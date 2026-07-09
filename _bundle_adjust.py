"""Multi-camera bundle adjustment for the AprilTag cube.

카메라 간 변환을 PnP 결과의 SE(3) 평균으로 구하는 대신, 모든 카메라·모든 프레임의
마커 코너 재투영 오차를 하나의 비선형 최소제곱으로 직접 최소화한다.

최소화 대상 (unknowns):
  - T_Cref_Ci : 비참조 카메라의 외부 파라미터 (카메라당 6 DOF). 참조 카메라는 항등 고정.
  - T_Cref_O[f] : 프레임별 큐브 pose (프레임당 6 DOF)

잔차:
  r = project(K_i, D_i, inv(T_Cref_Ci) @ T_Cref_O[f] @ X_obj) - x_observed   (픽셀)

내부 파라미터(K, D)는 Step1 의 팩토리 intrinsics 를 신뢰해 고정한다.

게이지(gauge) 고정:
  - 참조 카메라를 항등으로 두어 전역 6 DOF 자유도를 없앤다.
  - 따라서 프레임은 반드시 2대 이상의 카메라에서 관측된 것만 사용한다.
    1대만 본 프레임은 T_Cref_O[f] 와 T_Cref_Ci 가 함께 미끄러져 아무것도 구속하지 못한다.

스케일은 큐브의 알려진 물리 치수(marker_corners_in_rig)가 고정하므로 별도 처리가 없다.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


def rotvec_t_to_T(v: np.ndarray) -> np.ndarray:
    """[rx, ry, rz, tx, ty, tz] -> 4x4"""
    T = np.eye(4, dtype=np.float64)
    R, _ = cv2.Rodrigues(np.asarray(v[:3], dtype=np.float64).reshape(3, 1))
    T[:3, :3] = R
    T[:3, 3] = np.asarray(v[3:6], dtype=np.float64)
    return T


def T_to_rotvec_t(T: np.ndarray) -> np.ndarray:
    """4x4 -> [rx, ry, rz, tx, ty, tz]"""
    rvec, _ = cv2.Rodrigues(np.asarray(T[:3, :3], dtype=np.float64))
    return np.concatenate([rvec.reshape(3), np.asarray(T[:3, 3], dtype=np.float64)])


def _inv_T(T: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


class Observation:
    """한 (카메라, 프레임) 쌍에서 검출된 마커 코너들."""

    __slots__ = ("cam_id", "frame_id", "obj_pts", "img_pts")

    def __init__(self, cam_id: int, frame_id: int, obj_pts: np.ndarray, img_pts: np.ndarray):
        self.cam_id = int(cam_id)
        self.frame_id = int(frame_id)
        self.obj_pts = np.asarray(obj_pts, dtype=np.float64).reshape(-1, 3)
        self.img_pts = np.asarray(img_pts, dtype=np.float64).reshape(-1, 2)
        if len(self.obj_pts) != len(self.img_pts):
            raise ValueError("obj_pts/img_pts length mismatch")

    @property
    def n_pts(self) -> int:
        return len(self.obj_pts)


class _Problem:
    """파라미터 벡터 <-> 포즈 사이의 인덱싱을 담당."""

    def __init__(self, ref_cam: int, cam_ids: List[int], frame_ids: List[int]):
        self.ref_cam = int(ref_cam)
        self.opt_cams = [c for c in cam_ids if c != ref_cam]   # 최적화되는 카메라
        self.frame_ids = list(frame_ids)
        self.cam_slot = {c: i for i, c in enumerate(self.opt_cams)}
        self.frame_slot = {f: i for i, f in enumerate(self.frame_ids)}
        self.n_cam_params = 6 * len(self.opt_cams)

    def cam_block(self, cam_id: int) -> Optional[slice]:
        if cam_id == self.ref_cam:
            return None
        i = self.cam_slot[cam_id]
        return slice(6 * i, 6 * i + 6)

    def frame_block(self, frame_id: int) -> slice:
        i = self.frame_slot[frame_id]
        return slice(self.n_cam_params + 6 * i, self.n_cam_params + 6 * i + 6)

    def pack(self, T_cams: Dict[int, np.ndarray], T_frames: Dict[int, np.ndarray]) -> np.ndarray:
        x = np.zeros(self.n_cam_params + 6 * len(self.frame_ids), dtype=np.float64)
        for c in self.opt_cams:
            x[self.cam_block(c)] = T_to_rotvec_t(T_cams[c])
        for f in self.frame_ids:
            x[self.frame_block(f)] = T_to_rotvec_t(T_frames[f])
        return x

    def unpack(self, x: np.ndarray) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        T_cams = {self.ref_cam: np.eye(4, dtype=np.float64)}
        for c in self.opt_cams:
            T_cams[c] = rotvec_t_to_T(x[self.cam_block(c)])
        T_frames = {f: rotvec_t_to_T(x[self.frame_block(f)]) for f in self.frame_ids}
        return T_cams, T_frames


def _residuals(x, prob, obs_list, K_map, D_map):
    T_cams, T_frames = prob.unpack(x)
    out = np.empty(2 * sum(o.n_pts for o in obs_list), dtype=np.float64)
    k = 0
    for o in obs_list:
        # T_Ci_O = inv(T_Cref_Ci) @ T_Cref_O
        T_Ci_O = _inv_T(T_cams[o.cam_id]) @ T_frames[o.frame_id]
        rvec, _ = cv2.Rodrigues(T_Ci_O[:3, :3])
        proj, _ = cv2.projectPoints(o.obj_pts, rvec, T_Ci_O[:3, 3], K_map[o.cam_id], D_map[o.cam_id])
        n = 2 * o.n_pts
        out[k:k + n] = (proj.reshape(-1, 2) - o.img_pts).ravel()
        k += n
    return out


def _jac_sparsity(prob, obs_list):
    """각 관측 블록은 자기 카메라 6 파라미터와 자기 프레임 6 파라미터에만 의존."""
    m = 2 * sum(o.n_pts for o in obs_list)
    n = prob.n_cam_params + 6 * len(prob.frame_ids)
    S = lil_matrix((m, n), dtype=int)
    row = 0
    for o in obs_list:
        rows = slice(row, row + 2 * o.n_pts)
        cb = prob.cam_block(o.cam_id)
        if cb is not None:
            S[rows, cb] = 1
        S[rows, prob.frame_block(o.frame_id)] = 1
        row += 2 * o.n_pts
    return S


def _per_obs_rms(res: np.ndarray, obs_list) -> np.ndarray:
    """관측 블록별 RMS 픽셀 오차."""
    out = np.empty(len(obs_list), dtype=np.float64)
    k = 0
    for i, o in enumerate(obs_list):
        n = 2 * o.n_pts
        out[i] = float(np.sqrt(np.mean(res[k:k + n] ** 2)))
        k += n
    return out


def _point_errors(res: np.ndarray) -> np.ndarray:
    """점별 재투영 오차 (픽셀 거리)."""
    r = res.reshape(-1, 2)
    return np.linalg.norm(r, axis=1)


def bundle_adjust_multicam(
    obs_list: List[Observation],
    K_map: Dict[int, np.ndarray],
    D_map: Dict[int, np.ndarray],
    T_Cref_Ci_init: Dict[int, np.ndarray],
    T_Cref_O_init: Dict[int, np.ndarray],
    ref_cam: int,
    huber_px: float = 2.0,
    max_nfev: int = 200,
    reject_px: Optional[float] = 4.0,
    reject_rounds: int = 2,
    verbose: bool = True,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], dict]:
    """재투영 오차를 직접 최소화해 T_Cref_Ci 와 프레임별 큐브 pose 를 동시에 구한다.

    Args:
        obs_list: 관측 목록. 2대 이상 카메라가 본 프레임만 넣을 것 (게이지 고정).
        T_Cref_Ci_init / T_Cref_O_init: 초기값 (보통 PnP + robust SE3 평균 결과).
        huber_px: Huber loss 의 f_scale (픽셀). 이보다 큰 잔차는 선형 가중.
        reject_px: 라운드 종료 후 이 값을 넘는 '점'을 버리고 재최적화. None 이면 비활성.
        reject_rounds: outlier 제거 + 재최적화 반복 횟수.

    Returns:
        (T_Cref_Ci, T_Cref_O, stats)
    """
    obs_list = [o for o in obs_list if o.n_pts > 0]
    if not obs_list:
        raise ValueError("no observations")

    cam_ids = sorted({o.cam_id for o in obs_list})
    if ref_cam not in cam_ids:
        raise ValueError(f"ref_cam {ref_cam} has no observations")

    # 2대 미만이 본 프레임은 게이지가 풀려 최적화를 망친다 → 제외
    cams_per_frame: Dict[int, set] = {}
    for o in obs_list:
        cams_per_frame.setdefault(o.frame_id, set()).add(o.cam_id)
    usable = {f for f, cs in cams_per_frame.items() if len(cs) >= 2}
    dropped_frames = sorted(set(cams_per_frame) - usable)
    obs_list = [o for o in obs_list if o.frame_id in usable]
    if not obs_list:
        raise ValueError("no frame is observed by >= 2 cameras; cannot fix the gauge")

    frame_ids = sorted(usable)
    missing = [f for f in frame_ids if f not in T_Cref_O_init]
    if missing:
        raise ValueError(f"missing initial cube pose for frames {missing[:5]}")
    missing_c = [c for c in cam_ids if c != ref_cam and c not in T_Cref_Ci_init]
    if missing_c:
        raise ValueError(f"missing initial extrinsics for cams {missing_c}")

    prob = _Problem(ref_cam, cam_ids, frame_ids)
    x0 = prob.pack(T_Cref_Ci_init, T_Cref_O_init)

    stats: dict = {
        "ref_cam": int(ref_cam),
        "cam_ids": [int(c) for c in cam_ids],
        "n_frames": len(frame_ids),
        "frames_dropped_single_view": [int(f) for f in dropped_frames],
        "n_params": int(x0.size),
        "rounds": [],
    }

    if verbose:
        print(f"[BA] cams={cam_ids} ref=cam{ref_cam}  frames={len(frame_ids)}"
              f"  params={x0.size}")
        if dropped_frames:
            print(f"[BA] 단일 시점 프레임 {len(dropped_frames)}개 제외 (게이지 미고정)")

    res0 = _residuals(x0, prob, obs_list, K_map, D_map)
    rms_init = float(np.sqrt(np.mean(res0 ** 2)))
    stats["rms_px_initial"] = rms_init
    if verbose:
        print(f"[BA] 초기 재투영 RMS = {rms_init:.4f} px  "
              f"(관측 {sum(o.n_pts for o in obs_list)}점)")

    x = x0
    for rnd in range(max(int(reject_rounds), 1)):
        sparsity = _jac_sparsity(prob, obs_list)
        sol = least_squares(
            _residuals, x,
            jac_sparsity=sparsity,
            args=(prob, obs_list, K_map, D_map),
            method="trf", loss="huber", f_scale=float(huber_px),
            xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=int(max_nfev),
            verbose=0,
        )
        x = sol.x
        res = _residuals(x, prob, obs_list, K_map, D_map)
        rms = float(np.sqrt(np.mean(res ** 2)))
        pt_err = _point_errors(res)
        round_stat = {
            "round": rnd,
            "rms_px": rms,
            "median_px": float(np.median(pt_err)),
            "p90_px": float(np.percentile(pt_err, 90)),
            "max_px": float(pt_err.max()),
            "n_points": int(pt_err.size),
            "nfev": int(sol.nfev),
        }
        stats["rounds"].append(round_stat)
        if verbose:
            print(f"[BA] round{rnd}: RMS={rms:.4f}px median={round_stat['median_px']:.4f} "
                  f"p90={round_stat['p90_px']:.4f} max={round_stat['max_px']:.4f} "
                  f"nfev={sol.nfev}")

        if reject_px is None or rnd == int(reject_rounds) - 1:
            break

        # 점 단위 outlier 제거 후 재최적화
        keep_obs, n_dropped = [], 0
        k = 0
        for o in obs_list:
            e = pt_err[k:k + o.n_pts]
            k += o.n_pts
            m = e <= float(reject_px)
            n_dropped += int((~m).sum())
            if m.sum() >= 4:      # 마커 하나 분량은 남아야 의미가 있다
                keep_obs.append(Observation(o.cam_id, o.frame_id, o.obj_pts[m], o.img_pts[m]))
        if n_dropped == 0:
            if verbose:
                print(f"[BA] outlier 없음 → 수렴")
            break

        # 프레임이 다시 단일 시점이 되면 게이지가 풀리므로 함께 제거
        cams_per_frame = {}
        for o in keep_obs:
            cams_per_frame.setdefault(o.frame_id, set()).add(o.cam_id)
        usable = {f for f, cs in cams_per_frame.items() if len(cs) >= 2}
        keep_obs = [o for o in keep_obs if o.frame_id in usable]
        if not keep_obs:
            if verbose:
                print("[BA] outlier 제거 후 남은 관측이 없어 이전 해를 유지")
            break

        # 파라미터 재구성 (프레임 집합이 줄었을 수 있음)
        T_cams, T_frames = prob.unpack(x)
        obs_list = keep_obs
        frame_ids = sorted(usable)
        prob = _Problem(ref_cam, sorted({o.cam_id for o in obs_list}), frame_ids)
        x = prob.pack(T_cams, {f: T_frames[f] for f in frame_ids})
        if verbose:
            print(f"[BA] outlier {n_dropped}점 제거 (>{reject_px}px) → 재최적화 "
                  f"(frames={len(frame_ids)})")

    T_cams, T_frames = prob.unpack(x)
    res = _residuals(x, prob, obs_list, K_map, D_map)
    pt_err = _point_errors(res)
    stats["rms_px_final"] = float(np.sqrt(np.mean(res ** 2)))
    stats["median_px_final"] = float(np.median(pt_err))
    stats["p90_px_final"] = float(np.percentile(pt_err, 90))
    stats["max_px_final"] = float(pt_err.max())
    stats["n_points_final"] = int(pt_err.size)
    stats["n_frames_final"] = len(frame_ids)

    # 카메라별 최종 재투영 오차
    per_cam: Dict[str, dict] = {}
    k = 0
    for o in obs_list:
        e = pt_err[k:k + o.n_pts]
        k += o.n_pts
        d = per_cam.setdefault(str(o.cam_id), {"errs": []})
        d["errs"].extend(e.tolist())
    for ci, d in per_cam.items():
        arr = np.asarray(d.pop("errs"))
        d["rms_px"] = float(np.sqrt(np.mean(arr ** 2)))
        d["median_px"] = float(np.median(arr))
        d["p90_px"] = float(np.percentile(arr, 90))
        d["n_points"] = int(arr.size)
    stats["per_camera"] = per_cam

    if verbose:
        print(f"[BA] 최종 재투영 RMS = {stats['rms_px_final']:.4f} px "
              f"(초기 {rms_init:.4f} px)")
        for ci in sorted(per_cam, key=int):
            d = per_cam[ci]
            print(f"[BA]   cam{ci}: RMS={d['rms_px']:.4f}px "
                  f"median={d['median_px']:.4f} p90={d['p90_px']:.4f} n={d['n_points']}")

    return T_cams, T_frames, stats
