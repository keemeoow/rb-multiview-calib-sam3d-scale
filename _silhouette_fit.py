"""CAD-to-observation fitting by multi-view silhouette matching.

왜 실루엣인가
-------------
마스크 외곽선은 K 와 카메라 포즈만 있으면 metric 이고 depth 를 전혀 쓰지 않는다.
이 저장소의 실제 데이터에서 depth 는 mm 단위로 신뢰할 수 없다:

    흰 종이 (평면 잔차)      : 카메라 간 불일치 +/- 1.4 mm
    검은 원기둥 (obj1)       : 5.3 ~ 7.1 mm
    검은 정육면체 (obj2)     : 7.8 ~ 14.7 mm

검은 무광 표면에서 RealSense 는 점을 카메라로부터 멀어지는 쪽(= 물체 안쪽)으로 밀어낸다.
그 결과:
  - 점군 OBB 는 바깥쪽 노이즈 점을 껴안아 **과대추정**하고,
  - 점군에 대한 depth-ICP 는 편향을 그대로 따라가 **과소추정**한다.

참값을 아는 합성 실험 (실제 카메라 배치, depth 편향 3mm 주입) 에서의 평균 |치수 오차|:

    점군 최소부피 OBB   3.37 mm
    메시 depth-ICP      0.79 mm
    메시 실루엣 정합    0.28 mm      <- 이 모듈

미지수
------
단서 메시는 형상만 준다 (data/meshes/peg.glb, hole.glb 는 각각 독립적으로 정규화되어 있다).
YCB 처럼 이미 실척인 mesh 를 넣어도 된다 — 그 경우 추정 scale 이 1.0 근처로 나와야 정상이다.
따라서 스케일 1개 + 6-DoF 포즈 = 7개 파라미터를 푼다.
depth 는 초기값(대략적인 위치/방향)에만 쓰고, 스케일은 실루엣이 정한다.

단서 메시는 두 곳에서 온다 — 이 모듈은 둘을 구분하지 않는다 (엔진은 동일):
  baseline  Obj_Step3_sam3d_scale.py  SAM3D 가 만든 메시 (형상 추정치). **운용 경로.**
  oracle    Obj_Step3c_cad_scale.py   원본 정답 CAD (형상 참값). 상한/피규어 기준값.
형상이 참값이면 스케일 하나만 맞추면 되므로 sub-mm 가 나온다. SAM3D 처럼 형상이
추정치면 균등 스케일로 형상 오차를 못 고치므로, per-view IoU 가 치수 신뢰도가 된다.

주의: 점군을 CAD 좌표계로 축소시키며 맞추는 방향은 퇴화한다 (스케일 -> 0 이면 모든
점-표면 거리가 0). 그래서 항상 점군을 고정하고 CAD 를 움직인다.
"""

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import open3d as o3d
import trimesh
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from _obb import min_volume_obb


SUPERSAMPLE = 2   # 하위 픽셀 스케일 변화에도 손실이 매끄럽게 변하도록


class SurfaceQuery:
    """메시 표면 위 최근접점. open3d BVH 사용 (rtree 의존성 없음)."""

    def __init__(self, mesh: trimesh.Trimesh):
        tm = o3d.t.geometry.TriangleMesh(
            o3d.core.Tensor(np.asarray(mesh.vertices), dtype=o3d.core.Dtype.Float32),
            o3d.core.Tensor(np.asarray(mesh.faces), dtype=o3d.core.Dtype.UInt32))
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(tm)

    def closest(self, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        q = o3d.core.Tensor(np.asarray(pts, dtype=np.float32), dtype=o3d.core.Dtype.Float32)
        c = self.scene.compute_closest_points(q)["points"].numpy().astype(np.float64)
        return c, np.linalg.norm(c - pts, axis=1)


class View:
    """한 카메라의 관측: K, world->cam, 이진 마스크."""

    def __init__(self, K: np.ndarray, T_cam_to_world: np.ndarray, mask: np.ndarray):
        self.K = np.asarray(K, dtype=np.float64)
        self.W_c = np.linalg.inv(np.asarray(T_cam_to_world, dtype=np.float64))
        self.mask = np.asarray(mask, dtype=bool)
        self.shape = self.mask.shape
        self.mask_ss = cv2.resize(self.mask.astype(np.uint8), None,
                                  fx=SUPERSAMPLE, fy=SUPERSAMPLE,
                                  interpolation=cv2.INTER_NEAREST) > 0


def obb_frame(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(center, R, extent). R 의 열이 상자 축, extent 는 내림차순."""
    c, ext, R = min_volume_obb(pts)
    order = np.argsort(ext)[::-1]
    R = R[:, order]
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
    return c, R, ext[order]


def umeyama(X: np.ndarray, Y: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """s,R,t minimizing ||sRX + t - Y||."""
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    U, D, Vt = np.linalg.svd(Xc.T @ Yc / len(X))
    d = np.ones(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        d[2] = -1
    R = Vt.T @ np.diag(d) @ U.T
    s = float((D * d).sum() / ((Xc ** 2).sum() / len(X)))
    return s, R, my - s * R @ mx


_PERMS: Optional[List[np.ndarray]] = None


def _axis_rotations() -> List[np.ndarray]:
    """상자 축을 뒤바꾸는 24개의 proper rotation."""
    global _PERMS
    if _PERMS is None:
        out = []
        for p in [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]:
            P = np.eye(3)[list(p)]
            for sx in (1,-1):
                for sy in (1,-1):
                    for sz in (1,-1):
                        G = P * np.array([sx, sy, sz])[:, None]
                        if abs(np.linalg.det(G) - 1.0) < 1e-9:
                            out.append(G)
        _PERMS = out
    return _PERMS


def render_silhouette(V, F, s, R, t, view: View, ss: int = 1) -> np.ndarray:
    """포즈된 메시를 이진 실루엣으로 래스터화. 실루엣 = 투영된 삼각형들의 합집합.

    삼각형을 한 번의 ``cv2.fillPoly`` 호출로 모두 넘기면 안 된다. OpenCV 는 여러
    폴리곤을 even-odd 규칙으로 채우므로 겹치는 부분(앞면과 뒷면)이 서로 상쇄되어
    물체 내부가 뚫린다. 실측: 그렇게 그리면 IoU 0.85, 면적이 마스크의 85% 로 줄고
    슈퍼샘플링을 올릴수록 더 심해진다 (ss=4 에서 IoU 0.67).
    삼각형마다 따로 채워 진짜 합집합을 만든다 (ss=2 에서 3.3ms -> 4.0ms).
    """
    Vw = (s * (R @ V.T)).T + t
    Vc = (view.W_c[:3, :3] @ Vw.T).T + view.W_c[:3, 3]
    K = view.K.copy()
    K[:2, :] *= ss
    uv = (K @ Vc.T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = uv[:, :2] / uv[:, 2:3]
    sil = np.zeros((view.shape[0] * ss, view.shape[1] * ss), np.uint8)
    ok = (Vc[:, 2][F] > 1e-6).all(axis=1) & np.isfinite(uv[F]).all(axis=(1, 2))
    for tri in uv[F[ok]].astype(np.int32):
        cv2.fillConvexPoly(sil, tri, 255)
    return sil > 0


def per_view_iou(V, F, s, R, t, views: Sequence[View]) -> List[float]:
    out = []
    for v in views:
        sil = render_silhouette(V, F, s, R, t, v, ss=SUPERSAMPLE)
        inter = np.logical_and(sil, v.mask_ss).sum()
        union = np.logical_or(sil, v.mask_ss).sum()
        out.append(float(inter) / max(int(union), 1))
    return out


def _sil_loss(V, F, s, R, t, views) -> float:
    return 1.0 - float(np.mean(per_view_iou(V, F, s, R, t, views)))


def depth_rms(pq: SurfaceQuery, cloud, s, R, t, trim: float = 0.10) -> float:
    """점군 -> CAD 표면 거리 (m). trim 만큼 최악의 대응을 버린다."""
    q = ((cloud - t) @ R) / s
    _, d = pq.closest(q)
    k = max(int(len(cloud) * (1 - trim)), 10)
    return float(np.sqrt(np.mean(np.sort(d * s)[:k] ** 2)))


def _icp_fixed_scale(V, pq, cloud, s, trim=0.10, iters=30):
    """스케일 고정 ICP. 24개 축 정렬에서 시작해 최적 포즈를 고른다."""
    c_m, R_m, _ = obb_frame(V)
    c_c, R_c, _ = obb_frame(cloud)
    best = (np.inf, None, None)
    for G in _axis_rotations():
        R = R_c @ G @ R_m.T
        t = c_c - s * R @ c_m
        for _ in range(iters):
            q = ((cloud - t) @ R) / s
            closest, d = pq.closest(q)
            k = max(int(len(cloud) * (1 - trim)), 10)
            keep = np.argsort(d * s)[:k]
            _, R, t = umeyama(closest[keep] * s, cloud[keep])
        rms = depth_rms(pq, cloud, s, R, t, trim)
        if rms < best[0]:
            best = (rms, R, t)
    return best


def _pack(s, R, t):
    return np.concatenate([[np.log(s)], Rotation.from_matrix(R).as_rotvec(), t])


def _unpack(x):
    return float(np.exp(x[0])), Rotation.from_rotvec(x[1:4]).as_matrix(), x[4:7]


def fit_cad_to_views(
    mesh: trimesh.Trimesh,
    cloud: np.ndarray,
    views: Sequence[View],
    w_depth: float = 0.0,
    max_fev: int = 4000,
    verbose: bool = False,
) -> dict:
    """스케일 + 6-DoF 포즈를 실루엣으로 추정한다.

    w_depth > 0 이면 depth 잔차(m)를 손실에 더한다. 기본 0 = 순수 실루엣
    (depth 는 초기값에만 사용). 검은 물체에서는 0 을 권장한다.

    반환: scale, R, t (CAD -> world), extents_m, per-view IoU, depth_rms_mm.
    """
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces)
    if len(cloud) < 10:
        raise RuntimeError(f"cloud too small for initialisation: {len(cloud)} points")
    pq = SurfaceQuery(mesh)
    _, _, e_m = obb_frame(V)
    _, _, e_c = obb_frame(cloud)
    s_obb = float(np.mean(e_c / e_m))

    def loss(s, R, t):
        L = _sil_loss(V, F, s, R, t, views)
        if w_depth > 0:
            L += w_depth * depth_rms(pq, cloud, s, R, t)
        return L

    # 초기화: 여러 스케일에서 depth-ICP 포즈를 뽑고, 손실이 가장 낮은 것을 고른다.
    best = None
    for f in (0.80, 0.90, 1.00, 1.10):
        _, R, t = _icp_fixed_scale(V, pq, cloud, s_obb * f)
        L = loss(s_obb * f, R, t)
        if best is None or L < best[0]:
            best = (L, s_obb * f, R, t)
    _, s0, R0, t0 = best

    res = minimize(lambda x: loss(*_unpack(x)), _pack(s0, R0, t0), method="Powell",
                   options={"maxiter": 100000, "maxfev": int(max_fev),
                            "xtol": 1e-5, "ftol": 1e-7, "disp": bool(verbose)})
    s, R, t = _unpack(res.x)

    ious = per_view_iou(V, F, s, R, t, views)
    return {
        "scale": float(s),
        "R_cad_to_world": R,
        "t_cad_to_world": t,
        "extents_m": e_m * s,
        "per_view_iou": ious,
        "mean_iou": float(np.mean(ious)),
        "depth_rms_mm": depth_rms(pq, cloud, s, R, t) * 1000.0,
        "cloud_obb_extents_m": e_c,
        "n_fev": int(res.nfev),
    }


def fit_mesh_aniso(
    mesh: trimesh.Trimesh,
    cloud: np.ndarray,
    views: Sequence[View],
    w_depth: float = 0.0,
    max_fev: int = 6000,
    aniso_reg: float = 0.0,
    warm: Optional[dict] = None,
    verbose: bool = False,
) -> dict:
    """비등방(축별) 스케일 + 6-DoF 포즈를 실루엣으로 추정한다.

    fit_cad_to_views 는 단일 스칼라 스케일만 쓰므로, 형상 비율이 틀린 메시(예: SAM3D
    가 만든 peg — 너무 짧고 뚱뚱)는 균일 확대로 세 뷰 실루엣을 동시에 못 맞춘다.
    여기서는 메시 OBB 세 축에 독립 스케일 (sx,sy,sz) 을 허용해 비율까지 교정한다.
    등방 fit 결과에서 warm-start 하므로 등방보다 나빠지지 않는다.

    aniso_reg > 0 이면 log-scale 분산에 페널티를 줘 과도한 비등방(과적합)을 억제한다.
    반환 키는 fit_cad_to_views 와 호환(scale=기하평균) + scale_vec, V_scaled_meshframe.
    """
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces)
    if len(cloud) < 10:
        raise RuntimeError(f"cloud too small for initialisation: {len(cloud)} points")

    c_m, R_m, e_m = obb_frame(V)          # OBB center, axes(cols, desc), extent(desc)
    Vo = (V - c_m) @ R_m                  # 정점을 OBB 중심·축 프레임으로 (mesh units)
    pqo = SurfaceQuery(trimesh.Trimesh(vertices=Vo, faces=F, process=False))  # depth 용

    # warm-start: 등방 fit (제공되면 재계산 생략)
    iso = warm if warm is not None else fit_cad_to_views(mesh, cloud, views, w_depth=0.0, max_fev=max_fev)
    s0, R0, t0 = iso["scale"], iso["R_cad_to_world"], iso["t_cad_to_world"]
    # 등방 fit 은 원점 기준 스케일; 여기선 OBB 중심 기준이라 t 를 보정
    t0 = t0 + s0 * (R0 @ c_m)

    def cand_vertices(svec):
        return (Vo * svec) @ R_m.T        # mesh 프레임(중심0), 축별 스케일

    def loss(svec, R, t):
        Vc = cand_vertices(svec)
        L = 1.0 - float(np.mean(per_view_iou(Vc, F, 1.0, R, t, views)))
        if w_depth > 0:
            b = ((cloud - t) @ R) @ R_m   # world -> OBB 축 프레임(스케일된)
            b_un = b / svec               # 비등방 스케일 되돌림 -> 단위 프레임(Vo)
            _, d = pqo.closest(b_un)
            d_world = d * float(np.cbrt(np.prod(svec)))   # 대략 world meter 로 환산
            k = max(int(len(cloud) * 0.9), 10)            # 최악 10% 대응 trim (robust)
            L += w_depth * float(np.sqrt(np.mean(np.sort(d_world)[:k] ** 2)))
        if aniso_reg > 0:
            L += aniso_reg * float(np.var(np.log(svec)))
        return L

    def pack(svec, R, t):
        return np.concatenate([np.log(svec), Rotation.from_matrix(R).as_rotvec(), t])

    def unpack(x):
        return np.exp(x[0:3]), Rotation.from_rotvec(x[3:6]).as_matrix(), x[6:9]

    x0 = pack(np.array([s0, s0, s0]), R0, t0)
    res = minimize(lambda x: loss(*unpack(x)), x0, method="Powell",
                   options={"maxiter": 100000, "maxfev": int(max_fev),
                            "xtol": 1e-5, "ftol": 1e-7, "disp": bool(verbose)})
    svec, R, t = unpack(res.x)

    Vc = cand_vertices(svec)
    ious = per_view_iou(Vc, F, 1.0, R, t, views)
    _, _, ext = obb_frame(Vc)             # 최종 스케일 메시의 최소부피 OBB 치수
    return {
        "scale": float(np.cbrt(np.prod(svec))),
        "scale_vec": [float(v) for v in svec],
        "R_cad_to_world": R,
        "t_cad_to_world": t,
        "V_scaled_meshframe": Vc,
        "extents_m": ext,
        "per_view_iou": ious,
        "mean_iou": float(np.mean(ious)),
        "cloud_obb_extents_m": obb_frame(cloud)[2],
        "n_fev": int(res.nfev),
    }


def scale_cad_to_extents(mesh: trimesh.Trimesh, target_ext_desc: np.ndarray) -> np.ndarray:
    """CAD 정점을 OBB 축별로 비균등 스케일해 target extents(내림차순)를 갖게 한다.

    반환된 정점은 여전히 CAD 좌표계이므로, fit 이 준 (R, t) 를 그대로 적용하면
    실측 크기의 물체를 같은 자리에 놓은 것이 된다. (균등 스케일 s 인 경우와 일치)
    """
    V = np.asarray(mesh.vertices, dtype=np.float64)
    _, R_m, e_m = obb_frame(V)
    f = np.asarray(target_ext_desc, dtype=np.float64) / e_m
    return ((V @ R_m) * f) @ R_m.T


def _flying_pixel_edges(z: np.ndarray, mask: np.ndarray,
                        n_sigma: float, min_valid: float) -> np.ndarray:
    """물체 자신의 depth 그래디언트 통계로 경계 오염(mixed/flying pixel)을 탐지.

    고정 침식 대신 쓴다. 실제 표면은 픽셀 간 depth 변화가 완만하지만, 실루엣 경계의
    flying pixel 은 배경으로 튀며 국소 그래디언트가 급증한다. 그 그래디언트의 robust
    통계(median + n_sigma·MAD)를 넘는 픽셀만 제거하므로, 거리·물체 크기·형상과 무관하게
    "이 물체 기준으로 튀는 경계"에만 자동으로 맞춰진다. 반환: 제거 대상 True 마스크.
    """
    valid = mask & (z > min_valid)
    g = np.zeros_like(z)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        zs = np.roll(z, (dy, dx), (0, 1))
        vs = np.roll(valid, (dy, dx), (0, 1))
        d = np.abs(z - zs)
        d[~(valid & vs)] = 0.0                 # 유효 이웃끼리만 그래디언트 정의
        g = np.maximum(g, d)
    gin = g[valid]
    if gin.size < 20:
        return np.zeros_like(mask, bool)
    gmed = float(np.median(gin))
    gmad = 1.4826 * float(np.median(np.abs(gin - gmed))) + 1e-6
    return valid & (g > gmed + n_sigma * gmad)


def _auto_voxel(pts: np.ndarray, k_nn: float = 1.5,
                target_samples: Tuple[int, int] = (40, 300),
                noise_floor_m: Optional[float] = None) -> float:
    """점 밀도와 물체 스케일에서 voxel 크기를 유도한다 (고정 2mm 대체).

        voxel = clip(k_nn · d_nn,  L/N_hi,  L/N_lo),   voxel ≥ noise_floor

      d_nn : 최근접 이웃 간격의 median. 카메라가 멀어져 점이 성겨지면(≈ Z/fx) 함께 커진다
             → 거리·해상도 자동 적응 (물리항).
      L    : 중심 95% 점의 AABB 대각. 이상치를 뺀 물체 스케일 (데이터항).
             작은 물체는 촘촘, 큰 물체는 성기게 해 물체당 샘플 수를 N_lo~N_hi 로 유지.
    """
    P = np.asarray(pts, np.float64)
    if len(P) < 10:
        return 0.002
    if len(P) > 5000:                          # 결정론적 스트라이드 서브샘플(속도)
        P = P[:: len(P) // 5000]
    tree = cKDTree(P)
    dists, _ = tree.query(P, k=2)
    d_nn = float(np.median(dists[:, 1]))
    c = np.median(P, axis=0)
    r = np.linalg.norm(P - c, axis=1)
    core = P[r <= np.quantile(r, 0.95)]
    L = float(np.linalg.norm(core.max(0) - core.min(0))) if len(core) else 0.0
    v = k_nn * d_nn
    n_lo, n_hi = target_samples
    if L > 0:
        v = float(np.clip(v, L / n_hi, L / n_lo))
    if noise_floor_m:
        v = max(v, float(noise_floor_m))
    return max(v, 1e-4)


def cloud_from_masked_depth(K, T_cam_to_world, depth_u16, mask, depth_scale=0.001,
                            erode_px="auto", z_range=(0.05, 2.0),
                            flying_nsigma=4.0) -> np.ndarray:
    """마스크 안쪽 depth 픽셀을 world 로 backproject. 실루엣 경계 픽셀은 제외.

    erode_px="auto": 물체의 depth 그래디언트 통계로 오염 픽셀만 제거(권장) + 1px 안전 여유.
                     거리·물체 크기와 무관하게 자동 적응 (_flying_pixel_edges 참고).
    erode_px=<int>:  기존 고정 침식(rim = erode_px 픽셀). 하위호환용.
    """
    m = np.asarray(mask, bool)
    z = np.asarray(depth_u16, np.float64) * float(depth_scale)
    if erode_px == "auto":
        m = m & ~_flying_pixel_edges(z, m, flying_nsigma, z_range[0])
        m = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8), 1) > 0
    elif isinstance(erode_px, (int, np.integer)) and erode_px > 0:
        k = np.ones((int(erode_px) * 2 + 1,) * 2, np.uint8)
        m = cv2.erode(m.astype(np.uint8), k, 1) > 0
    H, W = z.shape
    v, u = np.mgrid[0:H, 0:W]
    sel = m & (z > z_range[0]) & (z < z_range[1])
    if not sel.any():
        return np.zeros((0, 3))
    zz = z[sel]
    P = np.stack([(u[sel] - K[0, 2]) * zz / K[0, 0],
                  (v[sel] - K[1, 2]) * zz / K[1, 1], zz], axis=1)
    T = np.asarray(T_cam_to_world, dtype=np.float64)
    return (T[:3, :3] @ P.T).T + T[:3, 3]


def clean_cloud(pts: np.ndarray, voxel_m="auto", nb_neighbors=20, std_ratio=2.0,
                dbscan_eps_m="auto", dbscan_min_points=10,
                noise_floor_m=None, verbose=True) -> np.ndarray:
    """다운샘플 -> 통계적 이상치 제거 -> DBSCAN 최대 군집만 유지.

    DBSCAN 이 없으면 마스크 가장자리에서 새어 들어온 배경 점 뭉치가 남아 초기 OBB 가
    수백 mm 로 부풀고, 그러면 ICP 초기 포즈가 엉뚱한 골짜기에서 시작한다.

    voxel_m="auto"     : 점 밀도·물체 스케일에서 유도(_auto_voxel). 고정값 대신 물체마다 최적화.
    dbscan_eps_m="auto": 5·voxel 로 연동. voxel 하나가 정해지면 군집 반경도 따라간다.
    """
    if len(pts) < 10:
        return pts
    src = "auto" if voxel_m == "auto" else "fixed"
    if voxel_m == "auto":
        voxel_m = _auto_voxel(pts, noise_floor_m=noise_floor_m)
    if dbscan_eps_m == "auto":
        dbscan_eps_m = 5.0 * voxel_m
    if verbose:
        print(f"  [{src}] voxel {voxel_m*1000:.2f} mm  dbscan_eps {dbscan_eps_m*1000:.2f} mm "
              f"(from {len(pts)} pts)")
    pcd = o3d.geometry.PointCloud()
    # Open3D 0.18(aarch64)의 Vector3dVector 는 non-C-contiguous 버퍼를 받으면 segfault 한다.
    # vstack 결과가 F-contiguous 로 나올 수 있어 C-contiguous float64 로 강제한다.
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, dtype=np.float64))
    if voxel_m > 0:
        pcd = pcd.voxel_down_sample(voxel_m)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors, std_ratio)
    p = np.asarray(pcd.points)
    if len(p) < 10:
        return p

    labels = np.asarray(pcd.cluster_dbscan(eps=dbscan_eps_m, min_points=dbscan_min_points))
    valid = labels >= 0
    if not valid.any():
        return p
    counts = np.bincount(labels[valid])
    return p[labels == int(np.argmax(counts))]


# ============================================================
# depth 신뢰도 자동 추정 (--w_depth auto)
# ============================================================
#
# depth 를 손실에 넣을지(w_depth)를 물체마다 자동으로 정한다. 판단 근거는
# **카메라 간 표면 일치도**: depth 가 믿을 만하면 한 카메라의 표면점을 다른 카메라로
# 재투영했을 때 그 카메라가 측정한 depth 와 일치한다. 검은/광택/투명처럼 depth 가
# 계통 편향(bias)을 가지면 카메라마다 표면 위치가 어긋나 불일치가 커진다
# (이 저장소 관측: 흰 종이 ~1.4mm, 검은 물체 5~15mm). 편향은 점을 더 모아도 줄지
# 않으므로(분산과 달리 평균으로 상쇄 안 됨), 이 불일치가 곧 depth 를 얼마나 믿을지의
# 지표다. 불일치가 작을수록 depth 가중을 키운다.


def _backproject_world(K, T_cam_to_world, depth_m, mask, z_range):
    """마스크 안의 유효 depth 픽셀을 world 3D 점으로 역투영."""
    z = np.asarray(depth_m, np.float64)
    m = np.asarray(mask, bool) & np.isfinite(z) & (z > z_range[0]) & (z < z_range[1])
    if not m.any():
        return np.zeros((0, 3))
    v, u = np.where(m)
    zz = z[m]
    K = np.asarray(K, np.float64)
    x = (u - K[0, 2]) * zz / K[0, 0]
    y = (v - K[1, 2]) * zz / K[1, 1]
    Pc = np.stack([x, y, zz], axis=1)
    T = np.asarray(T_cam_to_world, np.float64)
    return (T[:3, :3] @ Pc.T).T + T[:3, 3]


def cross_view_depth_disagreement(cams, z_range=(0.05, 2.0), reject_m=0.03,
                                  min_overlap=50) -> float:
    """카메라 간 표면 불일치(m)의 robust 중앙값. 평가 불가면 np.nan.

    cams: list of dict{K, T(cam->world), depth_m(HxW, meter), mask(HxW bool)}.
    한 카메라의 표면점을 다른 카메라로 재투영해, 그 카메라 mask 안에서 측정 depth 와
    비교한다. reject_m 초과 차이는 가림(occlusion)으로 서로 다른 표면에 대응된 것으로
    보고 버린다 (편향이 아니라 대응 오류이므로).
    """
    if len(cams) < 2:
        return np.nan
    world = [_backproject_world(c["K"], c["T"], c["depth_m"], c["mask"], z_range)
             for c in cams]
    pair_meds = []
    for a in range(len(cams)):
        Pw = world[a]
        if len(Pw) < min_overlap:
            continue
        for b in range(len(cams)):
            if a == b:
                continue
            cb = cams[b]
            Kb = np.asarray(cb["K"], np.float64)
            Wb = np.linalg.inv(np.asarray(cb["T"], np.float64))
            Xb = (Wb[:3, :3] @ Pw.T).T + Wb[:3, 3]
            front = Xb[:, 2] > z_range[0]
            if front.sum() < min_overlap:
                continue
            Xf = Xb[front]
            uv = (Kb @ Xf.T).T
            uu = uv[:, 0] / uv[:, 2]
            vv = uv[:, 1] / uv[:, 2]
            depth_b = np.asarray(cb["depth_m"], np.float64)
            mask_b = np.asarray(cb["mask"], bool)
            H, W = depth_b.shape
            ui = np.round(uu).astype(int)
            vi = np.round(vv).astype(int)
            inb = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            if inb.sum() < min_overlap:
                continue
            ui, vi, zf = ui[inb], vi[inb], Xf[inb, 2]
            meas = depth_b[vi, ui]
            good = (mask_b[vi, ui] & np.isfinite(meas)
                    & (meas > z_range[0]) & (meas < z_range[1]))
            if good.sum() < min_overlap:
                continue
            diff = np.abs(zf[good] - meas[good])
            diff = diff[diff < reject_m]
            if len(diff) >= min_overlap:
                pair_meds.append(float(np.median(diff)))
    if not pair_meds:
        return np.nan
    return float(np.median(pair_meds))


def auto_w_depth(cams, w_max=20.0, d_lo_m=0.0015, d_hi_m=0.006,
                 z_range=(0.05, 2.0), verbose=True) -> Tuple[float, dict]:
    """카메라 간 depth 일치도에서 w_depth 를 유도한다.

    반환 (w_depth, info). 불일치 δ 가 d_lo 이하면 신뢰도 1(→ w_max), d_hi 이상이면
    0(→ 순수 실루엣). 그 사이는 선형. 평가 불가(카메라<2 또는 겹침 부족)면 0 (안전).

    depth 가 계통 편향을 가질 때만 낮아지므로, 흰/무광처럼 정합에 도움되는 depth 는
    켜지고 검은/광택처럼 편향된 depth 는 자동으로 꺼진다. 최악의 경우에도 순수
    실루엣으로 우아하게 물러난다.
    """
    delta = cross_view_depth_disagreement(cams, z_range=z_range)
    if not np.isfinite(delta):
        info = {"disagreement_mm": None, "confidence": 0.0, "w_depth": 0.0,
                "mode": "auto", "d_lo_mm": d_lo_m * 1000.0, "d_hi_mm": d_hi_m * 1000.0,
                "w_max": float(w_max),
                "reason": "cross-view depth 평가 불가 (카메라<2 또는 겹침 부족) -> 순수 실루엣"}
        if verbose:
            print(f"  [w_depth auto] {info['reason']}")
        return 0.0, info
    conf = float(np.clip((d_hi_m - delta) / (d_hi_m - d_lo_m), 0.0, 1.0))
    w = float(w_max * conf)
    info = {"disagreement_mm": delta * 1000.0, "confidence": conf, "w_depth": w,
            "mode": "auto", "d_lo_mm": d_lo_m * 1000.0, "d_hi_mm": d_hi_m * 1000.0,
            "w_max": float(w_max), "reason": None}
    if verbose:
        print(f"  [w_depth auto] cross-view disagreement {delta*1000:.2f} mm "
              f"-> confidence {conf:.2f} -> w_depth {w:.1f}")
    return w, info
