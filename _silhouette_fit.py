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
CAD 는 형상만 준다 (이 저장소의 Peg.glb / Hole.glb 는 각각 독립적으로 정규화되어 있다).
따라서 스케일 1개 + 6-DoF 포즈 = 7개 파라미터를 푼다.
depth 는 초기값(대략적인 위치/방향)에만 쓰고, 스케일은 실루엣이 정한다.

주의: 점군을 CAD 좌표계로 축소시키며 맞추는 방향은 퇴화한다 (스케일 -> 0 이면 모든
점-표면 거리가 0). 그래서 항상 점군을 고정하고 CAD 를 움직인다.
"""

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import open3d as o3d
import trimesh
from scipy.optimize import minimize
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


def scale_cad_to_extents(mesh: trimesh.Trimesh, target_ext_desc: np.ndarray) -> np.ndarray:
    """CAD 정점을 OBB 축별로 비균등 스케일해 target extents(내림차순)를 갖게 한다.

    반환된 정점은 여전히 CAD 좌표계이므로, fit 이 준 (R, t) 를 그대로 적용하면
    실측 크기의 물체를 같은 자리에 놓은 것이 된다. (균등 스케일 s 인 경우와 일치)
    """
    V = np.asarray(mesh.vertices, dtype=np.float64)
    _, R_m, e_m = obb_frame(V)
    f = np.asarray(target_ext_desc, dtype=np.float64) / e_m
    return ((V @ R_m) * f) @ R_m.T


def cloud_from_masked_depth(K, T_cam_to_world, depth_u16, mask, depth_scale=0.001,
                            erode_px=3, z_range=(0.05, 2.0)) -> np.ndarray:
    """마스크 안쪽 depth 픽셀을 world 로 backproject. 실루엣 경계 픽셀은 침식으로 제외."""
    m = np.asarray(mask, bool)
    if erode_px > 0:
        k = np.ones((erode_px * 2 + 1,) * 2, np.uint8)
        m = cv2.erode(m.astype(np.uint8), k, 1) > 0
    z = np.asarray(depth_u16, np.float64) * float(depth_scale)
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


def clean_cloud(pts: np.ndarray, voxel_m=0.002, nb_neighbors=20, std_ratio=2.0,
                dbscan_eps_m=0.01, dbscan_min_points=10) -> np.ndarray:
    """다운샘플 -> 통계적 이상치 제거 -> DBSCAN 최대 군집만 유지.

    DBSCAN 이 없으면 마스크 가장자리에서 새어 들어온 배경 점 뭉치가 남아 초기 OBB 가
    수백 mm 로 부풀고, 그러면 ICP 초기 포즈가 엉뚱한 골짜기에서 시작한다.
    """
    if len(pts) < 10:
        return pts
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
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
