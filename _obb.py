"""Oriented bounding box for object point clouds.

왜 별도 모듈인가
----------------
open3d 의 ``get_oriented_bounding_box()`` 는 최소부피 상자가 아니다.
볼록껍질 점들의 공분산 고유벡터(PCA)를 상자 축으로 쓴다. 물체가 등방적일수록
(세 특이값이 비슷할수록) 그 축은 물체의 실제 모서리와 무관한 방향으로 정해지고,
상자는 물체를 감싸느라 크게 부푼다.

실측 예 (data/capture_obj 의 obj2, 검은 정육면체, 특이값 1 / 0.96 / 0.87):

    open3d PCA OBB : 71 x 70 x 60 mm
    최소부피 OBB   : 55 x 54 x 53 mm    <- 실제 치수
    부피비 1.85x

길쭉한 물체(obj1, 특이값 1 / 0.49 / 0.47)에서는 PCA 축이 잘 잡혀 차이가 작다
(부피비 1.09x). 그래서 이 결함은 등방적인 물체에서만 드러난다.

최소부피 OBB 는 "최소부피 상자는 반드시 볼록껍질의 한 면과 맞닿는다"는 성질을 써서
껍질의 모든 면을 후보 축으로 놓고 회전 캘리퍼스로 부피를 최소화한다.
``trimesh.bounds.oriented_bounds`` 가 이걸 한다.

이제 이 OBB 는 **크기 추정기가 아니라** Obj_Step3c 실루엣 정합의 초기 포즈/스케일을
잡는 용도로만 쓰인다. 크기는 실루엣이 정한다.

반환 규약은 open3d 와 동일하게 맞췄다: ``(center, extent, R)``,
R 의 열이 상자 축(box -> world), extent 는 각 축의 전체 길이.
"""

from typing import Tuple

import numpy as np
import trimesh


def min_volume_obb(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """최소부피 OBB. 반환: (center, extent, R_box_to_world).

    볼록껍질을 만들 수 없는 퇴화 점군(점이 너무 적거나 완전 공면)에서는
    PCA 축 기반 OBB 로 물러선다. 그런 경우 두 방법의 차이도 거의 없다.
    """
    pts = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) < 4:
        raise RuntimeError(f"need >= 4 points for an OBB, got {len(pts)}")

    try:
        T_world_to_box, extent = trimesh.bounds.oriented_bounds(pts)
    except Exception as e:  # QhullError 등: 공면/퇴화 점군
        return _pca_obb(pts, reason=str(e))

    T_box_to_world = np.linalg.inv(np.asarray(T_world_to_box, dtype=np.float64))
    center = T_box_to_world[:3, 3].copy()
    R = T_box_to_world[:3, :3].copy()
    extent = np.asarray(extent, dtype=np.float64)

    # 상자가 실제로 모든 점을 담는지 확인한다. 아니면 조용히 틀린 치수가 나간다.
    local = (pts - center) @ R
    slack = np.abs(local).max(axis=0) - extent / 2.0
    if np.any(slack > 1e-6):
        raise RuntimeError(f"min-volume OBB does not contain all points (slack={slack})")

    return center, extent, R


def _pca_obb(pts: np.ndarray, reason: str = "") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    print(f"[OBB] convex hull failed ({reason[:60]}); falling back to PCA axes")
    mean = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - mean, full_matrices=False)
    R = Vt.T
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1
    local = (pts - mean) @ R
    lo, hi = local.min(axis=0), local.max(axis=0)
    center = mean + R @ (0.5 * (lo + hi))
    return center, hi - lo, R
