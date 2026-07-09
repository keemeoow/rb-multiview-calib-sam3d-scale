#!/usr/bin/env python3
"""
_marker_box.py

물체(peg/hole)에 부착한 ArUco 마커로 **정답(ground-truth) 직육면체 크기**를 측정하는
코어 라이브러리.  카메라 캘리브레이션 point-cloud OBB(=추정 크기)와 비교할 기준값을
마커로 산출한다.

[측정 원리]
  - peg/hole 을 직육면체로 보고, 중심에 로컬 프레임(축 x/y/z = 박스 모서리 방향)을 둔다.
  - 측정할 각 면(face)에 ArUco 마커를 **평평하게, 가급적 면 중앙**에 1장 이상 부착.
    한 축(axis)의 '+' 면과 '-' 면 **양쪽 모두** 최소 1장 필요.
  - 마커의 world(로봇 base) 3D 포즈를 두 가지 방법으로 구할 수 있다:
      (1) triangulate  : 마커 코너를 ≥2대 카메라에서 멀티뷰 삼각측량 → 코너 3D.
                         넓은 카메라 baseline 덕에 깊이가 강건 → **sub-mm 권장 기본값**.
                         (Calib_verify_calibration_accuracy.py 와 동일 원리)
      (2) pnp          : 단일 마커 PnP(T_cam_marker) 후 cam→world 합성.
                         스케일은 인쇄된 마커 크기에서 옴(카메라 간 캘리브에 덜 의존)이나,
                         정면(fronto-parallel)에서 **깊이 불확실성이 커** 뷰축 방향 치수가
                         부정확할 수 있다. 교차검증/단일뷰 fallback 용.
  - 축별 크기(extent):
        c_plus  = '+' 면 마커 중심들의 평균 (world)
        c_minus = '-' 면 마커 중심들의 평균 (world)
        d_vec   = c_plus - c_minus
        n_axis  = 마커 면-법선 평균 (d_vec 방향으로 부호 정렬)
        extent  = |d_vec · n_axis| - 2 * surface_offset
    d_vec 를 면 법선에 투영하므로 마커가 면 안에서 약간 치우쳐 붙어도 강건.
  - GT extents (x,y,z) 를 내림차순 정렬해 추정 파이프라인의 OBB bbox_extents_m 과 비교.

[마커 로컬 프레임 규약 — OpenCV ArUco / SOLVEPNP_IPPE_SQUARE]
  코너 순서 TL,TR,BR,BL. objPoints 는 마커 중심 기준
      (-s,+s,0),(+s,+s,0),(+s,-s,0),(-s,-s,0),  s = size/2
  +X 오른쪽, +Y 위, +Z 스티커 바깥.  → outward-normal = R_cam_marker[:,2].

하드웨어/이미지 없이 단위테스트 가능한 순수 함수 위주.
(검출→PnP/삼각측량→world→축 solver 전 구간을 test_marker_box.py 가 합성 데이터로 검증)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------- #
# Marker layout config (물리 부착 규약)
# ---------------------------------------------------------------- #

_VALID_AXES = ("x", "y", "z")
_VALID_SIDES = ("+", "-")


@dataclass
class MarkerLayout:
    """마커 ID → (물체 축, 면 부호) 매핑 + 물리 파라미터.  JSON 예시: gt_marker_layout_example.json"""
    dictionary: str = "DICT_4X4_50"
    marker_size_m: float = 0.030          # 인쇄된 마커 한 변(검은 테두리 포함) [m]
    surface_offset_m: float = 0.0         # 마커면이 물체 표면에서 뜬 두께(스탠드오프) [m]
    marker_axis_side: Dict[int, Tuple[str, str]] = field(default_factory=dict)
    size_override: Dict[int, float] = field(default_factory=dict)
    offset_override: Dict[int, float] = field(default_factory=dict)
    axis_labels: Dict[str, str] = field(default_factory=dict)

    def size_for(self, marker_id: int) -> float:
        return float(self.size_override.get(marker_id, self.marker_size_m))

    def offset_for(self, marker_id: int) -> float:
        return float(self.offset_override.get(marker_id, self.surface_offset_m))

    def known_ids(self) -> set:
        return set(self.marker_axis_side.keys())


def load_layout(path: Path | str) -> MarkerLayout:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    markers = d.get("markers", {})
    axis_side: Dict[int, Tuple[str, str]] = {}
    size_override: Dict[int, float] = {}
    offset_override: Dict[int, float] = {}
    for k, v in markers.items():
        mid = int(k)
        axis = str(v["axis"]).lower()
        side = str(v["side"]).strip()
        if axis not in _VALID_AXES:
            raise ValueError(f"marker {mid}: axis must be one of {_VALID_AXES}, got {axis!r}")
        if side not in _VALID_SIDES:
            raise ValueError(f"marker {mid}: side must be '+' or '-', got {side!r}")
        axis_side[mid] = (axis, side)
        if "size_m" in v:
            size_override[mid] = float(v["size_m"])
        if "surface_offset_m" in v:
            offset_override[mid] = float(v["surface_offset_m"])
    if not axis_side:
        raise ValueError(f"layout {path} has no 'markers' entries")
    return MarkerLayout(
        dictionary=str(d.get("dictionary", "DICT_4X4_50")),
        marker_size_m=float(d.get("marker_size_m", 0.030)),
        surface_offset_m=float(d.get("surface_offset_m", 0.0)),
        marker_axis_side=axis_side,
        size_override=size_override,
        offset_override=offset_override,
        axis_labels={str(k): str(v) for k, v in d.get("axis_labels", {}).items()},
    )


# ---------------------------------------------------------------- #
# ArUco detection (params mirror _aruco_cube.py)
# ---------------------------------------------------------------- #

def build_detector(dictionary_name: str = "DICT_4X4_50") -> "cv2.aruco.ArucoDetector":
    dic_id = getattr(cv2.aruco, dictionary_name, None)
    if dic_id is None:
        raise ValueError(f"unknown ArUco dictionary: {dictionary_name}")
    dictionary = cv2.aruco.getPredefinedDictionary(dic_id)
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    p.minMarkerPerimeterRate = 0.02
    return cv2.aruco.ArucoDetector(dictionary, p)


def detect_markers(image: np.ndarray,
                   detector: "cv2.aruco.ArucoDetector") -> List[Tuple[int, np.ndarray]]:
    """returns [(marker_id, corners(4,2) float64)], corners in TL,TR,BR,BL order."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    corners, ids, _ = detector.detectMarkers(gray)
    out: List[Tuple[int, np.ndarray]] = []
    if ids is None:
        return out
    for c, i in zip(corners, ids.flatten()):
        out.append((int(i), np.asarray(c, dtype=np.float64).reshape(4, 2)))
    return out


# ---------------------------------------------------------------- #
# Single-marker PnP
# ---------------------------------------------------------------- #

def marker_object_points(size_m: float) -> np.ndarray:
    """SOLVEPNP_IPPE_SQUARE 규약 4개 코너(마커 로컬, z=0). 순서 TL,TR,BR,BL."""
    s = float(size_m) / 2.0
    return np.array([[-s, s, 0.0], [s, s, 0.0], [s, -s, 0.0], [-s, -s, 0.0]], dtype=np.float64)


def _D(D) -> np.ndarray:
    return np.zeros((5, 1)) if D is None else np.asarray(D, dtype=np.float64).reshape(-1, 1)


def pnp_marker(corners: np.ndarray, size_m: float,
               K: np.ndarray, D: Optional[np.ndarray] = None
               ) -> Tuple[Optional[np.ndarray], float]:
    """단일 정사각 마커 PnP. 반환 (T_cam_marker 4x4|None, mean reproj px)."""
    obj = marker_object_points(size_m)
    img = np.asarray(corners, dtype=np.float64).reshape(4, 1, 2)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, _D(D), flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None, float("nan")
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec.ravel()
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, _D(D))
    err = float(np.linalg.norm(proj.reshape(4, 2) - np.asarray(corners).reshape(4, 2), axis=1).mean())
    return T, err


def marker_world_pose(T_cam_marker: np.ndarray, T_world_cam: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """T_world_cam = cam{i}_T_cam_to_world. 반환 (center(3), outward_normal(3), T_world_marker)."""
    T_world_marker = np.asarray(T_world_cam, float) @ np.asarray(T_cam_marker, float)
    center = T_world_marker[:3, 3].copy()
    normal = T_world_marker[:3, 2].copy()
    normal /= (np.linalg.norm(normal) + 1e-12)
    return center, normal, T_world_marker


# ---------------------------------------------------------------- #
# Multi-view corner triangulation (base 좌표계, 넓은 baseline → 강건 depth)
# ---------------------------------------------------------------- #

def _undistort_normalized(corners: np.ndarray, K: np.ndarray, D) -> np.ndarray:
    pts = np.asarray(corners, dtype=np.float64).reshape(-1, 1, 2)
    und = cv2.undistortPoints(pts, K, _D(D))
    return und.reshape(-1, 2)


def _world_to_cam_P(T_world_cam: np.ndarray) -> np.ndarray:
    """정규화 좌표용 3x4 projection (world→cam)."""
    T_cam_world = np.linalg.inv(np.asarray(T_world_cam, float))
    return T_cam_world[:3, :]


def triangulate_dlt(norm_pts: np.ndarray, Ps: Sequence[np.ndarray]) -> np.ndarray:
    """정규화 이미지 좌표 + world→cam P 들로 DLT 삼각측량 → world 3D."""
    rows = []
    for (x, y), P in zip(norm_pts, Ps):
        rows.append(x * P[2] - P[0])
        rows.append(y * P[2] - P[1])
    A = np.stack(rows, axis=0)
    _, _, Vt = np.linalg.svd(A)
    Xh = Vt[-1]
    if abs(Xh[3]) < 1e-12:
        return np.array([np.nan, np.nan, np.nan])
    return Xh[:3] / Xh[3]


def marker_world_from_triangulation(obs_list: Sequence[dict]
                                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """obs_list: [{"corners","K","D","T_world_cam"}...] (len>=2).
    반환 (center(3), normal(3), corners3d(4,3), mean_edge_len_m).
    """
    corners3d = np.zeros((4, 3), dtype=np.float64)
    for k in range(4):
        pts, Ps = [], []
        for o in obs_list:
            und = _undistort_normalized(o["corners"], o["K"], o["D"])
            pts.append(und[k])
            Ps.append(_world_to_cam_P(o["T_world_cam"]))
        corners3d[k] = triangulate_dlt(np.asarray(pts), Ps)
    center = corners3d.mean(axis=0)
    cc = corners3d - center
    _, _, Vt = np.linalg.svd(cc)
    normal = Vt[2]
    normal /= (np.linalg.norm(normal) + 1e-12)
    edges = [np.linalg.norm(corners3d[i] - corners3d[(i + 1) % 4]) for i in range(4)]
    return center, normal, corners3d, float(np.mean(edges))


# ---------------------------------------------------------------- #
# Per-marker world pose  (method: triangulate | pnp | auto)
# ---------------------------------------------------------------- #

def compute_marker_poses(raw_obs: Sequence[dict], layout: MarkerLayout,
                         method: str = "auto") -> Dict[int, dict]:
    """검출 관측 → 마커별 world 포즈.

    raw_obs: [{"cam":str, "id":int, "corners":(4,2), "K":3x3, "D":(5,)|None,
               "T_world_cam":4x4}, ...]
    method: 'triangulate'(≥2뷰 필요) | 'pnp' | 'auto'(≥2뷰면 삼각측량, 아니면 PnP).
    반환 {id -> {center, normal, method, n_views, cams, reproj_px_mean,
                 cross_cam_spread_mm, marker_edge_err_mm(옵션)}}.
    """
    if method not in ("triangulate", "pnp", "auto"):
        raise ValueError(f"method must be triangulate|pnp|auto, got {method!r}")

    by_id: Dict[int, List[dict]] = {}
    for o in raw_obs:
        by_id.setdefault(int(o["id"]), []).append(o)

    per: Dict[int, dict] = {}
    for mid, lst in by_id.items():
        size = layout.size_for(mid)

        # --- per-view PnP (항상 계산: cross-cam spread / reproj 진단 + PnP/fallback용) ---
        pnp_centers, pnp_normals, reprojs = [], [], []
        for o in lst:
            T_cm, rp = pnp_marker(o["corners"], size, o["K"], o.get("D"))
            if T_cm is None:
                continue
            c, n, _ = marker_world_pose(T_cm, o["T_world_cam"])
            pnp_centers.append(c)
            pnp_normals.append(n)
            reprojs.append(rp)
        if not pnp_centers:
            continue
        pnp_centers = np.array(pnp_centers)
        spread_mm = (max(np.linalg.norm(a - b) for a, b in combinations(pnp_centers, 2)) * 1000.0
                     if len(pnp_centers) > 1 else 0.0)

        use_tri = (method == "triangulate") or (method == "auto" and len(lst) >= 2)
        if method == "triangulate" and len(lst) < 2:
            # 삼각측량 강제인데 단일 뷰 → 측정 불가로 표시
            per[mid] = {"center": None, "normal": None, "method": "triangulate",
                        "n_views": len(lst), "cams": [o["cam"] for o in lst],
                        "reproj_px_mean": float(np.mean(reprojs)),
                        "cross_cam_spread_mm": float(spread_mm),
                        "error": "triangulate needs >=2 views"}
            continue

        edge_err_mm = None
        if use_tri:
            center, normal, _c3d, edge_len = marker_world_from_triangulation(lst)
            method_used = "triangulate"
            edge_err_mm = abs(edge_len - size) * 1000.0
        else:
            center = pnp_centers.mean(axis=0)
            normal = np.mean(pnp_normals, axis=0)
            normal /= (np.linalg.norm(normal) + 1e-12)
            method_used = "pnp"

        per[mid] = {
            "center": center,
            "normal": normal,
            "method": method_used,
            "n_views": len(lst),
            "cams": [o["cam"] for o in lst],
            "reproj_px_mean": float(np.mean(reprojs)),
            "cross_cam_spread_mm": float(spread_mm),   # PnP 뷰 간 center 산포(캘리브 자기일관성)
            "marker_edge_err_mm": edge_err_mm,          # 삼각측량 변길이 vs 인쇄크기(스케일 체크)
        }
    return per


# ---------------------------------------------------------------- #
# Axis extent solver  (GT box size)
# ---------------------------------------------------------------- #

def solve_axis_extents(per_marker: Dict[int, dict], layout: MarkerLayout) -> dict:
    """축별 extent 산출.  per_marker[id] 는 {"center","normal", ...} (center None 이면 제외)."""
    usable = {m: v for m, v in per_marker.items() if v.get("center") is not None}
    axes: Dict[str, dict] = {}
    gt: Dict[str, Optional[float]] = {}

    for axis in _VALID_AXES:
        plus_ids = [m for m, (a, s) in layout.marker_axis_side.items()
                    if a == axis and s == "+" and m in usable]
        minus_ids = [m for m, (a, s) in layout.marker_axis_side.items()
                     if a == axis and s == "-" and m in usable]

        if not plus_ids or not minus_ids:
            axes[axis] = {"extent_m": None,
                          "reason": "need >=1 usable marker on BOTH + and - faces",
                          "n_plus": len(plus_ids), "n_minus": len(minus_ids),
                          "label": layout.axis_labels.get(axis, axis)}
            gt[axis] = None
            continue

        c_plus = np.mean([usable[m]["center"] for m in plus_ids], axis=0)
        c_minus = np.mean([usable[m]["center"] for m in minus_ids], axis=0)
        d_vec = c_plus - c_minus
        center_dist = float(np.linalg.norm(d_vec))
        d_hat = d_vec / (center_dist + 1e-12)

        aligned = []
        for m in plus_ids + minus_ids:
            n = usable[m]["normal"]
            aligned.append(n if float(np.dot(n, d_hat)) >= 0 else -n)
        n_axis = np.mean(aligned, axis=0)
        n_axis /= (np.linalg.norm(n_axis) + 1e-12)

        extent_perp = abs(float(np.dot(d_vec, n_axis)))
        offset = 0.5 * (layout.offset_for(plus_ids[0]) + layout.offset_for(minus_ids[0]))
        extent = extent_perp - 2.0 * offset

        in_plane_mm = float(np.sqrt(max(center_dist**2 - extent_perp**2, 0.0))) * 1000.0
        tilt_deg = float(np.degrees(np.arccos(np.clip(abs(np.dot(d_hat, n_axis)), 0.0, 1.0))))
        spread = max(usable[m]["cross_cam_spread_mm"] for m in plus_ids + minus_ids)
        methods = sorted({usable[m]["method"] for m in plus_ids + minus_ids})

        axes[axis] = {
            "extent_m": extent,
            "extent_mm": extent * 1000.0,
            "center_to_center_mm": center_dist * 1000.0,
            "in_plane_offset_mm": in_plane_mm,
            "face_tilt_deg": tilt_deg,
            "surface_offset_m": offset,
            "method": methods[0] if len(methods) == 1 else "mixed",
            "n_plus": len(plus_ids), "n_minus": len(minus_ids),
            "plus_ids": sorted(plus_ids), "minus_ids": sorted(minus_ids),
            "max_cross_cam_spread_mm": float(spread),
            "label": layout.axis_labels.get(axis, axis),
        }
        gt[axis] = extent

    have_all = all(gt[a] is not None for a in _VALID_AXES)
    xyz = [gt[a] for a in _VALID_AXES] if have_all else None
    sorted_desc = sorted([v * 1000.0 for v in xyz], reverse=True) if have_all else None
    return {"axes": axes, "gt_extents_m": gt, "gt_extents_m_xyz": xyz,
            "gt_extents_mm_sorted_desc": sorted_desc}


# ---------------------------------------------------------------- #
# Compare GT vs estimated OBB extents
# ---------------------------------------------------------------- #

def compare_extents(gt_extents_m: Sequence[float], est_extents_m: Sequence[float]) -> dict:
    """내림차순 정렬 후 rank 대응 비교(OBB 축 순서 임의라 정렬 비교가 공정).
    signed err = estimate - ground_truth."""
    gt = np.sort(np.asarray(gt_extents_m, float))[::-1]
    est = np.sort(np.asarray(est_extents_m, float))[::-1]
    if gt.shape != (3,) or est.shape != (3,):
        raise ValueError(f"need 3 extents each, got gt={gt.shape}, est={est.shape}")
    gt_mm, est_mm = gt * 1000.0, est * 1000.0
    signed = est_mm - gt_mm
    abs_mm = np.abs(signed)
    pct = 100.0 * abs_mm / np.where(gt_mm != 0, gt_mm, np.nan)
    return {
        "gt_extents_mm_sorted_desc": gt_mm.tolist(),
        "est_extents_mm_sorted_desc": est_mm.tolist(),
        "signed_error_mm": signed.tolist(),
        "abs_error_mm": abs_mm.tolist(),
        "abs_error_pct": pct.tolist(),
        "mae_mm": float(abs_mm.mean()),
        "max_abs_error_mm": float(abs_mm.max()),
        "rmse_mm": float(np.sqrt(np.mean(signed ** 2))),
        "mean_abs_error_pct": float(np.nanmean(pct)),
    }


# ---------------------------------------------------------------- #
# Estimation-pipeline OBB extents loader
# ---------------------------------------------------------------- #

def load_estimated_extents_m(estimate_json: Path | str) -> Tuple[List[float], str]:
    """추정 파이프라인 출력에서 OBB extents(m, 3개) 로드.
    지원 키: bbox_extents_m, extents_m, extents_mm_sorted_desc, target_obb_extents_mm_desc.
    반환 (extents_m, 사용 key)."""
    with open(estimate_json, "r", encoding="utf-8") as f:
        d = json.load(f)
    for key in ("bbox_extents_m", "extents_m"):
        if key in d and len(d[key]) == 3:
            return [float(v) for v in d[key]], key
    for key in ("extents_mm_sorted_desc", "target_obb_extents_mm_desc"):
        if key in d and len(d[key]) == 3:
            return [float(v) / 1000.0 for v in d[key]], key
    raise KeyError(f"{estimate_json}: no known extents key "
                   f"(bbox_extents_m/extents_m/extents_mm_sorted_desc/target_obb_extents_mm_desc)")
