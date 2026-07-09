#!/usr/bin/env python3
"""
test_marker_box.py — _marker_box.py 합성 데이터 자기검증 (하드웨어 불필요).

  (A) 순수 기하 solver: 회전 박스 + 면중앙 이탈 + 스탠드오프 → extent 정확 복원
  (B) 엔드투엔드: 실제 ArUco 마커를 합성 카메라(8뷰)에 렌더 → 실제 detector →
      삼각측량/PnP → world → 축 solver → GT extents 복원 + 비교
      · triangulate(멀티뷰) = sub-mm 이어야 함
      · pnp(단일마커 평균)  = 더 부정확(뷰축 depth 약함) — 코드 경로 동작만 확인

실행:  python3 test_marker_box.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _marker_box as mb


def rot_xyz(rx, ry, rz):
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def marker_frame_from_normal(nrm):
    nrm = nrm / np.linalg.norm(nrm)
    a = np.array([0.0, 0.0, 1.0]) if abs(nrm[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(a, nrm); u /= np.linalg.norm(u)
    v = np.cross(nrm, u)
    return np.column_stack([u, v, nrm])


def look_at_T_world_cam(eye, target):
    f = np.asarray(target, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f)
    a = np.array([0.0, 0.0, 1.0]) if abs(f[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    r = np.cross(a, f); r /= np.linalg.norm(r)
    u = np.cross(f, r)
    T = np.eye(4); T[:3, :3] = np.column_stack([r, u, f]); T[:3, 3] = eye
    return T


def render_marker(img, dictionary, mid, T_cam_marker, size_m, K, marker_px=240):
    """마커를 카메라 이미지에 워프 합성. 카메라를 충분히 향한 면만 그림."""
    R = T_cam_marker[:3, :3]
    t = T_cam_marker[:3, 3]
    if R[2, 2] >= -0.35:                       # 면 법선이 카메라를 향하는 각도 제한(≲70°)
        return False
    obj = mb.marker_object_points(size_m)
    cam_pts = (R @ obj.T).T + t
    if np.any(cam_pts[:, 2] <= 1e-3):
        return False
    rvec, _ = cv2.Rodrigues(R)
    proj, _ = cv2.projectPoints(obj, rvec, t.reshape(3, 1), K, None)
    dst = proj.reshape(4, 2)
    H, W = img.shape[:2]
    if (dst[:, 0].min() < 2 or dst[:, 0].max() > W - 3 or
            dst[:, 1].min() < 2 or dst[:, 1].max() > H - 3):
        return False
    # 너무 작으면 검출 실패 → skip
    side_px = np.linalg.norm(dst[0] - dst[1])
    if side_px < 18:
        return False
    m = cv2.aruco.generateImageMarker(dictionary, mid, marker_px)
    qz = marker_px // 4
    padded = np.full((marker_px + 2 * qz, marker_px + 2 * qz), 255, np.uint8)
    padded[qz:qz + marker_px, qz:qz + marker_px] = m
    src = np.array([[qz, qz], [qz + marker_px, qz],
                    [qz + marker_px, qz + marker_px], [qz, qz + marker_px]], np.float64)
    Hmg, _ = cv2.findHomography(src, dst)
    warped = cv2.warpPerspective(cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR), Hmg, (W, H),
                                 borderValue=(255, 255, 255))
    mask = cv2.warpPerspective(np.full_like(padded, 255), Hmg, (W, H), borderValue=0) > 0
    img[mask] = warped[mask]
    return True


def _make_layout(size):
    return mb.MarkerLayout(
        dictionary="DICT_4X4_50", marker_size_m=size,
        marker_axis_side={10: ("x", "+"), 11: ("x", "-"),
                          12: ("y", "+"), 13: ("y", "-"),
                          14: ("z", "+"), 15: ("z", "-")},
        axis_labels={"x": "length", "y": "width", "z": "height"})


# ---------------------------------------------------------------- #
# (A) pure geometry solver
# ---------------------------------------------------------------- #

def test_geometry_solver():
    print("\n[A] pure geometry solver")
    R_box = rot_xyz(0.3, -0.5, 0.8)
    C = np.array([0.10, -0.20, 0.35])
    E = np.array([0.120, 0.080, 0.050])
    ax = {"x": R_box[:, 0], "y": R_box[:, 1], "z": R_box[:, 2]}
    aidx = {"x": 0, "y": 1, "z": 2}

    layout = _make_layout(0.03)
    layout.offset_override = {14: 0.006, 15: 0.006}
    standoff = {14: 0.006, 15: 0.006}
    inplane = {10: 0.015 * R_box[:, 1]}          # +x 마커 면중앙 15mm 이탈

    per = {}
    for mid, (a, s) in layout.marker_axis_side.items():
        sign = 1.0 if s == "+" else -1.0
        half = E[aidx[a]] / 2.0
        off = standoff.get(mid, 0.0)
        center = C + sign * (half + off) * ax[a] + inplane.get(mid, 0.0)
        per[mid] = {"center": center, "normal": sign * ax[a], "method": "synthetic",
                    "n_views": 1, "cams": ["camA"], "reproj_px_mean": 0.1,
                    "cross_cam_spread_mm": 0.0, "marker_edge_err_mm": None}

    res = mb.solve_axis_extents(per, layout)
    got = np.array(res["gt_extents_m_xyz"])
    err_mm = np.abs(got - E) * 1000.0
    for a in "xyz":
        print(f"    {a}: truth={E[aidx[a]]*1000:6.2f}mm  got={got[aidx[a]]*1000:6.2f}mm  "
              f"err={err_mm[aidx[a]]:.4f}mm")
    print(f"    +x in-plane diag = {res['axes']['x']['in_plane_offset_mm']:.2f}mm (expect ~15)")
    assert np.all(err_mm < 1e-3), f"solver err too large: {err_mm}"
    assert abs(res["axes"]["x"]["in_plane_offset_mm"] - 15.0) < 0.5
    print("    PASS")


# ---------------------------------------------------------------- #
# (B) end-to-end
# ---------------------------------------------------------------- #

def _build_scene():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_size = 0.040
    layout = _make_layout(marker_size)
    R_box = rot_xyz(0.15, 0.25, -0.35)
    C = np.array([0.05, 0.45, 0.10])
    E = np.array([0.130, 0.090, 0.070])
    ax = {"x": R_box[:, 0], "y": R_box[:, 1], "z": R_box[:, 2]}
    aidx = {"x": 0, "y": 1, "z": 2}

    T_world_marker = {}
    for mid, (a, s) in layout.marker_axis_side.items():
        sign = 1.0 if s == "+" else -1.0
        Rm = marker_frame_from_normal(sign * ax[a])
        center = C + sign * (E[aidx[a]] / 2.0) * ax[a]
        T = np.eye(4); T[:3, :3] = Rm; T[:3, 3] = center
        T_world_marker[mid] = T

    K = np.array([[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1.0]])
    Wpx, Hpx = 640, 480
    d = 0.42
    eyes = {}
    for sx in (+1, -1):
        for sy in (+1, -1):
            for sz in (+1, -1):
                v = sx * ax["x"] + sy * ax["y"] + sz * ax["z"]
                v = v / np.linalg.norm(v)
                eyes[f"cam_{sx:+d}{sy:+d}{sz:+d}"] = C + d * v

    raw_obs = []
    n_rendered = 0
    for name, eye in eyes.items():
        T_world_cam = look_at_T_world_cam(eye, C)
        T_cam_world = np.linalg.inv(T_world_cam)
        img = np.full((Hpx, Wpx, 3), 255, np.uint8)
        for mid in layout.marker_axis_side:
            T_cm = T_cam_world @ T_world_marker[mid]
            if render_marker(img, dictionary, mid, T_cm, layout.size_for(mid), K):
                n_rendered += 1
        for mid, corners in mb.detect_markers(img, mb.build_detector("DICT_4X4_50")):
            if mid in layout.known_ids():
                raw_obs.append({"cam": name, "id": mid, "corners": corners,
                                "K": K, "D": None, "T_world_cam": T_world_cam})
    return layout, E, aidx, raw_obs, n_rendered


def test_end_to_end():
    print("\n[B] end-to-end render/detect/(tri,pnp)/world/solve  (8 oblique views)")
    layout, E, aidx, raw_obs, n_rendered = _build_scene()
    print(f"    rendered={n_rendered}, detections={len(raw_obs)}")
    views_per = {}
    for o in raw_obs:
        views_per[o["id"]] = views_per.get(o["id"], 0) + 1
    print(f"    views per marker: {dict(sorted(views_per.items()))}")
    assert len(views_per) == 6 and min(views_per.values()) >= 2

    # --- triangulate: sub-mm 기대 ---
    per_t = mb.compute_marker_poses(raw_obs, layout, method="triangulate")
    res_t = mb.solve_axis_extents(per_t, layout)
    got_t = np.array(res_t["gt_extents_m_xyz"])
    err_t = np.abs(got_t - E) * 1000.0
    edge_err = max(v["marker_edge_err_mm"] for v in per_t.values())
    for a in "xyz":
        info = res_t["axes"][a]
        print(f"    [tri] {a}({info['label']:>6}): truth={E[aidx[a]]*1000:6.2f}  "
              f"got={info['extent_mm']:6.2f}mm  err={err_t[aidx[a]]:.3f}mm  tilt={info['face_tilt_deg']:.2f}°")
    print(f"    [tri] marker edge-length self-check max err = {edge_err:.3f}mm")
    assert np.all(err_t < 1.0), f"triangulate extent err too large: {err_t} mm"
    assert edge_err < 1.0

    # --- pnp: 경로 동작 확인(정확도는 느슨) ---
    per_p = mb.compute_marker_poses(raw_obs, layout, method="pnp")
    res_p = mb.solve_axis_extents(per_p, layout)
    got_p = np.array(res_p["gt_extents_m_xyz"])
    err_p = np.abs(got_p - E) * 1000.0
    print(f"    [pnp] extents mm = {np.round(got_p*1000,2)}  err mm = {np.round(err_p,2)}  "
          f"(단일마커 depth 약함 → tri 대비 부정확)")
    assert res_p["gt_extents_m_xyz"] is not None

    # --- compare 함수: 추정이 GT+4mm 라 가정 ---
    cmp = mb.compare_extents(res_t["gt_extents_m_xyz"], (E + 0.004).tolist())
    print(f"    compare: gt_desc={np.round(cmp['gt_extents_mm_sorted_desc'],2)}  "
          f"mae={cmp['mae_mm']:.2f}mm  max={cmp['max_abs_error_mm']:.2f}mm")
    assert abs(cmp["mae_mm"] - 4.0) < 1.0
    print("    PASS (triangulate sub-mm ✔)")


if __name__ == "__main__":
    test_geometry_solver()
    test_end_to_end()
    print("\nALL TESTS PASSED ✅")
