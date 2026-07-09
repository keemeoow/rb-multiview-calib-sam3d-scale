#!/usr/bin/env python3
"""
test_gt_pipeline.py — Gt_Step2_measure_marker_box.py 통합 테스트 (하드웨어 불필요).

합성 캡처 폴더(렌더된 마커 PNG + K/T + calib_info)를 만들고, 실제 Gt_Step2 스크립트를
subprocess 로 돌려 gt_size_report.json 을 검증한다.
  (1) 단일 폴더 전축 측정 + 추정치 비교
  (2) 물체를 돌려 두 번 찍은(폴더 2개) 축별 병합

실행:  python3 test_gt_pipeline.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_marker_box as tmb   # render/scene helpers 재사용
import _marker_box as mb

DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
K = np.array([[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1.0]])


def octant_eyes(C, R_box, d=0.42):
    eyes = []
    for sx in (+1, -1):
        for sy in (+1, -1):
            for sz in (+1, -1):
                v = sx * R_box[:, 0] + sy * R_box[:, 1] + sz * R_box[:, 2]
                eyes.append((f"{sx:+d}{sy:+d}{sz:+d}", C + d * (v / np.linalg.norm(v))))
    return eyes


def gen_capture(dirpath: Path, E, C, R_box, layout, eyes, only_ids=None):
    dirpath.mkdir(parents=True, exist_ok=True)
    ax = {"x": R_box[:, 0], "y": R_box[:, 1], "z": R_box[:, 2]}
    aidx = {"x": 0, "y": 1, "z": 2}
    Twm = {}
    for mid, (a, s) in layout.marker_axis_side.items():
        sign = 1.0 if s == "+" else -1.0
        Rm = tmb.marker_frame_from_normal(sign * ax[a])
        T = np.eye(4); T[:3, :3] = Rm; T[:3, 3] = C + sign * (E[aidx[a]] / 2.0) * ax[a]
        Twm[mid] = T
    cams_info = []
    for i, (_name, eye) in enumerate(eyes):
        Twc = tmb.look_at_T_world_cam(eye, C)
        Tcw = np.linalg.inv(Twc)
        img = np.full((480, 640, 3), 255, np.uint8)
        for mid in layout.marker_axis_side:
            if only_ids is not None and mid not in only_ids:
                continue
            tmb.render_marker(img, DICT, mid, Tcw @ Twm[mid], layout.size_for(mid), K)
        cv2.imwrite(str(dirpath / f"cam{i}_rgb.png"), img)
        np.savetxt(dirpath / f"cam{i}_K.txt", K)
        np.savetxt(dirpath / f"cam{i}_T_cam_to_world.txt", Twc)
        cams_info.append({"cam_id": i, "serial": f"SYN{i}",
                          "depth_scale_m_per_unit": 0.001, "K_source": "color_K"})
    (dirpath / "calib_info.json").write_text(json.dumps(
        {"ref_cam_idx": 0, "world_frame": "robot_base", "cameras": cams_info}))


def run_gt2(args):
    no_intr = str(HERE / "__nonexistent_intr__")   # → D=0 가정(합성이 무왜곡이라 일치)
    cmd = [sys.executable, str(HERE / "Gt_Step2_measure_marker_box.py"),
           "--intrinsics_dir", no_intr, "--no_overlay"] + args
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-1500:])
    if r.returncode != 0:
        print(r.stderr[-2000:])
        raise RuntimeError("Gt_Step2 failed")
    return r


def test_single_dir_with_compare(tmp):
    print("\n[1] single capture dir, all axes + compare")
    layout_path = HERE / "gt_marker_layout_example.json"
    # layout_example 은 30mm; 합성은 40mm 로 렌더하므로 size 맞춘 임시 layout 작성
    layout = mb.load_layout(layout_path)
    layout.marker_size_m = 0.040
    lp = tmp / "layout40.json"
    lp.write_text(json.dumps({
        "dictionary": "DICT_4X4_50", "marker_size_m": 0.040, "surface_offset_m": 0.0,
        "axis_labels": {"x": "length", "y": "width", "z": "height"},
        "markers": {"10": {"axis": "x", "side": "+"}, "11": {"axis": "x", "side": "-"},
                    "12": {"axis": "y", "side": "+"}, "13": {"axis": "y", "side": "-"},
                    "14": {"axis": "z", "side": "+"}, "15": {"axis": "z", "side": "-"}}}))

    E = np.array([0.130, 0.090, 0.070])       # truth (x,y,z)
    C = np.array([0.05, 0.45, 0.10])
    R_box = tmb.rot_xyz(0.15, 0.25, -0.35)
    cap = tmp / "cap_single"
    gen_capture(cap, E, C, R_box, mb.load_layout(lp), octant_eyes(C, R_box))

    # 추정치: truth 대비 각 축 다르게 오차 주고(정렬 비교), 순서도 섞어서 저장
    est = {"bbox_extents_m": [0.072, 0.126, 0.093]}   # ~ z+2, x-4, y+3 (정렬 후 대응)
    est_path = tmp / "obj_bbox_metric.json"
    est_path.write_text(json.dumps(est))

    run_gt2(["--capture_dir", str(cap), "--layout", str(lp),
             "--estimate_json", str(est_path), "--label", "peg",
             "--method", "triangulate", "--out_dir", str(cap)])

    rep = json.loads((cap / "gt_size_report.json").read_text())
    gt_xyz = np.array(rep["ground_truth"]["gt_extents_m_xyz"])
    err = np.abs(gt_xyz - E) * 1000.0
    print(f"    GT xyz mm = {np.round(gt_xyz*1000,3)}  err mm = {np.round(err,3)}")
    assert np.all(err < 1.0), f"GT extent err too large: {err}"

    cmp = rep["comparison"]
    gt_desc = np.sort(E)[::-1] * 1000.0
    est_desc = np.sort([0.072, 0.126, 0.093])[::-1] * 1000.0
    assert np.allclose(cmp["gt_extents_mm_sorted_desc"], gt_desc, atol=1.0)
    assert np.allclose(cmp["est_extents_mm_sorted_desc"], est_desc, atol=1e-6)
    exp_signed = est_desc - gt_desc
    assert np.allclose(cmp["signed_error_mm"], exp_signed, atol=1.0), \
        f"signed err {cmp['signed_error_mm']} vs {exp_signed}"
    print(f"    compare Δmm(signed) = {np.round(cmp['signed_error_mm'],2)}  "
          f"MAE={cmp['mae_mm']:.2f}  RMSE={cmp['rmse_mm']:.2f}  MAX={cmp['max_abs_error_mm']:.2f}")
    print("    PASS")


def test_multi_dir_merge(tmp):
    print("\n[2] two capture dirs merged per-axis (object repositioned)")
    lp = tmp / "layout40.json"      # 재사용
    layout = mb.load_layout(lp)
    E = np.array([0.130, 0.090, 0.070])

    # dir A: 물체 자세 1 — x,y 축 면만 마커 보이게 (z 마커 렌더 제외)
    C1 = np.array([0.0, 0.40, 0.12]); R1 = tmb.rot_xyz(0.1, 0.2, -0.3)
    capA = tmp / "cap_A"
    gen_capture(capA, E, C1, R1, layout, octant_eyes(C1, R1), only_ids={10, 11, 12, 13})
    # dir B: 물체 자세 2(다른 위치/회전) — z 축 면만
    C2 = np.array([0.10, 0.50, 0.08]); R2 = tmb.rot_xyz(-0.2, 0.4, 0.5)
    capB = tmp / "cap_B"
    gen_capture(capB, E, C2, R2, layout, octant_eyes(C2, R2), only_ids={14, 15})

    run_gt2(["--capture_dir", str(capA), str(capB), "--layout", str(lp),
             "--label", "hole", "--method", "triangulate", "--out_dir", str(capB)])

    rep = json.loads((capB / "gt_size_report.json").read_text())
    pa = rep["ground_truth"]["per_axis"]
    print(f"    x from {pa['x']['n_measurements']} cap, z from {pa['z']['n_measurements']} cap")
    gt_xyz = np.array(rep["ground_truth"]["gt_extents_m_xyz"])
    err = np.abs(gt_xyz - E) * 1000.0
    print(f"    merged GT xyz mm = {np.round(gt_xyz*1000,3)}  err mm = {np.round(err,3)}")
    # x,y 는 capA 에서만, z 는 capB 에서만 측정됐어야
    assert pa["x"]["n_measurements"] == 1 and pa["z"]["n_measurements"] == 1
    assert np.all(err < 1.0), f"merged GT err too large: {err}"
    print("    PASS")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        test_single_dir_with_compare(tmp)
        test_multi_dir_merge(tmp)
    print("\nGT PIPELINE TESTS PASSED ✅")
