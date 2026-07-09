#!/usr/bin/env python3
"""
Gt_Step2_measure_marker_box.py

부착 마커로 peg/hole 의 **정답 직육면체 크기(x/y/z extent)** 를 측정하고, (옵션으로)
추정 파이프라인의 OBB 크기(bbox_extents_m)와 비교한다.

[입력]  Gt_Step1 이 저장한 캡처 폴더 (하나 이상):
          cam{i}_rgb.png, cam{i}_K.txt, cam{i}_T_cam_to_world.txt  (+calib_info.json)
        distortion(D)은 --intrinsics_dir/cam{i}.npz(color_D)에서 로드(없으면 0 가정).

[측정]  각 카메라에서 마커 검출 → 방법(method) 로 마커 world 포즈 산출 → 축별 extent.
          method=triangulate : ≥2 뷰 마커를 삼각측량(권장, sub-mm)
          method=pnp         : 단일마커 PnP 평균(뷰축 depth 약함)
          method=auto        : ≥2뷰면 삼각측량, 아니면 PnP (기본)
        한 축의 +,- 면이 한 폴더에 다 안 보이면, 물체를 돌려 찍은 **다른 폴더**를
        같이 넘기면 축별로 병합된다(각 축 extent = 측정된 폴더들의 평균).

[비교]  --estimate_json 제공 시 (예: data/outputs_set1/obj1/obj1_bbox_metric.json)
        GT vs 추정 extents 를 내림차순 정렬해 rank 대응으로 오차(mm/%) 산출.

사용 예:
  # 단일 폴더, 추정치와 비교
  python Gt_Step2_measure_marker_box.py \
    --capture_dir data/gt_capture_peg_set1 \
    --layout      gt_marker_layout_example.json \
    --intrinsics_dir intrinsics \
    --estimate_json data/outputs_set1/obj1/obj1_bbox_metric.json \
    --label peg --out_dir data/gt_results/peg

  # 물체를 돌려 여러 번 찍은 경우 폴더 여러 개
  python Gt_Step2_measure_marker_box.py \
    --capture_dir data/gt_capture_peg_a data/gt_capture_peg_b \
    --layout gt_marker_layout_example.json --label peg

[출력]  out_dir/gt_size_report.json   (+ 콘솔 표)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
for _p in (str(THIS_DIR.parent), str(THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _marker_box as mb  # noqa: E402


def _load_D(intr_dir: Optional[Path], cam_idx: int) -> Optional[np.ndarray]:
    if intr_dir is None:
        return None
    npz = intr_dir / f"cam{cam_idx}.npz"
    if not npz.exists():
        return None
    try:
        d = np.load(npz, allow_pickle=True)
        if "color_D" in d.files:
            return np.asarray(d["color_D"], dtype=np.float64).reshape(-1)
    except Exception as e:
        print(f"[WARN] cam{cam_idx} color_D load 실패: {e}")
    return None


def _cam_index(cam_id: str) -> int:
    m = re.search(r"(\d+)", cam_id)
    return int(m.group(1)) if m else -1


def measure_one_capture(cap_dir: Path, layout: mb.MarkerLayout, detector,
                        intr_dir: Optional[Path], method: str,
                        save_overlay: bool = True) -> dict:
    """한 캡처 폴더 → 마커 검출·측정 결과 dict."""
    rgbs = sorted(cap_dir.glob("cam*_rgb.png"))
    if not rgbs:
        raise FileNotFoundError(f"{cap_dir}: cam*_rgb.png 없음")

    world_frame = "unknown"
    ci_path = cap_dir / "calib_info.json"
    if ci_path.exists():
        try:
            world_frame = json.loads(ci_path.read_text()).get("world_frame", "unknown")
        except Exception:
            pass

    raw_obs: List[dict] = []
    per_cam_detect: Dict[str, list] = {}
    for rgb_p in rgbs:
        cam_id = rgb_p.stem.replace("_rgb", "")
        ci = _cam_index(cam_id)
        K_p = cap_dir / f"{cam_id}_K.txt"
        T_p = cap_dir / f"{cam_id}_T_cam_to_world.txt"
        if not (K_p.exists() and T_p.exists()):
            print(f"[WARN] {cam_id}: K/T 파일 없음 → skip")
            continue
        K = np.loadtxt(K_p, dtype=np.float64).reshape(3, 3)
        T_world_cam = np.loadtxt(T_p, dtype=np.float64).reshape(4, 4)
        D = _load_D(intr_dir, ci)
        img = cv2.imread(str(rgb_p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] {rgb_p} 읽기 실패 → skip")
            continue
        dets = mb.detect_markers(img, detector)
        per_cam_detect[cam_id] = [(mid, c) for mid, c in dets if mid in layout.known_ids()]
        for mid, corners in dets:
            if mid in layout.known_ids():
                raw_obs.append({"cam": cam_id, "id": mid, "corners": corners,
                                "K": K, "D": D, "T_world_cam": T_world_cam})
        if save_overlay and dets:
            ov = img.copy()
            cv2.aruco.drawDetectedMarkers(
                ov, [c.reshape(1, 4, 2).astype(np.float32) for _, c in dets],
                np.array([[mid] for mid, _ in dets]))
            cv2.imwrite(str(cap_dir / f"gt_detect_{cam_id}.png"), ov)

    per_marker = mb.compute_marker_poses(raw_obs, layout, method=method)
    res = mb.solve_axis_extents(per_marker, layout)

    # 마커별 진단(직렬화 가능한 형태로)
    marker_diag = {}
    for mid, v in per_marker.items():
        marker_diag[str(mid)] = {
            "axis_side": list(layout.marker_axis_side[mid]),
            "method": v.get("method"), "n_views": v.get("n_views"),
            "cams": v.get("cams"),
            "reproj_px_mean": round(v.get("reproj_px_mean", float("nan")), 4),
            "cross_cam_spread_mm": round(v.get("cross_cam_spread_mm", 0.0), 3),
            "marker_edge_err_mm": (round(v["marker_edge_err_mm"], 3)
                                   if v.get("marker_edge_err_mm") is not None else None),
            "usable": v.get("center") is not None,
        }
    return {"capture_dir": str(cap_dir), "world_frame": world_frame,
            "n_cameras": len(per_cam_detect), "axes": res["axes"],
            "gt_extents_m": res["gt_extents_m"], "markers": marker_diag}


def merge_axes(per_capture: List[dict]) -> dict:
    """폴더별 축 측정 → 축별 병합(측정된 폴더 평균)."""
    merged = {}
    for axis in ("x", "y", "z"):
        vals, srcs = [], []
        for pc in per_capture:
            e = pc["gt_extents_m"].get(axis)
            if e is not None:
                vals.append(e)
                srcs.append({"capture_dir": pc["capture_dir"],
                             "extent_mm": round(pc["axes"][axis]["extent_mm"], 3),
                             "method": pc["axes"][axis].get("method"),
                             "in_plane_offset_mm": round(pc["axes"][axis]["in_plane_offset_mm"], 3),
                             "face_tilt_deg": round(pc["axes"][axis]["face_tilt_deg"], 3),
                             "max_cross_cam_spread_mm": round(pc["axes"][axis]["max_cross_cam_spread_mm"], 3)})
        if vals:
            merged[axis] = {"extent_m": float(np.mean(vals)),
                            "extent_mm": float(np.mean(vals) * 1000.0),
                            "n_measurements": len(vals),
                            "spread_across_captures_mm": (float((max(vals) - min(vals)) * 1000.0)
                                                          if len(vals) > 1 else 0.0),
                            "sources": srcs}
        else:
            merged[axis] = {"extent_m": None, "n_measurements": 0,
                            "reason": "어느 캡처에서도 +,- 면이 함께 검출되지 않음"}
    have_all = all(merged[a]["extent_m"] is not None for a in ("x", "y", "z"))
    xyz = [merged[a]["extent_m"] for a in ("x", "y", "z")] if have_all else None
    return {"per_axis": merged,
            "gt_extents_m_xyz": xyz,
            "gt_extents_mm_sorted_desc": (sorted([v * 1000 for v in xyz], reverse=True)
                                          if have_all else None)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture_dir", nargs="+", required=True, type=Path,
                    help="Gt_Step1 캡처 폴더(들). 물체 돌려 여러 번 찍었으면 여러 개.")
    ap.add_argument("--layout", required=True, type=Path)
    ap.add_argument("--intrinsics_dir", type=Path, default=Path("intrinsics"),
                    help="cam{i}.npz(color_D) 위치. 없으면 D=0 가정.")
    ap.add_argument("--method", choices=["auto", "triangulate", "pnp"], default="auto")
    ap.add_argument("--estimate_json", type=Path, default=None,
                    help="비교할 추정 OBB json (bbox_extents_m 등).")
    ap.add_argument("--label", default="object", help="peg/hole 등 표시용 라벨.")
    ap.add_argument("--out_dir", type=Path, default=None,
                    help="report 저장 폴더 (기본: 첫 capture_dir).")
    ap.add_argument("--no_overlay", action="store_true", help="검출 오버레이 png 저장 안 함.")
    args = ap.parse_args()

    layout = mb.load_layout(args.layout)
    detector = mb.build_detector(layout.dictionary)
    intr_dir = args.intrinsics_dir if args.intrinsics_dir and args.intrinsics_dir.exists() else None
    if intr_dir is None:
        print("[WARN] intrinsics_dir 없음 → distortion D=0 으로 가정(정확도 영향 가능).")

    per_capture = []
    for cd in args.capture_dir:
        if not cd.exists():
            print(f"[WARN] {cd} 없음 → skip")
            continue
        print(f"[INFO] measuring {cd} (method={args.method}) ...")
        per_capture.append(measure_one_capture(cd, layout, detector, intr_dir,
                                               args.method, save_overlay=not args.no_overlay))
    if not per_capture:
        ap.error("측정 가능한 캡처 폴더가 없음")

    merged = merge_axes(per_capture)

    report = {
        "label": args.label,
        "layout": str(args.layout),
        "method": args.method,
        "capture_dirs": [str(c) for c in args.capture_dir],
        "world_frame": per_capture[0]["world_frame"],
        "ground_truth": merged,
        "per_capture": per_capture,
    }

    # ---- 비교 ----
    comparison = None
    if args.estimate_json is not None:
        if merged["gt_extents_m_xyz"] is None:
            print("[WARN] 세 축이 다 측정되지 않아 비교 생략. (누락 축을 위해 추가 캡처 필요)")
        else:
            est_m, est_key = mb.load_estimated_extents_m(args.estimate_json)
            comparison = mb.compare_extents(merged["gt_extents_m_xyz"], est_m)
            comparison["estimate_json"] = str(args.estimate_json)
            comparison["estimate_key"] = est_key
            report["comparison"] = comparison

    out_dir = args.out_dir or args.capture_dir[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "gt_size_report.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # ---------------- 콘솔 리포트 ----------------
    print("\n" + "=" * 66)
    print(f" GROUND-TRUTH SIZE  [{args.label}]   method={args.method}")
    print("=" * 66)
    labels = {a: layout.axis_labels.get(a, a) for a in ("x", "y", "z")}
    for a in ("x", "y", "z"):
        m = merged["per_axis"][a]
        if m["extent_m"] is None:
            print(f"  {a} ({labels[a]:>7}): --- 측정 불가 ({m.get('reason','')})")
        else:
            extra = ""
            if m["n_measurements"] > 1:
                extra = f"  (n={m['n_measurements']}, 폴더간 편차 {m['spread_across_captures_mm']:.2f}mm)"
            worst_tilt = max(s["face_tilt_deg"] for s in m["sources"])
            worst_spread = max(s["max_cross_cam_spread_mm"] for s in m["sources"])
            print(f"  {a} ({labels[a]:>7}): {m['extent_mm']:7.2f} mm{extra}")
            print(f"        └ tilt≤{worst_tilt:.2f}°  cross-cam spread≤{worst_spread:.2f}mm")

    # 마커 스케일 자기검증 (삼각측량시)
    edge_errs = [md["marker_edge_err_mm"] for pc in per_capture for md in pc["markers"].values()
                 if md["marker_edge_err_mm"] is not None]
    if edge_errs:
        print(f"\n  마커 변길이 self-check: 평균 {np.mean(edge_errs):.3f}mm, "
              f"최대 {np.max(edge_errs):.3f}mm (0에 가까울수록 스케일 신뢰)")

    if comparison is not None:
        gt = comparison["gt_extents_mm_sorted_desc"]
        est = comparison["est_extents_mm_sorted_desc"]
        print("\n" + "-" * 66)
        print(f" COMPARE vs ESTIMATE  ({Path(comparison['estimate_json']).name} :: {comparison['estimate_key']})")
        print("-" * 66)
        print(f"  {'rank':>4} {'GT(mm)':>10} {'EST(mm)':>10} {'Δmm':>9} {'Δ%':>8}")
        for i in range(3):
            print(f"  {i+1:>4} {gt[i]:>10.2f} {est[i]:>10.2f} "
                  f"{comparison['signed_error_mm'][i]:>+9.2f} {comparison['abs_error_pct'][i]:>7.2f}%")
        print(f"  {'':>4} {'':>10} {'MAE':>10} {comparison['mae_mm']:>9.2f} "
              f"{comparison['mean_abs_error_pct']:>7.2f}%")
        print(f"  {'':>4} {'':>10} {'RMSE':>10} {comparison['rmse_mm']:>9.2f}")
        print(f"  {'':>4} {'':>10} {'MAXerr':>10} {comparison['max_abs_error_mm']:>9.2f}")

    print(f"\n[SAVE] {out_json}")


if __name__ == "__main__":
    main()
