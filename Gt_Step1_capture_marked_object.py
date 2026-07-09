#!/usr/bin/env python3
"""
Gt_Step1_capture_marked_object.py

**정답(ground-truth) 크기 실험용** 캡처.  peg/hole 에 ArUco 마커를 부착한 상태로
고정 카메라 3대의 RGB(+depth) 동기 프레임 + 캘리브(K, T_cam_to_world) 를
Obj_Step1 과 동일한 flat layout 으로 저장한다.  이후 Gt_Step2 가 이 폴더를 읽어
마커로 직육면체 크기(정답)를 측정한다.

Obj_Step1_capture_object.py 와 캡처/덤프 로직은 공유하고, 여기에 **마커 커버리지
프리뷰**를 더한다: 각 카메라 화면에 검출된 마커를 그려주고, layout 의 마커 ID 별로
"몇 대 카메라가 지금 보고 있는지"를 실시간 표로 보여준다.

  ▸ 삼각측량(권장, sub-mm)은 마커가 **≥2대** 카메라에 보여야 함.
  ▸ 한 축(axis)의 '+' 면과 '-' 면 마커가 모두 커버돼야 그 축을 측정 가능.
  ▸ 3대가 같은 쪽에서 보면 물체의 반대편 면은 안 보이므로, 필요하면 물체를
    돌려가며 **여러 번 캡처**해서(각 폴더) Gt_Step2 에 함께 넘기면 축별로 병합됨.

사용 예:
  python Gt_Step1_capture_marked_object.py \
    --out_dir         data/gt_capture_peg_set1 \
    --intrinsics_dir  intrinsics \
    --transforms_json data/handeye_session_01/T_R_Ci_all.json \
    --layout          gt_marker_layout_example.json \
    --show

저장 결과 (out_dir/):
  cam{i}_rgb.png
  cam{i}_depth.png            # --no_depth 아니면
  cam{i}_K.txt
  cam{i}_T_cam_to_world.txt
  calib_info.json
  gt_marker_coverage.json     # 저장 시점 마커별 검출 카메라 목록(진단)
  detect_cam{i}.png           # 검출 마커 오버레이(진단)

다음: python Gt_Step2_measure_marker_box.py --capture_dir out_dir --layout ... [--estimate_json ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict

import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
for _p in (str(THIS_DIR.parent), str(THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _camera import RealSenseCamera  # noqa: E402
# Obj_Step1 의 순수 캡처/덤프 헬퍼 재사용 (DRY)
from Obj_Step1_capture_object import (  # noqa: E402
    capture_depth_burst, median_depth_ignore_zero, load_device_map, dump_calib_to_flat,
)
import _marker_box as mb  # noqa: E402


def _coverage(imgs: Dict[int, np.ndarray], detector, known_ids: set):
    """{cam_id: bgr} → (marker_id -> [cam_id,...], {cam_id: [(id,corners)]})."""
    seen: Dict[int, list] = {}
    per_cam: Dict[int, list] = {}
    for ci, img in imgs.items():
        if img is None:
            per_cam[ci] = []
            continue
        dets = mb.detect_markers(img, detector)
        per_cam[ci] = dets
        for mid, _ in dets:
            if mid in known_ids:
                seen.setdefault(mid, []).append(ci)
    return seen, per_cam


def _draw_overlay(img, dets, known_ids):
    out = img.copy()
    if dets:
        corners = [c.reshape(1, 4, 2).astype(np.float32) for _, c in dets]
        ids = np.array([[mid] for mid, _ in dets])
        cv2.aruco.drawDetectedMarkers(out, corners, ids)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--intrinsics_dir", required=True)
    p.add_argument("--transforms_json", required=True,
                   help="cam{i}_K.txt / cam{i}_T_cam_to_world.txt dump 용 (T_R_Ci_all.json 등)")
    p.add_argument("--layout", required=True, help="마커 layout json (커버리지 표시용)")
    p.add_argument("--use_depth_K", action="store_true")
    p.add_argument("--no_depth", action="store_true",
                   help="정답 측정은 마커(RGB)만 쓰므로 depth 저장 생략 가능.")

    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--depth_burst_n", type=int, default=10)
    p.add_argument("--depth_burst_max_wait_ms", type=int, default=1500)
    p.add_argument("--no_align_depth_to_color", action="store_true")

    p.add_argument("--warmup_seconds", type=float, default=2.0)
    p.add_argument("--auto_save", action="store_true",
                   help="warmup 후 자동 저장. (단, 모든 마커 축 커버 확인은 사용자 책임)")
    p.add_argument("--show", action="store_true", help="프리뷰. SPACE=저장, q/ESC=종료.")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    intr_dir = Path(args.intrinsics_dir)

    layout = mb.load_layout(args.layout)
    detector = mb.build_detector(layout.dictionary)
    known_ids = layout.known_ids()
    use_depth = not args.no_depth
    align_depth = not args.no_align_depth_to_color

    serial_to_idx = load_device_map(intr_dir)
    devs = RealSenseCamera.list_devices()
    if not devs:
        raise RuntimeError("RealSense 카메라를 찾을 수 없습니다.")
    if serial_to_idx is None:
        print("[WARN] device_map.json 없음 → serial 정렬 순서 사용.")
        idx_serial = [(i, s) for i, s in enumerate(sorted(devs.keys()))]
    else:
        idx_serial = []
        for s in devs.keys():
            if s in serial_to_idx:
                idx_serial.append((int(serial_to_idx[s]), s))
            else:
                print(f"[WARN] serial {s} not in device_map.json (skip)")
        idx_serial.sort(key=lambda x: x[0])
    if not idx_serial:
        raise RuntimeError("device_map 필터 후 사용 가능한 카메라 없음.")

    print("[INFO] camera mapping:")
    for ci, s in idx_serial:
        print(f"  cam{ci}: {s} ({devs.get(s, '?')})")

    cams: Dict[int, RealSenseCamera] = {}
    for ci, s in idx_serial:
        cam = RealSenseCamera(serial=s, width=args.width, height=args.height, fps=args.fps,
                              use_color=True, use_depth=use_depth,
                              align_depth_to_color=align_depth, warmup_frames=10,
                              frame_timeout_ms=2000, log_timeouts=True, log_errors=True)
        cam.start()
        cams[ci] = cam

    print(f"\n[INFO] warming up {args.warmup_seconds:.1f}s...")
    time.sleep(args.warmup_seconds)

    auto = args.auto_save or not args.show
    if args.show:
        print("\nControls: SPACE=save  q/ESC=quit")
        print("[INFO] 각 축(x/y/z)의 +,- 면 마커가 모두 커버돼야 Gt_Step2 에서 측정됩니다.\n")

    def print_coverage(seen):
        axis_ok = {"x": [False, False], "y": [False, False], "z": [False, False]}
        idxmap = {"+": 0, "-": 1}
        cells = []
        for mid in sorted(known_ids):
            a, s = layout.marker_axis_side[mid]
            ncams = len(seen.get(mid, []))
            tag = "tri" if ncams >= 2 else ("pnp" if ncams == 1 else "--")
            cells.append(f"id{mid}({a}{s}):{ncams}[{tag}]")
            if ncams >= 1:
                axis_ok[a][idxmap[s]] = True
        ax_str = " ".join(f"{a}:{'OK' if (o[0] and o[1]) else 'x'}" for a, o in axis_ok.items())
        print("  markers " + "  ".join(cells) + "   | axes " + ax_str)

    try:
        while True:
            imgs: Dict[int, dict] = {}
            all_valid = True
            for ci, cam in cams.items():
                color, depth, ts_ms = cam.get_latest()
                imgs[ci] = {"color": color, "depth": depth, "ts_ms": ts_ms}
                if color is None or (use_depth and depth is None):
                    all_valid = False

            colors = {ci: imgs[ci]["color"] for ci in imgs}
            seen, per_cam = _coverage(colors, detector, known_ids)

            if args.show:
                panels = []
                for ci in sorted(cams.keys()):
                    img = colors[ci]
                    if img is None:
                        img = np.zeros((args.height, args.width, 3), np.uint8)
                        cv2.putText(img, f"cam{ci} NO FRAME", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        img = _draw_overlay(img, per_cam.get(ci, []), known_ids)
                        msg = f"cam{ci} SPACE=save q=quit"
                        cv2.putText(img, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                        cv2.putText(img, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    panels.append(img)
                h0, w0 = panels[0].shape[:2]
                panels = [cv2.resize(pp, (w0, h0)) if pp.shape[:2] != (h0, w0) else pp for pp in panels]
                while len(panels) < 4:
                    panels.append(np.zeros((h0, w0, 3), np.uint8))
                cv2.imshow("gt_capture (marker overlay)",
                           np.vstack([np.hstack(panels[0:2]), np.hstack(panels[2:4])]))
                print_coverage(seen)

            key = cv2.waitKey(1) & 0xFF if args.show else 255
            if args.show and key in (27, ord("q")):
                print("[INFO] aborted by user.")
                return

            do_save = (key == 32) or (auto and all_valid)
            if not do_save:
                continue
            if not all_valid:
                print("[INFO] 아직 저장 불가: 프레임 없는 카메라 있음.")
                continue

            for ci in sorted(imgs.keys()):
                cv2.imwrite(str(out_dir / f"cam{ci}_rgb.png"), imgs[ci]["color"])
                if use_depth:
                    if args.depth_burst_n > 1:
                        burst = capture_depth_burst(cams[ci], args.depth_burst_n,
                                                    args.depth_burst_max_wait_ms)
                        fused = median_depth_ignore_zero(burst)
                        if fused is None:
                            fused = imgs[ci]["depth"]
                    else:
                        fused = imgs[ci]["depth"]
                    cv2.imwrite(str(out_dir / f"cam{ci}_depth.png"), fused)
                # 검출 오버레이 진단 저장
                cv2.imwrite(str(out_dir / f"detect_cam{ci}.png"),
                            _draw_overlay(imgs[ci]["color"], per_cam.get(ci, []), known_ids))
                print(f"[cam{ci}] saved rgb"
                      f"{' + depth' if use_depth else ''}  markers={[m for m,_ in per_cam.get(ci, [])]}")

            dump_calib_to_flat(intrinsics_dir=intr_dir,
                               transforms_json=Path(args.transforms_json),
                               out_dir=out_dir, cam_ids=[ci for ci, _ in idx_serial],
                               use_depth_K=args.use_depth_K)

            cov = {str(mid): {"cams": sorted(seen.get(mid, [])),
                              "axis": layout.marker_axis_side[mid][0],
                              "side": layout.marker_axis_side[mid][1],
                              "n_cams": len(seen.get(mid, []))}
                   for mid in sorted(known_ids)}
            (out_dir / "gt_marker_coverage.json").write_text(
                json.dumps({"layout": str(args.layout), "coverage": cov}, indent=2, ensure_ascii=False))

            # 커버되지 않은 축 경고
            missing = []
            for a in ("x", "y", "z"):
                plus = any(seen.get(m) for m, (ax_, s_) in layout.marker_axis_side.items()
                           if ax_ == a and s_ == "+")
                minus = any(seen.get(m) for m, (ax_, s_) in layout.marker_axis_side.items()
                            if ax_ == a and s_ == "-")
                if not (plus and minus):
                    missing.append(a)
            print(f"\n[INFO] saved to {out_dir.resolve()}")
            if missing:
                print(f"[WARN] 이 캡처에서 축 {missing} 는 +/- 면이 다 안 보임 → "
                      f"물체를 돌려 추가 캡처(다른 폴더) 후 Gt_Step2 에 함께 넘기세요.")
            print("Next: python Gt_Step2_measure_marker_box.py --capture_dir "
                  f"{out_dir} --layout {args.layout} [--estimate_json <obj_bbox_metric.json>]")
            return
    finally:
        for cam in cams.values():
            cam.stop()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
