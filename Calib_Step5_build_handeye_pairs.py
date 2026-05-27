#!/usr/bin/env python3
"""
Calib_Step5_build_handeye_pairs.py

EE에 부착된 ArUco 큐브를 이용해 (cam0 좌표계 점 ↔ robot base 좌표계 점) 쌍을
calibrate_camera_to_robot.py 입력 형식으로 빌드한다.

전제:
  - Step2 캡처 meta.json 각 event마다 tcp_base_ee_4x4 임베드되어 있음
    (Calib_Step2_capture_multi_cam.py --tcp_poses_file 사용)
  - 큐브는 EE에 단단히 부착되어 있고, T_ee_cube (EE -> cube origin, 4x4)가 알려져 있음
    (CAD 또는 캘리퍼로 측정)

출력:
  pairs.json — list of {event_id, camera_point[3], robot_point[3]} (meters)
  그대로 calibrate_camera_to_robot.py 의 --input 으로 사용.

Usage:
  python Calib_Step5_build_handeye_pairs.py \
    --root_folder ./data/handeye_session_01 \
    --intrinsics_dir ./intrinsics \
    --ref_cam_idx 0 \
    --t_ee_cube_xyz 0,0,0.07 \
    --t_ee_cube_rpy_deg 0,0,0 \
    --use_cube_corners \
    --reproj_max_px 3.0 \
    --out pairs.json
"""

import os
import json
import argparse
from typing import List, Dict

import numpy as np
import cv2

from _aruco_cube import CubeConfig, ArucoCubeTarget, rodrigues_to_Rt


def load_intrinsics(intr_dir: str, cam_idx: int):
    p = os.path.join(intr_dir, f"cam{cam_idx}.npz")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing intrinsics npz: {p}")
    d = np.load(p)
    return d["color_K"].astype(np.float64), d["color_D"].astype(np.float64)


def rpy_deg_to_R(rpy_deg) -> np.ndarray:
    r, p, y = [np.deg2rad(float(v)) for v in rpy_deg]
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def cube_corners_local(side_m: float) -> np.ndarray:
    d = side_m / 2.0
    return np.array(
        [[x, y, z] for x in (-d, d) for y in (-d, d) for z in (-d, d)],
        dtype=np.float64,
    )  # (8, 3)


def parse_xyz(s: str) -> np.ndarray:
    parts = [float(v) for v in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"expected 3 comma-separated floats, got: {s!r}")
    return np.array(parts, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_folder", required=True,
                    help="Step2 캡처 root (meta.json + camN/ 포함)")
    ap.add_argument("--intrinsics_dir", required=True)
    ap.add_argument("--ref_cam_idx", type=int, default=0,
                    help="hand-eye base 정합 기준 카메라 (= 멀티캠 reference)")

    ap.add_argument("--t_ee_cube_xyz", default="0,0,0",
                    help="EE 좌표계에서 cube origin까지의 translation [m]. 'x,y,z'. "
                         "meta.json 에 tcp_base_cube_4x4 (robot_pose_server 가 보낸 cube center) 가 "
                         "있으면 이 값은 무시되고 socket 값이 우선 사용됨.")
    ap.add_argument("--t_ee_cube_rpy_deg", default="0,0,0",
                    help="EE -> cube rotation, roll,pitch,yaw [deg]. "
                         "큐브를 EE축에 평행 정렬했으면 0,0,0 (회전 모르면 origin 1점만 사용 권장). "
                         "tcp_base_cube_4x4 가 있으면 무시.")

    ap.add_argument("--use_cube_corners", action="store_true",
                    help="포즈당 큐브 8 corner를 점쌍으로 사용 "
                         "(default: cube origin 1점만 — 회전 오차에 강건)")

    ap.add_argument("--reproj_max_px", type=float, default=3.0,
                    help="cam0 PnP 평균 reproj 임계 (초과 프레임 제외)")
    ap.add_argument("--min_markers", type=int, default=2,
                    help="포즈당 최소 검출 마커 수 (IPPE flip 방지 위해 2 권장)")

    ap.add_argument("--out", default="pairs.json",
                    help="출력 파일명. 상대경로면 root_folder 안에 저장.")
    args = ap.parse_args()

    # --- T_ee_cube ---
    t_ec = parse_xyz(args.t_ee_cube_xyz)
    R_ec = rpy_deg_to_R(args.t_ee_cube_rpy_deg.split(","))
    T_ee_cube = np.eye(4, dtype=np.float64)
    T_ee_cube[:3, :3] = R_ec
    T_ee_cube[:3, 3] = t_ec
    print(f"[INFO] T_ee_cube: t={t_ec.tolist()} m, rpy_deg={args.t_ee_cube_rpy_deg}")

    # --- meta.json ---
    meta_path = os.path.join(args.root_folder, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.json not found: {meta_path}")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    # --- intrinsics ---
    K, D = load_intrinsics(args.intrinsics_dir, args.ref_cam_idx)

    # --- cube ---
    cfg = CubeConfig()
    cube = ArucoCubeTarget(cfg)
    print(f"[INFO] cube_side_m={cfg.cube_side_m}")

    corners_local = (cube_corners_local(cfg.cube_side_m)
                     if args.use_cube_corners
                     else np.array([[0.0, 0.0, 0.0]], dtype=np.float64))
    print(f"[INFO] points per pose: {len(corners_local)}  "
          f"(use_cube_corners={args.use_cube_corners})")

    # --- per-event PnP + pairing ---
    pairs: List[Dict] = []
    n_total = n_ok = n_skip_no_tcp = n_skip_no_img = n_skip_pnp = n_skip_reproj = 0
    used_event_ids: List[int] = []

    n_used_cube_direct = 0
    n_used_ee_offset = 0

    for cap in meta["captures"]:
        n_total += 1
        eid = cap.get("event_id")
        ref_rec = cap.get("cams", {}).get(str(args.ref_cam_idx))

        # 우선순위 1: robot_pose_server 가 보내준 cube center 4x4 (tool 4 = cube center).
        #   이 경우 별도 T_ee_cube 측정/입력 불필요 — 로봇 키네매틱스가 이미 cube 위치를 알고 있음.
        # 우선순위 2: tcp_base_ee_4x4 + 사용자 입력 T_ee_cube (수동 측정).
        cube_4x4 = cap.get("tcp_base_cube_4x4")
        ee_4x4 = cap.get("tcp_base_ee_4x4")

        if cube_4x4 is not None:
            T_base_cube = np.array(cube_4x4, dtype=np.float64).reshape(4, 4)
            n_used_cube_direct += 1
        elif ee_4x4 is not None:
            T_base_ee = np.array(ee_4x4, dtype=np.float64).reshape(4, 4)
            T_base_cube = T_base_ee @ T_ee_cube
            n_used_ee_offset += 1
        else:
            n_skip_no_tcp += 1
            continue

        if not ref_rec or not ref_rec.get("saved"):
            n_skip_no_img += 1
            continue

        rgb_path = os.path.join(args.root_folder, ref_rec["rgb_path"])
        img = cv2.imread(rgb_path)
        if img is None:
            print(f"[WARN] image read failed: {rgb_path}")
            n_skip_no_img += 1
            continue

        ok, rvec, tvec, used, reproj = cube.solve_pnp_cube(
            img, K, D,
            min_markers=args.min_markers,
            reproj_thr_mean_px=args.reproj_max_px,
            single_marker_only=False,
            return_reproj=True,
        )
        if not ok or reproj is None:
            n_skip_pnp += 1
            continue
        if reproj["err_mean"] > args.reproj_max_px:
            n_skip_reproj += 1
            print(f"  event {eid}: skip — reproj {reproj['err_mean']:.2f}px > {args.reproj_max_px}")
            continue

        T_C0_cube = rodrigues_to_Rt(rvec, tvec)

        for p_local in corners_local:
            ph = np.append(p_local, 1.0)
            p_cam = (T_C0_cube @ ph)[:3]
            p_rob = (T_base_cube @ ph)[:3]
            pairs.append({
                "event_id": int(eid),
                "camera_point": p_cam.tolist(),
                "robot_point": p_rob.tolist(),
            })
        n_ok += 1
        used_event_ids.append(int(eid))
        print(f"  event {eid}: PnP ok  err_mean={reproj['err_mean']:.2f}px  "
              f"markers={len(used)}  cube_z_cam={T_C0_cube[2, 3]:.3f}m")

    print(
        f"\n[STAT] captures={n_total}  used_poses={n_ok}  "
        f"skip(no_tcp={n_skip_no_tcp}, no_img={n_skip_no_img}, "
        f"pnp_fail={n_skip_pnp}, reproj={n_skip_reproj})"
    )
    print(f"[STAT] robot_point source — cube_direct(tcp_base_cube_4x4)={n_used_cube_direct}, "
          f"ee+offset(tcp_base_ee_4x4 @ T_ee_cube)={n_used_ee_offset}")
    print(f"[STAT] total point pairs: {len(pairs)}")
    if n_ok < 3:
        raise SystemExit(
            f"[ERROR] need >=3 valid poses for Kabsch, got {n_ok}. "
            "더 캡처하거나 큐브/조명/거리/회전다양성 확인."
        )

    # diversity warn
    if n_ok < 6:
        print(f"[WARN] used_poses={n_ok}: 정확도를 위해 8개 이상 권장.")

    out_path = (args.out if os.path.isabs(args.out)
                else os.path.join(args.root_folder, args.out))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"[SAVE] {out_path}")
    print(f"[NEXT] python calibrate_camera_to_robot.py --input {out_path} "
          f"--output {os.path.join(os.path.dirname(out_path) or '.', 'T_R_C0.json')}")


if __name__ == "__main__":
    main()
