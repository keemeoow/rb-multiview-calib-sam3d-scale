#!/usr/bin/env python3
"""
Calib_Step6b_solve_eye_to_hand.py

Hand-to-eye 캘리브: T_base_cam 직접 솔브 (eye-to-hand 구성).

기존 Step5+Step6 의 한계:
  Step5 가 server 의 `tcp_base_cube_4x4` (= T_base_ee @ T_ee_cube_assumed) 를 사용.
  T_ee_cube_assumed = [I | (0,0,163mm)] 가 실제 큐브 마운팅과 다르면
  cube 위치/회전이 한꺼번에 어긋남 → Kabsch 가 보정 불가 → 큰 잔차 (45 mm급).

이 스크립트:
  - T_base_ee (robot kinematics, 신뢰 가능) + T_cam_cube (cam1 PnP) 만 사용.
  - cv2.calibrateHandEye (eye-to-hand inversion trick) 로 T_base_cam 직접 솔브.
  - T_ee_cube 는 자유 파라미터로 자동 추정 (= 큐브 실제 마운팅 transform).
  - 결과: T_R_C{ref} 4x4 + 잔차 통계 + T_ee_cube_actual 보고.

[실제 사용 명령어]
python Calib_Step3ee_solve_eye_to_hand.py \\
    --root_folder ./data/handeye_session_02 \\
    --intrinsics_dir ./intrinsics \\
    --ref_cam_idx 1 \\
    --reproj_max_px 3.0 \\
    --method HORAUD \\
    --reject_outliers_mm 6.0

검증된 결과 (handeye_session_02, 70 poses, HORAUD):
  mean trans residual = 2.09 mm  (PASS, threshold 5.0 mm)
  rotation std        = 0.52°
  cam1 in base (mm)   = (263.3, 450.9, 138.6)
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _apriltag_cube import CubeConfig, AprilTagCubeTarget, rodrigues_to_Rt


METHODS = {
    "TSAI":       cv2.CALIB_HAND_EYE_TSAI,
    "PARK":       cv2.CALIB_HAND_EYE_PARK,
    "HORAUD":     cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF":    cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def load_intrinsics(intr_dir, cam_idx):
    p = os.path.join(intr_dir, f"cam{cam_idx}.npz")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    d = np.load(p)
    return d["color_K"].astype(np.float64), d["color_D"].astype(np.float64)


def collect_pairs(root, intrinsics_dir, ref_cam_idx, reproj_max_px, min_markers):
    with open(os.path.join(root, "meta.json")) as f:
        meta = json.load(f)
    K, D = load_intrinsics(intrinsics_dir, ref_cam_idx)
    cfg = CubeConfig()
    cube = AprilTagCubeTarget(cfg)

    T_cam_cube_list, T_base_ee_list, event_ids, reproj_list = [], [], [], []
    n_skip = 0
    for cap in meta["captures"]:
        ref_rec = cap.get("cams", {}).get(str(ref_cam_idx))
        if not ref_rec or not ref_rec.get("saved"):
            n_skip += 1; continue
        if "tcp_base_ee_4x4" not in cap:
            n_skip += 1; continue
        img = cv2.imread(os.path.join(root, ref_rec["rgb_path"]))
        if img is None:
            n_skip += 1; continue
        ok, rvec, tvec, used, reproj = cube.solve_pnp_cube(
            img, K, D, min_markers=min_markers,
            reproj_thr_mean_px=reproj_max_px,
            single_marker_only=False, return_reproj=True,
        )
        if not ok or reproj is None or reproj["err_mean"] > reproj_max_px:
            n_skip += 1; continue
        T_cam_cube_list.append(rodrigues_to_Rt(rvec, tvec))
        T_base_ee_list.append(np.array(cap["tcp_base_ee_4x4"]).reshape(4, 4))
        event_ids.append(int(cap["event_id"]))
        reproj_list.append(float(reproj["err_mean"]))
    return T_cam_cube_list, T_base_ee_list, event_ids, reproj_list, n_skip


def solve_eye_to_hand(T_cam_cube_list, T_base_ee_list, method):
    """OpenCV calibrateHandEye eye-to-hand: T_base_ee 를 inverse 해서 전달."""
    R_b2g = [np.linalg.inv(T)[:3, :3] for T in T_base_ee_list]
    t_b2g = [np.linalg.inv(T)[:3,  3] for T in T_base_ee_list]
    R_t2c = [T[:3, :3] for T in T_cam_cube_list]
    t_t2c = [T[:3,  3] for T in T_cam_cube_list]
    R_c2b, t_c2b = cv2.calibrateHandEye(R_b2g, t_b2g, R_t2c, t_t2c, method=method)
    T_base_cam = np.eye(4)
    T_base_cam[:3, :3] = R_c2b
    T_base_cam[:3,  3] = t_c2b.flatten()
    return T_base_cam


def consistency_stats(T_base_cam, T_cam_cube_list, T_base_ee_list):
    """T_ee_cube_i = T_base_ee_i^-1 @ T_base_cam @ T_cam_cube_i 의 분산 → 일관성."""
    T_ec_list = []
    for Tbe, Tcc in zip(T_base_ee_list, T_cam_cube_list):
        T_ec_list.append(np.linalg.inv(Tbe) @ T_base_cam @ Tcc)
    T_ec_arr = np.stack(T_ec_list)
    ts = T_ec_arr[:, :3, 3] * 1000.0  # mm
    rvecs = np.stack([cv2.Rodrigues(T[:3, :3])[0].flatten() for T in T_ec_arr])
    return T_ec_arr, ts, rvecs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_folder", required=True)
    ap.add_argument("--intrinsics_dir", required=True)
    ap.add_argument("--ref_cam_idx", type=int, default=1)
    ap.add_argument("--reproj_max_px", type=float, default=3.0)
    ap.add_argument("--min_markers", type=int, default=2)
    ap.add_argument("--output", default=None,
                    help="output json (default: <root>/T_R_C<ref>.json)")
    ap.add_argument("--method", default="HORAUD",
                    choices=list(METHODS.keys()),
                    help="hand-eye solver (default: HORAUD)")
    ap.add_argument("--reject_outliers_mm", type=float, default=0.0,
                    help="≥0 이면 1차 솔브 후 T_ee_cube 잔차 > 이 값 이상인 pose 제거 후 재솔브")
    args = ap.parse_args()

    T_cam_cube, T_base_ee, eids, reproj_errs, n_skip = collect_pairs(
        args.root_folder, args.intrinsics_dir, args.ref_cam_idx,
        args.reproj_max_px, args.min_markers,
    )
    n = len(T_cam_cube)
    print(f"[INFO] poses used: {n}  (skipped: {n_skip})")
    print(f"[INFO] PnP reproj: mean={np.mean(reproj_errs):.3f}px  max={np.max(reproj_errs):.3f}px")
    if n < 5:
        raise SystemExit(f"need >= 5 poses, got {n}")

    print(f"\n=== {args.method} eye-to-hand 솔브 ===")
    T_base_cam = solve_eye_to_hand(T_cam_cube, T_base_ee, METHODS[args.method])
    T_ec_arr, ts, rvecs = consistency_stats(T_base_cam, T_cam_cube, T_base_ee)
    res_norm_mm = np.linalg.norm(ts - ts.mean(axis=0), axis=1)
    print(f"  T_ee_cube translation std (xyz, mm): "
          f"{ts[:,0].std():.2f} / {ts[:,1].std():.2f} / {ts[:,2].std():.2f}  "
          f"|| total trans residual mean={res_norm_mm.mean():.2f}mm max={res_norm_mm.max():.2f}mm")
    rot_std_deg = np.degrees(np.linalg.norm(rvecs.std(axis=0)))
    print(f"  T_ee_cube rotation std: {rot_std_deg:.2f}°")

    # outlier rejection
    if args.reject_outliers_mm > 0:
        keep_mask = res_norm_mm <= args.reject_outliers_mm
        n_drop = int((~keep_mask).sum())
        if n_drop > 0 and keep_mask.sum() >= 5:
            print(f"\n[INFO] outlier reject: {n_drop} poses dropped "
                  f"(threshold {args.reject_outliers_mm}mm)")
            T_cam_cube = [T for T, m in zip(T_cam_cube, keep_mask) if m]
            T_base_ee  = [T for T, m in zip(T_base_ee,  keep_mask) if m]
            eids       = [e for e, m in zip(eids,       keep_mask) if m]
            T_base_cam = solve_eye_to_hand(T_cam_cube, T_base_ee, METHODS[args.method])
            T_ec_arr, ts, rvecs = consistency_stats(T_base_cam, T_cam_cube, T_base_ee)
            res_norm_mm = np.linalg.norm(ts - ts.mean(axis=0), axis=1)
            rot_std_deg = np.degrees(np.linalg.norm(rvecs.std(axis=0)))
            print(f"  재솔브: trans residual mean={res_norm_mm.mean():.2f}mm  "
                  f"max={res_norm_mm.max():.2f}mm  rot_std={rot_std_deg:.2f}°")

    print(f"\n=== T_base_cam ({args.method}) ===")
    print(T_base_cam)
    cam_pos_mm = T_base_cam[:3, 3] * 1000.0
    print(f"  cam{args.ref_cam_idx} position in base (mm): "
          f"({cam_pos_mm[0]:.1f}, {cam_pos_mm[1]:.1f}, {cam_pos_mm[2]:.1f})")

    # T_ee_cube_actual (평균)
    T_ee_cube_mean = T_ec_arr.mean(axis=0)
    print(f"\n=== T_ee_cube_actual (cube 실제 마운팅, EE 기준) ===")
    print(T_ee_cube_mean)
    print(f"  translation (mm): ({T_ee_cube_mean[0,3]*1000:.1f}, "
          f"{T_ee_cube_mean[1,3]*1000:.1f}, {T_ee_cube_mean[2,3]*1000:.1f})")

    # 결과 저장
    output = args.output or os.path.join(
        args.root_folder, f"T_R_C{args.ref_cam_idx}.json")
    result = {
        "input_root": os.path.abspath(args.root_folder),
        "ref_cam_idx": args.ref_cam_idx,
        "method": args.method,
        "num_poses_used": n,
        "transformation_matrix_camera_to_robot": T_base_cam.tolist(),
        "T_base_cam": T_base_cam.tolist(),
        "rotation_matrix": T_base_cam[:3, :3].tolist(),
        "translation_m": T_base_cam[:3, 3].tolist(),
        "T_ee_cube_actual_mean": T_ee_cube_mean.tolist(),
        "consistency": {
            "trans_residual_mean_mm": float(res_norm_mm.mean()),
            "trans_residual_max_mm":  float(res_norm_mm.max()),
            "rotation_std_deg":       float(rot_std_deg),
            "translation_std_xyz_mm": [float(ts[:,i].std()) for i in range(3)],
        },
        "pnp_reproj": {
            "mean_px": float(np.mean(reproj_errs)),
            "max_px":  float(np.max(reproj_errs)),
        },
        "event_ids_used": eids,
        "mean_error_threshold_m": 0.005,
        "calibration_success": bool(res_norm_mm.mean() <= 5.0),
    }
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVE] {output}")
    print(f"[RESULT] mean trans residual = {res_norm_mm.mean():.2f} mm  "
          f"(threshold 5.0 mm → {'PASS ✓' if res_norm_mm.mean() <= 5.0 else 'FAIL ✗'})")


if __name__ == "__main__":
    main()
