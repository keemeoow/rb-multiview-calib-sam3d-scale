#!/usr/bin/env python3
"""
Obj_Step5_foundationpose_register.py

FoundationPose 으로 cam0 좌표계에서 6D 포즈 (T_C0_obj) 추정 후,
Eye-to-Hand 결과 T_R_C0 와 합성해 로봇 base 기준 포즈 (T_R_obj) 저장.

[전제]
  - foundationpose-arm64-ready docker 컨테이너 안에서 실행:
      -v ~/Desktop/FoundationPose:/workspace/FoundationPose
      -v ~/Desktop/my_multicam_repo:/workspace/my_repo
  - 컨테이너 ENV: TORCH_CUDA_ARCH_LIST=12.1, PYOPENGL_PLATFORM=egl
  - Mesh 는 Obj_Step3 의 *_scaled.glb 권장 (이미 origin-centered + meters).
  - Depth PNG 는 uint16, mm 단위 (RealSense aligned_depth_to_color 출력).
  - T_R_C0.json 는 Calib_Step6 출력 파일 (없으면 cam0 기준 포즈만 저장).

[실행 예시 — 컨테이너 안]
  python3 /workspace/my_repo/Obj_Step5_foundationpose_register.py \
    --scene_dir "/workspace/my_repo/data(1)/capture_obj_set1" \
    --mesh      "/workspace/my_repo/data(1)/outputs_set1/obj1/obj1_scaled.glb" \
    --mask      "/workspace/my_repo/data(1)/masks_set1/obj1/cam1_mask.png" \
    --t_r_c0    "/workspace/my_repo/data/handeye_session_01/T_R_C1.json" \
    --out_dir   "/workspace/my_repo/data(1)/outputs_set1/obj1/fp_pose" \
    --iter 5


출력:
  out_dir/T_C0_obj.json       4x4 (cam0 -> object)
  out_dir/T_R_obj.json        4x4 (base -> object)  ← T_R_C0 가 있을 때만
  out_dir/overlay.png         mesh re-projection 합성 이미지 (sanity check)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh


# ---------------------------------------------------------------- #
# I/O helpers
# ---------------------------------------------------------------- #

def load_K(path: Path) -> np.ndarray:
    K = np.loadtxt(path, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"K must be 3x3, got {K.shape} ({path})")
    return K


def load_rgb(path: Path) -> np.ndarray:
    """Returns HxWx3 uint8 RGB."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"RGB not readable: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_depth_m(path: Path, depth_scale: float = 1.0 / 1000.0) -> np.ndarray:
    """uint16 mm PNG -> float32 meters."""
    d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(f"Depth not readable: {path}")
    if d.dtype != np.uint16:
        print(f"[WARN] depth dtype={d.dtype}, expected uint16. Assuming raw values are mm.")
    depth_m = d.astype(np.float32) * float(depth_scale)
    depth_m[~np.isfinite(depth_m)] = 0.0
    return depth_m


def load_mask(path: Path, target_hw: tuple[int, int]) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Mask not readable: {path}")
    if m.shape != target_hw:
        m = cv2.resize(m, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
    return (m > 0).astype(bool)


def load_mesh(path: Path) -> trimesh.Trimesh:
    obj = trimesh.load(str(path), force="mesh")
    if isinstance(obj, trimesh.Scene):
        geoms = [g for g in obj.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise RuntimeError(f"No mesh geometry in {path}")
        m = trimesh.util.concatenate(tuple(geoms))
    else:
        m = obj
    if len(m.vertices) == 0:
        raise RuntimeError(f"Empty mesh: {path}")
    return m


# ---------------------------------------------------------------- #
# Sanity checks
# ---------------------------------------------------------------- #

def assert_mesh_canonical(mesh: trimesh.Trimesh, max_off_m: float = 0.01) -> dict:
    """FP 입력 가정: AABB 중심이 원점 근처 (<= max_off_m), extents 가 metric 범위."""
    v = np.asarray(mesh.vertices)
    aabb_min = v.min(axis=0)
    aabb_max = v.max(axis=0)
    center = 0.5 * (aabb_min + aabb_max)
    extents = aabb_max - aabb_min
    off = float(np.linalg.norm(center))
    info = {
        "aabb_center_m": center.tolist(),
        "extents_m": extents.tolist(),
        "center_offset_m": off,
    }
    if off > max_off_m:
        raise RuntimeError(
            f"Mesh AABB center {center} is {off:.4f}m from origin (>{max_off_m}m). "
            f"FoundationPose 가 객체 canonical frame 을 가정합니다 — Step3 _scaled.glb 사용 권장."
        )
    if extents.max() > 1.0 or extents.max() < 0.005:
        print(f"[WARN] mesh extents={extents} m — 단위가 미터인지 확인하세요 (FP 기본=meters).")
    return info


# ---------------------------------------------------------------- #
# FoundationPose adapter
# ---------------------------------------------------------------- #

def add_fp_to_sys_path(fp_root: Path) -> None:
    if not fp_root.exists():
        raise FileNotFoundError(f"FoundationPose root not found: {fp_root}")
    p = str(fp_root)
    if p not in sys.path:
        sys.path.insert(0, p)


def build_estimator(mesh: trimesh.Trimesh, debug_dir: Path):
    """FoundationPose 표준 API. estimater.py 가 fp_root 에 있다고 가정."""
    import torch  # noqa
    from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
    import nvdiffrast.torch as dr

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=np.asarray(mesh.vertices, dtype=np.float32),
        model_normals=np.asarray(mesh.vertex_normals, dtype=np.float32),
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=str(debug_dir),
        debug=1,
        glctx=glctx,
    )
    return est


def run_register(est, K: np.ndarray, rgb: np.ndarray, depth_m: np.ndarray,
                 mask: np.ndarray, n_iter: int) -> np.ndarray:
    pose = est.register(
        K=K.astype(np.float64),
        rgb=rgb,
        depth=depth_m,
        ob_mask=mask,
        iteration=int(n_iter),
    )
    pose = np.asarray(pose, dtype=np.float64).reshape(4, 4)
    return pose


# ---------------------------------------------------------------- #
# Post-FP Y-up (vertical axis) enforcement
# ---------------------------------------------------------------- #

def enforce_axis_up(T_cam_obj: np.ndarray, T_R_cam: np.ndarray | None) -> tuple[np.ndarray, dict]:
    """
    FP 가 추정한 pose 에서 mesh local 축 중 **가장 수직 (world Z 와 정렬도 최대)** 인 축이
    world +Z 방향을 향하도록 강제. mesh 축은 SAM3D 출력 + rank-매칭으로 임의 방향이라,
    대칭 객체에서 FP 가 위/아래 180° 모호한 pose 를 잡으면 사용자 시각으로는 '뒤집힌' 결과.

    동작:
      1. T_R_obj = T_R_cam @ T_cam_obj 계산
      2. mesh local axis k 의 world 방향 = R[:, k]; |R[2, k]| 최대인 k 가 수직 축
      3. R[2, k] < 0 (수직 축이 world -Z) 이면 → pose 180° 회전으로 뒤집어 +Z 로
         (회전축: 수직 축이 아닌 mesh 로컬 축 중 하나)
      4. 보정된 T_cam_obj 와 어느 축이 수직인지 (overlay 색깔용) 반환
    """
    info = {"applied": False, "vertical_axis_idx": None}
    if T_R_cam is None:
        info["reason"] = "no T_R_cam provided"
        return T_cam_obj, info

    T_R_obj = T_R_cam @ T_cam_obj
    R = T_R_obj[:3, :3]
    z_components = R[2, :]  # mesh +X, +Y, +Z 각각의 world Z 성분
    k_vert = int(np.argmax(np.abs(z_components)))
    z_dot = float(z_components[k_vert])
    axis_names = {0: "X", 1: "Y", 2: "Z"}
    info["vertical_axis_idx"] = k_vert
    info["vertical_axis_name"] = axis_names[k_vert]
    info["vertical_axis_world_z"] = z_dot

    if z_dot >= 0:
        info["reason"] = "vertical axis already points up"
        return T_cam_obj, info

    # 뒤집기: 수직 축이 아닌 어느 한 축을 중심으로 180° 회전 → 수직 축 부호 반전
    axis_to_rotate = (k_vert + 1) % 3
    R_flip = np.eye(3)
    for i in range(3):
        if i != axis_to_rotate:
            R_flip[i, i] = -1.0
    T_flip = np.eye(4)
    T_flip[:3, :3] = R_flip
    T_cam_obj_new = T_cam_obj @ T_flip

    R_new = (T_R_cam @ T_cam_obj_new)[:3, :3]
    info["applied"] = True
    info["rotation_axis_name"] = axis_names[axis_to_rotate]
    info["new_vertical_axis_world_z"] = float(R_new[2, k_vert])
    return T_cam_obj_new, info


# ---------------------------------------------------------------- #
# Overlay viz (FP 결과 sanity check)
# ---------------------------------------------------------------- #

def draw_axes(
    img: np.ndarray, K: np.ndarray, T_cam_obj: np.ndarray,
    axis_len_m: float = 0.03, vertical_axis_idx: int | None = None,
) -> np.ndarray:
    """object 좌표 원점 + XYZ 축을 cam 이미지에 투영.

    vertical_axis_idx 가 주어지면 그 축을 muted GREEN 으로 (= 사용자 컨벤션: green=up).
    채도 낮은 색깔 + 굵고 짧은 선.
    """
    pts3d = np.array([
        [0, 0, 0],
        [axis_len_m, 0, 0],
        [0, axis_len_m, 0],
        [0, 0, axis_len_m],
    ], dtype=np.float64)
    pts_h = np.hstack([pts3d, np.ones((4, 1))])
    pts_cam = (T_cam_obj @ pts_h.T).T[:, :3]
    z = pts_cam[:, 2]
    if (z <= 1e-6).any():
        return img
    uv = (K @ pts_cam.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    uv = uv.astype(int)
    out = img.copy()
    o = tuple(uv[0])

    # 진한 (saturated) 색 — RGB 입력
    BOLD_GREEN = (80, 200, 80)    # 수직 (up) - vivid green
    BOLD_BLUE  = (50, 110, 230)   # 다른 horizontal 1 - vivid blue
    BOLD_RED   = (230, 60, 60)    # 다른 horizontal 2 - vivid red

    if vertical_axis_idx is not None:
        colors = [BOLD_RED, BOLD_RED, BOLD_RED]
        colors[vertical_axis_idx] = BOLD_GREEN
        other = [i for i in range(3) if i != vertical_axis_idx]
        colors[other[0]] = BOLD_BLUE
        colors[other[1]] = BOLD_RED
    else:
        colors = [BOLD_BLUE, BOLD_GREEN, BOLD_RED]  # X / Y / Z

    THICK = 5
    for k in range(3):
        cv2.line(out, o, tuple(uv[k + 1]), colors[k], THICK, lineType=cv2.LINE_AA)
    cv2.circle(out, o, 5, (245, 245, 245), -1, lineType=cv2.LINE_AA)
    cv2.circle(out, o, 5, (40, 40, 40), 1, lineType=cv2.LINE_AA)
    return out


def draw_mesh_silhouette(img_rgb: np.ndarray, K: np.ndarray, T_cam_obj: np.ndarray,
                          mesh: trimesh.Trimesh) -> np.ndarray:
    """mesh silhouette 를 쨍한 민트초록 (vivid mint) 톤으로 알파블렌딩."""
    H, W = img_rgb.shape[:2]
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.faces, dtype=np.int32)
    Vh = np.hstack([V, np.ones((len(V), 1))])
    Vc = (T_cam_obj @ Vh.T).T[:, :3]
    z = Vc[:, 2]
    u = K[0, 0] * Vc[:, 0] / np.where(z > 1e-6, z, 1e-6) + K[0, 2]
    v = K[1, 1] * Vc[:, 1] / np.where(z > 1e-6, z, 1e-6) + K[1, 2]
    u[z <= 1e-6] = -1e6
    v[z <= 1e-6] = -1e6
    silh = np.zeros((H, W), dtype=np.uint8)
    pts2d = np.stack([u, v], axis=1).astype(np.int32)
    for tri in F:
        cv2.fillConvexPoly(silh, pts2d[tri], 255)
    overlay = img_rgb.copy()
    # 쨍한 민트초록 (RGB): vivid mint green
    TINT = np.array([90, 255, 170], dtype=np.float32)  # RGB
    ALPHA = 0.55
    if (silh > 0).any():
        region = overlay[silh > 0].astype(np.float32)
        overlay[silh > 0] = ((1.0 - ALPHA) * region + ALPHA * TINT).astype(np.uint8)
    return overlay


# ---------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------- #

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", type=Path, required=True,
                    help="cam{N}_rgb.png / cam{N}_depth.png / cam{N}_K.txt 가 있는 폴더")
    ap.add_argument("--cam", default="cam0",
                    help="사용할 카메라 prefix (default: cam0 — calib reference)")
    ap.add_argument("--mesh", type=Path, required=True,
                    help="obj{N}_scaled.glb (origin-centered, meters)")
    ap.add_argument("--mask", type=Path, required=True,
                    help="대상 객체의 cam0_mask.png (boolean/uint8)")
    ap.add_argument("--t_r_c0", type=Path, default=None,
                    help="Calib_Step6 출력 T_R_C0.json. 없으면 cam0 기준만 저장.")
    ap.add_argument("--out_dir", type=Path, required=True)

    ap.add_argument("--fp_root", type=Path,
                    default=Path("/workspace/FoundationPose"),
                    help="FoundationPose 소스 트리 (estimater.py 가 있는 곳)")
    ap.add_argument("--iter", type=int, default=5,
                    help="register iteration 횟수 (FP 권장 5)")
    ap.add_argument("--depth_scale", type=float, default=1.0 / 1000.0,
                    help="depth PNG 값 -> meter 변환 계수 (default: 1/1000 = mm).")
    ap.add_argument("--mesh_max_off_m", type=float, default=0.01,
                    help="mesh AABB 중심이 원점에서 떨어진 최대 허용 거리(m).")
    ap.add_argument("--no_yup", action="store_true",
                    help="post-FP Y-up (vertical axis) 강제 보정 비활성. "
                         "기본은 활성 — mesh local 축 중 수직 정렬도가 가장 큰 축이 "
                         "world +Z 를 향하도록 180° flip 적용, overlay 에서 그 축을 GREEN 으로.")
    return ap.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- inputs ---
    scene = args.scene_dir
    rgb_path   = scene / f"{args.cam}_rgb.png"
    depth_path = scene / f"{args.cam}_depth.png"
    K_path     = scene / f"{args.cam}_K.txt"
    for p in (rgb_path, depth_path, K_path):
        if not p.exists():
            raise FileNotFoundError(p)

    K = load_K(K_path)
    rgb = load_rgb(rgb_path)
    depth_m = load_depth_m(depth_path, depth_scale=args.depth_scale)
    mask = load_mask(args.mask, target_hw=depth_m.shape[:2])

    if int(mask.sum()) < 50:
        raise RuntimeError(f"Mask 가 너무 작음: {int(mask.sum())} px ({args.mask})")

    # --- mesh sanity ---
    mesh = load_mesh(args.mesh)
    canon_info = assert_mesh_canonical(mesh, max_off_m=args.mesh_max_off_m)
    print(f"[mesh] vertices={len(mesh.vertices)}  extents_m={canon_info['extents_m']}  "
          f"center_off={canon_info['center_offset_m']*1000:.3f}mm")

    # --- estimator ---
    add_fp_to_sys_path(args.fp_root)
    est = build_estimator(mesh, debug_dir=args.out_dir / "fp_debug")
    print(f"[FP] register iter={args.iter}, image={rgb.shape}, depth>0={int((depth_m>0).sum())} px, "
          f"mask={int(mask.sum())} px")

    T_C0_obj_raw = run_register(est, K, rgb, depth_m, mask, n_iter=args.iter)

    print("[FP] T_C0_obj (raw) =")
    print(T_C0_obj_raw)

    # --- T_R_C0 (handeye) 먼저 로드: Y-up 보정에 필요 ---
    T_R_C0 = None
    if args.t_r_c0 is not None:
        if not args.t_r_c0.exists():
            print(f"[WARN] T_R_C0 file missing: {args.t_r_c0} — base 합성 / Y-up 비활성.")
        else:
            with args.t_r_c0.open() as f:
                tcal = json.load(f)
            T_R_C0 = np.array(tcal["transformation_matrix_camera_to_robot"], dtype=np.float64).reshape(4, 4)

    # --- post-FP Y-up (vertical axis) 강제 보정 ---
    yup_info = {"applied": False, "vertical_axis_idx": None}
    if not args.no_yup:
        T_C0_obj, yup_info = enforce_axis_up(T_C0_obj_raw, T_R_C0)
        if yup_info["applied"]:
            print(f"[Y-up] mesh local +{yup_info['vertical_axis_name']} 가 world -Z 방향 "
                  f"(z={yup_info['vertical_axis_world_z']:.3f}) → "
                  f"+{yup_info['rotation_axis_name']} 축 중심 180° flip 적용. "
                  f"now z={yup_info['new_vertical_axis_world_z']:.3f}")
        else:
            print(f"[Y-up] no flip needed ({yup_info.get('reason', '?')})")
    else:
        T_C0_obj = T_C0_obj_raw

    out = {
        "input": {
            "rgb":  str(rgb_path),
            "depth": str(depth_path),
            "K":    str(K_path),
            "mask": str(args.mask),
            "mesh": str(args.mesh),
        },
        "cam_id": args.cam,
        "iterations": int(args.iter),
        "mesh_check": canon_info,
        "T_cam_obj_raw_4x4": T_C0_obj_raw.tolist(),
        "T_cam_obj_4x4": T_C0_obj.tolist(),
        "yup_correction": yup_info,
    }

    if not yup_info["applied"]:
        # 동일하면 raw key 제거해 출력 간소화
        out.pop("T_cam_obj_raw_4x4", None)

    print("[FP] T_C0_obj (final) =")
    print(T_C0_obj)

    # --- 합성: T_R_obj = T_R_C0 @ T_C0_obj ---
    if T_R_C0 is not None:
        T_R_obj = T_R_C0 @ T_C0_obj
        out["T_R_C0_4x4"] = T_R_C0.tolist()
        out["T_R_obj_4x4"] = T_R_obj.tolist()
        print("[FP] T_R_obj =")
        print(T_R_obj)
        (args.out_dir / "T_R_obj.json").write_text(
            json.dumps({"T_R_obj_4x4": T_R_obj.tolist(),
                        "translation_m": T_R_obj[:3, 3].tolist(),
                        "rotation_3x3": T_R_obj[:3, :3].tolist(),
                        "yup_correction": yup_info},
                       indent=2)
        )

    (args.out_dir / "T_C0_obj.json").write_text(json.dumps(out, indent=2))

    # --- overlay viz ---
    try:
        overlay = draw_mesh_silhouette(rgb, K, T_C0_obj, mesh)
        overlay = draw_axes(
            overlay, K, T_C0_obj, axis_len_m=0.03,
            vertical_axis_idx=yup_info.get("vertical_axis_idx"),
        )
        cv2.imwrite(str(args.out_dir / "overlay.png"),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"[FP] overlay saved: {args.out_dir / 'overlay.png'}")
    except Exception as e:
        print(f"[WARN] overlay draw failed: {e}")

    print(f"[FP] done. results -> {args.out_dir}")


if __name__ == "__main__":
    main()
