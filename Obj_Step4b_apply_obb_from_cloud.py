#!/usr/bin/env python3
"""
Obj_Step4b_apply_obb_from_cloud.py

[목적]
  SAM3D 메시의 **형상** + 점군 OBB 의 **크기** 조합.
  - Obj_Step3 출력 `<obj>_sam3d.glb` 는 형상은 그럴듯하지만 비율/scale 부정확
  - 멀티뷰 `<obj>_cloud_clean.ply` 는 실제 크기 정확 (metric)
  → 점군 OBB extents 로 SAM3D 메시에 **비균등 축별 scale** 강제 적용
  → 형상은 SAM3D 그대로, 크기는 점군 OBB 와 일치

  점군에서 OBB 를 직접 뽑아 자동화 — 실측 입력 불필요.

[실행]
  # 디렉토리 일괄 (outputs_set*/obj*/*_sam3d.glb + *_cloud_clean.ply 자동 매칭)
  python3 Obj_Step4b_apply_obb_from_cloud.py --in_root data

  # [정밀] cross-cam mask 검증 활성 (배경 leak 제거 → 더 작고 정확한 OBB)
  python3 Obj_Step4b_apply_obb_from_cloud.py --in_root data \
    --mask_validate --mask_dir data --min_cams 2 --mask_dilate_px 5

  # 단일 객체
  python3 Obj_Step4b_apply_obb_from_cloud.py \
    --sam3d_glb data/outputs_set1/obj1/obj1_sam3d.glb \
    --cloud     data/outputs_set1/obj1/obj1_cloud_clean.ply \
    --out       data/outputs_set1/obj1/obj1_sam3d_obb.glb

[출력]
  <obj>_sam3d_obb.glb       origin-centered, OBB extents 강제 적용된 GLB
                            (FoundationPose canonical 입력으로 바로 사용)
  <obj>_sam3d_obb.json      적용된 scale + 원본/목표 extents 기록

[FP 사용]
  python3 Obj_Step5_foundationpose_register.py \
    --mesh data/outputs_set1/obj1/obj1_sam3d_obb.glb \
    ... (나머지 동일)
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import trimesh


def _load_cam_masks(cap_dir: Path, msk_dir: Path, dilate_px: int = 0):
    info = {}
    for K_p in sorted(cap_dir.glob("cam*_K.txt")):
        cid = K_p.stem.replace("_K", "")
        T_p = cap_dir / f"{cid}_T_cam_to_world.txt"
        M_p = msk_dir / f"{cid}_mask.png"
        if not (T_p.exists() and M_p.exists()):
            continue
        K = np.loadtxt(K_p)
        T_C_R = np.linalg.inv(np.loadtxt(T_p))
        m = cv2.imread(str(M_p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        mb = m > 0
        if dilate_px > 0:
            k = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), np.uint8)
            mb = cv2.dilate(mb.astype(np.uint8), k, 1) > 0
        info[cid] = (K, T_C_R, mb)
    return info


def _validate_by_masks(pts: np.ndarray, cam_info: dict, min_cams: int):
    n = len(pts)
    counts = np.zeros(n, dtype=np.int32)
    for K, T_C_R, mask in cam_info.values():
        Vh = np.hstack([pts, np.ones((n, 1))])
        Vc = (T_C_R @ Vh.T).T[:, :3]
        z = Vc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = K[0, 0] * Vc[:, 0] / z + K[0, 2]
            v = K[1, 1] * Vc[:, 1] / z + K[1, 2]
        H, W = mask.shape
        in_front = z > 1e-6
        ui = np.clip(np.round(u), 0, W - 1).astype(np.int32)
        vi = np.clip(np.round(v), 0, H - 1).astype(np.int32)
        in_img = in_front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        hit = np.zeros(n, dtype=bool)
        hit[in_img] = mask[vi[in_img], ui[in_img]]
        counts += hit.astype(np.int32)
    keep = counts >= min_cams
    return keep, {int(k): int((counts == k).sum()) for k in range(len(cam_info) + 1)}


def _find_capture_mask_dirs(obj_dir: Path, data_root: Path | None):
    if data_root is None:
        return None, None
    m = re.search(r"outputs_set(\d+)", str(obj_dir))
    if not m:
        return None, None
    s = m.group(1)
    cap = data_root / f"capture_obj_set{s}"
    msk = data_root / f"masks_set{s}" / obj_dir.name
    if not (cap.exists() and msk.exists()):
        return None, None
    return cap, msk


def cloud_obb_extents(
    cloud_path: Path,
    min_points: int = 50,
    cam_info: dict | None = None,
    min_cams: int = 0,
) -> tuple[np.ndarray, int, dict]:
    """점군에서 OBB extents (m) 추출. desc 정렬. 옵션: mask 검증 적용."""
    pcd = o3d.io.read_point_cloud(str(cloud_path))
    pts = np.asarray(pcd.points)
    n_orig = len(pts)
    if n_orig < min_points:
        raise RuntimeError(f"point cloud too sparse: {n_orig} < {min_points} ({cloud_path})")

    val_info = {"applied": False}
    if cam_info and min_cams > 0:
        keep, hist = _validate_by_masks(pts, cam_info, min_cams)
        n_keep = int(keep.sum())
        val_info = {"applied": True, "min_cams": min_cams,
                    "n_original": n_orig, "n_kept": n_keep,
                    "hist_inside_count": hist}
        if n_keep >= min_points:
            pts = pts[keep]
            val_info["fallback"] = False
        else:
            val_info["fallback"] = True

    pcd_use = o3d.geometry.PointCloud()
    pcd_use.points = o3d.utility.Vector3dVector(pts)
    obb = pcd_use.get_oriented_bounding_box(robust=True)
    extents = np.asarray(obb.extent, dtype=np.float64)
    return np.sort(extents)[::-1], int(len(pts)), val_info


def apply_obb_to_mesh(mesh_path: Path, target_extents_desc: np.ndarray) -> tuple[trimesh.Scene, dict]:
    """SAM3D 메시에 OBB extents 강제 적용 (rank-매칭 비균등 scale)."""
    sc = trimesh.load(str(mesh_path), force="scene")
    geoms = list(sc.geometry.values()) if isinstance(sc, trimesh.Scene) else [sc]
    geoms = [g for g in geoms if hasattr(g, "vertices") and len(g.vertices)]
    if not geoms:
        raise RuntimeError(f"no usable mesh geometry: {mesh_path}")

    all_v = np.concatenate([g.vertices for g in geoms])
    cur_extents = all_v.max(0) - all_v.min(0)  # mesh-axis order
    order_cur = np.argsort(-cur_extents)        # largest -> smallest axis index
    target_sorted = np.sort(target_extents_desc)[::-1]
    if len(target_sorted) != 3:
        raise ValueError(f"target_extents must have 3 components, got {target_sorted}")

    scale_per_axis = np.zeros(3, dtype=np.float64)
    for rank, axis in enumerate(order_cur):
        scale_per_axis[axis] = target_sorted[rank] / cur_extents[axis]

    S = np.diag([scale_per_axis[0], scale_per_axis[1], scale_per_axis[2], 1.0])
    out_scene = trimesh.Scene()
    for name, g in (sc.geometry.items() if isinstance(sc, trimesh.Scene) else [("geom_0", sc)]):
        if not hasattr(g, "vertices"):
            continue
        g2 = g.copy()
        g2.apply_transform(S)
        out_scene.add_geometry(g2, geom_name=name)

    # origin-center (FP canonical)
    all_v2 = np.concatenate([g.vertices for g in out_scene.geometry.values()])
    centroid = 0.5 * (all_v2.max(0) + all_v2.min(0))
    T = np.eye(4)
    T[:3, 3] = -centroid
    for g in out_scene.geometry.values():
        g.apply_transform(T)

    final_v = np.concatenate([g.vertices for g in out_scene.geometry.values()])
    final_extents = final_v.max(0) - final_v.min(0)

    info = {
        "source_mesh": str(mesh_path),
        "mesh_extents_before_mm": (cur_extents * 1000.0).round(2).tolist(),
        "target_obb_extents_mm_desc": (target_sorted * 1000.0).round(2).tolist(),
        "scale_per_axis": scale_per_axis.tolist(),
        "mesh_extents_after_mm": (final_extents * 1000.0).round(2).tolist(),
        "centered_at_origin": True,
    }
    return out_scene, info


def iter_obj_dirs(in_root: Path):
    if (in_root / "outputs_set1").exists() or (in_root / "outputs_set2").exists():
        roots = sorted(in_root.glob("outputs_set*"))
    elif in_root.name.startswith("outputs_set"):
        roots = [in_root]
    else:
        roots = [in_root]
    out = []
    for r in roots:
        for od in sorted(r.glob("obj*")):
            if not od.is_dir():
                continue
            sam3d = od / f"{od.name}_sam3d.glb"
            ply = od / f"{od.name}_cloud_clean.ply"
            if sam3d.exists() and ply.exists():
                out.append((sam3d, ply, od))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", type=Path, default=None,
                    help="일괄 처리 루트. outputs_set*/obj*/ 자동 탐색.")
    ap.add_argument("--sam3d_glb", type=Path, default=None)
    ap.add_argument("--cloud", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--min_points", type=int, default=50)
    ap.add_argument("--mask_validate", action="store_true",
                    help="cross-cam SAM mask 교차 검증 활성 (정밀 모드).")
    ap.add_argument("--mask_dir", type=Path, default=None,
                    help="mask root (--mask_validate 시 필요)")
    ap.add_argument("--min_cams", type=int, default=2)
    ap.add_argument("--mask_dilate_px", type=int, default=5)
    args = ap.parse_args()

    if args.mask_validate and args.mask_dir is None:
        ap.error("--mask_validate 사용 시 --mask_dir 도 필요")

    # (sam3d_glb, cloud_ply, out_glb, info_json, cap_dir, msk_dir)
    targets = []
    if args.in_root is not None:
        for sam3d, ply, od in iter_obj_dirs(args.in_root):
            name = od.name
            cap, msk = (_find_capture_mask_dirs(od, args.mask_dir)
                        if args.mask_validate else (None, None))
            targets.append((sam3d, ply, od / f"{name}_sam3d_obb.glb",
                            od / f"{name}_sam3d_obb.json", cap, msk))
    elif args.sam3d_glb is not None and args.cloud is not None:
        out_glb = args.out if args.out is not None else \
                  args.sam3d_glb.with_name(args.sam3d_glb.stem.replace("_sam3d", "") + "_sam3d_obb.glb")
        info_json = out_glb.with_suffix(".json")
        cap, msk = (_find_capture_mask_dirs(args.sam3d_glb.parent, args.mask_dir)
                    if args.mask_validate else (None, None))
        targets.append((args.sam3d_glb, args.cloud, out_glb, info_json, cap, msk))
    else:
        ap.error("--in_root 또는 (--sam3d_glb --cloud) 필요")

    if not targets:
        print("[INFO] no matches found.")
        return

    print(f"{'obj':<28}{'before (mm)':<22}{'target OBB (mm)':<22}{'after (mm)':<22}{'val'}")
    print("-" * 110)
    for sam3d, ply, out_glb, info_json, cap, msk in targets:
        cam_info = None
        if args.mask_validate and cap is not None and msk is not None:
            cam_info = _load_cam_masks(cap, msk, dilate_px=args.mask_dilate_px)
        try:
            target_ext_desc, n_pts, val_info = cloud_obb_extents(
                ply, min_points=args.min_points,
                cam_info=cam_info, min_cams=args.min_cams if args.mask_validate else 0,
            )
            out_scene, info = apply_obb_to_mesh(sam3d, target_ext_desc)
            info["source_cloud"] = str(ply)
            info["cloud_points_used"] = n_pts
            info["mask_validation"] = val_info
        except Exception as e:
            print(f"[FAIL] {sam3d}: {e}")
            continue

        out_glb.parent.mkdir(parents=True, exist_ok=True)
        out_scene.export(str(out_glb))
        info_json.write_text(json.dumps(info, indent=2))

        b = info["mesh_extents_before_mm"]
        t = info["target_obb_extents_mm_desc"]
        a = info["mesh_extents_after_mm"]
        val_str = ""
        if val_info.get("applied"):
            val_str = "FB" if val_info.get("fallback") else "ok"
        label = f"{out_glb.parent.parent.name}/{out_glb.parent.name}"
        print(f"{label:<28}{str(b):<22}{str(t):<22}{str(a):<22}{val_str}")


if __name__ == "__main__":
    main()
