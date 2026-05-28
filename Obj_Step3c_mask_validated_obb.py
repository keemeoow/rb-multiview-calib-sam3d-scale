#!/usr/bin/env python3
"""
Obj_Step3c_mask_validated_obb.py

[목적]
  Step3 의 cloud 는 cam별 mask 로 depth 를 필터링했지만, 캘리브 오차/depth 노이즈로
  경계 부근에 배경 leak 점들이 남는 경우가 많음. 이 leak 가 OBB 를 부풀려
  메시 크기가 실제보다 커짐.

  해결: 점군의 각 3D 점을 **모든 카메라 마스크**에 재투영해서 K개 이상 카메라의 마스크
  안에 떨어지는 점만 신뢰. 그 점군으로 OBB 재계산.

  → 결과: tighter, 실제 객체 외곽에 부합하는 OBB
  → 후속: _box.glb / _sam3d_obb.glb 도 자동 재생성

[전제]
  - Obj_Step3 출력: data/outputs_set*/obj*/<obj>_cloud_clean.ply (base 좌표계)
  - capture session: data/capture_obj_set*/cam{N}_T_cam_to_world.txt (T_R_C{N})
                     data/capture_obj_set*/cam{N}_K.txt
  - SAM 마스크:      data/masks_set*/obj*/cam{N}_mask.png
  - capture/mask 폴더는 outputs 폴더와 set 번호로 매칭 (set1 ↔ set1).

[실행]
  python3 Obj_Step3c_mask_validated_obb.py --data_root data \
    --min_cams 2 \
    --regen_box_glb --regen_sam3d_obb_glb

[옵션]
  --min_cams K       3D 점을 신뢰하려면 K개 이상 카메라 마스크 안에 있어야 함 (default 2).
                     0 이면 검증 비활성화 (= Step3b 와 동일).
  --regen_box_glb           _box.glb 도 새 OBB 로 재생성 (Step3b 갱신)
  --regen_sam3d_obb_glb     _sam3d_obb.glb 도 새 OBB 로 재생성 (Step4b 갱신)

[출력]
  <obj>_cloud_mask_validated.ply   filtered point cloud
  <obj>_box_obb.json               **OVERWRITTEN** (validated OBB 로 갱신)
  <obj>_box.glb                    (--regen_box_glb) 새 OBB extents 로 재생성
  <obj>_sam3d_obb.glb              (--regen_sam3d_obb_glb) 새 OBB extents 로 재생성
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


# ---------------------------------------------------------------- #
# Projection / mask validation
# ---------------------------------------------------------------- #

def validate_points_by_masks(
    pts_world: np.ndarray,
    cam_info: dict,  # {cam_id: (K, T_C_R, mask_bool)}
    min_cams: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (filtered_pts, inside_counts, stats)."""
    n_pts = len(pts_world)
    counts = np.zeros(n_pts, dtype=np.int32)
    if min_cams <= 0:
        return pts_world, counts, {"total": n_pts, "kept": n_pts, "thresh": min_cams}

    for cam_id, (K, T_C_R, mask) in cam_info.items():
        Vh = np.hstack([pts_world, np.ones((n_pts, 1))])
        Vc = (T_C_R @ Vh.T).T[:, :3]
        z = Vc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = K[0, 0] * Vc[:, 0] / z + K[0, 2]
            v = K[1, 1] * Vc[:, 1] / z + K[1, 2]
        H, W = mask.shape
        in_front = z > 1e-6
        ui = np.clip(np.round(u), 0, W - 1).astype(np.int32)
        vi = np.clip(np.round(v), 0, H - 1).astype(np.int32)
        in_image = in_front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        hit = np.zeros(n_pts, dtype=bool)
        hit[in_image] = mask[vi[in_image], ui[in_image]]
        counts += hit.astype(np.int32)

    keep = counts >= min_cams
    stats = {
        "total": int(n_pts),
        "kept": int(keep.sum()),
        "thresh": int(min_cams),
        "hist": {int(k): int((counts == k).sum()) for k in range(len(cam_info) + 1)},
    }
    return pts_world[keep], counts, stats


# ---------------------------------------------------------------- #
# OBB
# ---------------------------------------------------------------- #

def compute_obb(pts: np.ndarray) -> dict:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    obb = pcd.get_oriented_bounding_box(robust=True)
    extents = np.asarray(obb.extent, dtype=np.float64)
    return {
        "extents_m": extents.tolist(),
        "extents_mm_sorted_desc": sorted([e * 1000.0 for e in extents], reverse=True),
        "obb_center_world_m": np.asarray(obb.center, dtype=np.float64).tolist(),
        "obb_R_world_from_box": np.asarray(obb.R, dtype=np.float64).tolist(),
    }


# ---------------------------------------------------------------- #
# Mesh regeneration
# ---------------------------------------------------------------- #

def regen_box_glb(out_path: Path, extents_m: list[float]):
    box = trimesh.creation.box(extents=np.array(extents_m))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    box.export(str(out_path))


def regen_sam3d_obb_glb(sam3d_glb: Path, out_path: Path, target_extents_desc: np.ndarray) -> dict:
    sc = trimesh.load(str(sam3d_glb), force="scene")
    geoms = list(sc.geometry.values()) if isinstance(sc, trimesh.Scene) else [sc]
    geoms = [g for g in geoms if hasattr(g, "vertices") and len(g.vertices)]
    if not geoms:
        raise RuntimeError(f"no mesh geometry: {sam3d_glb}")
    all_v = np.concatenate([g.vertices for g in geoms])
    cur_extents = all_v.max(0) - all_v.min(0)
    order_cur = np.argsort(-cur_extents)
    target_sorted = np.sort(target_extents_desc)[::-1]
    scale_per_axis = np.zeros(3, dtype=np.float64)
    for rank, axis in enumerate(order_cur):
        scale_per_axis[axis] = target_sorted[rank] / cur_extents[axis]
    S = np.diag([scale_per_axis[0], scale_per_axis[1], scale_per_axis[2], 1.0])
    out_scene = trimesh.Scene()
    for name, g in (sc.geometry.items() if isinstance(sc, trimesh.Scene) else [("g", sc)]):
        if not hasattr(g, "vertices"):
            continue
        g2 = g.copy()
        g2.apply_transform(S)
        out_scene.add_geometry(g2, geom_name=name)
    # origin-center
    all_v2 = np.concatenate([g.vertices for g in out_scene.geometry.values()])
    centroid = 0.5 * (all_v2.max(0) + all_v2.min(0))
    T = np.eye(4); T[:3, 3] = -centroid
    for g in out_scene.geometry.values():
        g.apply_transform(T)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_scene.export(str(out_path))
    return {
        "scale_per_axis": scale_per_axis.tolist(),
        "mesh_extents_before_mm": (cur_extents * 1000.0).round(2).tolist(),
        "mesh_extents_after_mm": [
            float(target_sorted[list(order_cur).index(i)] * 1000.0) for i in range(3)
        ],
    }


# ---------------------------------------------------------------- #
# Set/obj discovery
# ---------------------------------------------------------------- #

def find_jobs(data_root: Path):
    """Return list of (s, o, paths). Match outputs_set{N}/obj{X} with capture_obj_set{N}/ and masks_set{N}/obj{X}/."""
    jobs = []
    for od_root in sorted(data_root.glob("outputs_set*")):
        m = re.search(r"outputs_set(\d+)$", od_root.name)
        if not m:
            continue
        s = int(m.group(1))
        cap_dir = data_root / f"capture_obj_set{s}"
        msk_dir = data_root / f"masks_set{s}"
        if not (cap_dir.exists() and msk_dir.exists()):
            print(f"[skip set{s}] missing capture/mask dirs ({cap_dir}, {msk_dir})")
            continue
        for obj_dir in sorted(od_root.glob("obj*")):
            ply = obj_dir / f"{obj_dir.name}_cloud_clean.ply"
            msk_obj_dir = msk_dir / obj_dir.name
            if not (ply.exists() and msk_obj_dir.exists()):
                continue
            jobs.append({
                "set": s, "obj_dir": obj_dir, "obj_name": obj_dir.name,
                "ply": ply, "cap_dir": cap_dir, "msk_dir": msk_obj_dir,
            })
    return jobs


def load_cam_info(cap_dir: Path, msk_dir: Path, dilate_px: int = 0) -> dict:
    """Return {cam_id: (K, T_C_R, mask_bool)} for all cams with mask present.
    dilate_px > 0 적용 시 마스크를 그만큼 팽창 (캘리브 tolerance)."""
    info = {}
    for cam_K in sorted(cap_dir.glob("cam*_K.txt")):
        cam_id = cam_K.stem.replace("_K", "")
        T_R_C = cap_dir / f"{cam_id}_T_cam_to_world.txt"
        mask_p = msk_dir / f"{cam_id}_mask.png"
        if not (T_R_C.exists() and mask_p.exists()):
            continue
        K = np.loadtxt(cam_K, dtype=np.float64)
        T_R_C_mat = np.loadtxt(T_R_C, dtype=np.float64)
        T_C_R_mat = np.linalg.inv(T_R_C_mat)
        m = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        m_bool = m > 0
        if dilate_px > 0:
            kernel = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), np.uint8)
            m_bool = cv2.dilate(m_bool.astype(np.uint8), kernel, iterations=1) > 0
        info[cam_id] = (K, T_C_R_mat, m_bool)
    return info


# ---------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path, default=Path("data"),
                    help="outputs_set*/, capture_obj_set*/, masks_set*/ 가 있는 루트")
    ap.add_argument("--min_cams", type=int, default=2,
                    help="3D 점이 mask 안에 들어가야 할 카메라 최소 수 (default 2). 0=비활성")
    ap.add_argument("--min_kept_points", type=int, default=30,
                    help="검증 후 남은 점이 이 미만이면 검증 전 점군 그대로 사용")
    ap.add_argument("--mask_dilate_px", type=int, default=3,
                    help="검증 시 mask 를 이만큼 dilate (캘리브 오차 tolerance). "
                         "작은 객체일수록 크게 (예: 5). 0=비활성")
    ap.add_argument("--regen_box_glb", action="store_true",
                    help="새 OBB extents 로 _box.glb 재생성 (Step3b 갱신)")
    ap.add_argument("--regen_sam3d_obb_glb", action="store_true",
                    help="새 OBB extents 로 _sam3d_obb.glb 재생성 (Step4b 갱신)")
    ap.add_argument("--save_filtered_ply", action="store_true",
                    help="검증된 점군을 _cloud_mask_validated.ply 로 저장")
    args = ap.parse_args()

    jobs = find_jobs(args.data_root)
    if not jobs:
        print("[INFO] no jobs found.")
        return

    print(f"{'set/obj':<14}{'#pts':>7}{'kept':>7}  hist (0/1/2/3 cams)   "
          f"{'old OBB (mm)':<24}{'new OBB (mm)':<24}{'shrink':>8}")
    print("-" * 120)

    for j in jobs:
        ply = j["ply"]
        pcd = o3d.io.read_point_cloud(str(ply))
        pts = np.asarray(pcd.points)
        if len(pts) < 50:
            print(f"[skip] {ply}: too few points ({len(pts)})")
            continue
        cam_info = load_cam_info(j["cap_dir"], j["msk_dir"], dilate_px=args.mask_dilate_px)
        if len(cam_info) < args.min_cams and args.min_cams > 0:
            print(f"[skip] {ply}: not enough cams with masks ({len(cam_info)})")
            continue

        # old OBB (for comparison)
        old_obb = compute_obb(pts)
        old_ext_desc = old_obb["extents_mm_sorted_desc"]

        # validate
        pts_v, counts, stats = validate_points_by_masks(pts, cam_info, args.min_cams)
        hist = stats.get("hist", {})

        # fallback if too few validated
        if len(pts_v) < args.min_kept_points:
            print(f"[WARN] {j['obj_name']}: only {len(pts_v)} pts after validation, "
                  f"keeping original cloud OBB")
            new_obb = old_obb
            new_ext_desc = old_ext_desc
        else:
            new_obb = compute_obb(pts_v)
            new_ext_desc = new_obb["extents_mm_sorted_desc"]

        shrink_pct = 100.0 * (1.0 - new_ext_desc[0] / max(old_ext_desc[0], 1e-9))

        # overwrite _box_obb.json with new OBB + provenance
        out_json = j["obj_dir"] / f"{j['obj_name']}_box_obb.json"
        new_obb["source_cloud"] = str(ply)
        new_obb["num_points_original"] = int(len(pts))
        new_obb["num_points_validated"] = int(len(pts_v))
        new_obb["mask_validation"] = {
            "min_cams": int(args.min_cams),
            "cams_used": sorted(cam_info.keys()),
            "hist_inside_count": hist,
            "fallback_used": len(pts_v) < args.min_kept_points,
            "old_obb_extents_mm_desc": old_ext_desc,
        }
        out_json.write_text(json.dumps(new_obb, indent=2))

        # filtered ply
        if args.save_filtered_ply and len(pts_v) >= args.min_kept_points:
            pcd_v = o3d.geometry.PointCloud()
            pcd_v.points = o3d.utility.Vector3dVector(pts_v)
            o3d.io.write_point_cloud(str(j["obj_dir"] / f"{j['obj_name']}_cloud_mask_validated.ply"), pcd_v)

        # regen meshes
        if args.regen_box_glb:
            regen_box_glb(j["obj_dir"] / f"{j['obj_name']}_box.glb", new_obb["extents_m"])
        if args.regen_sam3d_obb_glb:
            sam3d_glb = j["obj_dir"] / f"{j['obj_name']}_sam3d.glb"
            if sam3d_glb.exists():
                info = regen_sam3d_obb_glb(
                    sam3d_glb, j["obj_dir"] / f"{j['obj_name']}_sam3d_obb.glb",
                    np.array(new_obb["extents_m"]),
                )
                # update _sam3d_obb.json
                info_json = j["obj_dir"] / f"{j['obj_name']}_sam3d_obb.json"
                info["source_mesh"] = str(sam3d_glb)
                info["target_obb_extents_mm_desc"] = new_ext_desc
                info["centered_at_origin"] = True
                info["mask_validation"] = new_obb["mask_validation"]
                info_json.write_text(json.dumps(info, indent=2))

        hist_str = "/".join(str(hist.get(k, 0)) for k in range(len(cam_info) + 1))
        print(f"set{j['set']}/{j['obj_name']:<8}{len(pts):>7}{len(pts_v):>7}  "
              f"{hist_str:<22}"
              f"{str([round(v,1) for v in old_ext_desc]):<24}"
              f"{str([round(v,1) for v in new_ext_desc]):<24}"
              f"{shrink_pct:>6.1f}%")


if __name__ == "__main__":
    main()
