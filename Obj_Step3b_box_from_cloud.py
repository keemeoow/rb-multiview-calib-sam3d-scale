#!/usr/bin/env python3
"""
Obj_Step3b_box_from_cloud.py

멀티뷰 point cloud 의 oriented bounding box (OBB) 로부터 origin-centered
직육면체 mesh (.glb) 를 생성. SAM3D 메시가 형상을 잘못 만드는 단순 박스
객체용 대체 메시.

[전제]
  - Obj_Step3 의 출력 `<obj>_cloud_clean.ply` 가 base 좌표계 점군이라고 가정.
    (base 좌표계든 cam 좌표계든 OBB 자체는 동일한 직육면체로 추출됨)

[실행]
  # 단일 객체
  python3 Obj_Step3b_box_from_cloud.py \
    --cloud data/outputs_set1/obj1/obj1_cloud_clean.ply \
    --out   data/outputs_set1/obj1/obj1_box.glb

  # 디렉토리 일괄 (data/outputs_set*/obj*/  자동 처리)
  python3 Obj_Step3b_box_from_cloud.py --in_root data

  # 특정 set 만
  python3 Obj_Step3b_box_from_cloud.py --in_root data/outputs_set1

  # [정밀] cross-cam mask 검증 활성 (점군에 배경 leak 있을 때)
  python3 Obj_Step3b_box_from_cloud.py --in_root data \
    --mask_validate --mask_dir data --min_cams 2 --mask_dilate_px 5

[출력]
  <obj>_box.glb            origin-centered 직육면체 (FoundationPose canonical input)
  <obj>_box_obb.json       OBB extents / 위치 / 방향 (진단 + Obj_Step5 참고용)

[FP 사용]
  python3 Obj_Step5_foundationpose_register.py \
    --mesh data/outputs_set1/obj1/obj1_box.glb \
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

from _obb import obb as compute_obb


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
    hist = {int(k): int((counts == k).sum()) for k in range(len(cam_info) + 1)}
    return keep, hist


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


def make_box_from_cloud(
    cloud_path: Path,
    out_glb_path: Path,
    obb_json_path: Path | None = None,
    min_points: int = 50,
    cam_info: dict | None = None,
    min_cams: int = 0,
    obb_method: str = "min_volume",
) -> dict:
    pcd = o3d.io.read_point_cloud(str(cloud_path))
    pts = np.asarray(pcd.points)
    n_orig = len(pts)
    if n_orig < min_points:
        raise RuntimeError(f"point cloud too sparse: {n_orig} < {min_points} ({cloud_path})")

    val_info = {"applied": False}
    if cam_info and min_cams > 0:
        keep, hist = _validate_by_masks(pts, cam_info, min_cams)
        n_keep = int(keep.sum())
        val_info = {"applied": True, "min_cams": min_cams, "n_original": n_orig,
                    "n_kept": n_keep, "hist_inside_count": hist}
        if n_keep >= min_points:
            pts = pts[keep]
            val_info["fallback"] = False
        else:
            val_info["fallback"] = True

    # 최소부피 OBB. open3d 의 PCA 축 OBB 는 등방적인 물체를 크게 부풀린다 (_obb.py 참고).
    center, extents, R = compute_obb(pts, method=obb_method)   # R: box(local) -> world

    # trimesh box: 원점 중심, axis-aligned. (FoundationPose canonical 입력 OK)
    box = trimesh.creation.box(extents=extents)
    out_glb_path.parent.mkdir(parents=True, exist_ok=True)
    box.export(str(out_glb_path))

    info = {
        "source_cloud": str(cloud_path),
        "num_points": int(n_orig),
        "num_points_used": int(len(pts)),
        "mask_validation": val_info,
        "extents_m": extents.tolist(),
        "extents_mm_sorted_desc": sorted([float(e) * 1000.0 for e in extents], reverse=True),
        "obb_center_world_m": center.tolist(),
        "obb_R_world_from_box": R.tolist(),
        "obb_method": str(obb_method),
        "note": (
            "Box mesh is origin-centered, AXIS-ALIGNED in its own canonical frame. "
            "FoundationPose 가 scene 에서 회전/위치를 추정. "
            "obb_R_world_from_box / obb_center_world_m 는 진단 참고용."
        ),
    }
    if obb_json_path is not None:
        obb_json_path.write_text(json.dumps(info, indent=2))
    return info


def iter_obj_dirs(in_root: Path):
    """Look for outputs_set*/obj*/<obj>_cloud_clean.ply patterns."""
    found = []
    if (in_root / "outputs_set1").exists() or (in_root / "outputs_set2").exists():
        roots = sorted(in_root.glob("outputs_set*"))
    elif in_root.name.startswith("outputs_set"):
        roots = [in_root]
    else:
        roots = [in_root]
    for r in roots:
        for od in sorted(r.glob("obj*")):
            if not od.is_dir():
                continue
            ply = od / f"{od.name}_cloud_clean.ply"
            if ply.exists():
                found.append(od)
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud", type=Path, default=None,
                    help="단일 ply 경로. 미지정 시 --in_root 사용.")
    ap.add_argument("--out", type=Path, default=None,
                    help="단일 모드의 출력 .glb (미지정 시 ply 옆에 <obj>_box.glb)")
    ap.add_argument("--in_root", type=Path, default=None,
                    help="디렉토리 일괄 처리. outputs_set*/obj*/*_cloud_clean.ply 찾음.")
    ap.add_argument("--min_points", type=int, default=50)
    ap.add_argument("--mask_validate", action="store_true",
                    help="cross-cam SAM mask 교차 검증 활성 (정밀 모드). "
                         "≥ --min_cams 카메라 mask 안에 들어가는 점만 사용 → 배경 leak 제거.")
    ap.add_argument("--mask_dir", type=Path, default=None,
                    help="mask root (예: data). --mask_validate 시 필요.")
    ap.add_argument("--min_cams", type=int, default=2,
                    help="--mask_validate 시 점이 들어가야 할 최소 카메라 수 (default 2).")
    ap.add_argument("--mask_dilate_px", type=int, default=5,
                    help="--mask_validate 시 mask dilation (캘리브 tolerance, default 5).")
    ap.add_argument("--obb_method", choices=["min_volume", "pca"], default="min_volume",
                    help="OBB 축 결정 방식. min_volume=최소부피 상자(기본). "
                         "pca=open3d 구 동작(등방적 물체를 부풀림, 옛 결과 재현용).")
    args = ap.parse_args()

    if args.cloud is None and args.in_root is None:
        ap.error("--cloud 또는 --in_root 중 하나는 필요")
    if args.mask_validate and args.mask_dir is None:
        ap.error("--mask_validate 사용 시 --mask_dir 도 필요")

    # (ply, glb, info_json, cap_dir, msk_dir)
    targets = []
    if args.cloud is not None:
        ply = args.cloud
        stem = ply.stem.replace("_cloud_clean", "")
        glb = args.out if args.out is not None else ply.with_name(f"{stem}_box.glb")
        info_json = ply.with_name(f"{stem}_box_obb.json")
        cap, msk = (_find_capture_mask_dirs(ply.parent, args.mask_dir)
                    if args.mask_validate else (None, None))
        targets.append((ply, glb, info_json, cap, msk))
    else:
        for od in iter_obj_dirs(args.in_root):
            stem = od.name
            ply = od / f"{stem}_cloud_clean.ply"
            glb = od / f"{stem}_box.glb"
            info_json = od / f"{stem}_box_obb.json"
            cap, msk = (_find_capture_mask_dirs(od, args.mask_dir)
                        if args.mask_validate else (None, None))
            targets.append((ply, glb, info_json, cap, msk))

    if not targets:
        print("[INFO] no matching point clouds found.")
        return

    print(f"{'src':<55}{'extents (mm sorted)':<25}{'#pts orig/used':<18}{'val'}")
    print("-" * 110)
    for ply, glb, info_json, cap, msk in targets:
        cam_info = None
        if args.mask_validate and cap is not None and msk is not None:
            cam_info = _load_cam_masks(cap, msk, dilate_px=args.mask_dilate_px)
        try:
            info = make_box_from_cloud(
                ply, glb, info_json,
                min_points=args.min_points,
                cam_info=cam_info, min_cams=args.min_cams if args.mask_validate else 0,
                obb_method=args.obb_method,
            )
        except Exception as e:
            print(f"[FAIL] {ply}: {e}")
            continue
        ext_str = " x ".join(f"{v:.1f}" for v in info["extents_mm_sorted_desc"])
        pts_str = f"{info['num_points']}/{info['num_points_used']}"
        v = info.get("mask_validation", {})
        val_str = ""
        if v.get("applied"):
            val_str = "FB" if v.get("fallback") else "ok"
        try:
            rel = ply.relative_to(ply.parents[2])
        except Exception:
            rel = ply
        print(f"{str(rel):<55}{ext_str:<25}{pts_str:<18}{val_str}")


if __name__ == "__main__":
    main()
