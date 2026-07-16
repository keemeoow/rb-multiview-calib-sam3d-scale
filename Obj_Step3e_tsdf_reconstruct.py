#!/usr/bin/env python3
"""
Multi-view depth fusion -> metric mesh (경로 3: 순수 재구성, 형상 prior 없음)
============================================================================

경로 1(CAD 실루엣 정합)·경로 2(SAM3D 메시 실루엣 정합)와 달리, 형상 prior 없이
캘리브레이션된 멀티뷰 depth 를 융합해 메시를 만든다. depth 가 이미 metric 이라
스케일 추정이 필요 없고 형상 hallucination 도 없다 — 관측 표면을 그대로 측정한다.

한계: (a) 뷰가 적으면 안 보이는 뒷면이 비어 non-watertight(열린 껍질)이고,
(b) 어둡/광택 표면은 depth 가 카메라 간 수 mm 어긋나 융합 메시도 흔들린다. 그리고
치수를 점군 OBB(바운딩)로 재면 노이즈가 상자를 바깥으로 밀어 과대측정 편향이 있다.

주의(aarch64): Open3D 0.18 의 ScalableTSDFVolume.integrate 가 segfault 하므로 native
TSDF 대신 numpy 역투영 point-cloud 융합을 쓴다.

입력 규약은 Obj_Step3c/Obj_Step3 와 동일:
  capture_dir/{cam}_rgb.png {cam}_depth.png {cam}_K.txt {cam}_T_cam_to_world.txt
  mask_dir/{obj}/{cam}_mask.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import trimesh

from _obb import min_volume_obb


def discover_cams(capture_dir: Path):
    return sorted({p.name.split("_")[0] for p in capture_dir.glob("cam*_rgb.png")})


def load_mask(mask_dir: Path, obj: str, cid: str, erode_px: int = 2) -> np.ndarray:
    p = mask_dir / obj / f"{cid}_mask.png"
    if not p.exists():
        raise SystemExit(f"마스크가 없습니다: {p}")
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 127
    if erode_px > 0:
        m = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=erode_px) > 0
    return m


def _backproject(K, T_cam_to_world, depth_m, mask, rgb, z_range):
    z = np.asarray(depth_m, np.float64)
    m = np.asarray(mask, bool) & np.isfinite(z) & (z > z_range[0]) & (z < z_range[1])
    if not m.any():
        return np.zeros((0, 3)), np.zeros((0, 3))
    v, u = np.where(m)
    zz = z[m]
    K = np.asarray(K, np.float64)
    x = (u - K[0, 2]) * zz / K[0, 0]
    y = (v - K[1, 2]) * zz / K[1, 1]
    Pc = np.stack([x, y, zz], axis=1)
    T = np.asarray(T_cam_to_world, np.float64)
    Pw = (T[:3, :3] @ Pc.T).T + T[:3, 3]
    col = rgb[v, u].astype(np.float64) / 255.0
    return Pw, col


def fuse_pointcloud(capture_dir: Path, mask_dir: Path, obj: str, cams,
                    depth_scale: float, voxel_m: float,
                    z_range=(0.05, 2.0), erode_px: int = 2):
    pts_parts, col_parts = [], []
    for cid in cams:
        d_p = capture_dir / f"{cid}_depth.png"
        rgb_p = capture_dir / f"{cid}_rgb.png"
        if not d_p.exists() or not rgb_p.exists():
            continue
        depth = cv2.imread(str(d_p), cv2.IMREAD_UNCHANGED)
        rgb = cv2.cvtColor(cv2.imread(str(rgb_p)), cv2.COLOR_BGR2RGB)
        if depth is None or rgb is None:
            continue
        mask = load_mask(mask_dir, obj, cid, erode_px=erode_px)
        K = np.loadtxt(capture_dir / f"{cid}_K.txt")
        T = np.loadtxt(capture_dir / f"{cid}_T_cam_to_world.txt")
        Pw, col = _backproject(K, T, depth.astype(np.float64) * depth_scale,
                               mask, rgb, z_range)
        pts_parts.append(Pw)
        col_parts.append(col)
    pts = np.vstack(pts_parts) if pts_parts else np.zeros((0, 3))
    cols = np.vstack(col_parts) if col_parts else np.zeros((0, 3))
    if len(pts) < 10:
        raise SystemExit(f"{obj}: 융합 점이 너무 적음 ({len(pts)})")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, dtype=np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(cols, dtype=np.float64))
    raw_n = len(pcd.points)
    if voxel_m > 0:
        pcd = pcd.voxel_down_sample(voxel_m)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    labels = np.asarray(pcd.cluster_dbscan(eps=5 * voxel_m, min_points=10))
    if labels.max() >= 0:
        keep = labels == int(np.argmax(np.bincount(labels[labels >= 0])))
        pcd = pcd.select_by_index(np.where(keep)[0])
    print(f"  fused cloud: {raw_n} -> {len(pcd.points)} points (voxel {voxel_m*1000:.1f}mm)")
    return pcd


def poisson_mesh(pcd, depth_level: int = 9):
    try:
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(20)
        mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth_level)
        dens = np.asarray(dens)
        if len(dens):
            mesh.remove_vertices_by_mask(dens < np.quantile(dens, 0.05))
        mesh.remove_unreferenced_vertices()
        return mesh
    except Exception as e:
        print(f"  [WARN] Poisson 실패({str(e)[:60]}); 점군 OBB 로만 치수 측정")
        return None


def extents_mm(points_xyz):
    V = np.ascontiguousarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    if len(V) < 4:
        raise SystemExit("점이 너무 적어 OBB 를 만들 수 없습니다")
    _, ext, _ = min_volume_obb(V)
    return np.sort(ext)[::-1] * 1000.0


def parse_gt(gt_args):
    gt = {}
    for a in gt_args or []:
        name, vals = a.split("=")
        gt[name] = sorted([float(x) for x in vals.split(",")], reverse=True)
    return gt


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture_dir", type=Path, required=True)
    ap.add_argument("--mask_dir", type=Path, required=True)
    ap.add_argument("--obj", action="append", required=True, help="객체 이름(마스크 서브폴더). 반복 가능.")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--depth_scale", type=float, default=0.001)
    ap.add_argument("--voxel_mm", type=float, default=1.0)
    ap.add_argument("--erode_px", type=int, default=2)
    ap.add_argument("--gt", action="append", metavar="peg=45,30,30", help="실측 치수(mm)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cams = discover_cams(args.capture_dir)
    print(f"[INFO] cameras: {cams}")
    voxel_m = args.voxel_mm / 1000.0
    gt = parse_gt(args.gt)

    summary = {}
    for obj in args.obj:
        print(f"\n=== {obj} (multi-view depth fusion) ===")
        pcd = fuse_pointcloud(args.capture_dir, args.mask_dir, obj, cams,
                              args.depth_scale, voxel_m, erode_px=args.erode_px)
        pts = np.asarray(pcd.points)
        ext = extents_mm(pts)
        out_glb = args.out_dir / f"{obj}_tsdf.glb"
        mesh = poisson_mesh(pcd)
        if mesh is not None and len(mesh.triangles) > 0:
            trimesh.Trimesh(vertices=np.asarray(mesh.vertices),
                            faces=np.asarray(mesh.triangles), process=False).export(out_glb)
            n_tri = int(len(mesh.triangles))
            print(f"  Poisson mesh: {len(mesh.vertices)} verts, {n_tri} tris -> {out_glb}")
        else:
            out_glb = args.out_dir / f"{obj}_tsdf_cloud.ply"
            o3d.io.write_point_cloud(str(out_glb), pcd)
            n_tri = 0
        print(f"  depth-fusion extents : {ext[0]:6.1f} x {ext[1]:6.1f} x {ext[2]:6.1f} mm")
        rec = {"obj": obj, "method": "multiview_depth_fusion",
               "extents_mm_sorted_desc": [float(x) for x in ext],
               "voxel_mm": args.voxel_mm, "mesh_glb": str(out_glb),
               "n_points": int(len(pts)), "n_triangles": n_tri}
        if obj in gt:
            g = np.array(gt[obj])
            err = ext - g
            rec.update({"gt_mm": list(g), "errors_mm": [float(x) for x in err],
                        "mean_abs_error_mm": float(np.mean(np.abs(err))),
                        "max_abs_error_mm": float(np.max(np.abs(err)))})
            print(f"  GT {g[0]:.0f} x {g[1]:.0f} x {g[2]:.0f} mm  "
                  f"meanAbsErr {rec['mean_abs_error_mm']:.2f} mm  maxErr {rec['max_abs_error_mm']:.2f} mm")
        summary[obj] = rec

    with open(args.out_dir / "tsdf_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {args.out_dir / 'tsdf_summary.json'}")


if __name__ == "__main__":
    main()
