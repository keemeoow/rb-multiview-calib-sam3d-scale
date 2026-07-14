#!/usr/bin/env python3
"""
Multi-cam object point cloud + SAM3D mesh
========================================

이 스크립트는 **크기(metric scale)를 추정하지 않는다.**
멀티뷰 RGB-D + 마스크로 물체 점군을 만들고, 지정 카메라의 RGB+mask 로 SAM3D 메시를
생성할 뿐이다. 크기는 CAD 가 있으면 Obj_Step3c_fit_cad_silhouette.py 가 다중뷰 실루엣
정합으로 구한다 (자 실측 대비 평균 0.90mm; 점군 OBB 3.04mm, depth-ICP 6.07mm).

옛 크기 추정 경로(점군 OBB bbox_metric.json, SAM3D 메시 metric 스케일링 *_scaled.glb,
Obj_Step3b/Obj_Step4b 의 box.glb)는 제거되었다. 검은 무광 물체에서 depth 는 카메라 간
5~15mm 씩 어긋나 mm 단위 크기 추정에 쓸 수 없기 때문이다. 여기서 만드는 점군은
Obj_Step3c 의 **초기 포즈 추정용**으로는 충분하다.

목적
----
1. 멀티 RGB-D 카메라 + SAM/SAM2 mask + calibration 으로 물체 point cloud 생성
2. 지정 cam 의 원본 RGB + mask 로 SAM3D mesh 생성 (CAD 가 없는 물체용)

입력 폴더 예시
-------------
data_dir/
  cam0_rgb.png  cam0_depth.png  cam0_K.txt  cam0_T_cam_to_world.txt
  cam1_rgb.png  ...

mask_dir/
  cam0_mask.png ...                      # flat 단일 객체
  cam0_obj1_mask.png ...                 # flat 다중 객체
  obj1/cam0_mask.png ...                 # 객체별 서브폴더 (Obj_Step2 출력)

실행 예시
--------
source sam3d_env_gb10.sh

# 점군만 (CAD 가 있으면 이것으로 충분하다 -> Obj_Step3c)
python Obj_Step3_sam3d_pose.py \
  --data_dir ./data/capture_obj --mask_dir ./data/masks --out_dir ./data/outputs \
  --depth_scale 0.001 --mask_close_px 5 --mask_erode_px 3 --keep_largest_cc

# 점군 + SAM3D 메시 (CAD 가 없는 물체)
python Obj_Step3_sam3d_pose.py \
  --data_dir ./data/capture_obj --mask_dir ./data/masks --out_dir ./data/outputs \
  --depth_scale 0.001 --mask_erode_px 3 --keep_largest_cc \
  --run_sam3d --sam3d_cam cam1 --spconv_algo native

SAM3D 는 기본적으로 객체별 subprocess 에서 실행한다 (spconv/CUDA 오류 격리).

sam3d_mesh 가 디렉토리면 다음 패턴들로 자동 매칭:
  obj{id}_sam3d.glb, obj{id}.glb, obj{id}_mesh.glb, obj{id}_result.glb,
  object{n}_result.glb, object{n}.glb, mesh.glb (단일 객체 fallback)

출력
----
outputs/<obj_tag>/<obj_tag>_cloud_clean.ply — 멀티뷰 통합 point cloud
outputs/<obj_tag>/<obj_tag>_sam3d.glb       — SAM3D 원본 mesh (unitless)
outputs/objects_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
import trimesh
from sklearn.neighbors import LocalOutlierFactor


# ============================================================
# Data structure
# ============================================================

@dataclass
class CameraPacket:
    cam_id: str
    rgb: np.ndarray             # H x W x 3, uint8, RGB order
    depth: np.ndarray           # H x W, float32, meter
    K: np.ndarray               # 3 x 3
    T_cam_to_world: np.ndarray  # 4 x 4, camera frame -> world/base frame


@dataclass
class ViewCloud:
    cam_id: str
    points: np.ndarray
    colors: Optional[np.ndarray]
    raw_count: int
    clean_count: int


# ============================================================
# I/O utilities
# ============================================================

def load_matrix(path: str | Path, shape: Tuple[int, int]) -> np.ndarray:
    arr = np.loadtxt(path, dtype=np.float64)
    if arr.shape != shape:
        raise ValueError(f"Matrix shape mismatch: {path}, expected={shape}, got={arr.shape}")
    return arr


def load_rgb(path: str | Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"RGB image not found: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_depth(path: str | Path, depth_scale: float) -> np.ndarray:
    depth_raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(f"Depth image not found: {path}")
    depth = depth_raw.astype(np.float32) * depth_scale
    depth[~np.isfinite(depth)] = 0.0
    return depth


def load_cameras_from_folder(data_dir: str | Path, depth_scale: float = 0.001) -> Dict[str, CameraPacket]:
    data_dir = Path(data_dir)
    cameras: Dict[str, CameraPacket] = {}

    rgb_files = sorted(data_dir.glob("cam*_rgb.*"))
    if not rgb_files:
        raise FileNotFoundError(f"No cam*_rgb.* files found in {data_dir}")

    for rgb_path in rgb_files:
        cam_id = rgb_path.stem.replace("_rgb", "")
        depth_candidates = sorted(data_dir.glob(f"{cam_id}_depth.*"))
        if not depth_candidates:
            raise FileNotFoundError(f"Depth file missing for {cam_id}")

        K_path = data_dir / f"{cam_id}_K.txt"
        T_path = data_dir / f"{cam_id}_T_cam_to_world.txt"
        if not K_path.exists():
            raise FileNotFoundError(f"K file missing: {K_path}")
        if not T_path.exists():
            raise FileNotFoundError(f"T file missing: {T_path}")

        rgb = load_rgb(rgb_path)
        depth = load_depth(depth_candidates[0], depth_scale=depth_scale)
        K = load_matrix(K_path, (3, 3))
        T = load_matrix(T_path, (4, 4))

        if rgb.shape[:2] != depth.shape[:2]:
            raise ValueError(f"RGB/depth size mismatch in {cam_id}: rgb={rgb.shape}, depth={depth.shape}")

        cameras[cam_id] = CameraPacket(cam_id=cam_id, rgb=rgb, depth=depth, K=K, T_cam_to_world=T)

    return cameras


def load_mesh_any(path: str | Path) -> trimesh.Trimesh:
    mesh_or_scene = trimesh.load(str(path), force="scene")
    if isinstance(mesh_or_scene, trimesh.Scene):
        geoms = [g for g in mesh_or_scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise TypeError(f"No Trimesh geometry in scene: {path}")
        mesh = trimesh.util.concatenate(tuple(geoms))
    elif isinstance(mesh_or_scene, trimesh.Trimesh):
        mesh = mesh_or_scene
    else:
        raise TypeError(f"Unsupported mesh type: {type(mesh_or_scene)}")

    if mesh.vertices is None or len(mesh.vertices) == 0:
        raise RuntimeError(f"Mesh has no vertices: {path}")
    return mesh


# ============================================================
# Mask utilities
# ============================================================

def erode_mask(mask: np.ndarray, erode_px: int = 5, min_pixels_after: int = 200) -> np.ndarray:
    """SAM boundary에 섞인 배경 depth를 줄이기 위한 erosion."""
    mask = mask.astype(bool)
    if erode_px <= 0:
        return mask
    kernel = np.ones((erode_px, erode_px), np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    if eroded.sum() < min_pixels_after:
        print(f"[WARN] erosion made mask too small ({eroded.sum()} px). Use original mask instead.")
        return mask
    return eroded


def close_mask(mask: np.ndarray, close_px: int = 5) -> np.ndarray:
    """morphological close — SAM 결과 안에 생긴 작은 구멍을 메움."""
    if close_px <= 0:
        return mask.astype(bool)
    k = np.ones((close_px * 2 + 1, close_px * 2 + 1), np.uint8)
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k)
    return closed.astype(bool)


def keep_largest_connected_component(mask: np.ndarray) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n <= 1:
        return mask.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    return labels == largest


def load_masks(mask_dir: str | Path, cameras: Dict[str, CameraPacket]) -> Dict[str, np.ndarray]:
    mask_dir = Path(mask_dir)
    masks: Dict[str, np.ndarray] = {}
    for cam_id, cam in cameras.items():
        mask_path = mask_dir / f"{cam_id}_mask.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        if mask.shape != cam.depth.shape:
            raise ValueError(f"Mask/depth size mismatch in {cam_id}: mask={mask.shape}, depth={cam.depth.shape}")
        masks[cam_id] = mask > 0
    return masks


def load_masks_per_object(
    mask_dir: str | Path,
    cameras: Dict[str, CameraPacket],
    obj_ids: Optional[List[str]] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    세 가지 마스크 디렉토리 형식 지원 (자동 감지):
      1) 객체별 서브폴더:  mask_dir/<obj_name>/cam{N}_mask.png  (generate_masks_sam2.py 출력)
      2) flat 다중 객체:    mask_dir/cam{N}_obj{X}_mask.png
      3) flat 단일 객체:    mask_dir/cam{N}_mask.png  (obj_id="0"으로 묶임)

    객체 ID 명명:
      형식 1 → 폴더 이름 그대로 (예: "obj01")
      형식 2 → 정규식 캡처 그룹 (예: "1")
      형식 3 → "0"
    """
    mask_dir = Path(mask_dir)
    masks_by_obj: Dict[str, Dict[str, np.ndarray]] = {}

    # (1) 객체별 서브폴더 우선
    sub_dirs = sorted([d for d in mask_dir.iterdir() if d.is_dir()])
    for od in sub_dirs:
        obj_name = od.name
        if obj_ids is not None and obj_name not in obj_ids:
            continue
        sub_masks: Dict[str, np.ndarray] = {}
        for cam_id in cameras:
            mp = od / f"{cam_id}_mask.png"
            if not mp.exists():
                continue
            mk = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if mk is None:
                continue
            if mk.shape != cameras[cam_id].depth.shape:
                print(f"[WARN] mask/depth size mismatch in {obj_name}/{cam_id}: {mk.shape} vs {cameras[cam_id].depth.shape}")
                continue
            sub_masks[cam_id] = mk > 0
        if sub_masks:
            masks_by_obj[obj_name] = sub_masks

    if masks_by_obj:
        return masks_by_obj

    # (2) flat 다중 객체
    pat = re.compile(r"^(?P<cam>cam[^_]+)_obj(?P<obj>[^_]+)_mask\.[A-Za-z0-9]+$")
    multi_files = []
    for fp in mask_dir.iterdir():
        if not fp.is_file():
            continue
        m = pat.match(fp.name)
        if m:
            multi_files.append((m.group("cam"), m.group("obj"), fp))

    if multi_files:
        for cam_id, obj_id, fp in multi_files:
            if cam_id not in cameras:
                print(f"[WARN] mask file references unknown cam_id={cam_id}: {fp.name}")
                continue
            if obj_ids is not None and obj_id not in obj_ids:
                continue
            mk = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
            if mk is None:
                raise FileNotFoundError(f"Mask not readable: {fp}")
            if mk.shape != cameras[cam_id].depth.shape:
                raise ValueError(f"Mask/depth size mismatch in {cam_id}/obj{obj_id}: {mk.shape} vs {cameras[cam_id].depth.shape}")
            masks_by_obj.setdefault(obj_id, {})[cam_id] = mk > 0
    else:
        # (3) flat 단일 객체
        if obj_ids is not None and "0" not in obj_ids:
            raise FileNotFoundError("No cam*_obj*_mask.* files and fallback obj_id='0' was not requested.")
        masks_by_obj["0"] = load_masks(mask_dir, cameras)

    if not masks_by_obj:
        raise FileNotFoundError(f"No usable object masks found in {mask_dir}.")
    return masks_by_obj


def preprocess_masks(
    masks: Dict[str, np.ndarray],
    erode_px: int,
    largest_cc: bool,
    close_px: int = 0,
    min_pixels_after: int = 200,
) -> Dict[str, np.ndarray]:
    """largest_cc → close (구멍 메우기) → erode (경계 노이즈 제거) 순서."""
    out = {}
    for cam_id, mask in masks.items():
        m = keep_largest_connected_component(mask) if largest_cc else mask.astype(bool)
        if close_px > 0:
            m = close_mask(m, close_px=close_px)
        m = erode_mask(m, erode_px=erode_px, min_pixels_after=min_pixels_after)
        out[cam_id] = m
        print(f"[{cam_id}] mask pixels after preprocess: {int(m.sum())}")
    return out


# ============================================================
# Point cloud construction
# ============================================================

def depth_mask_to_world_points(
    depth: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    T_cam_to_world: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    min_depth: float = 0.05,
    max_depth: float = 2.0,
    stride: int = 1,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if mask.dtype != bool:
        mask = mask.astype(bool)

    if stride > 1:
        sampled = np.zeros_like(mask, dtype=bool)
        sampled[::stride, ::stride] = mask[::stride, ::stride]
        mask = sampled

    v, u = np.where(mask)
    z = depth[v, u]
    valid = np.isfinite(z) & (z > min_depth) & (z < max_depth)
    u, v, z = u[valid], v[valid], z[valid]
    if len(z) == 0:
        return np.empty((0, 3), dtype=np.float64), None

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x = (u.astype(np.float64) - cx) * z.astype(np.float64) / fx
    y = (v.astype(np.float64) - cy) * z.astype(np.float64) / fy

    pts_cam_h = np.stack([x, y, z.astype(np.float64), np.ones_like(z, dtype=np.float64)], axis=1)
    pts_world = (T_cam_to_world @ pts_cam_h.T).T[:, :3]

    colors = None
    if rgb is not None:
        colors = rgb[v, u].astype(np.float64) / 255.0
    return pts_world, colors


def filter_cloud_open3d(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    voxel_size: float = 0.002,
    nb_neighbors: int = 30,
    std_ratio: float = 2.0,
    radius: float = 0.01,
    min_points: int = 8,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if len(points) == 0:
        return points, colors
    points = np.ascontiguousarray(points, dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None and len(colors) == len(points):
        colors = np.ascontiguousarray(colors, dtype=np.float64)
        pcd.colors = o3d.utility.Vector3dVector(colors)

    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    if len(pcd.points) >= nb_neighbors:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    if len(pcd.points) >= min_points:
        pcd, _ = pcd.remove_radius_outlier(nb_points=min_points, radius=radius)

    clean_points = np.asarray(pcd.points)
    clean_colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    return clean_points, clean_colors


def filter_cloud_lof(points: np.ndarray, n_neighbors: int = 30, contamination: float = 0.03) -> np.ndarray:
    if len(points) < n_neighbors + 1 or contamination <= 0:
        return points
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    labels = lof.fit_predict(points)
    return points[labels == 1]


def keep_largest_cluster(
    points: np.ndarray,
    eps: float = 0.015,
    min_points: int = 20,
    min_cluster_ratio: float = 0.1,
    force: bool = False,
) -> np.ndarray:
    """DBSCAN으로 가장 큰 cluster만 keep.
    force=True면 largest_size/total < min_cluster_ratio여도 가장 큰 cluster를 강제 채택.
    """
    if len(points) < min_points:
        return points
    points = np.ascontiguousarray(points, dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    if labels.max() < 0:
        return points  # no cluster
    sizes = np.bincount(labels[labels >= 0])
    if len(sizes) == 0:
        return points
    largest_idx = int(np.argmax(sizes))
    largest_size = sizes[largest_idx]
    if not force and largest_size / len(points) < min_cluster_ratio:
        return points
    return points[labels == largest_idx]


def multi_stage_dbscan(
    points: np.ndarray,
    eps_list: List[float],
    min_points: int = 20,
) -> np.ndarray:
    """eps를 점차 좁혀가면서 largest cluster만 유지. stray chain 끊기에 효과적."""
    cur = points
    for i, eps in enumerate(eps_list):
        n_before = len(cur)
        # 마지막 단계로 갈수록 force=True (tighter pass는 cluster 작아져도 채택)
        force = (i > 0)
        cur = keep_largest_cluster(cur, eps=eps, min_points=min_points,
                                    min_cluster_ratio=0.05, force=force)
        print(f"    [DBSCAN stage {i+1}, eps={eps*1000:.1f}mm] {n_before} -> {len(cur)}")
        if len(cur) < min_points:
            break
    return cur


def build_view_clouds(
    cameras: Dict[str, CameraPacket],
    masks: Dict[str, np.ndarray],
    min_depth: float,
    max_depth: float,
    stride: int,
    voxel_size: float,
    radius: float,
    lof_contamination: float,
) -> List[ViewCloud]:
    view_clouds: List[ViewCloud] = []
    for cam_id, cam in cameras.items():
        if cam_id not in masks:
            continue
        raw_pts, raw_cols = depth_mask_to_world_points(
            depth=cam.depth,
            mask=masks[cam_id],
            K=cam.K,
            T_cam_to_world=cam.T_cam_to_world,
            rgb=cam.rgb,
            min_depth=min_depth,
            max_depth=max_depth,
            stride=stride,
        )
        clean_pts, clean_cols = filter_cloud_open3d(
            raw_pts,
            raw_cols,
            voxel_size=voxel_size,
            nb_neighbors=30,
            std_ratio=2.0,
            radius=radius,
            min_points=8,
        )
        clean_pts = filter_cloud_lof(clean_pts, n_neighbors=30, contamination=lof_contamination)
        view_clouds.append(ViewCloud(cam_id, clean_pts, clean_cols, len(raw_pts), len(clean_pts)))
        print(f"[{cam_id}] raw={len(raw_pts)}, clean={len(clean_pts)}")
    return view_clouds


def merge_view_clouds(view_clouds: List[ViewCloud]) -> np.ndarray:
    pts = [vc.points for vc in view_clouds if len(vc.points) > 0]
    if not pts:
        raise RuntimeError("No valid points from any camera after filtering.")
    return np.concatenate(pts, axis=0)


def save_cloud_ply(path: str | Path, points: np.ndarray, colors: Optional[np.ndarray] = None) -> None:
    points = np.ascontiguousarray(points, dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None and len(colors) == len(points):
        colors = np.ascontiguousarray(colors, dtype=np.float64)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(path), pcd)




# ============================================================
# SAM3D runner
# ============================================================


class Sam3DRunner:
    """Lazy in-process wrapper around third_party/sam-3d-objects notebook API."""

    def __init__(
        self,
        sam3d_root: str | Path = "third_party/sam-3d-objects",
        config_path: str | Path = "checkpoints/hf/pipeline.yaml",
        compile_model: bool = False,
        spconv_algo: str = "native",
    ) -> None:
        self.sam3d_root = Path(sam3d_root).resolve()
        self.config_path = Path(config_path)
        if not self.config_path.is_absolute():
            self.config_path = self.sam3d_root / self.config_path
        if not self.sam3d_root.exists():
            raise FileNotFoundError(f"SAM3D root not found: {self.sam3d_root}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"SAM3D config not found: {self.config_path}")

        if spconv_algo:
            os.environ["SPCONV_ALGO"] = spconv_algo
        print(f"[SAM3D] SPCONV_ALGO={os.environ.get('SPCONV_ALGO', '<unset>')}")

        for path in (self.sam3d_root, self.sam3d_root / "notebook"):
            path_s = str(path)
            if path_s not in sys.path:
                sys.path.insert(0, path_s)

        from inference import Inference, load_image  # type: ignore  # noqa: WPS433
        try:
            from inference import load_mask  # type: ignore  # noqa: WPS433
        except ImportError:
            from inference import load_single_mask as load_mask  # type: ignore  # noqa: WPS433

        print(f"[SAM3D] loading model: {self.config_path}")
        self.inference = Inference(str(self.config_path), compile=compile_model)
        self.load_image = load_image
        self.load_mask = load_mask

    @staticmethod
    def _tensor_to_jsonable(value: Any) -> Any:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    @staticmethod
    def _summarize_output(output: dict) -> dict:
        summary = {"output_keys": sorted(str(k) for k in output.keys())}
        for key, value in output.items():
            entry = {"type": str(type(value))}
            if hasattr(value, "vertices"):
                try:
                    entry["vertices"] = int(len(value.vertices))
                except Exception:
                    pass
            if hasattr(value, "faces"):
                try:
                    entry["faces"] = int(len(value.faces))
                except Exception:
                    pass
            summary[str(key)] = entry
        return summary

    @staticmethod
    def _export_native_glb(output: dict, out_glb_path: Path) -> str:
        preferred_keys = ["glb", "mesh", "meshes", "trimesh", "textured_mesh", "output_mesh"]
        for key in preferred_keys:
            obj = output.get(key)
            if obj is None or not hasattr(obj, "export"):
                continue
            try:
                obj.export(str(out_glb_path))
                if out_glb_path.exists() and out_glb_path.stat().st_size > 0:
                    return key
            except Exception as exc:
                print(f"[WARN] export failed for output['{key}']: {exc}")
        raise RuntimeError("SAM3D did not return an exportable native GLB mesh.")

    def _run_and_export(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        out_glb_path: str | Path,
        seed: int,
        image_label: str,
        mask_label: str,
        out_gs_ply_path: Optional[str | Path] = None,
        report_path: Optional[str | Path] = None,
    ) -> Path:
        out_glb_path = Path(out_glb_path)
        out_glb_path.parent.mkdir(parents=True, exist_ok=True)

        image = np.asarray(image, dtype=np.uint8)
        mask = np.asarray(mask)
        if mask.ndim == 3:
            mask = mask[..., -1]
        mask = mask > 0
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(f"SAM3D image/mask size mismatch: image={image.shape}, mask={mask.shape}")
        if int(mask.sum()) == 0:
            raise RuntimeError(f"SAM3D mask is empty: {mask_label}")

        print(f"[SAM3D] running inference: image={image_label}, mask={mask_label}")
        output = self.inference(image, mask, seed=seed)
        if not isinstance(output, dict):
            raise RuntimeError(f"SAM3D returned unsupported output type: {type(output)}")

        export_key = self._export_native_glb(output, out_glb_path)
        print(f"[SAM3D] saved mesh from output['{export_key}']: {out_glb_path}")

        if out_gs_ply_path is not None and output.get("gs") is not None:
            out_gs_ply_path = Path(out_gs_ply_path)
            out_gs_ply_path.parent.mkdir(parents=True, exist_ok=True)
            output["gs"].save_ply(str(out_gs_ply_path))
            print(f"[SAM3D] saved gaussian splat: {out_gs_ply_path}")

        if report_path is not None:
            report = {
                "image": image_label,
                "mask": mask_label,
                "sam3d_glb": str(out_glb_path),
                "sam3d_gs_ply": str(out_gs_ply_path) if out_gs_ply_path is not None else None,
                "seed": int(seed),
                "export_key": export_key,
                "output_summary": self._summarize_output(output),
            }
            for key in ("translation", "scale", "rotation"):
                if key in output:
                    report[key] = self._tensor_to_jsonable(output[key])
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(f"[SAM3D] saved report: {report_path}")

        return out_glb_path

    def run_arrays(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        out_glb_path: str | Path,
        seed: int = 42,
        image_label: str = "<array>",
        mask_label: str = "<array>",
        out_gs_ply_path: Optional[str | Path] = None,
        report_path: Optional[str | Path] = None,
    ) -> Path:
        return self._run_and_export(
            image=image,
            mask=mask,
            out_glb_path=out_glb_path,
            seed=seed,
            image_label=image_label,
            mask_label=mask_label,
            out_gs_ply_path=out_gs_ply_path,
            report_path=report_path,
        )

    def run_paths(
        self,
        image_path: str | Path,
        mask_path: str | Path,
        out_glb_path: str | Path,
        seed: int = 42,
        out_gs_ply_path: Optional[str | Path] = None,
        report_path: Optional[str | Path] = None,
    ) -> Path:
        image_path = Path(image_path)
        mask_path = Path(mask_path)
        image = self.load_image(str(image_path))
        mask = self.load_mask(str(mask_path))
        return self._run_and_export(
            image=image,
            mask=mask,
            out_glb_path=out_glb_path,
            seed=seed,
            image_label=str(image_path),
            mask_label=str(mask_path),
            out_gs_ply_path=out_gs_ply_path,
            report_path=report_path,
        )


def save_sam3d_image_mask_pair(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    image_path: str | Path,
    mask_path: str | Path,
) -> Tuple[Path, Path]:
    image_path = Path(image_path)
    mask_path = Path(mask_path)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)

    image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
    mask_u8 = (mask.astype(bool).astype(np.uint8)) * 255
    cv2.imwrite(str(image_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(mask_path), mask_u8)
    return image_path, mask_path


def run_sam3d_subprocess(
    image_path: str | Path,
    mask_path: str | Path,
    out_glb_path: str | Path,
    out_dir: str | Path,
    sam3d_root: str | Path,
    sam3d_config: str | Path,
    sam3d_seed: int,
    spconv_algo: str = "native",
    sam3d_compile: bool = False,
    sam3d_save_gs: bool = False,
) -> Path:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--image_path", str(image_path),
        "--sam3d_mask_path", str(mask_path),
        "--mesh_out", str(out_glb_path),
        "--out_dir", str(out_dir),
        "--sam3d_root", str(sam3d_root),
        "--sam3d_config", str(sam3d_config),
        "--sam3d_seed", str(sam3d_seed),
        "--spconv_algo", str(spconv_algo),
    ]
    if sam3d_compile:
        cmd.append("--sam3d_compile")
    if sam3d_save_gs:
        cmd.append("--sam3d_save_gs")
    print("[SAM3D] subprocess:", " ".join(cmd))
    env = os.environ.copy()
    if spconv_algo:
        env["SPCONV_ALGO"] = spconv_algo
    subprocess.run(cmd, check=True, env=env)
    out_glb_path = Path(out_glb_path)
    if not out_glb_path.exists() or out_glb_path.stat().st_size <= 0:
        raise RuntimeError(f"SAM3D subprocess did not produce mesh: {out_glb_path}")
    return out_glb_path


def resolve_sam3d_mesh_path(mesh_arg: Optional[Path], obj_tag: str, num_objects: int) -> Optional[Path]:
    """
    인자가 디렉토리면 obj_tag와 매칭되는 mesh 파일을 자동 탐색.
    지원 패턴 (확장자 .glb/.obj/.ply/.stl 순):
      {obj_tag}_sam3d.X, {obj_tag}.X, {obj_tag}_mesh.X, {obj_tag}_result.X,
      object{n}_sam3d.X, object{n}_result.X, object{n}.X    (obj_tag가 obj{NN} 형식이면 n=int(NN))
      mesh.X    (단일 객체 fallback)
    """
    if mesh_arg is None:
        return None
    if mesh_arg.is_dir():
        # obj_tag에서 숫자 추출 (예: "obj01" → 1, "1" → 1)
        m = re.search(r"(\d+)", obj_tag)
        num = int(m.group(1)) if m else None

        candidates = [f"{obj_tag}_sam3d", f"{obj_tag}", f"{obj_tag}_mesh", f"{obj_tag}_result"]
        if num is not None:
            candidates += [f"object{num}_sam3d", f"object{num}_result", f"object{num}", f"obj{num}_result"]
        candidates.append("mesh")

        for stem in candidates:
            for ext in (".glb", ".obj", ".ply", ".stl"):
                cand = mesh_arg / f"{stem}{ext}"
                if cand.exists():
                    return cand
        return None
    if num_objects == 1:
        if not mesh_arg.exists():
            raise FileNotFoundError(f"SAM3D mesh not found: {mesh_arg}")
        return mesh_arg
    print(f"[{obj_tag}] Multi-object mode requires --sam3d_mesh to be a directory.")
    return None


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="",
                        help="멀티뷰 metric scale 모드 입력 폴더. direct SAM3D 모드에서는 생략 가능.")
    parser.add_argument("--mask_dir", default="",
                        help="멀티뷰 metric scale 모드 mask 폴더. direct SAM3D 모드에서는 생략 가능.")
    parser.add_argument("--image_path", default="",
                        help="Direct SAM3D mode: RGB image path.")
    parser.add_argument("--sam3d_mask_path", default="",
                        help="Direct SAM3D mode: mask image path.")
    parser.add_argument("--mesh_out", default="",
                        help="Direct SAM3D mode output GLB path. Empty = out_dir/<mask_stem>_sam3d.glb.")
    parser.add_argument("--out_dir", default="outputs_multicam_sam3d")

    parser.add_argument("--depth_scale", type=float, default=0.001, help="RealSense uint16 mm -> meter: 0.001")
    parser.add_argument("--min_depth", type=float, default=0.05)
    parser.add_argument("--max_depth", type=float, default=2.0)
    parser.add_argument("--stride", type=int, default=1)

    parser.add_argument("--mask_close_px", type=int, default=5,
                        help="morphological close (SAM mask 안 구멍 메우기) 반경. 0=비활성")
    parser.add_argument("--mask_erode_px", type=int, default=5,
                        help="mask 경계 erode (배경 depth 침범 방지). 0=비활성")
    parser.add_argument("--keep_largest_cc", action="store_true",
                        help="erosion 전에 가장 큰 connected component만 유지")
    parser.add_argument("--voxel_size", type=float, default=0.002, help="meter. 0.002 = 2 mm")
    parser.add_argument("--lof_contamination", type=float, default=0.03)
    parser.add_argument("--dbscan_eps", type=float, default=0.015,
                        help="DBSCAN 이웃 거리 (m). 객체 크기에 비례, 작을수록 stricter")
    parser.add_argument("--dbscan_min_points", type=int, default=20,
                        help="DBSCAN cluster 최소 점 개수")
    parser.add_argument("--dbscan_min_cluster_ratio", type=float, default=0.1,
                        help="largest cluster가 전체의 이 비율 이상이어야 적용. 미만이면 원본 유지")
    parser.add_argument("--dbscan_eps_stages", default="0.015,0.005",
                        help="다단계 DBSCAN eps(m). 콤마 구분. 예: '0.02,0.008,0.004' "
                             "넓게 chain 뚫고 점점 좁혀가며 stray 제거. 비우면 단일 단계로 fallback")
    parser.add_argument("--cams_for_cloud", default="",
                        help="콤마 구분 cam id (예: cam0). 비우면 모든 cam 사용. "
                             "마스크 품질이 들쭉날쭉할 때 좋은 cam만 골라 cloud 생성")
    parser.add_argument("--obj_ids", default="", help="Comma-separated object ids. Empty = all detected. Single-object fallback id is 0.")

    parser.add_argument("--run_sam3d", action="store_true",
                        help="Run SAM3D on image+mask and save <obj_tag>_sam3d.glb.")
    parser.add_argument("--sam3d_root", default="third_party/sam-3d-objects")
    parser.add_argument("--sam3d_config", default="checkpoints/hf/pipeline.yaml")
    parser.add_argument("--sam3d_compile", action="store_true")
    parser.add_argument("--sam3d_seed", type=int, default=42)
    parser.add_argument("--spconv_algo", choices=["auto", "implicit_gemm", "native"], default="native",
                        help="spconv convolution algorithm for SAM3D. GB10 defaults to native to avoid implicit_gemm CUDA faults.")
    parser.add_argument("--sam3d_save_gs", action="store_true",
                        help="Also save SAM3D gaussian splat as <obj_tag>_sam3d_splat.ply.")
    parser.add_argument("--sam3d_mesh", default="",
                        help="Pre-generated SAM3D mesh file or directory containing obj{id}_sam3d.glb/.obj/.ply")
    parser.add_argument("--sam3d_cam", default="",
                        help="멀티뷰 모드에서 SAM3D에 넣을 cam id. 비우면 객체 mask가 있는 첫 cam을 사용.")
    parser.add_argument("--sam3d_in_process", action="store_true",
                        help="멀티뷰 모드에서도 SAM3D를 같은 Python process에서 실행. 기본은 CUDA fault 격리를 위해 subprocess.")

    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    direct_sam3d_mode = bool(args.image_path or args.sam3d_mask_path)
    if direct_sam3d_mode:
        if not args.image_path or not args.sam3d_mask_path:
            raise ValueError("Direct SAM3D mode requires both --image_path and --sam3d_mask_path.")
        image_path = Path(args.image_path)
        mask_path = Path(args.sam3d_mask_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        out_glb = Path(args.mesh_out) if args.mesh_out else out_dir / f"{mask_path.stem.replace('_mask', '')}_sam3d.glb"
        runner = Sam3DRunner(
            sam3d_root=args.sam3d_root,
            config_path=args.sam3d_config,
            compile_model=args.sam3d_compile,
            spconv_algo=args.spconv_algo,
        )
        runner.run_paths(
            image_path=image_path,
            mask_path=mask_path,
            out_glb_path=out_glb,
            seed=args.sam3d_seed,
            out_gs_ply_path=out_glb.with_name(f"{out_glb.stem}_splat.ply") if args.sam3d_save_gs else None,
            report_path=out_glb.with_name(f"{out_glb.stem}_report.json"),
        )
        print(f"\nSaved SAM3D native mesh: {out_glb}")
        return

    if not args.data_dir or not args.mask_dir:
        raise ValueError("Multiview metric mode requires --data_dir and --mask_dir. "
                         "For direct SAM3D GLB generation, use --image_path and --sam3d_mask_path.")

    cameras = load_cameras_from_folder(args.data_dir, depth_scale=args.depth_scale)
    print(f"Loaded cameras: {list(cameras.keys())}")

    selected_obj_ids = [s.strip() for s in args.obj_ids.split(",") if s.strip()] or None
    masks_by_obj_raw = load_masks_per_object(args.mask_dir, cameras, obj_ids=selected_obj_ids)
    print(f"Objects to process: {sorted(masks_by_obj_raw.keys())}")

    mesh_arg = Path(args.sam3d_mesh) if args.sam3d_mesh else None
    sam3d_runner: Optional[Sam3DRunner] = None
    results_summary: Dict[str, dict] = {}

    for obj_id in sorted(masks_by_obj_raw.keys()):
        # obj_id가 이미 "obj"로 시작하면 그대로, 아니면 obj{id} 형식
        obj_tag = obj_id if obj_id.startswith("obj") else f"obj{obj_id}"
        print(f"\n=== Processing {obj_tag} ===")

        raw_masks = masks_by_obj_raw[obj_id]
        masks = preprocess_masks(
            raw_masks,
            erode_px=args.mask_erode_px,
            largest_cc=args.keep_largest_cc,
            close_px=args.mask_close_px,
        )
        obj_cams = {cid: cameras[cid] for cid in masks.keys() if cid in cameras}

        radius = max(args.voxel_size * 4.0, 0.006)
        view_clouds = build_view_clouds(
            cameras=obj_cams,
            masks=masks,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            stride=args.stride,
            voxel_size=args.voxel_size,
            radius=radius,
            lof_contamination=args.lof_contamination,
        )
        # 사용자가 cam 일부만 선택했으면 그 view만 사용
        if args.cams_for_cloud.strip():
            allowed = {s.strip() for s in args.cams_for_cloud.split(",")}
            kept = [vc for vc in view_clouds if vc.cam_id in allowed]
            print(f"[{obj_tag}] cams_for_cloud filter: {[vc.cam_id for vc in view_clouds]} -> "
                  f"{[vc.cam_id for vc in kept]}")
            view_clouds = kept
            if not view_clouds:
                print(f"[{obj_tag}] Skip: no view cloud after cams_for_cloud filter.")
                continue
        cloud_pts = merge_view_clouds(view_clouds)
        n_before_dbscan = len(cloud_pts)
        eps_list = [float(s) for s in args.dbscan_eps_stages.split(",") if s.strip()]
        if eps_list:
            print(f"[{obj_tag}] multi-stage DBSCAN cleanup:")
            cloud_pts = multi_stage_dbscan(cloud_pts, eps_list=eps_list,
                                            min_points=args.dbscan_min_points)
        else:
            cloud_pts = keep_largest_cluster(
                cloud_pts,
                eps=args.dbscan_eps,
                min_points=args.dbscan_min_points,
                min_cluster_ratio=args.dbscan_min_cluster_ratio,
            )
        print(f"[{obj_tag}] DBSCAN total: {n_before_dbscan} -> {len(cloud_pts)} points")
        if len(cloud_pts) < 50:
            print(f"[{obj_tag}] Skip: too few clean points ({len(cloud_pts)}).")
            continue
        print(f"[{obj_tag}] merged clean cloud: {len(cloud_pts)} points")
        save_cloud_ply(out_dir / f"{obj_tag}_cloud_clean.ply", cloud_pts)
        # 객체별 폴더에도 복사
        obj_out_dir_early = out_dir / obj_tag
        obj_out_dir_early.mkdir(parents=True, exist_ok=True)
        save_cloud_ply(obj_out_dir_early / f"{obj_tag}_cloud_clean.ply", cloud_pts)

        # 객체별 출력 디렉토리
        obj_out_dir = out_dir / obj_tag
        obj_out_dir.mkdir(parents=True, exist_ok=True)

        mesh_path = resolve_sam3d_mesh_path(mesh_arg, obj_tag=obj_tag, num_objects=len(masks_by_obj_raw))
        sam3d_cam: Optional[str] = args.sam3d_cam.strip() or None
        if mesh_path is None and args.run_sam3d:
            if sam3d_cam:
                if sam3d_cam not in raw_masks or sam3d_cam not in obj_cams:
                    print(f"[{obj_tag}] Skip SAM3D: --sam3d_cam {sam3d_cam} has no RGB/mask for this object.")
                    continue
            else:
                available = sorted(cid for cid in raw_masks.keys() if cid in obj_cams)
                if not available:
                    print(f"[{obj_tag}] Skip SAM3D: no camera has both RGB and mask.")
                    continue
                sam3d_cam = available[0]
            print(f"[{obj_tag}] SAM3D input: original {sam3d_cam}_rgb + {sam3d_cam}_mask")

            sam3d_image_path, sam3d_mask_path = save_sam3d_image_mask_pair(
                image_rgb=obj_cams[sam3d_cam].rgb,
                mask=raw_masks[sam3d_cam],
                image_path=obj_out_dir / f"{obj_tag}_{sam3d_cam}_rgb.png",
                mask_path=obj_out_dir / f"{obj_tag}_{sam3d_cam}_mask.png",
            )
            out_sam3d_glb = obj_out_dir / f"{obj_tag}_sam3d.glb"
            if args.sam3d_in_process:
                if sam3d_runner is None:
                    sam3d_runner = Sam3DRunner(
                        sam3d_root=args.sam3d_root,
                        config_path=args.sam3d_config,
                        compile_model=args.sam3d_compile,
                        spconv_algo=args.spconv_algo,
                    )
                mesh_path = sam3d_runner.run_paths(
                    image_path=sam3d_image_path,
                    mask_path=sam3d_mask_path,
                    out_glb_path=out_sam3d_glb,
                    seed=args.sam3d_seed,
                    out_gs_ply_path=(obj_out_dir / f"{obj_tag}_sam3d_splat.ply") if args.sam3d_save_gs else None,
                    report_path=obj_out_dir / f"{obj_tag}_sam3d_report.json",
                )
            else:
                mesh_path = run_sam3d_subprocess(
                    image_path=sam3d_image_path,
                    mask_path=sam3d_mask_path,
                    out_glb_path=out_sam3d_glb,
                    out_dir=obj_out_dir,
                    sam3d_root=args.sam3d_root,
                    sam3d_config=args.sam3d_config,
                    sam3d_seed=args.sam3d_seed,
                    spconv_algo=args.spconv_algo,
                    sam3d_compile=args.sam3d_compile,
                    sam3d_save_gs=args.sam3d_save_gs,
                )

        if mesh_path is None:
            print(f"[{obj_tag}] No SAM3D mesh provided/generated (cloud only).")

        results_summary[obj_id] = {
            "obj_tag": obj_tag,
            "cloud_clean_ply": str(obj_out_dir / f"{obj_tag}_cloud_clean.ply"),
            "sam3d_mesh": None if mesh_path is None else str(mesh_path),
            "sam3d_cam": sam3d_cam,
            "merged_clean_points": int(len(cloud_pts)),
            "note": ("크기 추정은 하지 않는다. metric 크기는 CAD 가 있으면 "
                     "Obj_Step3c_fit_cad_silhouette.py 로 구한다."),
        }

    if results_summary:
        summary_path = out_dir / "objects_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results_summary, f, indent=2)
        print(f"\nSaved summary: {summary_path}")
    else:
        print("\nNo object produced a cloud. Check masks and depth.")


if __name__ == "__main__":
    main()
