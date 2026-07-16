#!/usr/bin/env python3
"""
Obj_Step3c_cad_scale.py

CAD (또는 형상만 아는 메시) 가 있을 때의 **최선의 크기 추정** (경로 1).
다중뷰 SAM 마스크의 실루엣에 CAD 를 맞춰 스케일 + 6-DoF 포즈를 구한다.
depth 는 기본적으로 초기값에만 쓴다 (근거는 _silhouette_fit.py 상단 참고).
다양한 색/재질 물체에는 --w_depth auto 를 권장한다: 카메라 간 depth 일치도로 depth 를
신뢰할 만할 때(흰/무광)만 손실에 섞고, 편향될 때(검은/광택)는 자동으로 0 이 된다.

원본 CAD 가 없으면(경로 2) Obj_Step3_sam3d_scale.py --estimate_size 가 SAM3D 로 메시를
만들어 **같은 엔진**(_silhouette_fit.fit_cad_to_views)으로 크기를 구한다. 차이는 SAM3D
형상이 추정치라 per-view IoU 가 곧 치수 신뢰도라는 점. CAD 가 있으면 이 경로 1 이 더 정확하다.

  점군 최소부피 OBB   3.37 mm   (제거됨)
  메시 depth-ICP      0.79 mm
  메시 실루엣 정합    0.28 mm   <- 이 스크립트
  (참값 아는 합성 실험, 실제 카메라 배치, depth 편향 3mm 주입, 평균 |치수 오차|)

[실행]
  python Obj_Step3c_cad_scale.py \
    --capture_dir data/capture_obj \
    --mask_dir    data/masks \
    --cad peg=data/meshes/peg.glb --cad hole=data/meshes/hole.glb \
    --out_dir     data/outputs_cad_fit \
    --w_depth auto \
    --save_overlay --mask_uncertainty

[출력]  <out_dir>/
  <obj>_cad_fit.json      스케일, T_world_cad, 치수, per-view IoU, depth 잔차
  <obj>_cad_scaled.glb    실척으로 스케일된 CAD, 원점 중심 (FoundationPose 입력)
  <obj>_sil_overlay.jpg   마스크 외곽선 vs 정합된 CAD 실루엣 (--save_overlay)

[FoundationPose]
  --mesh 에 <obj>_cad_scaled.glb 를 넘긴다 (실척 CAD, 원점 중심).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh

from _silhouette_fit import (
    View, auto_w_depth, clean_cloud, cloud_from_masked_depth, fit_cad_to_views,
    obb_frame, render_silhouette,
)


def parse_cad(pairs):
    out = {}
    for p in pairs:
        if "=" not in p:
            raise SystemExit(f"--cad 형식은 obj1=path/to/model.glb 입니다 (받은 값: {p!r})")
        k, v = p.split("=", 1)
        path = Path(v)
        if not path.exists():
            raise SystemExit(f"CAD 파일이 없습니다: {path}")
        out[k] = path
    return out


def discover_cams(capture_dir: Path):
    cams = []
    for K_p in sorted(capture_dir.glob("cam*_K.txt")):
        cid = K_p.stem.replace("_K", "")
        T_p = capture_dir / f"{cid}_T_cam_to_world.txt"
        if T_p.exists():
            cams.append(cid)
    if not cams:
        raise SystemExit(f"카메라를 찾지 못했습니다: {capture_dir}/cam*_K.txt")
    return cams


def load_mask(mask_dir: Path, obj: str, cid: str, morph: int = 0) -> np.ndarray:
    p = mask_dir / obj / f"{cid}_mask.png"
    if not p.exists():
        raise SystemExit(f"마스크가 없습니다: {p}")
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 127
    if morph:
        k = np.ones((3, 3), np.uint8)
        m = (cv2.dilate(m.astype(np.uint8), k, 1) if morph > 0
             else cv2.erode(m.astype(np.uint8), k, 1)) > 0
    return m


def build_views(capture_dir: Path, mask_dir: Path, obj: str, cams, morph: int = 0):
    views, cam_meta = [], []
    for cid in cams:
        K = np.loadtxt(capture_dir / f"{cid}_K.txt")
        T = np.loadtxt(capture_dir / f"{cid}_T_cam_to_world.txt")
        m = load_mask(mask_dir, obj, cid, morph)
        if not m.any():
            print(f"  [WARN] {obj} {cid}: 마스크가 비어 있어 제외")
            continue
        views.append(View(K, T, m))
        cam_meta.append((cid, K, T))
    if len(views) < 2:
        raise SystemExit(f"{obj}: 실루엣 정합에는 최소 2개 시점이 필요합니다 (가용 {len(views)})")
    return views, cam_meta


def build_cloud(capture_dir: Path, mask_dir: Path, obj: str, cams, depth_scale: float,
                erode_px, voxel_m) -> np.ndarray:
    parts = []
    for cid in cams:
        d_p = capture_dir / f"{cid}_depth.png"
        if not d_p.exists():
            continue
        depth = cv2.imread(str(d_p), cv2.IMREAD_UNCHANGED)
        if depth is None:
            continue
        K = np.loadtxt(capture_dir / f"{cid}_K.txt")
        T = np.loadtxt(capture_dir / f"{cid}_T_cam_to_world.txt")
        m = load_mask(mask_dir, obj, cid)
        parts.append(cloud_from_masked_depth(K, T, depth, m, depth_scale, erode_px))
    if not parts:
        raise SystemExit(f"{obj}: depth 를 찾지 못해 초기값을 만들 수 없습니다")
    return clean_cloud(np.vstack(parts), voxel_m=voxel_m)


def object_depth_cams(capture_dir: Path, mask_dir: Path, obj: str, cams,
                      depth_scale: float):
    """--w_depth auto 의 카메라 간 일치도 평가용. 물체별 (K, T, depth[m], mask) 목록."""
    out = []
    for cid in cams:
        d_p = capture_dir / f"{cid}_depth.png"
        if not d_p.exists():
            continue
        depth = cv2.imread(str(d_p), cv2.IMREAD_UNCHANGED)
        if depth is None:
            continue
        out.append({
            "cid": cid,
            "K": np.loadtxt(capture_dir / f"{cid}_K.txt"),
            "T": np.loadtxt(capture_dir / f"{cid}_T_cam_to_world.txt"),
            "depth_m": depth.astype(np.float64) * float(depth_scale),
            "mask": load_mask(mask_dir, obj, cid),
        })
    return out


def save_scaled_cad(mesh: trimesh.Trimesh, scale: float, out_path: Path) -> np.ndarray:
    """실척 CAD 를 원점 중심으로 내보낸다 (FoundationPose canonical 입력)."""
    m = mesh.copy()
    m.apply_scale(float(scale))
    m.apply_translation(-m.bounding_box.centroid)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.export(str(out_path))
    return np.asarray(m.vertices, dtype=np.float64)


def overlay(capture_dir: Path, cam_meta, views, mesh, fit, obj: str, out_path: Path):
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces)
    s, R, t = fit["scale"], fit["R_cad_to_world"], fit["t_cad_to_world"]
    tiles = []
    for (cid, _, _), v, iou in zip(cam_meta, views, fit["per_view_iou"]):
        rgb = cv2.imread(str(capture_dir / f"{cid}_rgb.png"))
        if rgb is None:
            continue
        img = rgb.copy()
        cs, _ = cv2.findContours(v.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cs, -1, (0, 255, 0), 2)                 # SAM mask
        sil = render_silhouette(V, F, s, R, t, v, ss=1)
        cs2, _ = cv2.findContours(sil.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cs2, -1, (0, 0, 255), 2)                # fitted CAD
        ys, xs = np.where(v.mask)
        x0, x1 = max(int(xs.min()) - 40, 0), min(int(xs.max()) + 40, img.shape[1])
        y0, y1 = max(int(ys.min()) - 40, 0), min(int(ys.max()) + 40, img.shape[0])
        crop = cv2.resize(img[y0:y1, x0:x1], None, fx=2.5, fy=2.5, interpolation=cv2.INTER_NEAREST)
        cv2.putText(crop, f"{obj} {cid} IoU={iou:.3f}", (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        tiles.append(crop)
    if not tiles:
        return
    h = max(c.shape[0] for c in tiles)
    tiles = [cv2.copyMakeBorder(c, 0, h - c.shape[0], 0, 8, cv2.BORDER_CONSTANT, value=(255, 255, 255))
             for c in tiles]
    strip = cv2.hconcat(tiles)
    leg = np.full((44, strip.shape[1], 3), 255, np.uint8)
    cv2.line(leg, (16, 22), (60, 22), (0, 255, 0), 3)
    cv2.putText(leg, "SAM mask", (70, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    cv2.line(leg, (200, 22), (244, 22), (0, 0, 255), 3)
    cv2.putText(leg, "fitted CAD silhouette", (254, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.vconcat([strip, leg]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture_dir", type=Path, required=True)
    ap.add_argument("--mask_dir", type=Path, required=True,
                    help="<mask_dir>/<obj>/cam*_mask.png 구조")
    ap.add_argument("--cad", action="append", required=True, metavar="obj1=model.glb",
                    help="객체별 CAD. 여러 번 지정 가능.")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--depth_scale", type=float, default=0.001)
    ap.add_argument("--init_erode_px", default="auto",
                    help="초기 점군의 경계 오염 제거. 'auto'(권장)=물체 depth 그래디언트로 "
                         "flying-pixel 자동 제거, 정수=고정 침식 rim(px)")
    ap.add_argument("--init_voxel_m", default="auto",
                    help="초기 점군 voxel 크기. 'auto'(권장)=점 밀도·물체 스케일에서 유도, "
                         "실수=고정(m, 예: 0.002)")
    ap.add_argument("--w_depth", default="0",
                    help="depth 잔차 가중치. 'auto'(권장, 다양한 색/재질)=카메라 간 depth "
                         "일치도로 물체마다 자동(흰/무광이면 켜지고 검은/광택이면 꺼짐), "
                         "0=순수 실루엣, 실수=고정 가중.")
    ap.add_argument("--w_depth_max", type=float, default=20.0,
                    help="--w_depth auto 에서 depth 를 완전히 신뢰할 때의 상한 가중.")
    ap.add_argument("--max_fev", type=int, default=4000)
    ap.add_argument("--mask_uncertainty", action="store_true",
                    help="마스크를 +/-1px 팽창/침식해 재적합 → 치수 불확실성 보고 (3배 느림)")
    ap.add_argument("--save_overlay", action="store_true")
    args = ap.parse_args()

    cad = parse_cad(args.cad)
    erode_px = "auto" if str(args.init_erode_px) == "auto" else int(args.init_erode_px)
    voxel_m = "auto" if str(args.init_voxel_m) == "auto" else float(args.init_voxel_m)
    w_depth_auto = str(args.w_depth).strip().lower() == "auto"
    w_depth_fixed = None if w_depth_auto else float(args.w_depth)
    cams = discover_cams(args.capture_dir)
    print(f"[INFO] cameras: {cams}")
    print(f"[INFO] init cloud: erode={erode_px}  voxel={voxel_m}")
    if w_depth_auto:
        print(f"[INFO] w_depth=auto (cross-view 일치도로 물체마다 결정, max {args.w_depth_max})")
    else:
        print(f"[INFO] w_depth={w_depth_fixed} "
              f"({'pure silhouette' if w_depth_fixed == 0 else 'joint'})")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for obj, cad_path in cad.items():
        print(f"\n=== {obj}  <-  {cad_path.name} ===")
        mesh = trimesh.load(str(cad_path), force="mesh")
        cloud = build_cloud(args.capture_dir, args.mask_dir, obj, cams,
                            args.depth_scale, erode_px, voxel_m)
        print(f"  init cloud: {len(cloud)} points")

        if w_depth_auto:
            dcams = object_depth_cams(args.capture_dir, args.mask_dir, obj, cams, args.depth_scale)
            w_depth, wd_info = auto_w_depth(dcams, w_max=args.w_depth_max)
        else:
            w_depth, wd_info = w_depth_fixed, {"mode": "fixed", "w_depth": w_depth_fixed}

        views, cam_meta = build_views(args.capture_dir, args.mask_dir, obj, cams)
        fit = fit_cad_to_views(mesh, cloud, views, w_depth=w_depth, max_fev=args.max_fev)

        ext = np.sort(fit["extents_m"])[::-1] * 1000.0
        obb = np.sort(fit["cloud_obb_extents_m"])[::-1] * 1000.0
        print(f"  cloud OBB (no CAD) : {obb[0]:6.1f} x {obb[1]:6.1f} x {obb[2]:6.1f} mm")
        print(f"  CAD silhouette fit : {ext[0]:6.1f} x {ext[1]:6.1f} x {ext[2]:6.1f} mm")
        print(f"  mean IoU {fit['mean_iou']:.3f}  per-view "
              f"{[round(x, 3) for x in fit['per_view_iou']]}  "
              f"depth_rms {fit['depth_rms_mm']:.2f} mm  nfev {fit['n_fev']}")

        spread = None
        morph_ext = {}
        if args.mask_uncertainty:
            rows = [ext]
            for morph, key in ((+1, "dilate1px"), (-1, "erode1px")):
                v2, _ = build_views(args.capture_dir, args.mask_dir, obj, cams, morph=morph)
                f2 = fit_cad_to_views(mesh, cloud, v2, w_depth=w_depth, max_fev=args.max_fev)
                e2 = np.sort(f2["extents_m"])[::-1] * 1000.0
                rows.append(e2)
                morph_ext[key] = [float(x) for x in e2]
                print(f"  mask {key:>10}   : {e2[0]:6.1f} x {e2[1]:6.1f} x {e2[2]:6.1f} mm")
            spread = (np.max(rows, axis=0) - np.min(rows, axis=0))
            print(f"  mask +/-1px spread : {spread[0]:6.1f} x {spread[1]:6.1f} x {spread[2]:6.1f} mm"
                  f"   <- dominant uncertainty")

        T = np.eye(4)
        T[:3, :3] = fit["R_cad_to_world"]
        T[:3, 3] = fit["t_cad_to_world"]

        glb = args.out_dir / f"{obj}_cad_scaled.glb"
        Vs = save_scaled_cad(mesh, fit["scale"], glb)
        _, _, e_saved = obb_frame(Vs)
        if not np.allclose(np.sort(e_saved)[::-1] * 1000.0, ext, atol=1e-3):
            raise RuntimeError(f"{obj}: 저장된 glb 치수가 보고값과 다릅니다 "
                               f"({np.sort(e_saved)[::-1]*1000} vs {ext})")
        print(f"  [SAVE] {glb}")

        info = {
            "obj": obj,
            "cad": str(cad_path),
            "method": "cad_multiview_silhouette" if w_depth == 0 else "cad_silhouette_plus_depth",
            "w_depth": float(w_depth),
            "w_depth_mode": "auto" if w_depth_auto else "fixed",
            "depth_confidence": wd_info.get("confidence"),
            "cross_view_disagreement_mm": wd_info.get("disagreement_mm"),
            "scale_cad_to_world": fit["scale"],
            "T_world_cad_4x4": T.tolist(),
            "extents_m": fit["extents_m"].tolist(),
            "extents_mm_sorted_desc": [float(x) for x in ext],
            "mask_pm1px_spread_mm": None if spread is None else [float(x) for x in spread],
            "extents_mm_mask_dilate1px": morph_ext.get("dilate1px"),
            "extents_mm_mask_erode1px": morph_ext.get("erode1px"),
            "per_view_iou": fit["per_view_iou"],
            "mean_iou": fit["mean_iou"],
            "depth_rms_mm": fit["depth_rms_mm"],
            "cloud_obb_extents_mm_sorted_desc": [float(x) for x in obb],
            "init_cloud_points": int(len(cloud)),
            "cameras": cams,
            "note": ("depth is used only to initialise; the scale comes from the silhouettes. "
                     "cloud_obb_* is the CAD-free estimate and overestimates on noisy clouds."),
        }
        js = args.out_dir / f"{obj}_cad_fit.json"
        js.write_text(json.dumps(info, indent=2))
        print(f"  [SAVE] {js}")

        if args.save_overlay:
            ov = args.out_dir / f"{obj}_sil_overlay.jpg"
            overlay(args.capture_dir, cam_meta, views, mesh, fit, obj, ov)
            print(f"  [SAVE] {ov}")

        summary[obj] = info

    (args.out_dir / "cad_fit_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[SAVE] {args.out_dir / 'cad_fit_summary.json'}")
    print("\n[FINAL]")
    for obj, info in summary.items():
        e = info["extents_mm_sorted_desc"]
        sp = info["mask_pm1px_spread_mm"]
        tail = f"  (mask +/-1px: +/-{max(sp):.1f} mm)" if sp else ""
        print(f"  {obj}: {e[0]:.1f} x {e[1]:.1f} x {e[2]:.1f} mm   "
              f"IoU {info['mean_iou']:.3f}{tail}")


if __name__ == "__main__":
    main()
