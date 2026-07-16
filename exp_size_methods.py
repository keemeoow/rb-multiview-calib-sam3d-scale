#!/usr/bin/env python3
"""SAM3D 메시 크기추정 방법 비교 실험 (형상은 SAM3D 그대로, 정합 방식만 변형).

변형:
  1) iso/sil     : 등방 스케일, 실루엣만 (현재 기본)
  2) iso/+depth  : 등방 스케일, depth 잔차 추가
  3) aniso/sil   : 비등방(축별) 스케일, 실루엣만
  4) aniso/+depth: 비등방 + depth
치수는 최종 스케일 메시의 최소부피 OBB (다른 경로와 동일)로 재고 GT 와 비교한다.
"""
from pathlib import Path
import json
import numpy as np
import open3d as o3d
import trimesh

from Obj_Step3c_cad_scale import build_views, build_cloud, discover_cams
from _silhouette_fit import fit_cad_to_views, fit_mesh_aniso, obb_frame

CAP = Path("data(3)/capture_obj")
MK = Path("data(3)/masks")
GT = {"peg": [45, 30, 30], "hole": [50, 50, 50]}
MESH = {"peg": "data(3)/outputs_sam3d_fit/objpeg/objpeg_sam3d.glb",
        "hole": "data(3)/outputs_sam3d_fit/objhole/objhole_sam3d.glb"}
W_DEPTH = 3.0   # +depth 변형에서 강제로 넣는 depth 가중치 (auto 는 0 이었음)


def load_decimated(path, target=30000):
    m = trimesh.load(path, force="mesh")
    me = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.ascontiguousarray(m.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.ascontiguousarray(m.faces, dtype=np.int32)))
    if len(m.faces) > target:
        me = me.simplify_quadric_decimation(target)
    return trimesh.Trimesh(vertices=np.asarray(me.vertices),
                           faces=np.asarray(me.triangles), process=False)


def ext_mm(fit):
    return np.sort(np.asarray(fit["extents_m"]))[::-1] * 1000.0


def err(ext, g):
    e = np.asarray(ext) - np.asarray(sorted(g, reverse=True))
    return float(np.mean(np.abs(e))), float(np.max(np.abs(e)))


def main():
    cams = discover_cams(CAP)
    rows = []
    out = {}
    for obj in ["peg", "hole"]:
        print(f"\n########## {obj} ##########")
        mesh = load_decimated(MESH[obj])
        views, _ = build_views(CAP, MK, obj, cams, morph=0)
        cloud = build_cloud(CAP, MK, obj, cams, 0.001, "auto", "auto")
        g = GT[obj]

        MF = 3000
        iso_sil = fit_cad_to_views(mesh, cloud, views, w_depth=0.0, max_fev=MF)
        iso_dep = fit_cad_to_views(mesh, cloud, views, w_depth=W_DEPTH, max_fev=MF)
        variants = {
            "iso/sil":      lambda: iso_sil,
            "iso/+depth":   lambda: iso_dep,
            "aniso/sil":    lambda: fit_mesh_aniso(mesh, cloud, views, w_depth=0.0, max_fev=MF, warm=iso_sil),
            "aniso/+depth": lambda: fit_mesh_aniso(mesh, cloud, views, w_depth=W_DEPTH, max_fev=MF, warm=iso_dep),
        }
        out[obj] = {}
        for name, fn in variants.items():
            fit = fn()
            ext = ext_mm(fit)
            mae, mx = err(ext, g)
            iou = fit["mean_iou"]
            sv = fit.get("scale_vec")
            rows.append((obj, name, ext, iou, mae, mx))
            out[obj][name] = {"extents_mm": [float(x) for x in ext], "mean_iou": iou,
                              "mean_abs_err_mm": mae, "max_err_mm": mx, "scale_vec": sv}
            svs = ("  s=[" + ", ".join(f"{v:.3f}" for v in sv) + "]") if sv else ""
            print(f"  {name:13} {ext[0]:5.1f} x {ext[1]:5.1f} x {ext[2]:5.1f} mm  "
                  f"IoU {iou:.3f}  MAE {mae:.2f}mm  max {mx:.2f}mm{svs}")

    print("\n\n================= 요약 (GT 대비) =================")
    print(f"{'obj':5} {'method':13} {'ext(mm)':>22} {'IoU':>6} {'MAE':>7} {'max':>7}")
    last = None
    for obj, name, ext, iou, mae, mx in rows:
        if obj != last:
            print("-" * 70); last = obj
        es = " x ".join(f"{v:5.1f}" for v in ext)
        print(f"{obj:5} {name:13} {es:>22} {iou:6.3f} {mae:6.2f}mm {mx:6.2f}mm")
    Path("data(3)/outputs_sam3d_fit/size_method_experiment.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    print("\nSaved data(3)/outputs_sam3d_fit/size_method_experiment.json")


if __name__ == "__main__":
    main()
