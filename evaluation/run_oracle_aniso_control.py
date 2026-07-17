#!/usr/bin/env python3
"""통제 실험 — Oracle 을 **비등방**으로 재실행해 변환족을 Baseline 과 맞춘다.

왜 필요한가
-----------
메인 평가의 Baseline 은 비등방(축별 scale_vec, `sam3d_anisotropic_silhouette`)이고
Oracle 은 등방(단일 scale, `cad_multiview_silhouette`)이다. 변환족이 달라서
Baseline−Oracle gap 에 "SAM3D 형상 오차"와 "등방 vs 비등방" 차이가 섞여 있다.

이 스크립트는 **정답 CAD 를 비등방으로** 다시 정합해 Oracle 을 Baseline 과 같은
변환족에 놓는다. 그러면 gap 이 단서 메시(SAM3D vs 정답 CAD) 차이만 반영한다.

  기존 gap  = f(SAM3D 형상 오차, 등방/비등방)
  통제 gap  = f(SAM3D 형상 오차)                <- 이 스크립트

기존 추정 코드는 **읽기만** 한다 (Obj_Step3c_cad_scale 의 로더와 _silhouette_fit 의
fit_mesh_aniso 를 그대로 재사용). 결과는 evaluation/ 아래에만 쓴다.

  python evaluation/run_oracle_aniso_control.py \
    --config evaluation/evaluation_config.yaml --output evaluation/results
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_common as ec  # noqa: E402

sys.path.insert(0, str(ec._ROOT))
import Obj_Step3c_cad_scale as s3c            # noqa: E402  (읽기 전용 재사용)
from _silhouette_fit import fit_mesh_aniso, obb_frame  # noqa: E402

# Baseline(estimate_size_silhouette)이 쓴 값과 맞춘다 — 변환족 외의 변수를 통제
W_DEPTH = 0.0        # oracle *_cad_fit.json 의 w_depth 와 동일
MAX_FEV = 4000       # Obj_Step3_sam3d_scale --size_max_fev 기본값
ANISO_REG = 0.0      # --size_aniso_reg 기본값


def run_one(obj_cfg: dict, cfg: dict, out_dir: Path):
    """정답 CAD 를 비등방 정합. 반환: size_json 스키마 dict (baseline 과 동일 형식)."""
    name = obj_cfg["name"]
    mask_dir = Path(obj_cfg["mask_dir"])
    mask_parent, obj_key = mask_dir.parent, mask_dir.name      # build_* 는 <parent>/<obj>/cam*_mask.png 규약
    cap = ec.rp(obj_cfg["capture_dir"])
    spec = obj_cfg.get("oracle")
    if not spec:
        return None, "정답 CAD 없음 — 이 물체는 baseline 전용 (통제 실험 대상 아님)"
    mp = ec.rp(spec["mesh"])
    jp = ec.rp(spec["fit_json"])
    if not mp.exists():
        return None, f"CAD 없음 ({mp})"
    if not jp.exists():
        return None, f"oracle iso fit 없음 ({jp}) — warm start 에 필요"

    cams = s3c.discover_cams(cap)
    mesh = trimesh.load(str(mp), force="mesh")
    mesh = ec.decimate_like_pipeline(mesh, int(cfg.get("decimate_faces", 30000)))
    cloud = s3c.build_cloud(cap, ec.rp(mask_parent), obj_key, cams, 0.001, "auto", "auto")
    views, _ = s3c.build_views(cap, ec.rp(mask_parent), obj_key, cams)

    # ── 통제의 핵심: **발표된 iso oracle 해**를 warm start 로 넘긴다.
    # fit_mesh_aniso 는 warm 이 없으면 iso 를 스스로 다시 푸는데, 그러면 초기 점군/국소해가
    # 원본 실행과 달라져 "등방 대비 나빠지지 않는다"는 보장이 깨지고, gap 에 최적화 노이즈가
    # 섞인다. 같은 출발점에서 변환족만 넓히는 것이 이 실험의 목적이다.
    iso = json.loads(jp.read_text())
    Ti = np.array(iso["T_world_cad_4x4"], float)
    warm = dict(scale=float(iso["scale_cad_to_world"]),
                R_cad_to_world=Ti[:3, :3], t_cad_to_world=Ti[:3, 3])
    iso_ext = np.sort(np.asarray(iso["extents_mm_sorted_desc"], float))[::-1]

    print(f"  [{name}] CAD {len(mesh.faces)} faces, cloud {len(cloud)} pts, {len(views)} views")
    print(f"    warm start = 발표된 iso oracle (mean_iou {iso['mean_iou']:.4f}, "
          f"{iso_ext[0]:.2f} x {iso_ext[1]:.2f} x {iso_ext[2]:.2f} mm)")
    fit = fit_mesh_aniso(mesh, np.asarray(cloud, float), views,
                         w_depth=W_DEPTH, max_fev=MAX_FEV, aniso_reg=ANISO_REG, warm=warm)
    if fit["mean_iou"] + 1e-9 < float(iso["mean_iou"]):
        print(f"    [WARN] 비등방 IoU {fit['mean_iou']:.4f} < 등방 warm start "
              f"{iso['mean_iou']:.4f} — 최적화가 출발점보다 나빠졌습니다 (예상 밖)")

    ext = np.sort(fit["extents_m"])[::-1] * 1000.0
    T = np.eye(4)
    T[:3, :3] = fit["R_cad_to_world"]
    T[:3, 3] = fit["t_cad_to_world"]
    thr = cfg["iou_threshold"]

    # baseline 과 동일한 size_json 스키마로 저장 -> eval_common 의 baseline 로더가 그대로 읽는다
    d = dict(
        obj=name, mesh_source="oracle_cad_aniso_control", mesh=str(mp.relative_to(ec._ROOT)),
        method="cad_anisotropic_silhouette_control", anisotropic=True,
        shape_trusted=True, shape_ok_by_iou=bool(fit["mean_iou"] >= thr),
        min_iou_threshold=float(thr), w_depth=float(W_DEPTH), max_fev=int(MAX_FEV),
        scale_mesh_to_world=float(fit["scale"]),
        scale_vec=[float(v) for v in fit["scale_vec"]],
        T_world_mesh_4x4=T.tolist(),
        extents_m=np.asarray(fit["extents_m"]).tolist(),
        extents_mm_sorted_desc=[float(x) for x in ext],
        per_view_iou=[float(x) for x in fit["per_view_iou"]],
        mean_iou=float(fit["mean_iou"]),
        cloud_obb_extents_mm_sorted_desc=[float(x) for x in np.sort(fit["cloud_obb_extents_m"])[::-1] * 1000.0],
        init_cloud_points=int(len(cloud)), n_fev=int(fit["n_fev"]),
        cameras=cams,
        note=("통제 실험: 정답 CAD 를 Baseline 과 같은 비등방 변환족으로 정합. "
              "Baseline-Oracle gap 에서 등방/비등방 차이를 제거하기 위한 것."),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{name}_cad_fit_aniso.json"
    p.write_text(json.dumps(d, indent=2, ensure_ascii=False))
    print(f"    -> {ext[0]:7.2f} x {ext[1]:6.2f} x {ext[2]:6.2f} mm   mean_iou {fit['mean_iou']:.3f}  "
          f"nfev {fit['n_fev']}   [SAVE] {p.name}")
    return d, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--only", default="", help="쉼표구분 물체명 (기본: 전체)")
    a = ap.parse_args()

    cfg = ec.load_config(a.config)
    np.random.seed(cfg.get("seed", 0))
    out_dir = a.output / "control_oracle_aniso"
    only = {s.strip() for s in a.only.split(",") if s.strip()}

    print("\n[통제 실험] Oracle 을 비등방으로 재정합 (변환족을 Baseline 과 일치)")
    made = {}
    for o in cfg["objects"]:
        if only and o["name"] not in only:
            continue
        try:
            d, err = run_one(o, cfg, out_dir)
        except Exception as e:                      # 한 물체 실패가 전체를 막지 않게
            d, err = None, f"{type(e).__name__}: {e}"
        if d is None:
            print(f"  [SKIP] {o['name']}: {err}")
            continue
        made[o["name"]] = str((out_dir / f"{o['name']}_cad_fit_aniso.json").resolve()
                              .relative_to(ec._ROOT))

    idx = out_dir / "control_index.json"
    idx.write_text(json.dumps(made, indent=2, ensure_ascii=False))
    print(f"\n[SAVE] {idx}  ({len(made)}개 물체)")
    print("\n다음: analyze_controlled_gap.py 로 통제 gap 을 계산한다")


if __name__ == "__main__":
    main()
