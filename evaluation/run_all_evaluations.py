#!/usr/bin/env python3
"""Real-to-Sim 크기 추정 정량 평가 — 전체 재생성 진입점.

  python evaluation/run_all_evaluations.py \
    --config evaluation/evaluation_config.yaml --output evaluation/results

보고 지표는 아래 6가지로 한정한다. 그 밖의 파생 수치(gap, 통계검정, IoU 통과율,
R², 상관계수 등)는 계산하지도 출력하지도 않는다.

  1. 축별 절대오차           e_L / e_W / e_H          [mm]   -> Figure 3
  2. 평균 Dimension Error    E_dim                    [mm]   -> Figure 1
  3. 평균 상대 Dimension Error E_rel                  [%]    -> Figure 2
  4. Cross-view Silhouette IoU                               -> Figure 4
  5. Normalized Contour Distance                      [%]    -> Figure 5
  6. 정성 오버레이 이미지                                     -> Figure 6

기존 추정 코드는 읽기만 하고 수정하지 않는다. 결과 파일에서 값을 읽어 계산하며,
없는 값은 만들어내지 않고 skip 사유와 함께 기록한다.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_common as ec                       # noqa: E402
import evaluate_metric_scale as ems            # noqa: E402
import evaluate_silhouette_alignment as esa    # noqa: E402
import generate_evaluation_figures as gef      # noqa: E402
import generate_qualitative_overlays as gqo    # noqa: E402

BASE, ORA = "baseline_sam3d", "oracle_cad"

# 이 스크립트가 예전에 만들었던, 이제 보고하지 않거나 이름이 바뀐 산출물.
# 남아 있으면 최신 결과로 오해되므로 매 실행마다 지운다.
# (analyze_controlled_gap.py 같은 별도 실행 스크립트의 산출물은 건드리지 않는다.)
RETIRED = ("fig2_gt_vs_estimated_dimensions", "fig6_iou_vs_dimension_error",
           "fig7_qualitative_real_to_sim_grid", "fig7_qualitative_three_views",
           "fig1_baseline_vs_oracle_dim_error", "fig1_baseline_vs_oracle_dim_error_horizontal",
           "fig1_mean_dimension_error_horizontal", "fig2_scatter_points")
RETIRED_FILES = ("csv/paired_tests.json", "csv/fig2_scatter_points.csv",
                 "captions/fig7_caption.txt")


def build_summary(obj: pd.DataFrame) -> pd.DataFrame:
    """방법별 mean ± std — 보고 지표 4종(E_dim, E_rel, cross IoU, D_contour)만."""
    rows = []
    for m in ec.METHODS:
        s = obj[obj.method == m]
        if s.empty:
            continue
        de_m, de_s, n = ec.mean_std(s.mean_dimension_error_mm.tolist())
        re_m, re_s, _ = ec.mean_std(s.mean_relative_dimension_error_percent.tolist())
        iou_m, iou_s, _ = ec.mean_std(s.cross_view_iou.tolist())
        cd_m, cd_s, _ = ec.mean_std(s.cross_view_normalized_contour_distance_percent.tolist())
        rows.append(dict(
            method=m, object_count=n,
            mean_dimension_error_mean_mm=de_m, mean_dimension_error_std_mm=de_s,
            mean_relative_error_mean_percent=re_m, mean_relative_error_std_percent=re_s,
            cross_view_iou_mean=iou_m, cross_view_iou_std=iou_s,
            normalized_contour_distance_mean_percent=cd_m,
            normalized_contour_distance_std_percent=cd_s,
        ))
    return pd.DataFrame(rows)


def clean_retired(out: Path):
    """이전 버전이 만든, 이제 보고하지 않는 그림·CSV 를 지운다."""
    n = 0
    for d in [out / "figures", *sorted((out / "per_object").glob("*"))]:
        for stem in RETIRED:
            for f in d.glob(f"{stem}.*"):
                f.unlink(); n += 1
    for rel in RETIRED_FILES:
        f = out / rel
        if f.exists():
            f.unlink(); n += 1
    if n:
        print(f"  [CLEAN] 보고 대상 아닌 예전 산출물 {n}개 삭제")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--skip_qualitative", action="store_true")
    args = ap.parse_args()

    cfg = ec.load_config(args.config)
    random.seed(cfg.get("seed", 0)); np.random.seed(cfg.get("seed", 0))

    out = args.output
    (out / "csv").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "qualitative").mkdir(parents=True, exist_ok=True)
    (out / "captions").mkdir(parents=True, exist_ok=True)
    clean_retired(out)

    print("\n[1/6] 축별 절대오차 · E_dim · E_rel")
    m_rows, m_skip = ems.evaluate(cfg)
    print("\n[2/6] Cross-view IoU · Normalized contour distance")
    cam_rows, s_rows, s_skip = esa.evaluate(cfg)

    m = pd.DataFrame(m_rows)
    s = pd.DataFrame(s_rows)
    cam = pd.DataFrame(cam_rows)
    if m.empty:
        raise SystemExit("평가 가능한 물체가 없습니다 — config 경로를 확인하세요")
    obj = m.merge(s, on=["object_name", "method"], how="left", suffixes=("", "_sil"))
    obj["per_camera_iou"] = obj["per_camera_iou"].apply(
        lambda v: json.dumps(v) if isinstance(v, dict) else v)

    print("\n[3/6] CSV")
    obj.to_csv(out / "csv" / "evaluation_per_object.csv", index=False)
    cam.to_csv(out / "csv" / "evaluation_per_camera.csv", index=False)
    summ = build_summary(obj)
    summ.to_csv(out / "csv" / "evaluation_summary.csv", index=False)
    skipped = pd.DataFrame(m_skip + s_skip)
    if not skipped.empty:
        skipped.to_csv(out / "csv" / "skipped_objects.csv", index=False)
    for f in ("evaluation_per_object.csv", "evaluation_per_camera.csv", "evaluation_summary.csv"):
        print(f"  [SAVE] csv/{f}")

    print("\n[4/6] Figures 1~5 — 통합 (전체 객체)")
    gef.generate_all(obj, cam, out / "figures", cfg)

    print("\n[5/6] Figures 1~5 — 물체별")
    for name in list(dict.fromkeys(obj.object_name)):
        d = out / "per_object" / name
        disp = obj[obj.object_name == name].display_name.iloc[0]
        print(f"  --- {disp} -> per_object/{name}/")
        gef.generate_all(obj[obj.object_name == name], cam[cam.object_name == name],
                         d, cfg, suffix=disp)

    reps = []
    if not args.skip_qualitative:
        print("\n[6/6] Figure 6 — 정성 오버레이")
        gqo.per_object(cfg, cam, out / "qualitative", also_dir=out / "per_object")
        reps = gqo.grid(cfg, obj, cam, out / "figures")

    write_captions(out / "captions", cfg)
    write_report(out, cfg, obj, summ, skipped, reps)
    print(f"\n완료 → {out}")


def write_captions(d: Path, cfg):
    d.mkdir(parents=True, exist_ok=True)
    caps = {
        "fig1": ("Figure 1. Mean Dimension Error (E_dim) per object for Baseline (SAM3D cue mesh) and "
                 "Oracle (ground-truth CAD cue mesh) under the identical Sim(3) silhouette-fitting engine. "
                 "Lower is better. The rightmost group is the across-object mean."),
        "fig2": ("Figure 2. Mean Relative Dimension Error (E_rel, %) per object — each axis error divided "
                 "by its ground-truth length before averaging, so objects of different size are "
                 "comparable. Lower is better."),
        "fig3": ("Figure 3. Per-axis absolute error (mm). Lower is better. Cells marked n/d have "
                 "undetermined ground truth. This view exposes anisotropic shape distortion that the "
                 "per-object mean hides."),
        "fig4": ("Figure 4. Cross-view silhouette IoU (mean over cameras, SAM3D source view excluded "
                 "because the cue mesh was generated from it). Higher is better. Open circles are the "
                 "individual cross-view cameras."),
        "fig5": ("Figure 5. Normalized contour distance (bidirectional mean nearest-neighbour distance "
                 "between real and simulated mask contours, normalized by the real-mask bbox diagonal), "
                 "averaged over the cross-view cameras. Lower is better."),
        "fig6": ("Figure 6. Qualitative Real-to-Sim alignment. Green: real SAM mask contour. "
                 "Red: simulated silhouette of the fitted mesh. The SAM3D source view is faded because "
                 "the cue mesh was generated from it; the cross-view panels show generalization."),
    }
    for k, v in caps.items():
        (d / f"{k}_caption.txt").write_text(v + "\n")
    print(f"  [SAVE] captions/ ({len(caps)} files)")


def _fmt(v, p=2):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/d"
    return f"{v:.{p}f}"


def write_report(out: Path, cfg, obj, summ, skipped, reps):
    L = []
    A = L.append
    A("# Real-to-Sim Object Size Estimation — Evaluation Report\n")
    A("생성: `python evaluation/run_all_evaluations.py --config evaluation/evaluation_config.yaml "
      "--output evaluation/results`\n")

    A("\n## 1. 탐색한 기존 코드와 결과 파일\n")
    A("| 종류 | 파일 | 역할 |")
    A("|---|---|---|")
    A("| 엔진 | `_silhouette_fit.py` | `fit_cad_to_views`(등방), `fit_mesh_aniso`(비등방), "
      "`render_silhouette`, `per_view_iou`, `obb_frame` |")
    A("| Baseline | `Obj_Step3_sam3d_scale.py` | SAM3D mesh 생성 + `--estimate_size` (비등방 기본) → `*_size.json` |")
    A("| Oracle | `Obj_Step3c_cad_scale.py` | 정답 CAD 실루엣 정합 → `*_cad_fit.json` |")
    A("| 기존 비교 | `Obj_Step3d_compare_gt.py` | GT 대비 비교 그림 (이번 평가와 별개, 수정하지 않음) |")
    A("| 방법 선택 근거 | `data(3)/outputs_sam3d_fit/size_method_experiment.json` | iso vs aniso 실험 "
      "(peg: iso 4.37mm → aniso 1.05mm) → 비등방 채택 |")
    A("\n> 기존 추정 코드는 **읽기만** 했고 수정하지 않았다. 평가 코드는 `evaluation/` 에 신규 작성했다.\n")

    A("\n## 2. 실제 사용한 데이터 경로\n")
    A("| Object | capture | mask | Baseline (`*_size.json`) | Oracle (`*_cad_fit.json`) | source cam |")
    A("|---|---|---|---|---|---|")
    for o in cfg["objects"]:
        ora = (o.get("oracle") or {}).get("fit_json")     # 정답 CAD 가 없는 물체도 있다
        A(f"| {o['name']} | `{o['capture_dir']}` | `{o['mask_dir']}` | "
          f"`{o['baseline']['size_json']}` | {f'`{ora}`' if ora else '— (정답 CAD 없음)'} | "
          f"{o['source_camera']} |")

    A("\n## 3. GT dimension 출처\n")
    A("| Object | GT (mm, L≥W≥H) | 출처 | 비고 |")
    A("|---|---|---|---|")
    for o in cfg["objects"]:
        g = " × ".join("n/d" if x is None else f"{x:.1f}" for x in o["gt_mm"])
        A(f"| {o['name']} | {g} | `{o['gt_source']}` | {o.get('gt_note') or '—'} |")
    A("\n- `caliper_manual`: `configs/evaluation.yaml` 의 `measured.extents_mm` (캘리퍼 실측).")
    A("- `cad_design_nominal` (T_shape): `T_shape.glb` 의 고유 OBB 가 정확히 150/100/50 이고 추정 "
      "scale 이 0.001(mm→m)에 0.36% 로 수렴 → mm 단위 설계 CAD 로 판단해 **설계값**을 GT 로 사용. "
      "**캘리퍼 실측이 아니다.**")
    A("- `caliper_manual_partial`: 일부 축만 확정. 미확정 축은 오차 계산에서 제외한다 (§9).")

    A("\n## 4. Baseline / Oracle 정의\n")
    A("- **Baseline (`baseline_sam3d`)**: SAM3D 가 단일 RGB+mask 로 만든 unitless 단서 mesh 를 "
      "3대 카메라 SAM2 실루엣에 Sim(3) 7-DoF 로 정합. 최종 치수 = mesh OBB extents × 추정 scale. "
      "**실제 운용 방법** (정답 CAD 불필요).")
    A("- **Oracle (`oracle_cad`)**: 단서 mesh 만 정답 CAD 로 교체. 엔진·손실·최적화 동일. "
      "**형상이 완벽할 때의 정확도 상한**.")
    A(f"- 현재 데이터의 Baseline 은 모두 `sam3d_anisotropic_silhouette` (축별 scale_vec), "
      f"Oracle 은 `cad_multiview_silhouette` (등방 scale) 이다. "
      f"→ **두 방법이 완전히 동일한 변환족은 아니다** (§11).")

    A("\n## 5. 평가 지표 수식\n")
    A("보고하는 지표는 아래 5개(+정성 오버레이)뿐이다.\n")
    A("```")
    A("[1] e_L = |L_hat - L_GT|,  e_W = |W_hat - W_GT|,  e_H = |H_hat - H_GT|   [mm]")
    A("[2] E_dim = (e_L + e_W + e_H) / 3                                        [mm]")
    A("[3] E_rel = (1/3)[e_L/L_GT + e_W/W_GT + e_H/H_GT] x 100                  [%]")
    A("[4] IoU_i = |M_real_i ∩ M_sim_i| / |M_real_i ∪ M_sim_i|")
    A("    Cross-view IoU = mean over cameras EXCLUDING the SAM3D source view")
    A("[5] D_contour = 0.5[ mean_{p∈C_real} min_q ||p-q|| + mean_{q∈C_sim} min_p ||q-p|| ]  [px]")
    A("    D_contour_norm = D_contour / sqrt(w_real^2 + h_real^2)               [%]")
    A("[6] 정성 오버레이: real(green) / sim(red) 실루엣 외곽선 중첩 이미지")
    A("```")
    A("- 축 대응은 파이프라인·GT 모두 내림차순(L≥W≥H) 정렬 기준이다 (`rank_descending`).")
    A("- GT 미확정 축은 e_* 를 계산하지 않고 평균에서 제외 (`gt_axes_used` 에 사용 축 수 기록).")
    A("- sim 마스크는 결과 JSON 의 fit 파라미터로 재렌더링했고, IoU 는 파이프라인이 저장한 "
      "`per_view_iou` 를 그대로 쓴다. contour distance 만 재렌더 마스크에서 계산한다.")

    A("\n## 6. 객체별 결과\n")
    A("| Object | Method | GT (mm) | Estimated (mm) | e_L | e_W | e_H | **E_dim** (mm) | E_rel (%) | "
      "**cross IoU** | cross D_contour (%) |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in obj.iterrows():
        gt = " × ".join(_fmt(r[f"gt_{a}_mm"], 1) for a in cfg["axis_labels"])
        es = " × ".join(_fmt(r[f"estimated_{a}_mm"], 1) for a in cfg["axis_labels"])
        A(f"| {r.display_name} | `{r.method}` | {gt} | {es} | "
          f"{_fmt(r.abs_error_L_mm)} | {_fmt(r.abs_error_W_mm)} | {_fmt(r.abs_error_H_mm)} | "
          f"**{_fmt(r.mean_dimension_error_mm)}** | {_fmt(r.mean_relative_dimension_error_percent)} | "
          f"**{_fmt(r.cross_view_iou, 3)}** | "
          f"{_fmt(r.cross_view_normalized_contour_distance_percent)} |")
    A("\n> `n/d` = GT 미확정 축이라 그 축의 오차를 계산하지 않았다는 뜻이다 (§9).")

    A("\n## 7. 전체 (mean ± std)\n")
    A("| Method | n | E_dim (mm) | E_rel (%) | Cross-view IoU | D_contour (%) |")
    A("|---|---|---|---|---|---|")
    for _, r in summ.iterrows():
        A(f"| `{r.method}` | {int(r.object_count)} | "
          f"{_fmt(r.mean_dimension_error_mean_mm)} ± {_fmt(r.mean_dimension_error_std_mm)} | "
          f"{_fmt(r.mean_relative_error_mean_percent)} ± {_fmt(r.mean_relative_error_std_percent)} | "
          f"{_fmt(r.cross_view_iou_mean, 3)} ± {_fmt(r.cross_view_iou_std, 3)} | "
          f"{_fmt(r.normalized_contour_distance_mean_percent)} ± "
          f"{_fmt(r.normalized_contour_distance_std_percent)} |")

    A("\n## 8. 생성한 그래프와 피규어\n")
    A("모두 PNG(300dpi). 각 그림 상단에 `번호. 제목` 이 표기된다. "
      "지표 6개에 그림 6개가 1:1 로 대응한다.\n")
    A("### 통합 (전체 객체) — `figures/`\n")
    for n, t in [("fig1_mean_dimension_error", "1. 평균 Dimension Error (E_dim)"),
                 ("fig2_mean_relative_dimension_error", "2. 평균 상대 Dimension Error (E_rel)"),
                 ("fig3_per_axis_absolute_error_baseline", "3a. 축별 절대오차 (Baseline)"),
                 ("fig3_per_axis_absolute_error_oracle", "3b. 축별 절대오차 (Oracle)"),
                 ("fig4_cross_view_silhouette_iou", "4. Cross-view Silhouette IoU"),
                 ("fig5_normalized_contour_distance", "5. Normalized Contour Distance"),
                 ("fig6_qualitative_real_to_sim_grid", "6. 정성 오버레이 통합 그리드")]:
        A(f"- [{t}](figures/{n}.png)")
    if reps:
        A(f"\n대표 객체 (6. 그리드 행): {', '.join(reps)} — 최소오차/중간/최대오차 + 단순·복잡 형상 포함")
    A("\n### 물체별 — `per_object/<object>/`  ·  오버레이 원본 — `qualitative/`\n")
    A("| Object | 그림 |")
    A("|---|---|")
    for o in cfg["objects"]:
        n = o["name"]
        links = " · ".join(
            f"[{t}](per_object/{n}/{f}.png)" for f, t in [
                ("fig1_mean_dimension_error", "1"),
                ("fig2_mean_relative_dimension_error", "2"),
                ("fig3_per_axis_absolute_error_baseline", "3a"),
                ("fig3_per_axis_absolute_error_oracle", "3b"),
                ("fig4_cross_view_silhouette_iou", "4"),
                ("fig5_normalized_contour_distance", "5"),
                ("fig6_qualitative_three_views", "6")])
        A(f"| {o.get('display_name', n)} | {links} |")

    A("\n## 9. 누락·제외된 객체와 이유\n")
    A("| 대상 | 문제 | 처리 |")
    A("|---|---|---|")
    # GT 가 null 인 축은 config 에서 읽어 실제 상태만 적는다 (하드코딩하면 config 와 어긋난다)
    for o in cfg["objects"]:
        und = [cfg["axis_labels"][i] for i, g in enumerate(o["gt_mm"]) if g is None]
        if und:
            A(f"| {o['name']} {'/'.join(und)} 축 | GT 미확정 ({o.get('gt_note') or '사유 미기재'}) | "
              f"**해당 축만** 오차 계산에서 제외 (`gt_axes_used={3 - len(und)}/3`). "
              f"객체는 유지하고 추정치는 CSV/그림에 남김 |")
    if not skipped.empty:
        # 같은 사유가 metric/silhouette 두 단계에서 각각 기록되므로 표에서는 한 번만 보인다
        for _, r in skipped.drop_duplicates(subset=["object_name", "method", "reason"]).iterrows():
            A(f"| {r.object_name} / {r.method} | {r.reason} | skip (`csv/skipped_objects.csv`) |")
    else:
        A("| — | 파일 누락으로 skip 된 객체 없음 | — |")
    A("\n필요한 추가 데이터:")
    for o in cfg["objects"]:
        und = [i for i, g in enumerate(o["gt_mm"]) if g is None]
        if und:
            A(f"- **{o['name']} 캘리퍼 실측** — `evaluation_config.yaml` 의 "
              f"`{o['name']}.gt_mm[{und[0]}]` 을 null → 숫자로 바꾸면 3/3 축 평가가 된다.")
    if any(o.get("gt_source") == "cad_design_nominal" for o in cfg["objects"]):
        A("- **설계값(nominal) GT 물체의 캘리퍼 실측** — 제조 공차가 반영돼 있지 않다.")
    A(f"- 객체 수 {len(cfg['objects'])}개는 일반화를 주장하기에 부족하다. YCB 등 추가 객체 권장.")

    A("\n## 10. 결과 요약\n")
    bm = summ[summ.method == BASE].iloc[0]
    A(f"1. **크기 정확도 (지표 1~3)**: Baseline E_dim = {_fmt(bm.mean_dimension_error_mean_mm)} ± "
      f"{_fmt(bm.mean_dimension_error_std_mm)} mm, E_rel = "
      f"{_fmt(bm.mean_relative_error_mean_percent)} ± {_fmt(bm.mean_relative_error_std_percent)} %. "
      f"축별 분포는 Figure 3 참조 — 물체 평균이 가리는 축별 편차가 보인다.")
    A(f"2. **Real-to-Sim 정합 (지표 4~5)**: Baseline cross-view IoU = "
      f"{_fmt(bm.cross_view_iou_mean, 3)} ± {_fmt(bm.cross_view_iou_std, 3)}, "
      f"D_contour = {_fmt(bm.normalized_contour_distance_mean_percent)} ± "
      f"{_fmt(bm.normalized_contour_distance_std_percent)} %.")
    A("3. **정성 확인 (지표 6)**: Figure 6 및 `qualitative/` 의 오버레이에서 real/sim 외곽선 "
      "일치를 눈으로 확인한다. source view 는 mesh 를 만든 뷰라 흐리게 처리했다.")

    A("\n## 11. 지표 해석 시 주의\n")
    A("- **IoU 높음 → 3D 치수 정확** : 성립하지 않는다. Cross-view IoU 와 D_contour 는 "
      "**크기·형상·pose·캘리브레이션이 뒤섞인 투영 정합 품질**이고, E_dim/E_rel 만이 실측 기반 "
      "3D 크기 정확도다. 두 지표군을 바꿔 쓰지 말 것.")
    A("- **Baseline 과 Oracle 의 직접 비교** : 현재 Baseline 은 비등방(축별 scale_vec), "
      "Oracle 은 등방(단일 scale) 이라 **변환족이 다르다**. 두 열은 각각의 지표값으로 읽고, "
      "차이를 형상 오차로 환산하지 말 것.")
    A("- **일반화** : 4개 객체(단순 3 + 복잡 1), 단일 촬영 세션, 단일 센서(RealSense), "
      "source view 는 전부 cam0. 다른 재질·크기·카메라 배치로의 일반화는 이 데이터로 알 수 없다.")
    nominal = [o["name"] for o in cfg["objects"] if o.get("gt_source") == "cad_design_nominal"]
    if nominal:
        A(f"- **{', '.join(nominal)}** : GT 가 설계값이라 '실측 대비 정확도'가 아니라 "
          "'설계값 대비 일치도'다.")
    partial = [o["name"] for o in cfg["objects"] if any(g is None for g in o["gt_mm"])]
    if partial:
        A(f"- **{', '.join(partial)} 절대 정확도** : 미확정 축이 있어 3축 전체 정확도는 알 수 없다.")

    (out / "evaluation_report.md").write_text("\n".join(L) + "\n")
    print(f"  [SAVE] evaluation_report.md")


if __name__ == "__main__":
    main()
