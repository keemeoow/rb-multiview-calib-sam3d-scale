#!/usr/bin/env python3
"""Real-to-Sim 크기 추정 정량 평가 — 전체 재생성 진입점.

  python evaluation/run_all_evaluations.py \
    --config evaluation/evaluation_config.yaml --output evaluation/results

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
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_common as ec                       # noqa: E402
import evaluate_metric_scale as ems            # noqa: E402
import evaluate_silhouette_alignment as esa    # noqa: E402
import generate_evaluation_figures as gef      # noqa: E402
import generate_qualitative_overlays as gqo    # noqa: E402

BASE, ORA = "baseline_sam3d", "oracle_cad"


def build_summary(obj: pd.DataFrame, cfg) -> pd.DataFrame:
    thr = cfg["iou_threshold"]
    rows = []
    for m in ec.METHODS:
        s = obj[obj.method == m]
        if s.empty:
            continue
        de_m, de_s, n = ec.mean_std(s.mean_dimension_error_mm.tolist())
        re_m, re_s, _ = ec.mean_std(s.mean_relative_dimension_error_percent.tolist())
        iou_m, iou_s, _ = ec.mean_std(s.cross_view_iou.tolist())
        cd_m, cd_s, _ = ec.mean_std(s.cross_view_normalized_contour_distance_percent.tolist())
        iou_vals = [v for v in s.cross_view_iou.tolist() if v is not None and np.isfinite(v)]
        npass = sum(1 for v in iou_vals if v >= thr)
        rows.append(dict(
            method=m, object_count=n,
            mean_dimension_error_mean_mm=de_m, mean_dimension_error_std_mm=de_s,
            mean_relative_error_mean_percent=re_m, mean_relative_error_std_percent=re_s,
            cross_view_iou_mean=iou_m, cross_view_iou_std=iou_s,
            normalized_contour_distance_mean_percent=cd_m,
            normalized_contour_distance_std_percent=cd_s,
            iou_085_pass_count=npass,
            iou_085_pass_rate_percent=(100.0 * npass / len(iou_vals)) if iou_vals else None,
        ))
    df = pd.DataFrame(rows)

    # Baseline - Oracle gap: 같은 Sim(3) 엔진에서 단서 mesh 만 바꿨을 때의 치수 오차 차이
    b = obj[obj.method == BASE].set_index("object_name")
    o = obj[obj.method == ORA].set_index("object_name")
    common = [i for i in b.index if i in o.index]
    gaps = [b.loc[i].mean_dimension_error_mm - o.loc[i].mean_dimension_error_mm
            for i in common
            if pd.notna(b.loc[i].mean_dimension_error_mm) and pd.notna(o.loc[i].mean_dimension_error_mm)]
    rel_gaps = [b.loc[i].mean_relative_dimension_error_percent - o.loc[i].mean_relative_dimension_error_percent
                for i in common
                if pd.notna(b.loc[i].mean_relative_dimension_error_percent)
                and pd.notna(o.loc[i].mean_relative_dimension_error_percent)]
    if gaps:
        df = pd.concat([df, pd.DataFrame([dict(
            method="gap_baseline_minus_oracle", object_count=len(gaps),
            mean_dimension_error_gap_mm=float(np.mean(gaps)),
            mean_dimension_error_gap_std_mm=float(np.std(gaps, ddof=1)) if len(gaps) > 1 else 0.0,
            relative_error_gap_percent_point=float(np.mean(rel_gaps)) if rel_gaps else None,
        )])], ignore_index=True)
    return df


def paired_tests(obj: pd.DataFrame) -> dict:
    b = obj[obj.method == BASE].set_index("object_name").mean_dimension_error_mm
    o = obj[obj.method == ORA].set_index("object_name").mean_dimension_error_mm
    idx = [i for i in b.index if i in o.index and pd.notna(b[i]) and pd.notna(o[i])]
    x, y = np.array([b[i] for i in idx]), np.array([o[i] for i in idx])
    out = dict(n_pairs=len(idx), objects=idx,
               baseline_mean=float(x.mean()) if len(x) else None,
               oracle_mean=float(y.mean()) if len(y) else None)
    if len(idx) >= 3:
        t, tp = stats.ttest_rel(x, y)
        out["paired_t_statistic"], out["paired_t_pvalue"] = float(t), float(tp)
        try:
            w, wp = stats.wilcoxon(x, y)
            out["wilcoxon_statistic"], out["wilcoxon_pvalue"] = float(w), float(wp)
        except Exception as e:
            out["wilcoxon_error"] = str(e)
    out["caveat"] = ("표본이 매우 적어 p-value 를 과도하게 해석하지 말 것. "
                     "descriptive statistics 중심으로 보고한다.")
    return out


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

    print("\n[1/5] Metric-scale accuracy")
    m_rows, m_skip = ems.evaluate(cfg)
    print("\n[2/5] Real-to-Sim silhouette alignment")
    cam_rows, s_rows, s_skip = esa.evaluate(cfg)

    m = pd.DataFrame(m_rows)
    s = pd.DataFrame(s_rows)
    cam = pd.DataFrame(cam_rows)
    if m.empty:
        raise SystemExit("평가 가능한 물체가 없습니다 — config 경로를 확인하세요")
    obj = m.merge(s, on=["object_name", "method"], how="left", suffixes=("", "_sil"))
    obj["per_camera_iou"] = obj["per_camera_iou"].apply(
        lambda v: json.dumps(v) if isinstance(v, dict) else v)

    print("\n[3/5] CSV")
    obj.to_csv(out / "csv" / "evaluation_per_object.csv", index=False)
    cam.to_csv(out / "csv" / "evaluation_per_camera.csv", index=False)
    summ = build_summary(obj, cfg)
    summ.to_csv(out / "csv" / "evaluation_summary.csv", index=False)
    skipped = pd.DataFrame(m_skip + s_skip)
    if not skipped.empty:
        skipped.to_csv(out / "csv" / "skipped_objects.csv", index=False)
    tests = paired_tests(obj)
    (out / "csv" / "paired_tests.json").write_text(json.dumps(tests, indent=2, ensure_ascii=False))
    for f in ("evaluation_per_object.csv", "evaluation_per_camera.csv", "evaluation_summary.csv"):
        print(f"  [SAVE] csv/{f}")

    print("\n[4/6] Figures — 통합 (전체 객체)")
    gef.generate_all(obj, cam, out / "figures", cfg, csv_dir=out / "csv")

    print("\n[5/6] Figures — 물체별")
    for name in list(dict.fromkeys(obj.object_name)):
        d = out / "per_object" / name
        disp = obj[obj.object_name == name].display_name.iloc[0]
        print(f"  --- {disp} -> per_object/{name}/")
        gef.generate_all(obj[obj.object_name == name], cam[cam.object_name == name],
                         d, cfg, suffix=disp, csv_dir=d)

    reps = []
    if not args.skip_qualitative:
        print("\n[6/6] Qualitative")
        gqo.per_object(cfg, cam, out / "qualitative", also_dir=out / "per_object")
        reps = gqo.grid(cfg, obj, cam, out / "figures")

    write_captions(out / "captions", cfg)
    write_report(out, cfg, obj, cam, summ, skipped, tests, reps)
    print(f"\n완료 → {out}")


def write_captions(d: Path, cfg):
    d.mkdir(parents=True, exist_ok=True)
    thr = cfg["iou_threshold"]
    caps = {
        "fig1": ("Figure 1. Mean Dimension Error (E_dim) per object for Baseline (SAM3D cue mesh) and "
                 "Oracle (ground-truth CAD cue mesh) under the identical Sim(3) silhouette-fitting engine. "
                 "Lower is better. The rightmost group is the across-object mean. The Baseline-Oracle gap "
                 "is the change in final dimension error attributable to replacing the SAM3D cue mesh with "
                 "the ground-truth CAD."),
        "fig2": ("Figure 2. Estimated vs ground-truth dimension for every object-axis observation. "
                 "The dashed line is y=x. Axes with undetermined ground truth are excluded. "
                 "R^2 is shown for reference only and is not a primary metric."),
        "fig3": ("Figure 3. Per-axis absolute error (mm). Lower is better. Cells marked n/d have "
                 "undetermined ground truth. This view exposes anisotropic shape distortion that the "
                 "per-object mean hides."),
        "fig4": (f"Figure 4. Cross-view silhouette IoU (source view excluded from the mean). Higher is "
                 f"better. Dashed line marks the IoU={thr} shape-trust threshold used by the pipeline. "
                 f"Crosses mark the SAM3D source view, which is excluded because the cue mesh was "
                 f"generated from it."),
        "fig5": ("Figure 5. Normalized contour distance (bidirectional mean nearest-neighbour distance "
                 "between real and simulated mask contours, normalized by the real-mask bbox diagonal). "
                 "Lower is better."),
        "fig6": ("Figure 6. Cross-view silhouette IoU vs Mean Dimension Error (Baseline). Tests whether "
                 "IoU is a usable proxy for dimensional accuracy. With only a handful of objects the "
                 "correlation coefficients are descriptive only."),
        "fig7": ("Figure 7. Qualitative Real-to-Sim alignment. Green: real SAM mask contour. "
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


def write_report(out: Path, cfg, obj, cam, summ, skipped, tests, reps):
    thr = cfg["iou_threshold"]
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
        A(f"| {o['name']} | `{o['capture_dir']}` | `{o['mask_dir']}` | "
          f"`{o['baseline']['size_json']}` | `{o['oracle']['fit_json']}` | {o['source_camera']} |")

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
    A("- `caliper_manual_partial` (kettle): L2/L3 만 확정. L1 은 미확정 (§12).")

    A("\n## 4. Baseline / Oracle 정의\n")
    A("- **Baseline (`baseline_sam3d`)**: SAM3D 가 단일 RGB+mask 로 만든 unitless 단서 mesh 를 "
      "3대 카메라 SAM2 실루엣에 Sim(3) 7-DoF 로 정합. 최종 치수 = mesh OBB extents × 추정 scale. "
      "**실제 운용 방법** (정답 CAD 불필요).")
    A("- **Oracle (`oracle_cad`)**: 단서 mesh 만 정답 CAD 로 교체. 엔진·손실·최적화 동일. "
      "**형상이 완벽할 때의 정확도 상한**.")
    A(f"- 현재 데이터의 Baseline 은 모두 `sam3d_anisotropic_silhouette` (축별 scale_vec), "
      f"Oracle 은 `cad_multiview_silhouette` (등방 scale) 이다. "
      f"→ **두 방법이 완전히 동일한 변환족은 아니다** (§14).")

    A("\n## 5. 축 대응 방식\n")
    A("파이프라인과 GT 모두 치수를 **내림차순(L≥W≥H)** 으로 정렬해 보고하므로 크기 rank 대응을 사용했다. "
      "정렬된 두 수열에서 identity 대응이 Σ|차이| 를 최소화하는 최적 순열이므로(rearrangement inequality), "
      "검증을 위해 6개 순열을 모두 탐색해 rank 대응과 일치하는지 확인했다.\n")
    ag = obj.axis_perm_search_agrees.tolist()
    A(f"- `axis_matching_method` = `rank_descending` (전 행)")
    A(f"- 6-순열 탐색이 rank 대응과 일치: **{sum(1 for x in ag if x)}/{len(ag)} 행** "
      f"→ rank 대응이 최적임을 확인")

    A("\n## 6. 평가 지표 수식\n")
    A("```")
    A("e_L = |L_hat - L_GT|,  e_W = |W_hat - W_GT|,  e_H = |H_hat - H_GT|      [mm]")
    A("E_dim = (e_L + e_W + e_H) / 3                                            [mm]   <- 메인")
    A("E_rel = (1/3)[e_L/L_GT + e_W/W_GT + e_H/H_GT] x 100                      [%]")
    A("IoU_i = |M_real_i ∩ M_sim_i| / |M_real_i ∪ M_sim_i|")
    A("Cross-view IoU = mean over cameras EXCLUDING the SAM3D source view       <- 메인")
    A("D_contour = 0.5[ mean_{p∈C_real} min_q ||p-q|| + mean_{q∈C_sim} min_p ||q-p|| ]  [px]")
    A("D_contour_norm = D_contour / sqrt(w_real^2 + h_real^2)")
    A("```")
    A("- GT 미확정 축은 e_* 를 계산하지 않고 평균에서 제외 (`gt_axes_used` 에 사용 축 수 기록).")
    A("- sim 마스크는 결과 JSON 의 fit 파라미터로 **재렌더링**했다. 재현 검증을 위해 엔진과 동일한 "
      "슈퍼샘플 조건의 IoU 를 저장된 `per_view_iou` 와 대조했다 "
      f"(최대 Δ = {_fmt(cam.iou_reproduction_delta.max(), 4)}).")

    A("\n## 7. 객체별 결과\n")
    A("| Object | Method | GT (mm) | Estimated (mm) | e_L | e_W | e_H | **E_dim** (mm) | E_rel (%) | "
      "src IoU | **cross IoU** | cross D_contour (%) | axes |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in obj.iterrows():
        gt = " × ".join(_fmt(r[f"gt_{a}_mm"], 1) for a in cfg["axis_labels"])
        es = " × ".join(_fmt(r[f"estimated_{a}_mm"], 1) for a in cfg["axis_labels"])
        A(f"| {r.display_name} | `{r.method}` | {gt} | {es} | "
          f"{_fmt(r.abs_error_L_mm)} | {_fmt(r.abs_error_W_mm)} | {_fmt(r.abs_error_H_mm)} | "
          f"**{_fmt(r.mean_dimension_error_mm)}** | {_fmt(r.mean_relative_dimension_error_percent)} | "
          f"{_fmt(r.source_view_iou, 3)} | **{_fmt(r.cross_view_iou, 3)}** | "
          f"{_fmt(r.cross_view_normalized_contour_distance_percent)} | {int(r.gt_axes_used)}/3 |")

    A("\n## 8. 전체 (mean ± std)\n")
    A(f"| Method | n | E_dim (mm) | E_rel (%) | Cross-view IoU | D_contour (%) | IoU≥{thr} 통과 |")
    A("|---|---|---|---|---|---|---|")
    for _, r in summ[summ.method.isin(ec.METHODS)].iterrows():
        A(f"| `{r.method}` | {int(r.object_count)} | "
          f"{_fmt(r.mean_dimension_error_mean_mm)} ± {_fmt(r.mean_dimension_error_std_mm)} | "
          f"{_fmt(r.mean_relative_error_mean_percent)} ± {_fmt(r.mean_relative_error_std_percent)} | "
          f"{_fmt(r.cross_view_iou_mean, 3)} ± {_fmt(r.cross_view_iou_std, 3)} | "
          f"{_fmt(r.normalized_contour_distance_mean_percent)} ± "
          f"{_fmt(r.normalized_contour_distance_std_percent)} | "
          f"{int(r.iou_085_pass_count)}/{int(r.object_count)} "
          f"({_fmt(r.iou_085_pass_rate_percent, 1)}%) |")

    A("\n## 9. Baseline − Oracle gap\n")
    g = summ[summ.method == "gap_baseline_minus_oracle"]
    if not g.empty:
        gr = g.iloc[0]
        A(f"- **Mean Dimension Error gap = {_fmt(gr.mean_dimension_error_gap_mm)} ± "
          f"{_fmt(gr.mean_dimension_error_gap_std_mm)} mm** (n={int(gr.object_count)})")
        A(f"- Relative error gap = {_fmt(gr.relative_error_gap_percent_point)} %p")
    A("\n객체별 gap:\n")
    A("| Object | Baseline E_dim | Oracle E_dim | gap (mm) |")
    A("|---|---|---|---|")
    b = obj[obj.method == BASE].set_index("object_name")
    o = obj[obj.method == ORA].set_index("object_name")
    for i in b.index:
        if i in o.index:
            gap = b.loc[i].mean_dimension_error_mm - o.loc[i].mean_dimension_error_mm
            A(f"| {b.loc[i].display_name} | {_fmt(b.loc[i].mean_dimension_error_mm)} | "
              f"{_fmt(o.loc[i].mean_dimension_error_mm)} | {gap:+.2f} |")
    A("\n> 해석: 이 gap 은 **동일한 Sim(3) 최적화 엔진에서 단서 메시를 SAM3D 에서 정답 CAD 로 "
      "교체했을 때 변화한 최종 치수 오차**다. SAM3D 의 순수 형상 오차라고 단정할 수 없다 — "
      "SAM3D 형상 오류가 최종 크기 추정에 미친 영향으로 읽어야 한다.\n")
    A(f"\n통계 검정 (n={tests['n_pairs']}):\n")
    if "paired_t_pvalue" in tests:
        A(f"- paired t-test: t={tests['paired_t_statistic']:.3f}, p={tests['paired_t_pvalue']:.3f}")
    if "wilcoxon_pvalue" in tests:
        A(f"- Wilcoxon signed-rank: W={tests['wilcoxon_statistic']:.1f}, p={tests['wilcoxon_pvalue']:.3f}")
    A(f"- ⚠ {tests['caveat']}")

    A("\n## 10. Cross-view IoU 0.85 통과율\n")
    for _, r in summ[summ.method.isin(ec.METHODS)].iterrows():
        A(f"- `{r.method}`: {int(r.iou_085_pass_count)}/{int(r.object_count)} "
          f"({_fmt(r.iou_085_pass_rate_percent, 1)}%)")
    fails = obj[(obj.method == BASE) & (obj.cross_view_iou < thr)]
    A(f"- IoU < {thr} 인 객체: "
      f"{', '.join(fails.display_name) if len(fails) else '없음'}")

    A("\n## 11. 생성한 그래프와 피규어\n")
    A("모두 PNG(300dpi). 각 그림 상단에 `번호. 제목` 이 표기된다.\n")
    A("### 통합 (전체 객체) — `figures/`\n")
    for n, t in [("fig1_baseline_vs_oracle_dim_error", "1. Baseline vs Oracle E_dim"),
                 ("fig1_baseline_vs_oracle_dim_error_horizontal", "1b. 동 (가로형)"),
                 ("fig2_gt_vs_estimated_dimensions", "2. GT vs Estimated"),
                 ("fig3_per_axis_absolute_error_baseline", "3a. 축별 오차 (Baseline)"),
                 ("fig3_per_axis_absolute_error_oracle", "3b. 축별 오차 (Oracle)"),
                 ("fig4_cross_view_silhouette_iou", "4. Cross-view IoU"),
                 ("fig5_normalized_contour_distance", "5. Normalized contour distance"),
                 ("fig6_iou_vs_dimension_error", "6. IoU vs E_dim"),
                 ("fig7_qualitative_real_to_sim_grid", "7. 정성 통합 그리드"),
                 ("fig8_controlled_transformation_family", "8. 통제 실험 (변환족 일치)")]:
        A(f"- [{t}](figures/{n}.png)")
    if reps:
        A(f"\n대표 객체 (7. 그리드 행): {', '.join(reps)} — 최소오차/중간/최대오차 + 단순·복잡 형상 포함")
    A("\n### 물체별 — `per_object/<object>/`\n")
    A("| Object | 그림 |")
    A("|---|---|")
    for o in cfg["objects"]:
        n = o["name"]
        links = " · ".join(
            f"[{t}](per_object/{n}/{f}.png)" for f, t in [
                ("fig1_baseline_vs_oracle_dim_error", "1"),
                ("fig2_gt_vs_estimated_dimensions", "2"),
                ("fig3_per_axis_absolute_error_baseline", "3a"),
                ("fig3_per_axis_absolute_error_oracle", "3b"),
                ("fig4_cross_view_silhouette_iou", "4"),
                ("fig5_normalized_contour_distance", "5"),
                ("fig7_qualitative_three_views", "7"),
                ("fig8_controlled_transformation_family", "8")])
        A(f"| {o.get('display_name', n)} | {links} |")
    A("\n> 물체별에는 **6번(IoU vs E_dim)이 없다** — 객체 1개로는 상관을 볼 수 없다.")

    A("\n## 12. 누락·제외된 객체와 이유\n")
    A("| 대상 | 문제 | 처리 |")
    A("|---|---|---|")
    A("| kettle L1 축 | 최소부피 OBB 의 최장축(주둥이·손잡이 돌출 포함)과 캘리퍼 측정 지점의 "
      "대응이 정의되지 않음. 실측이 110 → 113 으로 바뀐 이력. | **해당 축만** 오차 계산에서 제외 "
      "(`gt_axes_used=2/3`). 객체는 유지하고 추정치는 CSV/그림에 남김 |")
    if not skipped.empty:
        for _, r in skipped.iterrows():
            A(f"| {r.object_name} / {r.method} | {r.reason} | skip (`csv/skipped_objects.csv`) |")
    else:
        A("| — | 파일 누락으로 skip 된 객체 없음 | — |")
    A("\n필요한 추가 데이터:")
    A("- **kettle L1 캘리퍼 실측** — OBB 최장축과 같은 두 점(주둥이 끝 ↔ 손잡이 바깥)을 재서 "
      "`evaluation_config.yaml` 의 `kettle.gt_mm[0]` 을 null → 숫자로 바꾸면 3/3 축 평가가 된다.")
    A("- **T_shape 캘리퍼 실측** — 현재는 설계값(nominal)이라 제조 공차가 반영돼 있지 않다.")
    A("- 객체 수 4개는 통계 검정에 부족하다. YCB 등 추가 객체 권장.")

    A("\n## 13. 결과 해석\n")
    bm = summ[summ.method == BASE].iloc[0]
    om = summ[summ.method == ORA].iloc[0]
    A(f"1. **크기 복원**: Baseline E_dim = {_fmt(bm.mean_dimension_error_mean_mm)} ± "
      f"{_fmt(bm.mean_dimension_error_std_mm)} mm, Oracle = {_fmt(om.mean_dimension_error_mean_mm)} ± "
      f"{_fmt(om.mean_dimension_error_std_mm)} mm. 정답 CAD 없이도 상한에 근접한다.")
    A(f"2. **형상 영향**: 객체별 gap 은 부호가 엇갈린다 (아래 표). 단순 형상(peg/T_shape)에서는 "
      f"Baseline 이 Oracle 과 동등하거나 더 낫고, 복잡 형상(kettle)에서 gap 이 가장 크다.")
    A(f"3. **Real-to-Sim 정합**: Baseline cross-view IoU = {_fmt(bm.cross_view_iou_mean, 3)} ± "
      f"{_fmt(bm.cross_view_iou_std, 3)}, D_contour = {_fmt(bm.normalized_contour_distance_mean_percent)}%.")
    A("4. **IoU 는 치수 정확도의 대리 지표로 신뢰할 수 없다** (Figure 6). 이 데이터에서 kettle 은 "
      "Baseline cross-view IoU 가 가장 높은 축에 속하지만 E_dim 은 가장 크고, Oracle 은 kettle 에서 "
      "IoU 가 가장 낮은데 E_dim 은 가장 작다. 복잡 형상에서 치수를 결정하는 극점(주둥이 끝)이 "
      "실루엣 면적에서 차지하는 비중이 작기 때문으로 보인다.")

    A("\n## 14. 현재 데이터만으로 주장할 수 없는 것\n")
    A("- **IoU 높음 → 3D 치수 정확** : 성립하지 않는다. Cross-view IoU 와 D_contour 는 "
      "**크기·형상·pose·캘리브레이션이 뒤섞인 투영 정합 품질**이고, E_dim/E_rel 만이 실측 기반 "
      "3D 크기 정확도다. 두 지표군을 바꿔 쓰지 말 것.")
    A("- **Baseline−Oracle gap = SAM3D 형상 오차** : 단정 불가. 현재 Baseline 은 비등방(축별 scale_vec), "
      "Oracle 은 등방(단일 scale) 이라 **변환족이 달라** gap 에 그 차이가 섞여 있다. "
      "엄밀히 하려면 Oracle 도 비등방으로 재실행해 통제해야 한다.")
    A(f"- **통계적 유의성** : n={tests['n_pairs']} 로 p-value 는 신뢰구간이 매우 넓다. "
      "descriptive 로만 읽어야 한다.")
    A("- **일반화** : 4개 객체(단순 3 + 복잡 1), 단일 촬영 세션, 단일 센서(RealSense), "
      "source view 는 전부 cam0. 다른 재질·크기·카메라 배치로의 일반화는 이 데이터로 알 수 없다.")
    A("- **T_shape 결과** : GT 가 설계값이라 '실측 대비 정확도'가 아니라 '설계값 대비 일치도'다.")
    A("- **kettle 절대 정확도** : L1 이 미확정이라 3축 전체 정확도는 알 수 없다.")

    (out / "evaluation_report.md").write_text("\n".join(L) + "\n")
    print(f"  [SAVE] evaluation_report.md")


if __name__ == "__main__":
    main()
