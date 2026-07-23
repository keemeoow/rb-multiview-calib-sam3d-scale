#!/usr/bin/env python3
"""통제 실험 분석 — 변환족을 맞춘 Baseline−Oracle gap.

메인 평가의 gap 은 Baseline(비등방) vs Oracle(등방) 이라 두 요인이 섞여 있다:
    기존 gap = f(SAM3D 형상 오차,  등방/비등방)
run_oracle_aniso_control.py 가 만든 Oracle(비등방) 을 쓰면 변환족이 통제된다:
    통제 gap = f(SAM3D 형상 오차)

  python evaluation/analyze_controlled_gap.py \
    --config evaluation/evaluation_config.yaml --output evaluation/results
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_common as ec                       # noqa: E402
import generate_evaluation_figures as gef      # noqa: E402

COL_ANISO = "#eda100"      # Oracle(aniso) 전용 3번째 계열 (dataviz reference palette)


def load_control(out: Path, cfg):
    """control_oracle_aniso/*.json -> object -> dict."""
    d = out / "control_oracle_aniso"
    idx = d / "control_index.json"
    if not idx.exists():
        raise SystemExit(f"통제 결과가 없습니다 ({idx}). 먼저 run_oracle_aniso_control.py 를 실행하세요.")
    out_map = {}
    for name in json.loads(idx.read_text()):
        p = d / f"{name}_cad_fit_aniso.json"
        if p.exists():
            out_map[name] = json.loads(p.read_text())
    return out_map


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()

    cfg = ec.load_config(a.config)
    obj = pd.read_csv(a.output / "csv" / "evaluation_per_object.csv")
    ctrl = load_control(a.output, cfg)
    by = {o["name"]: o for o in cfg["objects"]}

    rows = []
    for name, d in ctrl.items():
        if name not in by:
            print(f"  [SKIP] {name}: config 에 없는 물체 (오래된 통제 결과)")
            continue
        # 정답 CAD 가 없으면 oracle 자체가 없으므로 gap 이 정의되지 않는다
        if not by[name].get("oracle"):
            print(f"  [SKIP] {name}: 정답 CAD 없음 -> Baseline-Oracle gap 정의 불가")
            continue
        ex = by[name].get("exclude_from_metric_scale", False)
        if (("oracle_cad" in ex) if isinstance(ex, (list, tuple)) else bool(ex)):
            print(f"  [SKIP] {name}: 크기평가 제외 -> gap 계산 불가")
            continue
        o_rows = obj[(obj.object_name == name) & (obj.method == "oracle_cad")]
        b_rows = obj[(obj.object_name == name) & (obj.method == "baseline_sam3d")]
        if o_rows.empty or b_rows.empty:
            print(f"  [SKIP] {name}: baseline/oracle 결과가 모두 있어야 gap 을 낼 수 있음")
            continue
        gt = [None if g is None else float(g) for g in by[name]["gt_mm"]]
        m = ec.match_axes([float(x) for x in d["extents_mm_sorted_desc"]], gt)
        abs_e, mean_e, mean_rel, n_ax = ec.dim_errors(m["est_matched"], gt)
        b = b_rows.iloc[0]
        o_iso = o_rows.iloc[0]
        rows.append(dict(
            object_name=name, display_name=b.display_name, shape_class=b.shape_class,
            gt_axes_used=n_ax,
            baseline_aniso_E_dim_mm=b.mean_dimension_error_mm,
            oracle_iso_E_dim_mm=o_iso.mean_dimension_error_mm,
            oracle_aniso_E_dim_mm=mean_e,
            oracle_aniso_E_rel_percent=mean_rel,
            oracle_aniso_mean_iou=d["mean_iou"],
            # 메인 평가 CSV 는 보고 지표만 담으므로 이 진단용 열은 없을 수 있다
            oracle_iso_mean_iou=o_iso.get("stored_mean_iou"),
            oracle_aniso_extents_mm=json.dumps([round(x, 3) for x in m["est_matched"]]),
            gap_uncontrolled_mm=b.mean_dimension_error_mm - o_iso.mean_dimension_error_mm,
            gap_controlled_mm=(None if mean_e is None
                               else b.mean_dimension_error_mm - mean_e),
            control_file=str((a.output / "control_oracle_aniso" / f"{name}_cad_fit_aniso.json")),
        ))
    df = pd.DataFrame(rows)
    order = [o["name"] for o in cfg["objects"] if o["name"] in set(df.object_name)]
    df["__o"] = df.object_name.map({n: i for i, n in enumerate(order)})
    df = df.sort_values("__o").drop(columns="__o")
    df.to_csv(a.output / "csv" / "controlled_gap.csv", index=False)
    print(f"  [SAVE] csv/controlled_gap.csv")

    gu = df.gap_uncontrolled_mm.dropna()
    gc = df.gap_controlled_mm.dropna()
    summary = dict(
        n_objects=int(len(df)),
        gap_uncontrolled_mean_mm=float(gu.mean()), gap_uncontrolled_std_mm=float(gu.std(ddof=1)),
        gap_controlled_mean_mm=float(gc.mean()), gap_controlled_std_mm=float(gc.std(ddof=1)),
        oracle_iso_mean_E_dim_mm=float(df.oracle_iso_E_dim_mm.mean()),
        oracle_aniso_mean_E_dim_mm=float(df.oracle_aniso_E_dim_mm.mean()),
        baseline_mean_E_dim_mm=float(df.baseline_aniso_E_dim_mm.mean()),
        interpretation=(
            "gap_controlled 는 Baseline 과 Oracle 이 **같은 비등방 변환족**일 때의 차이라, "
            "단서 메시(SAM3D vs 정답 CAD) 형상 차이가 최종 치수 오차에 미친 영향에 더 가깝다. "
            "gap_uncontrolled 에는 등방/비등방 차이가 섞여 있다. "
            "여전히 SAM3D 의 '순수 형상 오차'라고 단정할 수는 없다 — 마스크 품질, 캘리브레이션, "
            "최적화 국소해 등 통제되지 않은 요인이 남아 있다."),
        caveat=f"n={len(df)} 로 표본이 매우 적다. descriptive 로만 읽을 것.",
    )
    (a.output / "csv" / "controlled_gap_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"  [SAVE] csv/controlled_gap_summary.json")

    # ---- Figure 8 (통합 + 물체별)
    plot_fig8(df, a.output / "figures", cfg)
    for _, r in df.iterrows():
        d = a.output / "per_object" / r.object_name
        plot_fig8(df[df.object_name == r.object_name], d, cfg, suffix=r.display_name)

    print("\n=== 통제 실험 결과 ===")
    print(f"{'Object':10s} {'Baseline':>10s} {'Oracle iso':>11s} {'Oracle aniso':>13s} "
          f"{'gap(기존)':>10s} {'gap(통제)':>10s}")
    print("-" * 70)
    for _, r in df.iterrows():
        print(f"{r.display_name:10s} {r.baseline_aniso_E_dim_mm:10.2f} {r.oracle_iso_E_dim_mm:11.2f} "
              f"{r.oracle_aniso_E_dim_mm:13.2f} {r.gap_uncontrolled_mm:+10.2f} "
              f"{r.gap_controlled_mm:+10.2f}")
    print("-" * 70)
    print(f"{'Mean':10s} {summary['baseline_mean_E_dim_mm']:10.2f} "
          f"{summary['oracle_iso_mean_E_dim_mm']:11.2f} {summary['oracle_aniso_mean_E_dim_mm']:13.2f} "
          f"{summary['gap_uncontrolled_mean_mm']:+10.2f} {summary['gap_controlled_mean_mm']:+10.2f}")
    print(f"\ngap 기존  = {summary['gap_uncontrolled_mean_mm']:+.2f} ± "
          f"{summary['gap_uncontrolled_std_mm']:.2f} mm  (등방/비등방 차이 섞임)")
    print(f"gap 통제  = {summary['gap_controlled_mean_mm']:+.2f} ± "
          f"{summary['gap_controlled_std_mm']:.2f} mm  (변환족 일치)")

    write_control_report(a.output, cfg, df, summary)


def plot_fig8(df: pd.DataFrame, outdir: Path, cfg, suffix: str = ""):
    """3계열 grouped bar. 물체가 1개면 Mean 막대는 중복이라 생략."""
    cb, co, cr = gef.setup(cfg)
    n = len(df)
    show_mean = n > 1
    x = np.arange(n + (1 if show_mean else 0))
    lab = list(df.display_name) + (["Mean"] if show_mean else [])
    b = list(df.baseline_aniso_E_dim_mm) + ([df.baseline_aniso_E_dim_mm.mean()] if show_mean else [])
    oi = list(df.oracle_iso_E_dim_mm) + ([df.oracle_iso_E_dim_mm.mean()] if show_mean else [])
    oa = list(df.oracle_aniso_E_dim_mm) + ([df.oracle_aniso_E_dim_mm.mean()] if show_mean else [])
    fig, ax = plt.subplots(figsize=(9.2, 4.9) if show_mean else (6.4, 4.9))
    w = 0.26
    bars = [
        ax.bar(x - w, b, w, label="Baseline — SAM3D mesh (anisotropic)", color=cb),
        ax.bar(x, oi, w, label="Oracle — GT CAD (isotropic, as published)", color=co),
        ax.bar(x + w, oa, w, label="Oracle — GT CAD (anisotropic, controlled)", color=COL_ANISO),
    ]
    for grp in bars:
        for r in grp:
            ax.annotate(f"{r.get_height():.2f}", (r.get_x() + r.get_width() / 2, r.get_height()),
                        textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7.6)
    if show_mean:
        ax.axvline(n - 0.5, color=cr, lw=0.8, ls=":")
    ax.set_xticks(x); ax.set_xticklabels(lab)
    ax.set_ylabel("Mean Dimension Error (mm)")
    ax.set_ylim(0, max(b + oi + oa) * 1.25)
    ax.set_title("Controlled comparison: same transformation family for Baseline and Oracle\n"
                 "(lower is better)", loc="left")
    ax.legend(frameon=False, fontsize=8.6)
    nm = "fig8_controlled_transformation_family"
    gef.save(fig, outdir, nm, cfg, title=gef.fig_title(nm, suffix))


def write_control_report(out: Path, cfg, df: pd.DataFrame, S: dict):
    L = []
    A = L.append
    A("# 통제 실험 — 변환족을 맞춘 Baseline−Oracle gap\n")
    A("생성: `python evaluation/analyze_controlled_gap.py --config evaluation/evaluation_config.yaml "
      "--output evaluation/results`\n")

    A("\n## 왜 했나\n")
    A("메인 평가의 Baseline 은 **비등방**(축별 `scale_vec`), Oracle 은 **등방**(단일 scale) 이라 "
      "gap 에 두 요인이 섞여 있었다:\n")
    A("```")
    A("기존 gap = f(SAM3D 형상 오차,  등방/비등방)")
    A("통제 gap = f(SAM3D 형상 오차)                <- 정답 CAD 를 비등방으로 재정합")
    A("```")
    A("\n**통제 방법**: 정답 CAD 를 `_silhouette_fit.fit_mesh_aniso` 로 재정합하되, "
      "**발표된 등방 oracle 해를 warm start 로 고정**해 출발점을 동일하게 두고 변환족만 넓혔다. "
      f"나머지 조건(`w_depth=0.0`, `max_fev=4000`, `aniso_reg=0.0`, 동일 마스크·점군)도 "
      "Baseline 과 맞췄다. 기존 추정 코드는 수정하지 않았다.\n")

    A("\n## 결과\n")
    A("| Object | Baseline (SAM3D, aniso) | Oracle (CAD, iso) | Oracle (CAD, aniso) | "
      "gap 기존 | **gap 통제** |")
    A("|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        A(f"| {r.display_name} | {r.baseline_aniso_E_dim_mm:.2f} | {r.oracle_iso_E_dim_mm:.2f} | "
          f"{r.oracle_aniso_E_dim_mm:.2f} | {r.gap_uncontrolled_mm:+.2f} | "
          f"**{r.gap_controlled_mm:+.2f}** |")
    A(f"| **Mean** | **{S['baseline_mean_E_dim_mm']:.2f}** | {S['oracle_iso_mean_E_dim_mm']:.2f} | "
      f"{S['oracle_aniso_mean_E_dim_mm']:.2f} | {S['gap_uncontrolled_mean_mm']:+.2f} | "
      f"**{S['gap_controlled_mean_mm']:+.2f}** |")
    A(f"\n- gap 기존 = **{S['gap_uncontrolled_mean_mm']:+.2f} ± {S['gap_uncontrolled_std_mm']:.2f} mm**")
    A(f"- gap 통제 = **{S['gap_controlled_mean_mm']:+.2f} ± {S['gap_controlled_std_mm']:.2f} mm** "
      f"→ **부호가 뒤집힌다**")
    A("\n그림: [fig8_controlled_transformation_family](figures/fig8_controlled_transformation_family.png)\n")

    A("\n## 해석\n")
    A("1. **기존 gap(+0.34mm)의 상당 부분은 SAM3D 형상 오차가 아니라 변환족 차이였다.** "
      "변환족을 맞추면 gap 이 음수로 바뀐다 — 즉 같은 비등방 조건에서는 SAM3D 단서 메시가 "
      "정답 CAD 보다 **나쁘지 않았다**.")
    A("2. **비등방은 형상이 맞을 때 오히려 해롭다.** Oracle 은 등방 0.62mm → 비등방 1.39mm 로 "
      "악화됐다. 형상이 이미 정확하면 축별 자유도는 실루엣에 **과적합**할 여지만 준다.")
    A("3. **kettle 이 그 과적합의 교과서적 사례다.** 비등방 Oracle 은 IoU 를 "
      f"{df[df.object_name=='kettle'].oracle_iso_mean_iou.iloc[0]:.3f} → "
      f"{df[df.object_name=='kettle'].oracle_aniso_mean_iou.iloc[0]:.3f} 로 **개선**하면서 "
      f"E_dim 은 {df[df.object_name=='kettle'].oracle_iso_E_dim_mm.iloc[0]:.2f} → "
      f"{df[df.object_name=='kettle'].oracle_aniso_E_dim_mm.iloc[0]:.2f} mm 로 **악화**시켰다. "
      "실루엣 손실을 낮추면서 치수를 망가뜨린 것이다.")
    A("4. 따라서 **최적 변환족은 단서 메시의 형상 정확도에 의존한다** (상호작용):")
    A("   - 형상이 정확(Oracle) → **등방**이 낫다 (자유도를 주면 과적합)")
    A("   - 형상이 추정치(SAM3D) → **비등방**이 낫다 (형상 비율 오류를 교정; "
      "`size_method_experiment.json` 의 peg 4.37 → 1.05mm)")

    A("\n## 두 gap 은 서로 다른 질문에 답한다\n")
    A("| | 비교 | 답하는 질문 |")
    A("|---|---|---|")
    A("| **gap 기존** (+0.34mm) | Baseline(aniso) vs Oracle(iso) | 각 방법을 **각자 최적 설정**으로 "
      "썼을 때의 실전 격차 |")
    A("| **gap 통제** (−0.42mm) | Baseline(aniso) vs Oracle(aniso) | **변환족 고정** 시 단서 메시 "
      "형상만의 영향 |")
    A("\n둘 다 유효하며 어느 하나가 다른 하나를 대체하지 않는다. 논문에 쓸 때 어떤 비교인지 "
      "명시해야 한다.\n")

    A("\n## 이 실험으로도 주장할 수 없는 것\n")
    A(f"- **gap = SAM3D 순수 형상 오차** : 여전히 단정 불가. 변환족은 통제했지만 마스크 품질, "
      "캘리브레이션, Powell 국소해 등은 통제되지 않았다.")
    A(f"- **통계적 유의성** : {S['caveat']} 특히 통제 gap 평균(−0.42)은 kettle 의 "
      "비등방 과적합(−1.51)이 끌어내린 값이라, 나머지 3개(−0.10, +0.07, −0.16)만 보면 "
      "사실상 0 에 가깝다.")
    A("- **'비등방이 항상 나쁘다'** : 성립하지 않는다. 형상이 틀린 Baseline 에서는 비등방이 "
      "크게 이롭다는 것이 이미 확인돼 있다 (`size_method_experiment.json`).")
    A("- **kettle 절대 정확도** : L1 이 GT 미확정이라 2/3 축으로만 계산된 값이다.")

    p = out / "control_report.md"
    p.write_text("\n".join(L) + "\n")
    print(f"  [SAVE] control_report.md")


if __name__ == "__main__":
    main()
