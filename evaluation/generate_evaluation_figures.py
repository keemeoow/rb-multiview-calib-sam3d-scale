#!/usr/bin/env python3
"""정량 그래프 6종 생성 (Figure 1~6).

축 라벨/범례는 영어, 배경 흰색, PNG 300dpi + 벡터(SVG/PDF).
Baseline/Oracle 색상은 config 에 고정돼 모든 그림에서 동일하다.
GT 미확정 축(kettle L1)은 값이 없으므로 해당 셀/점을 그리지 않고 'n/d' 로 표기한다.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import eval_common as ec

BASE, ORA = "baseline_sam3d", "oracle_cad"

# 각 피규어 상단에 찍는 제목 (파일명 -> "번호. 타이틀명").
# save() 가 이 표를 보고 suptitle 을 달아, 그림만 떼어놔도 어느 번호인지 알 수 있게 한다.
# 'Figure' 라는 단어는 넣지 않는다 (번호.제목 형식).
FIG_TITLES = {
    "fig1_baseline_vs_oracle_dim_error":
        "1. Baseline vs Oracle Mean Dimension Error",
    "fig1_baseline_vs_oracle_dim_error_horizontal":
        "1b. Baseline vs Oracle Mean Dimension Error (horizontal)",
    "fig2_gt_vs_estimated_dimensions":
        "2. GT Dimension vs Estimated Dimension",
    "fig3_per_axis_absolute_error_baseline":
        "3a. Per-axis Absolute Error — Baseline",
    "fig3_per_axis_absolute_error_oracle":
        "3b. Per-axis Absolute Error — Oracle",
    "fig4_cross_view_silhouette_iou":
        "4. Cross-view Silhouette IoU",
    "fig5_normalized_contour_distance":
        "5. Normalized Contour Distance",
    "fig6_iou_vs_dimension_error":
        "6. Cross-view IoU vs Dimension Error",
    "fig7_qualitative_real_to_sim_grid":
        "7. Qualitative Real-to-Sim Grid",
    "fig8_controlled_transformation_family":
        "8. Controlled Transformation Family (Baseline vs Oracle)",
}


def fig_title(name: str, suffix: str = "") -> str:
    """'번호. 제목' (+ 물체별 그림이면 ' — <물체>')."""
    t = FIG_TITLES.get(name, name)
    return f"{t} — {suffix}" if suffix else t


def setup(cfg):
    s = cfg["style"]
    plt.rcParams.update({
        "figure.facecolor": s["facecolor"], "savefig.facecolor": s["facecolor"],
        "axes.facecolor": s["facecolor"], "font.size": s["font_size"],
        "axes.titlesize": s["font_size"] + 1.5, "axes.labelsize": s["font_size"],
        "xtick.labelsize": s["font_size"] - 1, "ytick.labelsize": s["font_size"] - 1,
        "legend.fontsize": s["font_size"] - 1, "axes.grid": False,
        "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight",
    })
    return s["color_baseline"], s["color_oracle"], s["color_ref"]


def save(fig, outdir: Path, name: str, cfg, title: str | None = None):
    """저장 + 상단에 'Figure N. 타이틀명' 표기.

    축 제목(설명문)은 그대로 두고 그 위에 피규어 이름을 얹는다. bbox_inches='tight' 라
    suptitle 이 잘리지 않는다.
    """
    t = title if title is not None else FIG_TITLES.get(name)
    if t:
        fig.suptitle(t, fontsize=cfg["style"]["font_size"] + 2.5, fontweight="bold",
                     x=0.0, y=1.0, ha="left", va="bottom", color="#0b0b0b")
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in cfg["style"]["formats"]:
        fig.savefig(outdir / f"{name}.{ext}", dpi=cfg["style"]["dpi"])
    plt.close(fig)
    print(f"  [SAVE] {name}.{{{','.join(cfg['style']['formats'])}}}")


def _disp(df):
    return dict(zip(df.object_name, df.display_name))


def metric_only(df: pd.DataFrame) -> pd.DataFrame:
    """크기(metric scale) 지표를 쓸 수 있는 물체만.

    GT 나 정답 CAD 가 신뢰 불가해 제외된 물체(excluded_from_metric_scale)는 E_dim 이 없으므로
    Figure 1/2/3/6/8 에서 뺀다. 실루엣 지표(Figure 4/5)는 GT 가 필요 없어 그대로 쓴다.
    """
    if "excluded_from_metric_scale" not in df.columns:
        return df
    return df[~df.excluded_from_metric_scale.fillna(False).astype(bool)]


# ---------------------------------------------------------------- Figure 1
def fig1(obj: pd.DataFrame, outdir: Path, cfg, suffix="", variants=True):
    cb, co, cr = setup(cfg)
    obj = metric_only(obj)
    if obj.empty:
        print(f"  [SKIP] fig1 {suffix}: 크기평가 가능한 물체 없음")
        return
    names = list(dict.fromkeys(obj.object_name))
    d = _disp(obj)

    def get(n, meth):
        """물체×방법이 없을 수 있다 (예: kettle 은 oracle 만 크기평가 제외) -> NaN."""
        r = obj[(obj.object_name == n) & (obj.method == meth)]
        return float(r.mean_dimension_error_mm.iloc[0]) if len(r) else np.nan

    b = [get(n, BASE) for n in names]
    o = [get(n, ORA) for n in names]
    bm = np.nanmean(b) if np.isfinite(b).any() else np.nan
    om = np.nanmean(o) if np.isfinite(o).any() else np.nan
    show_mean = len(names) > 1        # 물체가 하나면 Mean 막대는 같은 값의 중복이라 생략

    for horiz in ((False, True) if variants else (False,)):
        x = np.arange(len(names) + (1 if show_mean else 0))
        bb, oo = (b + [bm], o + [om]) if show_mean else (list(b), list(o))
        lab = [d[n] for n in names] + (["Mean"] if show_mean else [])
        wf = max(4.6, 1.2 * len(x) + 2.4)          # 막대 수에 비례한 폭
        fig, ax = plt.subplots(figsize=(wf, 4.6) if not horiz
                               else (7.2, max(3.2, 0.85 * len(x) + 2.2)))
        w = 0.38
        if not horiz:
            r1 = ax.bar(x - w / 2, bb, w, label="Baseline (SAM3D)", color=cb)
            r2 = ax.bar(x + w / 2, oo, w, label="Oracle (GT CAD)", color=co)
            ax.set_xticks(x); ax.set_xticklabels(lab)
            ax.set_ylabel("Mean Dimension Error (mm)")
            for r in list(r1) + list(r2):
                if not np.isfinite(r.get_height()):
                    continue                     # 크기평가 제외 조합은 막대·라벨 없음
                ax.annotate(f"{r.get_height():.2f}", (r.get_x() + r.get_width() / 2, r.get_height()),
                            textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
            if show_mean:
                ax.axvline(len(names) - 0.5, color=cr, lw=0.8, ls=":")
            ax.set_ylim(0, np.nanmax(bb + oo) * 1.22)
        else:
            r1 = ax.barh(x - w / 2, bb, w, label="Baseline (SAM3D)", color=cb)
            r2 = ax.barh(x + w / 2, oo, w, label="Oracle (GT CAD)", color=co)
            ax.set_yticks(x); ax.set_yticklabels(lab); ax.invert_yaxis()
            ax.set_xlabel("Mean Dimension Error (mm)")
            for r in list(r1) + list(r2):
                if not np.isfinite(r.get_width()):
                    continue
                ax.annotate(f"{r.get_width():.2f}", (r.get_width(), r.get_y() + r.get_height() / 2),
                            textcoords="offset points", xytext=(3, 0), va="center", fontsize=8)
            ax.set_xlim(0, np.nanmax(bb + oo) * 1.18)
        ax.set_title("Mean Dimension Error: Baseline vs Oracle  (lower is better)", loc="left")
        ax.legend(frameon=False)
        nm = "fig1_baseline_vs_oracle_dim_error" + ("_horizontal" if horiz else "")
        save(fig, outdir, nm, cfg, title=fig_title(nm, suffix))


# ---------------------------------------------------------------- Figure 2
def fig2(obj: pd.DataFrame, outdir: Path, cfg, axis_labels, suffix="", csv_out: Path | None = None):
    cb, co, cr = setup(cfg)
    obj = metric_only(obj)
    rec = []
    for _, r in obj.iterrows():
        for i, ax_l in enumerate(axis_labels):
            g, e = r[f"gt_{ax_l}_mm"], r[f"estimated_{ax_l}_mm"]
            if g is None or (isinstance(g, float) and np.isnan(g)):
                continue                      # GT 미확정 축은 점을 찍을 수 없다
            rec.append(dict(object_name=r.object_name, axis=ax_l, method=r.method,
                            gt_mm=float(g), est_mm=float(e)))
    p = pd.DataFrame(rec)
    if p.empty:
        print(f"  [SKIP] fig2{' ' + suffix if suffix else ''}: 확정 GT 축이 없어 점을 찍을 수 없음")
        return
    if csv_out is not None:
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        p.to_csv(csv_out, index=False)

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    lim = [0, max(p.gt_mm.max(), p.est_mm.max()) * 1.10]
    ax.plot(lim, lim, color=cr, lw=1.0, ls="--", zorder=1, label="y = x")
    txt = []
    for m, c, mk, nm in ((BASE, cb, "o", "Baseline (SAM3D)"), (ORA, co, "^", "Oracle (GT CAD)")):
        s = p[p.method == m]
        ax.scatter(s.gt_mm, s.est_mm, s=58, marker=mk, facecolor=c, edgecolor="white",
                   linewidth=1.0, zorder=3, label=nm)
        err = (s.est_mm - s.gt_mm).abs()
        mae = err.mean()
        rel = (err / s.gt_mm * 100).mean()
        ss_res = ((s.est_mm - s.gt_mm) ** 2).sum()
        ss_tot = ((s.gt_mm - s.gt_mm.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        txt.append(f"{nm}:  MAE {mae:.2f} mm   Rel {rel:.2f}%   (R²={r2:.4f})")
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("GT Dimension (mm)"); ax.set_ylabel("Estimated Dimension (mm)")
    ax.set_title("GT vs Estimated Dimensions  (per object-axis)", loc="left")
    ax.legend(frameon=False, loc="lower right")
    ax.text(0.03, 0.97, "\n".join(txt), transform=ax.transAxes, va="top", ha="left",
            fontsize=8.6, color="#0b0b0b",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f0efec", ec="#e8e7e3"))
    save(fig, outdir, "fig2_gt_vs_estimated_dimensions", cfg,
         title=fig_title("fig2_gt_vs_estimated_dimensions", suffix))


# ---------------------------------------------------------------- Figure 3
def fig3(obj: pd.DataFrame, outdir: Path, cfg, axis_labels, suffix=""):
    setup(cfg)
    obj = metric_only(obj)
    if obj.empty:
        print(f"  [SKIP] fig3 {suffix}: 크기평가 가능한 물체 없음")
        return
    names = list(dict.fromkeys(obj.object_name))
    d = _disp(obj)
    for method, tag in ((BASE, "baseline"), (ORA, "oracle")):
        # 그 방법이 없는 물체(예: kettle 은 정답 CAD 없음)는 행 자체를 만들지 않는다
        names_m = [n for n in names if len(obj[(obj.object_name == n) & (obj.method == method)])]
        if not names_m:
            print(f"  [SKIP] fig3 {tag} {suffix}: 해당 방법 결과가 없음")
            continue
        M = np.full((len(names_m), 3), np.nan)
        for i, n in enumerate(names_m):
            r = obj[(obj.object_name == n) & (obj.method == method)]
            for j, a in enumerate(axis_labels):
                v = r[f"abs_error_{a}_mm"].iloc[0]
                M[i, j] = np.nan if v is None else v
        fig, ax = plt.subplots(figsize=(5.0, 0.62 * len(names_m) + 2.2))
        vmax = np.nanmax(M) if np.isfinite(M).any() else 1.0
        im = ax.imshow(np.ma.masked_invalid(M), cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
        im.cmap.set_bad("#e8e7e3")
        ax.set_xticks(range(3)); ax.set_xticklabels(axis_labels)
        ax.set_yticks(range(len(names_m))); ax.set_yticklabels([d[n] for n in names_m])
        for i in range(len(names_m)):
            for j in range(3):
                v = M[i, j]
                if np.isnan(v):
                    ax.text(j, i, "n/d", ha="center", va="center", fontsize=8.5,
                            color="#52514e", style="italic")
                else:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9.5,
                            color="white" if v > vmax * 0.55 else "#0b0b0b")
        ax.set_title(f"Per-axis Absolute Error — {'Baseline (SAM3D)' if method == BASE else 'Oracle (GT CAD)'}"
                     "\n(mm, lower is better;  n/d = GT undetermined)", loc="left")
        fig.colorbar(im, ax=ax, label="Absolute error (mm)", fraction=0.046, pad=0.04)
        nm = f"fig3_per_axis_absolute_error_{tag}"
        save(fig, outdir, nm, cfg, title=fig_title(nm, suffix))


# ---------------------------------------------------------------- Figure 4
def fig4(obj: pd.DataFrame, cam: pd.DataFrame, outdir: Path, cfg, suffix=""):
    cb, co, cr = setup(cfg)
    thr = cfg["iou_threshold"]
    b = obj[obj.method == BASE]
    names = list(b.object_name)
    d = _disp(obj)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(4.8, 1.6 * len(names) + 2.2), 4.8))
    vals = [b[b.object_name == n].cross_view_iou.iloc[0] for n in names]
    r = ax.bar(x, vals, 0.55, color=cb, label="Cross-view IoU (mean, source excluded)")
    for rr in r:
        ax.annotate(f"{rr.get_height():.3f}", (rr.get_x() + rr.get_width() / 2, rr.get_height()),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8.5)
    first = True
    for i, n in enumerate(names):
        cs = cam[(cam.object_name == n) & (cam.method == BASE)]
        for _, c in cs.iterrows():
            src = bool(c.is_source_view)
            ax.scatter(i + (0.16 if src else -0.16), c.silhouette_iou, s=42,
                       marker="x" if src else "o",
                       facecolor="none" if not src else cr, edgecolor=cr,
                       linewidth=1.3, zorder=4,
                       label=("Source view (excluded)" if src else "Individual camera") if first else None)
            if src:
                first = False
    ax.set_ylim(0.80, 1.0)          # shape_ok 라벨을 축 하단에 붙이려면 범위를 먼저 확정해야 한다
    ax.axhline(thr, color="#c0392b", lw=1.1, ls="--", zorder=2)
    ax.text(len(names) - 0.45, thr + 0.003, f"IoU = {thr}", color="#c0392b", fontsize=8.5, ha="right")
    y0 = ax.get_ylim()[0]
    for i, n in enumerate(names):
        ok = b[b.object_name == n].shape_ok_by_iou.iloc[0]
        ax.annotate("shape_ok" if ok else "shape flagged", (i, y0),
                    textcoords="offset points", xytext=(0, 5), ha="center", fontsize=7.5,
                    color="#1baf7a" if ok else "#c0392b")
    ax.set_xticks(x); ax.set_xticklabels([d[n] for n in names])
    ax.set_ylabel("Silhouette IoU")
    ax.set_title("Cross-view Silhouette IoU — Baseline  (higher is better)", loc="left")
    ax.legend(frameon=False, loc="lower right", ncol=1)
    save(fig, outdir, "fig4_cross_view_silhouette_iou", cfg,
         title=fig_title("fig4_cross_view_silhouette_iou", suffix))


# ---------------------------------------------------------------- Figure 5
def fig5(obj: pd.DataFrame, cam: pd.DataFrame, outdir: Path, cfg, suffix=""):
    cb, co, cr = setup(cfg)
    b = obj[obj.method == BASE]
    names = list(b.object_name)
    d = _disp(obj)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(4.8, 1.6 * len(names) + 2.2), 4.8))
    w = 0.36
    cvv = [b[b.object_name == n].cross_view_normalized_contour_distance_percent.iloc[0] for n in names]
    svv = [b[b.object_name == n].source_view_normalized_contour_distance_percent.iloc[0] for n in names]
    r1 = ax.bar(x - w / 2, cvv, w, color=cb, label="Cross-view (source excluded)")
    r2 = ax.bar(x + w / 2, svv, w, color=cb, alpha=0.35, label="Source view")
    for rr in list(r1) + list(r2):
        ax.annotate(f"{rr.get_height():.2f}", (rr.get_x() + rr.get_width() / 2, rr.get_height()),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
    first = True
    for i, n in enumerate(names):
        cs = cam[(cam.object_name == n) & (cam.method == BASE)]
        for _, c in cs.iterrows():
            if c.is_source_view:
                continue
            ax.scatter(i - w / 2, c.normalized_contour_distance_percent, s=34, marker="o",
                       facecolor="none", edgecolor=cr, linewidth=1.2, zorder=4,
                       label="Individual camera" if first else None)
            first = False
    ax.set_xticks(x); ax.set_xticklabels([d[n] for n in names])
    ax.set_ylabel("Normalized Contour Distance (%)")
    ax.set_title("Normalized Contour Distance — Baseline  (lower is better)\n"
                 "normalized by real-mask bbox diagonal", loc="left")
    ax.legend(frameon=False)
    ax.set_ylim(0, max(cvv + svv) * 1.25)
    save(fig, outdir, "fig5_normalized_contour_distance", cfg,
         title=fig_title("fig5_normalized_contour_distance", suffix))


# ---------------------------------------------------------------- Figure 6
def fig6(obj: pd.DataFrame, outdir: Path, cfg):
    cb, co, cr = setup(cfg)
    thr = cfg["iou_threshold"]
    b = metric_only(obj)
    b = b[b.method == BASE].copy()
    d = _disp(obj)
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    ax.scatter(b.cross_view_iou, b.mean_dimension_error_mm, s=80, color=cb,
               edgecolor="white", linewidth=1.0, zorder=3)
    for _, r in b.iterrows():
        ax.annotate(d[r.object_name], (r.cross_view_iou, r.mean_dimension_error_mm),
                    textcoords="offset points", xytext=(7, 4), fontsize=9)
    ax.axvline(thr, color="#c0392b", lw=1.1, ls="--")
    # 라벨은 아래쪽에 둔다 (위쪽은 통계 박스가 차지)
    ax.text(thr, ax.get_ylim()[0], f" IoU={thr}", color="#c0392b", fontsize=8.5,
            va="bottom", ha="left")
    x, y = b.cross_view_iou.values, b.mean_dimension_error_mm.values
    ok = np.isfinite(x) & np.isfinite(y)
    note = f"n = {ok.sum()} objects"
    if ok.sum() >= 3:
        pr, pp = stats.pearsonr(x[ok], y[ok])
        sr, sp = stats.spearmanr(x[ok], y[ok])
        note += (f"\nPearson r = {pr:+.3f} (p={pp:.3f})"
                 f"\nSpearman ρ = {sr:+.3f} (p={sp:.3f})"
                 f"\nToo few samples — descriptive only")
    ax.text(0.03, 0.97, note, transform=ax.transAxes, va="top", ha="left", fontsize=8.6,
            bbox=dict(boxstyle="round,pad=0.4", fc="#f0efec", ec="#e8e7e3"))
    ax.set_xlabel("Cross-view Silhouette IoU  (higher is better)")
    ax.set_ylabel("Mean Dimension Error (mm)  (lower is better)")
    ax.set_title("Does cross-view IoU predict dimensional accuracy?  — Baseline", loc="left")
    save(fig, outdir, "fig6_iou_vs_dimension_error", cfg,
         title=fig_title("fig6_iou_vs_dimension_error"))


def generate_all(obj: pd.DataFrame, cam: pd.DataFrame, outdir: Path, cfg,
                 suffix: str = "", csv_dir: Path | None = None):
    """통합(전체 객체) 또는 물체별 부분집합 그림 생성.

    물체가 1개면 Figure 6(IoU vs E_dim 상관)은 점이 하나뿐이라 의미가 없어 건너뛴다.
    Figure 1 의 가로형 변형도 통합본에서만 만든다.
    """
    ax_l = cfg["axis_labels"]
    multi = obj.object_name.nunique() > 1
    multi_metric = metric_only(obj).object_name.nunique() > 1
    fig1(obj, outdir, cfg, suffix=suffix, variants=multi_metric)
    fig2(obj, outdir, cfg, ax_l, suffix=suffix,
         csv_out=(csv_dir / "fig2_scatter_points.csv") if csv_dir else None)
    fig3(obj, outdir, cfg, ax_l, suffix=suffix)
    fig4(obj, cam, outdir, cfg, suffix=suffix)
    fig5(obj, cam, outdir, cfg, suffix=suffix)
    if multi_metric:
        fig6(obj, outdir, cfg)
    else:
        print(f"  [SKIP] fig6 {suffix}: 크기평가 가능한 객체가 2개 미만이라 상관을 볼 수 없음")
