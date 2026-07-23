#!/usr/bin/env python3
"""정량 그래프 생성 — 보고 지표 5종에 대응하는 Figure 1~5.

  1. Mean Dimension Error (E_dim)
  2. Mean Relative Dimension Error (E_rel)
  3. Per-axis Absolute Error (e_L / e_W / e_H)
  4. Cross-view Silhouette IoU
  5. Normalized Contour Distance

정성 오버레이(Figure 6)는 generate_qualitative_overlays.py 가 만든다.
축 라벨/범례는 영어, 배경 흰색, PNG 300dpi + 벡터(SVG/PDF).
Baseline/Oracle 색상은 config 에 고정돼 모든 그림에서 동일하다.
GT 미확정 축(kettle L1)은 값이 없으므로 해당 셀을 그리지 않고 'n/d' 로 표기한다.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import eval_common as ec

BASE, ORA = "baseline_sam3d", "oracle_cad"

# 각 피규어 상단에 찍는 제목 (파일명 -> "번호. 타이틀명").
# save() 가 이 표를 보고 suptitle 을 달아, 그림만 떼어놔도 어느 번호인지 알 수 있게 한다.
# 'Figure' 라는 단어는 넣지 않는다 (번호.제목 형식).
FIG_TITLES = {
    "fig1_mean_dimension_error":
        "1. Mean Dimension Error",
    "fig2_mean_relative_dimension_error":
        "2. Mean Relative Dimension Error",
    "fig3_per_axis_absolute_error_baseline":
        "3a. Per-axis Absolute Error — Baseline",
    "fig3_per_axis_absolute_error_oracle":
        "3b. Per-axis Absolute Error — Oracle",
    "fig4_cross_view_silhouette_iou":
        "4. Cross-view Silhouette IoU",
    "fig5_normalized_contour_distance":
        "5. Normalized Contour Distance",
    "fig6_qualitative_real_to_sim_grid":
        "6. Qualitative Real-to-Sim Grid",
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

    GT 나 정답 CAD 가 신뢰 불가해 제외된 물체(excluded_from_metric_scale)는 오차가 없으므로
    Figure 1/2/3 에서 뺀다. 실루엣 지표(Figure 4/5)는 GT 가 필요 없어 그대로 쓴다.
    """
    if "excluded_from_metric_scale" not in df.columns:
        return df
    return df[~df.excluded_from_metric_scale.fillna(False).astype(bool)]


# ------------------------------------------------------- Figure 1 / 2 (공통)
def _error_bars(obj: pd.DataFrame, outdir: Path, cfg, column: str, ylabel: str,
                title: str, name: str, fmt: str, suffix=""):
    """물체별 오차 막대 (Baseline vs Oracle). Figure 1(E_dim)/2(E_rel) 공통 본체."""
    cb, co, cr = setup(cfg)
    obj = metric_only(obj)
    if obj.empty:
        print(f"  [SKIP] {name} {suffix}: 크기평가 가능한 물체 없음")
        return
    names = list(dict.fromkeys(obj.object_name))
    d = _disp(obj)

    def get(n, meth):
        """물체×방법이 없을 수 있다 (예: kettle 은 oracle 만 크기평가 제외) -> NaN."""
        r = obj[(obj.object_name == n) & (obj.method == meth)]
        return float(r[column].iloc[0]) if len(r) else np.nan

    b = [get(n, BASE) for n in names]
    o = [get(n, ORA) for n in names]
    bm = np.nanmean(b) if np.isfinite(b).any() else np.nan
    om = np.nanmean(o) if np.isfinite(o).any() else np.nan
    show_mean = len(names) > 1        # 물체가 하나면 Mean 막대는 같은 값의 중복이라 생략

    x = np.arange(len(names) + (1 if show_mean else 0))
    bb, oo = (b + [bm], o + [om]) if show_mean else (list(b), list(o))
    lab = [d[n] for n in names] + (["Mean"] if show_mean else [])
    wf = max(4.6, 1.2 * len(x) + 2.4)          # 막대 수에 비례한 폭
    fig, ax = plt.subplots(figsize=(wf, 4.6))
    w = 0.38
    r1 = ax.bar(x - w / 2, bb, w, label="Baseline (SAM3D)", color=cb)
    r2 = ax.bar(x + w / 2, oo, w, label="Oracle (GT CAD)", color=co)
    ax.set_xticks(x); ax.set_xticklabels(lab)
    ax.set_ylabel(ylabel)
    for r in list(r1) + list(r2):
        if not np.isfinite(r.get_height()):
            continue                             # 크기평가 제외 조합은 막대·라벨 없음
        ax.annotate(fmt.format(r.get_height()), (r.get_x() + r.get_width() / 2, r.get_height()),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
    if show_mean:
        ax.axvline(len(names) - 0.5, color=cr, lw=0.8, ls=":")
    ax.set_ylim(0, np.nanmax(bb + oo) * 1.22)
    ax.set_title(title, loc="left")
    ax.legend(frameon=False)
    save(fig, outdir, name, cfg, title=fig_title(name, suffix))


# ---------------------------------------------------------------- Figure 1
def fig1(obj: pd.DataFrame, outdir: Path, cfg, suffix=""):
    """E_dim = 축별 |오차| 의 평균 [mm]."""
    _error_bars(obj, outdir, cfg,
                column="mean_dimension_error_mm",
                ylabel="Mean Dimension Error (mm)",
                title="Mean Dimension Error  (lower is better)",
                name="fig1_mean_dimension_error", fmt="{:.2f}", suffix=suffix)


# ---------------------------------------------------------------- Figure 2
def fig2(obj: pd.DataFrame, outdir: Path, cfg, suffix=""):
    """E_rel = 축별 |오차|/GT 의 평균 [%]. 크기가 다른 물체를 함께 볼 때 쓴다."""
    _error_bars(obj, outdir, cfg,
                column="mean_relative_dimension_error_percent",
                ylabel="Mean Relative Dimension Error (%)",
                title="Mean Relative Dimension Error  (lower is better)",
                name="fig2_mean_relative_dimension_error", fmt="{:.2f}", suffix=suffix)


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
    """Cross-view Silhouette IoU. source view 는 mesh 를 만든 뷰라 막대·점 모두에서 제외."""
    cb, co, cr = setup(cfg)
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
        cs = cam[(cam.object_name == n) & (cam.method == BASE) & (~cam.is_source_view)]
        for _, c in cs.iterrows():
            # 막대 안쪽 왼편에 찍는다 — 중앙에 두면 막대 위 값 라벨과 겹친다
            ax.scatter(i - 0.2, c.silhouette_iou, s=42, marker="o", facecolor="none",
                       edgecolor=cr, linewidth=1.3, zorder=4,
                       label="Individual camera" if first else None)
            first = False
    ax.set_ylim(0.80, 1.0)
    ax.set_xticks(x); ax.set_xticklabels([d[n] for n in names])
    ax.set_ylabel("Silhouette IoU")
    ax.set_title("Cross-view Silhouette IoU — Baseline  (higher is better)", loc="left")
    ax.legend(loc="lower right", ncol=1, facecolor="white", edgecolor="#e8e7e3", framealpha=0.95)
    save(fig, outdir, "fig4_cross_view_silhouette_iou", cfg,
         title=fig_title("fig4_cross_view_silhouette_iou", suffix))


# ---------------------------------------------------------------- Figure 5
def fig5(obj: pd.DataFrame, cam: pd.DataFrame, outdir: Path, cfg, suffix=""):
    """Normalized Contour Distance. Figure 4 와 같이 cross-view 평균만 보고한다."""
    cb, co, cr = setup(cfg)
    b = obj[obj.method == BASE]
    names = list(b.object_name)
    d = _disp(obj)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(4.8, 1.6 * len(names) + 2.2), 4.8))
    cvv = [b[b.object_name == n].cross_view_normalized_contour_distance_percent.iloc[0] for n in names]
    r1 = ax.bar(x, cvv, 0.55, color=cb, label="Cross-view (source excluded)")
    for rr in r1:
        ax.annotate(f"{rr.get_height():.2f}", (rr.get_x() + rr.get_width() / 2, rr.get_height()),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
    first = True
    hi = max(cvv)
    for i, n in enumerate(names):
        cs = cam[(cam.object_name == n) & (cam.method == BASE) & (~cam.is_source_view)]
        for _, c in cs.iterrows():
            hi = max(hi, float(c.normalized_contour_distance_percent))   # 점이 잘리지 않게
            ax.scatter(i - 0.2, c.normalized_contour_distance_percent, s=34, marker="o",
                       facecolor="none", edgecolor=cr, linewidth=1.2, zorder=4,
                       label="Individual camera" if first else None)
            first = False
    ax.set_xticks(x); ax.set_xticklabels([d[n] for n in names])
    ax.set_ylabel("Normalized Contour Distance (%)")
    ax.set_title("Normalized Contour Distance — Baseline  (lower is better)\n"
                 "normalized by real-mask bbox diagonal", loc="left")
    ax.legend(loc="upper right", facecolor="white", edgecolor="#e8e7e3", framealpha=0.95)
    ax.set_ylim(0, hi * 1.30)
    save(fig, outdir, "fig5_normalized_contour_distance", cfg,
         title=fig_title("fig5_normalized_contour_distance", suffix))


def generate_all(obj: pd.DataFrame, cam: pd.DataFrame, outdir: Path, cfg, suffix: str = ""):
    """통합(전체 객체) 또는 물체별 부분집합 그림 생성."""
    fig1(obj, outdir, cfg, suffix=suffix)
    fig2(obj, outdir, cfg, suffix=suffix)
    fig3(obj, outdir, cfg, cfg["axis_labels"], suffix=suffix)
    fig4(obj, cam, outdir, cfg, suffix=suffix)
    fig5(obj, cam, outdir, cfg, suffix=suffix)
