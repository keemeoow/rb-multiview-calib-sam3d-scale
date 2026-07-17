#!/usr/bin/env python3
"""정성적 Real-to-Sim 피규어 — 실제 RGB 위에 real/sim 실루엣 외곽선을 겹친다.

물체당 카메라 3대를 한 행에 배치하고, 각 패널에 다음을 표기한다:
  Camera ID / SOURCE 또는 CROSS-VIEW / IoU / normalized contour distance
SAM3D source camera 는 mesh 를 만든 뷰라 거의 자기 자신에 맞으므로, 실제 일반화를
보여주는 cross-view 패널이 두드러지도록 source 패널은 흐리게 처리한다.

물체 중심점은 표시하지 않는다 (pose 는 이번 평가 대상이 아님).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import eval_common as ec

C_REAL = "#1baf7a"     # 실제 SAM mask 외곽선
C_SIM = "#c0392b"      # 시뮬레이션 렌더 mask 외곽선
C_BBOX = "#8a8a86"


def _crop_box(real, sim, shape, pad=42):
    m = np.logical_or(real, sim)
    if not m.any():
        m = real
    ys, xs = np.where(m)
    if len(xs) == 0:
        return 0, shape[1], 0, shape[0]
    x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad, shape[1])
    y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad, shape[0])
    return x0, x1, y0, y1


def panel(ax, rgb_path, real, sim, title, dim_title, show_bbox=False, faded=False):
    bgr = cv2.imread(str(rgb_path)) if rgb_path else None
    if bgr is None:
        bgr = np.full((*real.shape, 3), 240, np.uint8)
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x0, x1, y0, y1 = _crop_box(real, sim, real.shape)
    img = img[y0:y1, x0:x1].copy()
    if faded:
        img = (img * 0.55 + 255 * 0.45).astype(np.uint8)
    ax.imshow(img)

    def draw(mask, color, lw):
        cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in cs:
            p = c.reshape(-1, 2).astype(float)
            p[:, 0] -= x0; p[:, 1] -= y0
            ax.plot(np.append(p[:, 0], p[0, 0]), np.append(p[:, 1], p[0, 1]),
                    color=color, lw=lw, alpha=0.5 if faded else 1.0)

    draw(real, C_REAL, 1.8)
    draw(sim, C_SIM, 1.5)
    if show_bbox:
        ys, xs = np.where(real)
        if len(xs):
            ax.add_patch(plt.Rectangle((xs.min() - x0, ys.min() - y0),
                                       xs.max() - xs.min(), ys.max() - ys.min(),
                                       fill=False, ec=C_BBOX, lw=0.6, ls=":"))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#c8c7c3"); s.set_linewidth(1.0)
    ax.set_title(title, fontsize=8.8, color="#52514e" if faded else "#0b0b0b",
                 loc="left", pad=3)
    if dim_title:
        ax.set_xlabel(dim_title, fontsize=8.2, color="#52514e", labelpad=2)


def _panels_for(obj_cfg, cfg, method="baseline_sam3d"):
    """(cid, rgb_path, real, sim, is_source) 목록."""
    views = ec.load_views(obj_cfg["capture_dir"], obj_cfg["mask_dir"], cfg["cameras"])
    fit, err = ec.load_fit(obj_cfg, method)
    if fit is None or not views:
        return None, err or "뷰 없음"
    src = obj_cfg.get("source_camera")
    out = []
    for cid, v in views.items():
        sim = ec.render_sim_mask(fit, v["view"])
        out.append((cid, v["rgb_path"], v["mask"], sim, cid == src))
    return out, None


def per_object(cfg, cam_df, outdir: Path, formats=None, also_dir: Path | None = None):
    """물체별 3-view 오버레이. also_dir 가 주어지면 <also_dir>/<object>/ 에도 같이 저장한다."""
    outdir.mkdir(parents=True, exist_ok=True)
    formats = formats or cfg["style"]["formats"]
    made = []
    for o in cfg["objects"]:
        panels, err = _panels_for(o, cfg)
        if panels is None:
            print(f"  [SKIP] qualitative {o['name']}: {err}")
            continue
        n = len(panels)
        fig, axes = plt.subplots(1, n, figsize=(4.1 * n, 4.5))
        axes = np.atleast_1d(axes)
        for ax, (cid, rgb, real, sim, is_src) in zip(axes, panels):
            r = cam_df[(cam_df.object_name == o["name"]) & (cam_df.method == "baseline_sam3d")
                       & (cam_df.camera_id == cid)]
            iou = float(r.silhouette_iou.iloc[0]) if len(r) else float("nan")
            dc = float(r.normalized_contour_distance_percent.iloc[0]) if len(r) else float("nan")
            tag = "SOURCE (SAM3D input)" if is_src else "CROSS-VIEW"
            panel(ax, rgb, real, sim, f"{cid}  ·  {tag}",
                  f"IoU {iou:.3f}   ·   D_contour {dc:.2f}%", faded=is_src)
        name_disp = o.get("display_name", o["name"])
        fig.suptitle(f"7. Qualitative Real-to-Sim — {name_disp} (three views)\n"
                     f"Real (green) vs Sim (red) silhouette   |   "
                     f"SAM3D source: {o.get('source_camera')}",
                     fontsize=12.5, fontweight="bold", x=0.0, y=1.0, ha="left", va="bottom")
        fig.tight_layout()
        targets = [outdir / f"qualitative_{o['name']}_three_views.{{}}"]
        if also_dir is not None:
            d = also_dir / o["name"]
            d.mkdir(parents=True, exist_ok=True)
            targets.append(d / "fig7_qualitative_three_views.{}")
        for t in targets:
            for ext in formats:
                fig.savefig(str(t).format(ext), dpi=cfg["style"]["dpi"],
                            facecolor="white", bbox_inches="tight")
        plt.close(fig)
        print(f"  [SAVE] qualitative_{o['name']}_three_views.{{{','.join(formats)}}}"
              f"{'  (+per_object/)' if also_dir else ''}")
        made.append(o["name"])
    return made


def pick_representatives(obj_df):
    """가장 정확 / 중간 / 최대 오차 + 단순·복잡 형상을 모두 포함하도록 선정."""
    b = obj_df[obj_df.method == "baseline_sam3d"].dropna(subset=["mean_dimension_error_mm"])
    b = b.sort_values("mean_dimension_error_mm")
    if b.empty:
        return []
    picks = [b.iloc[0].object_name, b.iloc[len(b) // 2].object_name, b.iloc[-1].object_name]
    picks = list(dict.fromkeys(picks))
    for cls in ("simple", "complex"):                 # 형상 다양성 보장
        if cls in set(b.shape_class) and not any(
                b[b.object_name == p].shape_class.iloc[0] == cls for p in picks):
            picks.append(b[b.shape_class == cls].iloc[-1].object_name)
    return picks


def grid(cfg, obj_df, cam_df, outdir: Path):
    names = pick_representatives(obj_df)
    if not names:
        print("  [SKIP] fig7: 대표 객체를 고를 수 없음")
        return []
    by = {o["name"]: o for o in cfg["objects"]}
    rows = []
    for n in names:
        p, err = _panels_for(by[n], cfg)
        if p is None:
            print(f"  [SKIP] fig7 row {n}: {err}")
            continue
        rows.append((n, p))
    if not rows:
        return []
    ncol = max(len(p) for _, p in rows)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(4.0 * ncol + 2.6, 4.3 * len(rows)))
    axes = np.atleast_2d(axes)
    for i, (n, panels) in enumerate(rows):
        o = by[n]
        r = obj_df[(obj_df.object_name == n) & (obj_df.method == "baseline_sam3d")].iloc[0]
        for j in range(ncol):
            ax = axes[i, j]
            if j >= len(panels):
                ax.axis("off"); continue
            cid, rgb, real, sim, is_src = panels[j]
            c = cam_df[(cam_df.object_name == n) & (cam_df.method == "baseline_sam3d")
                       & (cam_df.camera_id == cid)]
            iou = float(c.silhouette_iou.iloc[0]) if len(c) else float("nan")
            dc = float(c.normalized_contour_distance_percent.iloc[0]) if len(c) else float("nan")
            tag = "SOURCE (SAM3D input)" if is_src else "CROSS-VIEW"
            panel(ax, rgb, real, sim, f"{cid}  ·  {tag}",
                  f"IoU {iou:.3f}   ·   D_contour {dc:.2f}%", faded=is_src)
        fmt = lambda v: "n/d" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.1f}"
        gt = f"GT      {fmt(r.gt_L_mm)} × {fmt(r.gt_W_mm)} × {fmt(r.gt_H_mm)} mm"
        es = f"Baseline {fmt(r.estimated_L_mm)} × {fmt(r.estimated_W_mm)} × {fmt(r.estimated_H_mm)} mm"
        ed = f"E_dim   {r.mean_dimension_error_mm:.2f} mm"
        axes[i, 0].text(-0.06, 0.5, f"{r.display_name}\n\n{gt}\n{es}\n{ed}",
                        transform=axes[i, 0].transAxes, ha="right", va="center",
                        fontsize=9.2, family="monospace", color="#0b0b0b")
    fig.suptitle("Figure 7. Qualitative Real-to-Sim Grid\n"
                 "Real (green) vs Sim (red) silhouette;  source view is faded, "
                 "cross-view panels show generalization",
                 fontsize=13.5, fontweight="bold", x=0.0, y=1.0, ha="left", va="bottom")
    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    fmts = cfg["style"]["formats"]
    for ext in fmts:
        fig.savefig(outdir / f"fig7_qualitative_real_to_sim_grid.{ext}",
                    dpi=cfg["style"]["dpi"], facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVE] fig7_qualitative_real_to_sim_grid.{{{','.join(fmts)}}}  "
          f"(rows: {', '.join(names)})")
    return names
