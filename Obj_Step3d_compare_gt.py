#!/usr/bin/env python3
"""
Rebuttal_Obj_Step3d_compare_gt.py

실측(ground truth) vs 추정 크기 비교.

파이프라인의 유일한 크기 추정기인 CAD 다중뷰 실루엣 정합(Rebuttal_Obj_Step3c)을 실측과 비교한다.
자 실측 기준 평균 |오차| 0.90mm. (제거된 옛 방식들은 각각 3.04mm / 6.07mm 였다.)

[출력]
  gt_vs_estimate.png       부호 있는 오차 점 도표 + 값 표
  gt_vs_estimate.csv       표 뷰 (차트와 같은 숫자)
  gt_comparison.json       기계 판독용
  gt_overlay_<obj>.jpg     실측 크기 CAD 실루엣 vs 추정 크기 CAD 실루엣 (같은 포즈)

[실행]
  python Rebuttal_Obj_Step3d_compare_gt.py \
    --fit_dir     data/outputs_cad_fit \
    --capture_dir data/capture_obj \
    --mask_dir    data/masks \
    --cad peg=data/meshes/peg.glb --cad hole=data/meshes/hole.glb \
    --gt  peg=45,30,30      --gt  hole=50,50,50 \
    --gt_tol_mm 0.5 \
    --out_dir data/outputs_cad_fit

  실측은 --gt (mm, 내림차순 정렬해 rank 대응) 또는 --gt_json 으로 준다.
  --gt_json 은 {"peg": {"extents_mm": [...], "tol_mm": 0.5}} 형식과
  Gt_Step2 리포트({"label":..., "gt_extents_mm_sorted_desc":[...]}) 를 모두 읽는다.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh

import Obj_Step3c_cad_scale as s3c
from _silhouette_fit import obb_frame, render_silhouette, scale_cad_to_extents

def _use_cjk_font() -> bool:
    """한글 폰트가 있으면 쓰고, 없으면 영어 라벨로 물러선다.

    폰트가 없는 환경(예: 서버)에서 matplotlib 은 경고만 내고 한글을 두부(□)로 그린다.
    조용히 깨진 그림을 내보내느니 영어로 그리는 편이 낫다.
    """
    from matplotlib import font_manager as fm
    names = {f.name for f in fm.fontManager.ttflist}
    for c in ("Apple SD Gothic Neo", "NanumGothic", "Noto Sans CJK KR",
              "Noto Sans KR", "Malgun Gothic", "AppleGothic"):
        if c in names:
            plt.rcParams["font.family"] = c
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


CJK = None   # main() 에서 설정

STR = {
    True: dict(
        title="실측 대비 크기 추정 오차",
        xlabel="추정 - 실측  (mm)        ← 과소추정        과대추정 →",
        table="표 뷰 - 괄호 안은 실측 대비 오차",
        gt="GT (mm)", obj="물체", axis="축",
    ),
    False: dict(
        title="Size estimation error vs ruler ground truth",
        xlabel="estimate - ground truth  (mm)      <- under      over ->",
        table="Table view - parentheses show error vs ground truth",
        gt="GT (mm)", obj="object", axis="axis",
    ),
}

# ---- design tokens (dataviz reference palette; validated with validate_palette.js) ----
THEME = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e8e7e3",
                  band="#f0efec", series=("#2a78d6", "#1baf7a", "#eda100")),
    "dark":  dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", grid="#383835",
                  band="#2a2a28", series=("#3987e5", "#199e70", "#c98500")),
}
# 계열은 엔티티에 고정 (순위에 따라 다시 칠하지 않는다)
METHODS = [
    ("silhouette", "CAD silhouette fit (Step3c)"),
]


def parse_gt(args) -> dict:
    gt = {}
    if args.gt_json:
        data = json.loads(Path(args.gt_json).read_text())
        if "gt_extents_mm_sorted_desc" in data:            # Gt_Step2 리포트 (단일 물체)
            label = data.get("label", "object")
            gt[label] = {"extents_mm": data["gt_extents_mm_sorted_desc"], "source": "Gt_Step2"}
        else:
            for k, v in data.items():
                if isinstance(v, dict) and "gt_extents_mm_sorted_desc" in v:
                    gt[k] = {"extents_mm": v["gt_extents_mm_sorted_desc"], "source": "Gt_Step2"}
                elif isinstance(v, dict) and "extents_mm" in v:
                    gt[k] = {"extents_mm": v["extents_mm"], "source": v.get("source", "manual"),
                             "tol_mm": v.get("tol_mm")}
                else:
                    gt[k] = {"extents_mm": list(v), "source": "manual"}
    for spec in (args.gt or []):
        if "=" not in spec:
            raise SystemExit(f"--gt 형식은 peg=45,30,30 입니다 (받은 값: {spec!r})")
        k, v = spec.split("=", 1)
        vals = [float(x) for x in v.split(",")]
        if len(vals) != 3:
            raise SystemExit(f"--gt {k}: 3개 치수(mm)가 필요합니다 (받은 값: {vals})")
        gt[k] = {"extents_mm": vals, "source": "manual"}
    if not gt:
        raise SystemExit("--gt 또는 --gt_json 으로 실측값을 주세요")
    for k, v in gt.items():
        v["extents_mm"] = sorted([float(x) for x in v["extents_mm"]], reverse=True)
        if v.get("tol_mm") is None:
            v["tol_mm"] = float(args.gt_tol_mm)
    return gt


def chart(rows, out_png: Path, mode: str, gt_tol_mm: float):
    """rows: [(obj, axis_idx, gt_mm, {method: (est_mm, lo_mm, hi_mm|None)})]"""
    T = THEME[mode]
    n = len(rows)
    fig = plt.figure(figsize=(11.6, 0.92 * n + 3.9), facecolor=T["surface"])
    gs = fig.add_gridspec(2, 1, height_ratios=[1.30 * n, 0.55 * n + 1.2], hspace=0.55)
    ax = fig.add_subplot(gs[0], facecolor=T["surface"])
    axt = fig.add_subplot(gs[1], facecolor=T["surface"])

    ys, labels = [], []
    y = 0.0
    prev_obj = None
    for obj, ai, gt_mm, ests in rows:
        if prev_obj is not None and obj != prev_obj:
            y += 0.9                              # 물체 사이 여백
        ys.append(y)
        labels.append(f"{obj}  ·  L{ai+1}   (GT {gt_mm:.1f} mm)")
        y += 1.0
        prev_obj = obj

    # 실측 허용오차 밴드 + 0 기준선
    ax.axvspan(-gt_tol_mm, gt_tol_mm, color=T["band"], zorder=0, lw=0)
    ax.axvline(0.0, color=T["ink2"], lw=1.0, zorder=1)

    off = {"silhouette": 0.0, "depth_icp": +0.22, "cloud_obb": -0.22}
    for si, (key, name) in enumerate(METHODS):
        col = T["series"][si]
        xs, yy, lo, hi = [], [], [], []
        for (obj, ai, gt_mm, ests), yv in zip(rows, ys):
            if key not in ests:
                continue
            e, l, h = ests[key]
            xs.append(e - gt_mm); yy.append(yv + off[key])
            lo.append(0.0 if l is None else (e - gt_mm) - (l - gt_mm))
            hi.append(0.0 if h is None else (h - gt_mm) - (e - gt_mm))
        if any(v > 0 for v in lo + hi):
            ax.errorbar(xs, yy, xerr=[lo, hi], fmt="none", ecolor=col,
                        elinewidth=2, alpha=0.45, capsize=0, zorder=2)
        ax.scatter(xs, yy, s=90, color=col, label=name, zorder=3,
                   edgecolors=T["surface"], linewidths=2)   # 2px surface ring

    # 선택적 직접 라벨: 권장 방식(실루엣)만
    for (obj, ai, gt_mm, ests), yv in zip(rows, ys):
        if "silhouette" not in ests:
            continue
        d = ests["silhouette"][0] - gt_mm
        ax.annotate(f"{d:+.1f}", (d, yv + off["silhouette"]),
                    textcoords="offset points", xytext=(0, 9), ha="center",
                    fontsize=8.5, color=T["ink2"])

    ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=9.5, color=T["ink"])
    ax.invert_yaxis()
    S = STR[bool(CJK)]
    ax.set_xlabel(S["xlabel"], fontsize=10, color=T["ink2"], labelpad=9)
    ax.set_title(S["title"], fontsize=13.5, color=T["ink"],
                 pad=(34 if len(METHODS) > 1 else 14), loc="left")
    ax.grid(axis="x", color=T["grid"], lw=1.0, ls="-", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(T["grid"])
    ax.tick_params(colors=T["ink2"], length=0)
    # 계열이 하나면 범례 상자는 제목을 반복할 뿐이다 (제목이 무엇을 그리는지 말한다).
    if len(METHODS) > 1:
        leg = ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=len(METHODS),
                        frameon=False, fontsize=9.5, handletextpad=0.4, columnspacing=1.6)
        for t in leg.get_texts():
            t.set_color(T["ink"])

    # 표 뷰 (대비 WARN 완화 규칙: 라벨 또는 표가 반드시 존재해야 한다)
    axt.axis("off")
    head = [S["obj"], S["axis"], S["gt"]] + [f"{n}\n(mm / d mm)" for _, n in METHODS]
    body = []
    for obj, ai, gt_mm, ests in rows:
        r = [obj, f"L{ai+1}", f"{gt_mm:.1f}"]
        for key, _ in METHODS:
            if key in ests:
                e = ests[key][0]
                r.append(f"{e:.1f}  ({e - gt_mm:+.1f})")
            else:
                r.append("—")
        body.append(r)
    tb = axt.table(cellText=body, colLabels=head, cellLoc="center", loc="upper center")
    tb.auto_set_font_size(False); tb.set_fontsize(8.6); tb.scale(1, 1.55)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor(T["grid"])
        cell.get_text().set_color(T["ink"] if r else T["ink2"])
        cell.set_facecolor(T["surface"])
        if r == 0:
            cell.get_text().set_fontweight("bold")
    axt.set_title(S["table"], fontsize=10, color=T["ink2"], loc="left", pad=2)

    fig.savefig(out_png, dpi=170, facecolor=T["surface"], bbox_inches="tight")
    plt.close(fig)


def overlay(capture_dir: Path, cams, views, mesh, fit_json, gt_desc_mm, obj, out_path, mode="light"):
    """같은 포즈에서 '추정 크기 CAD' vs '실측 크기 CAD' 실루엣을 겹쳐 그린다."""
    T = THEME[mode]
    bgr = lambda h: tuple(int(h[i:i+2], 16) for i in (5, 3, 1))   # '#rrggbb' -> BGR
    C_EST = bgr(T["series"][0])
    C_GT = (255, 255, 255)

    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces)
    R = np.array(fit_json["T_world_cad_4x4"])[:3, :3]
    t = np.array(fit_json["T_world_cad_4x4"])[:3, 3]
    s = float(fit_json["scale_cad_to_world"])
    V_gt = scale_cad_to_extents(mesh, np.asarray(gt_desc_mm) / 1000.0)

    tiles = []
    for cid, v in zip(cams, views):
        rgb = cv2.imread(str(capture_dir / f"{cid}_rgb.png"))
        if rgb is None:
            continue
        img = rgb.copy()
        cs, _ = cv2.findContours(v.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cs, -1, (170, 170, 170), 1)                      # 마스크 (문맥)

        sil_gt = render_silhouette(V_gt, F, 1.0, R, t, v, ss=1)
        cg, _ = cv2.findContours(sil_gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cg, -1, (0, 0, 0), 4)                            # halo
        cv2.drawContours(img, cg, -1, C_GT, 2)                                 # 실측 크기

        sil_est = render_silhouette(V, F, s, R, t, v, ss=1)
        ce, _ = cv2.findContours(sil_est.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, ce, -1, C_EST, 2)                                # 추정 크기

        ys, xs = np.where(v.mask)
        x0, x1 = max(int(xs.min()) - 45, 0), min(int(xs.max()) + 45, img.shape[1])
        y0, y1 = max(int(ys.min()) - 45, 0), min(int(ys.max()) + 45, img.shape[0])
        crop = cv2.resize(img[y0:y1, x0:x1], None, fx=3.0, fy=3.0, interpolation=cv2.INTER_NEAREST)
        cv2.putText(crop, f"{obj} {cid}", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(crop, f"{obj} {cid}", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        tiles.append(crop)

    if not tiles:
        return
    h = max(c.shape[0] for c in tiles)
    tiles = [cv2.copyMakeBorder(c, 0, h - c.shape[0], 0, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))
             for c in tiles]
    strip = cv2.hconcat(tiles)
    leg = np.full((52, strip.shape[1], 3), 255, np.uint8)
    est = fit_json["extents_mm_sorted_desc"]
    cv2.line(leg, (16, 20), (60, 20), (0, 0, 0), 4)
    cv2.line(leg, (16, 20), (60, 20), C_GT, 2)
    cv2.putText(leg, f"GT (ruler)  {gt_desc_mm[0]:.1f} x {gt_desc_mm[1]:.1f} x {gt_desc_mm[2]:.1f} mm",
                (70, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.line(leg, (16, 42), (60, 42), C_EST, 3)
    cv2.putText(leg, f"silhouette fit  {est[0]:.1f} x {est[1]:.1f} x {est[2]:.1f} mm",
                (70, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.vconcat([strip, leg]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit_dir", type=Path, required=True, help="Obj_Step3c 출력 폴더")
    ap.add_argument("--capture_dir", type=Path, required=True)
    ap.add_argument("--mask_dir", type=Path, required=True)
    ap.add_argument("--cad", action="append", required=True, metavar="peg=model.glb")
    ap.add_argument("--gt", action="append", metavar="peg=45,30,30", help="실측 치수(mm)")
    ap.add_argument("--gt_json", type=Path, default=None)
    ap.add_argument("--gt_tol_mm", type=float, default=0.5, help="실측 도구 허용오차 (자=0.5mm)")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--dark", action="store_true", help="다크 서피스용 차트도 저장")
    ap.add_argument("--from_report", action="store_true",
                    help="gt_comparison.json 을 읽어 차트만 다시 그린다 (정합/오버레이 생략)")
    args = ap.parse_args()

    global CJK
    CJK = _use_cjk_font()
    if not CJK:
        print("[INFO] 한글 폰트를 찾지 못해 차트 라벨을 영어로 그립니다")

    cad = s3c.parse_cad(args.cad)
    gt = parse_gt(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.from_report:
        report = json.loads((args.out_dir / "gt_comparison.json").read_text())
        rows = []
        for obj, r in report.items():
            lo = (r.get("silhouette_mask_pm1px") or {}).get("erode1px")
            hi = (r.get("silhouette_mask_pm1px") or {}).get("dilate1px")
            for ai in range(3):
                ests = {}
                for key, _ in METHODS:
                    if key not in r["estimates_mm_sorted_desc"]:
                        continue
                    ests[key] = (float(r["estimates_mm_sorted_desc"][key][ai]),
                                 lo[ai] if (key == "silhouette" and lo) else None,
                                 hi[ai] if (key == "silhouette" and hi) else None)
                rows.append((obj, ai, float(r["gt_extents_mm_sorted_desc"][ai]), ests))
        tol = float(np.mean([report[o]["gt_tol_mm"] for o in report]))
        chart(rows, args.out_dir / "gt_vs_estimate.png", "light", tol)
        print(f"[SAVE] {args.out_dir / 'gt_vs_estimate.png'}")
        if args.dark:
            chart(rows, args.out_dir / "gt_vs_estimate_dark.png", "dark", tol)
            print(f"[SAVE] {args.out_dir / 'gt_vs_estimate_dark.png'}")
        return

    cams = s3c.discover_cams(args.capture_dir)

    rows, report = [], {}
    for obj, cad_path in cad.items():
        if obj not in gt:
            print(f"[WARN] {obj}: 실측값이 없어 건너뜁니다")
            continue
        fit_json = json.loads((args.fit_dir / f"{obj}_cad_fit.json").read_text())
        mesh = trimesh.load(str(cad_path), force="mesh")
        gt_mm = gt[obj]["extents_mm"]

        est = {"silhouette": fit_json["extents_mm_sorted_desc"]}
        lo = fit_json.get("extents_mm_mask_erode1px")
        hi = fit_json.get("extents_mm_mask_dilate1px")

        print(f"\n=== {obj} (GT {gt_mm[0]:.1f} x {gt_mm[1]:.1f} x {gt_mm[2]:.1f} mm, "
              f"source={gt[obj]['source']}) ===")
        for key, name in METHODS:
            if key not in est:
                continue
            e = np.asarray(est[key])
            d = e - np.asarray(gt_mm)
            print(f"  {name:<28} {e[0]:6.1f} {e[1]:6.1f} {e[2]:6.1f}   "
                  f"Δ {d[0]:+5.1f} {d[1]:+5.1f} {d[2]:+5.1f}   "
                  f"|Δ|mean {np.abs(d).mean():5.2f} mm  ({np.abs(d / np.asarray(gt_mm)).mean()*100:4.1f}%)")

        for ai in range(3):
            ests = {}
            for key, _ in METHODS:
                if key not in est:
                    continue
                l = lo[ai] if (key == "silhouette" and lo) else None
                h = hi[ai] if (key == "silhouette" and hi) else None
                ests[key] = (float(est[key][ai]), l, h)
            rows.append((obj, ai, float(gt_mm[ai]), ests))

        views, _ = s3c.build_views(args.capture_dir, args.mask_dir, obj, cams)
        overlay(args.capture_dir, cams, views, mesh, fit_json, gt_mm, obj,
                args.out_dir / f"gt_overlay_{obj}.jpg")
        print(f"  [SAVE] {args.out_dir / f'gt_overlay_{obj}.jpg'}")

        report[obj] = {
            "gt_extents_mm_sorted_desc": gt_mm,
            "gt_source": gt[obj]["source"],
            "gt_tol_mm": gt[obj]["tol_mm"],
            "estimates_mm_sorted_desc": {k: [float(x) for x in v] for k, v in est.items()},
            "errors_mm": {k: [float(x) for x in (np.asarray(v) - np.asarray(gt_mm))]
                          for k, v in est.items()},
            "mean_abs_error_mm": {k: float(np.abs(np.asarray(v) - np.asarray(gt_mm)).mean())
                                  for k, v in est.items()},
            "silhouette_mask_pm1px": {"erode1px": lo, "dilate1px": hi},
        }

    if not rows:
        raise SystemExit("비교할 물체가 없습니다")

    tol = float(np.mean([gt[o]["tol_mm"] for o in report]))
    chart(rows, args.out_dir / "gt_vs_estimate.png", "light", tol)
    print(f"\n[SAVE] {args.out_dir / 'gt_vs_estimate.png'}")
    if args.dark:
        chart(rows, args.out_dir / "gt_vs_estimate_dark.png", "dark", tol)
        print(f"[SAVE] {args.out_dir / 'gt_vs_estimate_dark.png'}")

    with open(args.out_dir / "gt_vs_estimate.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["object", "axis", "gt_mm"] +
                   [c for key, _ in METHODS for c in (f"{key}_mm", f"{key}_err_mm")])
        for obj, ai, gt_mm_v, ests in rows:
            r = [obj, f"L{ai+1}", f"{gt_mm_v:.3f}"]
            for key, _ in METHODS:
                if key in ests:
                    r += [f"{ests[key][0]:.3f}", f"{ests[key][0] - gt_mm_v:+.3f}"]
                else:
                    r += ["", ""]
            w.writerow(r)
    print(f"[SAVE] {args.out_dir / 'gt_vs_estimate.csv'}")

    (args.out_dir / "gt_comparison.json").write_text(json.dumps(report, indent=2))
    print(f"[SAVE] {args.out_dir / 'gt_comparison.json'}")

    print("\n[SUMMARY] mean |error| vs ruler")
    for key, name in METHODS:
        vals = [report[o]["mean_abs_error_mm"][key] for o in report
                if key in report[o]["mean_abs_error_mm"]]
        if vals:
            print(f"  {name:<28} {np.mean(vals):5.2f} mm")


if __name__ == "__main__":
    main()
