#!/usr/bin/env python3
"""
Obj_Step3d_compare_gt.py

실측(ground truth) vs 추정 크기 비교 — 피규어 생성용.

--fit_dir 로 준 크기 추정 결과를 캘리퍼 실측과 비교한다. 두 경로 모두 넣을 수 있다:
  oracle    Obj_Step3c_cad_scale.py  출력 (data/outputs_cad_fit)  — 정답 CAD 기준 = 상한
  baseline  Obj_Step3_sam3d_scale.py 출력 (SAM3D 로 만든 단서 CAD) — 실제 운용 방법
정합 엔진은 둘이 같으므로, 이 비교가 보여주는 건 "단서 메시 형상이 참값일 때(oracle) vs
SAM3D 추정일 때(baseline) 치수 오차가 얼마나 벌어지는가" 다.

정답 CAD(oracle) 기준 자 실측 평균 |오차| 0.90mm. (제거된 옛 방식들은 3.04mm / 6.07mm.)

[출력]
  gt_vs_estimate.png       부호 있는 오차 점 도표 + 값 표 (baseline vs oracle 계열)
  gt_vs_estimate.csv       표 뷰 (차트와 같은 숫자)
  gt_comparison.json       기계 판독용 (baseline_mean_iou 포함)
  gt_overlay_<obj>.jpg     실측 크기 CAD 실루엣 vs oracle 추정 크기 CAD 실루엣 (같은 포즈)
                           — 정답 CAD 기준 그림이라 --fit_dir + --cad 있을 때만 생성

[실행] GT / oracle / baseline 3-way 비교 (리뷰탈 피규어)
  python Obj_Step3d_compare_gt.py \
    --fit_dir      data/outputs_cad_fit \
    --baseline_dir data/outputs \
    --capture_dir  data/capture_obj \
    --mask_dir     data/masks \
    --cad peg=data/meshes/peg.glb --cad hole=data/meshes/hole.glb \
    --gt  peg=45,30,30      --gt  hole=50,50,50 \
    --gt_tol_mm 0.5 \
    --out_dir data/outputs_cad_fit

  --fit_dir / --baseline_dir 중 하나만 줘도 된다 (그 계열만 그린다).
  baseline 만 볼 때는 --cad 생략 가능 (gt_overlay 는 생략됨).

  실측은 --gt (mm, 내림차순 정렬해 rank 대응) 또는 --gt_json 으로 준다.
  --gt_json 은 {"peg": {"extents_mm": [...], "tol_mm": 0.5}} 형식과
  Gt_Step2 리포트({"label":..., "gt_extents_mm_sorted_desc":[...]}) 를 모두 읽는다.

[GT 미확정 축]
  OBB 최장축이 캘리퍼로 잴 지점과 대응되지 않는 물체가 있다 (예: 주둥이·손잡이가
  튀어나온 주전자 — L2/L3 는 0.1mm 로 맞는데 L1 만 어긋난다). 그 축에 억지 숫자를
  넣으면 의미 없는 오차가 보고되고, 조용히 빼면 cherry-picking 이 된다.
  그래서 축 단위로 '?' 를 허용한다 — 오차 계산에서만 빼고, 그림·표에는
  'GT 미확정' 으로 사유와 함께 남긴다.

    --gt obj1=?,68,65        # L1 미확정, L2=68, L3=65
    --gt_json '{"obj1": {"extents_mm": [null, 68, 65], "note": "L1 은 주둥이 포함 여부 불명"}}'

  주의: '?' 를 쓰면 자동 정렬을 못 하므로 (None 은 순서를 모른다) 내림차순
  rank(L1>=L2>=L3) 순서로 직접 줘야 한다.
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
        gt_unknown="GT 미확정", unknown_short="미확정",
        unknown_cap="GT 미확정 축은 오차를 계산하지 않는다 (추정치만 표기)",
    ),
    False: dict(
        title="Size estimation error vs ruler ground truth",
        xlabel="estimate - ground truth  (mm)      <- under      over ->",
        table="Table view - parentheses show error vs ground truth",
        gt="GT (mm)", obj="object", axis="axis",
        gt_unknown="GT undetermined", unknown_short="n/d",
        unknown_cap="axes with undetermined GT are excluded from error (estimate shown only)",
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
# baseline 이 관심 대상이라 primary 계열을 준다. oracle 은 상한 기준.
METHODS = [
    ("baseline", "Baseline: SAM3D mesh (Step3)"),
    ("oracle", "Oracle: GT CAD (Step3c)"),
]
PRIMARY = "baseline"      # 직접 라벨을 붙일 계열 (없으면 present 첫 계열)


def find_baseline_json(baseline_dir: Path, obj: str):
    """Obj_Step3_sam3d_scale 의 <obj>_size.json 을 찾는다.

    obj_tag 규칙이 'peg' -> 'objpeg' 라 폴더명이 어긋날 수 있어 여러 패턴을 시도하고,
    마지막엔 아무 *_size.json 이나 열어 내부 "obj" 필드로 매칭한다.
    """
    for c in (baseline_dir / obj / f"{obj}_size.json",
              baseline_dir / f"obj{obj}" / f"obj{obj}_size.json",
              baseline_dir / f"{obj}_size.json"):
        if c.exists():
            return c
    for p in sorted(baseline_dir.glob("**/*_size.json")):
        try:
            tag = str(json.loads(p.read_text()).get("obj", ""))
        except Exception:
            continue
        if tag in (obj, f"obj{obj}"):
            return p
    return None


GT_UNKNOWN_TOKENS = {"?", "null", "none", "na", "n/a", "-", ""}


def gt_val(tok):
    """'?' / null / '-' -> None (GT 미확정). 그 외는 float(mm).

    OBB 최장축이 캘리퍼로 잴 지점과 대응되지 않는 경우(예: 주둥이·손잡이가 튀어나온
    주전자)가 있다. 그 축에 억지 숫자를 넣으면 의미 없는 오차가 보고되므로, 축 단위로
    '미확정'을 허용해 오차 계산에서만 빼고 그림에는 사유와 함께 남긴다.
    """
    if tok is None:
        return None
    s = str(tok).strip().lower()
    return None if s in GT_UNKNOWN_TOKENS else float(s)


def axis_errors(est_desc, gt_desc):
    """축별 오차(mm). GT 미확정 축은 None."""
    return [None if g is None else float(e - g) for e, g in zip(est_desc, gt_desc)]


def mean_abs_err(errs):
    """확정된 축에 대해서만 평균 |오차|. 전부 미확정이면 None."""
    v = [abs(x) for x in errs if x is not None]
    return float(np.mean(v)) if v else None


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
                             "tol_mm": v.get("tol_mm"), "note": v.get("note")}
                else:
                    gt[k] = {"extents_mm": list(v), "source": "manual"}
    for spec in (args.gt or []):
        if "=" not in spec:
            raise SystemExit(f"--gt 형식은 peg=45,30,30 입니다 (받은 값: {spec!r})")
        k, v = spec.split("=", 1)
        vals = [gt_val(x) for x in v.split(",")]
        if len(vals) != 3:
            raise SystemExit(f"--gt {k}: 3개 치수(mm)가 필요합니다 (받은 값: {vals})")
        gt[k] = {"extents_mm": vals, "source": "manual"}
    if not gt:
        raise SystemExit("--gt 또는 --gt_json 으로 실측값을 주세요")
    for k, v in gt.items():
        e = [gt_val(x) for x in v["extents_mm"]]
        if any(x is None for x in e):
            # 미확정 축이 있으면 정렬할 수 없다 (None 은 순서를 모른다).
            # 이 경우 사용자가 이미 내림차순 rank(L1>=L2>=L3) 로 준 것으로 본다.
            known = [x for x in e if x is not None]
            if known != sorted(known, reverse=True):
                raise SystemExit(
                    f"--gt {k}: 미확정 축('?')이 있으면 내림차순 rank(L1>=L2>=L3) 순서로 "
                    f"주세요. 확정 축이 내림차순이 아닙니다: {e}")
            v["extents_mm"] = e
        else:
            v["extents_mm"] = sorted([float(x) for x in e], reverse=True)
        if v.get("tol_mm") is None:
            v["tol_mm"] = float(args.gt_tol_mm)
    return gt


def chart(rows, out_png: Path, mode: str, gt_tol_mm: float):
    """rows: [(obj, axis_idx, gt_mm, {method: (est_mm, lo_mm, hi_mm|None)})]"""
    T = THEME[mode]
    S = STR[bool(CJK)]
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
        labels.append(f"{obj}  ·  L{ai+1}   (GT {gt_mm:.1f} mm)" if gt_mm is not None
                      else f"{obj}  ·  L{ai+1}   ({S['gt_unknown']})")
        y += 1.0
        prev_obj = obj

    # 실측 허용오차 밴드 + 0 기준선
    ax.axvspan(-gt_tol_mm, gt_tol_mm, color=T["band"], zorder=0, lw=0)
    ax.axvline(0.0, color=T["ink2"], lw=1.0, zorder=1)

    # 실제로 값이 있는 계열만 그린다 (baseline 단독 / oracle 단독 실행도 지원)
    present = [k for k, _ in METHODS if any(k in e for _, _, _, e in rows)]
    if len(present) <= 1:
        off = {k: 0.0 for k in present}
    else:
        span = 0.18
        off = {k: (i - (len(present) - 1) / 2.0) * 2 * span for i, k in enumerate(present)}

    for key, name in METHODS:
        if key not in present:
            continue
        col = T["series"][[k for k, _ in METHODS].index(key)]
        xs, yy, lo, hi = [], [], [], []
        for (obj, ai, gt_mm, ests), yv in zip(rows, ys):
            if key not in ests or gt_mm is None:      # GT 미확정 축은 오차가 정의되지 않는다
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

    # 직접 라벨: 관심 계열(baseline) 하나에만 — 여러 계열에 붙이면 서로 겹친다
    label_key = PRIMARY if PRIMARY in present else (present[0] if present else None)
    for (obj, ai, gt_mm, ests), yv in zip(rows, ys):
        if label_key is None or label_key not in ests or gt_mm is None:
            continue
        d = ests[label_key][0] - gt_mm
        ax.annotate(f"{d:+.1f}", (d, yv + off[label_key]),
                    textcoords="offset points", xytext=(0, 9), ha="center",
                    fontsize=8.5, color=T["ink2"])

    # GT 미확정 축: 점을 못 찍으므로, 왜 비어 있는지 그 자리에 명시한다 (숨기지 않는다)
    for (obj, ai, gt_mm, ests), yv in zip(rows, ys):
        if gt_mm is not None:
            continue
        ax.annotate(S["gt_unknown"], (0.0, yv), xycoords=("data", "data"),
                    textcoords="offset points", xytext=(0, 0), ha="center", va="center",
                    fontsize=8.5, color=T["ink2"], style="italic",
                    bbox=dict(boxstyle="round,pad=0.28", fc=T["band"], ec=T["grid"], lw=0.8))

    ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=9.5, color=T["ink"])
    ax.invert_yaxis()
    ax.set_xlabel(S["xlabel"], fontsize=10, color=T["ink2"], labelpad=9)
    ax.set_title(S["title"], fontsize=13.5, color=T["ink"],
                 pad=(34 if len(present) > 1 else 14), loc="left")
    ax.grid(axis="x", color=T["grid"], lw=1.0, ls="-", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(T["grid"])
    ax.tick_params(colors=T["ink2"], length=0)
    # 계열이 하나면 범례 상자는 제목을 반복할 뿐이다 (제목이 무엇을 그리는지 말한다).
    if len(present) > 1:
        leg = ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=len(present),
                        frameon=False, fontsize=9.5, handletextpad=0.4, columnspacing=1.6)
        for t in leg.get_texts():
            t.set_color(T["ink"])

    # 표 뷰 (대비 WARN 완화 규칙: 라벨 또는 표가 반드시 존재해야 한다)
    axt.axis("off")
    head = [S["obj"], S["axis"], S["gt"]] + [f"{n}\n(mm / d mm)"
                                             for k, n in METHODS if k in present]
    body = []
    for obj, ai, gt_mm, ests in rows:
        r = [obj, f"L{ai+1}", f"{gt_mm:.1f}" if gt_mm is not None else S["unknown_short"]]
        for key, _ in [(k, n) for k, n in METHODS if k in present]:
            if key in ests:
                e = ests[key][0]
                # GT 미확정 축은 추정치는 보여주되 오차는 계산하지 않는다
                r.append(f"{e:.1f}  ({e - gt_mm:+.1f})" if gt_mm is not None else f"{e:.1f}  (—)")
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
    tcap = S["table"]
    if any(g is None for _, _, g, _ in rows):      # 왜 비었는지 표에도 명시
        tcap += "   ·   " + S["unknown_cap"]
    axt.set_title(tcap, fontsize=10, color=T["ink2"], loc="left", pad=2)

    fig.savefig(out_png, dpi=170, facecolor=T["surface"], bbox_inches="tight")
    plt.close(fig)


def overlay(capture_dir: Path, cams, views, mesh, fit_json, gt_desc_mm, obj, out_path, mode="light"):
    """같은 포즈에서 '추정 크기 CAD' vs '실측 크기 CAD' 실루엣을 겹쳐 그린다."""
    T = THEME[mode]
    bgr = lambda h: tuple(int(h[i:i+2], 16) for i in (5, 3, 1))   # '#rrggbb' -> BGR
    # 이 그림은 oracle(정답 CAD) 정합을 그린다 -> 차트의 oracle 계열색과 맞춘다
    C_EST = bgr(T["series"][[k for k, _ in METHODS].index("oracle")])
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
    cv2.putText(leg, f"oracle fit (GT CAD)  {est[0]:.1f} x {est[1]:.1f} x {est[2]:.1f} mm",
                (70, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.vconcat([strip, leg]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit_dir", type=Path, default=None,
                    help="oracle: Obj_Step3c 출력 폴더 (<obj>_cad_fit.json). 정답 CAD 기준 상한.")
    ap.add_argument("--baseline_dir", type=Path, default=None,
                    help="baseline: Obj_Step3_sam3d_scale --estimate_size 출력 root "
                         "(<obj>/<obj>_size.json). 실제 운용 방법.")
    ap.add_argument("--capture_dir", type=Path, required=True)
    ap.add_argument("--mask_dir", type=Path, required=True)
    ap.add_argument("--cad", action="append", metavar="peg=model.glb",
                    help="정답 CAD. oracle 및 gt_overlay 그림에 필요. baseline 만 볼 땐 생략 가능.")
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

    if not args.fit_dir and not args.baseline_dir:
        raise SystemExit("--fit_dir (oracle) 또는 --baseline_dir (baseline) 중 최소 하나가 필요합니다")
    cad = s3c.parse_cad(args.cad) if args.cad else {}
    gt = parse_gt(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.from_report:
        report = json.loads((args.out_dir / "gt_comparison.json").read_text())
        rows = []
        for obj, r in report.items():
            pm = r.get("oracle_mask_pm1px") or r.get("silhouette_mask_pm1px") or {}
            lo = pm.get("erode1px")
            hi = pm.get("dilate1px")
            for ai in range(3):
                ests = {}
                for key, _ in METHODS:
                    if key not in r["estimates_mm_sorted_desc"]:
                        continue
                    ests[key] = (float(r["estimates_mm_sorted_desc"][key][ai]),
                                 lo[ai] if (key == "oracle" and lo) else None,
                                 hi[ai] if (key == "oracle" and hi) else None)
                g = r["gt_extents_mm_sorted_desc"][ai]
                rows.append((obj, ai, None if g is None else float(g), ests))
        tol = float(np.mean([report[o]["gt_tol_mm"] for o in report]))
        chart(rows, args.out_dir / "gt_vs_estimate.png", "light", tol)
        print(f"[SAVE] {args.out_dir / 'gt_vs_estimate.png'}")
        if args.dark:
            chart(rows, args.out_dir / "gt_vs_estimate_dark.png", "dark", tol)
            print(f"[SAVE] {args.out_dir / 'gt_vs_estimate_dark.png'}")
        return

    cams = s3c.discover_cams(args.capture_dir)

    rows, report = [], {}
    for obj in sorted(gt.keys()):                    # 비교 대상 = 실측이 있는 물체
        gt_mm = gt[obj]["extents_mm"]
        est, lo, hi = {}, None, None
        fit_json = None
        b_iou, b_shape_ok = None, None

        if args.fit_dir:                             # oracle (정답 CAD)
            p = args.fit_dir / f"{obj}_cad_fit.json"
            if p.exists():
                fit_json = json.loads(p.read_text())
                est["oracle"] = fit_json["extents_mm_sorted_desc"]
                lo = fit_json.get("extents_mm_mask_erode1px")
                hi = fit_json.get("extents_mm_mask_dilate1px")
            else:
                print(f"[WARN] {obj}: oracle 결과 없음 ({p})")

        if args.baseline_dir:                        # baseline (SAM3D 단서 CAD)
            p = find_baseline_json(args.baseline_dir, obj)
            if p is not None:
                bj = json.loads(p.read_text())
                est["baseline"] = bj["extents_mm_sorted_desc"]
                b_iou = bj.get("mean_iou")
                b_shape_ok = bj.get("shape_ok_by_iou")
            else:
                print(f"[WARN] {obj}: baseline 결과 없음 ({args.baseline_dir} 안에 *_size.json)")

        if not est:
            print(f"[WARN] {obj}: 어떤 추정 결과도 없어 건너뜁니다")
            continue

        gt_str = " x ".join(f"{g:.1f}" if g is not None else "미확정" for g in gt_mm)
        n_unk = sum(1 for g in gt_mm if g is None)
        print(f"\n=== {obj} (GT {gt_str} mm, source={gt[obj]['source']}) ===")
        if n_unk:
            note = gt[obj].get("note")
            print(f"  [GT] 미확정 축 {n_unk}개 — 오차 계산에서 제외"
                  f"{(': ' + note) if note else ''}")
        for key, name in METHODS:
            if key not in est:
                continue
            e = [float(x) for x in est[key]]
            d = axis_errors(e, gt_mm)
            mae = mean_abs_err(d)
            dstr = " ".join(f"{x:+5.1f}" if x is not None else "    —" for x in d)
            pct = [abs(x / g) * 100 for x, g in zip(d, gt_mm) if x is not None and g]
            maestr = (f"{mae:5.2f} mm  ({np.mean(pct):4.1f}%)" if mae is not None
                      else "  —   (확정 축 없음)")
            tail = ""
            if key == "baseline" and b_iou is not None:
                tail = f"   IoU {b_iou:.3f}{'' if b_shape_ok else '  <- 형상 의심'}"
            print(f"  {name:<30} {e[0]:6.1f} {e[1]:6.1f} {e[2]:6.1f}   "
                  f"Δ {dstr}   |Δ|mean {maestr}{tail}")

        for ai in range(3):
            ests = {}
            for key, _ in METHODS:
                if key not in est:
                    continue
                l = lo[ai] if (key == "oracle" and lo) else None
                h = hi[ai] if (key == "oracle" and hi) else None
                ests[key] = (float(est[key][ai]), l, h)
            g = gt_mm[ai]
            rows.append((obj, ai, None if g is None else float(g), ests))

        # gt_overlay 는 '실측 크기 CAD vs 추정 크기 CAD' 그림이라 GT 3축이 모두 확정이어야
        # 만들 수 있다. 미확정 축을 추정치로 채우면 GT 가 맞는 것처럼 보이는 가짜 그림이 된다.
        if fit_json is not None and obj in cad and n_unk:
            print(f"  [SKIP] gt_overlay_{obj}.jpg — GT 미확정 축이 있어 실측 크기 CAD 를 "
                  f"만들 수 없습니다")
        elif fit_json is not None and obj in cad:
            mesh = trimesh.load(str(cad[obj]), force="mesh")
            views, _ = s3c.build_views(args.capture_dir, args.mask_dir, obj, cams)
            overlay(args.capture_dir, cams, views, mesh, fit_json, gt_mm, obj,
                    args.out_dir / f"gt_overlay_{obj}.jpg")
            print(f"  [SAVE] {args.out_dir / f'gt_overlay_{obj}.jpg'}")

        report[obj] = {
            "gt_extents_mm_sorted_desc": gt_mm,
            "gt_source": gt[obj]["source"],
            "gt_tol_mm": gt[obj]["tol_mm"],
            "gt_note": gt[obj].get("note"),
            "gt_undetermined_axes": [i for i, g in enumerate(gt_mm) if g is None],
            "estimates_mm_sorted_desc": {k: [float(x) for x in v] for k, v in est.items()},
            "errors_mm": {k: axis_errors([float(x) for x in v], gt_mm)
                          for k, v in est.items()},
            "mean_abs_error_mm": {k: mean_abs_err(axis_errors([float(x) for x in v], gt_mm))
                                  for k, v in est.items()},
            "oracle_mask_pm1px": {"erode1px": lo, "dilate1px": hi},
            "baseline_mean_iou": b_iou,
            "baseline_shape_ok_by_iou": b_shape_ok,
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
            r = [obj, f"L{ai+1}", f"{gt_mm_v:.3f}" if gt_mm_v is not None else ""]
            for key, _ in METHODS:
                if key in ests:
                    # GT 미확정 축: 추정치는 남기고 오차 칸은 비운다
                    r += [f"{ests[key][0]:.3f}",
                          f"{ests[key][0] - gt_mm_v:+.3f}" if gt_mm_v is not None else ""]
                else:
                    r += ["", ""]
            w.writerow(r)
    print(f"[SAVE] {args.out_dir / 'gt_vs_estimate.csv'}")

    (args.out_dir / "gt_comparison.json").write_text(json.dumps(report, indent=2))
    print(f"[SAVE] {args.out_dir / 'gt_comparison.json'}")

    print("\n[SUMMARY] mean |error| vs ruler  (GT 미확정 축 제외)")
    for key, name in METHODS:
        vals = [v for o in report
                for v in [report[o]["mean_abs_error_mm"].get(key)] if v is not None]
        if vals:
            print(f"  {name:<28} {np.mean(vals):5.2f} mm   ({len(vals)}개 물체)")


if __name__ == "__main__":
    main()
