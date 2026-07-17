#!/usr/bin/env python3
"""물체별 크기 표 — GT / SAM3D(baseline) / Oracle(GT CAD) 를 한 표에 모은다.

물체마다 표 1개. 축은 파이프라인 규약대로 내림차순 rank (L1>=L2>=L3) 이며
L1=최장, L3=최단이다. 정답 CAD 가 없는 물체는 Oracle 행을 'CAD 없음' 으로 표기한다.

  python evaluation/make_dimension_tables.py \
    --config evaluation/evaluation_config.yaml --output evaluation/results
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_common as ec  # noqa: E402

AX = ["L1", "L2", "L3"]
AXK = ["L", "W", "H"]     # CSV 열 이름 규약 (L1->L, L2->W, L3->H)


def build(cfg, obj: pd.DataFrame):
    """반환: (per-object rows, 긴 형식 DataFrame)."""
    long = []
    for o in cfg["objects"]:
        n = o["name"]
        gt = [None if g is None else float(g) for g in o["gt_mm"]]
        for i, a in enumerate(AX):
            long.append(dict(object_name=n, display_name=o.get("display_name", n),
                             axis=a, source="GT", value_mm=gt[i], error_mm=None,
                             note=o.get("gt_source")))
        for meth, label in (("baseline_sam3d", "SAM3D"), ("oracle_cad", "Oracle")):
            r = obj[(obj.object_name == n) & (obj.method == meth)]
            if r.empty:
                for i, a in enumerate(AX):
                    long.append(dict(object_name=n, display_name=o.get("display_name", n),
                                     axis=a, source=label, value_mm=None, error_mm=None,
                                     note="정답 CAD 없음" if meth == "oracle_cad" else "결과 없음"))
                continue
            r = r.iloc[0]
            for i, (a, k) in enumerate(zip(AX, AXK)):
                v = r[f"estimated_{k}_mm"]
                e = r[f"abs_error_{k}_mm"]
                long.append(dict(object_name=n, display_name=r.display_name, axis=a,
                                 source=label, value_mm=None if pd.isna(v) else float(v),
                                 error_mm=None if pd.isna(e) else float(e),
                                 note=r.metric_exclusion_reason if pd.notna(
                                     r.get("metric_exclusion_reason")) else None))
    return pd.DataFrame(long)


def fmt(v, p=2):
    return "—" if v is None or pd.isna(v) else f"{v:.{p}f}"


def md_tables(cfg, obj: pd.DataFrame, long: pd.DataFrame) -> str:
    out = ["# 물체별 크기 표 — GT / SAM3D / Oracle\n",
           "축은 내림차순 rank (L1≥L2≥L3). 단위 mm. 괄호 안은 GT 대비 |오차|.\n"]
    for o in cfg["objects"]:
        n = o["name"]
        disp = o.get("display_name", n)
        b = obj[(obj.object_name == n) & (obj.method == "baseline_sam3d")]
        orc = obj[(obj.object_name == n) & (obj.method == "oracle_cad")]
        gt = [None if g is None else float(g) for g in o["gt_mm"]]

        out.append(f"\n## {disp}\n")
        out.append("| | L1 | L2 | L3 | E_dim |")
        out.append("|---|---|---|---|---|")
        out.append(f"| **GT ({o.get('gt_source')})** | {fmt(gt[0],1)} | {fmt(gt[1],1)} | "
                   f"{fmt(gt[2],1)} | — |")

        def row(df, label):
            if df.empty:
                return f"| **{label}** | — | — | — | CAD 없음 |"
            r = df.iloc[0]
            cells = []
            for k in AXK:
                v, e = r[f"estimated_{k}_mm"], r[f"abs_error_{k}_mm"]
                cells.append(fmt(v) if pd.isna(e) else f"{fmt(v)}  ({fmt(e)})")
            ed = fmt(r.mean_dimension_error_mm)
            return f"| **{label}** | {cells[0]} | {cells[1]} | {cells[2]} | {ed} |"

        out.append(row(b, "SAM3D (baseline)"))
        out.append(row(orc, "Oracle (GT CAD)"))
        if orc.empty:
            out.append(f"\n> Oracle 없음: **{disp} 는 정답 CAD 가 없다** (baseline 전용).")
        meta = o.get("shape_class")
        out.append(f"\n- 형상: `{meta}` · SAM3D source view: `{o.get('source_camera')}`")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()

    cfg = ec.load_config(a.config)
    obj = pd.read_csv(a.output / "csv" / "evaluation_per_object.csv")
    long = build(cfg, obj)

    (a.output / "csv").mkdir(parents=True, exist_ok=True)
    long.to_csv(a.output / "csv" / "dimension_tables_long.csv", index=False)
    print(f"  [SAVE] csv/dimension_tables_long.csv")

    # 물체별 wide CSV (물체당 파일 1개)
    for n, g in long.groupby("object_name"):
        w = g.pivot_table(index="source", columns="axis", values="value_mm", aggfunc="first")
        w = w.reindex(index=["GT", "SAM3D", "Oracle"], columns=AX)
        d = a.output / "per_object" / n
        d.mkdir(parents=True, exist_ok=True)
        w.to_csv(d / "dimensions.csv")
    print(f"  [SAVE] per_object/<object>/dimensions.csv  ({long.object_name.nunique()}개)")

    md = md_tables(cfg, obj, long)
    (a.output / "dimension_tables.md").write_text(md)
    print(f"  [SAVE] dimension_tables.md")
    print("\n" + md)


if __name__ == "__main__":
    main()
