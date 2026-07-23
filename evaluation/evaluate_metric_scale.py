#!/usr/bin/env python3
"""평가 1 — 실측 대비 metric scale / mesh 크기 정확도.

결과 JSON 의 extents_mm_sorted_desc 를 GT 와 비교해 축별 |오차|,
Mean Dimension Error(E_dim), Mean Relative Dimension Error(E_rel) 를 낸다.
GT 미확정 축(null)은 오차 계산에서 제외하고 사용 축 수를 기록한다.
"""
from __future__ import annotations

import eval_common as ec


def evaluate(cfg: dict):
    """반환: (rows, skipped). rows 는 물체×방법."""
    rows, skipped = [], []
    for obj in cfg["objects"]:
        gt = [None if g is None else float(g) for g in obj["gt_mm"]]
        for method in ec.METHODS:
            fit, err = ec.load_fit(obj, method)
            if fit is None:
                skipped.append(dict(object_name=obj["name"], method=method,
                                    stage="metric_scale", reason=err))
                print(f"  [SKIP] {obj['name']}/{method}: {err}")
                continue
            meta = fit[5]
            est = [float(x) for x in meta["extents"]]
            m = ec.match_axes(est, gt)
            abs_e, mean_e, mean_rel, n_axes = ec.dim_errors(m["est_matched"], gt)

            # GT 나 단서 mesh 가 신뢰 불가하면 오차를 계산하지 않는다.
            # 행은 남겨 추정치와 제외 사유를 보이게 한다 (조용히 빼면 cherry-picking).
            # 설정값: true = 그 물체의 모든 방법 제외, 리스트 = 해당 방법만 제외.
            ex = obj.get("exclude_from_metric_scale", False)
            excluded = (method in ex) if isinstance(ex, (list, tuple)) else bool(ex)
            if excluded:
                abs_e, mean_e, mean_rel, n_axes = [None] * 3, None, None, 0
                skipped.append(dict(object_name=obj["name"], method=method,
                                    stage="metric_scale",
                                    reason=" ".join(str(obj.get("metric_exclusion_reason", "")).split())))
            # 보고 지표는 축별 |오차| / E_dim / E_rel 셋뿐이다. GT·추정치는 그 오차를
            # 읽기 위한 최소 문맥으로만 남기고, 나머지 진단용 필드는 저장하지 않는다.
            rows.append(dict(
                object_name=obj["name"],
                display_name=obj.get("display_name", obj["name"]),
                shape_class=obj.get("shape_class"),
                method=method,
                source_camera=obj.get("source_camera"),
                gt_source=obj.get("gt_source"),
                gt_L_mm=gt[0], gt_W_mm=gt[1], gt_H_mm=gt[2],
                estimated_L_mm=m["est_matched"][0],
                estimated_W_mm=m["est_matched"][1],
                estimated_H_mm=m["est_matched"][2],
                abs_error_L_mm=abs_e[0], abs_error_W_mm=abs_e[1], abs_error_H_mm=abs_e[2],
                mean_dimension_error_mm=mean_e,
                mean_relative_dimension_error_percent=mean_rel,
                gt_axes_used=n_axes,
                excluded_from_metric_scale=excluded,
                metric_exclusion_reason=(" ".join(str(obj["metric_exclusion_reason"]).split())
                                         if excluded else None),
                result_file_path=str(meta["json_path"].relative_to(ec._ROOT)),
            ))
            fmt = lambda x: "   —  " if x is None else f"{x:6.2f}"
            tail = ("  <- 크기평가 제외 (GT/CAD 불량)" if excluded else "")
            print(f"  {obj['name']:8s} {method:14s} "
                  f"|e|=[{fmt(abs_e[0])},{fmt(abs_e[1])},{fmt(abs_e[2])}] mm  "
                  f"E_dim={fmt(mean_e)} mm  E_rel="
                  f"{'  —  ' if mean_rel is None else f'{mean_rel:5.2f}%'}{tail}")
    return rows, skipped


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    a = ap.parse_args()
    evaluate(ec.load_config(a.config))
