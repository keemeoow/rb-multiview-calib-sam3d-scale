#!/usr/bin/env python3
"""평가 2 — 멀티뷰 Real-to-Sim silhouette 정합 정확도.

카메라별로 sim 마스크를 재렌더링해 IoU 와 contour distance 를 낸다.
SAM3D source view 는 그 뷰의 RGB+mask 로 mesh 를 만들었으므로 거의 자기 자신에 맞는다.
따라서 **메인 지표는 source view 를 제외한 cross-view 평균**이다.

주의: IoU 는 결과 JSON 의 per_view_iou(저장값)를 메인으로 쓰고, 재렌더링 IoU 는
재현 검증용으로 함께 저장한다 (GLB float32 왕복 때문에 미세 차이가 날 수 있다).
contour distance 는 저장된 값이 없어 재렌더링에서만 나온다.
"""
from __future__ import annotations

import numpy as np

import eval_common as ec


def evaluate(cfg: dict):
    """반환: (cam_rows, obj_rows, skipped)."""
    cams_cfg = cfg["cameras"]
    cam_rows, obj_rows, skipped = [], [], []

    for obj in cfg["objects"]:
        views = ec.load_views(obj["capture_dir"], obj["mask_dir"], cams_cfg)
        if not views:
            skipped.append(dict(object_name=obj["name"], method="*", stage="silhouette",
                                reason=f"사용 가능한 뷰 없음 ({obj['capture_dir']}, {obj['mask_dir']})"))
            print(f"  [SKIP] {obj['name']}: 뷰 없음")
            continue
        src = obj.get("source_camera")

        for method in ec.METHODS:
            fit, err = ec.load_fit(obj, method)
            if fit is None:
                skipped.append(dict(object_name=obj["name"], method=method,
                                    stage="silhouette", reason=err))
                print(f"  [SKIP] {obj['name']}/{method}: {err}")
                continue
            stored = (fit[5].get("per_view_iou") or [])
            # 재현 검증: 엔진과 같은 슈퍼샘플 조건으로 IoU 를 다시 계산해 저장값과 대조
            eng = ec.engine_iou(fit, [v["view"] for v in views.values()])

            per = {}
            for i, (cid, v) in enumerate(views.items()):
                real = v["mask"]
                sim = ec.render_sim_mask(fit, v["view"])          # ss=1 (contour/면적용)
                iou_re = float(eng[i])
                # 저장된 per_view_iou 는 cam0,cam1,cam2 순 (파이프라인이 정렬된 glob 순으로 뷰 구성)
                iou_stored = float(stored[i]) if i < len(stored) else None
                d_px, d_norm, d_pct = ec.contour_distance(real, sim)
                w, h = ec.bbox_wh(real)
                per[cid] = dict(iou=iou_stored if iou_stored is not None else iou_re,
                                iou_recomputed=iou_re, iou_stored=iou_stored,
                                d_px=d_px, d_norm=d_norm, d_pct=d_pct)
                cam_rows.append(dict(
                    object_name=obj["name"], method=method, camera_id=cid,
                    is_source_view=(cid == src),
                    silhouette_iou=per[cid]["iou"],
                    silhouette_iou_recomputed=iou_re,
                    silhouette_iou_stored=iou_stored,
                    iou_reproduction_delta=(None if iou_stored is None
                                            else abs(iou_re - iou_stored)),
                    contour_distance_px=d_px,
                    normalized_contour_distance=d_norm,
                    normalized_contour_distance_percent=d_pct,
                    real_mask_area_px=int(real.sum()),
                    sim_mask_area_px=int(sim.sum()),
                    real_bbox_width_px=w, real_bbox_height_px=h,
                    image_path=(str(v["rgb_path"].relative_to(ec._ROOT))
                                if v["rgb_path"] else None),
                    real_mask_path=str(v["mask_path"].relative_to(ec._ROOT)),
                    sim_mask_path=None,   # 메모리 상 렌더 (파일로 저장하지 않음)
                ))

            cross = [c for c in per if c != src]
            f = lambda key, ids: ec.mean_std([per[c][key] for c in ids])[0]
            obj_rows.append(dict(
                object_name=obj["name"], method=method, source_camera=src,
                source_view_iou=(per[src]["iou"] if src in per else None),
                cross_view_iou=f("iou", cross),
                cross_view_contour_distance_px=f("d_px", cross),
                cross_view_normalized_contour_distance=f("d_norm", cross),
                cross_view_normalized_contour_distance_percent=f("d_pct", cross),
                source_view_normalized_contour_distance_percent=(
                    per[src]["d_pct"] if src in per else None),
                n_cross_views=len(cross),
                per_camera_iou={c: per[c]["iou"] for c in per},
                max_iou_reproduction_delta=max(
                    [abs(per[c]["iou_recomputed"] - per[c]["iou_stored"])
                     for c in per if per[c]["iou_stored"] is not None] or [np.nan]),
            ))
            r = obj_rows[-1]
            print(f"  {obj['name']:8s} {method:14s} src({src})_IoU="
                  f"{r['source_view_iou']:.3f}  cross_IoU={r['cross_view_iou']:.3f}  "
                  f"cross_Dcontour={r['cross_view_normalized_contour_distance_percent']:.2f}%  "
                  f"(재현Δ≤{r['max_iou_reproduction_delta']:.4f})")
    return cam_rows, obj_rows, skipped


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    a = ap.parse_args()
    evaluate(ec.load_config(a.config))
