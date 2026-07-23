#!/usr/bin/env python3
"""평가 2 — 멀티뷰 Real-to-Sim silhouette 정합 정확도.

카메라별로 sim 마스크를 재렌더링해 IoU 와 contour distance 를 낸다.
SAM3D source view 는 그 뷰의 RGB+mask 로 mesh 를 만들었으므로 거의 자기 자신에 맞는다.
따라서 **메인 지표는 source view 를 제외한 cross-view 평균**이다.

주의: IoU 는 결과 JSON 의 per_view_iou(저장값)를 쓴다. 저장값이 없을 때만 엔진과 동일한
슈퍼샘플 조건으로 재계산한다. contour distance 는 저장된 값이 없어 재렌더링에서만 나온다.
"""
from __future__ import annotations

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
                sim = ec.render_sim_mask(fit, v["view"])          # ss=1 (contour 용)
                # 저장된 per_view_iou 는 cam0,cam1,cam2 순 (파이프라인이 정렬된 glob 순으로 뷰 구성).
                # 저장값이 없을 때만 엔진과 동일 조건으로 재계산한 값을 쓴다.
                iou = float(stored[i]) if i < len(stored) else float(eng[i])
                d_px, d_norm, d_pct = ec.contour_distance(real, sim)
                per[cid] = dict(iou=iou, d_px=d_px, d_norm=d_norm, d_pct=d_pct)
                cam_rows.append(dict(
                    object_name=obj["name"], method=method, camera_id=cid,
                    is_source_view=(cid == src),
                    silhouette_iou=iou,
                    normalized_contour_distance=d_norm,
                    normalized_contour_distance_percent=d_pct,
                    image_path=(str(v["rgb_path"].relative_to(ec._ROOT))
                                if v["rgb_path"] else None),
                    real_mask_path=str(v["mask_path"].relative_to(ec._ROOT)),
                ))

            # 보고 지표는 cross-view 평균뿐이다 (source view 는 그 뷰로 mesh 를 만들었으므로 제외).
            cross = [c for c in per if c != src]
            f = lambda key: ec.mean_std([per[c][key] for c in cross])[0]
            obj_rows.append(dict(
                object_name=obj["name"], method=method, source_camera=src,
                cross_view_iou=f("iou"),
                cross_view_normalized_contour_distance=f("d_norm"),
                cross_view_normalized_contour_distance_percent=f("d_pct"),
                n_cross_views=len(cross),
                per_camera_iou={c: per[c]["iou"] for c in cross},
            ))
            r = obj_rows[-1]
            print(f"  {obj['name']:8s} {method:14s} "
                  f"cross_IoU={r['cross_view_iou']:.3f}  "
                  f"cross_Dcontour={r['cross_view_normalized_contour_distance_percent']:.2f}%")
    return cam_rows, obj_rows, skipped


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    a = ap.parse_args()
    evaluate(ec.load_config(a.config))
