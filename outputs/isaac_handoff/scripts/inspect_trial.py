#!/usr/bin/env python3
"""
inspect_trial.py — handoff package 의 trial 하나를 눈으로 확인한다.

package 안에서 단독 실행된다 (repo 모듈 import 없음).
의존성: numpy, pyyaml, trimesh, opencv-python. **Isaac Sim 없이 돌아간다.**

  python scripts/inspect_trial.py --trial trial_0001
  python scripts/inspect_trial.py --trial trial_0001 --out /tmp/check.png

하는 일
  1. trial 의 pose/mesh/camera 를 읽어 수치를 출력한다.
  2. 카메라마다 mesh 를 pose 대로 재투영해 실제 RGB 위에 얹고,
     SAM mask 외곽선(초록)과 비교한 overlay 를 만든다.
  3. IoU 를 다시 계산해 pose json 에 기록된 값과 일치하는지 본다.

Isaac Sim 에서 객체가 이상한 위치/크기로 뜬다면, 먼저 이걸 돌려라.
  - 여기 overlay 가 맞다  → package 는 정상. Isaac import (단위/up-axis/카메라) 문제다.
  - 여기 overlay 가 틀리다 → package 가 잘못됐다. 사용자에게 재-export 를 요청하라.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh
import yaml


def load_mesh(path: Path) -> trimesh.Trimesh:
    m = trimesh.load(str(path), force="mesh")
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(m.geometry.values()))
    return m


def render_silhouette(mesh, K, T_cam_obj, hw):
    H, W = hw
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int32)
    Vc = (T_cam_obj[:3, :3] @ V.T).T + T_cam_obj[:3, 3]
    z = Vc[:, 2]
    zz = np.where(z > 1e-6, z, 1e-6)
    u = K[0, 0] * Vc[:, 0] / zz + K[0, 2]
    v = K[1, 1] * Vc[:, 1] / zz + K[1, 2]
    u[z <= 1e-6], v[z <= 1e-6] = -1e6, -1e6
    pts = np.stack([u, v], axis=1).astype(np.int32)
    sil = np.zeros((H, W), np.uint8)
    for f in F:
        cv2.fillConvexPoly(sil, pts[f], 255)
    return sil > 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", type=Path, default=Path("."))
    ap.add_argument("--trial", required=True)
    ap.add_argument("--out", type=Path, default=None,
                    help="overlay 저장 경로 (기본: <package>/trials/<trial>/inspect_overlay.png)")
    args = ap.parse_args()

    root = args.package.resolve()
    manifest = yaml.safe_load((root / "manifest.yaml").read_text())
    if args.trial not in (manifest.get("trials") or {}):
        print(f"[FATAL] trial '{args.trial}' 없음. 있는 trial: "
              f"{sorted(manifest.get('trials') or {})}", file=sys.stderr)
        return 2

    cams = yaml.safe_load((root / "calibration/cameras.yaml").read_text())["cameras"]
    trial = yaml.safe_load((root / manifest["trials"][args.trial]["yaml"]).read_text())

    print(f"trial       : {args.trial}")
    print(f"world frame : {trial['world_frame']}")
    print(f"cameras     : {sorted(trial['cameras'])}")
    print(f"objects     : {sorted(trial['objects'])}")

    tiles = []
    worst = 1.0
    for oid, oentry in trial["objects"].items():
        pose = json.loads((root / oentry["pose"]).read_text())
        obj = yaml.safe_load((root / manifest["objects"][oid]["yaml"]).read_text())
        mesh = load_mesh(root / obj["assets"]["mesh_scaled"])

        T_wo = np.asarray(pose["T_world_object"], dtype=np.float64)
        ext = np.asarray(obj["size"]["estimated_extents_m"]) * 1000
        meas = obj["size"].get("measured_extents_m")
        print(f"\n--- {oid} ({obj.get('name')}) ---")
        print(f"  pose_source     : {pose['pose_source']}")
        print(f"  T_world_object t: {np.round(T_wo[:3, 3], 4).tolist()} m")
        print(f"  quat_wxyz       : {[round(v, 4) for v in pose['quat_wxyz']]}")
        print(f"  추정 치수       : {np.round(np.sort(ext)[::-1], 2).tolist()} mm")
        if meas:
            mm = np.sort(np.asarray(meas) * 1000)[::-1]
            print(f"  실측 치수       : {np.round(mm, 2).tolist()} mm")
            print(f"  오차            : {np.round(np.sort(ext)[::-1] - mm, 2).tolist()} mm")
        print(f"  mass_kg         : {obj.get('mass_kg')}"
              f"{'   <-- 미측정' if obj.get('mass_kg') is None else ''}")
        print(f"  mesh vertices   : {len(mesh.vertices)}")

        for cid in sorted(oentry.get("masks") or {}):
            c = cams[cid]
            K = np.asarray(c["K"], dtype=np.float64)
            T_cam_obj = np.asarray(pose["T_cam_object"][cid], dtype=np.float64)
            rgb = cv2.imread(str(root / trial["cameras"][cid]["rgb"]))
            mask = cv2.imread(str(root / oentry["masks"][cid]), 0) > 0
            sil = render_silhouette(mesh, K, T_cam_obj, (c["height"], c["width"]))
            union = (sil | mask).sum()
            v = float((sil & mask).sum()) / union if union else 0.0
            rec = (pose.get("reprojection_iou") or {}).get(cid)
            worst = min(worst, v)
            drift = f"  (기록값 {rec:.4f})" if rec is not None else ""
            print(f"  {cid}: 재투영 IoU = {v:.4f}{drift}")

            img = rgb.copy()
            cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cs, -1, (0, 255, 0), 2)          # 실제 SAM mask
            cs2, _ = cv2.findContours(sil.astype(np.uint8), cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cs2, -1, (0, 0, 255), 2)          # mesh + pose 재투영
            cv2.putText(img, f"{oid} {cid} IoU={v:.3f}", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            tiles.append(img)

    if tiles:
        h = max(t.shape[0] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 6, cv2.BORDER_CONSTANT,
                                    value=(255, 255, 255)) for t in tiles]
        strip = cv2.hconcat(tiles)
        legend = np.full((40, strip.shape[1], 3), 255, np.uint8)
        cv2.line(legend, (16, 20), (56, 20), (0, 255, 0), 3)
        cv2.putText(legend, "real SAM mask", (64, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        cv2.line(legend, (220, 20), (260, 20), (0, 0, 255), 3)
        cv2.putText(legend, "handoff mesh + pose (reprojected)", (268, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        out = args.out or (root / f"trials/{args.trial}/inspect_overlay.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), cv2.vconcat([strip, legend]))
        print(f"\n[SAVE] overlay -> {out}")

    gate = (manifest.get("quality_gates") or {}).get("min_reprojection_iou")
    if gate is not None and worst < gate:
        print(f"\n[FAIL] 최저 IoU {worst:.3f} < 임계 {gate} — package 정합이 깨졌다.")
        return 1
    print(f"\n[OK] 최저 IoU {worst:.3f}. 초록(실제 mask)과 빨강(재투영)이 겹치면 정상이다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
