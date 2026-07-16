#!/usr/bin/env python3
"""
export_isaac_handoff.py — 사용자 파이프라인 결과를 Isaac Sim 담당자용 package 로 내보낸다.

  python scripts/export_isaac_handoff.py \
    --config configs/evaluation.yaml \
    --trials data/trials \
    --output outputs/isaac_handoff

--trials 는 선택이다. 없으면 config 의 trials: 항목만 내보낸다.

[무엇을 모으는가]
  Obj_Step1  capture_obj/  cam{i}_rgb.png, cam{i}_depth.png, cam{i}_K.txt,
                           cam{i}_T_cam_to_world.txt, calib_info.json
  Obj_Step2  masks/<obj>/cam{i}_mask.png
  Obj_Step3c outputs_cad_fit/<obj>_cad_fit.json, <obj>_cad_scaled.glb
  Obj_Step5  <fp_dir>/<obj>/T_C0_obj.json          (있을 때만)
  config     measured size / mass / 실험 파라미터

[pose 를 어떻게 만드는가 — 두 경로가 다르다]
  (A) FoundationPose (fp_dir 지정 시)
      Obj_Step5 는 이미 <obj>_cad_scaled.glb (원점 중심, 실척) 를 mesh 로 넣고 돌린다.
      따라서 T_cam_obj 가 곧 내보낼 mesh 의 pose 다. 추가 보정 없음.
        T_world_obj = T_world_cam @ T_cam_obj

  (B) CAD silhouette fit (기본)
      Obj_Step3c 의 T_world_cad 는 **원본 CAD** 좌표에 대한 변환이다:
        p_world = R @ (s * p_cad) + t
      그런데 내보내는 mesh 는 save_scaled_cad() 가 s 배 스케일 후 AABB 중심을
      원점으로 옮긴 것이다:  v = s * p_cad - c   (c = centroid(s * p_cad))
      그러므로 내보낸 mesh 의 pose 는 translation 이 다르다:
        p_world = R @ v + (R @ c + t)
        T_world_obj = [R | R @ c + t]
      이 보정을 빼먹으면 객체가 c 만큼 (Hole 기준 0.78 mm) 밀린 채 stage 에 올라간다.

두 경로 모두 **내보낸 mesh + 내보낸 pose 를 실제 mask 에 재투영해 IoU 를 재계산**하고
manifest 에 기록한다. quality_gates.min_reprojection_iou 미만이면 validator 가 잡는다.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import trimesh
import yaml

REPO = Path(__file__).resolve().parent.parent

# OpenCV camera (x right, y down, z forward) -> USD/Isaac camera (x right, y up, -z forward)
# T_world_cam_isaac = T_world_cam_opencv @ CV_TO_USD
CV_TO_USD = np.diag([1.0, -1.0, -1.0, 1.0])


# --------------------------------------------------------------------------- #
# 기본 유틸
# --------------------------------------------------------------------------- #

def log(msg: str) -> None:
    print(msg, flush=True)


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def R_to_quat_wxyz(R: np.ndarray) -> list[float]:
    """scalar-first. USD Gf.Quatd(w, Gf.Vec3d(x, y, z)) 와 같은 순서."""
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q)
    return [float(v) for v in q]


def load_mesh(path: Path) -> trimesh.Trimesh:
    m = trimesh.load(str(path), force="mesh")
    if isinstance(m, trimesh.Scene):
        geoms = [g for g in m.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise RuntimeError(f"mesh geometry 없음: {path}")
        m = trimesh.util.concatenate(tuple(geoms))
    return m


def render_silhouette(mesh: trimesh.Trimesh, K: np.ndarray, T_cam_obj: np.ndarray,
                      hw: tuple[int, int]) -> np.ndarray:
    """mesh 를 T_cam_obj 로 놓고 K 로 투영한 채움 실루엣 (검증용, z-buffer 없이 union)."""
    H, W = hw
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int32)
    Vc = (T_cam_obj[:3, :3] @ V.T).T + T_cam_obj[:3, 3]
    z = Vc[:, 2]
    if (z <= 1e-6).all():
        return np.zeros((H, W), dtype=bool)
    zz = np.where(z > 1e-6, z, 1e-6)
    u = K[0, 0] * Vc[:, 0] / zz + K[0, 2]
    v = K[1, 1] * Vc[:, 1] / zz + K[1, 2]
    bad = z <= 1e-6
    u[bad], v[bad] = -1e6, -1e6
    pts = np.stack([u, v], axis=1).astype(np.int32)
    sil = np.zeros((H, W), dtype=np.uint8)
    for f in F:
        cv2.fillConvexPoly(sil, pts[f], 255)
    return sil > 0


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int((a | b).sum())
    return float((a & b).sum()) / union if union else 0.0


def copy_file(src: Path, dst: Path, symlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if symlink:
        # package 를 옮겨도 깨지지 않도록 절대경로 symlink
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


class ExportError(Exception):
    """export 를 계속 진행하되, 해당 항목은 건너뛰고 report 에 남긴다."""


# --------------------------------------------------------------------------- #
# 입력 읽기
# --------------------------------------------------------------------------- #

def discover_cams(capture_dir: Path) -> list[str]:
    cams = sorted({p.name.split("_")[0] for p in capture_dir.glob("cam*_K.txt")})
    if not cams:
        raise ExportError(f"카메라를 찾지 못함 (cam*_K.txt 없음): {capture_dir}")
    return cams


def read_cameras(capture_dir: Path) -> dict:
    """cam{i}_K.txt + cam{i}_T_cam_to_world.txt + rgb 해상도 → 카메라 dict."""
    info_path = capture_dir / "calib_info.json"
    calib_info = json.loads(info_path.read_text()) if info_path.exists() else {}
    world_frame = calib_info.get("world_frame")

    cams = {}
    for cid in discover_cams(capture_dir):
        K = np.loadtxt(capture_dir / f"{cid}_K.txt", dtype=np.float64)
        T_path = capture_dir / f"{cid}_T_cam_to_world.txt"
        if not T_path.exists():
            raise ExportError(f"extrinsic 없음: {T_path}")
        T_world_cam = np.loadtxt(T_path, dtype=np.float64)   # camera -> world
        rgb = cv2.imread(str(capture_dir / f"{cid}_rgb.png"), cv2.IMREAD_COLOR)
        if rgb is None:
            raise ExportError(f"RGB 없음: {capture_dir / f'{cid}_rgb.png'}")
        H, W = rgb.shape[:2]

        # depth_scale: calib_info 의 카메라별 값이 있으면 그것을 쓴다 (RealSense 실측값)
        ds = None
        for c in calib_info.get("cameras", []):
            if f"cam{c.get('cam_id')}" == cid:
                ds = c.get("depth_scale_m_per_unit")
        cams[cid] = {
            "K": K, "T_world_cam": T_world_cam, "width": W, "height": H,
            "depth_scale_m_per_unit": ds,
            "serial": next((c.get("serial") for c in calib_info.get("cameras", [])
                            if f"cam{c.get('cam_id')}" == cid), None),
        }
    return {"cameras": cams, "world_frame": world_frame}


def pose_from_silhouette_fit(fit: dict, cad_path: Path) -> tuple[np.ndarray, dict]:
    """
    Obj_Step3c 의 (scale, T_world_cad) → **내보낸 scaled mesh** 의 T_world_obj.

    save_scaled_cad() 가 s 배 스케일 후 AABB 중심을 원점으로 옮겼으므로
    translation 에 R @ c 를 더해야 한다 (c = centroid of s*CAD).
    """
    s = float(fit["scale_cad_to_world"])
    T = np.asarray(fit["T_world_cad_4x4"], dtype=np.float64)
    R, t = T[:3, :3], T[:3, 3]

    raw = load_mesh(cad_path)
    scaled = raw.copy()
    scaled.apply_scale(s)
    c = np.asarray(scaled.bounding_box.centroid, dtype=np.float64)

    T_world_obj = np.eye(4)
    T_world_obj[:3, :3] = R
    T_world_obj[:3, 3] = R @ c + t
    meta = {
        "pose_source": "cad_silhouette_fit",
        "producer": "Obj_Step3c_fit_cad_silhouette.py",
        "scale_cad_to_world": s,
        "recenter_offset_m": [float(v) for v in c],
        "recenter_offset_mm": float(np.linalg.norm(c) * 1000.0),
        "note": ("T_world_cad 는 원본 CAD 기준이라 mesh 원점 이동분 R@c 를 더해 보정했다. "
                 "이 값이 곧 <obj>_cad_scaled.glb 의 world pose 다."),
    }
    return T_world_obj, meta


def pose_from_foundationpose(fp_json: Path, cams: dict) -> tuple[np.ndarray, dict]:
    """
    Obj_Step5 출력 → T_world_obj.
    Obj_Step5 는 mesh 로 이미 <obj>_cad_scaled.glb 를 쓰므로 recenter 보정이 필요 없다.
    """
    d = json.loads(fp_json.read_text())
    key = "T_cam_obj_4x4"
    if key not in d:
        raise ExportError(f"{fp_json}: '{key}' 없음 — Obj_Step5 출력이 맞는지 확인")
    T_cam_obj = np.asarray(d[key], dtype=np.float64).reshape(4, 4)
    cid = d.get("cam_id", "cam0")
    if cid not in cams:
        raise ExportError(f"{fp_json}: cam_id={cid} 가 capture 카메라에 없음")
    T_world_obj = cams[cid]["T_world_cam"] @ T_cam_obj
    meta = {
        "pose_source": "foundationpose",
        "producer": "Obj_Step5_foundationpose_register.py",
        "fp_camera": cid,
        "fp_iterations": d.get("iterations"),
        "yup_correction": d.get("yup_correction"),
        "note": "Obj_Step5 가 <obj>_cad_scaled.glb 를 그대로 mesh 로 썼으므로 추가 보정 없음.",
    }
    return T_world_obj, meta


# --------------------------------------------------------------------------- #
# trial 해석
# --------------------------------------------------------------------------- #

def discover_objects(mask_dir: Path, fit_dir: Path) -> list[str]:
    """
    mask_dir/<id>/ 와 fit_dir/<id>_cad_fit.json 이 **둘 다** 있는 id 만 객체로 본다.
    한쪽만 있으면 파이프라인이 덜 돌아간 것이므로 조용히 넣지 않는다 (호출부가 경고한다).
    """
    masks = {p.name for p in mask_dir.iterdir() if p.is_dir()} if mask_dir.is_dir() else set()
    fits = {p.name[: -len("_cad_fit.json")] for p in fit_dir.glob("*_cad_fit.json")} \
        if fit_dir.is_dir() else set()
    return sorted(masks & fits)


def resolve_trials(cfg: dict, trials_root: Path | None) -> list[dict]:
    trials = [dict(t) for t in (cfg.get("trials") or [])]
    known = {t["id"] for t in trials}

    if trials_root is not None:
        if not trials_root.exists():
            log(f"[WARN] --trials 경로가 없다: {trials_root} — config 의 trials 만 내보낸다")
        else:
            d = cfg.get("trial_discovery") or {}
            cap_sub = d.get("capture_subdir", "capture_obj")
            mask_sub = d.get("mask_subdir", "masks")
            fit_sub = d.get("fit_subdir", "outputs_cad_fit")
            fp_sub = d.get("fp_subdir", "fp_pose")
            for sub in sorted(p for p in trials_root.iterdir() if p.is_dir()):
                if sub.name in known:
                    continue
                cap, msk, fit = sub / cap_sub, sub / mask_sub, sub / fit_sub
                if not (cap.is_dir() and msk.is_dir() and fit.is_dir()):
                    log(f"[SKIP] {sub}: {cap_sub}/{mask_sub}/{fit_sub} layout 이 아니다")
                    continue
                fp = sub / fp_sub
                trials.append({
                    "id": sub.name,
                    "capture_dir": str(cap.relative_to(REPO)) if cap.is_relative_to(REPO) else str(cap),
                    "mask_dir": str(msk.relative_to(REPO)) if msk.is_relative_to(REPO) else str(msk),
                    "fit_dir": str(fit.relative_to(REPO)) if fit.is_relative_to(REPO) else str(fit),
                    "fp_dir": (str(fp.relative_to(REPO)) if fp.is_dir() and fp.is_relative_to(REPO)
                               else (str(fp) if fp.is_dir() else None)),
                    "objects": "auto",
                })
                log(f"[DISCOVER] trial {sub.name}")
    if not trials:
        raise SystemExit("내보낼 trial 이 없다. configs/evaluation.yaml 의 trials: 를 확인하라.")

    # objects: auto → mask_dir ∩ fit_dir 스캔. 객체를 추가해도 config 를 안 고쳐도 되게.
    for t in trials:
        if t.get("objects") in (None, "auto"):
            found = discover_objects(REPO / t["mask_dir"], REPO / t["fit_dir"])
            if not found:
                log(f"[WARN] trial {t['id']}: 객체를 못 찾았다 "
                    f"(mask_dir/<id>/ 와 fit_dir/<id>_cad_fit.json 이 둘 다 있어야 한다)")
            t["objects"] = found
            log(f"[DISCOVER] trial {t['id']}: objects={found}")
    return trials


# --------------------------------------------------------------------------- #
# export 본체
# --------------------------------------------------------------------------- #

def export(cfg: dict, cfg_path: Path, trials_root: Path | None, out: Path,
           symlink: bool) -> dict:
    conv = cfg["conventions"]
    gates = cfg.get("quality_gates") or {}
    min_iou = float(gates.get("min_reprojection_iou", 0.0))
    min_mask_px = int(gates.get("min_mask_pixels", 50))
    max_size_err = float(gates.get("max_size_error_mm", 5.0))
    scale_tol = float(gates.get("scale_tolerance", 0.2))

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    trials = resolve_trials(cfg, trials_root)
    obj_cfg = cfg["objects"]

    export_issues: list[str] = []
    manifest_objects: dict[str, dict] = {}
    manifest_trials: dict[str, dict] = {}
    cameras_global: dict | None = None
    world_frame_global: str | None = None
    # (obj -> 이 객체를 처음 만난 trial 의 fit) — objects/ 는 trial 간 공유 자산이다
    obj_fit_used: dict[str, dict] = {}
    size_rows: list[dict] = []

    for t in trials:
        tid = t["id"]
        cap = REPO / t["capture_dir"]
        mask_dir = REPO / t["mask_dir"]
        fit_dir = REPO / t["fit_dir"]
        fp_dir = (REPO / t["fp_dir"]) if t.get("fp_dir") else None
        log(f"\n=== trial {tid} ===")

        try:
            cam_info = read_cameras(cap)
        except ExportError as e:
            export_issues.append(f"trial {tid}: {e}")
            log(f"[ERROR] {e} — trial 건너뜀")
            continue
        cams = cam_info["cameras"]
        wf = cam_info["world_frame"]

        # 고정 카메라 3대는 모든 trial 에서 동일해야 한다. 다르면 재캘리브레이션된 것 —
        # 조용히 첫 trial 값으로 덮으면 나머지 trial 의 투영이 전부 틀어진다.
        if cameras_global is None:
            cameras_global, world_frame_global = cams, wf
            if wf and wf != conv.get("world_frame"):
                export_issues.append(
                    f"world_frame 불일치: calib_info.json='{wf}' vs config='{conv.get('world_frame')}'")
                log(f"[ERROR] world_frame 불일치: calib_info={wf}, config={conv.get('world_frame')}")
        else:
            for cid, c in cams.items():
                g = cameras_global.get(cid)
                if g is None:
                    export_issues.append(f"trial {tid}: 카메라 {cid} 가 첫 trial 에 없었다")
                    continue
                if not np.allclose(c["K"], g["K"], atol=1e-6) or \
                   not np.allclose(c["T_world_cam"], g["T_world_cam"], atol=1e-6):
                    export_issues.append(
                        f"trial {tid}: 카메라 {cid} 의 K 또는 extrinsic 이 첫 trial 과 다르다 "
                        f"(고정 카메라 가정 위반 — trial 별 calibration 이 필요)")

        # --- 카메라별 RGB/Depth 복사 ---
        trial_cams: dict[str, dict] = {}
        for cid in cams:
            rgb_src = cap / f"{cid}_rgb.png"
            dep_src = cap / f"{cid}_depth.png"
            if not rgb_src.exists() or not dep_src.exists():
                export_issues.append(f"trial {tid}/{cid}: rgb 또는 depth 파일 없음")
                continue
            rgb_rel = f"trials/{tid}/{cid}/rgb.png"
            dep_rel = f"trials/{tid}/{cid}/depth.png"
            copy_file(rgb_src, out / rgb_rel, symlink)
            copy_file(dep_src, out / dep_rel, symlink)
            trial_cams[cid] = {"rgb": rgb_rel, "depth": dep_rel}

        # --- 객체별 pose / mask ---
        trial_objs: dict[str, dict] = {}
        for oid in t["objects"]:
            # config 에 항목이 없어도 export 는 한다 (GT/질량만 빠진 상태로).
            # 그래야 객체를 추가하는 도중에도 파이프라인을 끝까지 돌려볼 수 있다.
            ocfg = obj_cfg.get(oid) or {}
            if oid not in obj_cfg:
                export_issues.append(
                    f"{oid}: configs/evaluation.yaml 의 objects: 에 항목이 없다 "
                    f"— 실측 치수/질량이 없어 Oracle·Pose-only 조건을 만들 수 없다")
                log(f"  [WARN] {oid}: config 항목 없음 (실측값 없이 export)")

            fit_key = ocfg.get("fit_key", oid)
            mask_key = ocfg.get("mask_key", oid)
            fit_json = fit_dir / f"{fit_key}_cad_fit.json"
            scaled_glb = fit_dir / f"{fit_key}_cad_scaled.glb"

            missing = [p for p in (fit_json, scaled_glb) if not p.exists()]
            if missing:
                for p in missing:
                    export_issues.append(f"trial {tid}/{oid}: 파일 없음 {p}")
                log(f"[ERROR] {tid}/{oid}: Obj_Step3c 출력 누락 — 건너뜀")
                continue

            fit = json.loads(fit_json.read_text())

            # 원본 CAD: config 의 cad: 가 우선, 없으면 fit json 이 기록한 경로.
            # (recenter 보정에 원본 CAD 의 AABB centroid 가 필요하다 — 아래 참고)
            cad_rel = ocfg.get("cad") or fit.get("cad")
            cad_raw = (REPO / cad_rel) if cad_rel else None
            if cad_raw is None or not cad_raw.exists():
                export_issues.append(
                    f"trial {tid}/{oid}: 원본 CAD 를 찾을 수 없다 (cad={cad_rel!r}). "
                    f"configs/evaluation.yaml 의 objects.{oid}.cad 에 경로를 적어라.")
                log(f"[ERROR] {tid}/{oid}: CAD 없음 ({cad_rel}) — 건너뜀")
                continue

            # YCB 처럼 이미 실척인 mesh 는 추정 scale 이 1.0 근처여야 한다.
            # 크게 벗어나면 mesh 단위(mm/cm) 나 마스크가 잘못된 것이다.
            expect = ocfg.get("expect_scale_near")
            if expect:
                s_est = float(fit["scale_cad_to_world"])
                rel = abs(s_est - float(expect)) / float(expect)
                if rel > scale_tol:
                    export_issues.append(
                        f"{oid}: 추정 scale {s_est:.4f} 가 기대값 {expect} 에서 {rel*100:.1f}% 벗어남 "
                        f"— mesh 단위(mm/cm/m)나 mask 를 확인하라")
                    log(f"  [WARN] {oid}: scale {s_est:.4f} vs 기대 {expect} ({rel*100:.1f}% 차이)")

            # pose: FoundationPose 가 있으면 그것을, 없으면 silhouette fit 을 쓴다
            fp_json = None
            if fp_dir is not None:
                for cand in (fp_dir / oid / "T_C0_obj.json", fp_dir / oid / "fp_pose" / "T_C0_obj.json"):
                    if cand.exists():
                        fp_json = cand
                        break
                if fp_json is None:
                    export_issues.append(
                        f"trial {tid}/{oid}: fp_dir 이 지정됐지만 T_C0_obj.json 이 없다 "
                        f"({fp_dir / oid}) — silhouette fit pose 로 대체")
            try:
                if fp_json is not None:
                    T_world_obj, pmeta = pose_from_foundationpose(fp_json, cams)
                else:
                    T_world_obj, pmeta = pose_from_silhouette_fit(fit, cad_raw)
            except ExportError as e:
                export_issues.append(f"trial {tid}/{oid}: {e}")
                log(f"[ERROR] {e}")
                continue

            mesh = load_mesh(scaled_glb)

            # --- mask 복사 + 재투영 IoU 재계산 (내보낸 mesh + 내보낸 pose 로) ---
            masks_rel: dict[str, str] = {}
            T_cam_obj_all: dict[str, list] = {}
            ious: dict[str, float] = {}
            for cid, c in cams.items():
                msrc = mask_dir / mask_key / f"{cid}_mask.png"
                if not msrc.exists():
                    export_issues.append(f"trial {tid}/{oid}/{cid}: mask 없음 {msrc}")
                    continue
                mrel = f"trials/{tid}/{cid}/mask_{oid}.png"
                copy_file(msrc, out / mrel, symlink)
                masks_rel[cid] = mrel

                T_cam_obj = np.linalg.inv(c["T_world_cam"]) @ T_world_obj
                T_cam_obj_all[cid] = T_cam_obj.tolist()

                m = cv2.imread(str(msrc), cv2.IMREAD_GRAYSCALE)
                if m is None:
                    export_issues.append(f"trial {tid}/{oid}/{cid}: mask 읽기 실패")
                    continue
                mb = m > 0
                if int(mb.sum()) < min_mask_px:
                    export_issues.append(
                        f"trial {tid}/{oid}/{cid}: mask foreground {int(mb.sum())} px < {min_mask_px}")
                sil = render_silhouette(mesh, c["K"], T_cam_obj, (c["height"], c["width"]))
                v = iou(sil, mb)
                ious[cid] = v
                flag = "" if v >= min_iou else f"  <-- 임계 {min_iou} 미만"
                log(f"  {oid} {cid}: 재투영 IoU = {v:.4f}{flag}")
                if v < min_iou:
                    export_issues.append(
                        f"trial {tid}/{oid}/{cid}: 재투영 IoU {v:.3f} < {min_iou}")

            R = T_world_obj[:3, :3]
            pose_doc = {
                "trial": tid,
                "object": oid,
                "frames": {
                    "world": world_frame_global or conv.get("world_frame"),
                    "object": f"{oid}_cad_scaled.glb 의 로컬 좌표계 (AABB 중심이 원점)",
                },
                "units": {"translation": "m"},
                "matrix_layout": conv["matrix_layout"],
                "quaternion_order": conv["quaternion_order"],
                "T_world_object": T_world_obj.tolist(),
                "translation_m": [float(v) for v in T_world_obj[:3, 3]],
                "quat_wxyz": R_to_quat_wxyz(R),
                "T_cam_object": T_cam_obj_all,
                "reprojection_iou": ious,
                "mean_reprojection_iou": (float(np.mean(list(ious.values()))) if ious else None),
                **pmeta,
            }
            prel = f"trials/{tid}/poses/{oid}.json"
            (out / prel).parent.mkdir(parents=True, exist_ok=True)
            (out / prel).write_text(json.dumps(pose_doc, indent=2, ensure_ascii=False))

            trial_objs[oid] = {"pose": prel, "masks": masks_rel,
                               "pose_source": pose_doc["pose_source"]}

            # --- objects/ 자산 (trial 간 공유; 처음 만난 fit 을 채택) ---
            if oid not in obj_fit_used:
                obj_fit_used[oid] = {"fit": fit, "trial": tid}
                odir = f"objects/{oid}"
                mesh_rel = f"{odir}/{oid}_cad_scaled.glb"
                raw_rel = f"{odir}/{oid}_cad_raw{cad_raw.suffix}"
                copy_file(scaled_glb, out / mesh_rel, symlink=False)  # 자산은 항상 실제 복사
                copy_file(cad_raw, out / raw_rel, symlink=False)

                est_m = [float(v) for v in fit["extents_m"]]
                meas_mm = (ocfg.get("measured") or {}).get("extents_mm")
                meas_m = [float(v) / 1000.0 for v in meas_mm] if meas_mm else None

                est_sorted = sorted(est_m, reverse=True)
                scale_to_measured = None
                if meas_m:
                    ms = sorted(meas_m, reverse=True)
                    scale_to_measured = [float(m / e) for m, e in zip(ms, est_sorted)]
                    for i, (g, e) in enumerate(zip(ms, est_sorted), start=1):
                        err = (e - g) * 1000.0
                        size_rows.append({
                            "object": oid, "axis": f"L{i}",
                            "measured_mm": round(g * 1000.0, 3),
                            "estimated_mm": round(e * 1000.0, 3),
                            "error_mm": round(err, 3),
                            "error_percent": round(100.0 * (e - g) / g, 3),
                        })
                        if abs(err) > max_size_err:
                            export_issues.append(
                                f"{oid} L{i}: 치수 오차 {err:+.2f} mm 가 임계 {max_size_err} mm 초과")

                obj_doc = {
                    "id": oid,
                    "name": ocfg.get("name"),
                    "assets": {"mesh_scaled": mesh_rel, "mesh_raw": raw_rel},
                    "size": {
                        "estimated_extents_m": est_m,
                        "estimated_extents_mm_sorted_desc": fit.get("extents_mm_sorted_desc"),
                        "measured_extents_m": meas_m,
                        "measured_extents_mm_sorted_desc": (sorted(meas_mm, reverse=True)
                                                            if meas_mm else None),
                        "measured_source": (ocfg.get("measured") or {}).get("source"),
                        "measured_tolerance_mm": (ocfg.get("measured") or {}).get("tolerance_mm"),
                        "estimation_method": fit.get("method"),
                        "mask_pm1px_spread_mm": fit.get("mask_pm1px_spread_mm"),
                        "mean_silhouette_iou": fit.get("mean_iou"),
                        "scale_estimated_to_measured": scale_to_measured,
                        "note": ("mesh_scaled 는 estimated 치수다. Oracle / Pose-only 조건에서는 "
                                 "scale_estimated_to_measured 를 mesh 에 곱해 measured 치수로 만든다."),
                    },
                    "mass_kg": ocfg.get("mass_kg"),
                    "com_m": ocfg.get("com_m"),
                    "material": ocfg.get("material"),
                    "source_fit": f"{fit_key}_cad_fit.json (trial {tid})",
                }
                (out / odir).mkdir(parents=True, exist_ok=True)
                (out / f"{odir}/object.yaml").write_text(
                    yaml.safe_dump(obj_doc, sort_keys=False, allow_unicode=True))
                manifest_objects[oid] = {"dir": odir, "yaml": f"{odir}/object.yaml"}
                if ocfg.get("mass_kg") is None:
                    log(f"  [TODO] {oid}: mass_kg 미측정 (null)")

        trial_doc = {
            "id": tid,
            "world_frame": world_frame_global or conv.get("world_frame"),
            "source": {"capture_dir": t["capture_dir"], "mask_dir": t["mask_dir"],
                       "fit_dir": t["fit_dir"], "fp_dir": t.get("fp_dir")},
            "cameras": trial_cams,
            "objects": trial_objs,
        }
        (out / f"trials/{tid}").mkdir(parents=True, exist_ok=True)
        (out / f"trials/{tid}/trial.yaml").write_text(
            yaml.safe_dump(trial_doc, sort_keys=False, allow_unicode=True))
        manifest_trials[tid] = {
            "dir": f"trials/{tid}", "yaml": f"trials/{tid}/trial.yaml",
            "cameras": sorted(trial_cams), "objects": sorted(trial_objs),
        }

    if cameras_global is None:
        raise SystemExit("카메라 정보를 하나도 읽지 못했다. capture_dir 경로를 확인하라.")

    # config 에 선언됐지만 데이터가 아직 없는 객체 = pending.
    # 오류가 아니다 (객체를 하나씩 추가하는 중일 수 있다) — 무엇이 빠졌는지만 정확히 알려준다.
    pending: dict[str, list[str]] = {}
    for oid, ocfg in obj_cfg.items():
        if oid in manifest_objects:
            continue
        fit_key = (ocfg or {}).get("fit_key", oid)
        mask_key = (ocfg or {}).get("mask_key", oid)
        need = []
        cad_rel = (ocfg or {}).get("cad")
        if not cad_rel or not (REPO / cad_rel).exists():
            need.append(f"mesh: data/meshes/{oid}.glb")
        for t in trials:
            if not (REPO / t["mask_dir"] / mask_key).is_dir():
                need.append(f"mask: {t['mask_dir']}/{mask_key}/cam*_mask.png  (Obj_Step2 --objects {oid})")
            if not (REPO / t["fit_dir"] / f"{fit_key}_cad_fit.json").exists():
                need.append(f"fit : {t['fit_dir']}/{fit_key}_cad_fit.json  (Obj_Step3c --cad {oid}=...)")
            break   # trial 하나만 예시로 보여준다
        pending[oid] = need

    if pending:
        log("\n[PENDING] config 에 선언됐지만 데이터가 없는 객체:")
        for oid, need in pending.items():
            log(f"  {oid}:")
            for n in need:
                log(f"    - 없음 → {n}")

    # --- calibration/cameras.yaml ---
    cam_doc = {
        "world_frame": world_frame_global or conv.get("world_frame"),
        "camera_model": conv["camera_model"],
        "note": ("T_world_cam 은 OpenCV 카메라 좌표를 world 로 보내는 4x4 (camera -> world). "
                 "Isaac/USD 카메라는 -Z 를 보고 +Y 가 위라서 T_world_cam_isaac 을 따로 넣었다."),
        "opencv_to_usd_camera": CV_TO_USD.tolist(),
        "cameras": {},
    }
    for cid, c in cameras_global.items():
        T = c["T_world_cam"]
        K = c["K"]
        T_isaac = T @ CV_TO_USD
        cam_doc["cameras"][cid] = {
            "serial": c["serial"],
            "width": int(c["width"]), "height": int(c["height"]),
            "K": K.tolist(),
            "fx": float(K[0, 0]), "fy": float(K[1, 1]),
            "cx": float(K[0, 2]), "cy": float(K[1, 2]),
            "distortion": None,   # RealSense color 는 rectified 출력 — 왜곡계수 미적용
            "distortion_model": "none",
            "T_world_cam": T.tolist(),
            "quat_wxyz": R_to_quat_wxyz(T[:3, :3]),
            "translation_m": [float(v) for v in T[:3, 3]],
            "T_world_cam_isaac": T_isaac.tolist(),
            "quat_wxyz_isaac": R_to_quat_wxyz(T_isaac[:3, :3]),
            "depth_scale_m_per_unit": c["depth_scale_m_per_unit"],
        }
    (out / "calibration").mkdir(parents=True, exist_ok=True)
    (out / "calibration/cameras.yaml").write_text(
        yaml.safe_dump(cam_doc, sort_keys=False, allow_unicode=True))

    # --- measurements/ ---
    (out / "measurements").mkdir(parents=True, exist_ok=True)
    meas_doc = {
        oid: {
            "name": o.get("name"),
            "measured_extents_mm": (o.get("measured") or {}).get("extents_mm"),
            "measured_source": (o.get("measured") or {}).get("source"),
            "tolerance_mm": (o.get("measured") or {}).get("tolerance_mm"),
            "mass_kg": o.get("mass_kg"),
            "com_m": o.get("com_m"),
            "material": o.get("material"),
        }
        for oid, o in obj_cfg.items() if oid in manifest_objects
    }
    (out / "measurements/measured_sizes.yaml").write_text(
        yaml.safe_dump(meas_doc, sort_keys=False, allow_unicode=True))

    if size_rows:
        with (out / "measurements/size_error.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(size_rows[0]))
            w.writeheader()
            w.writerows(size_rows)

    # --- schemas/ ---
    write_schemas(out / "schemas")

    # --- scripts/ ---
    (out / "scripts").mkdir(parents=True, exist_ok=True)
    for name in ("validate_handoff.py", "inspect_trial.py"):
        src = REPO / "scripts" / name
        if src.exists():
            shutil.copy2(src, out / "scripts" / name)
        else:
            export_issues.append(f"scripts/{name} 이 repo 에 없어 package 에 넣지 못했다")

    # --- README (repo 는 README_ISAAC_SIM.md, package 안에서는 ISAAC_SIM_README.md) ---
    readme = next((REPO / n for n in ("README_ISAAC_SIM.md", "ISAAC_SIM_README.md")
                   if (REPO / n).exists()), None)
    if readme is not None:
        shutil.copy2(readme, out / "ISAAC_SIM_README.md")
    else:
        export_issues.append("README_ISAAC_SIM.md 가 repo root 에 없다")

    write_conventions_md(out / "calibration/coordinate_conventions.md", conv,
                         world_frame_global or conv.get("world_frame"))

    # --- manifest.yaml ---
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "scripts/export_isaac_handoff.py",
        "git_commit": git_commit(),
        "source_config": str(cfg_path.relative_to(REPO) if cfg_path.is_relative_to(REPO) else cfg_path),
        "conventions": dict(conv),
        "quality_gates": dict(gates),
        "cameras": {"file": "calibration/cameras.yaml", "ids": sorted(cameras_global)},
        "objects": manifest_objects,
        "pending_objects": pending or None,   # 선언됐지만 데이터가 없는 객체 (오류 아님)
        "trials": manifest_trials,
        "measurements": {
            "sizes": "measurements/measured_sizes.yaml",
            "size_error_csv": ("measurements/size_error.csv" if size_rows else None),
        },
        "isaac": cfg.get("isaac"),
        "export_issues": export_issues,
    }
    (out / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True))

    # --- 객체 준비 상태 요약 ---
    log("\n" + "=" * 72)
    log(f"객체 준비 상태  ({len(manifest_objects)} exported / {len(pending)} pending)")
    log("=" * 72)
    log(f"{'객체':<16} {'mesh':<6} {'mask':<6} {'fit':<6} {'실측치수':<10} {'질량':<8}")
    log("-" * 72)
    for oid in sorted(set(manifest_objects) | set(pending)):
        ok = oid in manifest_objects
        ocfg = obj_cfg.get(oid) or {}
        has_meas = bool((ocfg.get("measured") or {}).get("extents_mm")) and \
            all(v is not None for v in (ocfg.get("measured") or {}).get("extents_mm", [None]))
        has_mass = ocfg.get("mass_kg") is not None
        y, n = "  O   ", "  -   "
        log(f"{oid:<16} {y if ok else n} {y if ok else n} {y if ok else n} "
            f"{'  O       ' if has_meas else '  TODO    '} {'  O' if has_mass else '  TODO'}")
    log("-" * 72)
    log("O = 준비됨,  - = 데이터 없음(pending),  TODO = 사용자가 채워야 함")

    log(f"\n[SAVE] handoff package -> {out}")
    if export_issues:
        log(f"[WARN] export 중 {len(export_issues)} 건의 문제 (manifest.export_issues 참고)")
    return manifest


# --------------------------------------------------------------------------- #
# schema / 규약 문서
# --------------------------------------------------------------------------- #

def write_schemas(sdir: Path) -> None:
    sdir.mkdir(parents=True, exist_ok=True)
    mat4 = {
        "type": "array", "minItems": 4, "maxItems": 4,
        "items": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "number"}},
        "description": "row-major 4x4. p_dst = T @ p_src. 마지막 행은 [0,0,0,1].",
    }
    (sdir / "pose.schema.json").write_text(json.dumps({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "trials/<trial>/poses/<object>.json",
        "type": "object",
        "required": ["trial", "object", "T_world_object", "quat_wxyz",
                     "T_cam_object", "pose_source", "matrix_layout", "quaternion_order"],
        "properties": {
            "trial": {"type": "string"},
            "object": {"type": "string"},
            "T_world_object": mat4,
            "translation_m": {"type": "array", "minItems": 3, "maxItems": 3,
                              "items": {"type": "number"}},
            "quat_wxyz": {"type": "array", "minItems": 4, "maxItems": 4,
                          "items": {"type": "number"},
                          "description": "scalar-first (w, x, y, z)"},
            "T_cam_object": {"type": "object", "additionalProperties": mat4},
            "reprojection_iou": {"type": "object", "additionalProperties": {"type": "number"}},
            "pose_source": {"enum": ["foundationpose", "cad_silhouette_fit"]},
            "matrix_layout": {"const": "row_major"},
            "quaternion_order": {"const": "wxyz"},
        },
    }, indent=2, ensure_ascii=False))

    (sdir / "object.schema.json").write_text(json.dumps({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "objects/<object>/object.yaml",
        "type": "object",
        "required": ["id", "assets", "size"],
        "properties": {
            "id": {"type": "string"},
            "name": {"type": ["string", "null"]},
            "assets": {
                "type": "object", "required": ["mesh_scaled"],
                "properties": {"mesh_scaled": {"type": "string"}, "mesh_raw": {"type": "string"}},
            },
            "size": {
                "type": "object", "required": ["estimated_extents_m"],
                "properties": {
                    "estimated_extents_m": {"type": "array", "minItems": 3, "maxItems": 3,
                                            "items": {"type": "number", "exclusiveMinimum": 0}},
                    "measured_extents_m": {"type": ["array", "null"], "minItems": 3, "maxItems": 3,
                                           "items": {"type": "number", "exclusiveMinimum": 0}},
                    "scale_estimated_to_measured": {"type": ["array", "null"],
                                                    "minItems": 3, "maxItems": 3,
                                                    "items": {"type": "number"}},
                },
            },
            "mass_kg": {"type": ["number", "null"], "exclusiveMinimum": 0},
            "com_m": {"type": ["array", "null"], "minItems": 3, "maxItems": 3},
        },
    }, indent=2, ensure_ascii=False))

    (sdir / "camera.schema.json").write_text(json.dumps({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "calibration/cameras.yaml",
        "type": "object",
        "required": ["world_frame", "camera_model", "cameras"],
        "properties": {
            "world_frame": {"type": "string"},
            "camera_model": {"const": "opencv"},
            "opencv_to_usd_camera": mat4,
            "cameras": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "required": ["width", "height", "K", "T_world_cam"],
                    "properties": {
                        "width": {"type": "integer", "exclusiveMinimum": 0},
                        "height": {"type": "integer", "exclusiveMinimum": 0},
                        "K": {"type": "array", "minItems": 3, "maxItems": 3,
                              "items": {"type": "array", "minItems": 3, "maxItems": 3,
                                        "items": {"type": "number"}}},
                        "T_world_cam": mat4,
                        "T_world_cam_isaac": mat4,
                    },
                },
            },
        },
    }, indent=2, ensure_ascii=False))


def write_conventions_md(path: Path, conv: dict, world_frame: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"""# 좌표계와 단위 (자동 생성 — 고치지 말 것)

이 파일은 `scripts/export_isaac_handoff.py` 가 실제 export 값으로 생성한다.
사람이 손으로 고치면 package 와 어긋난다.

## 프레임

| 이름 | 정의 |
|---|---|
| `world` | **`{world_frame}`** 이다. 로봇 base 가 아니다. `Obj_Step1_capture_object.py` 가 reference 카메라를 world 로 잡고 dump 한다. |
| `camera` | OpenCV 규약: **x=오른쪽, y=아래, z=광축 전방**. |
| `object` | `objects/<id>/<id>_cad_scaled.glb` 의 로컬 좌표계. AABB 중심이 원점. |

## 변환

모든 4x4 는 **{conv['matrix_layout']}**, `p_dst = T @ p_src` (열벡터 오른쪽 곱).

```
T_world_object = T_world_cam @ T_cam_object
T_cam_object   = inv(T_world_cam) @ T_world_object
```

* `T_world_cam` — `calibration/cameras.yaml` 의 `T_world_cam`. **camera → world** 방향이다.
  `cam{{i}}_T_cam_to_world.txt` 를 그대로 읽은 값이다. 역방향으로 쓰면 객체가 엉뚱한 곳에 뜬다.
* `T_world_object` — `trials/<trial>/poses/<object>.json`.
* `T_cam_object` — 같은 파일의 카메라별 값. 위 식으로 이미 계산해 두었다.

## 단위

* 길이: **{conv['length_unit']}** (pose translation, mesh vertex, size 전부).
* depth PNG: uint16, **{conv['depth_unit']}**. meter 로 쓰려면 `depth_scale = {conv['depth_scale']}` 를 곱한다.
* mass: kg. (미측정이면 `null`)

## Quaternion

**`{conv['quaternion_order']}` (scalar-first)** — `quat_wxyz` 키 이름 그대로다.
USD `Gf.Quatd(w, Gf.Vec3d(x, y, z))` 와 순서가 같다.
scipy 는 `xyzw` 라서 `Rotation.from_quat` 에 그대로 넣으면 안 된다:

```python
from scipy.spatial.transform import Rotation
w, x, y, z = pose["quat_wxyz"]
R = Rotation.from_quat([x, y, z, w])      # scipy 는 xyzw
```

## OpenCV 카메라 vs Isaac/USD 카메라

OpenCV 는 **+Z 가 광축 전방, +Y 가 아래**. USD 카메라는 **-Z 가 전방, +Y 가 위**.
그래서 extrinsic 을 그대로 USD 카메라에 넣으면 상이 **상하로 뒤집힌다**.

```
T_world_cam_isaac = T_world_cam @ CV_TO_USD

CV_TO_USD =
  [ 1,  0,  0, 0]
  [ 0, -1,  0, 0]
  [ 0,  0, -1, 0]
  [ 0,  0,  0, 1]
```

`cameras.yaml` 에 **이미 변환된 `T_world_cam_isaac` 와 `quat_wxyz_isaac` 를 넣어 두었다.**
Isaac 카메라 prim 에는 그 값을 쓰고, 마스크/포즈 계산에는 `T_world_cam` 을 써라.

## Stage up-axis

world 는 `{world_frame}` (카메라 좌표계) 이라 **+Y 가 아래를 향한다.** USD 기본 up-axis 는 +Z 다.
stage 를 그대로 만들면 객체가 옆으로 누워 보인다. 두 가지 중 하나를 택하라.

1. stage up-axis 를 이 world 에 맞추지 말고, **모든 pose 를 그대로 쓰고 중력 방향만
   world 기준으로 계산**한다 (물리 없는 렌더링/alignment 평가에는 이걸로 충분하다).
2. 물리(grasp ablation)를 돌리려면 world → Isaac world 회전 `T_isaacworld_world` 를 한 번
   정의해 모든 pose 에 왼쪽 곱한다. 이 행렬은 **아직 정해지지 않았다 (README §17 TODO)** —
   테이블 평면을 어떻게 world 에 정렬할지는 Isaac 담당자가 정한다.
""")


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=REPO / "configs/evaluation.yaml")
    ap.add_argument("--trials", type=Path, default=None,
                    help="trial root (선택). 하위 폴더에서 capture/mask/fit layout 을 자동 탐색.")
    ap.add_argument("--output", type=Path, default=REPO / "outputs/isaac_handoff")
    ap.add_argument("--symlink", action="store_true",
                    help="RGB/depth/mask 를 복사 대신 symlink (용량 절약, 이식성 하락)")
    ap.add_argument("--skip_validate", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    export(cfg, args.config.resolve(), args.trials, args.output.resolve(), args.symlink)

    if args.skip_validate:
        return 0

    # export 직후 항상 검증한다. 통과 못 한 package 를 넘기면 Isaac 쪽에서 원인을 못 찾는다.
    log("\n" + "=" * 72)
    log("검증 실행 (scripts/validate_handoff.py)")
    log("=" * 72)
    out = args.output.resolve()
    rc = subprocess.run(
        [sys.executable, str(out / "scripts/validate_handoff.py"),
         "--package", str(out),
         "--json", str(out / "handoff_validation_report.json"),
         "--txt", str(out / "handoff_validation_report.txt")],
        cwd=REPO,
    ).returncode
    log(f"\n보고서: {out / 'handoff_validation_report.txt'}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
