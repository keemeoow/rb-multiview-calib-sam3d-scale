#!/usr/bin/env python3
"""
validate_handoff.py — Isaac Sim handoff package 검증기.

이 파일은 **package 안에 복사되어 단독 실행**된다. repo 의 다른 모듈을 import 하지 않는다.
의존성: numpy, pyyaml, trimesh, opencv-python.

  python scripts/validate_handoff.py                 # package root 에서
  python scripts/validate_handoff.py --package .     # 경로 명시
  python scripts/validate_handoff.py --package outputs/isaac_handoff --json report.json

종료 코드
  0  ready_for_isaac_sim = true   (error 0건)
  1  error 존재 → Isaac Sim 실행 전에 고쳐야 한다
  2  package 를 아예 읽지 못함 (manifest 없음 등)

실패해도 첫 오류에서 멈추지 않는다. 객체/trial/필드 단위로 전부 모아서 보고한다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

# meter 단위 sanity 범위. 이 밖이면 단위(mm/m) 혼동을 의심한다.
MAX_TRANSLATION_M = 10.0
MIN_EXTENT_M = 1e-4
MAX_EXTENT_M = 1.0
ORTHO_TOL = 1e-4
DET_TOL = 1e-3


class Report:
    """객체/trial/필드 단위로 오류를 모은다. 첫 오류에서 종료하지 않는다."""

    def __init__(self) -> None:
        self.errors: list[dict] = []
        self.warnings: list[dict] = []
        self.checks_run = 0

    def check(self, ok: bool, where: str, field: str, msg: str, *, warn: bool = False) -> bool:
        self.checks_run += 1
        if ok:
            return True
        entry = {"where": where, "field": field, "message": msg}
        (self.warnings if warn else self.errors).append(entry)
        return False

    def error(self, where: str, field: str, msg: str) -> None:
        self.check(False, where, field, msg)

    def warn(self, where: str, field: str, msg: str) -> None:
        self.check(False, where, field, msg, warn=True)


# --------------------------------------------------------------------------- #
# 저수준 검사
# --------------------------------------------------------------------------- #

def check_pose_matrix(rep: Report, where: str, field: str, value) -> np.ndarray | None:
    """4x4, 마지막 행 [0,0,0,1], R 직교 + det=+1, finite, translation 단위."""
    try:
        T = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        rep.error(where, field, f"4x4 행렬로 변환 불가: {value!r}")
        return None

    if T.shape != (4, 4):
        rep.error(where, field, f"shape 가 (4,4) 가 아님: {T.shape}")
        return None
    if not np.isfinite(T).all():
        rep.error(where, field, "NaN 또는 inf 포함")
        return None

    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-9):
        rep.error(where, field, f"마지막 행이 [0,0,0,1] 이 아님: {T[3].tolist()}")

    R = T[:3, :3]
    ortho_err = float(np.abs(R @ R.T - np.eye(3)).max())
    det = float(np.linalg.det(R))
    rep.check(ortho_err <= ORTHO_TOL, where, field,
              f"rotation 이 직교가 아님 (max|RR^T - I| = {ortho_err:.2e} > {ORTHO_TOL:.0e})")
    rep.check(abs(det - 1.0) <= DET_TOL, where, field,
              f"det(R) = {det:.6f} != +1 (반사 행렬이면 좌표계 handedness 가 뒤집힌 것)")

    t_norm = float(np.linalg.norm(T[:3, 3]))
    rep.check(t_norm <= MAX_TRANSLATION_M, where, field,
              f"translation |t| = {t_norm:.3f} m 가 {MAX_TRANSLATION_M} m 초과 "
              f"— mm 값을 m 로 넣었는지 확인 (1000배 오류)")
    return T


def check_quat(rep: Report, where: str, field: str, q, T: np.ndarray | None) -> None:
    """manifest 규약은 wxyz (scalar-first). 행렬과 실제로 일치하는지도 본다."""
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (4,):
        rep.error(where, field, f"quaternion 은 길이 4 여야 함: {q.shape}")
        return
    if not np.isfinite(q).all():
        rep.error(where, field, "quaternion 에 NaN/inf")
        return
    n = float(np.linalg.norm(q))
    rep.check(abs(n - 1.0) < 1e-6, where, field, f"quaternion 이 정규화되지 않음 (|q| = {n:.6f})")
    if T is None:
        return
    R_q = quat_wxyz_to_R(q / max(n, 1e-12))
    err = float(np.abs(R_q - T[:3, :3]).max())
    rep.check(err < 1e-5, where, field,
              f"quaternion(wxyz) 이 rotation 행렬과 불일치 (max diff {err:.2e}) "
              f"— xyzw 순서로 잘못 쓴 것 아닌지 확인")


def quat_wxyz_to_R(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def check_inside_package(rep: Report, where: str, field: str, root: Path, rel: str) -> Path | None:
    """상대경로가 package 밖 (../, 절대경로) 을 가리키면 error."""
    p = Path(rel)
    if p.is_absolute():
        rep.error(where, field, f"절대경로 금지 (package 이식 시 깨짐): {rel}")
        return None
    target = (root / p).resolve()
    if not str(target).startswith(str(root.resolve())):
        rep.error(where, field, f"상대경로가 package 밖을 참조: {rel}")
        return None
    if not target.exists():
        rep.error(where, field, f"파일 없음: {rel}")
        return None
    return target


def check_mesh(rep: Report, where: str, root: Path, rel: str,
               expect_extents_m=None, *, metric: bool = True) -> None:
    """로드 가능 여부, texture 참조, 치수, 원점 중심 여부.

    metric=False 는 **원본 CAD** (mesh_raw) 용이다. 원본은 단위가 임의이고
    원점 중심도 아닌 것이 정상이라 치수/원점 검사를 건너뛴다.
    Isaac 이 실제로 스테이지에 올리는 것은 metric=True 인 mesh_scaled 뿐이다.
    """
    path = check_inside_package(rep, where, "mesh", root, rel)
    if path is None:
        return
    try:
        import trimesh
        mesh = trimesh.load(str(path), force="mesh")
    except Exception as e:  # noqa: BLE001 — 어떤 로더 예외든 error 로 보고
        rep.error(where, "mesh", f"mesh 로드 실패 ({rel}): {e}")
        return

    if not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
        rep.error(where, "mesh", f"vertex 가 없음: {rel}")
        return

    V = np.asarray(mesh.vertices, dtype=np.float64)
    if not np.isfinite(V).all():
        rep.error(where, "mesh", f"vertex 에 NaN/inf: {rel}")
        return

    extents = V.max(axis=0) - V.min(axis=0)
    if metric:
        if extents.max() > MAX_EXTENT_M or extents.min() < MIN_EXTENT_M:
            rep.error(where, "mesh.extents",
                      f"mesh 치수 {np.round(extents * 1000, 1).tolist()} mm 가 예상 범위를 벗어남 "
                      f"— glb 단위가 meter 가 맞는지 확인 ({rel})")

        center = 0.5 * (V.max(axis=0) + V.min(axis=0))
        off_mm = float(np.linalg.norm(center)) * 1000.0
        rep.check(off_mm <= 1.0, where, "mesh.origin",
                  f"mesh AABB 중심이 원점에서 {off_mm:.2f} mm 떨어짐 — pose 가 그만큼 어긋난다 ({rel})",
                  warn=off_mm <= 5.0)

    # GLB 는 texture 를 embed 한다. 외부 파일을 참조하면 package 이식 시 깨진다.
    mat = getattr(getattr(mesh, "visual", None), "material", None)
    for attr in ("image", "baseColorTexture"):
        img = getattr(mat, attr, None) if mat is not None else None
        if isinstance(img, (str, Path)):
            rep.error(where, "mesh.texture",
                      f"texture 가 외부 파일을 참조 ({img}) — glb 에 embed 되어야 함")

    if expect_extents_m is not None and metric:
        exp = np.sort(np.asarray(expect_extents_m, dtype=np.float64))[::-1] * 1000.0
        got = np.sort(extents)[::-1] * 1000.0
        d = float(np.abs(exp - got).max())
        rep.check(d < 0.5, where, "mesh.extents",
                  f"mesh 실측 치수 {np.round(got, 2).tolist()} mm 가 "
                  f"metadata 값 {np.round(exp, 2).tolist()} mm 와 {d:.2f} mm 차이")


# --------------------------------------------------------------------------- #
# package 단위 검증
# --------------------------------------------------------------------------- #

REQUIRED_FILES = [
    "manifest.yaml",
    "calibration/cameras.yaml",
    "calibration/coordinate_conventions.md",
    "ISAAC_SIM_README.md",
]

REQUIRED_CONVENTION_KEYS = [
    "length_unit", "depth_unit", "depth_scale", "quaternion_order",
    "matrix_layout", "camera_model", "world_frame",
]


def validate(root: Path, min_iou: float | None = None) -> tuple[Report, dict]:
    rep = Report()
    stats = {
        "package": str(root),
        "num_objects": 0, "num_trials": 0, "num_cameras": 0,
        "valid_trials": 0, "excluded_trials": [], "missing_files": [],
        "bad_poses": [], "bad_masks": [], "todo_fields": [], "pending_objects": [],
    }

    for rel in REQUIRED_FILES:
        if not (root / rel).exists():
            rep.error("package", "required_file", f"필수 파일 없음: {rel}")
            stats["missing_files"].append(rel)

    manifest_path = root / "manifest.yaml"
    if not manifest_path.exists():
        return rep, stats
    manifest = yaml.safe_load(manifest_path.read_text())

    # --- 규약 (pose convention 누락 검사) ---
    conv = manifest.get("conventions") or {}
    for k in REQUIRED_CONVENTION_KEYS:
        rep.check(k in conv and conv[k] is not None, "manifest", f"conventions.{k}",
                  f"pose convention 누락: conventions.{k} — 단위/축 해석이 불가능해진다")
    if conv.get("quaternion_order") not in (None, "wxyz", "xyzw"):
        rep.error("manifest", "conventions.quaternion_order",
                  f"알 수 없는 quaternion 순서: {conv.get('quaternion_order')}")

    cameras_cfg = yaml.safe_load((root / "calibration" / "cameras.yaml").read_text()) \
        if (root / "calibration" / "cameras.yaml").exists() else {}
    cams = (cameras_cfg or {}).get("cameras") or {}
    stats["num_cameras"] = len(cams)
    rep.check(len(cams) > 0, "calibration", "cameras", "cameras.yaml 에 카메라가 없음")

    cam_res: dict[str, tuple[int, int]] = {}
    for cid, c in cams.items():
        where = f"camera/{cid}"
        K = np.asarray(c.get("K", []), dtype=np.float64)
        if K.shape != (3, 3) or not np.isfinite(K).all():
            rep.error(where, "K", f"intrinsic K 가 3x3 finite 가 아님: shape={K.shape}")
        else:
            rep.check(K[0, 0] > 0 and K[1, 1] > 0, where, "K",
                      f"focal length 가 양수가 아님: fx={K[0,0]}, fy={K[1,1]}")
        T = check_pose_matrix(rep, where, "T_world_cam", c.get("T_world_cam"))
        if T is not None and c.get("quat_wxyz") is not None:
            check_quat(rep, where, "quat_wxyz", c["quat_wxyz"], T)
        w, h = c.get("width"), c.get("height")
        if isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0:
            cam_res[cid] = (h, w)
        else:
            rep.error(where, "resolution", f"width/height 가 유효한 양의 정수가 아님: {w}x{h}")

    # --- 객체 ---
    objects = manifest.get("objects") or {}
    stats["num_objects"] = len(objects)
    obj_ids = set(objects)

    # 선언만 되고 데이터가 없는 객체. 오류가 아니다 — 아직 촬영/추정을 안 한 것뿐이다.
    # 다만 Isaac 담당자가 "객체가 왜 5개뿐이지?" 하고 헤매지 않도록 반드시 보고한다.
    pending = manifest.get("pending_objects") or {}
    stats["pending_objects"] = sorted(pending)
    for oid, need in pending.items():
        rep.warn(f"object/{oid}", "pending",
                 "데이터가 아직 없어 package 에 포함되지 않았다: " + "; ".join(need or ["?"]))
        stats["todo_fields"].append(f"{oid} (데이터 미수집)")
    for oid, entry in objects.items():
        where = f"object/{oid}"
        yrel = entry.get("yaml")
        ypath = check_inside_package(rep, where, "yaml", root, yrel) if yrel else None
        if ypath is None:
            rep.error(where, "yaml", "object.yaml 을 찾을 수 없음")
            continue
        obj = yaml.safe_load(ypath.read_text())

        rep.check(obj.get("id") == oid, where, "id",
                  f"object.yaml 의 id ({obj.get('id')!r}) 가 manifest key ({oid!r}) 와 불일치")

        est = (obj.get("size") or {}).get("estimated_extents_m")
        meas = (obj.get("size") or {}).get("measured_extents_m")
        for label, ext in (("estimated", est), ("measured", meas)):
            if ext is None:
                rep.check(label != "estimated", where, f"size.{label}_extents_m",
                          "추정 치수가 없음 — scale condition 을 구성할 수 없다")
                continue
            e = np.asarray(ext, dtype=np.float64)
            if e.shape != (3,) or not np.isfinite(e).all():
                rep.error(where, f"size.{label}_extents_m", f"3-vector 가 아니거나 NaN: {ext}")
            elif (e <= 0).any():
                rep.error(where, f"size.{label}_extents_m", f"치수가 양수가 아님: {e.tolist()}")
            elif e.max() > MAX_EXTENT_M:
                rep.error(where, f"size.{label}_extents_m",
                          f"치수 {np.round(e * 1000, 1).tolist()} mm 가 비현실적 — m/mm 혼동 의심")

        if obj.get("mass_kg") is None:
            rep.warn(where, "mass_kg", "질량 미측정 (null) — Isaac 물리 시뮬레이션에 필요")
            stats["todo_fields"].append(f"{oid}.mass_kg")
        elif not (0 < float(obj["mass_kg"]) < 50):
            rep.error(where, "mass_kg", f"질량이 비현실적: {obj['mass_kg']} kg (kg 단위가 맞는가?)")

        for key in ("mesh_scaled", "mesh_raw"):
            rel = (obj.get("assets") or {}).get(key)
            if rel is None:
                rep.check(key != "mesh_scaled", where, f"assets.{key}", "필수 mesh 경로 없음")
                continue
            is_scaled = key == "mesh_scaled"
            check_mesh(rep, f"{where}.{key}", root, rel,
                       expect_extents_m=est if is_scaled else None,
                       metric=is_scaled)

    # --- trial ---
    trials = manifest.get("trials") or {}
    stats["num_trials"] = len(trials)
    for tid, entry in trials.items():
        where = f"trial/{tid}"
        trial_ok = True
        yrel = entry.get("yaml")
        ypath = check_inside_package(rep, where, "yaml", root, yrel) if yrel else None
        if ypath is None:
            rep.error(where, "yaml", "trial.yaml 을 찾을 수 없음")
            stats["excluded_trials"].append(tid)
            continue
        trial = yaml.safe_load(ypath.read_text())

        t_cams = list((trial.get("cameras") or {}))
        missing_cams = [c for c in cams if c not in t_cams]
        if missing_cams:
            trial_ok = False
            rep.error(where, "cameras",
                      f"calibration 에 있는 카메라의 데이터가 trial 에 없음: {missing_cams}")

        # 카메라별 RGB/depth 존재 + 해상도 일치
        for cid, cam_entry in (trial.get("cameras") or {}).items():
            cw = f"{where}/{cid}"
            if cid not in cams:
                trial_ok = False
                rep.error(cw, "cameras", f"cameras.yaml 에 없는 카메라: {cid}")
                continue
            for key in ("rgb", "depth"):
                rel = cam_entry.get(key)
                if rel is None:
                    trial_ok = False
                    rep.error(cw, key, f"{key} 경로 없음")
                    continue
                p = check_inside_package(rep, cw, key, root, rel)
                if p is None:
                    trial_ok = False
                    stats["missing_files"].append(rel)
                    continue
                hw = image_shape(p)
                if hw is None:
                    trial_ok = False
                    rep.error(cw, key, f"이미지를 읽을 수 없음: {rel}")
                elif cid in cam_res and hw != cam_res[cid]:
                    trial_ok = False
                    rep.error(cw, key,
                              f"{key} 해상도 {hw[1]}x{hw[0]} 가 카메라 intrinsic 해상도 "
                              f"{cam_res[cid][1]}x{cam_res[cid][0]} 와 불일치 "
                              f"— K 를 그대로 쓰면 투영이 틀어진다")

        # 객체별 mask + pose
        t_objs = trial.get("objects") or {}
        for oid, oentry in t_objs.items():
            ow = f"{where}/{oid}"
            if oid not in obj_ids:
                trial_ok = False
                rep.error(ow, "object_id", f"manifest objects 에 없는 객체 id: {oid}")

            for cid, mrel in (oentry.get("masks") or {}).items():
                mw = f"{ow}/{cid}"
                p = check_inside_package(rep, mw, "mask", root, mrel)
                if p is None:
                    trial_ok = False
                    stats["bad_masks"].append({"trial": tid, "object": oid, "camera": cid,
                                               "reason": "파일 없음", "path": mrel})
                    continue
                hw = image_shape(p)
                if hw is None:
                    trial_ok = False
                    stats["bad_masks"].append({"trial": tid, "object": oid, "camera": cid,
                                               "reason": "읽기 실패", "path": mrel})
                    rep.error(mw, "mask", f"mask 를 읽을 수 없음: {mrel}")
                elif cid in cam_res and hw != cam_res[cid]:
                    trial_ok = False
                    stats["bad_masks"].append({"trial": tid, "object": oid, "camera": cid,
                                               "reason": "해상도 불일치", "path": mrel})
                    rep.error(mw, "mask",
                              f"mask 해상도 {hw[1]}x{hw[0]} != 카메라 {cam_res[cid][1]}x{cam_res[cid][0]}")
                else:
                    npix = mask_pixels(p)
                    if npix is not None and npix < 50:
                        trial_ok = False
                        stats["bad_masks"].append({"trial": tid, "object": oid, "camera": cid,
                                                   "reason": f"foreground {npix} px (비어 있음)",
                                                   "path": mrel})
                        rep.error(mw, "mask", f"mask foreground 가 {npix} px 뿐 — 사실상 빈 mask")

            prel = oentry.get("pose")
            if prel is None:
                trial_ok = False
                rep.error(ow, "pose", "pose json 경로 없음")
                stats["bad_poses"].append({"trial": tid, "object": oid, "reason": "경로 없음"})
                continue
            ppath = check_inside_package(rep, ow, "pose", root, prel)
            if ppath is None:
                trial_ok = False
                stats["bad_poses"].append({"trial": tid, "object": oid, "reason": "파일 없음"})
                continue

            pose = json.loads(ppath.read_text())
            T_wo = check_pose_matrix(rep, ow, "T_world_object", pose.get("T_world_object"))
            if T_wo is None:
                trial_ok = False
                stats["bad_poses"].append({"trial": tid, "object": oid,
                                           "reason": "T_world_object 형식 오류"})
            elif pose.get("quat_wxyz") is not None:
                check_quat(rep, ow, "quat_wxyz", pose["quat_wxyz"], T_wo)

            if not pose.get("pose_source"):
                rep.error(ow, "pose_source",
                          "pose_source 누락 — FoundationPose 결과인지 silhouette fit 인지 알 수 없다")

            for cid, Tc in (pose.get("T_cam_object") or {}).items():
                check_pose_matrix(rep, f"{ow}/{cid}", "T_cam_object", Tc)

            # exporter 가 기록한 재투영 IoU (mesh+pose 를 실제 mask 에 다시 투영한 값)
            ious = pose.get("reprojection_iou") or {}
            for cid, iou in ious.items():
                if iou is None or not np.isfinite(iou):
                    rep.error(f"{ow}/{cid}", "reprojection_iou", "IoU 가 NaN/None")
                    trial_ok = False
                elif min_iou is not None and iou < min_iou:
                    trial_ok = False
                    stats["bad_poses"].append({"trial": tid, "object": oid, "camera": cid,
                                               "reason": f"재투영 IoU {iou:.3f} < {min_iou}"})
                    rep.error(f"{ow}/{cid}", "reprojection_iou",
                              f"재투영 IoU {iou:.3f} < 임계 {min_iou} "
                              f"— pose/scale/mask 정합이 깨졌다")

        if trial_ok:
            stats["valid_trials"] += 1
        else:
            stats["excluded_trials"].append(tid)

    return rep, stats


def image_shape(path: Path) -> tuple[int, int] | None:
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return None if img is None else (int(img.shape[0]), int(img.shape[1]))


def mask_pixels(path: Path) -> int | None:
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return None if img is None else int((img > 0).sum())


# --------------------------------------------------------------------------- #
# 보고서
# --------------------------------------------------------------------------- #

def render_text(rep: Report, stats: dict, ready: bool) -> str:
    L = []
    L.append("=" * 72)
    L.append("Isaac Sim handoff — 검증 보고서")
    L.append("=" * 72)
    L.append(f"package        : {stats['package']}")
    L.append(f"객체 수        : {stats['num_objects']}"
             + (f"  (+ pending {len(stats['pending_objects'])}: "
                f"{', '.join(stats['pending_objects'])})" if stats["pending_objects"] else ""))
    L.append(f"trial 수       : {stats['num_trials']}")
    L.append(f"카메라 수      : {stats['num_cameras']}")
    L.append(f"유효 trial     : {stats['valid_trials']}")
    L.append(f"제외된 trial   : {len(stats['excluded_trials'])} {stats['excluded_trials'] or ''}")
    L.append(f"수행한 검사    : {rep.checks_run}")
    L.append(f"error          : {len(rep.errors)}")
    L.append(f"warning        : {len(rep.warnings)}")
    L.append("")

    if rep.errors:
        L.append("-" * 72)
        L.append("ERROR — Isaac Sim 실행 전에 반드시 수정")
        L.append("-" * 72)
        for e in rep.errors:
            L.append(f"  [{e['where']}] {e['field']}")
            L.append(f"      {e['message']}")

    if rep.warnings:
        L.append("-" * 72)
        L.append("WARNING — 실행은 가능하지만 결과 해석에 영향")
        L.append("-" * 72)
        for w in rep.warnings:
            L.append(f"  [{w['where']}] {w['field']}")
            L.append(f"      {w['message']}")

    if stats["missing_files"]:
        L.append("-" * 72)
        L.append("누락 파일")
        for f in stats["missing_files"]:
            L.append(f"  - {f}")

    if stats["bad_poses"]:
        L.append("-" * 72)
        L.append("잘못된 pose")
        for p in stats["bad_poses"]:
            L.append(f"  - {p}")

    if stats["bad_masks"]:
        L.append("-" * 72)
        L.append("잘못된 mask")
        for m in stats["bad_masks"]:
            L.append(f"  - {m}")

    if stats["todo_fields"]:
        L.append("-" * 72)
        L.append("TODO — 사용자가 채워야 하는 값 (null)")
        for t in stats["todo_fields"]:
            L.append(f"  - {t}")

    L.append("=" * 72)
    L.append(f"Isaac Sim 실행 가능 여부: {'YES' if ready else 'NO'}")
    if not ready:
        L.append("  → 위 ERROR 를 먼저 해결한 뒤 export 를 다시 실행하라.")
    L.append("=" * 72)
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", type=Path, default=Path("."),
                    help="handoff package root (manifest.yaml 이 있는 폴더)")
    ap.add_argument("--min_iou", type=float, default=None,
                    help="재투영 IoU 하한. 생략하면 manifest 의 quality_gates 값을 쓴다.")
    ap.add_argument("--json", type=Path, default=None, help="JSON 보고서 저장 경로")
    ap.add_argument("--txt", type=Path, default=None, help="텍스트 보고서 저장 경로")
    args = ap.parse_args()

    root = args.package.resolve()
    if not (root / "manifest.yaml").exists():
        print(f"[FATAL] manifest.yaml 이 없다: {root}", file=sys.stderr)
        return 2

    min_iou = args.min_iou
    if min_iou is None:
        manifest = yaml.safe_load((root / "manifest.yaml").read_text())
        min_iou = ((manifest.get("quality_gates") or {}).get("min_reprojection_iou"))

    rep, stats = validate(root, min_iou=min_iou)
    ready = len(rep.errors) == 0

    payload = {
        **stats,
        "checks_run": rep.checks_run,
        "errors": rep.errors,
        "warnings": rep.warnings,
        "ready_for_isaac_sim": ready,
    }
    text = render_text(rep, stats, ready)
    print(text)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.txt:
        args.txt.parent.mkdir(parents=True, exist_ok=True)
        args.txt.write_text(text)

    return 0 if ready else 1


if __name__ == "__main__":
    sys.exit(main())
