#!/usr/bin/env python3
"""평가 공용 모듈 — 로딩, sim 마스크 재렌더링, 지표 계산.

기존 추정 코드(Obj_Step3*, _silhouette_fit)는 **수정하지 않고 읽기만** 한다.
정합 엔진(_silhouette_fit)의 렌더러를 그대로 재사용해, 평가가 파이프라인과
같은 방식으로 실루엣을 만든다.

sim 마스크 재렌더링 근거
------------------------
IoU 는 결과 JSON 에 per_view_iou 로 이미 저장돼 있지만, contour distance 는 저장돼
있지 않아 sim 마스크를 다시 만들어야 한다. 재현식은 방법마다 다르다:

  oracle_cad     : V = CAD 정점,  s = scale_cad_to_world,  (R,t) = T_world_cad_4x4
                   -> Obj_Step3d_compare_gt.overlay() 와 동일. 모호함 없음.

  baseline_sam3d : 저장된 *_sam3d_scaled.glb 는 이미 실척(비등방 포함)이지만,
                   export 시 AABB 중심으로 평행이동됐다 (export_scaled_mesh).
                   반면 T_world_mesh_4x4 는 **OBB 중심이 원점인 프레임** 기준이다.
                   OBB 중심이 원점이라는 성질로 offset 을 복원한다:
                       V_fit = V_glb - obb_center(V_glb),  s = 1.0
                   T_shape 에서 저장된 per_view_iou 와 소수점 4자리까지 일치함을 확인.

재현 정확도는 recompute_iou 로 검증해 CSV/리포트에 남긴다 (GLB float32 왕복 때문에
물체에 따라 ~0.001 오차가 남을 수 있다). 메인 IoU 는 결과 파일의 저장값을 쓴다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import trimesh
import yaml
from scipy.spatial import cKDTree

# repo root 를 import 경로에 넣어 기존 엔진을 재사용한다
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from _silhouette_fit import View, obb_frame, render_silhouette  # noqa: E402
from _silhouette_fit import per_view_iou as _engine_per_view_iou  # noqa: E402

METHODS = ["baseline_sam3d", "oracle_cad"]
METHOD_LABEL = {"baseline_sam3d": "Baseline (SAM3D)", "oracle_cad": "Oracle (GT CAD)"}


# ============================================================
# 설정 / 경로
# ============================================================

def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    cfg["_root"] = _ROOT
    # 물체마다 감산 목표를 들고 다니게 해 load_fit 이 파이프라인과 동일하게 재현하도록 한다
    for o in cfg.get("objects", []):
        o["_decimate_faces"] = cfg.get("decimate_faces", 30000)
    return cfg


def decimate_like_pipeline(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    """Obj_Step3_sam3d_scale.decimate_mesh 와 **동일한** open3d quadric decimation.

    trimesh 의 simplify_quadric_decimation 은 다른 라이브러리(fast_simplification)를
    쓰므로 결과가 달라진다. 정합 때와 같은 정점을 얻어야 저장된 per_view_iou 를
    재현할 수 있어 open3d 로 맞춘다.
    """
    if target_faces <= 0 or len(mesh.faces) <= target_faces:
        return mesh
    import open3d as o3d
    me = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.ascontiguousarray(mesh.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.ascontiguousarray(mesh.faces, dtype=np.int32)))
    dec = me.simplify_quadric_decimation(int(target_faces))
    V = np.asarray(dec.vertices, dtype=np.float64)
    F = np.asarray(dec.triangles)
    if len(F) == 0:
        return mesh
    return trimesh.Trimesh(vertices=V, faces=F, process=False)


def rp(rel) -> Path:
    """repo root 기준 상대경로 -> 절대경로."""
    p = Path(rel)
    return p if p.is_absolute() else _ROOT / p


# ============================================================
# 관측 로딩
# ============================================================

def load_views(capture_dir, mask_dir, cameras):
    """카메라별 (View, rgb_path, mask_path). 파일이 없으면 그 카메라는 제외."""
    out = {}
    cap, mk = rp(capture_dir), rp(mask_dir)
    for cid in cameras:
        K_p, T_p = cap / f"{cid}_K.txt", cap / f"{cid}_T_cam_to_world.txt"
        m_p, rgb_p = mk / f"{cid}_mask.png", cap / f"{cid}_rgb.png"
        if not (K_p.exists() and T_p.exists() and m_p.exists()):
            continue
        mask = cv2.imread(str(m_p), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        mask = mask > 127                       # IoU 전 binary 화
        if not mask.any():                      # 빈 마스크는 지표가 정의되지 않는다
            continue
        K = np.loadtxt(K_p)
        T = np.loadtxt(T_p)
        out[cid] = dict(view=View(K, T, mask), mask=mask,
                        rgb_path=rgb_p if rgb_p.exists() else None, mask_path=m_p)
    return out


def resolve_sibling(json_path: Path, stored: Optional[str]) -> Optional[Path]:
    """결과 JSON 에 적힌 산출물 경로. 폴더가 이동됐으면 JSON 옆에서 같은 이름을 찾는다."""
    if not stored:
        return None
    p = rp(stored)
    if p.exists():
        return p
    sib = Path(json_path).parent / Path(stored).name
    return sib if sib.exists() else None


# ============================================================
# 방법별 fit 로딩 + sim 마스크 재렌더링
# ============================================================

def load_fit(obj_cfg: dict, method: str):
    """(V, F, s, R, t, meta) 또는 None. V/s/R/t 는 render_silhouette 인자 규약."""
    if method == "oracle_cad":
        spec = obj_cfg.get("oracle")
        if not spec:
            # 정답 CAD 가 없는 물체 (예: kettle). oracle 은 정의 자체가 성립하지 않는다.
            return None, "정답 CAD 없음 — 이 물체는 baseline 전용"
        jp = rp(spec["fit_json"])
        mp = rp(spec["mesh"])
        if not jp.exists() or not mp.exists():
            return None, f"oracle 파일 없음 ({jp if not jp.exists() else mp})"
        d = json.loads(jp.read_text())
        mesh = trimesh.load(str(mp), force="mesh")
        V = np.asarray(mesh.vertices, float)
        T = np.array(d["T_world_cad_4x4"], float)
        meta = dict(json_path=jp, extents=d["extents_mm_sorted_desc"],
                    per_view_iou=d.get("per_view_iou"), mean_iou=d.get("mean_iou"),
                    scale=d.get("scale_cad_to_world"), shape_ok=None,
                    scale_vec=None, method_name=d.get("method"))
        return (V, np.asarray(mesh.faces), float(d["scale_cad_to_world"]),
                T[:3, :3], T[:3, 3], meta), None

    if method == "baseline_sam3d":
        jp = rp(obj_cfg["baseline"]["size_json"])
        if not jp.exists():
            return None, f"baseline 파일 없음 ({jp})"
        d = json.loads(jp.read_text())
        T = np.array(d["T_world_mesh_4x4"], float)
        meta = dict(json_path=jp, extents=d["extents_mm_sorted_desc"],
                    per_view_iou=d.get("per_view_iou"), mean_iou=d.get("mean_iou"),
                    scale=d.get("scale_mesh_to_world"), shape_ok=d.get("shape_ok_by_iou"),
                    scale_vec=d.get("scale_vec"), method_name=d.get("method"))

        # (1) 정확 재현: 원본 SAM3D 메시 + 파이프라인과 동일한 감산 + scale_vec.
        #     T_world_mesh 는 이 프레임(OBB 중심 원점, 축별 스케일 적용)을 기준으로 저장됐다.
        mp = obj_cfg["baseline"].get("mesh") or d.get("mesh")
        mp = resolve_sibling(jp, mp) or (rp(mp) if mp else None)
        sv = d.get("scale_vec")
        if mp is not None and mp.exists() and sv:
            mesh = decimate_like_pipeline(trimesh.load(str(mp), force="mesh"),
                                          int(obj_cfg.get("_decimate_faces", 30000)))
            V = np.asarray(mesh.vertices, float)
            F = np.asarray(mesh.faces)
            c_m, R_m, _ = obb_frame(V)
            Vc = (((V - c_m) @ R_m) * np.asarray(sv, float)) @ R_m.T
            meta["reconstruction"] = "source_mesh+decimate+scale_vec (exact)"
            return (Vc, F, 1.0, T[:3, :3], T[:3, 3], meta), None

        # (2) 대체: 내보낸 scaled_glb. export 가 AABB 중심으로 옮겼으므로 OBB 중심으로 되돌린다.
        #     min_volume_obb 가 비등방 스케일 후 다른 축을 찾을 수 있어 복잡 메시에서는
        #     오차가 남는다 (재현 delta 로 확인 가능).
        gp = resolve_sibling(jp, d.get("scaled_glb"))
        if gp is None:
            return None, (f"baseline mesh/scaled_glb 를 찾을 수 없음 "
                          f"(mesh={d.get('mesh')}, scaled_glb={d.get('scaled_glb')})")
        mesh = trimesh.load(str(gp), force="mesh")
        V = np.asarray(mesh.vertices, float)
        V = V - obb_frame(V)[0]
        meta["reconstruction"] = "scaled_glb+obb_recenter (approx)"
        return (V, np.asarray(mesh.faces), 1.0, T[:3, :3], T[:3, 3], meta), None

    raise ValueError(f"unknown method: {method}")


def render_sim_mask(fit, view: View) -> np.ndarray:
    """네이티브 해상도(ss=1) sim 마스크. contour distance/면적은 픽셀 공간에서 재야 한다."""
    V, F, s, R, t = fit[0], fit[1], fit[2], fit[3], fit[4]
    return render_silhouette(V, F, s, R, t, view, ss=1).astype(bool)


def engine_iou(fit, views_list):
    """엔진과 **동일 조건**(ss=SUPERSAMPLE)으로 계산한 per-view IoU.

    결과 JSON 의 per_view_iou 는 엔진이 슈퍼샘플로 계산한 값이므로, 재현 검증은
    반드시 같은 조건으로 해야 한다 (ss=1 로 비교하면 래스터화 차이만큼 어긋난다).
    """
    V, F, s, R, t = fit[0], fit[1], fit[2], fit[3], fit[4]
    return _engine_per_view_iou(V, F, s, R, t, views_list)


# ============================================================
# 지표
# ============================================================

def mask_iou(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """binary IoU. 합집합이 비면 정의되지 않음(None)."""
    a, b = a.astype(bool), b.astype(bool)
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return None
    return float(np.logical_and(a, b).sum()) / union


def _contour_points(mask: np.ndarray) -> np.ndarray:
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return np.zeros((0, 2))
    return np.vstack([c.reshape(-1, 2) for c in cs]).astype(np.float64)


def bbox_wh(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None, None
    return float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)


def contour_distance(real: np.ndarray, sim: np.ndarray):
    """양방향 평균 최근접 거리(px) 와 real bbox 대각선 정규화 값.

    D = 0.5*[ mean_{p in C_real} min_q ||p-q|| + mean_{q in C_sim} min_p ||q-p|| ]
    D_norm = D / sqrt(w_real^2 + h_real^2)
    """
    cr, cs = _contour_points(real), _contour_points(sim)
    if len(cr) == 0 or len(cs) == 0:
        return None, None, None
    d_rs = cKDTree(cs).query(cr)[0].mean()
    d_sr = cKDTree(cr).query(cs)[0].mean()
    d = 0.5 * float(d_rs + d_sr)
    w, h = bbox_wh(real)
    if not w or not h:
        return d, None, None
    diag = float(np.hypot(w, h))
    if diag <= 0:
        return d, None, None
    n = d / diag
    return d, n, n * 100.0


# ============================================================
# 축 대응
# ============================================================

_PERMS6 = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]


def match_axes(est_desc, gt_desc):
    """GT 축 대응 선택.

    파이프라인과 GT 모두 내림차순(크기 rank)이라 identity 가 기본 대응이다.
    정렬된 두 수열에서 identity 는 sum|차이| 를 최소화하는 최적 순열이므로
    (rearrangement inequality), 6개 순열 탐색 결과와 일치해야 한다.
    검증 목적으로 둘 다 계산해 일치 여부를 기록한다.

    GT 미확정(None) 축은 비교에서 제외한다.
    반환: dict(est_matched, method, perm, agrees_with_rank)
    """
    est = [float(x) for x in est_desc]
    idx_known = [i for i, g in enumerate(gt_desc) if g is not None]
    if not idx_known:
        return dict(est_matched=est, method="rank_descending",
                    perm=(0, 1, 2), agrees_with_rank=True)

    def cost(perm):
        v = [abs(est[perm[i]] - gt_desc[i]) for i in idx_known]
        return float(np.mean(v))

    best = min(_PERMS6, key=cost)
    rank_perm = (0, 1, 2)
    agrees = abs(cost(best) - cost(rank_perm)) < 1e-9
    # 프로젝트가 크기 rank 대응을 보장하므로 rank 를 우선 사용하고, 일치 여부만 기록
    return dict(est_matched=[est[i] for i in rank_perm],
                method="rank_descending",
                perm=rank_perm, agrees_with_rank=agrees,
                best_perm=best, best_perm_cost=cost(best), rank_cost=cost(rank_perm))


def dim_errors(est_matched, gt_desc):
    """축별 |오차|(mm), 평균, 상대오차(%). GT 미확정 축은 None 이고 평균에서 제외."""
    abs_e = [None if g is None else abs(float(e) - float(g))
             for e, g in zip(est_matched, gt_desc)]
    known = [x for x in abs_e if x is not None]
    mean_e = float(np.mean(known)) if known else None
    rel = [None if (g is None or not g) else abs(float(e) - float(g)) / float(g) * 100.0
           for e, g in zip(est_matched, gt_desc)]
    rel_known = [x for x in rel if x is not None]
    mean_rel = float(np.mean(rel_known)) if rel_known else None
    return abs_e, mean_e, mean_rel, len(known)


def mean_std(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    if not v:
        return None, None, 0
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0), len(v)
