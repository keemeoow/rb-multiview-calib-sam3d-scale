#!/usr/bin/env python3
"""
Calib_Step4ee_compose_all_cams_to_base.py

cam1 의 hand-to-eye 결과(T_R_C1)와 멀티캠 캘리브(T_C1_Ci)를 합성하여
**모든 카메라의 base 좌표계 transform** 을 한 JSON 으로 저장.

  T_R_Ci = T_R_C1 @ T_C1_Ci   (i = 0, 1, 2, ...)

FoundationPose / depth fusion 등에서 객체 pose 를 base 좌표계로 변환할 때 사용.

[입력]
  --handeye_result : T_R_C1.json  (Calib_Step3ee 출력)
  --multicam_dir   : T_C1_C0.npy, T_C1_C2.npy 가 있는 폴더
                     (예: ./data/static_cams_session_01/calib_out_cube)

[출력]
  --output         : T_R_Ci_all.json
    {
      "ref_cam_idx": 1,
      "T_R_C0": [4x4 list],
      "T_R_C1": [4x4 list],
      "T_R_C2": [4x4 list],
      "source": {...}
    }

[실제 사용 명령어 — 저장만]
python Calib_Step4ee_all_cams_to_base.py \\
  --handeye_result ./data/handeye_session_01/T_R_C1.json \\
  --multicam_dir   ./data/static_cams_session_01/calib_out_cube \\
  --output         ./data/handeye_session_01/T_R_Ci_all.json

[실제 사용 명령어 — 저장 + matplotlib 3D 인터랙티브 창]
python Calib_Step4ee_all_cams_to_base.py \\
  --handeye_result ./data/handeye_session_01/T_R_C1.json \\
  --multicam_dir   ./data/static_cams_session_01/calib_out_cube \\
  --output         ./data/handeye_session_01/T_R_Ci_all.json \\
  --visualize \\
  --save_fig       ./data/handeye_session_01/cams_in_base.png
"""
import argparse
import json
import os
import re
from glob import glob

import numpy as np


# -------------------------------------------------------------------- #
# Matplotlib 3D 시각화
# -------------------------------------------------------------------- #
def visualize_cameras_in_base(T_R_Ci_all, ref_cam_idx,
                              save_path=None, show=True,
                              cam_axis_len_mm=80.0,
                              base_axis_len_mm=100.0,
                              frustum_depth_mm=120.0,
                              frustum_half_w_mm=60.0,
                              frustum_half_h_mm=45.0):
    """3 개 카메라 + base 좌표계 frame + 카메라 frustum 을 base 좌표계에 그림.

    축 색상 컨벤션: X=red, Y=green, Z=blue.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    def draw_frame(T, label, axis_len=cam_axis_len_mm, lw=2.5, label_color="k"):
        """T (4x4, meters) 의 원점과 X/Y/Z 축을 그리고 라벨 표시. 좌표는 mm 로 표시."""
        origin_m = T[:3, 3]
        origin_mm = origin_m * 1000.0
        axes_dirs = T[:3, :3]  # 컬럼이 각 축의 base-좌표 방향
        for k, color in enumerate(("r", "g", "b")):
            end_mm = origin_mm + axes_dirs[:, k] * axis_len
            ax.plot([origin_mm[0], end_mm[0]],
                    [origin_mm[1], end_mm[1]],
                    [origin_mm[2], end_mm[2]],
                    color=color, linewidth=lw)
        ax.scatter([origin_mm[0]], [origin_mm[1]], [origin_mm[2]],
                   color="black", s=25)
        ax.text(origin_mm[0], origin_mm[1], origin_mm[2] + axis_len * 0.25,
                label, color=label_color, fontsize=10, weight="bold")

    def draw_frustum(T, depth_mm=frustum_depth_mm,
                     hw_mm=frustum_half_w_mm, hh_mm=frustum_half_h_mm,
                     color="dimgray"):
        """카메라 frustum (원뿔 대신 직사각 피라미드) 을 base 좌표계로 그림."""
        # 카메라 local 4 corner at depth (camera +Z 방향)
        local_corners = np.array([
            [ hw_mm,  hh_mm, depth_mm],
            [-hw_mm,  hh_mm, depth_mm],
            [-hw_mm, -hh_mm, depth_mm],
            [ hw_mm, -hh_mm, depth_mm],
        ]) / 1000.0  # → meters
        # base 좌표계로 변환
        T_local = T  # T_R_Ci (cam → base)
        base_corners = (T_local[:3, :3] @ local_corners.T).T + T_local[:3, 3]
        base_corners_mm = base_corners * 1000.0
        origin_mm = T_local[:3, 3] * 1000.0
        # origin → 각 corner 4 lines + corner 끼리 닫는 4 lines
        segs = []
        for i in range(4):
            segs.append([origin_mm.tolist(), base_corners_mm[i].tolist()])
        for i in range(4):
            segs.append([base_corners_mm[i].tolist(),
                         base_corners_mm[(i + 1) % 4].tolist()])
        lc = Line3DCollection(segs, colors=color, linewidths=0.8, alpha=0.6)
        ax.add_collection3d(lc)

    # base frame
    draw_frame(np.eye(4), "base (robot)",
               axis_len=base_axis_len_mm, lw=3.0, label_color="navy")

    # 각 카메라
    palette = {0: "darkorange", 1: "crimson", 2: "purple"}
    for ci in sorted(T_R_Ci_all.keys()):
        T = T_R_Ci_all[ci]
        label = f"cam{ci}" + (" (ref/HE)" if ci == ref_cam_idx else "")
        draw_frame(T, label)
        draw_frustum(T, color=palette.get(ci, "gray"))

    # 카메라 위치들을 base xy 평면(z=0)에 점 + 수직선으로 표시 (높이 가시화)
    for ci, T in T_R_Ci_all.items():
        pos_mm = T[:3, 3] * 1000.0
        ax.plot([pos_mm[0], pos_mm[0]], [pos_mm[1], pos_mm[1]],
                [0, pos_mm[2]], "k:", alpha=0.3, linewidth=0.7)
        ax.scatter([pos_mm[0]], [pos_mm[1]], [0],
                   color="lightgray", s=15, marker="x")

    # base xy plane grid (workspace 짐작용)
    all_pos = np.array([T[:3, 3] for T in T_R_Ci_all.values()]) * 1000.0
    pad = 200
    xmin, xmax = float(all_pos[:, 0].min()) - pad, float(all_pos[:, 0].max()) + pad
    ymin, ymax = float(all_pos[:, 1].min()) - pad, float(all_pos[:, 1].max()) + pad
    xs = np.linspace(xmin, xmax, 5)
    ys = np.linspace(ymin, ymax, 5)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)
    ax.plot_wireframe(X, Y, Z, color="lightgray", linewidth=0.3, alpha=0.5)

    # 축 라벨 및 동등 비율
    ax.set_xlabel("X (mm) [base]")
    ax.set_ylabel("Y (mm) [base]")
    ax.set_zlabel("Z (mm) [base]")
    ax.set_title(f"Cameras in base frame  (ref=cam{ref_cam_idx})")

    # 동등 비율 (matplotlib 3D 에는 set_box_aspect 가 가장 정확)
    zmin = min(0.0, float(all_pos[:, 2].min()) - 50)
    zmax = max(float(all_pos[:, 2].max()) + 100, 200)
    span_x = xmax - xmin
    span_y = ymax - ymin
    span_z = zmax - zmin
    try:
        ax.set_box_aspect((span_x, span_y, span_z))
    except Exception:
        pass
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.view_init(elev=22, azim=-60)

    # 범례 (X=red, Y=green, Z=blue 컨벤션)
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color="r", lw=2, label="X axis"),
        Line2D([0], [0], color="g", lw=2, label="Y axis"),
        Line2D([0], [0], color="b", lw=2, label="Z axis (cam +Z = optical axis)"),
    ]
    ax.legend(handles=legend_elems, loc="upper left", fontsize=8)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"[SAVE] {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--handeye_result", required=True,
                    help="T_R_C{ref}.json (Calib_Step3ee 출력)")
    ap.add_argument("--multicam_dir", required=True,
                    help="T_C{ref}_C{i}.npy 들이 있는 폴더")
    ap.add_argument("--output", required=True)

    ap.add_argument("--visualize", action="store_true",
                    help="저장 후 matplotlib 3D 창으로 시각화")
    ap.add_argument("--save_fig", default=None,
                    help="시각화 png 저장 경로 (--visualize 없이도 저장 가능)")
    args = ap.parse_args()

    # 1) Hand-eye 결과 로드
    with open(args.handeye_result) as f:
        he = json.load(f)
    ref_cam_idx = int(he["ref_cam_idx"])
    T_R_Cref = np.array(he["transformation_matrix_camera_to_robot"]).reshape(4, 4)
    print(f"[INFO] ref_cam_idx = {ref_cam_idx}")
    print(f"[INFO] T_R_C{ref_cam_idx} (hand-eye) loaded from {args.handeye_result}")
    print(f"       cam{ref_cam_idx} in base (mm): "
          f"({T_R_Cref[0,3]*1000:.1f}, {T_R_Cref[1,3]*1000:.1f}, {T_R_Cref[2,3]*1000:.1f})")
    if "consistency" in he:
        c = he["consistency"]
        print(f"       hand-eye residual mean = {c.get('trans_residual_mean_mm', 0):.2f} mm, "
              f"rotation std = {c.get('rotation_std_deg', 0):.2f}°")

    # 2) 멀티캠 변환 로드 (T_C{ref}_C{i}.npy 패턴)
    pat = re.compile(rf"T_C{ref_cam_idx}_C(\d+)\.npy$")
    npy_files = sorted(glob(os.path.join(args.multicam_dir, f"T_C{ref_cam_idx}_C*.npy")))
    T_Cref_Ci_map = {}
    for p in npy_files:
        m = pat.search(p)
        if not m:
            continue
        ci = int(m.group(1))
        T_Cref_Ci_map[ci] = np.load(p).astype(np.float64)
        print(f"[INFO] T_C{ref_cam_idx}_C{ci}  loaded ({os.path.basename(p)})")

    # 3) 합성: T_R_Ci = T_R_Cref @ T_Cref_Ci
    T_R_Ci_all = {ref_cam_idx: T_R_Cref.copy()}
    for ci, T_ref_i in T_Cref_Ci_map.items():
        T_R_Ci_all[ci] = T_R_Cref @ T_ref_i

    print(f"\n=== base 좌표계 카메라 위치 ===")
    for ci in sorted(T_R_Ci_all.keys()):
        T = T_R_Ci_all[ci]
        t_mm = T[:3, 3] * 1000.0
        # 카메라 광축 방향 (cam +Z 가 base 에서 어디 향하는지)
        z_axis = T[:3, 2]
        tag = " (ref/hand-eye)" if ci == ref_cam_idx else " (composed)"
        print(f"  cam{ci}: pos=({t_mm[0]:>7.1f}, {t_mm[1]:>7.1f}, {t_mm[2]:>7.1f}) mm  "
              f"+Z=({z_axis[0]:>+.3f}, {z_axis[1]:>+.3f}, {z_axis[2]:>+.3f}){tag}")

    # 4) 저장
    out = {
        "ref_cam_idx": ref_cam_idx,
        "T_R_Ci": {str(ci): T.tolist() for ci, T in T_R_Ci_all.items()},
        # 별칭 — 일부 코드에서 transformation_matrix_camera_to_robot_cam{ci} 형태로 찾을 수 있어
        # 평면 형태도 함께 제공
        "T_R_C0": T_R_Ci_all.get(0, np.eye(4)).tolist() if 0 in T_R_Ci_all else None,
        "T_R_C1": T_R_Ci_all.get(1, np.eye(4)).tolist() if 1 in T_R_Ci_all else None,
        "T_R_C2": T_R_Ci_all.get(2, np.eye(4)).tolist() if 2 in T_R_Ci_all else None,
        "source": {
            "handeye_result": os.path.abspath(args.handeye_result),
            "multicam_dir":   os.path.abspath(args.multicam_dir),
            "handeye_method": he.get("method"),
            "handeye_consistency": he.get("consistency"),
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVE] {args.output}")
    print(f"[NEXT] FoundationPose 등에서 다음과 같이 사용:")
    print(f"       T_R_Ci = np.array(json.load(open('{args.output}'))['T_R_C{ref_cam_idx}'])")
    print(f"       T_R_obj = T_R_Ci @ T_Ci_obj   # T_Ci_obj 는 cam{ref_cam_idx} 기준 FoundationPose output")

    # 5) 시각화 (옵션)
    if args.visualize or args.save_fig:
        print()
        visualize_cameras_in_base(
            T_R_Ci_all, ref_cam_idx,
            save_path=args.save_fig,
            show=bool(args.visualize),
        )


if __name__ == "__main__":
    main()
