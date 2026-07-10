"""OBB 시각화: open3d hull-PCA OBB vs 최소부피 OBB.

obj2 처럼 등방적인(거의 정육면체) 물체에서 open3d 의 get_oriented_bounding_box() 는
볼록껍질 PCA 축을 쓰기 때문에 주축이 대각선으로 잡히고 상자가 크게 부푼다.
이 스크립트는 그 차이를 세 장의 그림으로 보여준다.

  viz_overlay.png : cam0/1/2 RGB 위에 두 상자 + 재투영된 점군
  viz_crops.png   : cam1 확대 크롭 + 치수/부피비
  viz_3d.png      : 점군과 두 상자의 3D 뷰

실행 (레포 루트에서):
  python data/obb_viz/make_obb_viz.py

점군은 clouds/ 에 동봉되어 있다. 다시 만들려면:
  python Obj_Step3_sam3d_pose.py --data_dir data/capture_obj --mask_dir data/masks \
      --out_dir data/obb_viz/clouds --depth_scale 0.001 \
      --mask_close_px 5 --mask_erode_px 3 --keep_largest_cc --use_oriented_bbox
"""
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CAP = ROOT / "data" / "capture_obj"
CLOUDS = HERE / "clouds"

OBJS = ["obj1", "obj2"]
EDGES = [(0,1),(1,3),(3,2),(2,0),(4,5),(5,7),(7,6),(6,4),(0,4),(1,5),(2,6),(3,7)]
RED, GRN = (60, 60, 235), (80, 220, 80)      # BGR


def corners_from(center, R, extent):
    """8 corners. R columns are the box axes, extent holds full side lengths."""
    s = np.array([[i, j, k] for i in (-.5, .5) for j in (-.5, .5) for k in (-.5, .5)])
    return center + (s * extent) @ R.T


def boxes(pts):
    """(open3d hull-PCA OBB, minimum-volume OBB), each as (corners, extent)."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    ob = pcd.get_oriented_bounding_box(robust=True)
    o3d_c = corners_from(np.asarray(ob.center), np.asarray(ob.R), np.asarray(ob.extent))

    T, ext = trimesh.bounds.oriented_bounds(pts)    # T maps world -> box
    Tinv = np.linalg.inv(T)
    mv_c = corners_from(Tinv[:3, 3], Tinv[:3, :3], np.asarray(ext))
    return (o3d_c, np.asarray(ob.extent)), (mv_c, np.asarray(ext))


def project(P, K, T_cam_to_world):
    """world = cam0 frame (see capture_obj/calib_info.json)."""
    W = np.linalg.inv(T_cam_to_world)
    pc = (W[:3, :3] @ P.T).T + W[:3, 3]
    uv = (K @ pc.T).T
    return uv[:, :2] / uv[:, 2:3]


def draw_box(img, uv, color, thick=2):
    for a, b in EDGES:
        cv2.line(img, tuple(np.round(uv[a]).astype(int)),
                 tuple(np.round(uv[b]).astype(int)), color, thick, cv2.LINE_AA)


def cam(i):
    return (np.loadtxt(CAP / f"cam{i}_K.txt"),
            np.loadtxt(CAP / f"cam{i}_T_cam_to_world.txt"))


def main():
    clouds = {o: np.asarray(o3d.io.read_point_cloud(str(CLOUDS / f"{o}_cloud_clean.ply")).points)
              for o in OBJS}
    B = {o: boxes(clouds[o]) for o in OBJS}

    # 그림을 믿기 전에: 점군을 각 카메라로 재투영해 마스크 안에 떨어지는지 확인한다.
    for i in range(3):
        K, T = cam(i)
        for o in OBJS:
            m = cv2.imread(str(ROOT / "data" / "masks" / o / f"cam{i}_mask.png"), 0) > 127
            uv = project(clouds[o], K, T)
            u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
            ok = (u >= 0) & (u < m.shape[1]) & (v >= 0) & (v < m.shape[0])
            print(f"[check] cam{i} {o}: {m[v[ok], u[ok]].mean()*100:.1f}% of reprojected "
                  f"points fall inside the mask")

    # ---- 1) 3-view overlay ----
    tiles = []
    for i in range(3):
        K, T = cam(i)
        img = cv2.imread(str(CAP / f"cam{i}_rgb.png"))
        for o in OBJS:
            (o3c, _), (mvc, _) = B[o]
            for u, v in project(clouds[o], K, T).astype(int):
                cv2.circle(img, (u, v), 1, (255, 255, 0), -1)
            draw_box(img, project(o3c, K, T), RED)
            draw_box(img, project(mvc, K, T), GRN)
        cv2.putText(img, f"cam{i}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 3)
        cv2.putText(img, f"cam{i}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 0), 1)
        tiles.append(img)

    strip = cv2.hconcat(tiles)
    legend = np.full((78, strip.shape[1], 3), 255, np.uint8)
    cv2.line(legend, (20, 26), (70, 26), RED, 3)
    cv2.putText(legend, "open3d get_oriented_bounding_box  (current pipeline)", (84, 32),
                cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.line(legend, (20, 58), (70, 58), GRN, 3)
    cv2.putText(legend, "trimesh minimum-volume OBB", (84, 64),
                cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.circle(legend, (620, 26), 3, (255, 255, 0), -1)
    cv2.putText(legend, "merged clean point cloud (reprojected)", (640, 32),
                cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(HERE / "viz_overlay.png"), cv2.vconcat([strip, legend]))

    # ---- 2) cam1 zoomed crops ----
    K, T = cam(1)
    base = cv2.imread(str(CAP / "cam1_rgb.png"))
    crops = []
    for o in OBJS:
        img = base.copy()
        (o3c, o3e), (mvc, mve) = B[o]
        draw_box(img, project(o3c, K, T), RED)
        draw_box(img, project(mvc, K, T), GRN)
        uv = project(np.vstack([o3c, mvc]), K, T)
        x0, y0 = np.clip(uv.min(0) - 30, 0, None).astype(int)
        x1, y1 = (uv.max(0) + 30).astype(int)
        crop = img[y0:min(y1, img.shape[0]), x0:min(x1, img.shape[1])]
        crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        e1, e2 = np.sort(o3e)[::-1] * 1000, np.sort(mve)[::-1] * 1000
        hdr = np.full((66, crop.shape[1], 3), 255, np.uint8)
        cv2.putText(hdr, o, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 0), 2)
        cv2.putText(hdr, f"open3d {e1[0]:.0f}x{e1[1]:.0f}x{e1[2]:.0f}mm   "
                         f"min-vol {e2[0]:.0f}x{e2[1]:.0f}x{e2[2]:.0f}mm   "
                         f"vol {np.prod(e1)/np.prod(e2):.2f}x",
                    (10, 52), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1, cv2.LINE_AA)
        crops.append(cv2.vconcat([hdr, crop]))
    h = max(c.shape[0] for c in crops)
    crops = [cv2.copyMakeBorder(c, 0, h - c.shape[0], 0, 12, cv2.BORDER_CONSTANT, value=(255, 255, 255))
             for c in crops]
    cv2.imwrite(str(HERE / "viz_crops.png"), cv2.hconcat(crops))

    # ---- 3) 3D ----
    fig = plt.figure(figsize=(13, 6))
    for i, o in enumerate(OBJS):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        P = clouds[o] * 1000
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=1, c="#1f77b4", alpha=.25, linewidths=0)
        (o3c, o3e), (mvc, mve) = B[o]
        for C, col, lab in ((o3c * 1000, "#d62728", "open3d OBB"),
                            (mvc * 1000, "#2ca02c", "min-volume OBB")):
            for j, (a, b) in enumerate(EDGES):
                ax.plot(*zip(C[a], C[b]), color=col, lw=1.6, label=lab if j == 0 else None)
        e1, e2 = np.sort(o3e)[::-1] * 1000, np.sort(mve)[::-1] * 1000
        ax.set_title(f"{o}\nopen3d {e1[0]:.0f}x{e1[1]:.0f}x{e1[2]:.0f}  |  "
                     f"min-vol {e2[0]:.0f}x{e2[1]:.0f}x{e2[2]:.0f} mm", fontsize=10)
        lim = np.vstack([P, o3c * 1000, mvc * 1000])
        c, r = lim.mean(0), (lim.max(0) - lim.min(0)).max() / 2 * 1.05
        ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)"); ax.set_zlabel("z (mm)")
        ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(str(HERE / "viz_3d.png"), dpi=130)

    print()
    for o in OBJS:
        (_, e1), (_, e2) = B[o]
        e1, e2 = np.sort(e1)[::-1] * 1000, np.sort(e2)[::-1] * 1000
        print(f"{o}: open3d {np.round(e1,1)}  min-vol {np.round(e2,1)} mm  "
              f"(vol {np.prod(e1)/np.prod(e2):.2f}x)")
    print(f"\nwrote {HERE}/viz_overlay.png, viz_crops.png, viz_3d.png")


if __name__ == "__main__":
    main()
