#!/usr/bin/env python3
"""_silhouette_fit.auto_w_depth 합성 검증.

카메라 간 depth 일치도로 w_depth 를 자동 산출하는 로직이
  - 일치(consistent) depth  -> 신뢰도 1 -> w_depth = w_max   (흰/무광에서 depth 활용)
  - 편향(biased)     depth  -> 신뢰도 0 -> w_depth = 0        (검은/광택에서 실루엣만)
로 동작하는지 확인한다. 편향은 점을 더 모아도 줄지 않으므로(분산과 달리), 카메라 간
불일치가 곧 depth 신뢰도의 지표가 된다.

numpy 필요. sam3d 환경에서:
  source sam3d_env_gb10.sh
  python test_auto_w_depth.py
"""
import numpy as np

from _silhouette_fit import auto_w_depth, cross_view_depth_disagreement


def make_cam(K, T, S, H, W):
    """world 점군 S 를 카메라(K, T=cam->world)에 투영해 (depth[m], mask) 생성."""
    Wc = np.linalg.inv(T)
    Xc = (Wc[:3, :3] @ S.T).T + Wc[:3, 3]
    z = Xc[:, 2]
    u = np.round(K[0, 0] * Xc[:, 0] / z + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * Xc[:, 1] / z + K[1, 2]).astype(int)
    ok = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    depth = np.zeros((H, W), np.float64)
    mask = np.zeros((H, W), bool)
    depth[v[ok], u[ok]] = z[ok]
    mask[v[ok], u[ok]] = True
    return depth, mask


def main():
    H, W = 240, 320
    K = np.array([[300.0, 0, 160.0], [0, 300.0, 120.0], [0, 0, 1.0]])

    # world 평면 z=0.5 m, x,y in [-0.08, 0.08] (상수-깊이 → 픽셀 양자화에 강건)
    g = np.arange(-0.08, 0.08, 0.0015)
    xx, yy = np.meshgrid(g, g)
    S = np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, 0.5)], axis=1)

    T0 = np.eye(4)
    T1 = np.eye(4)
    T1[0, 3] = 0.05                       # 옆으로 5cm 이동한 두 번째 카메라

    d0, m0 = make_cam(K, T0, S, H, W)
    d1, m1 = make_cam(K, T1, S, H, W)

    def cams(bias1_m=0.0):
        d1b = d1.copy()
        d1b[m1] += bias1_m                # cam1 depth 에만 계통 편향 주입
        return [
            {"K": K, "T": T0, "depth_m": d0, "mask": m0},
            {"K": K, "T": T1, "depth_m": d1b, "mask": m1},
        ]

    print("=== consistent depth (bias 0 mm) ===")
    delta = cross_view_depth_disagreement(cams(0.0))
    w, info = auto_w_depth(cams(0.0), w_max=20.0)
    assert delta is not None and delta < 1e-4, f"delta={delta}"
    assert abs(w - 20.0) < 1e-6, f"w={w}"
    print(f"  OK  disagreement={delta*1000:.4f} mm  w_depth={w}")

    print("=== biased depth (+6 mm on cam1) ===")
    w, info = auto_w_depth(cams(0.006), w_max=20.0)
    assert info["disagreement_mm"] > 5.0, info
    assert w == 0.0, f"w={w}"
    print(f"  OK  disagreement={info['disagreement_mm']:.2f} mm  w_depth={w}")

    print("=== partial bias (+3 mm) -> 중간 신뢰도 ===")
    w, info = auto_w_depth(cams(0.003), w_max=20.0)
    assert 8.0 < w < 18.0, (w, info)
    print(f"  OK  disagreement={info['disagreement_mm']:.2f} mm  "
          f"conf={info['confidence']:.3f}  w_depth={w:.2f}")

    print("=== single camera -> 평가 불가, 안전하게 0 ===")
    w, info = auto_w_depth(cams(0.0)[:1], w_max=20.0)
    assert w == 0.0 and info["disagreement_mm"] is None, info
    print(f"  OK  w_depth={w} (reason: {info['reason']})")

    print("\nALL PASS")


if __name__ == "__main__":
    main()
