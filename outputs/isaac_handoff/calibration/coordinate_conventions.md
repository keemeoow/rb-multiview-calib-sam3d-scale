# 좌표계와 단위 (자동 생성 — 고치지 말 것)

이 파일은 `scripts/export_isaac_handoff.py` 가 실제 export 값으로 생성한다.
사람이 손으로 고치면 package 와 어긋난다.

## 프레임

| 이름 | 정의 |
|---|---|
| `world` | **`cam0`** 이다. 로봇 base 가 아니다. `Obj_Step1_capture_object.py` 가 reference 카메라를 world 로 잡고 dump 한다. |
| `camera` | OpenCV 규약: **x=오른쪽, y=아래, z=광축 전방**. |
| `object` | `objects/<id>/<id>_cad_scaled.glb` 의 로컬 좌표계. AABB 중심이 원점. |

## 변환

모든 4x4 는 **row_major**, `p_dst = T @ p_src` (열벡터 오른쪽 곱).

```
T_world_object = T_world_cam @ T_cam_object
T_cam_object   = inv(T_world_cam) @ T_world_object
```

* `T_world_cam` — `calibration/cameras.yaml` 의 `T_world_cam`. **camera → world** 방향이다.
  `cam{i}_T_cam_to_world.txt` 를 그대로 읽은 값이다. 역방향으로 쓰면 객체가 엉뚱한 곳에 뜬다.
* `T_world_object` — `trials/<trial>/poses/<object>.json`.
* `T_cam_object` — 같은 파일의 카메라별 값. 위 식으로 이미 계산해 두었다.

## 단위

* 길이: **m** (pose translation, mesh vertex, size 전부).
* depth PNG: uint16, **mm**. meter 로 쓰려면 `depth_scale = 0.001` 를 곱한다.
* mass: kg. (미측정이면 `null`)

## Quaternion

**`wxyz` (scalar-first)** — `quat_wxyz` 키 이름 그대로다.
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

world 는 `cam0` (카메라 좌표계) 이라 **+Y 가 아래를 향한다.** USD 기본 up-axis 는 +Z 다.
stage 를 그대로 만들면 객체가 옆으로 누워 보인다. 두 가지 중 하나를 택하라.

1. stage up-axis 를 이 world 에 맞추지 말고, **모든 pose 를 그대로 쓰고 중력 방향만
   world 기준으로 계산**한다 (물리 없는 렌더링/alignment 평가에는 이걸로 충분하다).
2. 물리(grasp ablation)를 돌리려면 world → Isaac world 회전 `T_isaacworld_world` 를 한 번
   정의해 모든 pose 에 왼쪽 곱한다. 이 행렬은 **아직 정해지지 않았다 (README §17 TODO)** —
   테이블 평면을 어떻게 world 에 정렬할지는 Isaac 담당자가 정한다.
