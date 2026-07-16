# Isaac Sim handoff — 담당자 안내서

이 문서는 handoff package (`outputs/isaac_handoff/`) 안에 함께 들어간다.
**모든 경로는 package root 기준 상대경로다.** package 를 통째로 복사해 가면 그대로 동작한다.

이 README 에 적힌 명령 중 **`scripts/validate_handoff.py` 와 `scripts/inspect_trial.py` 두 개만
실제로 구현되어 있다.** Isaac Sim 쪽 스크립트는 아직 없다 — §6 에서 `TODO` 로 표시했고,
그것들을 만드는 게 담당자의 첫 작업이다. 없는 명령을 그럴듯하게 적어두지 않았다.

---

## 1. 목적

### handoff package 가 무엇인가

실제 카메라 3대로 찍은 RGB/depth/mask, 카메라 calibration, 그리고 **실물 객체의 3D 크기를
추정한 결과(metric scale + 6-DoF pose)** 를 Isaac Sim 에서 바로 쓸 수 있는 형태로 묶은 것이다.

### 사용자가 완료한 단계

1. 객체 mesh 준비 (`data/meshes/<id>.glb`)
2. 3D 프린트 출력물의 실제 치수 측정 (캘리퍼). **질량은 아직 미측정 — §17 참고**
3. 고정 카메라 3대 calibration (AprilTag cube + bundle adjustment)
4. 실제 RGB / depth 촬영, SAM2 로 객체 mask 생성
5. 다중뷰 실루엣 정합으로 **metric scale + pose** 추정
6. 실측 대비 크기 오차 정량화
7. 이 handoff package export

### Isaac Sim 담당자가 수행할 단계

* object asset import 및 USD 변환
* visual / collision geometry 설정
* 카메라 3대 생성 (실제와 동일 시점)
* object pose 및 scale 적용
* segmentation mask rendering
* Real-to-Sim alignment metric 생성
* scale sensitivity sweep
* grasp ablation
* 결과 CSV 및 figure 생성

### 검증하는 연구 질문

> **객체 크기 추정 오차가 시뮬레이션 grasp 성능을 얼마나 망가뜨리는가?**

현재 파이프라인의 크기 오차는 **평균 절대 1.19 mm (peg) / 0.87 mm (hole)**, 최대 1.89 mm 다
(`measurements/size_error.csv`). 이 정도 오차가 grasp 성공률에 유의미한 차이를 만드는지,
그리고 몇 % 부터 무너지는지를 scale sweep (§11) 과 grasp ablation (§12) 으로 확인한다.

### 객체 구성 — 총 7개 예정

| 객체 | 상태 |
|---|---|
| `peg`, `hole` | **준비 완료** (3D 프린트 출력물, 캘리퍼 실측 완료) |
| YCB 객체 5개 | **수집 예정** — 사용자가 촬영·추정 후 재-export 한다 |

**package 에 지금 몇 개가 들어 있는지는 `manifest.yaml` 이 정답이다.**
`objects:` 에 있는 것이 쓸 수 있는 객체이고, `pending_objects:` 는 아직 데이터가 없는 객체다.
객체가 5개뿐이라고 당황하지 말고 `handoff_validation_report.txt` 의 `pending` 줄을 보라.
객체가 늘어나도 **Isaac 쪽 코드는 바뀌지 않는다** — `manifest.yaml: objects` 를 순회하도록 짜라.
객체 id 는 `obj1` 같은 번호가 아니라 **`peg`, `hole`, `mustard` 같은 의미 있는 이름**이다.

---

## 2. 담당 범위

| 사용자 (완료) | Isaac Sim 담당자 (해야 함) |
|---|---|
| 카메라 calibration | object asset import / USD 변환 |
| RGB · depth · mask 수집 | visual / collision geometry |
| metric scale + pose 추정 | 카메라 생성 및 extrinsic 적용 |
| world frame 변환 · 다중뷰 융합 | object pose / scale 적용 |
| 실측 치수 입력 | segmentation mask rendering |
| 크기 오차 정량화 | Real-to-Sim alignment metric |
| handoff package export | scale sweep · grasp ablation · figure |

**FoundationPose 는 아직 돌리지 않았다.** 스크립트(`Obj_Step5_foundationpose_register.py`)는
사용자 repo 에 있지만 결과가 없어서, 현재 package 의 pose 는 전부
`pose_source: cad_silhouette_fit` 이다 (§5 참고). Isaac 쪽 작업에는 영향이 없다 —
pose 를 읽는 방법은 동일하다.

---

## 3. 지원 환경

저장소에서 **확인된** 것만 적는다.

| 항목 | 값 | 근거 |
|---|---|---|
| 카메라 | Intel RealSense 3대 (serial `314522062542`, `319522062138`, `912322060991`) | `calibration/cameras.yaml` |
| 해상도 | 640 × 480 (RGB / depth / mask 전부) | 〃 |
| depth | uint16 PNG, mm 단위 (`depth_scale = 0.001`) | `manifest.yaml` |
| package 검증 스크립트 | Python 3.11, `numpy` `pyyaml` `trimesh` `opencv-python` | `scripts/*.py` — **Isaac Sim 없이 실행됨** |
| mesh 포맷 | `.glb` (texture embed) | `objects/*/` |

아래는 **저장소에서 확인되지 않는다. 담당자가 정하고 §17 표에 채워라.**

* Isaac Sim 버전, Python 실행 방식 (`./python.sh` vs `isaacsim` 런처)
* OS / GPU 요구사항
* 필요한 extension 또는 package
* 로봇 asset, gripper asset, policy checkpoint
* environment variables

---

## 4. Handoff package 구조

```text
isaac_handoff/
├── manifest.yaml                     # 진입점. 모든 경로가 여기서 시작한다.
├── ISAAC_SIM_README.md               # 이 문서
├── handoff_validation_report.json    # export 직후 자동 검증 결과
├── handoff_validation_report.txt     # 같은 내용, 사람이 읽는 형식
├── calibration/
│   ├── cameras.yaml                  # K, T_world_cam, T_world_cam_isaac, 해상도
│   └── coordinate_conventions.md     # 좌표계/단위 (export 시 자동 생성)
├── objects/
│   └── <obj>/
│       ├── object.yaml               # 추정/실측 치수, mass, scale 계수
│       ├── <obj>_cad_scaled.glb      # ★ Isaac 에 올릴 mesh (실척, meter, 원점 중심)
│       └── <obj>_cad_raw.glb         # 원본 CAD (단위 임의 — 쓰지 마라)
├── trials/
│   └── <trial>/
│       ├── trial.yaml                # 이 trial 의 카메라/객체 index
│       ├── cam{0,1,2}/rgb.png        # 실제 RGB
│       ├── cam{0,1,2}/depth.png      # 실제 depth (uint16 mm)
│       ├── cam{0,1,2}/mask_<obj>.png # 실제 SAM2 mask (객체별)
│       └── poses/<obj>.json          # ★ T_world_object, T_cam_object, 재투영 IoU
├── measurements/
│   ├── measured_sizes.yaml           # 캘리퍼 실측 치수 + mass (mass 는 null)
│   └── size_error.csv                # 추정 - 실측 오차
├── schemas/                          # pose / object / camera JSON Schema
└── scripts/
    ├── validate_handoff.py           # package 검증
    └── inspect_trial.py              # trial 1개 시각 확인 (Isaac 불필요)
```

### 파일별 상세

| 파일 | 생성 주체 | 의미 | 좌표계 | 단위 | Isaac 에서 쓰는 곳 |
|---|---|---|---|---|---|
| `objects/<obj>/<obj>_cad_scaled.glb` | `Obj_Step3c` | 추정 실척 CAD, AABB 중심이 원점 | object local | m | USD 변환 → visual/collision mesh |
| `objects/<obj>/object.yaml` | exporter | 추정·실측 치수, mass, scale 계수 | — | m / kg | scale 조건 구성 (§9) |
| `calibration/cameras.yaml` | exporter | intrinsic + extrinsic | world = `cam0` | m | 카메라 prim 생성 (§8) |
| `trials/<t>/poses/<obj>.json` | exporter | 객체 6-DoF pose | world = `cam0` | m | object prim 배치 (§5) |
| `trials/<t>/cam{i}/rgb.png` | `Obj_Step1` | 실제 RGB | — | — | overlay 비교 대상 |
| `trials/<t>/cam{i}/depth.png` | `Obj_Step1` | 실제 depth | camera | **mm** (uint16) | (선택) depth 비교 |
| `trials/<t>/cam{i}/mask_<obj>.png` | `Obj_Step2` (SAM2) | 실제 객체 mask | — | — | **IoU 의 ground truth** (§10) |
| `measurements/size_error.csv` | exporter | 추정 − 실측 오차 | — | mm | sweep 범위 근거 (§11) |

---

## 5. 좌표계와 단위

> 전체 규약은 `calibration/coordinate_conventions.md` 에 export 시점 값으로 자동 생성된다.
> 아래는 그중 반드시 알아야 할 것.

### world 는 로봇 base 가 아니라 `cam0` 다

`manifest.yaml` 의 `conventions.world_frame: cam0`. `cam0` 의 `T_world_cam` 은 **단위행렬**이다.
즉 **world 좌표계 = cam0 의 OpenCV 카메라 좌표계**다. 따라서:

* **+Y 가 아래를 향한다** (OpenCV 규약). USD 기본 up-axis(+Z)와 다르다.
* 중력·테이블 평면을 물리적으로 맞추려면 world → Isaac world 회전을 담당자가 한 번 정의해야
  한다. **이 행렬은 아직 정해지지 않았다 (§17 TODO).** 렌더링/alignment 평가(§10)만 할 거면
  pose 를 그대로 써도 되고, grasp ablation(§12) 을 하려면 반드시 정의해야 한다.

### 변환식

모든 4×4 는 **row-major**, `p_dst = T @ p_src` (열벡터 오른쪽 곱).

```
  W        W     C
   T    =   T  ·  T
    O        C     O

T_world_object = T_world_cam @ T_cam_object
T_cam_object   = inv(T_world_cam) @ T_world_object
```

* `T_world_cam` — `calibration/cameras.yaml`. **camera → world 방향이다.** 역으로 쓰면 객체가 엉뚱한 곳에 뜬다.
* `T_world_object` — `trials/<t>/poses/<obj>.json`. **이미 다중뷰 융합된 값**이라 카메라별로 따로
  융합할 필요가 없다. `T_cam_object` 는 위 식으로 미리 계산해 같은 파일에 넣어 두었다.

### 이 pose 는 어떤 mesh 에 대한 것인가

**`objects/<obj>/<obj>_cad_scaled.glb` 에 대한 것이다.** 원본 CAD (`_cad_raw.glb`) 가 아니다.

내부적으로 `Obj_Step3c` 의 `T_world_cad` 는 원본 CAD 기준이고, 내보내는 mesh 는 스케일 후
AABB 중심을 원점으로 옮긴 것이라 translation 이 다르다. exporter 가 이 보정을 이미 적용했다:

```
p_world = R · (s · p_cad) + t          # Obj_Step3c 가 푼 것 (원본 CAD 기준)
v       = s · p_cad − c                # 내보낸 mesh (c = centroid of s·CAD)
p_world = R · v + (R·c + t)            # ⇒ T_world_object = [ R | R·c + t ]
```

보정을 빼먹으면 hole 이 **0.78 mm** 밀린다. `poses/<obj>.json` 의 `recenter_offset_mm` 에 그 값이 있다.

### 단위

* 길이 **meter** — pose translation, mesh vertex, `object.yaml` 의 size 전부.
* depth PNG 만 **uint16 mm** — meter 로 쓰려면 `× 0.001`.
* mass **kg** (현재 `null`).

### Quaternion

**`wxyz` (scalar-first).** 키 이름이 `quat_wxyz` 라 헷갈릴 일이 없다.
USD `Gf.Quatd(w, Gf.Vec3d(x, y, z))` 와 같은 순서. **scipy 는 `xyzw` 라 그대로 넣으면 안 된다:**

```python
from scipy.spatial.transform import Rotation
w, x, y, z = pose["quat_wxyz"]
R = Rotation.from_quat([x, y, z, w])     # scipy 는 xyzw
```

### OpenCV 카메라 → Isaac/USD 카메라

OpenCV 는 +Z 가 광축 전방, +Y 가 아래. USD 카메라는 **−Z 가 전방, +Y 가 위**.
extrinsic 을 그대로 USD 카메라 prim 에 넣으면 **상이 상하로 뒤집힌다.**

```
T_world_cam_isaac = T_world_cam @ CV_TO_USD

              ⎡ 1   0   0   0 ⎤
CV_TO_USD  =  ⎢ 0  −1   0   0 ⎥
              ⎢ 0   0  −1   0 ⎥
              ⎣ 0   0   0   1 ⎦
```

**변환된 값(`T_world_cam_isaac`, `quat_wxyz_isaac`)이 `cameras.yaml` 에 이미 들어 있다.**
카메라 prim 에는 그것을 쓰고, pose 계산·IoU 계산에는 `T_world_cam` 을 써라.

---

## 6. 빠른 시작

### 1) handoff package 검증 — **구현됨**

```bash
cd outputs/isaac_handoff
python scripts/validate_handoff.py --package .
```

`ready_for_isaac_sim: YES` 가 나와야 다음으로 간다. `NO` 면 `handoff_validation_report.txt` 의
ERROR 를 사용자에게 그대로 전달하라 (mask 해상도, pose 형식, 재투영 IoU 등 항목별로 찍힌다).

### 2) trial 1개 시각 확인 — **구현됨** (Isaac Sim 불필요)

```bash
python scripts/inspect_trial.py --package . --trial trial_0001
```

`trials/trial_0001/inspect_overlay.png` 가 생긴다.
**초록 = 실제 SAM mask, 빨강 = handoff mesh 를 handoff pose 로 재투영한 실루엣.**
둘이 겹치면 package 는 정상이다. 이후 Isaac 에서 객체가 이상하게 뜨면 **package 가 아니라 import
쪽 문제**라는 뜻이다. 이 그림이 §15 문제 해결의 출발점이다.

### 3~10) Isaac Sim 단계 — **미구현. 담당자가 만들어야 한다.**

아래는 만들어야 할 스크립트와 각 단계가 소비/생성하는 파일이다.
**명령어를 임의로 지어내지 않았다.** 이름은 담당자가 정하면 된다.

| # | 단계 | 입력 | 출력 | 상태 |
|---|---|---|---|---|
| 3 | mesh → USD 변환 | `objects/<obj>/<obj>_cad_scaled.glb` | `<obj>.usd` | `TODO: 팀원이 구현` |
| 4 | scene smoke test (trial 1개) | `trials/trial_0001/`, `calibration/cameras.yaml` | stage + 카메라 3대 | `TODO: 팀원이 구현` |
| 5 | 카메라 RGB / segmentation 렌더링 | 위 stage | sim RGB, sim mask (640×480) | `TODO: 팀원이 구현` |
| 6 | real / sim overlay 확인 | sim mask + `trials/*/cam*/mask_<obj>.png` | overlay png | `TODO: 팀원이 구현` |
| 7 | 전체 alignment evaluation (§10) | 위 | `alignment_trials.csv`, `alignment_summary.csv` | `TODO: 팀원이 구현` |
| 8 | scale sweep (§11) | `manifest.yaml: isaac.scale_sweep` | `scale_sweep_trials.csv` | `TODO: 팀원이 구현` |
| 9 | grasp ablation (§12) | `manifest.yaml: isaac.grasp_ablation` | `grasp_ablation_trials.csv` | `TODO: 팀원이 구현` |
| 10 | figure / LaTeX table 생성 | 위 CSV 전부 | `figures/`, `tables/` | `TODO: 팀원이 구현` |

---

## 7. 객체 asset import

* **mesh 단위는 meter다.** `<obj>_cad_scaled.glb` 는 이미 실척이다.
  USD 변환 시 `metersPerUnit = 1.0` 로 두고 **추가 스케일을 곱하지 마라.** 기본값이 cm 인 툴이면
  100배 커진다 (§15 첫 항목).
* **pivot/origin**: AABB 중심이 원점이다. `poses/<obj>.json` 의 pose 는 이 원점 기준이다.
  USD 변환기가 pivot 을 바꾸면 그만큼 어긋난다. 변환 후 bbox 중심이 (0,0,0) 인지 확인하라.
* **texture**: glb 에 embed 되어 있다. 외부 파일 참조가 없어서 package 를 옮겨도 안 깨진다
  (validator 가 검사한다).
* **visual mesh**: 변환된 mesh 를 그대로 쓴다.
* **collision mesh**: convex hull 로 충분하다 (peg 는 볼록, hole 은 구멍이 있어 **convex
  decomposition 이 필요하다** — convex hull 로 감싸면 구멍이 막혀 peg 가 안 들어간다).
* **mass**: `objects/<obj>/object.yaml` 의 `mass_kg`. **현재 `null` 이다 (§17).**
  값이 들어오기 전까지는 균질 밀도 + 재질 밀도 가정으로 임시 진행하고, 그 사실을 결과에 남겨라.
* **center of mass**: `com_m` 도 `null` — 균질 밀도 가정 (mesh centroid).
* **방향 확인**: `scripts/inspect_trial.py` 의 overlay 가 정답이다. Isaac 에서 카메라 3대로 렌더한
  그림이 그 overlay 와 같은 방향으로 보이면 맞다.

---

## 8. 카메라 설정

`calibration/cameras.yaml` 의 각 카메라에 `width`, `height`, `K`, `fx/fy/cx/cy`,
`T_world_cam`, `T_world_cam_isaac` 이 들어 있다.

* **resolution**: 640 × 480 으로 고정하라. 실제 mask 와 크기가 같아야 IoU 를 계산할 수 있다.
* **intrinsic → Isaac**: `K` 는 픽셀 단위 pinhole 이다. USD 카메라는 `focalLength` 와
  `horizontalAperture` (mm) 로 표현하므로 아래 관계를 쓴다.

  ```
  horizontal_aperture = sensor_width          (임의로 하나 고르면 된다. 예: 36.0)
  focal_length        = fx * horizontal_aperture / width
  vertical_aperture   = horizontal_aperture * height / width * (fx / fy)
  ```

  `fx ≠ fy` 라 `vertical_aperture` 를 위처럼 보정하지 않으면 세로 방향이 미세하게 틀어진다
  (cam0: fx=607.25, fy=606.30 — 0.16 % 차이).
* **principal point**: `cx, cy` 가 정확히 중심이 아니다 (cam0: 318.95, 241.15 vs 중심 320, 240).
  USD 표준 카메라는 principal point offset 을 직접 지원하지 않는다. 오차가 1 px 수준이라
  **무시해도 IoU 에 거의 영향이 없지만**, 엄밀히 하려면 렌더 후 shift 하거나 offset 을 지원하는
  카메라 모델을 써라. 무시하기로 했다면 그 사실을 결과에 적어라.
* **clipping range**: 객체가 카메라에서 0.46 ~ 0.64 m 거리에 있다. `near = 0.01`, `far = 5.0`
  정도면 충분하다. near 를 너무 크게 잡으면 객체가 잘린다.
* **extrinsic**: 카메라 prim 에는 **`T_world_cam_isaac`** 를 써라 (§5). `T_world_cam` 을 그대로
  쓰면 상하 반전된다.
* **distortion**: RealSense color 스트림은 rectified 출력이라 왜곡계수를 쓰지 않았다
  (`distortion: null`, `distortion_model: none`). Isaac 카메라도 왜곡 없이 두면 된다.
* **segmentation**: semantic/instance segmentation annotator 를 켜고 객체 prim 에 semantic label
  (`manifest.yaml: objects` 의 id — 예: `peg`, `hole`) 을 붙여라.
  출력 mask 는 640×480 uint8 로 저장해 실제 mask 와 같은 형식을 맞춘다.

---

## 9. 실험 조건

| Condition | Size | Pose |
|---|---|---|
| **Oracle** | measured GT size | predefined Isaac pose |
| **Scale-only** | estimated size | predefined Isaac pose |
| **Pose-only** | measured GT size | pipeline pose |
| **Full-pipeline** | estimated size | pipeline pose |

정의는 `manifest.yaml` 의 `isaac.conditions` 에도 기계가 읽는 형태로 들어 있다.

### 입력 파일

* **estimated size** → `objects/<obj>/<obj>_cad_scaled.glb` 를 **그대로** 쓴다.
* **measured GT size** → 같은 mesh 에 `object.yaml` 의 `size.scale_estimated_to_measured`
  (축별 3-vector, 내림차순 축 기준) 를 곱한다. 실측 치수는 `size.measured_extents_m`.
* **pipeline pose** → `trials/<t>/poses/<obj>.json` 의 `T_world_object`.
* **predefined Isaac pose** → **담당자가 정한다.** 실제 관측과 무관하게 stage 위에 놓는
  결정론적 pose 를 쓰면 된다 (seed 고정).

### ⚠️ Oracle 조건을 오해하지 마라

Oracle 의 pose 는 **실측한 ground-truth pose 가 아니다.** 사용자 파이프라인은 pose 의 GT 를
측정하지 않았다 (크기의 GT 만 캘리퍼로 쟀다). Oracle 은 **scale 의 영향을 분리하기 위해 pose 를
상수로 고정하는 조건**일 뿐이며, **Real-to-Sim 전체 정확도의 상한이 아니다.**
"Oracle 대비 Full-pipeline 이 x % 하락" 은 *pose 추정 오차 + scale 추정 오차* 를 합친 값이고,
"Oracle 대비 Scale-only" 만이 **순수한 크기 오차의 효과**다. 논문에 이 구분을 명시하라.

---

## 10. Real-to-Sim alignment

* **실제 mask**: `trials/<t>/cam{i}/mask_<obj>.png` (SAM2, 640×480, `>0` 이면 전경)
* **sim mask**: Isaac segmentation 렌더 결과 (§8). 같은 카메라, 같은 해상도여야 한다.

객체·카메라·trial 마다 다음을 계산해 `alignment_trials.csv` 로 남겨라.

| metric | 정의 |
|---|---|
| silhouette IoU | `|A∩B| / |A∪B|` |
| Dice | `2|A∩B| / (|A|+|B|)` |
| centroid error | 두 mask 무게중심의 픽셀 거리 |
| bounding-box error | bbox 의 `[x, y, w, h]` 차이 (px) |
| contour Chamfer distance | 두 외곽선 점집합 간 양방향 평균 최근접 거리 (px) |

overlay 는 실제 RGB 위에 실제 mask 외곽선과 sim mask 외곽선을 서로 다른 색으로 그린다.
`scripts/inspect_trial.py` 가 만드는 그림이 정확히 이 형식이니 그대로 흉내내면 된다
(거기서는 sim mask 자리에 "mesh 재투영 실루엣" 이 들어간다).

**기준선**: package 의 mesh+pose 를 재투영한 IoU 는 **0.96 ~ 0.98** 이다
(`poses/<obj>.json` 의 `reprojection_iou`). Isaac 렌더 mask 의 IoU 가 이보다 크게 낮으면
alignment 문제가 아니라 **import 문제**다 (§15).

---

## 11. Scale sensitivity sweep

기본 scale error (`manifest.yaml: isaac.scale_sweep.uniform_percent`):

```text
-10%, -5%, -3%, 0%, +3%, +5%, +10%
```

* **uniform scale** 이 기본이다 (`anisotropic: false`). 세 축에 같은 배율을 곱한다.
* **anisotropic** 을 켜면 축별로 독립 배율을 준다. 실제 파이프라인 오차가 축별로 다르므로
  (peg: −0.69 / −0.99 / −1.89 mm) 필요하면 켜라.
* **measured error distribution sampling**: 실제 관측된 오차 범위는
  **−1.89 mm ~ +0.86 mm (약 −4.2 % ~ +1.7 %)** 다 (`measurements/size_error.csv`).
  sweep 범위 ±10 % 는 이 실측 오차를 충분히 포함하도록 잡은 것이다.
* **mass 고정**: `hold_mass_constant: true` — scale 만 바꾸고 mass 는 실측값으로 둔다.
  (부피에 비례해 mass 를 바꾸면 크기 효과와 관성 효과가 섞여 해석이 불가능해진다.)
* **seed / 초기 pose 고정**: `seed: 0`, 초기 pose 는 조건별로 동일하게. sweep 간 유일한
  차이가 scale 이어야 한다.

출력: `scale_sweep_trials.csv`.

---

## 12. Grasp ablation

`manifest.yaml: isaac.grasp_ablation` 에 파라미터가 있다.

| 항목 | 값 | 상태 |
|---|---|---|
| lift height | 0.10 m | 설정됨 |
| hold duration | 2.0 s | 설정됨 |
| episode timeout | 20.0 s | 설정됨 |
| seed | 0 | 설정됨 |
| robot USD | — | `TODO: 팀원이 입력 필요` |
| gripper USD | — | `TODO: 팀원이 입력 필요` |
| policy checkpoint | — | `TODO: 팀원이 입력 필요` |
| friction / contact | — | `TODO: 팀원이 입력 필요` |

* **object initial pose**: 조건에 따라 pipeline pose (`poses/<obj>.json`) 또는 predefined pose (§9).
* **grasp 성공 조건**: 객체를 `lift_height_m` 이상 들어올린 뒤 `hold_duration_s` 동안 유지.
* **drop 판정**: hold 중 객체가 gripper 접촉을 잃고 낙하.
* **collision 판정**: gripper/로봇이 객체 또는 테이블과 비정상 관통·충돌.
* **timeout**: `episode_timeout_s` 초과 시 실패.
* **실패 원인 코드**: 최소 `no_contact` / `slip_during_lift` / `drop_during_hold` /
  `collision` / `timeout` 로 구분해 CSV 에 남겨라. 원인 구분이 없으면 "작게 추정된 객체가 왜
  실패하는가" 를 설명할 수 없다.
* **결과 로그**: `grasp_ablation_trials.csv` (에피소드별), `grasp_ablation_summary.csv` (조건별 집계).

---

## 13. 결과 파일

담당자가 생성할 파일 (`TODO` — 아직 없음):

```text
alignment_trials.csv           # trial × object × camera 별 IoU/Dice/centroid/bbox/chamfer
alignment_summary.csv          # 조건별 집계
scale_sweep_trials.csv         # scale error × 결과
grasp_ablation_trials.csv      # 에피소드별 성공/실패 + 실패 원인
grasp_ablation_summary.csv     # 조건별 성공률
sim_real_comparison.csv        # 조건 4개 종합
figures/                       # 논문용 그림
tables/                        # LaTeX table
```

package 에 **이미 있는** 결과 파일:

```text
measurements/size_error.csv          # 추정 vs 실측 치수 오차 (mm, %)
handoff_validation_report.json/.txt  # package 검증 결과
```

---

## 14. 실험 재현성

각 실행마다 다음을 결과와 함께 저장하라.

* **random seed** — `manifest.yaml` 의 `isaac.scale_sweep.seed`, `isaac.grasp_ablation.seed`
* **config snapshot** — `manifest.yaml` 전체를 결과 폴더에 복사
* **Git commit hash** — package 는 `manifest.yaml: git_commit` 에 사용자 repo 커밋을 기록한다.
  Isaac 쪽 repo 커밋은 담당자가 별도로 기록하라.
* **Isaac Sim version** — `TODO: 팀원이 입력 필요`
* **policy checkpoint** — `TODO: 팀원이 입력 필요`
* **physics settings** — solver, timestep, friction, contact offset
* **실행 시간**, **GPU / 시스템 정보**

---

## 15. 문제 해결

**먼저 이걸 돌려라.** package 문제인지 Isaac 문제인지 1분 만에 갈린다.

```bash
python scripts/validate_handoff.py --package .
python scripts/inspect_trial.py --package . --trial trial_0001
```

`inspect_trial` 의 overlay 에서 초록(실제 mask)과 빨강(재투영)이 겹치면 **package 는 정상이다.**
아래 문제는 전부 Isaac import 쪽이다.

| 증상 | 원인 | 확인할 파일 / 진단 |
|---|---|---|
| 객체가 100배 크거나 작다 | USD 변환기가 cm/mm 를 가정 | `metersPerUnit` 확인. `object.yaml: size.estimated_extents_m` 와 stage bbox 를 비교 |
| 객체가 1000배 차이 | m ↔ mm 혼동 | pose translation 이 0.46~0.64 **m** 범위인지 확인 |
| 객체가 90°/180° 회전 | quaternion 순서 (`wxyz` vs `xyzw`) | `poses/<obj>.json: quat_wxyz` 와 `T_world_object` 는 일치한다 (validator 가 검사). scipy 를 썼다면 §5 |
| 객체 origin 이 안 맞음 | USD 변환기가 pivot 변경 | 변환된 USD 의 bbox 중심이 (0,0,0) 인지. glb 는 그렇다 |
| 카메라 상이 상하 반전 | `T_world_cam` 을 그대로 씀 | **`T_world_cam_isaac`** 를 써라 (§5) |
| 카메라 상이 좌우 반전 | 좌표계 handedness 뒤집힘 | `det(R) = +1` 이어야 한다 (validator 가 검사) |
| sim mask 가 비어 있다 | semantic label 미부착 / 카메라가 객체를 안 봄 | 먼저 sim RGB 에 객체가 보이는지 확인. 안 보이면 카메라 extrinsic 문제 |
| mask 해상도가 실제와 다름 | 렌더 해상도 불일치 | `cameras.yaml: width/height` = 640×480 |
| 객체가 stage 아래(바닥 밑)에 생성 | world 의 **+Y 가 아래**인데 USD up-axis 는 +Z | §5 "world 는 cam0" — up-axis 정합 행렬 필요 (§17) |
| collision mesh 가 과도하게 큼 | convex hull 이 hole 의 구멍을 메움 | hole 은 **convex decomposition** 필요 (§7) |
| gripper 가 객체를 관통 | collision mesh 미생성 / contact offset 과소 | collision prim 존재 여부, physics contact offset |
| texture 가 안 보임 | glb material 변환 실패 | `objects/<obj>/<obj>_cad_scaled.glb` 를 trimesh/Blender 로 열어 확인 (validator 는 외부 참조만 검사) |
| pose 가 전치/역변환된 듯 보임 | `T_world_cam` 을 `T_cam_world` 로 씀 | `cam0` 의 `T_world_cam` 은 **단위행렬**이다. 아니라면 잘못 읽은 것 |

---

## 16. 완료 체크리스트

```text
[ ] handoff validator 통과 (ready_for_isaac_sim: YES)
[ ] manifest.yaml 의 objects 전부가 올바른 실제 크기로 import 됨
    - peg : 44.31 × 29.01 × 28.11 mm (추정) / 45 × 30 × 30 mm (실측)
    - hole: 50.86 × 50.76 × 49.01 mm (추정) / 50 × 50 × 50 mm (실측)
    - YCB 5개: 사용자 수집 후 추가됨 (pending_objects 가 비면 완료)
[ ] object frame 방향 확인 (inspect_trial overlay 와 동일)
[ ] 카메라 3대 위치와 방향 확인
[ ] 실제 RGB 와 Sim view 가 같은 시점을 가짐
[ ] segmentation mask 생성 확인 (640×480)
[ ] trial 1개 smoke test 완료
[ ] scale sweep 완료
[ ] grasp ablation 완료
[ ] 모든 결과 CSV 생성
[ ] 논문용 figure 와 table 생성
[ ] config 와 로그 보관
```

> 최종 목표는 **7개** (peg, hole + YCB 5개) 이고, **현재 package 에 든 것은 `manifest.yaml` 이
> 정답이다.** trial 은 현재 **1개** (`trial_0001`). 객체·trial 모두 사용자가 재-export 하면 늘어난다.

---

## 17. 팀원에게 필요한 미확정 정보

저장소에서 확인되지 않는 값이다. 임의로 채우지 말고 담당자가 결정한 뒤 여기에 기록하라.

| 항목 | 현재 상태 | 담당자 | 필요한 조치 |
|---|---|---|---|
| YCB 객체 5개 | 미수집 (`pending_objects`) | 사용자 | mesh·촬영·마스크·크기추정 후 재-export |
| 객체 질량 `mass_kg` | `null` (미측정) | 사용자 | 저울로 측정 → `configs/evaluation.yaml` 에 기입 후 재-export |
| center of mass `com_m` | `null` | 사용자 | 미측정이면 균질 밀도 가정으로 진행 (그 사실을 결과에 명시) |
| 재질 `material` | `null` | 사용자 | 예: PLA. 밀도 추정에 필요 |
| FoundationPose pose | 미실행 (`pose_source: cad_silhouette_fit`) | 사용자 | 필요 시 `Obj_Step5` 실행 → `fp_dir` 지정해 재-export |
| world → Isaac up-axis 정합 행렬 | **미정의** | Isaac | world = cam0 (+Y 아래). 물리 실험 전 반드시 정의 (§5) |
| Isaac Sim 버전 | 미확인 | Isaac | 기록 |
| Python 실행 방식 | 미확인 | Isaac | `./python.sh` 인지 런처인지 |
| GPU / OS 요구사항 | 미확인 | Isaac | 기록 |
| robot USD | 미확인 | Isaac | 결정 후 `configs/evaluation.yaml: isaac.grasp_ablation.robot_usd` |
| gripper USD | 미확인 | Isaac | 〃 `gripper_usd` |
| policy checkpoint | 미확인 | Isaac | 〃 `policy_checkpoint` |
| friction / contact 파라미터 | 미확인 | Isaac | 〃 `friction` |
| grasp 성공 판정 세부 | 초안만 (§12) | Isaac | 실제 판정 코드와 일치시킬 것 |
| predefined Isaac pose (Oracle/Scale-only) | 미정의 | Isaac | 결정론적 초기 pose 정의 (seed 고정) |
