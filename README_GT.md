# 정답(Ground-truth) vs 추정 크기 비교 실험

peg / hole 의 **직육면체 크기(x·y·z extent)** 를 두 방법으로 구해 비교한다.

| 지표 | 방법 | 스케일 근거 | 코드 |
|---|---|---|---|
| **추정 (방법 B)** | 캘리브된 카메라 → 마스크 depth 융합 → point-cloud OBB | 카메라 간 캘리브 | 기존 `Obj_Step1~3` → `*_bbox_metric.json` 의 `bbox_extents_m` |
| **정답 (방법 A)** | 물체 면에 ArUco 마커 부착 → 마커 3D 포즈 → 면-대-면 거리 | **인쇄된 마커 크기(+캘리브)** | **신규** `Gt_Step1`, `Gt_Step2` |

정답이 추정보다 **정확해야** 의미가 있으므로, 기본 측정은 **멀티뷰 삼각측량**
(`method=triangulate`, 합성검증 오차 **<0.02 mm**)을 쓴다. 단일마커 PnP(`method=pnp`)는
정면에서 depth 가 약해 부정확 → 교차검증/단일뷰 fallback 용.

---

## 1. 마커 부착 (물리 준비)

- 물체를 직육면체로 보고, 중심에 로컬 축 x/y/z(모서리 방향)를 정한다.
- 측정할 **각 면에 ArUco 마커를 평평하게, 가급적 면 중앙**에 붙인다.
  - 한 축을 재려면 그 축의 **+면과 −면 둘 다** 마커가 있어야 한다.
  - `DICT_4X4_50`, 캘리브 큐브(ID 0~4)와 안 겹치게 **ID 10~15** 권장.
  - 마커 크기(`marker_size_m`)는 인쇄 실측값을 `gt_marker_layout_example.json` 에 기입.
- 부착 규약을 `gt_marker_layout_example.json` 처럼 (ID → axis/side) 로 작성.

> **가시성**: 삼각측량은 한 마커가 **≥2대** 카메라에 보여야 함. 3대가 같은 쪽이라
> 물체의 반대편 면이 안 보이면, **물체를 돌려 여러 번 캡처**(폴더 각각)해서 Gt_Step2 에
> 함께 넘기면 축별로 병합된다. (extent 는 자세 불변이라 자세가 달라도 OK)

---

## 2. 캡처 — `Gt_Step1`

```bash
python Gt_Step1_capture_marked_object.py \
  --out_dir         data/gt_capture_peg_set1 \
  --intrinsics_dir  intrinsics \
  --transforms_json data/handeye_session_01/T_R_Ci_all.json \
  --layout          gt_marker_layout_example.json \
  --show
```
프리뷰에 검출 마커가 그려지고, 하단에 마커별 검출 카메라 수 / 축 커버(OK/x)가 표시됨.
`SPACE`=저장. 저장물은 `Obj_Step1` 과 동일 flat layout
(`cam{i}_rgb.png`, `cam{i}_K.txt`, `cam{i}_T_cam_to_world.txt`, `calib_info.json`) +
`gt_marker_coverage.json`, `detect_cam{i}.png`.

## 3. 측정 + 비교 — `Gt_Step2`

```bash
# 단일 폴더 + 추정치 비교
python Gt_Step2_measure_marker_box.py \
  --capture_dir data/gt_capture_peg_set1 \
  --layout      gt_marker_layout_example.json \
  --intrinsics_dir intrinsics \
  --estimate_json  data/outputs_set1/obj1/obj1_bbox_metric.json \
  --label peg --out_dir data/gt_results/peg

# 물체를 돌려 여러 번 찍었으면 폴더 여러 개(축별 병합)
python Gt_Step2_measure_marker_box.py \
  --capture_dir data/gt_capture_peg_a data/gt_capture_peg_b \
  --layout gt_marker_layout_example.json --label peg
```
`--method auto|triangulate|pnp` (기본 auto: ≥2뷰 삼각측량, 아니면 PnP).
출력: 콘솔 표 + `gt_size_report.json` (축별 extent, 마커 진단, GT vs EST 오차 mm/%).

### 비교 방식
GT extents (x,y,z) 와 추정 `bbox_extents_m` 를 각각 **내림차순 정렬** 후 rank 대응으로
비교(OBB 축 순서는 임의라 정렬 비교가 공정). `signed = 추정 − 정답`.

---

## 4. 정확도 진단 (리포트에 포함)
- **marker 변길이 self-check** (삼각측량): 복원한 마커 한 변 vs 인쇄 크기. 0에 가까울수록 스케일 신뢰.
- **cross-cam spread**: 같은 마커를 여러 카메라가 볼 때 PnP center 산포 → 캘리브 자기일관성.
- **in-plane offset / face tilt**: 마커가 면 중앙에서 벗어난 정도 / 면 수직 정렬 오차.

## 5. 자기검증 (하드웨어 불필요)
```bash
python test_marker_box.py     # 코어: 삼각측량 sub-mm, 기하 solver 정확
python test_gt_pipeline.py    # Gt_Step2 통합: 합성 캡처 → 측정/비교/병합
```

## 파일
- `_marker_box.py` — 코어(검출·PnP·삼각측량·축 solver·비교)
- `Gt_Step1_capture_marked_object.py` — 마커 부착 물체 캡처(+커버리지 프리뷰)
- `Gt_Step2_measure_marker_box.py` — 정답 측정 + 추정 비교 리포트
- `gt_marker_layout_example.json` — 마커 부착 규약 예시
- `test_marker_box.py`, `test_gt_pipeline.py` — 합성 자기검증
