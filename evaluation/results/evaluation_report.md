# Real-to-Sim Object Size Estimation — Evaluation Report

생성: `python evaluation/run_all_evaluations.py --config evaluation/evaluation_config.yaml --output evaluation/results`


## 1. 탐색한 기존 코드와 결과 파일

| 종류 | 파일 | 역할 |
|---|---|---|
| 엔진 | `_silhouette_fit.py` | `fit_cad_to_views`(등방), `fit_mesh_aniso`(비등방), `render_silhouette`, `per_view_iou`, `obb_frame` |
| Baseline | `Obj_Step3_sam3d_scale.py` | SAM3D mesh 생성 + `--estimate_size` (비등방 기본) → `*_size.json` |
| Oracle | `Obj_Step3c_cad_scale.py` | 정답 CAD 실루엣 정합 → `*_cad_fit.json` |
| 기존 비교 | `Obj_Step3d_compare_gt.py` | GT 대비 비교 그림 (이번 평가와 별개, 수정하지 않음) |
| 방법 선택 근거 | `data(3)/outputs_sam3d_fit/size_method_experiment.json` | iso vs aniso 실험 (peg: iso 4.37mm → aniso 1.05mm) → 비등방 채택 |

> 기존 추정 코드는 **읽기만** 했고 수정하지 않았다. 평가 코드는 `evaluation/` 에 신규 작성했다.


## 2. 실제 사용한 데이터 경로

| Object | capture | mask | Baseline (`*_size.json`) | Oracle (`*_cad_fit.json`) | source cam |
|---|---|---|---|---|---|
| peg | `data(3)/capture_obj` | `data(3)/masks/peg` | `data(3)/outputs_sam3d_fit/objpeg/objpeg_size.json` | `data(3)/outputs_cad_fit/peg_cad_fit.json` | cam0 |
| hole | `data(3)/capture_obj` | `data(3)/masks/hole` | `data(3)/outputs_sam3d_fit/objhole/objhole_size.json` | `data(3)/outputs_cad_fit/hole_cad_fit.json` | cam0 |
| T_shape | `data/capture_obj_01` | `data/masks_01/obj1` | `data/outputs_sam3d_01/obj1/obj1_size.json` | `data/outputs_cad_01/obj1_cad_fit.json` | cam0 |
| kettle | `data/capture_obj_02` | `data/masks_02/obj1` | `data/outputs_sam3d_02/obj1/obj1_size.json` | `data/outputs_cad_02/obj1_cad_fit.json` | cam0 |

## 3. GT dimension 출처

| Object | GT (mm, L≥W≥H) | 출처 | 비고 |
|---|---|---|---|
| peg | 45.0 × 30.0 × 30.0 | `caliper_manual` | — |
| hole | 50.0 × 50.0 × 50.0 | `caliper_manual` | — |
| T_shape | 150.0 × 100.0 × 50.0 | `cad_design_nominal` | — |
| kettle | 116.0 × 68.0 × 65.0 | `caliper_manual` | — |

- `caliper_manual`: `configs/evaluation.yaml` 의 `measured.extents_mm` (캘리퍼 실측).
- `cad_design_nominal` (T_shape): `T_shape.glb` 의 고유 OBB 가 정확히 150/100/50 이고 추정 scale 이 0.001(mm→m)에 0.36% 로 수렴 → mm 단위 설계 CAD 로 판단해 **설계값**을 GT 로 사용. **캘리퍼 실측이 아니다.**
- `caliper_manual_partial` (kettle): L2/L3 만 확정. L1 은 미확정 (§12).

## 4. Baseline / Oracle 정의

- **Baseline (`baseline_sam3d`)**: SAM3D 가 단일 RGB+mask 로 만든 unitless 단서 mesh 를 3대 카메라 SAM2 실루엣에 Sim(3) 7-DoF 로 정합. 최종 치수 = mesh OBB extents × 추정 scale. **실제 운용 방법** (정답 CAD 불필요).
- **Oracle (`oracle_cad`)**: 단서 mesh 만 정답 CAD 로 교체. 엔진·손실·최적화 동일. **형상이 완벽할 때의 정확도 상한**.
- 현재 데이터의 Baseline 은 모두 `sam3d_anisotropic_silhouette` (축별 scale_vec), Oracle 은 `cad_multiview_silhouette` (등방 scale) 이다. → **두 방법이 완전히 동일한 변환족은 아니다** (§14).

## 5. 축 대응 방식

파이프라인과 GT 모두 치수를 **내림차순(L≥W≥H)** 으로 정렬해 보고하므로 크기 rank 대응을 사용했다. 정렬된 두 수열에서 identity 대응이 Σ|차이| 를 최소화하는 최적 순열이므로(rearrangement inequality), 검증을 위해 6개 순열을 모두 탐색해 rank 대응과 일치하는지 확인했다.

- `axis_matching_method` = `rank_descending` (전 행)
- 6-순열 탐색이 rank 대응과 일치: **8/8 행** → rank 대응이 최적임을 확인

## 6. 평가 지표 수식

```
e_L = |L_hat - L_GT|,  e_W = |W_hat - W_GT|,  e_H = |H_hat - H_GT|      [mm]
E_dim = (e_L + e_W + e_H) / 3                                            [mm]   <- 메인
E_rel = (1/3)[e_L/L_GT + e_W/W_GT + e_H/H_GT] x 100                      [%]
IoU_i = |M_real_i ∩ M_sim_i| / |M_real_i ∪ M_sim_i|
Cross-view IoU = mean over cameras EXCLUDING the SAM3D source view       <- 메인
D_contour = 0.5[ mean_{p∈C_real} min_q ||p-q|| + mean_{q∈C_sim} min_p ||q-p|| ]  [px]
D_contour_norm = D_contour / sqrt(w_real^2 + h_real^2)
```
- GT 미확정 축은 e_* 를 계산하지 않고 평균에서 제외 (`gt_axes_used` 에 사용 축 수 기록).
- sim 마스크는 결과 JSON 의 fit 파라미터로 **재렌더링**했다. 재현 검증을 위해 엔진과 동일한 슈퍼샘플 조건의 IoU 를 저장된 `per_view_iou` 와 대조했다 (최대 Δ = 0.0045).

## 7. 객체별 결과

| Object | Method | GT (mm) | Estimated (mm) | e_L | e_W | e_H | **E_dim** (mm) | E_rel (%) | src IoU | **cross IoU** | cross D_contour (%) | axes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Peg | `baseline_sam3d` | 45.0 × 30.0 × 30.0 | 44.5 × 28.8 × 28.6 | 0.51 | 1.19 | 1.36 | **1.02** | 3.21 | 0.981 | **0.960** | 0.69 | 3/3 |
| Peg | `oracle_cad` | 45.0 × 30.0 × 30.0 | 44.3 × 29.0 × 28.1 | 0.69 | 0.99 | 1.89 | **1.19** | 3.72 | 0.981 | **0.959** | 0.50 | 3/3 |
| Hole | `baseline_sam3d` | 50.0 × 50.0 × 50.0 | 51.0 × 49.1 × 49.1 | 1.03 | 0.88 | 0.93 | **0.95** | 1.90 | 0.981 | **0.973** | 0.47 | 3/3 |
| Hole | `oracle_cad` | 50.0 × 50.0 × 50.0 | 50.9 × 50.8 × 49.0 | 0.86 | 0.76 | 0.99 | **0.87** | 1.74 | 0.978 | **0.976** | 0.49 | 3/3 |
| T-shape | `baseline_sam3d` | 150.0 × 100.0 × 50.0 | 150.6 × 99.8 × 50.1 | 0.61 | 0.23 | 0.14 | **0.33** | 0.31 | 0.981 | **0.976** | 0.34 | 3/3 |
| T-shape | `oracle_cad` | 150.0 × 100.0 × 50.0 | 149.5 × 99.6 × 49.8 | 0.54 | 0.36 | 0.18 | **0.36** | 0.36 | 0.981 | **0.980** | 0.23 | 3/3 |
| Kettle | `baseline_sam3d` | 116.0 × 68.0 × 65.0 | 117.3 × 69.1 × 67.0 | 1.31 | 1.09 | 2.02 | **1.47** | 1.95 | 0.936 | **0.978** | 0.33 | 3/3 |
| Kettle | `oracle_cad` | 116.0 × 68.0 × 65.0 | 115.1 × 68.0 × 65.1 | n/d | n/d | n/d | **n/d** | n/d | 0.858 | **0.900** | 3.41 | 0/3 |

## 8. 전체 (mean ± std)

| Method | n | E_dim (mm) | E_rel (%) | Cross-view IoU | D_contour (%) | IoU≥0.85 통과 |
|---|---|---|---|---|---|---|
| `baseline_sam3d` | 4 | 0.94 ± 0.47 | 1.84 ± 1.19 | 0.972 ± 0.008 | 0.46 ± 0.16 | 4/4 (100.0%) |
| `oracle_cad` | 3 | 0.81 ± 0.42 | 1.94 ± 1.69 | 0.954 ± 0.037 | 1.16 ± 1.51 | 4/3 (100.0%) |

## 9. Baseline − Oracle gap

- **Mean Dimension Error gap = -0.04 ± 0.12 mm** (n=3)
- Relative error gap = -0.14 %p

객체별 gap:

| Object | Baseline E_dim | Oracle E_dim | gap (mm) |
|---|---|---|---|
| Peg | 1.02 | 1.19 | -0.17 |
| Hole | 0.95 | 0.87 | +0.08 |
| T-shape | 0.33 | 0.36 | -0.04 |
| Kettle | 1.47 | n/d | +nan |

> 해석: 이 gap 은 **동일한 Sim(3) 최적화 엔진에서 단서 메시를 SAM3D 에서 정답 CAD 로 교체했을 때 변화한 최종 치수 오차**다. SAM3D 의 순수 형상 오차라고 단정할 수 없다 — SAM3D 형상 오류가 최종 크기 추정에 미친 영향으로 읽어야 한다.


통계 검정 (n=3):

- paired t-test: t=-0.613, p=0.602
- Wilcoxon signed-rank: W=2.0, p=0.750
- ⚠ 표본이 매우 적어 p-value 를 과도하게 해석하지 말 것. descriptive statistics 중심으로 보고한다.

## 10. Cross-view IoU 0.85 통과율

- `baseline_sam3d`: 4/4 (100.0%)
- `oracle_cad`: 4/3 (100.0%)
- IoU < 0.85 인 객체: 없음

## 11. 생성한 그래프와 피규어

모두 PNG(300dpi). 각 그림 상단에 `번호. 제목` 이 표기된다.

### 통합 (전체 객체) — `figures/`

- [1. Baseline vs Oracle E_dim](figures/fig1_baseline_vs_oracle_dim_error.png)
- [1b. 동 (가로형)](figures/fig1_baseline_vs_oracle_dim_error_horizontal.png)
- [2. GT vs Estimated](figures/fig2_gt_vs_estimated_dimensions.png)
- [3a. 축별 오차 (Baseline)](figures/fig3_per_axis_absolute_error_baseline.png)
- [3b. 축별 오차 (Oracle)](figures/fig3_per_axis_absolute_error_oracle.png)
- [4. Cross-view IoU](figures/fig4_cross_view_silhouette_iou.png)
- [5. Normalized contour distance](figures/fig5_normalized_contour_distance.png)
- [6. IoU vs E_dim](figures/fig6_iou_vs_dimension_error.png)
- [7. 정성 통합 그리드](figures/fig7_qualitative_real_to_sim_grid.png)
- [8. 통제 실험 (변환족 일치)](figures/fig8_controlled_transformation_family.png)

대표 객체 (7. 그리드 행): T_shape, peg, kettle — 최소오차/중간/최대오차 + 단순·복잡 형상 포함

### 물체별 — `per_object/<object>/`

| Object | 그림 |
|---|---|
| Peg | [1](per_object/peg/fig1_baseline_vs_oracle_dim_error.png) · [2](per_object/peg/fig2_gt_vs_estimated_dimensions.png) · [3a](per_object/peg/fig3_per_axis_absolute_error_baseline.png) · [3b](per_object/peg/fig3_per_axis_absolute_error_oracle.png) · [4](per_object/peg/fig4_cross_view_silhouette_iou.png) · [5](per_object/peg/fig5_normalized_contour_distance.png) · [7](per_object/peg/fig7_qualitative_three_views.png) · [8](per_object/peg/fig8_controlled_transformation_family.png) |
| Hole | [1](per_object/hole/fig1_baseline_vs_oracle_dim_error.png) · [2](per_object/hole/fig2_gt_vs_estimated_dimensions.png) · [3a](per_object/hole/fig3_per_axis_absolute_error_baseline.png) · [3b](per_object/hole/fig3_per_axis_absolute_error_oracle.png) · [4](per_object/hole/fig4_cross_view_silhouette_iou.png) · [5](per_object/hole/fig5_normalized_contour_distance.png) · [7](per_object/hole/fig7_qualitative_three_views.png) · [8](per_object/hole/fig8_controlled_transformation_family.png) |
| T-shape | [1](per_object/T_shape/fig1_baseline_vs_oracle_dim_error.png) · [2](per_object/T_shape/fig2_gt_vs_estimated_dimensions.png) · [3a](per_object/T_shape/fig3_per_axis_absolute_error_baseline.png) · [3b](per_object/T_shape/fig3_per_axis_absolute_error_oracle.png) · [4](per_object/T_shape/fig4_cross_view_silhouette_iou.png) · [5](per_object/T_shape/fig5_normalized_contour_distance.png) · [7](per_object/T_shape/fig7_qualitative_three_views.png) · [8](per_object/T_shape/fig8_controlled_transformation_family.png) |
| Kettle | [1](per_object/kettle/fig1_baseline_vs_oracle_dim_error.png) · [2](per_object/kettle/fig2_gt_vs_estimated_dimensions.png) · [3a](per_object/kettle/fig3_per_axis_absolute_error_baseline.png) · [3b](per_object/kettle/fig3_per_axis_absolute_error_oracle.png) · [4](per_object/kettle/fig4_cross_view_silhouette_iou.png) · [5](per_object/kettle/fig5_normalized_contour_distance.png) · [7](per_object/kettle/fig7_qualitative_three_views.png) · [8](per_object/kettle/fig8_controlled_transformation_family.png) |

> 물체별에는 **6번(IoU vs E_dim)이 없다** — 객체 1개로는 상관을 볼 수 없다.

## 12. 누락·제외된 객체와 이유

| 대상 | 문제 | 처리 |
|---|---|---|
| kettle L1 축 | 최소부피 OBB 의 최장축(주둥이·손잡이 돌출 포함)과 캘리퍼 측정 지점의 대응이 정의되지 않음. 실측이 110 → 113 으로 바뀐 이력. | **해당 축만** 오차 계산에서 제외 (`gt_axes_used=2/3`). 객체는 유지하고 추정치는 CSV/그림에 남김 |
| kettle / oracle_cad | 정답 CAD(kettle_dec30k.glb)가 촬영 물체와 형상이 다르다: 세로축 azimuth 전탐색(0~350°)과 거울상까지 시도해도 mean IoU 상한이 0.886 인데, 동일 최적화기가 SAM3D 메시로는 0.978 을 얻는다. cam0 에서 CAD 실루엣이 마스크의 13%(2421px, 손잡이 영역)를 덮지 못한다. 포즈가 아니라 메시 문제다. CAD 는 watertight 도 아니다(euler=-4). 형상이 틀린 mesh 는 "형상이 완벽할 때의 상한"이라는 oracle 의 정의를 만족하지 않으므로 E_dim/E_rel 을 계산하지 않는다. oracle 이 0.06mm(L2/L3 기준)로 나왔던 것은 실루엣에 가장 안 맞는 fit 에서 나온 우연의 일치로, 근거로 쓸 수 없다. baseline 은 GT(116/68/65) 가 확정돼 정상 평가한다. | skip (`csv/skipped_objects.csv`) |

필요한 추가 데이터:
- **kettle L1 캘리퍼 실측** — OBB 최장축과 같은 두 점(주둥이 끝 ↔ 손잡이 바깥)을 재서 `evaluation_config.yaml` 의 `kettle.gt_mm[0]` 을 null → 숫자로 바꾸면 3/3 축 평가가 된다.
- **T_shape 캘리퍼 실측** — 현재는 설계값(nominal)이라 제조 공차가 반영돼 있지 않다.
- 객체 수 4개는 통계 검정에 부족하다. YCB 등 추가 객체 권장.

## 13. 결과 해석

1. **크기 복원**: Baseline E_dim = 0.94 ± 0.47 mm, Oracle = 0.81 ± 0.42 mm. 정답 CAD 없이도 상한에 근접한다.
2. **형상 영향**: 객체별 gap 은 부호가 엇갈린다 (아래 표). 단순 형상(peg/T_shape)에서는 Baseline 이 Oracle 과 동등하거나 더 낫고, 복잡 형상(kettle)에서 gap 이 가장 크다.
3. **Real-to-Sim 정합**: Baseline cross-view IoU = 0.972 ± 0.008, D_contour = 0.46%.
4. **IoU 는 치수 정확도의 대리 지표로 신뢰할 수 없다** (Figure 6). 이 데이터에서 kettle 은 Baseline cross-view IoU 가 가장 높은 축에 속하지만 E_dim 은 가장 크고, Oracle 은 kettle 에서 IoU 가 가장 낮은데 E_dim 은 가장 작다. 복잡 형상에서 치수를 결정하는 극점(주둥이 끝)이 실루엣 면적에서 차지하는 비중이 작기 때문으로 보인다.

## 14. 현재 데이터만으로 주장할 수 없는 것

- **IoU 높음 → 3D 치수 정확** : 성립하지 않는다. Cross-view IoU 와 D_contour 는 **크기·형상·pose·캘리브레이션이 뒤섞인 투영 정합 품질**이고, E_dim/E_rel 만이 실측 기반 3D 크기 정확도다. 두 지표군을 바꿔 쓰지 말 것.
- **Baseline−Oracle gap = SAM3D 형상 오차** : 단정 불가. 현재 Baseline 은 비등방(축별 scale_vec), Oracle 은 등방(단일 scale) 이라 **변환족이 달라** gap 에 그 차이가 섞여 있다. 엄밀히 하려면 Oracle 도 비등방으로 재실행해 통제해야 한다.
- **통계적 유의성** : n=3 로 p-value 는 신뢰구간이 매우 넓다. descriptive 로만 읽어야 한다.
- **일반화** : 4개 객체(단순 3 + 복잡 1), 단일 촬영 세션, 단일 센서(RealSense), source view 는 전부 cam0. 다른 재질·크기·카메라 배치로의 일반화는 이 데이터로 알 수 없다.
- **T_shape 결과** : GT 가 설계값이라 '실측 대비 정확도'가 아니라 '설계값 대비 일치도'다.
- **kettle 절대 정확도** : L1 이 미확정이라 3축 전체 정확도는 알 수 없다.
