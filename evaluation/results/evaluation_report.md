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
| kettle | `data/capture_obj_02` | `data/masks_02/obj1` | `data/outputs_sam3d_02/obj1/obj1_size.json` | — (정답 CAD 없음) | cam0 |

## 3. GT dimension 출처

| Object | GT (mm, L≥W≥H) | 출처 | 비고 |
|---|---|---|---|
| peg | 45.0 × 30.0 × 30.0 | `caliper_manual` | — |
| hole | 50.0 × 50.0 × 50.0 | `caliper_manual` | — |
| T_shape | 150.0 × 100.0 × 50.0 | `cad_design_nominal` | — |
| kettle | 116.0 × 68.0 × 65.0 | `caliper_manual` | — |

- `caliper_manual`: `configs/evaluation.yaml` 의 `measured.extents_mm` (캘리퍼 실측).
- `cad_design_nominal` (T_shape): `T_shape.glb` 의 고유 OBB 가 정확히 150/100/50 이고 추정 scale 이 0.001(mm→m)에 0.36% 로 수렴 → mm 단위 설계 CAD 로 판단해 **설계값**을 GT 로 사용. **캘리퍼 실측이 아니다.**
- `caliper_manual_partial`: 일부 축만 확정. 미확정 축은 오차 계산에서 제외한다 (§9).

## 4. Baseline / Oracle 정의

- **Baseline (`baseline_sam3d`)**: SAM3D 가 단일 RGB+mask 로 만든 unitless 단서 mesh 를 3대 카메라 SAM2 실루엣에 Sim(3) 7-DoF 로 정합. 최종 치수 = mesh OBB extents × 추정 scale. **실제 운용 방법** (정답 CAD 불필요).
- **Oracle (`oracle_cad`)**: 단서 mesh 만 정답 CAD 로 교체. 엔진·손실·최적화 동일. **형상이 완벽할 때의 정확도 상한**.
- 현재 데이터의 Baseline 은 모두 `sam3d_anisotropic_silhouette` (축별 scale_vec), Oracle 은 `cad_multiview_silhouette` (등방 scale) 이다. → **두 방법이 완전히 동일한 변환족은 아니다** (§11).

## 5. 평가 지표 수식

보고하는 지표는 아래 5개(+정성 오버레이)뿐이다.

```
[1] e_L = |L_hat - L_GT|,  e_W = |W_hat - W_GT|,  e_H = |H_hat - H_GT|   [mm]
[2] E_dim = (e_L + e_W + e_H) / 3                                        [mm]
[3] E_rel = (1/3)[e_L/L_GT + e_W/W_GT + e_H/H_GT] x 100                  [%]
[4] IoU_i = |M_real_i ∩ M_sim_i| / |M_real_i ∪ M_sim_i|
    Cross-view IoU = mean over cameras EXCLUDING the SAM3D source view
[5] D_contour = 0.5[ mean_{p∈C_real} min_q ||p-q|| + mean_{q∈C_sim} min_p ||q-p|| ]  [px]
    D_contour_norm = D_contour / sqrt(w_real^2 + h_real^2)               [%]
[6] 정성 오버레이: real(green) / sim(red) 실루엣 외곽선 중첩 이미지
```
- 축 대응은 파이프라인·GT 모두 내림차순(L≥W≥H) 정렬 기준이다 (`rank_descending`).
- GT 미확정 축은 e_* 를 계산하지 않고 평균에서 제외 (`gt_axes_used` 에 사용 축 수 기록).
- sim 마스크는 결과 JSON 의 fit 파라미터로 재렌더링했고, IoU 는 파이프라인이 저장한 `per_view_iou` 를 그대로 쓴다. contour distance 만 재렌더 마스크에서 계산한다.

## 6. 객체별 결과

| Object | Method | GT (mm) | Estimated (mm) | e_L | e_W | e_H | **E_dim** (mm) | E_rel (%) | **cross IoU** | cross D_contour (%) |
|---|---|---|---|---|---|---|---|---|---|---|
| Peg | `baseline_sam3d` | 45.0 × 30.0 × 30.0 | 44.5 × 28.8 × 28.6 | 0.51 | 1.19 | 1.36 | **1.02** | 3.21 | **0.960** | 0.69 |
| Peg | `oracle_cad` | 45.0 × 30.0 × 30.0 | 44.3 × 29.0 × 28.1 | 0.69 | 0.99 | 1.89 | **1.19** | 3.72 | **0.959** | 0.50 |
| Hole | `baseline_sam3d` | 50.0 × 50.0 × 50.0 | 51.0 × 49.1 × 49.1 | 1.03 | 0.88 | 0.93 | **0.95** | 1.90 | **0.973** | 0.47 |
| Hole | `oracle_cad` | 50.0 × 50.0 × 50.0 | 50.9 × 50.8 × 49.0 | 0.86 | 0.76 | 0.99 | **0.87** | 1.74 | **0.976** | 0.49 |
| T-shape | `baseline_sam3d` | 150.0 × 100.0 × 50.0 | 150.6 × 99.8 × 50.1 | 0.61 | 0.23 | 0.14 | **0.33** | 0.31 | **0.976** | 0.34 |
| T-shape | `oracle_cad` | 150.0 × 100.0 × 50.0 | 149.5 × 99.6 × 49.8 | 0.54 | 0.36 | 0.18 | **0.36** | 0.36 | **0.980** | 0.23 |
| Kettle | `baseline_sam3d` | 116.0 × 68.0 × 65.0 | 117.3 × 69.1 × 67.0 | 1.31 | 1.09 | 2.02 | **1.47** | 1.95 | **0.978** | 0.33 |

> `n/d` = GT 미확정 축이라 그 축의 오차를 계산하지 않았다는 뜻이다 (§9).

## 7. 전체 (mean ± std)

| Method | n | E_dim (mm) | E_rel (%) | Cross-view IoU | D_contour (%) |
|---|---|---|---|---|---|
| `baseline_sam3d` | 4 | 0.94 ± 0.47 | 1.84 ± 1.19 | 0.972 ± 0.008 | 0.46 ± 0.16 |
| `oracle_cad` | 3 | 0.81 ± 0.42 | 1.94 ± 1.69 | 0.972 ± 0.011 | 0.41 ± 0.15 |

## 8. 생성한 그래프와 피규어

모두 PNG(300dpi). 각 그림 상단에 `번호. 제목` 이 표기된다. 지표 6개에 그림 6개가 1:1 로 대응한다.

### 통합 (전체 객체) — `figures/`

- [1. 평균 Dimension Error (E_dim)](figures/fig1_mean_dimension_error.png)
- [2. 평균 상대 Dimension Error (E_rel)](figures/fig2_mean_relative_dimension_error.png)
- [3a. 축별 절대오차 (Baseline)](figures/fig3_per_axis_absolute_error_baseline.png)
- [3b. 축별 절대오차 (Oracle)](figures/fig3_per_axis_absolute_error_oracle.png)
- [4. Cross-view Silhouette IoU](figures/fig4_cross_view_silhouette_iou.png)
- [5. Normalized Contour Distance](figures/fig5_normalized_contour_distance.png)
- [6. 정성 오버레이 통합 그리드](figures/fig6_qualitative_real_to_sim_grid.png)

대표 객체 (6. 그리드 행): T_shape, peg, kettle — 최소오차/중간/최대오차 + 단순·복잡 형상 포함

### 물체별 — `per_object/<object>/`  ·  오버레이 원본 — `qualitative/`

| Object | 그림 |
|---|---|
| Peg | [1](per_object/peg/fig1_mean_dimension_error.png) · [2](per_object/peg/fig2_mean_relative_dimension_error.png) · [3a](per_object/peg/fig3_per_axis_absolute_error_baseline.png) · [3b](per_object/peg/fig3_per_axis_absolute_error_oracle.png) · [4](per_object/peg/fig4_cross_view_silhouette_iou.png) · [5](per_object/peg/fig5_normalized_contour_distance.png) · [6](per_object/peg/fig6_qualitative_three_views.png) |
| Hole | [1](per_object/hole/fig1_mean_dimension_error.png) · [2](per_object/hole/fig2_mean_relative_dimension_error.png) · [3a](per_object/hole/fig3_per_axis_absolute_error_baseline.png) · [3b](per_object/hole/fig3_per_axis_absolute_error_oracle.png) · [4](per_object/hole/fig4_cross_view_silhouette_iou.png) · [5](per_object/hole/fig5_normalized_contour_distance.png) · [6](per_object/hole/fig6_qualitative_three_views.png) |
| T-shape | [1](per_object/T_shape/fig1_mean_dimension_error.png) · [2](per_object/T_shape/fig2_mean_relative_dimension_error.png) · [3a](per_object/T_shape/fig3_per_axis_absolute_error_baseline.png) · [3b](per_object/T_shape/fig3_per_axis_absolute_error_oracle.png) · [4](per_object/T_shape/fig4_cross_view_silhouette_iou.png) · [5](per_object/T_shape/fig5_normalized_contour_distance.png) · [6](per_object/T_shape/fig6_qualitative_three_views.png) |
| Kettle | [1](per_object/kettle/fig1_mean_dimension_error.png) · [2](per_object/kettle/fig2_mean_relative_dimension_error.png) · [3a](per_object/kettle/fig3_per_axis_absolute_error_baseline.png) · [3b](per_object/kettle/fig3_per_axis_absolute_error_oracle.png) · [4](per_object/kettle/fig4_cross_view_silhouette_iou.png) · [5](per_object/kettle/fig5_normalized_contour_distance.png) · [6](per_object/kettle/fig6_qualitative_three_views.png) |

## 9. 누락·제외된 객체와 이유

| 대상 | 문제 | 처리 |
|---|---|---|
| kettle / oracle_cad | 정답 CAD 없음 — 이 물체는 baseline 전용 | skip (`csv/skipped_objects.csv`) |

필요한 추가 데이터:
- **설계값(nominal) GT 물체의 캘리퍼 실측** — 제조 공차가 반영돼 있지 않다.
- 객체 수 4개는 일반화를 주장하기에 부족하다. YCB 등 추가 객체 권장.

## 10. 결과 요약

1. **크기 정확도 (지표 1~3)**: Baseline E_dim = 0.94 ± 0.47 mm, E_rel = 1.84 ± 1.19 %. 축별 분포는 Figure 3 참조 — 물체 평균이 가리는 축별 편차가 보인다.
2. **Real-to-Sim 정합 (지표 4~5)**: Baseline cross-view IoU = 0.972 ± 0.008, D_contour = 0.46 ± 0.16 %.
3. **정성 확인 (지표 6)**: Figure 6 및 `qualitative/` 의 오버레이에서 real/sim 외곽선 일치를 눈으로 확인한다. source view 는 mesh 를 만든 뷰라 흐리게 처리했다.

## 11. 지표 해석 시 주의

- **IoU 높음 → 3D 치수 정확** : 성립하지 않는다. Cross-view IoU 와 D_contour 는 **크기·형상·pose·캘리브레이션이 뒤섞인 투영 정합 품질**이고, E_dim/E_rel 만이 실측 기반 3D 크기 정확도다. 두 지표군을 바꿔 쓰지 말 것.
- **Baseline 과 Oracle 의 직접 비교** : 현재 Baseline 은 비등방(축별 scale_vec), Oracle 은 등방(단일 scale) 이라 **변환족이 다르다**. 두 열은 각각의 지표값으로 읽고, 차이를 형상 오차로 환산하지 말 것.
- **일반화** : 4개 객체(단순 3 + 복잡 1), 단일 촬영 세션, 단일 센서(RealSense), source view 는 전부 cam0. 다른 재질·크기·카메라 배치로의 일반화는 이 데이터로 알 수 없다.
- **T_shape** : GT 가 설계값이라 '실측 대비 정확도'가 아니라 '설계값 대비 일치도'다.
