# evaluation/ — Real-to-Sim 객체 크기 추정 정량 평가

기존 추정 파이프라인(`Obj_Step3*`, `_silhouette_fit.py`)은 **읽기만** 하고 수정하지 않는다.
결과 파일에서 값을 읽어 계산하며, 없는 값은 만들어내지 않고 skip 사유를 기록한다.

## 실행

```bash
# 메인 평가 — CSV · 통합 그래프 · 물체별 그래프 · 정성 피규어 · 리포트
python evaluation/run_all_evaluations.py \
    --config evaluation/evaluation_config.yaml \
    --output evaluation/results

# 통제 실험 (선택) — Oracle 을 비등방으로 재정합해 변환족을 Baseline 과 맞춘다
python evaluation/run_oracle_aniso_control.py \
    --config evaluation/evaluation_config.yaml --output evaluation/results   # 약 3분 40초
python evaluation/analyze_controlled_gap.py \
    --config evaluation/evaluation_config.yaml --output evaluation/results
```

메인 평가는 한 명령으로 전부 재생성된다. `--skip_qualitative` 로 정성 피규어만 건너뛸 수 있다.

> ⚠ **`results/control_oracle_aniso/` 를 지우지 말 것.** 통제 실험 fit 결과이고 재생성에
> 약 3분 40초가 걸린다 (Powell 최적화 × 4객체). `rm -rf results` 하면 같이 사라지므로,
> 통제 결과를 유지한 채 메인만 다시 돌리려면 `results` 를 통째로 지우지 말고
> `run_all_evaluations.py` 를 그냥 재실행하면 된다 (덮어쓴다).

의존성: `numpy pandas matplotlib scipy opencv-python trimesh open3d pyyaml`
(파이프라인과 동일 환경. macOS 로컬 검증은 `/usr/local/bin/python3.11` 사용.)

## 평가 대상

| # | 지표군 | 의미 |
|---|---|---|
| 1 | **E_dim**, **E_rel** | 실측 대비 **3D 크기 정확도** (metric scale) |
| 2 | **Cross-view IoU**, **D_contour** | **Real-to-Sim 투영 정합 품질** (크기·형상·pose·캘리브레이션 혼합) |

> ⚠ 두 지표군은 서로 대체할 수 없다. IoU 가 높다고 3D 치수가 정확한 것은 아니다
> (Figure 6 에서 이 데이터로 실제 확인됨).

파지 관련 지표, BBox width/height/center, pose(위치·회전) 정확도는 **평가 대상에서 제외**한다.

## 비교 방법

| method | 단서 mesh | 성격 |
|---|---|---|
| `baseline_sam3d` | SAM3D 가 단일 RGB+mask 로 생성 (unitless) | **실제 운용 방법** |
| `oracle_cad` | 정답 CAD | **정확도 상한** (형상이 완벽할 때) |

동일한 Sim(3) 7-DoF 실루엣 정합 엔진(`_silhouette_fit`)을 쓰고 **단서 mesh 만 다르다.**
Baseline−Oracle gap 은 "단서 메시를 SAM3D → 정답 CAD 로 교체했을 때 변화한 최종 치수 오차"로
읽어야 하며, SAM3D 의 순수 형상 오차라고 단정할 수 없다.

## 파일

| 파일 | 역할 |
|---|---|
| `evaluation_config.yaml` | 물체별 입력 경로·GT·source camera·스타일. **경로 하드코딩 없음** |
| `eval_common.py` | 로딩, sim 마스크 재렌더링, IoU/contour/축대응 지표 |
| `evaluate_metric_scale.py` | 평가 1 — E_dim / E_rel |
| `evaluate_silhouette_alignment.py` | 평가 2 — per-camera IoU / contour distance |
| `generate_evaluation_figures.py` | Figure 1~6 |
| `generate_qualitative_overlays.py` | 물체별 3-view + Figure 7 그리드 |
| `run_all_evaluations.py` | 전체 진입점 + CSV + 통계 + 리포트 |

## sim 마스크 재현 방식 (중요)

IoU 는 결과 JSON 에 저장돼 있지만 contour distance 는 없어 sim 마스크를 **재렌더링**한다.

- `oracle_cad` — CAD 정점 + `scale_cad_to_world` + `T_world_cad_4x4`. 모호함 없음.
- `baseline_sam3d` — **원본 SAM3D mesh + 파이프라인과 동일한 open3d quadric decimation +
  `scale_vec`** 로 정합 당시 프레임을 그대로 복원한다.
  (내보낸 `*_sam3d_scaled.glb` 는 AABB 중심으로 평행이동돼 있어 `T_world_mesh_4x4` 와
  직접 맞지 않는다. glb 기반 복원은 대체 경로로만 남겨뒀다.)

재현 정확도는 엔진과 **동일한 슈퍼샘플 조건**의 IoU 를 저장된 `per_view_iou` 와 대조해
`iou_reproduction_delta` 로 CSV 에 남긴다 (현재 데이터: oracle 전부 0.0000, baseline 최대 0.0045).

## Cross-view 정의

SAM3D 는 **source view 의 RGB+mask 로 mesh 를 만들었으므로** 그 뷰에는 거의 자기 자신에 맞는다.
따라서 **메인 지표는 source view 를 제외한 cross-view 평균**이다.
`source_view_iou` / `cross_view_iou` / 카메라별 IoU 를 모두 구분 저장한다.

현재 데이터는 4개 물체 모두 source = `cam0` → cross-view = `cam1`, `cam2` 평균.

## GT 미확정 축

`gt_mm` 에 `null` 을 쓰면 그 축은 오차 계산에서 제외되고, 추정치는 CSV·그림에 남으며
`n/d` 로 표기된다 (`gt_axes_used` 에 사용 축 수 기록). 억지 숫자를 넣지 않기 위한 장치다.

현재: `kettle.gt_mm[0] = null` — 최소부피 OBB 의 최장축(주둥이·손잡이 돌출 포함)과
캘리퍼 측정 지점의 대응이 정의되지 않음.

## 출력

```
evaluation/results/
├── csv/            evaluation_per_object.csv / per_camera.csv / summary.csv
│                   fig2_scatter_points.csv, paired_tests.json, skipped_objects.csv
│                   controlled_gap.csv, controlled_gap_summary.json
├── figures/        ★ 통합 (전체 객체) — fig1~fig8
├── per_object/     ★ 물체별 폴더 — <object>/fig1~fig5, fig7, fig8
│   ├── peg/  hole/  T_shape/  kettle/
├── qualitative/    qualitative_<object>_three_views.png
├── captions/       figN_caption.txt
├── control_oracle_aniso/   통제 실험 fit 결과 (*_cad_fit_aniso.json)
├── evaluation_report.md
└── control_report.md
```

- **형식은 PNG(300dpi) 만** 저장한다 (`style.formats: [png]`). 벡터가 필요하면 config 에
  `[png, svg, pdf]` 로 되돌리면 된다.
- 각 그림 상단에 **`번호. 제목`** 이 표기된다 (`FIG_TITLES`). 물체별 그림은 `— <물체>` 가 붙는다.
- 물체별 폴더에는 **Figure 6 이 없다** — 객체 1개로는 IoU–오차 상관을 볼 수 없어 건너뛴다.
  Figure 1 의 가로형 변형도 통합본에만 만든다.

## 새 물체 추가

`evaluation_config.yaml` 의 `objects:` 에 블록을 추가한다. 필요한 것:

- `capture_dir` (`cam*_K.txt`, `cam*_T_cam_to_world.txt`, `cam*_rgb.png`)
- `mask_dir` (`cam*_mask.png`)
- `source_camera` — SAM3D 입력 뷰. `objects_summary.json` 의 `sam3d_cam` 또는
  SAM3D 입력 저장본 파일명(`<tag>_<cam>_rgb.png`)에서 확인
- `gt_mm` — 내림차순(L≥W≥H), 미확정 축은 `null`
- `oracle.fit_json` + `oracle.mesh`, `baseline.size_json` (+ 경로가 깨졌으면 `baseline.mesh`)

파일이 없으면 전체 실행이 멈추지 않고 warning 과 함께 skip 되며 `csv/skipped_objects.csv` 에 기록된다.
