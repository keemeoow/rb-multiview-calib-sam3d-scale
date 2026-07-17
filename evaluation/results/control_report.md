# 통제 실험 — 변환족을 맞춘 Baseline−Oracle gap

생성: `python evaluation/analyze_controlled_gap.py --config evaluation/evaluation_config.yaml --output evaluation/results`


## 왜 했나

메인 평가의 Baseline 은 **비등방**(축별 `scale_vec`), Oracle 은 **등방**(단일 scale) 이라 gap 에 두 요인이 섞여 있었다:

```
기존 gap = f(SAM3D 형상 오차,  등방/비등방)
통제 gap = f(SAM3D 형상 오차)                <- 정답 CAD 를 비등방으로 재정합
```

**통제 방법**: 정답 CAD 를 `_silhouette_fit.fit_mesh_aniso` 로 재정합하되, **발표된 등방 oracle 해를 warm start 로 고정**해 출발점을 동일하게 두고 변환족만 넓혔다. 나머지 조건(`w_depth=0.0`, `max_fev=4000`, `aniso_reg=0.0`, 동일 마스크·점군)도 Baseline 과 맞췄다. 기존 추정 코드는 수정하지 않았다.


## 결과

| Object | Baseline (SAM3D, aniso) | Oracle (CAD, iso) | Oracle (CAD, aniso) | gap 기존 | **gap 통제** |
|---|---|---|---|---|---|
| Peg | 1.02 | 1.19 | 1.12 | -0.17 | **-0.10** |
| Hole | 0.95 | 0.87 | 0.87 | +0.08 | **+0.07** |
| T-shape | 0.33 | 0.36 | 0.48 | -0.04 | **-0.16** |
| Kettle | 1.56 | 0.06 | 3.07 | +1.49 | **-1.51** |
| **Mean** | **0.96** | 0.62 | 1.39 | +0.34 | **-0.42** |

- gap 기존 = **+0.34 ± 0.77 mm**
- gap 통제 = **-0.42 ± 0.73 mm** → **부호가 뒤집힌다**

그림: [fig8_controlled_transformation_family](figures/fig8_controlled_transformation_family.png)


## 해석

1. **기존 gap(+0.34mm)의 상당 부분은 SAM3D 형상 오차가 아니라 변환족 차이였다.** 변환족을 맞추면 gap 이 음수로 바뀐다 — 즉 같은 비등방 조건에서는 SAM3D 단서 메시가 정답 CAD 보다 **나쁘지 않았다**.
2. **비등방은 형상이 맞을 때 오히려 해롭다.** Oracle 은 등방 0.62mm → 비등방 1.39mm 로 악화됐다. 형상이 이미 정확하면 축별 자유도는 실루엣에 **과적합**할 여지만 준다.
3. **kettle 이 그 과적합의 교과서적 사례다.** 비등방 Oracle 은 IoU 를 0.886 → 0.903 로 **개선**하면서 E_dim 은 0.06 → 3.07 mm 로 **악화**시켰다. 실루엣 손실을 낮추면서 치수를 망가뜨린 것이다.
4. 따라서 **최적 변환족은 단서 메시의 형상 정확도에 의존한다** (상호작용):
   - 형상이 정확(Oracle) → **등방**이 낫다 (자유도를 주면 과적합)
   - 형상이 추정치(SAM3D) → **비등방**이 낫다 (형상 비율 오류를 교정; `size_method_experiment.json` 의 peg 4.37 → 1.05mm)

## 두 gap 은 서로 다른 질문에 답한다

| | 비교 | 답하는 질문 |
|---|---|---|
| **gap 기존** (+0.34mm) | Baseline(aniso) vs Oracle(iso) | 각 방법을 **각자 최적 설정**으로 썼을 때의 실전 격차 |
| **gap 통제** (−0.42mm) | Baseline(aniso) vs Oracle(aniso) | **변환족 고정** 시 단서 메시 형상만의 영향 |

둘 다 유효하며 어느 하나가 다른 하나를 대체하지 않는다. 논문에 쓸 때 어떤 비교인지 명시해야 한다.


## 이 실험으로도 주장할 수 없는 것

- **gap = SAM3D 순수 형상 오차** : 여전히 단정 불가. 변환족은 통제했지만 마스크 품질, 캘리브레이션, Powell 국소해 등은 통제되지 않았다.
- **통계적 유의성** : n=4 로 표본이 매우 적다. descriptive 로만 읽을 것. 특히 통제 gap 평균(−0.42)은 kettle 의 비등방 과적합(−1.51)이 끌어내린 값이라, 나머지 3개(−0.10, +0.07, −0.16)만 보면 사실상 0 에 가깝다.
- **'비등방이 항상 나쁘다'** : 성립하지 않는다. 형상이 틀린 Baseline 에서는 비등방이 크게 이롭다는 것이 이미 확인돼 있다 (`size_method_experiment.json`).
- **kettle 절대 정확도** : L1 이 GT 미확정이라 2/3 축으로만 계산된 값이다.
