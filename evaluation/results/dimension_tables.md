# 물체별 크기 표 — GT / SAM3D / Oracle

축은 내림차순 rank (L1≥L2≥L3). 단위 mm. 괄호 안은 GT 대비 |오차|.


## Peg

| | L1 | L2 | L3 | E_dim |
|---|---|---|---|---|
| **GT (caliper_manual)** | 45.0 | 30.0 | 30.0 | — |
| **SAM3D (baseline)** | 44.49  (0.51) | 28.81  (1.19) | 28.64  (1.36) | 1.02 |
| **Oracle (GT CAD)** | 44.31  (0.69) | 29.01  (0.99) | 28.11  (1.89) | 1.19 |

- 형상: `simple` · SAM3D source view: `cam0`

## Hole

| | L1 | L2 | L3 | E_dim |
|---|---|---|---|---|
| **GT (caliper_manual)** | 50.0 | 50.0 | 50.0 | — |
| **SAM3D (baseline)** | 51.03  (1.03) | 49.12  (0.88) | 49.07  (0.93) | 0.95 |
| **Oracle (GT CAD)** | 50.86  (0.86) | 50.76  (0.76) | 49.01  (0.99) | 0.87 |

- 형상: `simple` · SAM3D source view: `cam0`

## T-shape

| | L1 | L2 | L3 | E_dim |
|---|---|---|---|---|
| **GT (cad_design_nominal)** | 150.0 | 100.0 | 50.0 | — |
| **SAM3D (baseline)** | 150.61  (0.61) | 99.77  (0.23) | 50.14  (0.14) | 0.33 |
| **Oracle (GT CAD)** | 149.46  (0.54) | 99.64  (0.36) | 49.82  (0.18) | 0.36 |

- 형상: `simple` · SAM3D source view: `cam0`

## Kettle

| | L1 | L2 | L3 | E_dim |
|---|---|---|---|---|
| **GT (caliper_manual)** | 116.0 | 68.0 | 65.0 | — |
| **SAM3D (baseline)** | 117.31  (1.31) | 69.09  (1.09) | 67.02  (2.02) | 1.47 |
| **Oracle (GT CAD)** | — | — | — | CAD 없음 |

> Oracle 없음: **Kettle 는 정답 CAD 가 없다** (baseline 전용).

- 형상: `complex` · SAM3D source view: `cam0`
