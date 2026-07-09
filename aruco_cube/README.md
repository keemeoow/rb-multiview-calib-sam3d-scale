# aruco_cube/ — 보관용 (더 이상 사용하지 않음)

30mm ArUco 큐브(`DICT_4X4_50`, 22mm 마커 5개) 구현을 보관한 폴더입니다.
**현재 캘리브레이션 파이프라인은 이 코드를 임포트하지 않습니다.**

활성 구현은 저장소 루트의 [`_apriltag_cube.py`](../_apriltag_cube.py) 이며,
59mm AprilTag 큐브(`DICT_APRILTAG_36h11`)를 정의합니다.

| | 보관된 ArUco 큐브 | 현재 AprilTag 큐브 |
|---|---|---|
| dictionary | `DICT_4X4_50` | `DICT_APRILTAG_36h11` |
| 큐브 한 변 | 30mm | 59mm |
| 마커 | 22mm × 5개 (ID 0~4) | 상면 25mm × 2개 + 측면 51mm × 4개 (ID 0~5) |
| 마커 크기 | 전부 동일 | 면마다 다름 (`marker_size_by_id`) |
| 마커 중심 | 면 중심 | 면별 지정 (`marker_center_m`) |
| `face_roll_deg` | 0/270/0/90/180 (실물 검증됨) | 전부 0 (**실물 검증 필요**) |

## 왜 남겨두었나

`data/` 아래 기존 세션(`meta.json`, `calib_out_cube/`)은 이 30mm ArUco 큐브로
촬영·계산된 것이라 새 큐브 정의로는 재현되지 않습니다. 옛 데이터를 다시 열어봐야
할 때 참조용으로만 사용하세요.

```python
# 옛 세션을 재현해야 할 때만
from aruco_cube._aruco_cube import CubeConfig, ArucoCubeTarget
```

새로 촬영하는 데이터는 `Calib_Step2` 부터 AprilTag 큐브로 진행하면 됩니다.
