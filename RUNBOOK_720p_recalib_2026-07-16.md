# 720p 재캘리브 → 새 물체 촬영 → 크기 추정 런북

작성일: 2026-07-16
대상: static D415 3대 (그리퍼 D435 제외), 해상도 1280×720

---

## 사전 정리
- **그리퍼 D435는 이번 3대 캘리브에서 제외** — USB에서 물리적으로 뽑거나 device_map에 안 들어오게 한다.
  연결돼 있으면 시리얼 정렬상 cam2를 차지해 인덱스가 밀린다. 3대만 연결하면 cam0/1/2 = 314/319/912로 깔끔.
- 폴더는 새로 시작하므로 캘리브는 **새 세션 폴더**(`static_cams_session_02`)에 저장한다.

---

## 1) intrinsics 덤프 @ 720p
```bash
python Calib_Step1_dump_intrinsics.py \
  --color_w 1280 --color_h 720 --color_fps 15 \
  --depth_w 1280 --depth_h 720 --depth_fps 15 --fresh_map
```
**확인:** `[SAVE] cam0.npz / cam1.npz / cam2.npz` 3개 다 찍히고, 실패 목록 경고가 없어야 함.

---

## 2) 멀티캠 촬영 @ 720p (큐브는 EE가 잡고 이동해도 OK)
```bash
python Calib_Step2_capture_multi_cam.py \
  --root_folder ./data/static_cams_session_02 \
  --intrinsics_dir ./intrinsics \
  --width 1280 --height 720 --min_markers 2 --show --preview_scale 0.5
```
큐브를 다양한 위치·자세로 옮기며 3대가 동시에 잘 보는 프레임을 충분히(수십 장) 저장.
팔/그리퍼가 큐브를 가리지 않게 한다.

- `--preview_scale` (기본 0.5): 미리보기 창 축소 배율. 720p 4패널(2560×1440)이 0.5면 1280×720으로 표시된다.
  저장 이미지는 원본 해상도 그대로이고 **미리보기만** 줄어든다. 더 작게: `--preview_scale 0.35`,
  축소 없이: `--preview_scale 1.0`. 창은 `WINDOW_NORMAL`이라 마우스 드래그로도 크기 조절 가능.

---

## 3) 캘리브레이션 (3대 한번에)
python Calib_Step3_calibrate_multi_cam_cube.py \
  --root_folder ./data/static_cams_session_02 \
  --intrinsics_dir ./intrinsics \
  --ref_cam_idx 0 --min_markers 1 --save_overlay
```
**확인:** 끝에 `reproj_rms_px` 값 — 낮을수록(대략 1px 내외) 좋음.
결과: `./data/static_cams_session_02/calib_out_cube/transforms/T_C0_Ci_all.json`

---

## 4) 새 물체 촬영 @ 720p (K·extrinsic 자동 생성)
```bash
python Obj_Step1_capture_object.py \
  --out_dir ./data/capture_obj \
  --intrinsics_dir ./intrinsics \
  --transforms_json ./data/static_cams_session_02/calib_out_cube/transforms/T_C0_Ci_all.json \
  --width 1280 --height 720
```
`--transforms_json`을 넣었으므로 `cam*_K.txt`, `cam*_T_cam_to_world.txt`가 자동 생성됨(수동 작업 불필요).

---

## 5) 마스킹 (SAM2, sam2env)
```bash
SET="data/capture_obj"; MASKS="data/masks"
PYTHONWARNINGS=ignore /home/sprout/anaconda3/envs/sam2env/bin/python3 \
  Obj_Step2_mask_sam2.py --capture_dir "$SET" --masks_dir "$MASKS" \
  --sam_checkpoint ~/sam2_checkpoints/sam2_hiera_large.pt \
  --sam_config configs/sam2/sam2_hiera_l.yaml --num_objects 2 --device cpu
```

---

## 6) 크기 추정 (SAM3D — sam2env 아님, SAM3D 환경 필요)
```bash
python Obj_Step3_sam3d_scale.py \
  --data_dir ./data/capture_obj --mask_dir ./data/masks --out_dir ./data/outputs \
  --depth_scale 0.001 --mask_erode_px 3 --keep_largest_cc \
  --run_sam3d --sam3d_cam cam1 --spconv_algo native \
  --estimate_size --size_save_overlay
```
출력:
- `<obj>_sam3d_scaled.glb` (실척·meter·원점중심, FoundationPose/Isaac 입력)
- `<obj>_size.json` (scale, 치수, per-view IoU)
- `<obj>_size_overlay.jpg`

---

## 주의
- **6단계는 SAM3D 환경**이 따로 필요하다(레포의 `sam3d_env_gb10.sh` 참고). 5단계 sam2env와 다르다.
- 크기는 **실루엣 정합**으로 정해진다. `_size.json`의 **mean IoU가 낮으면**(`--size_min_iou` 미만)
  SAM3D 형상이 관측과 어긋난 것이라 치수를 신뢰하지 말라는 뜻 — `--sam3d_cam`을 물체가 더 잘 보이는
  카메라로 바꿔 재시도한다.

---

## 참고 (배경)
- **640→720 해상도 변경은 기존 캘리브 전체를 무효화한다.** 인트린식이 바뀌므로 멀티캠 큐브 캘리브와
  eye-to-hand(base 정합) 결과 모두 720p로 다시 풀어야 한다.
- **그리퍼 D435를 4번째 static 카메라로 넣으려면** topview로 고정한 채 큐브를 손으로 옮겨야 한다
  (EE가 큐브를 잡으면 팔에 달린 그리퍼 카메라도 같이 움직여 static 가정이 깨진다).
- **base 좌표계 정합은 별도 작업** — EE에 큐브를 장착하고 로봇을 움직이는 eye-to-hand 촬영 +
  `Calib_Step3ee`/`Calib_Step4ee` 솔브가 필요하다. 크기 추정에는 base가 필요 없다(cam0 기준으로 충분).
