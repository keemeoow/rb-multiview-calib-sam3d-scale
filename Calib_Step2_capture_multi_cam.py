# Calib_Step2_capture_multi_cam.py
"""
    멀티캠 RGB+depth 캡처 (SPACE 또는 auto_save).
    --robot_ip 지정 시 SPACE 저장마다 robot_pose_server 로 get_pose 쿼리하여
    joint / tcp_base_ee_4x4 / tcp_base_cube_4x4 를 meta.json 에 임베드.
    저장 직후 notify_saved 도 전송 → 서버 터미널에 [CAPTURED] banner 출력.
"""
"""
 (1) 멀티캠 정적 캘리브용 (Step3, Step4 까지 진행 시):
     python Calib_Step2_capture_multi_cam.py \
       --root_folder ./data/static_cams_session_01 \
       --intrinsics_dir ./intrinsics \
       --min_markers 2 --show

 (2) Hand-to-eye 캡처용 (cube on EE, 로봇 자세별 SPACE):
     python Calib_Step2_capture_multi_cam.py \
       --root_folder ./data/handeye_session_02 \
       --intrinsics_dir ./intrinsics \
       --robot_ip 192.168.0.23 --robot_port 12348 \
       --min_markers 2 --show

         (2)-1 동시에 Terminal 3에서 Calib_Step2ee_replay_joint_sequence.py 가 로봇을 자동으로
 각 자세로 이동시키면, 사용자는 OpenCV 창에서 SPACE 만 누르면 됨.
"""

import os
import json
import time
import socket
import argparse
from typing import Dict, Optional

import cv2
import numpy as np
import sys

# Allow running this script from inside src/Step0_calibration by adding the
# repository root (parent of `src`) to sys.path so `import src...` works.
THIS_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from _camera import RealSenseCamera
from _apriltag_cube import CubeConfig, AprilTagCubeTarget


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


# ---------------------------------------------------------------- #
# Robot pose helpers (talks to robot_pose_server.py over TCP/JSON)
# ---------------------------------------------------------------- #

def pose6_mm_deg_to_T(pose6) -> np.ndarray:
    """[x_mm, y_mm, z_mm, rz_deg, ry_deg, rx_deg] -> 4x4 (meters).

    Rotation convention: ZYX intrinsic (R = Rz @ Ry @ Rx). i611 Position 의
    (rz, ry, rx) 표기에 맞춤. 캘리브 결과가 비정상이면 이 한 함수만 바꾸면 됨.
    """
    x, y, z = float(pose6[0]) / 1000.0, float(pose6[1]) / 1000.0, float(pose6[2]) / 1000.0
    rz, ry, rx = (np.deg2rad(float(pose6[3])),
                  np.deg2rad(float(pose6[4])),
                  np.deg2rad(float(pose6[5])))
    cz, sz = np.cos(rz), np.sin(rz)
    cy, sy = np.cos(ry), np.sin(ry)
    cx, sx = np.cos(rx), np.sin(rx)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    R = Rz @ Ry @ Rx
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


def robot_connect(ip: str, port: int, timeout_sec: float = 2.0):
    sk = socket.create_connection((ip, port), timeout=timeout_sec)
    sk.settimeout(timeout_sec)
    return sk


def robot_query_pose(sk: socket.socket, timeout_sec: float = 2.0) -> Optional[dict]:
    """Send {"command":"get_pose"}; return parsed reply dict or None on error."""
    try:
        sk.settimeout(timeout_sec)
        sk.sendall((json.dumps({"command": "get_pose"}) + "\n").encode("utf-8"))
        buf = b""
        while b"\n" not in buf:
            chunk = sk.recv(65536)
            if not chunk:
                return None
            buf += chunk
        line = buf.split(b"\n", 1)[0]
        resp = json.loads(line.decode("utf-8"))
        if not isinstance(resp, dict) or resp.get("status") != "ok":
            print("[WARN] robot pose error: {}".format(
                resp.get("reason") if isinstance(resp, dict) else resp))
            return None
        return resp
    except Exception as e:
        print("[WARN] robot pose query failed: {}".format(e))
        return None


def capture_depth_burst(cam, n_frames: int, max_wait_ms: int = 1500):
    """카메라에서 ts_ms 가 다른 N장의 depth 프레임을 모은다.
    Returns (n, H, W) uint16 ndarray 또는 None."""
    depths = []
    last_ts = None
    start_t = time.monotonic()
    while len(depths) < n_frames:
        elapsed_ms = (time.monotonic() - start_t) * 1000.0
        if elapsed_ms > max_wait_ms:
            break
        _, depth, ts_ms = cam.get_latest()
        if depth is None or ts_ms == last_ts:
            time.sleep(0.003)
            continue
        depths.append(depth)
        last_ts = ts_ms
    if not depths:
        return None
    return np.stack(depths, axis=0)


def median_depth_ignore_zero(stack: Optional[np.ndarray]):
    """Per-pixel median of (N, H, W) uint16 depth stack, ignoring zeros."""
    if stack is None or stack.shape[0] == 0:
        return None
    f = stack.astype(np.float32)
    f = np.where(f > 0, f, np.nan)
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(f, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    return med.astype(np.uint16)


def load_device_map(intr_dir: str):
    map_path = os.path.join(intr_dir, "device_map.json")
    if not os.path.exists(map_path):
        return None, None
    with open(map_path, "r") as f:
        m = json.load(f)
    serial_to_idx = m.get("serial_to_idx", {})
    return serial_to_idx, map_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_folder", required=True)
    parser.add_argument("--intrinsics_dir", required=True)

    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)

    parser.add_argument("--min_markers", type=int, default=2)

    parser.add_argument("--auto_save", action="store_true")
    parser.add_argument("--stable_frames", type=int, default=3)
    parser.add_argument("--cooldown_ms", type=int, default=700)

    parser.add_argument("--save_depth", dest="save_depth", action="store_true", default=True,
                        help="Save aligned depth (default: ON).")
    parser.add_argument("--no_save_depth", dest="save_depth", action="store_false",
                        help="Disable depth saving.")
    parser.add_argument("--depth_burst_n", type=int, default=10,
                        help="저장 시점 직후 N장 depth burst median으로 노이즈 감소 (default: 10). "
                             "1로 두면 단일 프레임 저장.")
    parser.add_argument("--depth_burst_max_wait_ms", type=int, default=1500,
                        help="N장을 모으는 데 허용하는 최대 시간(ms).")
    parser.add_argument("--no_align_depth_to_color", action="store_true")
    parser.add_argument("--camera_frame_timeout_ms", type=int, default=2000)
    parser.add_argument("--log_cam_timeouts", action="store_true")
    parser.add_argument("--log_cam_errors", action="store_true")
    parser.add_argument("--log_cam_stats_sec", type=float, default=0.0)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--preview_scale", type=float, default=0.5,
                        help="미리보기 창 축소 배율 (0<scale<=1). 2x2 그리드 합성 후 적용. "
                             "예: 720p 4패널(2560x1440)에 0.5 → 1280x720. 창은 드래그로도 조절 가능.")

    parser.add_argument("--tcp_poses_file", default=None,
                        help='저장 직전에 읽을 JSON. {"<event_id>": [[..4x4..]]} 형식 '
                             '(robot base -> EE, meters). 있으면 meta.json 캡처 레코드에 '
                             'tcp_base_ee_4x4로 임베드. 핸드-아이 캘리브용. '
                             '(robot_ip 와 동시 사용 시 socket 응답이 우선)')

    parser.add_argument("--robot_ip", default=None,
                        help="robot_pose_server.py 가 떠 있는 로봇 IP. "
                             "지정하면 SPACE 저장 직전 소켓으로 joint/tcp/cube_center 받아 "
                             "meta.json 캡처 레코드에 임베드. 핸드-아이 캘리브용.")
    parser.add_argument("--robot_port", type=int, default=12348)
    parser.add_argument("--robot_timeout_sec", type=float, default=2.0)

    args = parser.parse_args()

    root = ensure_dir(args.root_folder)
    intr_dir = ensure_dir(args.intrinsics_dir)

    devs = RealSenseCamera.list_devices()
    if len(devs) == 0:
        raise RuntimeError("No RealSense devices found.")

    serial_to_idx, map_path = load_device_map(intr_dir)
    if serial_to_idx is None:
        print("[WARN] device_map.json not found. Falling back to sorted(serial). (권장X)")
        # fallback (not recommended)
        serials = sorted(devs.keys())
        idx_serial_pairs = [(i, s) for i, s in enumerate(serials)]
    else:
        # map 기반으로 현재 연결된 장치만 가져오고 idx로 정렬
        idx_serial_pairs = []
        for serial in devs.keys():
            if serial not in serial_to_idx:
                print(f"[WARN] serial not in device_map.json: {serial} (Step1 다시 실행 권장)")
                continue
            idx_serial_pairs.append((int(serial_to_idx[serial]), serial))
        idx_serial_pairs.sort(key=lambda x: x[0])

    if len(idx_serial_pairs) == 0:
        raise RuntimeError(
            "No usable cameras found after applying device_map.json. "
            "Run Step1_dump_intrinsics.py again or reconnect mapped devices."
        )

    print("[INFO] devices (cam_idx fixed):")
    for idx, s in idx_serial_pairs:
        print(f"  cam{idx}: {s}  ({devs.get(s,'?')})")

    if args.save_depth:
        print(
            "[INFO] depth align_to_color:",
            "OFF" if args.no_align_depth_to_color else "ON",
        )
        if args.depth_burst_n > 1:
            print(f"[INFO] depth temporal median: {args.depth_burst_n} frames per save "
                  f"(max wait {args.depth_burst_max_wait_ms}ms)")
        else:
            print("[INFO] depth temporal median: OFF (single frame)")

    cams = {}
    for ci, serial in idx_serial_pairs:
        cam = RealSenseCamera(
            serial=serial,
            width=args.width,
            height=args.height,
            fps=args.fps,
            use_color=True,
            use_depth=args.save_depth,
            align_depth_to_color=(not args.no_align_depth_to_color),
            warmup_frames=10,
            frame_timeout_ms=args.camera_frame_timeout_ms,
            log_timeouts=args.log_cam_timeouts,
            log_errors=args.log_cam_errors,
        )
        cam.start()
        cams[ci] = cam

    if len(cams) == 0:
        raise RuntimeError("No cameras started successfully.")

    # intrinsics 존재 체크
    for ci in cams:
        intr_path = os.path.join(intr_dir, f"cam{ci}.npz")
        if not os.path.exists(intr_path):
            print(f"[WARN] intrinsics not found: {intr_path} (Step1_dump_intrinsics.py 필요)")

    # 로봇 pose 소켓 연결 (옵션). 실패해도 캡처는 진행.
    robot_sock: Optional[socket.socket] = None
    if args.robot_ip:
        try:
            robot_sock = robot_connect(args.robot_ip, args.robot_port,
                                       timeout_sec=args.robot_timeout_sec)
            print(f"[INFO] connected to robot pose server at "
                  f"{args.robot_ip}:{args.robot_port}")
        except Exception as e:
            print(f"[WARN] could not connect to robot pose server "
                  f"{args.robot_ip}:{args.robot_port}: {e} (capture will continue without robot pose)")
            robot_sock = None

    cfg = CubeConfig()
    cube = AprilTagCubeTarget(cfg)

    meta = {"root_folder": os.path.abspath(root), "captures": []}
    meta_path = os.path.join(root, "meta.json")

    for ci in cams:
        ensure_dir(os.path.join(root, f"cam{ci}"))

    event_id = 0
    stable_cnt = {ci: 0 for ci in cams}
    last_save_t = 0.0
    last_stats_log_t = time.monotonic()
    prev_cam_stats = {ci: cam.get_stats() for ci, cam in cams.items()}

    print("\nControls:")
    print("  SPACE : save once (all cams must satisfy min_markers & stable_frames)")
    print("  ESC/q : quit\n")

    if args.show:
        cv2.namedWindow("multi_cam", cv2.WINDOW_NORMAL)  # 드래그로 창 크기 조절 가능

    try:
        while True:
            frames: Dict[int, dict] = {}
            all_ok = True

            for ci, cam in cams.items():
                color, depth, ts_ms = cam.get_latest()
                if color is None:
                    all_ok = False
                    continue

                corners, ids = cube.detect(color)
                ok = (ids is not None) and (len(ids) >= args.min_markers)

                if ok:
                    stable_cnt[ci] += 1
                else:
                    stable_cnt[ci] = 0
                    all_ok = False

                frames[ci] = {
                    "color": color,
                    "depth": depth,
                    "ts_ms": ts_ms,
                    "ok": ok,
                    "ids": ([] if ids is None else [int(x) for x in ids]),
                    "corners": corners,
                    "ids_np": ids,
                }

            if args.show:
                panels = []
                for ci in sorted(cams.keys()):
                    if ci in frames and frames[ci]["color"] is not None:
                        img = frames[ci]["color"].copy()
                        ids_np = frames[ci]["ids_np"]
                        corners = frames[ci]["corners"]
                        if ids_np is not None:
                            try:
                                cv2.aruco.drawDetectedMarkers(img, corners, ids_np)
                            except Exception:
                                pass
                        txt = (f"cam{ci} ok={frames[ci]['ok']} "
                               f"stable={stable_cnt[ci]} ids={frames[ci]['ids']}")
                        color_ok = (0, 255, 0) if frames[ci]['ok'] else (0, 0, 255)
                        cv2.putText(img, txt, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                        cv2.putText(img, txt, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_ok, 1)
                    else:
                        img = np.zeros((args.height, args.width, 3), dtype=np.uint8)
                        cv2.putText(img, f"cam{ci} NO FRAME", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    panels.append(img)
                if panels:
                    h0, w0 = panels[0].shape[:2]
                    for i in range(len(panels)):
                        if panels[i].shape[:2] != (h0, w0):
                            panels[i] = cv2.resize(panels[i], (w0, h0))
                    # 2x2 그리드: 위 2개, 아래 2개 (cell이 부족하면 검은 placeholder 채움)
                    while len(panels) < 4:
                        panels.append(np.zeros((h0, w0, 3), dtype=np.uint8))
                    top    = np.hstack(panels[0:2])
                    bottom = np.hstack(panels[2:4])
                    combined = np.vstack([top, bottom])
                    s = args.preview_scale
                    if 0.0 < s < 1.0:
                        combined = cv2.resize(combined, None, fx=s, fy=s,
                                              interpolation=cv2.INTER_AREA)
                    cv2.imshow("multi_cam", combined)

            if args.log_cam_stats_sec > 0.0:
                now_mono = time.monotonic()
                dt = now_mono - last_stats_log_t
                if dt >= args.log_cam_stats_sec:
                    print(f"[CAM_STATS] dt={dt:.2f}s")
                    for ci in sorted(cams.keys()):
                        s = cams[ci].get_stats()
                        p = prev_cam_stats.get(ci, {})
                        d_frames = int(s["frames_received"]) - int(p.get("frames_received", 0))
                        d_timeouts = int(s["wait_timeouts"]) - int(p.get("wait_timeouts", 0))
                        d_errors = int(s["loop_errors"]) - int(p.get("loop_errors", 0))
                        d_stale = int(s["stale_frames"]) - int(p.get("stale_frames", 0))
                        fps_est = (d_frames / dt) if dt > 0.0 else 0.0
                        print(
                            f"  cam{ci} fps~{fps_est:.1f} (+frames={d_frames}, +timeouts={d_timeouts}, "
                            f"+errors={d_errors}, +stale={d_stale})"
                        )
                        prev_cam_stats[ci] = s
                    last_stats_log_t = now_mono

            key = cv2.waitKey(1) & 0xFF
            now_ms = time.time() * 1000.0

            manual_trigger = (key == 32)  # SPACE
            quit_trigger = (key == 27) or (key == ord("q"))
            if quit_trigger:
                break

            if args.auto_save:
                if all_ok and all(stable_cnt[ci] >= args.stable_frames for ci in cams) and (now_ms - last_save_t) >= args.cooldown_ms:
                    manual_trigger = True

            if manual_trigger:
                if not (all_ok and all(stable_cnt[ci] >= args.stable_frames for ci in cams)):
                    print("[INFO] save blocked: not all cams stable/ok")
                    continue

                cap_rec = {"event_id": int(event_id), "cams": {}}
                fid = int(event_id)

                if args.tcp_poses_file and os.path.exists(args.tcp_poses_file):
                    try:
                        with open(args.tcp_poses_file, "r") as _ftcp:
                            _tcp_map = json.load(_ftcp)
                        _tcp = _tcp_map.get(str(event_id))
                        if _tcp is None:
                            _tcp = _tcp_map.get(int(event_id)) if isinstance(_tcp_map, dict) else None
                        if _tcp is not None:
                            cap_rec["tcp_base_ee_4x4"] = _tcp
                        else:
                            print(f"[WARN] tcp_poses_file에 event_id={event_id} 항목 없음 (handeye 페어 생성 시 이 프레임은 skip)")
                    except Exception as _e:
                        print(f"[WARN] tcp_poses 읽기 실패: {_e}")

                # 로봇 소켓 (robot_pose_server) 응답이 우선 — tcp_poses_file 보다 신뢰성 높음.
                if robot_sock is not None:
                    pose_resp = robot_query_pose(robot_sock,
                                                 timeout_sec=args.robot_timeout_sec)
                    if pose_resp is not None:
                        joint_6dof = pose_resp.get("joint_6dof")
                        tcp_6dof = pose_resp.get("tcp_6dof")
                        cube_6dof = pose_resp.get("cube_center_6dof")
                        cap_rec["robot"] = {
                            "joint_6dof_deg": joint_6dof,
                            "tcp_6dof_mm_deg": tcp_6dof,
                            "cube_center_6dof_mm_deg": cube_6dof,
                            "gripper_state": pose_resp.get("gripper_state"),
                            "tool_gripper_z_mm": pose_resp.get("tool_gripper_z_mm"),
                            "tool_cube_center_z_mm": pose_resp.get("tool_cube_center_z_mm"),
                            "rotation_convention": "ZYX_intrinsic (Rz @ Ry @ Rx)",
                        }
                        if tcp_6dof is not None:
                            cap_rec["tcp_base_ee_4x4"] = pose6_mm_deg_to_T(tcp_6dof).tolist()
                        if cube_6dof is not None:
                            cap_rec["tcp_base_cube_4x4"] = pose6_mm_deg_to_T(cube_6dof).tolist()
                        print(f"[ROBOT] event_id={event_id}: "
                              f"joint d1={joint_6dof[0]:.2f}.. "
                              f"tcp_z={tcp_6dof[2]:.1f}mm cube_z={cube_6dof[2]:.1f}mm "
                              f"grip={pose_resp.get('gripper_state')}")
                    else:
                        print(f"[WARN] event_id={event_id}: robot pose query 실패 "
                              "(이번 프레임은 robot pose 없이 저장됨)")

                for ci in sorted(frames.keys()):
                    fr = frames[ci]
                    rgb_rel = f"cam{ci}/rgb_{fid:05d}.jpg"
                    rgb_abs = os.path.join(root, rgb_rel)
                    cv2.imwrite(rgb_abs, fr["color"])

                    depth_rel = None
                    if args.save_depth and (fr["depth"] is not None):
                        depth_rel = f"cam{ci}/depth_{fid:05d}.png"
                        depth_abs = os.path.join(root, depth_rel)
                        if args.depth_burst_n > 1:
                            burst = capture_depth_burst(
                                cams[ci],
                                args.depth_burst_n,
                                args.depth_burst_max_wait_ms,
                            )
                            depth_to_save = median_depth_ignore_zero(burst)
                            if depth_to_save is None:
                                depth_to_save = fr["depth"]
                            else:
                                got = int(burst.shape[0])
                                if got < args.depth_burst_n:
                                    print(f"[WARN] cam{ci}: burst {got}/{args.depth_burst_n} "
                                          f"frames in {args.depth_burst_max_wait_ms}ms")
                        else:
                            depth_to_save = fr["depth"]
                        cv2.imwrite(depth_abs, depth_to_save)

                    cap_rec["cams"][str(ci)] = {
                        "saved": True,
                        "ts_ms": fr["ts_ms"],
                        "rgb_path": rgb_rel,
                        "depth_path": depth_rel,
                        "ids": fr["ids"],
                    }

                meta["captures"].append(cap_rec)
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)

                n_caps = len(meta["captures"])
                print(f"[SAVE] event_id={event_id} -> meta.json updated ({n_caps} captures)")

                # 서버 터미널에도 저장 알림 (사용자가 확인할 수 있게 prominent banner)
                if robot_sock is not None:
                    try:
                        # representative rgb path (첫 cam)
                        first_ci = sorted(frames.keys())[0] if frames else None
                        rgb_hint = (cap_rec["cams"].get(str(first_ci), {}).get("rgb_path", "")
                                    if first_ci is not None else "")
                        notify = {
                            "command": "notify_saved",
                            "event_id": int(event_id),
                            "n_captures": int(n_caps),
                            "rgb_path": rgb_hint,
                        }
                        robot_sock.settimeout(args.robot_timeout_sec)
                        robot_sock.sendall((json.dumps(notify) + "\n").encode("utf-8"))
                        # 응답 비우기 (request-reply 프로토콜 동기 유지)
                        buf = b""
                        while b"\n" not in buf:
                            chunk = robot_sock.recv(65536)
                            if not chunk:
                                break
                            buf += chunk
                    except Exception as e:
                        print(f"[WARN] notify_saved 전송 실패: {e}")

                event_id += 1
                last_save_t = now_ms

    finally:
        for cam in cams.values():
            cam.stop()
        cv2.destroyAllWindows()
        if robot_sock is not None:
            try:
                robot_sock.sendall((json.dumps({"command": "quit"}) + "\n").encode("utf-8"))
            except Exception:
                pass
            try:
                robot_sock.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
