#!/usr/bin/env python3
"""
Calib_Step2c_replay_joint_sequence.py

parsed_poses.json (teach pendant 로그에서 파싱된 joint pose 시퀀스) 를
로봇에 socket gotoj 로 자동 재생하여 hand-eye 캡처를 무인화.

워크플로 (3 터미널):
  - 로봇PC (Terminal 1):  python2 robot_pose_server.py
  - PC      (Terminal 2):  python Calib_Step2_capture_multi_cam.py \\
                             --root_folder ./data/handeye_session_02 \\
                             --intrinsics_dir ./intrinsics \\
                             --robot_ip 192.168.0.23 --robot_port 12348 \\
                             --min_markers 2 --show
  - PC      (Terminal 3):  본 stepper (아래)

[실제 사용 명령어]
python Calib_Step2ee_replay_joint_sequence.py \\
  --poses ./data/handeye_session_01/parsed_poses.json \\
  --robot_ip 192.168.0.23 --robot_port 12348 \\
  --filter_min_deg 10.0 \\
  --start_idx 3 \\
  --manual_step

manual_step:
  각 pose 마다 1) robot 이동 → 2) settle_sec 대기 →
  3) "Terminal 2 에서 SPACE 누른 뒤 ENTER" 프롬프트 → 4) 다음 pose.

auto-save 모드를 원하면 --manual_step 빼고 Calib_Step2 에 --auto_save 추가.
"""
import argparse
import json
import os
import socket
import sys
import time
from typing import List, Optional


def load_poses(path: str) -> List[dict]:
    with open(path, "r") as f:
        return json.load(f)


def filter_diverse_poses(poses: List[dict], min_deg: float) -> List[dict]:
    """
    이전 채택 pose 와의 joint L_inf 거리가 min_deg 이상인 것만 유지.
    fine-tune (p ry,5 → p ry,4 → ...) 시리즈가 하나로 압축됨.
    """
    if not poses:
        return []
    kept = [poses[0]]
    for p in poses[1:]:
        prev = kept[-1]["joints_deg"]
        cur = p["joints_deg"]
        max_diff = max(abs(c - r) for c, r in zip(cur, prev))
        if max_diff >= min_deg:
            kept.append(p)
    return kept


def robot_connect(ip: str, port: int, timeout_sec: float):
    sk = socket.create_connection((ip, port), timeout=timeout_sec)
    sk.settimeout(timeout_sec)
    return sk


def send_json(sk: socket.socket, obj: dict, timeout_sec: float) -> Optional[dict]:
    """Send command, read newline-delimited JSON reply."""
    sk.settimeout(timeout_sec)
    sk.sendall((json.dumps(obj) + "\n").encode("utf-8"))
    buf = b""
    while b"\n" not in buf:
        chunk = sk.recv(65536)
        if not chunk:
            return None
        buf += chunk
    line = buf.split(b"\n", 1)[0]
    return json.loads(line.decode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", required=True,
                    help="parsed_poses.json 경로 (Calib_Step2b parse 결과)")
    ap.add_argument("--robot_ip", required=True)
    ap.add_argument("--robot_port", type=int, default=12348)
    ap.add_argument("--robot_timeout_sec", type=float, default=30.0,
                    help="이동 명령 타임아웃 (큰 자세 변환 고려해 30초 default)")

    ap.add_argument("--filter_min_deg", type=float, default=3.0,
                    help="이전 pose 와 joint L_inf 거리 임계 (deg). 0=필터 off")

    ap.add_argument("--settle_sec", type=float, default=1.5,
                    help="이동 완료 후 카메라 안정 대기 (Calib_Step2 stable_frames 충족용)")
    ap.add_argument("--cooldown_sec", type=float, default=1.0,
                    help="auto_save 후 다음 pose 까지 대기 (cooldown_ms 보다 길게)")

    ap.add_argument("--start_idx", type=int, default=0,
                    help="중단된 경우 이 idx 부터 재개")
    ap.add_argument("--end_idx", type=int, default=None,
                    help="여기 idx 까지만 (exclusive). default=끝까지")
    ap.add_argument("--manual_step", action="store_true",
                    help="자동 진행 대신 매 pose 마다 ENTER 대기 (auto_save 끄고 SPACE 캡처 모드용)")
    ap.add_argument("--dry_run", action="store_true",
                    help="실제 로봇 이동 없이 pose 목록만 출력")

    args = ap.parse_args()

    poses = load_poses(args.poses)
    print(f"[INFO] parsed_poses 로드: {len(poses)}개")

    if args.filter_min_deg > 0:
        kept = filter_diverse_poses(poses, args.filter_min_deg)
        print(f"[INFO] diversity filter (Δjoint ≥ {args.filter_min_deg}°): "
              f"{len(poses)} → {len(kept)}개 유지")
        poses = kept

    end_idx = args.end_idx if args.end_idx is not None else len(poses)
    todo = poses[args.start_idx:end_idx]
    print(f"[INFO] 실행 범위: [{args.start_idx}, {end_idx}) → {len(todo)}개")

    if args.dry_run:
        for i, p in enumerate(todo):
            j = p["joints_deg"]
            print(f"  [{i + args.start_idx:>3}] cmd={p['cmd']:<18} "
                  f"j=[{j[0]:>7.2f},{j[1]:>7.2f},{j[2]:>7.2f},"
                  f"{j[3]:>7.2f},{j[4]:>7.2f},{j[5]:>7.2f}]")
        return

    # 연결
    print(f"[INFO] connecting robot_pose_server at {args.robot_ip}:{args.robot_port}...")
    try:
        sk = robot_connect(args.robot_ip, args.robot_port, args.robot_timeout_sec)
    except Exception as e:
        print(f"[ERROR] connect failed: {e}")
        sys.exit(1)
    print("[INFO] connected.")

    # ping
    try:
        resp = send_json(sk, {"command": "ping"}, args.robot_timeout_sec)
        if resp is None or resp.get("status") != "ok":
            print(f"[ERROR] ping failed: {resp}")
            sys.exit(1)
    except Exception as e:
        print(f"[ERROR] ping exception: {e}")
        sys.exit(1)

    print("[INFO] server reachable. starting replay.\n")
    t_start = time.time()

    n_ok = 0
    n_fail = 0
    try:
        for i, p in enumerate(todo):
            abs_idx = args.start_idx + i
            joints = list(p["joints_deg"])
            cmd = p.get("cmd", "")

            print(f"[{abs_idx + 1}/{end_idx}] cmd_origin='{cmd}'  "
                  f"j=[{joints[0]:.2f},{joints[1]:.2f},{joints[2]:.2f},"
                  f"{joints[3]:.2f},{joints[4]:.2f},{joints[5]:.2f}]")

            t0 = time.time()
            try:
                resp = send_json(sk,
                                 {"command": "gotoj", "joints": joints},
                                 args.robot_timeout_sec)
            except socket.timeout:
                print(f"  [WARN] gotoj timeout ({args.robot_timeout_sec}s) — "
                      f"이동이 매우 길거나 서버 응답 없음. skip.")
                n_fail += 1
                continue
            except Exception as e:
                print(f"  [ERROR] gotoj exception: {e}")
                n_fail += 1
                # 연결 깨졌을 수도 있음 → 재연결 시도
                try:
                    sk.close()
                except Exception:
                    pass
                try:
                    sk = robot_connect(args.robot_ip, args.robot_port,
                                       args.robot_timeout_sec)
                    print("  [INFO] reconnected.")
                except Exception as re:
                    print(f"  [ERROR] reconnect failed: {re} → abort")
                    break
                continue

            dt = time.time() - t0
            if resp is None:
                print(f"  [WARN] no response ({dt:.1f}s) → skip")
                n_fail += 1
                continue
            if resp.get("status") != "ok":
                print(f"  [WARN] server returned: {resp.get('status')} "
                      f"reason={resp.get('reason')} → skip")
                n_fail += 1
                continue

            print(f"  [OK] moved in {dt:.2f}s. settling {args.settle_sec}s ...")
            time.sleep(args.settle_sec)

            if args.manual_step:
                input("  >> Calib_Step2 에서 SPACE 누른 뒤 여기서 ENTER 로 다음 진행: ")
            else:
                # auto_save 가 잡을 시간 + cooldown
                time.sleep(args.cooldown_sec)

            n_ok += 1
    finally:
        elapsed = time.time() - t_start
        print(f"\n[DONE] success={n_ok}  fail={n_fail}  elapsed={elapsed:.1f}s")
        try:
            send_json(sk, {"command": "quit"}, 1.0)
        except Exception:
            pass
        try:
            sk.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
