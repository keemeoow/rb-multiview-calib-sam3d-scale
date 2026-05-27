"""
robot_pose_server.py

PC 가 SPACE 를 누를 때마다 현재 로봇의 joint / TCP / cube_center 를 회신하는 서버.

  - foreground: goto_server.py 스타일 REPL (수동 이동/그리퍼/속도)
  - background: newline-delimited JSON socket. PC 의 `get_pose` 요청에 응답
  - single-threaded: select 로 stdin + socket 을 동시에 watch -> i611 SDK 호출은
    항상 메인 스레드에서만 발생 (rb.changetool/getpos/getjnt 의 thread-safety 걱정 X)

소켓 프로토콜 (newline-delimited JSON, port 12348):
  Request : {"command": "get_pose"}
  Reply   : {
    "status": "ok",
    "joint_6dof":       [d1..d6]                (deg),
    "tcp_6dof":         [x,y,z,rz,ry,rx]        (mm/deg, gripper tip = tool 3),
    "cube_center_6dof": [x,y,z,rz,ry,rx]        (mm/deg, cube center = tool 4),
    "gripper_state": "open"|"closed"|"unknown",
    "tool_gripper_z_mm": 150.0,
    "tool_cube_center_z_mm": 137.0
  }
  Request : {"command": "ping"}        -> {"status":"ok"}
  Request : {"command": "quit"}        -> {"status":"ok"}  (close that client)

REPL 명령어 (Calib_Step2 SPACE 와 독립):
  gotoj d1..d6           : joint abs move
  gotop x,y,z[,rz,ry,rx] : TCP abs move
  p <axis>,<v> / j <axis>,<v> : relative move (axis: x|y|z|rz|ry|rx, d1..d6)
  show / speed <0-100> / go / gc / q
"""

#!/usr/bin/python
# -*- coding: utf-8 -*-

from i611_MCS import *
from teachdata import *
from i611_extend import *
from rbsys import *
from i611_common import *
from i611_io import *
from i611shm import *
import sys
import time
import socket
import select
import json

HOST = '0.0.0.0'
PORT = 12348

GRIPPER_IO_PORT = 48
GRIPPER_TIMEOUT_SEC = 5.0

# Cube / tool 형상.
#
# 그리퍼가 잡는 위치 = 큐브 "윗면" (flange 와 가까운 쪽 면)의 정중앙.
# 즉 gripper tip 에서 tool +z 방향(= flange 에서 멀어지는 방향)으로
# 큐브가 (CUBE_SIZE/2 - CUBE_GRIP_DEPTH) mm 만큼 뻗어 있고, 그 끝에 cube center.
#
#   flange ──┬── (tool 0)
#            │   tool +z (down/away from flange)
#            │   ┊
#            ▼   ┊
#         gripper tip (tool 3, z = TOOL_GRIPPER_Z)
#            │   = 큐브 윗면 중점에 닿음
#            │   ┊ ← 그리퍼가 큐브에 2mm (CUBE_GRIP_DEPTH_MM) 박힘
#            ▼   ┊
#         cube center (tool 4, z = TOOL_GRIPPER_Z + CUBE_CENTER_OFFSET_Z)
#            │   ┊
#            ▼   ┊
#         cube bottom face
#
# robot_calb.py 는 TOOL_CUBE_CENTER_Z = TOOL_GRIPPER_Z - CUBE_CENTER_OFFSET_Z 로
# 부호가 반대로 되어 있음 — 그 코드는 cube center 를 flange 쪽으로 13mm
# (= 큐브가 위로 뻗는 형상) 잡아서 hand-eye 입력으로 쓰면 26mm 어긋남.
# 이 서버는 사용자 확인된 실제 기하 (윗면 그립) 로 정정.
CUBE_SIZE_MM = 30.0
CUBE_GRIP_DEPTH_MM = 2.0
CUBE_CENTER_OFFSET_Z = CUBE_SIZE_MM / 2.0 - CUBE_GRIP_DEPTH_MM  # +13.0 mm (away from flange)
TOOL_GRIPPER_Z = 150.0
TOOL_CUBE_CENTER_Z = TOOL_GRIPPER_Z + CUBE_CENTER_OFFSET_Z      # 163.0 mm

TCP_AXIS_MAP = {'x': 'dx', 'y': 'dy', 'z': 'dz',
                'rz': 'drz', 'ry': 'dry', 'rx': 'drx'}
JOINT_AXIS_MAP = {'d1': 'dj1', 'd2': 'dj2', 'd3': 'dj3',
                  'd4': 'dj4', 'd5': 'dj5', 'd6': 'dj6'}

_RECV_BUF = {'data': b''}


# -- Socket helpers --

def send_json(conn, obj):
    try:
        msg = json.dumps(obj)
        conn.sendall((msg + '\n').encode('utf-8'))
    except socket.error as e:
        print 'Send error: {}'.format(e)


def try_recv_json(conn):
    """Non-blocking: read available bytes, return (parsed_json|None, peer_closed_bool).
    Returns parsed_json when a full line is in buffer."""
    try:
        chunk = conn.recv(65536)
    except socket.error:
        return None, False
    if not chunk:
        return None, True
    _RECV_BUF['data'] += chunk
    if b'\n' not in _RECV_BUF['data']:
        return None, False
    line, _, rest = _RECV_BUF['data'].partition(b'\n')
    _RECV_BUF['data'] = rest
    try:
        return json.loads(line.decode('utf-8').strip()), False
    except Exception as e:
        print 'Recv parse error: {}'.format(e)
        return None, False


# -- Robot helpers --

def get_tcp():
    return rb.getpos().pos2list()[:6]


def get_joints():
    return rb.getjnt().jnt2list()[:6]


def get_cube_center():
    """Switch to tool 4 (cube center), read TCP, restore tool 3."""
    rb.changetool(4)
    tcp = rb.getpos().pos2list()[:6]
    rb.changetool(3)
    return tcp


def check_gripper():
    return [din(GRIPPER_IO_PORT + i) for i in [3, 2, 1, 0]]


def gripper_state():
    bits = check_gripper()
    if bits == ['0', '1', '0', '0']:
        return 'open'
    if bits == ['0', '0', '0', '1']:
        return 'closed'
    return 'unknown'


def gripper_open():
    print 'Gripper opening...'
    dout(GRIPPER_IO_PORT, '0000')
    t0 = time.time()
    while check_gripper() != ['0', '1', '0', '0']:
        dout(GRIPPER_IO_PORT, '0100')
        if time.time() - t0 > GRIPPER_TIMEOUT_SEC:
            print '[WARN] Gripper open timeout!'
            break
        time.sleep(0.05)
    print 'Gripper opened'


def gripper_close():
    print 'Gripper closing...'
    dout(GRIPPER_IO_PORT, '0000')
    t0 = time.time()
    while check_gripper() != ['0', '0', '0', '1']:
        dout(GRIPPER_IO_PORT, '0001')
        if time.time() - t0 > GRIPPER_TIMEOUT_SEC:
            print '[WARN] Gripper close timeout!'
            break
        time.sleep(0.05)
    print 'Gripper closed'


def show_pose():
    tcp = get_tcp()
    jnt = get_joints()
    cube = get_cube_center()
    print ''
    print '  joints: [{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}]'.format(
        jnt[0], jnt[1], jnt[2], jnt[3], jnt[4], jnt[5])
    print '  tcp:    ({:.1f}, {:.1f}, {:.1f}) / ({:.1f}, {:.1f}, {:.1f})'.format(
        tcp[0], tcp[1], tcp[2], tcp[3], tcp[4], tcp[5])
    print '  cube:   ({:.1f}, {:.1f}, {:.1f}) / ({:.1f}, {:.1f}, {:.1f})'.format(
        cube[0], cube[1], cube[2], cube[3], cube[4], cube[5])
    print '  grip:   {}'.format(gripper_state())
    print ''


def move_tcp(axis, value):
    if axis not in TCP_AXIS_MAP:
        print 'Invalid axis: {}. Use x,y,z,rz,ry,rx'.format(axis)
        return
    current = Position(*rb.getpos().pos2list()[:6])
    rb.line(current.offset(**{TCP_AXIS_MAP[axis]: value}))


def move_joint(axis, value):
    if axis not in JOINT_AXIS_MAP:
        print 'Invalid axis: {}. Use d1~d6'.format(axis)
        return
    current = Joint(*rb.getjnt().jnt2list()[:6])
    rb.move(current.offset(**{JOINT_AXIS_MAP[axis]: value}))


# -- Socket request handler --

def handle_socket_message(conn, obj):
    """Return one of: 'continue', 'disconnect'."""
    if not isinstance(obj, dict):
        send_json(conn, {"status": "error", "reason": "invalid request (not a dict)"})
        return 'continue'
    cmd = obj.get('command')
    if cmd == 'ping':
        send_json(conn, {"status": "ok"})
        return 'continue'
    if cmd == 'get_pose':
        try:
            jnt = get_joints()
            tcp = get_tcp()
            cube = get_cube_center()
            send_json(conn, {
                "status": "ok",
                "joint_6dof": jnt,
                "tcp_6dof": tcp,
                "cube_center_6dof": cube,
                "gripper_state": gripper_state(),
                "tool_gripper_z_mm": TOOL_GRIPPER_Z,
                "tool_cube_center_z_mm": TOOL_CUBE_CENTER_Z,
            })
            print '[Sock] get_pose -> joint d1={:.2f}.. tcp z={:.1f} cube z={:.1f}'.format(
                jnt[0], tcp[2], cube[2])
        except Exception as e:
            send_json(conn, {"status": "error", "reason": str(e)})
        return 'continue'
    if cmd == 'quit':
        send_json(conn, {"status": "ok"})
        return 'disconnect'
    send_json(conn, {"status": "error", "reason": "unknown command: {}".format(cmd)})
    return 'continue'


# -- REPL command handler --

def handle_stdin(cmd):
    """Return False to quit server."""
    cl = cmd.lower()
    if cl == 'q':
        return False
    if cl == 'show':
        show_pose()
    elif cl.startswith('speed'):
        parts = cmd.split()
        if len(parts) >= 2:
            try:
                rb.override(int(parts[1]))
                print 'Speed: {}'.format(int(parts[1]))
            except ValueError:
                print 'Usage: speed <0-100>'
    elif cl == 'go':
        gripper_open()
    elif cl == 'gc':
        gripper_close()
    elif cl.startswith('gotoj '):
        vals = [float(v.strip()) for v in cmd[6:].strip().split(',')]
        if len(vals) != 6:
            print 'Usage: gotoj d1,d2,d3,d4,d5,d6'
        else:
            rb.move(Joint(*vals))
            show_pose()
    elif cl.startswith('gotop '):
        vals = [float(v.strip()) for v in cmd[6:].strip().split(',')]
        if len(vals) == 6:
            rb.line(Position(*vals))
            show_pose()
        elif len(vals) == 3:
            tcp = get_tcp()
            rb.line(Position(vals[0], vals[1], vals[2], tcp[3], tcp[4], tcp[5]))
            show_pose()
        else:
            print 'Usage: gotop x,y,z[,rz,ry,rx]'
    elif cl.startswith('p '):
        parts = cmd[2:].strip().split(',')
        if len(parts) == 2:
            move_tcp(parts[0].strip(), float(parts[1].strip()))
            show_pose()
        else:
            print 'Usage: p <axis>,<value>'
    elif cl.startswith('j '):
        parts = cmd[2:].strip().split(',')
        if len(parts) == 2:
            move_joint(parts[0].strip(), float(parts[1].strip()))
            show_pose()
        else:
            print 'Usage: j <axis>,<value>'
    else:
        print 'Unknown: {}'.format(cmd)
    return True


def main():
    rbs = None
    srv = None
    conn = None
    try:
        rbs = RobSys()
        rbs.open()

        global rb
        rb = i611Robot()
        Base()
        rb.open()
        IOinit(rb)

        m = MotionParam(jnt_speed=100, lin_speed=100, pose_speed=100,
                        overlap=0, acctime=0.8, dacctime=0.8)
        rb.motionparam(m)
        rb.override(50)

        rb.settool(1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        rb.settool(3, 0.0, 0.0, TOOL_GRIPPER_Z, 0.0, 0.0, 0.0)
        rb.settool(4, 0.0, 0.0, TOOL_CUBE_CENTER_Z, 0.0, 0.0, 0.0)
        rb.changetool(3)
        rb.use_mt(True)

        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((HOST, PORT))
        srv.listen(1)
        srv.setblocking(False)

        print ''
        print '=========================================='
        print '  Robot Pose Server (port {})'.format(PORT)
        print '  - tool 3 (gripper) z = {:.1f} mm'.format(TOOL_GRIPPER_Z)
        print '  - tool 4 (cube ctr) z = {:.1f} mm'.format(TOOL_CUBE_CENTER_Z)
        print '  Socket: {"command":"get_pose"} -> joint/tcp/cube_center'
        print '  REPL:   gotoj | gotop | p | j | show | speed | go | gc | q'
        print '=========================================='
        print ''
        show_pose()
        sys.stdout.write('> '); sys.stdout.flush()

        running = True
        while running:
            rlist = [sys.stdin, srv]
            if conn is not None:
                rlist.append(conn)
            try:
                ready, _, _ = select.select(rlist, [], [], 0.2)
            except select.error:
                continue

            if srv in ready:
                try:
                    new_conn, addr = srv.accept()
                    new_conn.setblocking(False)
                    print '\n[Sock] client connected: {}'.format(addr)
                    if conn is not None:
                        try: conn.close()
                        except Exception: pass
                    conn = new_conn
                    _RECV_BUF['data'] = b''
                    sys.stdout.write('> '); sys.stdout.flush()
                except socket.error as e:
                    print '[Sock] accept error: {}'.format(e)

            if conn is not None and conn in ready:
                obj, peer_closed = try_recv_json(conn)
                if peer_closed:
                    print '\n[Sock] client disconnected'
                    try: conn.close()
                    except Exception: pass
                    conn = None
                    sys.stdout.write('> '); sys.stdout.flush()
                elif obj is not None:
                    action = handle_socket_message(conn, obj)
                    if action == 'disconnect':
                        try: conn.close()
                        except Exception: pass
                        conn = None

            if sys.stdin in ready:
                try:
                    cmd = sys.stdin.readline()
                except Exception:
                    break
                if not cmd:
                    break
                cmd = cmd.strip()
                if cmd:
                    try:
                        if not handle_stdin(cmd):
                            running = False
                    except Exception as e:
                        print 'Error: {}'.format(e)
                sys.stdout.write('> '); sys.stdout.flush()

    except KeyboardInterrupt:
        print '\nInterrupted'
    except Robot_emo as e:
        print(e)
    except Robot_error as e:
        print(e)
    except Robot_fatalerror as e:
        print(e)
    except Exception as e:
        print(e)
    finally:
        try:
            if conn is not None: conn.close()
        except Exception: pass
        try:
            if srv is not None: srv.close()
        except Exception: pass
        try:
            if rb is not None:
                rb.exit(0)
                rb.close()
        except Exception: pass
        try:
            if rbs is not None: rbs.close()
        except Exception: pass


if __name__ == '__main__':
    main()
