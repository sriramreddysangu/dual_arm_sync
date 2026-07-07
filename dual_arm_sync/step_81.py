#!/usr/bin/env python3
"""
step_81.py  —  Joint Config Input for Both Arms
=================================================
Input  : interactive (you type start + target joint configs)
Output : step81_joints.json

Reads current joint states live from Gazebo (ROS2).
If ROS2 is unavailable, uses zeros as start config.

Run:
    ros2 run dual_arm_sync step_81
"""

import json, os, sys, time
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    _ROS_OK = True
except ImportError:
    _ROS_OK = False

# ─────────────────────────────────────────────────────────────────────────────
# ROBOT CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

_PI   = np.pi
_PI_2 = np.pi / 2.0
L1, L2, L3, L4 = 0.1525, 0.6200, 0.5590, 0.1210
A = 0.0345

DH = np.array([
    [0.0,    0.0,  0.0,    L1],
    [-_PI_2, 0.0, -_PI_2,  A ],
    [0.0,    L2,   _PI_2,  0.0],
    [_PI_2,  0.0,  0.0,    L3],
    [-_PI_2, 0.0,  0.0,    0.0],
    [_PI_2,  0.0,  0.0,    L4],
], dtype=float)

POS_LIM = np.array([
    [-2*_PI,  2*_PI ], [-1.6493, 1.6493], [-2.7925, 2.7925],
    [-2*_PI,  2*_PI ], [-2*_PI,  2*_PI ], [-2*_PI,  2*_PI ],
], dtype=float)

VEL_LIM = np.array([2.094, 2.094, 3.140, 3.927, 3.927, 3.927])
NDOF    = 6

ROBOT_BASES: Dict[str, np.ndarray] = {
    'dsr01': np.array([0.0,  0.5, 0.0]),
    'dsr02': np.array([0.0, -0.5, 0.0]),
}
ROBOT_NAMES = ['dsr01', 'dsr02']

# Gazebo joint order vs DH order
# Gazebo: [j1, j2, j4, j5, j3, j6]
# DH:     [j1, j2, j3, j4, j5, j6]
GZ_TO_DH = [0, 1, 4, 2, 3, 5]   # q_dh = q_gz[GZ_TO_DH]


# ─────────────────────────────────────────────────────────────────────────────
# FK
# ─────────────────────────────────────────────────────────────────────────────

def fk_pos(q: np.ndarray, base: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    for i in range(NDOF):
        al, a, to, d = DH[i]
        th = q[i] + to
        ct, st = np.cos(th), np.sin(th)
        ca, sa = np.cos(al),  np.sin(al)
        T = T @ np.array([
            [ct,    -st,    0.,  a    ],
            [st*ca,  ct*ca, -sa, -sa*d],
            [st*sa,  ct*sa,  ca,  ca*d],
            [0.,     0.,    0.,  1.   ],
        ])
    return T[:3, 3] + base


# ─────────────────────────────────────────────────────────────────────────────
# ROS2 NODE
# ─────────────────────────────────────────────────────────────────────────────

if _ROS_OK:
    class JointReader(Node):
        def __init__(self):
            super().__init__('step_81_reader')
            self._arm: Dict[str, Dict] = {
                n: {'q': np.zeros(NDOF), 'ready': False}
                for n in ROBOT_NAMES
            }
            for name in ROBOT_NAMES:
                for topic in (f'/{name}/gz/joint_states',
                              f'/{name}/joint_states'):
                    self.create_subscription(
                        JointState, topic,
                        lambda msg, n=name: self._cb(msg, n), 10)

        def _cb(self, msg: JointState, name: str):
            if len(msg.position) < NDOF:
                return
            jmap = {n: i for i, n in enumerate(msg.name)}
            keys = [f'joint_{k}' for k in range(1, NDOF + 1)]
            if all(k in jmap for k in keys):
                q_gz = np.array([msg.position[jmap[k]] for k in keys])
            else:
                q_gz = np.array(msg.position[:NDOF])
            # Convert Gazebo joint order → DH order
            q_dh = q_gz[GZ_TO_DH]
            self._arm[name]['q']     = q_dh.astype(float)
            self._arm[name]['ready'] = True

        def both_ready(self) -> bool:
            return all(v['ready'] for v in self._arm.values())

        def joints(self, name: str) -> np.ndarray:
            return self._arm[name]['q'].copy()

        def wait(self, timeout: float = 20.0) -> bool:
            t0 = time.time()
            while rclpy.ok() and time.time() - t0 < timeout:
                rclpy.spin_once(self, timeout_sec=0.05)
                if self.both_ready():
                    return True
            return False


# ─────────────────────────────────────────────────────────────────────────────
# INPUT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _parse_joints(raw: str, use_deg: bool) -> Optional[np.ndarray]:
    try:
        vals = [float(v) for v in raw.replace(',', ' ').split()]
        if len(vals) != NDOF:
            print(f'    Need exactly {NDOF} values, got {len(vals)}')
            return None
        q = np.array(vals, dtype=float)
        if use_deg:
            q = np.radians(q)
        q = np.clip(q, POS_LIM[:, 0], POS_LIM[:, 1])
        return q
    except ValueError as e:
        print(f'    Parse error: {e}')
        return None


def _show_joints(label: str, q: np.ndarray):
    deg = [f'{np.degrees(v):7.2f}' for v in q]
    rad = [f'{v:7.4f}' for v in q]
    print(f'    {label}')
    print(f'      deg: [{", ".join(deg)}]')
    print(f'      rad: [{", ".join(rad)}]')
    ee = fk_pos(q, np.zeros(3))
    print(f'      EE (local frame): [{ee[0]:.4f}, {ee[1]:.4f}, {ee[2]:.4f}] m')


def prompt_arm(name: str, current_q: np.ndarray, use_deg: bool) -> Tuple[np.ndarray, np.ndarray]:
    ustr = 'degrees' if use_deg else 'radians'
    lims = 'J1=+-360  J2=+-94.5  J3=+-159.9  J4/5/6=+-360  (deg)' if use_deg \
           else 'J1=+-6.28  J2=+-1.65  J3=+-2.79  J4/5/6=+-6.28  (rad)'

    print(f'\n  {"─"*62}')
    print(f'  ARM: {name.upper()}    base = {ROBOT_BASES[name].tolist()}')
    print(f'  {"─"*62}')

    # Show current joints
    _show_joints('Current joints (from Gazebo):', current_q)
    print(f'    Limits: {lims}')
    print()

    # Start joints — default to current
    while True:
        raw = input(f'  [{name}] START joints J1..J6 ({ustr})'
                    f'  [Enter = use current]: ').strip()
        if raw == '':
            start_q = current_q.copy()
            _show_joints('  -> Using current as START:', start_q)
            break
        start_q = _parse_joints(raw, use_deg)
        if start_q is not None:
            _show_joints('  -> START:', start_q)
            break

    # Target joints
    while True:
        raw = input(f'  [{name}] TARGET joints J1..J6 ({ustr}): ').strip()
        target_q = _parse_joints(raw, use_deg)
        if target_q is not None:
            _show_joints('  -> TARGET:', target_q)
            break

    return start_q, target_q


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(args=None):
    print('\n' + '=' * 64)
    print('  STEP 81  —  Joint Config Input')
    print('=' * 64)
    print(f'  Arms   : {ROBOT_NAMES}')
    print(f'  dsr01  : base = {ROBOT_BASES["dsr01"].tolist()}')
    print(f'  dsr02  : base = {ROBOT_BASES["dsr02"].tolist()}')

    # ROS2 joint state read
    current_q: Dict[str, np.ndarray] = {n: np.zeros(NDOF) for n in ROBOT_NAMES}
    node = None

    if _ROS_OK:
        rclpy.init(args=args)
        node = JointReader()
        print('\n  Waiting for Gazebo joint states ...')
        if node.wait(20.0):
            for name in ROBOT_NAMES:
                current_q[name] = node.joints(name)
            print('  Both arms ready.')
        else:
            print('  Timeout — using zero start configs.')
    else:
        print('\n  No ROS2 — using zero start configs.')

    # Unit selection
    print()
    u_raw   = input('  Unit? [D]egrees / [R]adians  (Enter = D): ').strip().lower()
    use_deg = (u_raw != 'r')
    print(f'  -> {"degrees" if use_deg else "radians"}')

    # Collect configs for each arm
    arm_data: Dict = {}
    for name in ROBOT_NAMES:
        start_q, target_q = prompt_arm(name, current_q[name], use_deg)
        base = ROBOT_BASES[name]
        arm_data[name] = {
            'base'             : base.tolist(),
            'start_joints'     : start_q.tolist(),
            'target_joints'    : target_q.tolist(),
            'start_joints_deg' : np.degrees(start_q).tolist(),
            'target_joints_deg': np.degrees(target_q).tolist(),
            'start_ee_world'   : fk_pos(start_q,  base).tolist(),
            'target_ee_world'  : fk_pos(target_q, base).tolist(),
        }

    # Duration
    print()
    dur_raw  = input('  Duration [s]  (Enter = 10.0): ').strip()
    duration = float(dur_raw) if dur_raw else 10.0

    out = {'duration': duration, 'arm_names': ROBOT_NAMES}
    out.update(arm_data)

    with open('step81_joints.json', 'w') as fh:
        json.dump(out, fh, indent=2)

    print('\n' + '─' * 64)
    print('  Summary:')
    for name in ROBOT_NAMES:
        d = arm_data[name]
        sd = [f'{v:.1f}' for v in d['start_joints_deg']]
        td = [f'{v:.1f}' for v in d['target_joints_deg']]
        print(f'  [{name}]  start: [{", ".join(sd)}] deg')
        print(f'           target: [{", ".join(td)}] deg')
    print(f'  Duration : {duration:.2f}s')
    print(f'  Saved    : step81_joints.json')
    print(f'  Next     : ros2 run dual_arm_sync step_82')
    print('─' * 64 + '\n')

    if node is not None:
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()


if __name__ == '__main__':
    main()