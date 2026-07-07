#!/usr/bin/env python3
"""
step_11.py  —  IK Solver: Position + Orientation → Best Target Joint Configs
             (4-arm extension of step_1)
═══════════════════════════════════════════════════════════════════════════════
Input  : user inputs target pos+orientation per arm (interactive or ROS2)
Output : ik_solutions.json

{
  "duration": 10.0,
  "dsr01": { "start_joints": [...], "target_joints": [...], ... },
  "dsr02": { ... },
  "dsr03": { ... },
  "dsr04": { ... }
}

LAYOUT  (matches multi_arm_gazebo.launch.py)
─────────────────────────────────────────────
  dsr01 (0.0, +0.5)   dsr03 (1.0, +0.5)
  dsr02 (0.0, -0.5)   dsr04 (1.0, -0.5)

LOGIC
─────
1. Read current joint state from ROS2 (or zeros in standalone)
2. User inputs target position [x,y,z] + quaternion [w,x,y,z] per arm
3. Multi-seed SE(3) IK for each arm independently
4. Score solutions: maximize inter-arm clearance (all 6 pairs) + joint-limit
5. Pick best target config per arm
═══════════════════════════════════════════════════════════════════════════════
"""

import json, os, sys, time
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import minimize

from dual_arm_sync.dual_arm_sync.step_1 import W_TIME

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    _ROS_OK = True
except ImportError:
    _ROS_OK = False

# ─────────────────────────────────────────────────────────────────────────────
# ROBOT CONSTANTS  (Doosan M1013)
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

# 4-arm 2×2 grid, 1.5 m spacing
ROBOT_BASES: Dict[str, np.ndarray] = {
    'dsr01': np.array([0.0,  0.5, 0.0]),
    'dsr02': np.array([0.0, -0.5, 0.0]),
    'dsr03': np.array([1.0,  0.5, 0.0]),
    'dsr04': np.array([1.0, -0.5, 0.0]),
}
ROBOT_NAMES = ['dsr01', 'dsr02', 'dsr03', 'dsr04']

LINK_RADII    = np.array([0.10, 0.10, 0.08, 0.08, 0.06, 0.06])
SAFETY_MARGIN = 0.12

IK_TOL_POS  = 0.010
IK_TOL_ROT  = 0.05
IK_MAX_ITER = 800
IK_FTOL     = 1e-10
IK_UNIQ     = 0.12

W_POS = 1.0
W_ROT = 0.15

# scoring weights
W_CLEAR = 3.0
W_LIM   = 1.0
W_MANIP = 0.4

# ─────────────────────────────────────────────────────────────────────────────
# FK
# ─────────────────────────────────────────────────────────────────────────────

def fk(q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    T = np.eye(4)
    for i in range(NDOF):
        al, a, to, d = DH[i]; th = q[i] + to
        ct, st = np.cos(th), np.sin(th); ca, sa = np.cos(al), np.sin(al)
        T = T @ np.array([
            [ct,    -st,    0.,  a    ],
            [st*ca,  ct*ca, -sa, -sa*d],
            [st*sa,  ct*sa,  ca,  ca*d],
            [0.,     0.,    0.,  1.   ],
        ])
    return T[:3, 3].copy(), T

def link_origins(q: np.ndarray, base: np.ndarray) -> np.ndarray:
    T = np.eye(4); o = np.zeros((NDOF, 3))
    for i in range(NDOF):
        al, a, to, d = DH[i]; th = q[i] + to
        ct, st = np.cos(th), np.sin(th); ca, sa = np.cos(al), np.sin(al)
        T = T @ np.array([
            [ct,    -st,    0.,  a    ],
            [st*ca,  ct*ca, -sa, -sa*d],
            [st*sa,  ct*sa,  ca,  ca*d],
            [0.,     0.,    0.,  1.   ],
        ])
        o[i] = T[:3, 3] + base
    return o

def rot_err(R_got: np.ndarray, R_tgt: np.ndarray) -> float:
    R = R_tgt @ R_got.T
    return float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))

def manipulability(q: np.ndarray) -> float:
    eps = 1e-5; J = np.zeros((3, NDOF)); p0 = fk(q)[0]
    for i in range(NDOF):
        dq = q.copy(); dq[i] += eps; J[:, i] = (fk(dq)[0] - p0) / eps
    return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))

def pair_min_dist(qi: np.ndarray, bi: np.ndarray,
                  qj: np.ndarray, bj: np.ndarray) -> float:
    """Minimum link-origin distance — vectorised broadcasting."""
    oi = link_origins(qi, bi)  # (NDOF,3)
    oj = link_origins(qj, bj)  # (NDOF,3)
    diff = oi[:, np.newaxis, :] - oj[np.newaxis, :, :]  # (NDOF,NDOF,3)
    return float(np.min(np.linalg.norm(diff, axis=2)))

# ─────────────────────────────────────────────────────────────────────────────
# QUATERNION
# ─────────────────────────────────────────────────────────────────────────────

def quat_norm(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([1., 0., 0., 0.])

def quat_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_norm(q)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)  ],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)  ],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
    ])

# ─────────────────────────────────────────────────────────────────────────────
# IK SEEDS
# ─────────────────────────────────────────────────────────────────────────────

def ik_seeds(current: np.ndarray, tgt: np.ndarray) -> List[np.ndarray]:
    px, py, pz = tgt
    cl = lambda q: np.clip(q, POS_LIM[:, 0], POS_LIM[:, 1])
    # FIX: removed np.zeros(NDOF) seed — biases solver to home config;
    # always seed from robot's actual current state instead.
    seeds = [cl(current.copy())]
    for j in range(NDOF):
        for d in (0.3, -0.3, 0.6, -0.6):
            s = current.copy(); s[j] += d; seeds.append(cl(s))
    t1 = float(np.arctan2(py, px))
    rh = float(np.hypot(px, py))
    re = float(np.sqrt(max(rh**2 - A**2, 0.0)))
    h  = float(pz - L1)
    c3 = float(np.clip((re**2 + h**2 - L2**2 - L3**2) / (2*L2*L3), -1, 1))
    for sgn in (1., -1.):
        th3 = sgn * float(np.arccos(c3)); q3 = th3 - _PI_2
        th2 = float(np.arctan2(h, re)) - float(np.arctan2(L3*np.sin(th3), L2+L3*np.cos(th3)))
        q2  = th2 + _PI_2
        for q5 in (0., _PI_2, -_PI_2):
            for t in (t1, t1 + _PI_2, t1 - _PI_2):
                seeds.append(cl(np.array([t, q2, q3, 0., q5, 0.])))
    unique: List[np.ndarray] = []
    for s in seeds:
        if all(np.linalg.norm(s - u) > 0.05 for u in unique):
            unique.append(s)
    return unique

# ─────────────────────────────────────────────────────────────────────────────
# SE(3) IK
# ─────────────────────────────────────────────────────────────────────────────

def solve_ik(target_local: np.ndarray,
             target_rot  : np.ndarray,
             current     : np.ndarray) -> List[np.ndarray]:
    bds = [(POS_LIM[i, 0], POS_LIM[i, 1]) for i in range(NDOF)]
    def obj(q):
        p, T = fk(q)
        return (W_POS * float(np.sum((p - target_local)**2))
              + W_ROT * float(rot_err(T[:3,:3], target_rot)**2))
    solutions: List[np.ndarray] = []
    for seed in ik_seeds(current, target_local):
        res = minimize(obj, seed, method='SLSQP', bounds=bds,
                       options={'maxiter': IK_MAX_ITER, 'ftol': IK_FTOL})
        if not res.success: continue
        q = np.clip(res.x, POS_LIM[:, 0], POS_LIM[:, 1])
        p, T = fk(q)
        if np.linalg.norm(p - target_local) >= IK_TOL_POS: continue
        if rot_err(T[:3,:3], target_rot)    >= IK_TOL_ROT:  continue
        if all(np.linalg.norm(q - s) > IK_UNIQ for s in solutions):
            solutions.append(q)
    return solutions

# ─────────────────────────────────────────────────────────────────────────────
# SCORE & SELECT BEST CONFIG — all N arms, all C(N,2) pairs
# ─────────────────────────────────────────────────────────────────────────────

def score(q: np.ndarray,
          start_q: np.ndarray,
          base: np.ndarray,
          others: List[Tuple[np.ndarray, np.ndarray]]) -> float:
    """Higher = better.
    Priority: clearance (W_CLEAR) > motion-time sq-diff (W_TIME) > joint-limit (W_LIM) > manip.
    FIX: added start_q + W_TIME * time_score (velocity-weighted sq diff from start).
    """
    if others:
        min_d = min(pair_min_dist(q, base, oq, ob) for oq, ob in others)
    else:
        min_d = 1.0
    clear = float(np.tanh(min_d / 0.25))
    # Velocity-weighted squared difference: sum((Δq_i/VEL_LIM_i)^2)
    # exp(-cost/T_REF^2) in (0,1] — smaller cost = faster to reach = higher score
    cost       = float(np.sum(((q - start_q) / VEL_LIM) ** 2))
    time_score = float(np.exp(-cost / 25.0))  # T_REF=5s → T_REF^2=25
    mid   = (POS_LIM[:, 0] + POS_LIM[:, 1]) / 2.0
    rng   = POS_LIM[:, 1] - POS_LIM[:, 0]
    lim   = float(np.mean(1.0 - 2.0 * np.abs(q - mid) / rng))
    manip = float(np.tanh(manipulability(q) / 0.05))
    return W_CLEAR * clear + W_TIME * time_score + W_LIM * lim + W_MANIP * manip
#return W_CLEAR * clear + W_TIME * time_score + W_LIM * lim + W_MANIP * manip

def find_best_configs(arms_data: Dict) -> Optional[Dict[str, np.ndarray]]:
    """
    Round 1: solve IK for each arm independently.
    Round 2: re-score all solutions with inter-arm clearance from all other arms.
    Works for any number of arms — 6 pairs for 4 arms.
    """
    arm_names = sorted(arms_data.keys())
    all_sols:  Dict[str, List[np.ndarray]] = {}
    best:      Dict[str, np.ndarray]       = {}

    # ── Round 1: independent IK ──────────────────────────────────────────────
    for name in arm_names:
        d     = arms_data[name]
        start = d['start_q']; tloc = d['target_local']; trot = d['target_rot']
        print(f'  [{name}] IK solving ...')
        sols = solve_ik(tloc, trot, start)
        if not sols:
            print(f'  [{name}] SE(3) IK failed — trying position-only ...')
            bds = [(POS_LIM[i,0], POS_LIM[i,1]) for i in range(NDOF)]
            def pos_obj(q, tl=tloc):
                p, _ = fk(q); return float(np.sum((p - tl)**2))
            for seed in ik_seeds(start, tloc):
                res = minimize(pos_obj, seed, method='SLSQP', bounds=bds,
                               options={'maxiter': IK_MAX_ITER, 'ftol': IK_FTOL})
                q = np.clip(res.x, POS_LIM[:, 0], POS_LIM[:, 1])
                if np.linalg.norm(fk(q)[0] - tloc) < IK_TOL_POS:
                    sols.append(q); break
        if not sols:
            print(f'  [{name}] ❌  No IK solution found'); return None
        all_sols[name] = sols
        # FIX Round-1: bottleneck time = max(|Δq_i|/VEL_LIM_i), not Euclidean norm
        best[name] = min(sols, key=lambda q, s=start: float(np.max(np.abs(q - s) / VEL_LIM)))
        print(f'  [{name}] {len(sols)} solution(s)')

    # ── Round 2: re-score with clearance from all other arms ─────────────────
    for name in arm_names:
        base   = arms_data[name]['base']
        others = [(best[n], arms_data[n]['base']) for n in arm_names if n != name]
        # FIX Round-2: pass start_q so time score is computed
        start_q = arms_data[name]['start_q']
        scores  = [score(q, start_q, base, others) for q in all_sols[name]]
        best[name] = all_sols[name][int(np.argmax(scores))]
        # Report clearance to every other arm
        clears = [pair_min_dist(best[name], base, best[n], arms_data[n]['base'])
                  for n in arm_names if n != name]
        min_c = min(clears) if clears else 0.0
        print(f'  [{name}] best score={max(scores):.4f}  '
              f'min inter-arm clearance={min_c*100:.1f}cm')

    return best

# ─────────────────────────────────────────────────────────────────────────────
# ROS2 NODE
# ─────────────────────────────────────────────────────────────────────────────

if _ROS_OK:
    class Step11Node(Node):
        def __init__(self):
            super().__init__('step_11_ik')
            self._arm = {n: {'q': np.zeros(NDOF), 'ready': False}
                         for n in ROBOT_NAMES}
            for name in ROBOT_NAMES:
                for topic in (f'/{name}/gz/joint_states', f'/{name}/joint_states'):
                    self.create_subscription(
                        JointState, topic,
                        lambda msg, n=name: self._cb(msg, n), 10)

        def _cb(self, msg, name):
            if len(msg.position) < NDOF: return
            jmap = {n: i for i, n in enumerate(msg.name)}
            keys = [f'joint_{k}' for k in range(1, NDOF + 1)]
            q = (np.array([msg.position[jmap[k]] for k in keys])
                 if all(k in jmap for k in keys)
                 else np.array(msg.position[:NDOF]))
            self._arm[name]['q'] = q.astype(float)
            if not self._arm[name]['ready']:
                self._arm[name]['ready'] = True
                self.get_logger().info(f'{name} ready')

        def all_ready(self): return all(v['ready'] for v in self._arm.values())
        def joints(self, n):  return self._arm[n]['q'].copy()

        def wait(self, t=30.0):
            t0 = time.time()
            while rclpy.ok() and time.time() - t0 < t:
                rclpy.spin_once(self, timeout_sec=0.05)
                if self.all_ready(): return True
            return False

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def _prompt_pose(label: str) -> Tuple[np.ndarray, np.ndarray]:
    print(f'\n  [{label}] target pose:')
    x = float(input('    pos X (m): '))
    y = float(input('    pos Y (m): '))
    z = float(input('    pos Z (m): '))
    raw = input('    quaternion [w,x,y,z] (Enter = identity): ').strip()
    if raw:
        vals = [float(v) for v in raw.replace(',', ' ').split()]
        quat = quat_norm(np.array(vals[:4]))
    else:
        quat = np.array([1., 0., 0., 0.])
    return np.array([x, y, z]), quat

def main(args=None):
    print('\n' + '=' * 68)
    print('  STEP 11  —  IK: 4-Arm Position + Orientation → Target Configs')
    print('=' * 68)
    print(f'\n  Layout (2×2 grid, 1.5 m spacing):')
    for n, b in ROBOT_BASES.items():
        print(f'    {n}  base=({b[0]:+.2f}, {b[1]:+.2f}, {b[2]:.2f})')

    if _ROS_OK:
        rclpy.init(args=args)
        node = Step11Node()
        print('\n  Waiting for joint states from all 4 arms ...')
        if not node.wait(30.0):
            print('  ❌  Timeout'); node.destroy_node(); rclpy.shutdown(); return
        print('  ✅  All 4 arms ready')
        get_q = node.joints
        spin  = lambda: rclpy.spin_once(node, timeout_sec=0.05)
    else:
        print('  ⚠   No ROS2 — using zero start configs')
        get_q = lambda _: np.zeros(NDOF)
        spin  = lambda: None

    try:
        while True:
            print('\n' + '─' * 68)
            poses: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
            for name in ROBOT_NAMES:
                pos_w, quat = _prompt_pose(name.upper())
                poses[name] = (pos_w, quat)

            dur_raw  = input('\n  Duration [s] (Enter=10): ').strip()
            duration = float(dur_raw) if dur_raw else 10.0
            spin()

            arms_data: Dict = {}
            for name in ROBOT_NAMES:
                base         = ROBOT_BASES[name]
                start_q      = get_q(name)
                target_world = poses[name][0]
                target_quat  = poses[name][1]
                target_local = target_world - base
                target_rot   = quat_to_rot(target_quat)
                arms_data[name] = {
                    'base'        : base,
                    'start_q'     : start_q,
                    'target_world': target_world,
                    'target_local': target_local,
                    'target_rot'  : target_rot,
                    'target_quat' : target_quat,
                }

            print('\n  Solving IK for all 4 arms ...')
            best = find_best_configs(arms_data)
            if best is None:
                print('  ❌  IK failed for one or more arms — adjust targets')
                continue

            # Report all 6 pair clearances
            print('\n  Inter-arm clearances at target configs:')
            for i in range(len(ROBOT_NAMES)):
                for j in range(i+1, len(ROBOT_NAMES)):
                    ni, nj = ROBOT_NAMES[i], ROBOT_NAMES[j]
                    d = pair_min_dist(best[ni], ROBOT_BASES[ni],
                                      best[nj], ROBOT_BASES[nj])
                    icon = '✅' if d > SAFETY_MARGIN else '⚠ '
                    print(f'  {icon}  {ni}↔{nj}:  {d*100:.1f}cm')

            # Build output
            out = {'duration': duration}
            for name in ROBOT_NAMES:
                base     = arms_data[name]['base']
                start_q  = arms_data[name]['start_q']
                target_q = best[name]
                ps, _    = fk(start_q);  ps_w = (ps + base).tolist()
                pt, Tt   = fk(target_q); pt_w = (pt + base).tolist()
                pos_err  = float(np.linalg.norm(pt - arms_data[name]['target_local']) * 1000)
                rot_e    = float(np.degrees(rot_err(Tt[:3,:3], arms_data[name]['target_rot'])))
                out[name] = {
                    'robot_name'           : name,
                    'base'                 : base.tolist(),
                    'start_joints'         : start_q.tolist(),
                    'target_joints'        : target_q.tolist(),
                    'start_joints_deg'     : np.degrees(start_q).tolist(),
                    'target_joints_deg'    : np.degrees(target_q).tolist(),
                    'start_pos_world'      : ps_w,
                    'target_pos_world'     : pt_w,
                    'target_quaternion'    : arms_data[name]['target_quat'].tolist(),
                    'position_error_mm'    : round(pos_err, 3),
                    'orientation_error_deg': round(rot_e, 3),
                }
                print(f'  [{name}] pos_err={pos_err:.2f}mm  rot_err={rot_e:.2f}°')

            with open('ik_solutions.json', 'w') as fh:
                json.dump(out, fh, indent=2)
            print('\n  ✅  Saved: ik_solutions.json  →  run step_12.py')

            if input('\n  Again? (y/n): ').strip().lower() != 'y':
                break

    except KeyboardInterrupt:
        print('\n  Interrupted.')

    if _ROS_OK:
        node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()