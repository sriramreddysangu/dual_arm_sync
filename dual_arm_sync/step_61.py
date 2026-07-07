#!/usr/bin/env python3
"""
step_61.py  --  Dual-Arm SE(3) IK Solver
==========================================
INPUT  : interactive (user types targets) + live Gazebo joint states
OUTPUT : s61_ik.json

Round-1 selection: IK solution closest to current joint config in raw radians.
Round-2 selection: full scoring with inter-arm clearance, time cost, limits, manipulability.

Paper metric written to s61_ik.json:
  - ik_solve_time_ms  (planning time for IK stage)
  - n_solutions_found (per arm: how many valid IK solutions before dedup)
  - position_error_mm (IK residual)
"""

import json, os, sys, time
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(__file__))
from _robot import (DH, NDOF, POS_LIM, VEL_LIM, ROBOT_BASES, ARM_NAMES,
                    HOME_Q, fk, fk_world, pair_min_dist, manipulability)

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    _ROS_OK = True
except ImportError:
    _ROS_OK = False

# ── IK constants ──────────────────────────────────────────────────────────────
W_POS       = 1.0
W_ROT       = 0.15
IK_TOL_POS  = 0.010   # 10 mm
IK_TOL_ROT  = 0.050   # rad (~2.9°)
IK_UNIQ     = 0.12    # rad -- minimum distance between accepted solutions
T_REF       = 5.0     # sec -- time-score reference
W_CLEAR     = 3.0
W_TIME      = 2.0
W_LIM       = 1.0
W_MANIP     = 0.4


# ── FK helpers ────────────────────────────────────────────────────────────────

def rot_err(R_got, R_want) -> float:
    dR   = R_got.T @ R_want
    cos_ = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cos_))


# ── IK seeds ─────────────────────────────────────────────────────────────────

def _ik_seeds(current: np.ndarray, target_local: np.ndarray) -> List[np.ndarray]:
    seeds = [current.copy()]
    # Perturbations of current state
    for j in range(NDOF):
        for d in [0.3, -0.3, 0.6, -0.6]:
            s = current.copy(); s[j] += d
            s = np.clip(s, POS_LIM[:, 0], POS_LIM[:, 1])
            seeds.append(s)
    # Geometric seeds from target XYZ
    px, py, pz = target_local
    t1 = np.arctan2(py, px)
    r_xy = np.sqrt(px**2 + py**2)
    from _robot import L2, L3, A
    r2  = r_xy - A
    dist = np.sqrt(r2**2 + pz**2)
    c3  = np.clip((dist**2 - L2**2 - L3**2) / (2 * L2 * L3), -1, 1)
    for th3 in [np.arccos(c3), -np.arccos(c3)]:
        for wrist in [0, _PI_2, -_PI_2]:
            for base in [t1, t1 + _PI_2, t1 - _PI_2]:
                th2 = np.arctan2(pz, r2) - np.arctan2(L3 * np.sin(th3), L2 + L3 * np.cos(th3))
                s   = np.array([base, th2, th3, 0., wrist, 0.])
                s   = np.clip(s, POS_LIM[:, 0], POS_LIM[:, 1])
                seeds.append(s)
    return seeds

_PI   = np.pi
_PI_2 = np.pi / 2.0


# ── SE(3) IK ─────────────────────────────────────────────────────────────────

def solve_ik(target_local: np.ndarray,
             target_rot: np.ndarray,
             current: np.ndarray) -> List[np.ndarray]:
    valid = []

    def obj(q):
        p, T = fk(q)
        return W_POS * np.sum((p - target_local)**2) + \
               W_ROT * rot_err(T[:3, :3], target_rot)**2

    for seed in _ik_seeds(current, target_local):
        res = minimize(obj, seed, method='SLSQP',
                       bounds=list(zip(POS_LIM[:, 0], POS_LIM[:, 1])),
                       options={'maxiter': 800, 'ftol': 1e-10})
        if not res.success: continue
        q = np.clip(res.x, POS_LIM[:, 0], POS_LIM[:, 1])
        p, T = fk(q)
        pe = float(np.linalg.norm(p - target_local)) * 1000
        re = rot_err(T[:3, :3], target_rot)
        if pe > IK_TOL_POS * 1000 or re > IK_TOL_ROT: continue
        if all(float(np.linalg.norm(q - v)) >= IK_UNIQ for v in valid):
            valid.append(q)
    # Fallback: position-only
    if not valid:
        def pos_obj(q): return np.sum((fk(q)[0] - target_local)**2)
        for seed in _ik_seeds(current, target_local)[:15]:
            res = minimize(pos_obj, seed, method='SLSQP',
                           bounds=list(zip(POS_LIM[:, 0], POS_LIM[:, 1])),
                           options={'maxiter': 600, 'ftol': 1e-9})
            if not res.success: continue
            q = np.clip(res.x, POS_LIM[:, 0], POS_LIM[:, 1])
            pe = float(np.linalg.norm(fk(q)[0] - target_local)) * 1000
            if pe > IK_TOL_POS * 2000: continue
            if all(float(np.linalg.norm(q - v)) >= IK_UNIQ for v in valid):
                valid.append(q)
    return valid


# ── Scoring ───────────────────────────────────────────────────────────────────

def sq_vel_cost(q, start): return float(np.sum(((q - start) / VEL_LIM)**2))
def min_motion_time(q, start): return float(np.max(np.abs(q - start) / VEL_LIM))


def score(q, start, base, others) -> float:
    min_d    = min((pair_min_dist(q, base, oq, ob) for oq, ob in others), default=1.0)
    clear    = float(np.tanh(min_d / 0.25))
    svc      = sq_vel_cost(q, start)
    t_score  = float(np.exp(-svc / T_REF**2))
    mid      = (POS_LIM[:, 0] + POS_LIM[:, 1]) / 2
    rng      = POS_LIM[:, 1] - POS_LIM[:, 0]
    lim      = float(np.mean(1 - 2 * np.abs(q - mid) / rng))
    manip    = manipulability(q)
    return W_CLEAR * clear + W_TIME * t_score + W_LIM * lim + W_MANIP * manip


# ── Two-round selection ───────────────────────────────────────────────────────

def find_best_configs(arms_data: Dict) -> Optional[Dict]:
    arm_names = list(arms_data.keys())
    all_sols  = {}
    best      = {}

    for name in arm_names:
        start  = arms_data[name]['start_q']
        target = arms_data[name]['target_local']
        rot    = arms_data[name]['target_rot']
        t0     = time.time()
        sols   = solve_ik(target, rot, start)
        arms_data[name]['ik_time_ms'] = round((time.time() - t0) * 1000, 1)
        if not sols:
            print(f'  [{name}] FAIL: no IK solution found')
            return None
        all_sols[name] = sols
        arms_data[name]['n_sols'] = len(sols)
        # Round-1: closest to current config in raw radians
        best[name] = min(sols, key=lambda q, s=start: float(np.linalg.norm(q - s)))
        dist_r     = float(np.linalg.norm(best[name] - start))
        print(f'  [{name}] {len(sols)} sol(s)  '
              f'||Dq||={dist_r:.4f}rad  T≈{min_motion_time(best[name],start):.2f}s')

    # Round-2: re-score with inter-arm clearance
    for name in arm_names:
        start  = arms_data[name]['start_q']
        base   = arms_data[name]['base']
        others = [(best[n], arms_data[n]['base']) for n in arm_names if n != name]
        scores = [(score(q, start, base, others), q) for q in all_sols[name]]
        sc, best[name] = max(scores, key=lambda x: x[0])
        d_clear = min((pair_min_dist(best[name], base, oq, ob) for oq, ob in others), default=1.0)
        p, T    = fk(best[name])
        pos_err = float(np.linalg.norm(p - arms_data[name]['target_local'])) * 1000
        print(f'  [{name}] Round-2: score={sc:.4f}  '
              f'd_clear={d_clear*100:.1f}cm  pos_err={pos_err:.2f}mm')
        arms_data[name]['score']    = round(sc, 5)
        arms_data[name]['pos_err_mm'] = round(pos_err, 3)
        arms_data[name]['d_clear_m']  = round(d_clear, 4)
    return best


# ── ROS2 reader ───────────────────────────────────────────────────────────────

if _ROS_OK:
    class JointReader(Node):
        def __init__(self):
            super().__init__('step_61_reader')
            self._q = {n: np.zeros(NDOF) for n in ARM_NAMES}
            self._ready = {n: False for n in ARM_NAMES}
            for name in ARM_NAMES:
                for topic in (f'/{name}/gz/joint_states', f'/{name}/joint_states'):
                    self.create_subscription(JointState, topic,
                        lambda msg, n=name: self._cb(msg, n), 10)
        def _cb(self, msg, name):
            if len(msg.position) < NDOF: return
            jmap = {n: i for i, n in enumerate(msg.name)}
            keys = [f'joint_{k}' for k in range(1, NDOF + 1)]
            q    = (np.array([msg.position[jmap[k]] for k in keys])
                    if all(k in jmap for k in keys)
                    else np.array(msg.position[:NDOF]))
            self._q[name] = q.astype(float); self._ready[name] = True
        def joints(self, n): return self._q[n].copy()
        def wait(self, t=20.):
            import time as _t; t0 = _t.time()
            while rclpy.ok() and _t.time()-t0 < t:
                rclpy.spin_once(self, timeout_sec=0.05)
                if all(self._ready.values()): return True
            return False


# ── Input helpers ─────────────────────────────────────────────────────────────

def _parse(raw, deg):
    try:
        v = [float(x) for x in raw.replace(',', ' ').split()]
        if len(v) != NDOF: print(f'    Need {NDOF} values, got {len(v)}'); return None
        q = np.array(v); q = np.radians(q) if deg else q
        return np.clip(q, POS_LIM[:, 0], POS_LIM[:, 1])
    except ValueError as e: print(f'    {e}'); return None


def _show(label, q, base):
    deg = ', '.join(f'{np.degrees(v):7.2f}' for v in q)
    ee  = fk_world(q, base)
    print(f'    {label}'); print(f'      [{deg}] deg')
    print(f'      EE world: [{ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}] m')


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args=None):
    print('\n' + '='*66)
    print('  STEP 61  --  Dual-Arm SE(3) IK Solver')
    print('='*66)

    cur_q = {n: np.zeros(NDOF) for n in ARM_NAMES}
    node  = None
    if _ROS_OK:
        rclpy.init(args=args); node = JointReader()
        print('\n  Waiting for Gazebo joint states...')
        if node.wait(20.):
            for n in ARM_NAMES: cur_q[n] = node.joints(n)
            print('  Both arms ready.')
        else: print('  Timeout -- using zero start.')

    u_raw   = input('\n  Units? [D]egrees / [R]adians  (Enter=D): ').strip().lower()
    use_deg = (u_raw != 'r')

    arms_data = {}
    for name in ARM_NAMES:
        base = ROBOT_BASES[name]
        print(f'\n  {"─"*60}')
        print(f'  ARM: {name.upper()}  base={base.tolist()}')
        _show('Current Gazebo:', cur_q[name], base)

        while True:
            raw = input(f'  [{name}] TARGET pos X Y Z (m): ').strip()
            try: tgt_xyz = [float(v) for v in raw.replace(',', ' ').split()]
            except ValueError: continue
            if len(tgt_xyz) == 3: break
        tgt_world = np.array(tgt_xyz)
        tgt_local = tgt_world - base

        raw_q = input(f'  [{name}] TARGET quaternion w x y z (Enter=identity): ').strip()
        if raw_q == '':
            R = np.eye(3)
        else:
            try:
                w, x, y, z = [float(v) for v in raw_q.replace(',', ' ').split()]
                n2 = w*w+x*x+y*y+z*z
                if n2 < 1e-9: R = np.eye(3)
                else:
                    w,x,y,z = [v/np.sqrt(n2) for v in [w,x,y,z]]
                    R = np.array([
                        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)],
                    ])
            except ValueError: R = np.eye(3)

        arms_data[name] = {
            'base': base, 'start_q': cur_q[name],
            'target_world': tgt_world.tolist(),
            'target_local': tgt_local, 'target_rot': R,
        }

    dur_raw  = input('\n  Duration [s]  (Enter=10.0): ').strip()
    duration = float(dur_raw) if dur_raw else 10.0

    print('\n  Solving IK...')
    t_ik_total = time.time()
    best = find_best_configs(arms_data)
    ik_total_ms = round((time.time() - t_ik_total) * 1000, 1)

    if best is None:
        print('\n  FAIL: IK failed for one or more arms.'); sys.exit(1)

    out = {'duration': duration, 'arm_names': ARM_NAMES,
           'ik_total_time_ms': ik_total_ms}
    for name in ARM_NAMES:
        d   = arms_data[name]
        q   = best[name]
        ee  = fk_world(q, d['base'])
        out[name] = {
            'base'            : np.array(d['base']).tolist(),
            'start_joints'    : d['start_q'].tolist(),
            'target_joints'   : q.tolist(),
            'target_joints_deg': np.degrees(q).tolist(),
            'target_world'    : d['target_world'],
            'target_ee_world' : ee.tolist(),
            'position_error_mm': d['pos_err_mm'],
            'ik_time_ms'      : d['ik_time_ms'],
            'n_solutions'     : d['n_sols'],
            'ik_score'        : d['score'],
            'clearance_m'     : d['d_clear_m'],
        }
        _show(f'  [{name}] target:', q, d['base'])

    with open('s61_ik.json', 'w') as fh: json.dump(out, fh, indent=2)
    print(f'\n  IK total time : {ik_total_ms:.0f} ms')
    print(f'  Saved         : s61_ik.json')
    print(f'  Next          : ros2 run dual_arm_sync step_62\n')

    if node:
        try: node.destroy_node()
        except: pass
        rclpy.shutdown()


if __name__ == '__main__': main()