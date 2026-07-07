#!/usr/bin/env python3
"""
step_51.py  --  Joint-Aware IK Selection (Hierarchical Pipeline)
==================================================================
Identical to step_40 in role -- finds IK targets with neighbor awareness,
biased toward minimum joint motion + clearance at target. Output is
s51_ik.json which feeds step_52.
"""
import json, os, sys, time
from typing import Dict, List
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(__file__))
from _robot5x import (DH, NDOF, POS_LIM, ROBOT_BASES, ARM_NAMES,
                       fk, fk_world, pair_min_dist, pair_collides)

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    _ROS_OK = True
except ImportError:
    _ROS_OK = False

W_POS      = 1.0
W_ROT      = 0.15
IK_TOL_POS = 0.010
IK_TOL_ROT = 0.050
IK_UNIQ    = 0.08
N_PATH_SMP = 5

W_DIST_CUR  = 1.0
W_CLEAR_TGT = 2.0
W_PATH_COLL = 1.5
W_EE_LEN    = 0.8

_PI_2 = np.pi / 2.0


def rot_err(R_got, R_want):
    dR   = R_got.T @ R_want
    return float(np.arccos(np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)))


def _ik_seeds(current, target_local):
    seeds = [current.copy()]
    for j in range(NDOF):
        for d in [0.3, -0.3, 0.6, -0.6]:
            s = current.copy(); s[j] += d
            s = np.clip(s, POS_LIM[:, 0], POS_LIM[:, 1])
            seeds.append(s)
    px, py, pz = target_local
    t1 = np.arctan2(py, px)
    r_xy = np.sqrt(px**2 + py**2)
    from _robot5x import L2, L3, A
    r2 = r_xy - A
    dist = np.sqrt(r2**2 + pz**2)
    c3 = np.clip((dist**2 - L2**2 - L3**2) / (2 * L2 * L3), -1, 1)
    for th3 in [np.arccos(c3), -np.arccos(c3)]:
        for wrist in [0, _PI_2, -_PI_2]:
            # Cover all four j1 quadrants to catch alternate IK branches
            # (some EE positions are reachable from multiple j1 directions
            # with very different arm postures -- we want all of them)
            for base in [t1, t1 + _PI_2, t1 - _PI_2, t1 + np.pi, t1 - np.pi]:
                th2 = np.arctan2(pz, r2) - np.arctan2(L3 * np.sin(th3),
                                                       L2 + L3 * np.cos(th3))
                s = np.array([base, th2, th3, 0., wrist, 0.])
                s = np.clip(s, POS_LIM[:, 0], POS_LIM[:, 1])
                seeds.append(s)
    return seeds


def solve_ik(target_local, target_rot, current):
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


def linear_path(start_q, end_q, n=N_PATH_SMP):
    fracs = np.linspace(0., 1., n)
    return np.array([start_q + f * (end_q - start_q) for f in fracs])


# Joint weights for minimum-cost IK selection. Lower-numbered joints (the
# big proximal joints) cost more to move, so we prefer solutions that keep
# them close to start. j1 weighted lower so the solver is free to choose the
# natural j1 branch.
JOINT_COST_W = np.array([1.0, 1.5, 1.5, 0.6, 0.6, 0.4])

def joint_aware_score(q_cand, start_q, base, prior):
    """
    Minimum-cost-from-start IK selection (your specification).

    Cost = weighted joint distance from the arm's CURRENT configuration.
    Collision avoidance is NOT part of this score -- that is handled entirely
    by the tier hierarchy (step_54 Kuramoto, step_55/56 home-pull). This keeps
    concerns separated: IK picks the cheapest reachable target, tiers fix any
    resulting collision.

    We also report ee_path and an informational path_coll count (against
    already-selected arms) for logging, but they do NOT affect selection.
    """
    dq = q_cand - start_q
    weighted_dist = float(np.sqrt(np.sum((JOINT_COST_W * dq) ** 2)))
    dist_cur = float(np.linalg.norm(dq))
    ee_path  = float(np.linalg.norm(fk_world(q_cand, base) - fk_world(start_q, base)))
    # Informational only -- not used in selection
    path_coll = 0
    if prior:
        for p, ob in prior:
            path = linear_path(start_q, q_cand)
            for qa in path:
                for qb in p:
                    if pair_collides(qa, base, qb, ob): path_coll += 1
    total = weighted_dist     # selection is purely minimum weighted joint cost
    return total, {'dist_cur': round(dist_cur, 4),
                    'weighted_cost': round(weighted_dist, 4),
                    'ee_path_m': round(ee_path, 4),
                    'path_coll': int(path_coll)}


def greedy_select(arms_data):
    best = {}; prior = []
    for name in arms_data:
        d = arms_data[name]
        t0 = time.time()
        sols = solve_ik(d['target_local'], d['target_rot'], d['start_q'])
        d['ik_time_ms'] = round((time.time() - t0) * 1000, 1)
        if not sols: return None
        d['n_sols'] = len(sols)
        scored = [(joint_aware_score(q, d['start_q'], d['base'], prior), q) for q in sols]
        scored.sort(key=lambda x: x[0][0])
        (sc, br), best_q = scored[0]
        best[name] = best_q; d['joint_aware_score'] = br
        p, T = fk(best_q)
        d['pos_err_mm'] = round(float(np.linalg.norm(p - d['target_local'])) * 1000, 3)
        path = linear_path(d['start_q'], best_q)
        prior.append((path, d['base']))
        print(f'  [{name}] {len(sols)} sol(s) | dist={br["dist_cur"]:.2f}rad '
              f'wcost={br["weighted_cost"]:.2f} ee_len={br["ee_path_m"]:.2f}m '
              f'path_coll={br["path_coll"]}(info) pos_err={d["pos_err_mm"]:.2f}mm')
    return best


if _ROS_OK:
    class JointReader(Node):
        def __init__(self):
            super().__init__('step_51_reader')
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
            q = (np.array([msg.position[jmap[k]] for k in keys])
                 if all(k in jmap for k in keys) else np.array(msg.position[:NDOF]))
            self._q[name] = q.astype(float); self._ready[name] = True
        def joints(self, n): return self._q[n].copy()
        def wait(self, t=20.):
            import time as _t; t0 = _t.time()
            while rclpy.ok() and _t.time() - t0 < t:
                rclpy.spin_once(self, timeout_sec=0.05)
                if all(self._ready.values()): return True
            return False


def _show(label, q, base):
    deg = ', '.join(f'{np.degrees(v):7.2f}' for v in q)
    ee  = fk_world(q, base)
    print(f'    {label}'); print(f'      [{deg}] deg')
    print(f'      EE world: [{ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}] m')


def main(args=None):
    print('\n' + '='*66)
    print('  STEP 51  --  Joint-Aware IK Selection (Hierarchical)')
    print('='*66)

    cur_q = {n: np.zeros(NDOF) for n in ARM_NAMES}
    node = None
    if _ROS_OK:
        rclpy.init(args=args); node = JointReader()
        print('\n  Waiting for Gazebo joint states...')
        if node.wait(20.):
            for n in ARM_NAMES: cur_q[n] = node.joints(n)
            print('  Both arms ready.')
        else: print('  Timeout -- using zero start.')

    arms_data = {}
    for name in ARM_NAMES:
        base = ROBOT_BASES[name]
        print(f'\n  {"─"*60}\n  ARM: {name.upper()}  base={base.tolist()}')
        _show('Current Gazebo:', cur_q[name], base)
        while True:
            raw = input(f'  [{name}] TARGET pos X Y Z (m): ').strip()
            try: tgt_xyz = [float(v) for v in raw.replace(',', ' ').split()]
            except ValueError: continue
            if len(tgt_xyz) == 3: break
        tgt_world = np.array(tgt_xyz)
        tgt_local = tgt_world - base
        raw_q = input(f'  [{name}] TARGET quaternion w x y z (Enter=identity): ').strip()
        if raw_q == '': R = np.eye(3)
        else:
            try:
                w, x, y, z = [float(v) for v in raw_q.replace(',', ' ').split()]
                n2 = w*w + x*x + y*y + z*z
                if n2 < 1e-9: R = np.eye(3)
                else:
                    w, x, y, z = [v/np.sqrt(n2) for v in [w,x,y,z]]
                    R = np.array([
                        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)],
                    ])
            except ValueError: R = np.eye(3)
        arms_data[name] = {'base': base, 'start_q': cur_q[name],
                            'target_world': tgt_world.tolist(),
                            'target_local': tgt_local, 'target_rot': R}

    dur_raw  = input('\n  Duration [s]  (Enter=10.0): ').strip()
    duration = float(dur_raw) if dur_raw else 10.0

    print('\n  Solving IK with greedy joint-aware selection...')
    t0 = time.time()
    best = greedy_select(arms_data)
    ik_total = round((time.time() - t0) * 1000, 1)
    if best is None: print('\n  FAIL: no IK solution'); sys.exit(1)

    out = {'duration': duration, 'arm_names': ARM_NAMES,
            'ik_total_time_ms': ik_total, 'pipeline': 'hierarchical_5x'}
    for name in ARM_NAMES:
        d = arms_data[name]; q = best[name]
        out[name] = {
            'base'             : np.array(d['base']).tolist(),
            'start_joints'     : d['start_q'].tolist(),
            'target_joints'    : q.tolist(),
            'target_joints_deg': np.degrees(q).tolist(),
            'target_world'     : d['target_world'],
            'target_ee_world'  : fk_world(q, d['base']).tolist(),
            'position_error_mm': d['pos_err_mm'],
            'ik_time_ms'       : d['ik_time_ms'],
            'n_solutions'      : d['n_sols'],
            'joint_aware_score': d['joint_aware_score'],
        }
        _show(f'  [{name}] target:', q, d['base'])

    with open('s51_ik.json', 'w') as fh: json.dump(out, fh, indent=2)
    print(f'\n  IK total time : {ik_total:.0f} ms')
    print(f'  Saved : s51_ik.json')
    print(f'  Next  : ros2 run dual_arm_sync step_52\n')

    if node:
        try: node.destroy_node()
        except: pass
        rclpy.shutdown()


if __name__ == '__main__': main()