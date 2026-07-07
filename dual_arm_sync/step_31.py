#!/usr/bin/env python3
"""
step_31.py  --  Joint-Aware IK Selection (Greedy Pairwise)   [DUAL ARM]
============================================================
INPUT  : interactive (user types targets) + live Gazebo joint states
OUTPUT : s31_ik.json

Dual-arm version: imports _robot2x. It loops over ARM_NAMES (two arms here);
bases dsr01=(0,+0.5,0), dsr02=(0,-0.5,0).

MODIFIED: each arm accepts EITHER
  * 3 numbers  X Y Z (m)        -> position target, runs IK (as before)
  * 6 numbers  j1..j6 (deg)     -> joint configuration, used AS-IS (no IK)
so you can drive an exact configuration from the present position (e.g. to
demonstrate a collide-then-Kuramoto-skip trajectory in Gazebo).
"""

import json, os, sys, time
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(__file__))
from _robot2x import (DH, NDOF, POS_LIM, VEL_LIM, ROBOT_BASES, ARM_NAMES,
                       fk, fk_world, pair_min_dist, pair_collides,
                       _arm_caps, caps_collide, SAFETY_MARGIN)

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    _ROS_OK = True
except ImportError:
    _ROS_OK = False

# -- IK constants --------------------------------------------------------------
W_POS       = 1.0
W_ROT       = 0.15
IK_TOL_POS  = 0.010
IK_TOL_ROT  = 0.050
IK_UNIQ     = 0.12
IK_KEEP     = 40
N_PATH_SMP  = 5

JOINT_COST_W = np.array([1.0, 1.5, 1.5, 0.6, 0.6, 0.4])
PATH_COLL_W  = 0.10
CLEAR_W      = 0.05
ENDPOINT_COLL_W = 1000.0

_PI_2 = np.pi / 2.0


def rot_err(R_got, R_want):
    dR   = R_got.T @ R_want
    cos_ = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cos_))


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
    from _robot2x import L2, L3, A
    r2 = r_xy - A
    dist = np.sqrt(r2**2 + pz**2)
    c3 = np.clip((dist**2 - L2**2 - L3**2) / (2 * L2 * L3), -1, 1)
    for th3 in [np.arccos(c3), -np.arccos(c3)]:
        for wrist in [0, _PI_2, -_PI_2]:
            for base in [t1, t1 + _PI_2, t1 - _PI_2, t1 + np.pi, t1 - np.pi]:
                th2 = np.arctan2(pz, r2) - np.arctan2(L3 * np.sin(th3),
                                                      L2 + L3 * np.cos(th3))
                s   = np.array([base, th2, th3, 0., wrist, 0.])
                s   = np.clip(s, POS_LIM[:, 0], POS_LIM[:, 1])
                seeds.append(s)
    return seeds


def solve_ik(target_local, target_rot, current, constrain_orient=False):
    use_ori = bool(constrain_orient and target_rot is not None)
    valid = []
    def obj(q):
        p, T = fk(q)
        e = np.sum((p - target_local) ** 2)
        if use_ori:
            e += W_ROT * rot_err(T[:3, :3], target_rot) ** 2
        return e
    for seed in _ik_seeds(current, target_local):
        res = minimize(obj, seed, method='SLSQP',
                        bounds=list(zip(POS_LIM[:, 0], POS_LIM[:, 1])),
                        options={'maxiter': 400, 'ftol': 1e-9})
        if not res.success: continue
        q = np.clip(res.x, POS_LIM[:, 0], POS_LIM[:, 1])
        p, T = fk(q)
        if float(np.linalg.norm(p - target_local)) > IK_TOL_POS: continue
        if use_ori and float(rot_err(T[:3, :3], target_rot)) > IK_TOL_ROT: continue
        if all(float(np.linalg.norm(q - v)) >= IK_UNIQ for v in valid):
            valid.append(q)
    valid.sort(key=lambda q: (round(float(np.max(np.abs(nearest_wrap(q, current) - current) / VEL_LIM)), 3),
                              float(np.sum(JOINT_COST_W * np.abs(nearest_wrap(q, current) - current)))))
    return valid[:IK_KEEP]


def linear_interp_path(start_q, end_q, n_samples=N_PATH_SMP):
    fracs = np.linspace(0., 1., n_samples)
    return np.array([start_q + f * (end_q - start_q) for f in fracs])


def path_collision_count(start_q_A, end_q_A, base_A, path_other, base_other):
    path_A = linear_interp_path(start_q_A, end_q_A, N_PATH_SMP)
    coll = 0
    for q_a in path_A:
        for q_b in path_other:
            if pair_collides(q_a, base_A, q_b, base_other):
                coll += 1
    return coll


def nearest_wrap(q, start):
    qw = q.copy()
    for j in range(NDOF):
        cands = [q[j] + 2 * np.pi * k for k in (-2, -1, 0, 1, 2)]
        cands = [c for c in cands if POS_LIM[j, 0] <= c <= POS_LIM[j, 1]]
        if cands:
            qw[j] = min(cands, key=lambda c: abs(c - start[j]))
    return qw


def op_time(qw, start_q):
    return float(np.max(np.abs(qw - start_q) / VEL_LIM))


def joint_aware_score(q_candidate, start_q, base, prior_paths_with_bases=None):
    qw = nearest_wrap(q_candidate, start_q)
    travel = float(np.sum(JOINT_COST_W * np.abs(qw - start_q)))
    t_op   = op_time(qw, start_q)
    key = (round(t_op, 3), round(travel, 3))
    max_exc = float(np.max(np.abs(np.degrees(qw - start_q))))
    return key, qw, {
        'op_time_s': round(t_op, 3),
        'weighted_travel_rad': round(travel, 4),
        'max_joint_excursion_deg': round(max_exc, 1),
    }


def repair_endpoint_collisions(best, arms_data, restarts=8, iters=400, seed=0):
    names = list(best.keys())
    bases = {n: arms_data[n]["base"] for n in names}
    starts = {n: arms_data[n]["start_q"] for n in names}
    cand = {n: [nearest_wrap(q, starts[n]) for q in arms_data[n]["_sols"]] for n in names}
    travel = {n: np.array([float(np.sum(JOINT_COST_W * np.abs(q - starts[n]))) for q in cand[n]])
              for n in names}
    optime = {n: np.array([float(np.max(np.abs(q - starts[n]) / VEL_LIM)) for q in cand[n]])
              for n in names}
    caps = {n: [_arm_caps(q, bases[n]) for q in cand[n]] for n in names}
    tab = {}
    for i, ni in enumerate(names):
        for nj in names[i + 1:]:
            M = np.zeros((len(cand[ni]), len(cand[nj])), bool)
            for a in range(len(cand[ni])):
                for b in range(len(cand[nj])):
                    M[a, b] = caps_collide(caps[ni][a], caps[nj][b])
            tab[(ni, nj)] = M
    def coll_of(name, ki, asg):
        c = 0
        for o in names:
            if o == name:
                continue
            c += int(tab[(name, o)][ki, asg[o]]) if (name, o) in tab else int(tab[(o, name)][asg[o], ki])
        return c
    def total(asg):
        return sum(int(tab[(names[i], nj)][asg[names[i]], asg[nj]])
                   for i in range(len(names)) for nj in names[i + 1:])
    rng = np.random.default_rng(seed); best_asg = None; best_key = None
    for r in range(restarts):
        asg = {n: (int(min(range(len(cand[n])), key=lambda k: (round(optime[n][k], 3), travel[n][k]))) if r == 0 else int(rng.integers(len(cand[n])))) for n in names}
        for _ in range(iters):
            conf = [n for n in names if coll_of(n, asg[n], asg) > 0]
            if not conf:
                break
            n = conf[int(rng.integers(len(conf)))]
            asg[n] = min(range(len(cand[n])), key=lambda k: (coll_of(n, k, asg), round(optime[n][k], 3), travel[n][k]))
        for n in names:
            ks = [k for k in range(len(cand[n])) if coll_of(n, k, asg) == 0]
            if ks:
                asg[n] = min(ks, key=lambda k: (round(optime[n][k], 3), travel[n][k]))
        tc = total(asg)
        sys_optime = max(float(optime[n][asg[n]]) for n in names)
        tt = float(sum(travel[n][asg[n]] for n in names))
        if best_key is None or (tc, round(sys_optime, 3), tt) < best_key:
            best_key = (tc, round(sys_optime, 3), tt); best_asg = dict(asg)
        if best_key[0] == 0:
            break
    for n in names:
        best[n] = cand[n][best_asg[n]]
    if best_key[0] == 0:
        print("  [ok] all target poses are mutually collision-free")
    else:
        for i, ni in enumerate(names):
            for nj in names[i + 1:]:
                if pair_collides(best[ni], bases[ni], best[nj], bases[nj]):
                    print(f"  [warn] targets {ni}<->{nj} still collide "
                          f"(no collision-free IK branch combination exists)")
    return best


def greedy_pairwise_selection(arms_data: Dict) -> Optional[Dict]:
    arm_names = list(arms_data.keys())
    best = {}
    selected_paths_with_bases = []
    for name in arm_names:
        d = arms_data[name]
        start  = d['start_q']; target = d['target_local']
        rot    = d['target_rot']; base = d['base']
        con    = d.get('constrain_orient', False)

        # ---- JOINT CONFIG GIVEN: use as-is, no IK -----------------------------
        if d.get('given_q') is not None:
            q = np.asarray(d['given_q'], float)
            d['_sols'] = [q]; d['n_sols'] = 1; d['n_candidates_evaluated'] = 1
            best[name] = q
            qw = nearest_wrap(q, start)
            d['ik_time_ms'] = 0.0
            d['joint_aware_score'] = {
                'op_time_s': round(op_time(qw, start), 3),
                'weighted_travel_rad': round(float(np.sum(JOINT_COST_W * np.abs(qw - start))), 4),
                'max_joint_excursion_deg': round(float(np.max(np.abs(np.degrees(qw - start)))), 1)}
            p, T = fk(q); d['pos_err_mm'] = 0.0
            print(f'  [{name}] JOINTS given -> used as-is'
                  f'  op_time={d["joint_aware_score"]["op_time_s"]:.2f}s'
                  f'  max_exc={d["joint_aware_score"]["max_joint_excursion_deg"]:.0f}deg (no IK)')
            path = linear_interp_path(start, q, N_PATH_SMP)
            selected_paths_with_bases.append((path, base))
            continue

        t0 = time.time()
        sols = solve_ik(target, rot, start, constrain_orient=con)
        d['ik_time_ms'] = round((time.time() - t0) * 1000, 1)
        if not sols:
            if con:
                print(f'  [{name}] FAIL: no IK solution at the requested orientation '
                      f'(try Enter=FREE orientation, or a reachable quaternion)')
            else:
                print(f'  [{name}] FAIL: no IK solution')
            return None
        d['n_sols'] = len(sols)
        d['_sols'] = sols
        scored = []
        for q in sols:
            score, qw, breakdown = joint_aware_score(q, start, base, selected_paths_with_bases)
            scored.append((score, qw, breakdown))
        scored.sort(key=lambda x: x[0])
        sc, best_q, breakdown = scored[0]
        best[name] = best_q
        d['joint_aware_score'] = breakdown
        d['n_candidates_evaluated'] = len(scored)
        p, T = fk(best_q)
        pos_err = float(np.linalg.norm(p - target)) * 1000
        d['pos_err_mm'] = round(pos_err, 3)
        ori_txt = ''
        if d.get('constrain_orient', False) and d['target_rot'] is not None:
            oe = float(np.degrees(rot_err(T[:3, :3], d['target_rot'])))
            d['ori_err_deg'] = round(oe, 3)
            ori_txt = f'  ori_err={oe:.2f}deg'
        print(f'  [{name}] {len(sols)} sol(s)  selected:'
              f'  op_time={breakdown["op_time_s"]:.2f}s'
              f'  travel={breakdown["weighted_travel_rad"]:.3f}rad'
              f'  max_exc={breakdown["max_joint_excursion_deg"]:.0f}deg'
              f'{ori_txt}'
              f'  pos_err={pos_err:.2f}mm')
        path = linear_interp_path(start, best_q, N_PATH_SMP)
        selected_paths_with_bases.append((path, base))
    best = repair_endpoint_collisions(best, arms_data)
    print('  --- final selection (after endpoint collision resolution) ---')
    for name in arm_names:
        d = arms_data[name]; s = d['start_q']; qe = nearest_wrap(best[name], s)
        otime = float(np.max(np.abs(qe - s) / VEL_LIM))
        mexc  = float(np.max(np.abs(np.degrees(qe - s))))
        trav  = float(np.sum(JOINT_COST_W * np.abs(qe - s)))
        line  = (f'  [{name}] FINAL: op_time={otime:.2f}s  max_exc={mexc:.0f}deg'
                 f'  travel={trav:.2f}rad')
        if d.get('constrain_orient', False) and d['target_rot'] is not None:
            line += f'  ori_err={np.degrees(rot_err(fk(best[name])[1][:3, :3], d["target_rot"])):.2f}deg'
        if mexc > 150.0:
            line += '  [large excursion: forced by orientation+collision; try Enter=FREE orientation]'
        print(line)
        d['final_op_time_s'] = round(otime, 3)
        d['final_max_exc_deg'] = round(mexc, 1)
    return best


if _ROS_OK:
    class JointReader(Node):
        def __init__(self):
            super().__init__('step_31_reader')
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


def _show(label, q, base):
    deg = ', '.join(f'{np.degrees(v):7.2f}' for v in q)
    ee  = fk_world(q, base)
    print(f'    {label}'); print(f'      [{deg}] deg')
    print(f'      EE world: [{ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}] m')


def main(args=None):
    print('\n' + '='*66)
    print('  STEP 31  --  Joint-Aware IK Selection (Greedy Pairwise) [DUAL ARM]')
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

    arms_data = {}
    for name in ARM_NAMES:
        base = ROBOT_BASES[name]
        print(f'\n  {"-"*60}')
        print(f'  ARM: {name.upper()}  base={base.tolist()}')
        _show('Current Gazebo:', cur_q[name], base)

        given_q = None
        while True:
            raw = input(f'  [{name}] TARGET X Y Z (m)  OR  6 joints j1..j6 (deg): ').strip()
            try:
                vals = [float(v) for v in raw.replace(',', ' ').split()]
            except ValueError:
                continue
            if len(vals) == 3:
                tgt_xyz = vals; break
            elif len(vals) == 6:
                given_q = np.clip(np.radians(np.array(vals)), POS_LIM[:, 0], POS_LIM[:, 1]); break
            else:
                print('    enter 3 numbers (X Y Z, metres) or 6 numbers (joints, deg)')

        if given_q is not None:
            # joint configuration given -> use as-is, no IK, no orientation prompt
            p_local, _ = fk(given_q)
            arms_data[name] = {
                'base': base, 'start_q': cur_q[name],
                'target_world': (p_local + base).tolist(),
                'target_local': p_local, 'target_rot': None,
                'constrain_orient': False, 'given_q': given_q}
            _show('Given joints ->', given_q, base)
            continue

        tgt_world = np.array(tgt_xyz); tgt_local = tgt_world - base
        raw_q = input(f'  [{name}] TARGET quaternion w x y z (Enter=FREE orientation): ').strip()
        if raw_q == '':
            R = None
            constrain = False
        else:
            constrain = True
            try:
                w, x, y, z = [float(v) for v in raw_q.replace(',', ' ').split()]
                n2 = w*w + x*x + y*y + z*z
                if n2 < 1e-9: R = np.eye(3)
                else:
                    w, x, y, z = [v/np.sqrt(n2) for v in [w,x,y,z]]
                    R = np.array([
                        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
            except ValueError: R = np.eye(3)
        arms_data[name] = {
            'base': base, 'start_q': cur_q[name],
            'target_world': tgt_world.tolist(),
            'target_local': tgt_local, 'target_rot': R,
            'constrain_orient': constrain, 'given_q': None}

    dur_raw  = input('\n  Duration [s]  (Enter=10.0): ').strip()
    duration = float(dur_raw) if dur_raw else 10.0

    print('\n  Solving IK with greedy joint-aware selection...')
    t_ik_total = time.time()
    best = greedy_pairwise_selection(arms_data)
    ik_total_ms = round((time.time() - t_ik_total) * 1000, 1)
    if best is None: print('\n  FAIL'); sys.exit(1)

    out = {'duration': duration, 'arm_names': ARM_NAMES,
           'ik_total_time_ms': ik_total_ms,
           'selection_method': 'greedy_pairwise_joint_aware'}
    for name in ARM_NAMES:
        d = arms_data[name]; q = best[name]; ee = fk_world(q, d['base'])
        out[name] = {
            'base'             : np.array(d['base']).tolist(),
            'start_joints'     : d['start_q'].tolist(),
            'target_joints'    : q.tolist(),
            'target_joints_deg': np.degrees(q).tolist(),
            'target_world'     : d['target_world'],
            'target_ee_world'  : ee.tolist(),
            'position_error_mm': d['pos_err_mm'],
            'ik_time_ms'       : d['ik_time_ms'],
            'n_solutions'      : d['n_sols'],
            'n_candidates'     : d['n_candidates_evaluated'],
            'input_mode'       : ('joints' if d.get('given_q') is not None else 'position'),
            'joint_aware_score': d['joint_aware_score']}
        _show(f'  [{name}] target:', q, d['base'])

    with open('s31_ik.json', 'w') as fh: json.dump(out, fh, indent=2)
    print(f'\n  IK total time : {ik_total_ms:.0f} ms')
    print(f'  Saved         : s31_ik.json')
    print(f'  Next          : ros2 run dual_arm_sync step_32\n')
    if node:
        try: node.destroy_node()
        except: pass
        rclpy.shutdown()


if __name__ == '__main__': main()