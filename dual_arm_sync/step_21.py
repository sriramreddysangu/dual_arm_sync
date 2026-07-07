#!/usr/bin/env python3
"""
step_21.py  --  Joint-Aware IK Selection (Greedy Pairwise)   [6 ARM]
====================================================================
INPUT  : interactive targets + live Gazebo joint states
OUTPUT : s21_ik.json

Greedy pairwise selection: each arm solves IK, scores candidates by joint-space
distance from current + clearance at target + path-collision count against the
arms already chosen, and keeps the lowest-cost candidate. Loops over the six
arms of _robot6x.
"""
import json, os, sys, time
from typing import Dict, Optional
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(__file__))
from _robot6x import (NDOF, POS_LIM, VEL_LIM, ROBOT_BASES, ARM_NAMES,
                      fk, fk_world, pair_min_dist, pair_collides,
                      _arm_caps, caps_collide)

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    _ROS_OK = True
except ImportError:
    _ROS_OK = False

W_POS, W_ROT = 1.0, 0.15
IK_KEEP = 40          # keep many branches (more collision-free options); table stays fast via cached caps
IK_TOL_POS, IK_TOL_ROT, IK_UNIQ = 0.010, 0.050, 0.12
N_PATH_SMP = 5
# Minimum-cost-from-start: weighted joint TRAVEL dominates; clearance and
# path-collisions are gentle tiebreakers. Each solution is UNWRAPPED to the joint
# representation nearest the start (+245 deg j1 -> -114 deg). Residual collisions
# are left to the downstream resolver (step_24/25).
JOINT_COST_W = np.array([1.0, 1.5, 1.5, 0.6, 0.6, 0.4])
PATH_COLL_W, CLEAR_W = 0.10, 0.05
ENDPOINT_COLL_W = 1000.0   # target-pose collisions are UNRESOLVABLE -> near-hard penalty
_PI_2 = np.pi / 2.0


def rot_err(Rg, Rw):
    return float(np.arccos(np.clip((np.trace(Rg.T @ Rw) - 1.0) / 2.0, -1.0, 1.0)))


def _ik_seeds(current, target_local):
    seeds = [current.copy()]
    for j in range(NDOF):
        for d in [0.3, -0.3, 0.6, -0.6]:
            s = current.copy(); s[j] += d
            seeds.append(np.clip(s, POS_LIM[:, 0], POS_LIM[:, 1]))
    px, py, pz = target_local
    t1 = np.arctan2(py, px); r_xy = np.sqrt(px**2 + py**2)
    from _robot6x import L2, L3, A
    r2 = r_xy - A; dist = np.sqrt(r2**2 + pz**2)
    c3 = np.clip((dist**2 - L2**2 - L3**2) / (2 * L2 * L3), -1, 1)
    for th3 in [np.arccos(c3), -np.arccos(c3)]:
        for wrist in [0, _PI_2, -_PI_2]:
            for base in [t1, t1 + _PI_2, t1 - _PI_2, t1 + np.pi, t1 - np.pi]:
                th2 = np.arctan2(pz, r2) - np.arctan2(L3 * np.sin(th3), L2 + L3 * np.cos(th3))
                seeds.append(np.clip(np.array([base, th2, th3, 0., wrist, 0.]),
                                     POS_LIM[:, 0], POS_LIM[:, 1]))
    return seeds


def solve_ik(target_local, target_rot, current, constrain_orient=False):
    """IK from the present configuration.
      * constrain_orient=False: POSITION-ONLY (orientation free) -> more branches.
      * constrain_orient=True : full SE(3) POSE IK -> EE orientation driven to
        target_rot, every kept branch re-verified to IK_TOL_ROT.
    All branches re-verified to IK_TOL_POS; lowest-op-time branches returned."""
    use_ori = bool(constrain_orient and target_rot is not None)
    valid = []
    def obj(q):
        p, T = fk(q)
        e = np.sum((p - target_local)**2)
        if use_ori:
            e += W_ROT * rot_err(T[:3, :3], target_rot)**2
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
        if all(float(np.linalg.norm(q - v)) >= IK_UNIQ for v in valid): valid.append(q)
    valid.sort(key=lambda q: (round(float(np.max(np.abs(nearest_wrap(q, current) - current) / VEL_LIM)), 3),
                              float(np.sum(JOINT_COST_W * np.abs(nearest_wrap(q, current) - current)))))
    return valid[:IK_KEEP]


def lin_path(s, e, n=N_PATH_SMP):
    return np.array([s + f * (e - s) for f in np.linspace(0., 1., n)])


def nearest_wrap(q, start):
    """Unwrap each joint to the +-2pi representation nearest start, within limits."""
    qw = q.copy()
    for j in range(NDOF):
        cands = [q[j] + 2*np.pi*k for k in (-2, -1, 0, 1, 2)]
        cands = [c for c in cands if POS_LIM[j, 0] <= c <= POS_LIM[j, 1]]
        if cands: qw[j] = min(cands, key=lambda c: abs(c - start[j]))
    return qw


def op_time(qw, start):
    """Min trajectory duration from present config: max_k |dq_k|/VEL_LIM[k].
    One joint swinging far (e.g. j3 by ~200 deg) sets the duration even when the
    weighted-sum travel looks small. Minimizing it shortens the motion."""
    return float(np.max(np.abs(qw - start) / VEL_LIM))


def score(q, start, base, prior=None):
    """Optimal IK branch from the PRESENT configuration: minimal operational time,
    then minimal weighted travel. PATH collisions are NOT considered here (handled
    by step_22 scan / step_23 resolver / step_24 Kuramoto). Mutual collision-
    freedom of the FINAL target poses is enforced by repair_endpoint_collisions()."""
    qw = nearest_wrap(q, start)
    travel = float(np.sum(JOINT_COST_W * np.abs(qw - start)))
    t_op = op_time(qw, start)
    key = (round(t_op, 3), round(travel, 3))      # cost only; no path collision
    max_exc = round(float(np.max(np.abs(np.degrees(qw - start)))), 1)
    return key, qw, {'op_time_s': round(t_op, 3), 'weighted_travel_rad': round(travel, 4),
                     'max_joint_excursion_deg': max_exc}


def repair_endpoint_collisions(best, arms_data, restarts=8, iters=400, seed=0):
    """Min-conflicts CSP: choose one IK branch per arm so all TARGET poses are
    mutually collision-free (target collisions cannot be path-deformed away).
    Precomputes a pairwise branch-collision table, runs min-conflicts with
    restarts, then a min-travel pass over collision-free branches."""
    names = list(best.keys())
    bases = {n: arms_data[n]["base"] for n in names}
    starts = {n: arms_data[n]["start_q"] for n in names}
    cand = {n: [nearest_wrap(q, starts[n]) for q in arms_data[n]["_sols"]] for n in names}
    travel = {n: np.array([float(np.sum(JOINT_COST_W * np.abs(q - starts[n]))) for q in cand[n]])
              for n in names}
    optime = {n: np.array([float(np.max(np.abs(q - starts[n]) / VEL_LIM)) for q in cand[n]])
              for n in names}   # per-branch min duration (worst-joint excursion / vel)
    caps = {n: [_arm_caps(q, bases[n]) for q in cand[n]] for n in names}   # FK once per branch
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
        sys_optime = max(float(optime[n][asg[n]]) for n in names)   # arms run concurrently
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


def greedy(arms_data):
    best = {}; prior = []
    for name in arms_data:
        d = arms_data[name]
        con = d.get('constrain_orient', False)
        sols = solve_ik(d['target_local'], d['target_rot'], d['start_q'], constrain_orient=con)
        if not sols:
            if con:
                print(f'  [{name}] FAIL: no IK at requested orientation '
                      f'(use Enter=FREE orientation, or a reachable quaternion)')
            else:
                print(f'  [{name}] FAIL: no IK')
            return None
        d['_sols'] = sols
        scored = sorted((score(q, d['start_q'], d['base'], prior) for q in sols),
                        key=lambda x: x[0])
        sc, q, brk = scored[0]
        best[name] = q; d['joint_aware_score'] = brk; d['n_sols'] = len(sols)
        T = fk(q)[1]
        d['pos_err_mm'] = round(float(np.linalg.norm(fk(q)[0] - d['target_local'])) * 1000, 3)
        ori_txt = ''
        if con and d['target_rot'] is not None:
            oe = float(np.degrees(rot_err(T[:3, :3], d['target_rot'])))
            d['ori_err_deg'] = round(oe, 3); ori_txt = f'  ori_err={oe:.2f}deg'
        print(f'  [{name}] {len(sols)} sol(s)  op_time={brk["op_time_s"]:.2f}s  '
              f'travel={brk["weighted_travel_rad"]:.3f}rad  max_exc={brk["max_joint_excursion_deg"]:.0f}deg{ori_txt}  '
              f'pos_err={d["pos_err_mm"]:.2f}mm')
        prior.append((lin_path(d['start_q'], q), d['base']))
    best = repair_endpoint_collisions(best, arms_data)
    print('  --- final selection (after endpoint collision resolution) ---')
    for name in arms_data:
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
        d['final_op_time_s'] = round(otime, 3); d['final_max_exc_deg'] = round(mexc, 1)
    return best


if _ROS_OK:
    class JointReader(Node):
        def __init__(self):
            super().__init__('step_21_reader')
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
            while rclpy.ok() and _t.time()-t0 < t:
                rclpy.spin_once(self, timeout_sec=0.05)
                if all(self._ready.values()): return True
            return False


def main(args=None):
    print('\n' + '='*66)
    print('  STEP 21  --  Joint-Aware IK Selection [6 ARM]')
    print('='*66)
    cur_q = {n: np.zeros(NDOF) for n in ARM_NAMES}; node = None
    if _ROS_OK:
        rclpy.init(args=args); node = JointReader()
        print('\n  Waiting for Gazebo joint states...')
        if node.wait(20.):
            for n in ARM_NAMES: cur_q[n] = node.joints(n)
            print('  All arms ready.')
        else:
            print('\n  ' + '!' * 60)
            print('  !!  GAZEBO JOINT STATES NOT RECEIVED (20s timeout).')
            print('  !!  Planning from the ZERO/home pose, NOT the real robot pose.')
            print('  !!  Collisions found/avoided here will NOT match the running')
            print('  !!  robot -- a "SAFE" result may be false. Bring up Gazebo +')
            print('  !!  controllers and verify:  ros2 topic echo /dsr01/gz/joint_states --once')
            print('  ' + '!' * 60 + '\n')
    arms_data = {}
    for name in ARM_NAMES:
        base = ROBOT_BASES[name]
        print(f'\n  ARM: {name.upper()}  base={base.tolist()}')
        while True:
            raw = input(f'  [{name}] TARGET pos X Y Z (m): ').strip()
            try: xyz = [float(v) for v in raw.replace(',', ' ').split()]
            except ValueError: continue
            if len(xyz) == 3: break
        tw = np.array(xyz)
        raw_q = input(f'  [{name}] TARGET quaternion w x y z (Enter=FREE orientation): ').strip()
        if raw_q == '':
            R = None; constrain = False
        else:
            constrain = True
            try:
                w, x, y, z = [float(v) for v in raw_q.replace(',', ' ').split()]
                n2 = w*w + x*x + y*y + z*z
                if n2 < 1e-9:
                    R = np.eye(3)
                else:
                    w, x, y, z = [v/np.sqrt(n2) for v in [w, x, y, z]]
                    R = np.array([
                        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
            except ValueError:
                R = np.eye(3)
        arms_data[name] = {'base': base, 'start_q': cur_q[name],
            'target_world': tw.tolist(), 'target_local': tw - base, 'target_rot': R,
            'constrain_orient': constrain}
    dur_raw = input('\n  Duration [s] (Enter=10.0): ').strip()
    duration = float(dur_raw) if dur_raw else 10.0
    print('\n  Solving IK (greedy joint-aware)...')
    t0 = time.time(); best = greedy(arms_data); ik_ms = round((time.time()-t0)*1000, 1)
    if best is None: print('\n  FAIL'); sys.exit(1)
    out = {'duration': duration, 'arm_names': ARM_NAMES, 'ik_total_time_ms': ik_ms,
           'selection_method': 'greedy_pairwise_joint_aware'}
    for name in ARM_NAMES:
        d = arms_data[name]; q = best[name]; ee = fk_world(q, d['base'])
        out[name] = {'base': np.array(d['base']).tolist(), 'start_joints': d['start_q'].tolist(),
            'target_joints': q.tolist(), 'target_joints_deg': np.degrees(q).tolist(),
            'end_joints': q.tolist(), 'target_world': d['target_world'],
            'target_ee_world': ee.tolist(), 'position_error_mm': d['pos_err_mm'],
            'n_solutions': d['n_sols'], 'joint_aware_score': d['joint_aware_score']}
    with open('s21_ik.json', 'w') as fh: json.dump(out, fh, indent=2)
    print(f'\n  IK total: {ik_ms:.0f} ms   Saved: s21_ik.json')
    print(f'  Next : ros2 run dual_arm_sync step_22\n')
    if node:
        try: node.destroy_node()
        except: pass
        rclpy.shutdown()


if __name__ == '__main__': main()