
#!/usr/bin/env python3
"""
step_22.py  --  Global B-Spline Trajectory Seed   [6 ARM]
=========================================================
INPUT  : s21_ik.json
OUTPUT : s22_trajectories.json

One global clamped cubic B-spline per arm (SEED_NCP control points), straight
line start->target in joint space. Duration is scaled up until velocity and
acceleration peaks fit the M1013 limits. The global-CP form (not a 5seg x 4cp
grid) is what step_24's control-point retraction edits directly.
"""
import json, os, sys, time
import numpy as np
from scipy.interpolate import BSpline

sys.path.insert(0, os.path.dirname(__file__))
from _robot6x import (NDOF, POS_LIM, VEL_LIM, ACC_LIM, RATE_HZ,
                      ROBOT_BASES, ARM_NAMES, fk_world)

DEG      = 3
SEED_NCP = 12


def make_knots(ncp):
    ni  = max(0, ncp - DEG - 1)
    inn = np.linspace(0, 1, ni + 2)[1:-1] if ni > 0 else np.array([])
    return np.concatenate([np.zeros(DEG + 1), inn, np.ones(DEG + 1)])


def seed_cp(start_q, end_q, ncp=SEED_NCP):
    s = np.linspace(0., 1., ncp)
    cp = start_q[None, :] + s[:, None] * (end_q - start_q)[None, :]
    return np.clip(cp, POS_LIM[:, 0], POS_LIM[:, 1])


def eval_cp(cp, duration):
    kn = make_knots(len(cp)); n = max(2, int(round(duration * RATE_HZ)))
    s = np.linspace(0., 1., n)
    pos = np.zeros((n, NDOF)); vel = np.zeros((n, NDOF)); acc = np.zeros((n, NDOF))
    for j in range(NDOF):
        spl = BSpline(kn, cp[:, j], DEG, extrapolate=True)
        pos[:, j] = spl(s)
        vel[:, j] = spl.derivative(1)(s) / duration
        acc[:, j] = spl.derivative(2)(s) / duration ** 2
    return np.clip(pos, POS_LIM[:, 0], POS_LIM[:, 1]), vel, acc


def scale_duration(cp, duration):
    pos, vel, acc = eval_cp(cp, duration)
    sv = sa = 1.0
    for j in range(NDOF):
        vp = float(np.max(np.abs(vel[:, j]))); ap = float(np.max(np.abs(acc[:, j])))
        if vp > VEL_LIM[j]: sv = max(sv, vp / VEL_LIM[j])
        if ap > ACC_LIM[j]: sa = max(sa, float(np.sqrt(ap / ACC_LIM[j])))
    scale = max(sv, sa)
    if scale > 1.0:
        duration = duration * scale * 1.05
        pos, vel, acc = eval_cp(cp, duration)
    return pos, vel, acc, duration, sv, sa


def build_arm(name, start_q, end_q, base, duration):
    cp = seed_cp(start_q, end_q)
    pos, vel, acc, dur, sv, sa = scale_duration(cp, duration)
    n = len(pos)
    ee = np.array([fk_world(pos[k], base) for k in range(n)])
    plen = float(np.sum(np.linalg.norm(np.diff(ee, axis=0), axis=1)))
    ee_err = float(np.linalg.norm(fk_world(pos[-1], base) - fk_world(end_q, base)) * 1000)
    print(f'  [{name}] {n} smp  dur={dur:.2f}s  EE_path={plen*100:.1f}cm  '
          f'ee_err={ee_err:.2f}mm  vel_sc={sv:.2f}  acc_sc={sa:.2f}')
    return {
        'robot_name': name,
        'metadata': {'start_joints': start_q.tolist(), 'end_joints': end_q.tolist(),
                     'target_ee_world': fk_world(end_q, base).tolist(),
                     'duration': float(dur), 'n_samples': int(n),
                     'ee_path_length_m': round(plen, 5), 'ee_err_mm': round(ee_err, 3),
                     'vel_scale': round(sv, 4), 'acc_scale': round(sa, 4), 'degree': DEG},
        'spline': {'degree': DEG, 'n_cp': int(len(cp)), 'global_cp': cp.tolist()},
        'trajectory': {'time': np.linspace(0., dur, n).tolist(),
                       'positions': pos.tolist(), 'velocities': vel.tolist(),
                       'accelerations': acc.tolist(),
                       'arc_fracs': np.linspace(0., 1., n).tolist()}}


def resample(pos, nout):
    sin = np.linspace(0, 1, len(pos)); sout = np.linspace(0, 1, nout)
    return np.clip(np.vstack([np.interp(sout, sin, pos[:, j]) for j in range(NDOF)]).T,
                   POS_LIM[:, 0], POS_LIM[:, 1])


def main():
    print('\n' + '='*66)
    print('  STEP 22  --  Global B-Spline Trajectory Seed [6 ARM]')
    print('='*66)
    if not os.path.exists('s21_ik.json'):
        print('  s21_ik.json not found -- run step_21 first'); sys.exit(1)
    with open('s21_ik.json') as fh: ik = json.load(fh)
    arm_names = ik.get('arm_names', ARM_NAMES)
    requested = float(ik.get('duration', 10.0))
    print(f'\n  Arms: {arm_names}  dur={requested:.2f}s')
    out = {'duration': requested, 'arm_names': arm_names}
    max_dur = 0.; t0 = time.time()
    for name in arm_names:
        d = ik[name]
        out[name] = build_arm(name, np.array(d['start_joints'], float),
                              np.array(d['target_joints'], float),
                              np.array(ROBOT_BASES.get(name, [0, 0, 0])), requested)
        max_dur = max(max_dur, out[name]['metadata']['duration'])
    # sync all arms to the longest duration
    for name in arm_names:
        if out[name]['metadata']['duration'] < max_dur - 1e-6:
            pos = np.array(out[name]['trajectory']['positions'])
            nout = max(2, int(round(max_dur * RATE_HZ)))
            p2 = resample(pos, nout); dt = max_dur / max(nout - 1, 1)
            v2 = np.gradient(p2, dt, axis=0); a2 = np.gradient(v2, dt, axis=0)
            out[name]['trajectory'].update({'time': np.linspace(0, max_dur, nout).tolist(),
                'positions': p2.tolist(), 'velocities': v2.tolist(),
                'accelerations': a2.tolist(), 'arc_fracs': np.linspace(0, 1, nout).tolist()})
            out[name]['metadata']['duration'] = max_dur
            out[name]['metadata']['n_samples'] = nout
    out['duration'] = max_dur
    out['total_time_ms'] = round((time.time() - t0) * 1000, 1)
    with open('s22_trajectories.json', 'w') as fh: json.dump(out, fh, indent=2)
    print(f'\n  Synced duration: {max_dur:.2f}s')
    print(f'  Saved: s22_trajectories.json')
    print(f'  Next : ros2 run dual_arm_sync step_23\n')


if __name__ == '__main__': main()



