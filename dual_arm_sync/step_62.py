#!/usr/bin/env python3
"""
step_62.py  --  B-Spline Trajectory Generator
===============================================
INPUT  : s61_ik.json
OUTPUT : s62_trajectories.json

Builds one clamped cubic B-spline per arm:
  5 segments × 4 control points = 16 unique CPs (degree 3)
  Single global spline (C2 continuous everywhere, no stitching artifacts)
  Duration scaled up if velocity or acceleration limits are exceeded.

Paper metric written:
  - bspline_time_ms       (planning time for trajectory stage)
  - path_length_m         (Cartesian EE path length)
  - vel_scale, acc_scale  (how much duration was stretched)
  - optimality_ratio      (path_length vs straight-line lower bound)
"""

import json, os, sys, time
import numpy as np
from scipy.interpolate import BSpline
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(__file__))
from _robot import (NDOF, POS_LIM, VEL_LIM, ACC_LIM, RATE_HZ,
                    ROBOT_BASES, ARM_NAMES, fk_world)

N_SEG    = 5
N_CP_SEG = 4
DEG      = 3


# ── Spline utilities ──────────────────────────────────────────────────────────

def make_knots(ncp: int) -> np.ndarray:
    ni  = max(0, ncp - DEG - 1)
    inn = np.linspace(0, 1, ni + 2)[1:-1] if ni > 0 else np.array([])
    return np.concatenate([np.zeros(DEG + 1), inn, np.ones(DEG + 1)])


def build_cp(start_q: np.ndarray, end_q: np.ndarray) -> np.ndarray:
    """Linear CPs from start to end. Shape (N_SEG, N_CP_SEG, NDOF)."""
    total   = N_SEG * (N_CP_SEG - 1) + 1
    s_cp    = np.linspace(0., 1., total)
    cp_flat = start_q + s_cp[:, None] * (end_q - start_q)
    cp_flat = np.clip(cp_flat, POS_LIM[:, 0], POS_LIM[:, 1])
    cp_segs = np.zeros((N_SEG, N_CP_SEG, NDOF))
    for seg in range(N_SEG):
        cp_segs[seg] = cp_flat[seg * (N_CP_SEG - 1): seg * (N_CP_SEG - 1) + N_CP_SEG]
    return cp_segs


def eval_spline(cp_segs: np.ndarray, duration: float
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    total   = N_SEG * (N_CP_SEG - 1) + 1
    cp_flat = np.zeros((total, NDOF))
    for seg in range(N_SEG):
        i0             = seg * (N_CP_SEG - 1)
        cp_flat[i0: i0 + N_CP_SEG] = cp_segs[seg]
    knots   = make_knots(total)
    n_steps = max(2, int(round(duration * RATE_HZ)))
    s       = np.linspace(0., 1., n_steps)
    pos = np.zeros((n_steps, NDOF))
    vel = np.zeros((n_steps, NDOF))
    acc = np.zeros((n_steps, NDOF))
    for j in range(NDOF):
        spl       = BSpline(knots, cp_flat[:, j], DEG, extrapolate=True)
        pos[:, j] = spl(s)
        vel[:, j] = spl.derivative(1)(s) / duration
        acc[:, j] = spl.derivative(2)(s) / duration**2
    pos = np.clip(pos, POS_LIM[:, 0], POS_LIM[:, 1])
    return pos, vel, acc, np.linspace(0., duration, n_steps)


def scale_duration(cp_segs, duration) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                               np.ndarray, float, float, float]:
    pos, vel, acc, t = eval_spline(cp_segs, duration)
    sv = sa = 1.0
    for j in range(NDOF):
        vp = float(np.max(np.abs(vel[:, j])))
        ap = float(np.max(np.abs(acc[:, j])))
        if vp > VEL_LIM[j]: sv = max(sv, vp / VEL_LIM[j])
        if ap > ACC_LIM[j]: sa = max(sa, float(np.sqrt(ap / ACC_LIM[j])))
    scale = max(sv, sa)
    if scale > 1.0:
        duration = duration * scale * 1.05
        pos, vel, acc, t = eval_spline(cp_segs, duration)
    return pos, vel, acc, t, duration, sv, sa


# ── Build one arm ─────────────────────────────────────────────────────────────

def build_arm(name: str, start_q: np.ndarray, end_q: np.ndarray,
              base: np.ndarray, duration: float, t_build_start: float) -> Dict:

    cp_segs                  = build_cp(start_q, end_q)
    pos, vel, acc, t, dur, sv, sa = scale_duration(cp_segs, duration)

    n   = len(pos)
    arc = np.linspace(0., 1., n)
    ee  = np.array([fk_world(pos[k], base) for k in range(n)])
    plen = float(np.sum(np.linalg.norm(np.diff(ee, axis=0), axis=1)))

    # Optimality ratio: path length vs minimum possible (straight EE line)
    ee_direct = float(np.linalg.norm(fk_world(end_q, base) - fk_world(start_q, base)))
    opt_ratio = float(plen / ee_direct) if ee_direct > 1e-6 else 1.0

    j_err_deg = float(np.max(np.abs(np.degrees(pos[-1] - end_q))))
    ee_err_mm = float(np.linalg.norm(fk_world(pos[-1], base) -
                                      fk_world(end_q,   base)) * 1000)
    t_ms      = round((time.time() - t_build_start) * 1000, 1)

    print(f'  [{name}] {n} smp  dur={dur:.2f}s  '
          f'EE_path={plen*100:.1f}cm  j_err={j_err_deg:.3f}deg  '
          f'ee_err={ee_err_mm:.2f}mm  vel_sc={sv:.2f}  acc_sc={sa:.2f}')

    seg_info = []
    for seg in range(N_SEG):
        seg_info.append({
            'segment'  : seg,
            'arc_start': round(seg / N_SEG, 4),
            'arc_end'  : round((seg + 1) / N_SEG, 4),
            'cp'       : cp_segs[seg].tolist(),
            'n_cp'     : N_CP_SEG,
        })

    return {
        'robot_name': name,
        'metadata': {
            'start_joints'     : start_q.tolist(),
            'end_joints'       : end_q.tolist(),
            'target_ee_world'  : fk_world(end_q, base).tolist(),
            'duration'         : float(dur),
            'n_samples'        : int(n),
            'ee_path_length_m' : round(plen, 5),
            'ee_err_mm'        : round(ee_err_mm, 3),
            'vel_scale'        : round(sv, 4),
            'acc_scale'        : round(sa, 4),
            'optimality_ratio' : round(opt_ratio, 4),
            'bspline_time_ms'  : t_ms,
            'n_seg'            : N_SEG,
            'n_cp_seg'         : N_CP_SEG,
            'degree'           : DEG,
        },
        'spline': {'n_seg': N_SEG, 'n_cp_seg': N_CP_SEG,
                   'degree': DEG, 'segments': seg_info},
        'trajectory': {
            'time'         : t.tolist(),
            'positions'    : pos.tolist(),
            'velocities'   : vel.tolist(),
            'accelerations': acc.tolist(),
            'arc_fracs'    : arc.tolist(),
        },
        'ee_path': {'positions': ee.tolist()},
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('\n' + '='*66)
    print('  STEP 62  --  B-Spline Trajectory Generator')
    print('='*66)

    if not os.path.exists('s61_ik.json'):
        print('  s61_ik.json not found -- run step_61 first'); sys.exit(1)
    with open('s61_ik.json') as fh: ik = json.load(fh)

    arm_names = ik.get('arm_names', ARM_NAMES)
    requested = float(ik.get('duration', 10.0))
    print(f'\n  Arms: {arm_names}  requested_dur={requested:.2f}s')

    out = {'duration': requested, 'arm_names': arm_names}
    max_dur = 0.
    t0_total = time.time()

    for name in arm_names:
        t0 = time.time()
        d  = ik[name]
        out[name] = build_arm(
            name,
            np.array(d['start_joints'],  dtype=float),
            np.array(d['target_joints'], dtype=float),
            np.array(ROBOT_BASES.get(name, [0, 0, 0])),
            requested, t0)
        max_dur = max(max_dur, out[name]['metadata']['duration'])

    # Sync all arms to global (longest) duration
    if max_dur > requested + 0.01:
        print(f'\n  Syncing all arms to global duration {max_dur:.3f}s')
        for name in arm_names:
            cur = float(out[name]['metadata']['duration'])
            if cur < max_dur - 0.01:
                pos  = np.array(out[name]['trajectory']['positions'])
                nout = max(2, int(round(max_dur * RATE_HZ)))
                sin  = np.linspace(0, 1, len(pos))
                sout = np.linspace(0, 1, nout)
                p2   = np.clip(
                    np.vstack([np.interp(sout, sin, pos[:, j]) for j in range(NDOF)]).T,
                    POS_LIM[:, 0], POS_LIM[:, 1])
                dt   = max_dur / max(nout - 1, 1)
                v2   = np.gradient(p2, dt, axis=0)
                a2   = np.gradient(v2, dt, axis=0)
                out[name]['trajectory'].update({
                    'time'         : np.linspace(0, max_dur, nout).tolist(),
                    'positions'    : p2.tolist(),
                    'velocities'   : v2.tolist(),
                    'accelerations': a2.tolist(),
                    'arc_fracs'    : sout.tolist(),
                })
                out[name]['metadata']['duration']  = max_dur
                out[name]['metadata']['n_samples'] = nout
    out['duration'] = max_dur
    out['total_bspline_time_ms'] = round((time.time() - t0_total) * 1000, 1)

    with open('s62_trajectories.json', 'w') as fh: json.dump(out, fh, indent=2)
    kb = os.path.getsize('s62_trajectories.json') / 1024.
    print(f'\n  Total B-spline time : {out["total_bspline_time_ms"]:.0f} ms')
    print(f'  Saved               : s62_trajectories.json ({kb:.0f} KB)')
    print(f'  Next                : ros2 run dual_arm_sync step_63\n')


if __name__ == '__main__': main()