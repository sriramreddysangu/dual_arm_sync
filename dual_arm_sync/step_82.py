#!/usr/bin/env python3
"""
step_82.py  —  B-Spline Trajectory Generation from Joint Configs
=================================================================
Input  : step81_joints.json
Output : step82_trajectories.json

Builds a cubic B-spline from start_joints to target_joints for each arm.
5 segments x 4 control points, clamped, single global spline.
Scales duration if velocity or acceleration limits are exceeded.

Run:
    ros2 run dual_arm_sync step_82
"""

import json, os, sys
from typing import Dict, List, Tuple
import numpy as np
from scipy.interpolate import BSpline

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
ACC_LIM = np.array([8.0,   8.0,   8.0,  12.0,  12.0,  12.0])
NDOF    = 6
RATE_HZ = 100.0

ROBOT_BASES: Dict[str, np.ndarray] = {
    'dsr01': np.array([0.0,  0.5, 0.0]),
    'dsr02': np.array([0.0, -0.5, 0.0]),
}

# Spline structure — matches step_2 / rest of pipeline
N_SEG    = 5
N_CP_SEG = 4
DEG      = 3


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
# B-SPLINE CORE
# ─────────────────────────────────────────────────────────────────────────────

def make_knots(ncp: int, deg: int = DEG) -> np.ndarray:
    """Clamped uniform knot vector."""
    n_inner = max(0, ncp - deg - 1)
    inner   = np.linspace(0, 1, n_inner + 2)[1:-1] if n_inner > 0 else np.array([])
    return np.concatenate([np.zeros(deg + 1), inner, np.ones(deg + 1)])


def build_cp(start_q: np.ndarray, end_q: np.ndarray) -> np.ndarray:
    """
    Linearly spaced control points from start_q to end_q.
    Shape: (N_SEG, N_CP_SEG, NDOF).

    Monotone CPs -> monotone joint motion -> no back-and-forth.
    Single global spline built by flattening and de-duplicating boundaries.
    """
    total    = N_SEG * (N_CP_SEG - 1) + 1   # 16 unique CPs
    s_cp     = np.linspace(0.0, 1.0, total)
    cp_flat  = start_q + s_cp[:, None] * (end_q - start_q)
    cp_flat  = np.clip(cp_flat, POS_LIM[:, 0], POS_LIM[:, 1])

    cp_segs  = np.zeros((N_SEG, N_CP_SEG, NDOF))
    for seg in range(N_SEG):
        i0             = seg * (N_CP_SEG - 1)
        cp_segs[seg]   = cp_flat[i0: i0 + N_CP_SEG]
    return cp_segs


def eval_trajectory(cp_segs: np.ndarray,
                    duration: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate as ONE global clamped cubic B-spline.
    Flatten (N_SEG, N_CP_SEG, NDOF) -> (total_unique, NDOF), build single spline.
    Returns pos, vel, acc, t  — all at RATE_HZ.
    """
    # De-duplicate shared boundary CPs
    total    = N_SEG * (N_CP_SEG - 1) + 1
    cp_flat  = np.zeros((total, NDOF))
    for seg in range(N_SEG):
        i0             = seg * (N_CP_SEG - 1)
        cp_flat[i0: i0 + N_CP_SEG] = cp_segs[seg]

    knots   = make_knots(total)
    n_steps = max(2, int(round(duration * RATE_HZ)))
    s       = np.linspace(0.0, 1.0, n_steps)

    pos = np.zeros((n_steps, NDOF))
    vel = np.zeros((n_steps, NDOF))
    acc = np.zeros((n_steps, NDOF))

    for j in range(NDOF):
        spl       = BSpline(knots, cp_flat[:, j], DEG, extrapolate=True)
        pos[:, j] = spl(s)
        vel[:, j] = spl.derivative(1)(s) / duration
        acc[:, j] = spl.derivative(2)(s) / duration ** 2

    pos = np.clip(pos, POS_LIM[:, 0], POS_LIM[:, 1])
    t   = np.linspace(0.0, duration, n_steps)
    return pos, vel, acc, t


def scale_duration(cp_segs: np.ndarray, duration: float) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Scale duration up if vel/acc limits are exceeded. Returns pos,vel,acc,t,duration."""
    pos, vel, acc, t = eval_trajectory(cp_segs, duration)
    sv = sa = 1.0
    for j in range(NDOF):
        vp = float(np.max(np.abs(vel[:, j])))
        ap = float(np.max(np.abs(acc[:, j])))
        if vp > VEL_LIM[j]:
            sv = max(sv, vp / VEL_LIM[j])
        if ap > ACC_LIM[j]:
            sa = max(sa, float(np.sqrt(ap / ACC_LIM[j])))
    scale = max(sv, sa)
    if scale > 1.0:
        duration = duration * scale * 1.05
        pos, vel, acc, t = eval_trajectory(cp_segs, duration)
    return pos, vel, acc, t, duration


# ─────────────────────────────────────────────────────────────────────────────
# BUILD ONE ARM
# ─────────────────────────────────────────────────────────────────────────────

def build_arm(name: str,
              start_q: np.ndarray,
              end_q:   np.ndarray,
              base:    np.ndarray,
              duration: float) -> Dict:

    print(f'  [{name}] Building B-spline  '
          f'({N_SEG} segs x {N_CP_SEG} CPs, degree={DEG}) ...')

    cp_segs              = build_cp(start_q, end_q)
    pos, vel, acc, t, dur = scale_duration(cp_segs, duration)

    if dur > duration:
        print(f'  [{name}] Duration scaled {duration:.2f}s -> {dur:.2f}s'
              f'  (vel/acc limits)')

    n_steps   = len(pos)
    arc_fracs = np.linspace(0.0, 1.0, n_steps)
    ee_path   = np.array([fk_pos(pos[k], base) for k in range(n_steps)])
    path_len  = float(np.sum(np.linalg.norm(np.diff(ee_path, axis=0), axis=1)))

    # Check final joint error (how well spline ends at target)
    j_err_deg = float(np.max(np.abs(np.degrees(pos[-1] - end_q))))
    ee_end    = fk_pos(pos[-1], base)
    ee_tgt    = fk_pos(end_q,   base)
    ee_err_mm = float(np.linalg.norm(ee_end - ee_tgt) * 1000)

    print(f'  [{name}] {n_steps} samples  dur={dur:.3f}s  '
          f'EE path={path_len*100:.1f}cm  '
          f'j_err={j_err_deg:.3f}deg  ee_err={ee_err_mm:.2f}mm')

    seg_info = []
    for seg in range(N_SEG):
        seg_info.append({
            'segment'  : seg,
            'arc_start': round(seg / N_SEG, 4),
            'arc_end'  : round((seg + 1) / N_SEG, 4),
            'arc_mid'  : round((seg + 0.5) / N_SEG, 4),
            'cp'       : cp_segs[seg].tolist(),
        })

    return {
        'robot_name': name,
        'metadata': {
            'start_joints'     : start_q.tolist(),
            'end_joints'       : end_q.tolist(),
            'start_joints_deg' : np.degrees(start_q).tolist(),
            'end_joints_deg'   : np.degrees(end_q).tolist(),
            'target_ee_world'  : ee_tgt.tolist(),
            'duration'         : float(dur),
            'n_samples'        : int(n_steps),
            'ee_path_length_m' : round(path_len, 5),
            'n_seg'            : N_SEG,
            'n_cp_seg'         : N_CP_SEG,
            'degree'           : DEG,
        },
        'spline': {
            'n_seg'    : N_SEG,
            'n_cp_seg' : N_CP_SEG,
            'degree'   : DEG,
            'segments' : seg_info,
        },
        'trajectory': {
            'time'         : t.tolist(),
            'positions'    : pos.tolist(),
            'velocities'   : vel.tolist(),
            'accelerations': acc.tolist(),
            'arc_fracs'    : arc_fracs.tolist(),
            'n_samples'    : int(n_steps),
        },
        'ee_path': {
            'positions': ee_path.tolist(),
            'arc_fracs': arc_fracs.tolist(),
            'length_m' : round(path_len, 5),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print('\n' + '=' * 64)
    print('  STEP 82  —  B-Spline Trajectory Generation')
    print('=' * 64)

    if not os.path.exists('step81_joints.json'):
        print('\n  step81_joints.json not found — run step_81 first')
        sys.exit(1)

    with open('step81_joints.json') as fh:
        data = json.load(fh)

    arm_names = data.get('arm_names', ['dsr01', 'dsr02'])
    requested = float(data.get('duration', 10.0))

    print(f'\n  Arms       : {arm_names}')
    print(f'  Requested  : {requested:.2f}s')
    print(f'  Structure  : {N_SEG} segs x {N_CP_SEG} CPs  degree={DEG}')
    print(f'  Rate       : {RATE_HZ:.0f} Hz\n')

    out = {
        'duration'  : requested,
        'arm_names' : arm_names,
        'source'    : 'step_82',
    }

    max_dur = 0.0
    for name in arm_names:
        arm_in  = data[name]
        base    = np.array(ROBOT_BASES.get(name, [0, 0, 0]))
        start_q = np.array(arm_in['start_joints'],  dtype=float)
        end_q   = np.array(arm_in['target_joints'], dtype=float)

        arm_out       = build_arm(name, start_q, end_q, base, requested)
        out[name]     = arm_out
        max_dur       = max(max_dur, arm_out['metadata']['duration'])

    # Sync all arms to the same (longest) duration via resampling
    if max_dur > requested:
        print(f'\n  Syncing all arms to global duration {max_dur:.3f}s ...')
        for name in arm_names:
            cur_dur = float(out[name]['metadata']['duration'])
            if cur_dur < max_dur:
                pos_in  = np.array(out[name]['trajectory']['positions'])
                n_out   = max(2, int(round(max_dur * RATE_HZ)))
                s_in    = np.linspace(0, 1, len(pos_in))
                s_out   = np.linspace(0, 1, n_out)
                pos_out = np.zeros((n_out, NDOF))
                for j in range(NDOF):
                    pos_out[:, j] = np.interp(s_out, s_in, pos_in[:, j])
                pos_out = np.clip(pos_out, POS_LIM[:, 0], POS_LIM[:, 1])
                dt      = max_dur / max(n_out - 1, 1)
                vel_out = np.gradient(pos_out, dt, axis=0)
                acc_out = np.gradient(vel_out,  dt, axis=0)
                t_out   = np.linspace(0., max_dur, n_out)
                out[name]['metadata']['duration']         = max_dur
                out[name]['metadata']['n_samples']        = n_out
                out[name]['trajectory']['time']           = t_out.tolist()
                out[name]['trajectory']['positions']      = pos_out.tolist()
                out[name]['trajectory']['velocities']     = vel_out.tolist()
                out[name]['trajectory']['accelerations']  = acc_out.tolist()
                out[name]['trajectory']['arc_fracs']      = s_out.tolist()
                print(f'  [{name}] resampled {len(pos_in)} -> {n_out} steps')

    out['duration'] = max_dur

    with open('step82_trajectories.json', 'w') as fh:
        json.dump(out, fh, indent=2)

    kb = os.path.getsize('step82_trajectories.json') / 1024.0
    print(f'\n  Saved : step82_trajectories.json  ({kb:.1f} KB)')
    print(f'  Next  : ros2 run dual_arm_sync step_83\n')


if __name__ == '__main__':
    main()