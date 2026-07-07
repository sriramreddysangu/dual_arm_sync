#!/usr/bin/env python3
"""
step_12.py  —  Cubic B-Spline Trajectory Generation  (4-arm)
═══════════════════════════════════════════════════════════════════════════════
Input  : ik_solutions.json  (4 arms: dsr01–dsr04)
Output : trajectories.json

STRUCTURE  (fixed, per arm)
───────────────────────────
N_SEG = 5  segments
N_CP  = 4  control points per segment
Degree = 3 (cubic)

LOGIC  (identical to step_2, generalised to N arms)
─────
1. Per arm: start_joints → target_joints
2. Build initial control points (linear seed in joint space)
3. Evaluate cubic B-spline at 100 Hz
4. Scale duration if velocity/acceleration limits exceeded
5. Store per-segment control points for step_14 to modify
═══════════════════════════════════════════════════════════════════════════════
"""

import json, os, sys
from typing import Dict, List, Tuple
import numpy as np
from scipy.interpolate import BSpline

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
    'dsr03': np.array([1.0,  0.5, 0.0]),
    'dsr04': np.array([1.0, -0.5, 0.0]),
}

N_SEG    = 5
N_CP_SEG = 4
DEG      = 3

# ─────────────────────────────────────────────────────────────────────────────
# FK
# ─────────────────────────────────────────────────────────────────────────────

def fk_pos(q: np.ndarray, base: np.ndarray) -> np.ndarray:
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
    return T[:3, 3] + base

# ─────────────────────────────────────────────────────────────────────────────
# B-SPLINE BUILDING BLOCKS
# ─────────────────────────────────────────────────────────────────────────────

def make_knots(ncp: int, deg: int = DEG) -> np.ndarray:
    n_inner = max(0, ncp - deg - 1)
    inner   = np.linspace(0, 1, n_inner + 2)[1:-1] if n_inner > 0 else np.array([])
    return np.concatenate([np.zeros(deg + 1), inner, np.ones(deg + 1)])

def build_initial_cp(start_q: np.ndarray, end_q: np.ndarray,
                     n_seg: int = N_SEG, n_cp: int = N_CP_SEG) -> np.ndarray:
    """
    Linearly spaced CPs from start_q to end_q.
    Shape: (n_seg, n_cp, NDOF)
    """
    total     = n_seg * (n_cp - 1) + 1
    s_cp      = np.linspace(0.0, 1.0, total)
    cp_global = (start_q[np.newaxis, :]
                 + s_cp[:, np.newaxis] * (end_q - start_q)[np.newaxis, :])
    cp_global = np.clip(cp_global, POS_LIM[:, 0], POS_LIM[:, 1])
    cp_segs   = np.zeros((n_seg, n_cp, NDOF))
    for seg in range(n_seg):
        i0 = seg * (n_cp - 1)
        cp_segs[seg] = cp_global[i0 : i0 + n_cp]
    return cp_segs

def _flatten_cp(cp_segs: np.ndarray, n_seg: int, n_cp: int) -> np.ndarray:
    total = n_seg * (n_cp - 1) + 1
    out   = np.zeros((total, cp_segs.shape[2]))
    for seg in range(n_seg):
        i0 = seg * (n_cp - 1)
        out[i0 : i0 + n_cp] = cp_segs[seg]
    return out

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATE FULL TRAJECTORY  — single global B-spline
# ─────────────────────────────────────────────────────────────────────────────

def eval_trajectory(cp_segs : np.ndarray,
                    duration: float,
                    n_seg   : int = N_SEG,
                    n_cp    : int = N_CP_SEG
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cp_global = _flatten_cp(cp_segs, n_seg, n_cp)
    n_global  = len(cp_global)
    knots     = make_knots(n_global)
    n_steps   = max(2, int(round(duration * RATE_HZ)))
    s_full    = np.linspace(0.0, 1.0, n_steps)
    pos = np.zeros((n_steps, NDOF)); vel = np.zeros_like(pos); acc = np.zeros_like(pos)
    for j in range(NDOF):
        spl       = BSpline(knots, cp_global[:, j], DEG, extrapolate=True)
        pos[:, j] = spl(s_full)
        vel[:, j] = spl.derivative(1)(s_full) / duration
        acc[:, j] = spl.derivative(2)(s_full) / duration**2
    pos = np.clip(pos, POS_LIM[:, 0], POS_LIM[:, 1])
    t   = np.linspace(0.0, duration, n_steps)
    return pos, vel, acc, t

def scale_duration(cp_segs : np.ndarray,
                   duration: float,
                   n_seg   : int = N_SEG,
                   n_cp    : int = N_CP_SEG
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    pos, vel, acc, t = eval_trajectory(cp_segs, duration, n_seg, n_cp)
    sv = sa = 1.0
    for j in range(NDOF):
        vp = float(np.max(np.abs(vel[:, j]))); ap = float(np.max(np.abs(acc[:, j])))
        if vp > VEL_LIM[j]: sv = max(sv, vp / VEL_LIM[j])
        if ap > ACC_LIM[j]: sa = max(sa, float(np.sqrt(ap / ACC_LIM[j])))
    scale = max(sv, sa)
    if scale > 1.0:
        duration = duration * scale * 1.05
        pos, vel, acc, t = eval_trajectory(cp_segs, duration, n_seg, n_cp)
    return pos, vel, acc, t, duration

# ─────────────────────────────────────────────────────────────────────────────
# BUILD ONE ARM'S TRAJECTORY
# ─────────────────────────────────────────────────────────────────────────────

def build_arm_trajectory(name    : str,
                         start_q : np.ndarray,
                         end_q   : np.ndarray,
                         base    : np.ndarray,
                         duration: float) -> Dict:
    print(f'  [{name}] building cubic B-spline  '
          f'({N_SEG} segs × {N_CP_SEG} CPs, degree={DEG}) ...')
    cp_segs = build_initial_cp(start_q, end_q)
    pos, vel, acc, t, dur = scale_duration(cp_segs, duration)
    if dur > duration:
        print(f'  [{name}] duration scaled {duration:.2f}s → {dur:.2f}s')

    ee_path  = np.array([fk_pos(pos[k], base) for k in range(len(pos))])
    path_len = float(np.sum(np.linalg.norm(np.diff(ee_path, axis=0), axis=1)))
    arc_fracs = np.linspace(0.0, 1.0, len(pos))

    seg_info = []
    for seg in range(N_SEG):
        seg_info.append({
            'segment'  : seg,
            'arc_start': round(seg / N_SEG, 4),
            'arc_end'  : round((seg + 1) / N_SEG, 4),
            'arc_mid'  : round((seg + 0.5) / N_SEG, 4),
            'cp'       : cp_segs[seg].tolist(),
        })

    print(f'  [{name}] ✅  {len(pos)} samples  dur={dur:.3f}s  '
          f'EE path={path_len*100:.1f}cm')

    return {
        'robot_name': name,
        'metadata'  : {
            'start_joints'    : start_q.tolist(),
            'end_joints'      : end_q.tolist(),
            'start_joints_deg': np.degrees(start_q).tolist(),
            'end_joints_deg'  : np.degrees(end_q).tolist(),
            'duration'        : float(dur),
            'n_samples'       : len(pos),
            'ee_path_length_m': round(path_len, 5),
            'n_seg'           : N_SEG,
            'n_cp_seg'        : N_CP_SEG,
            'degree'          : DEG,
        },
        'spline'    : {
            'n_seg'   : N_SEG,
            'n_cp_seg': N_CP_SEG,
            'degree'  : DEG,
            'segments': seg_info,
        },
        'trajectory': {
            'time'         : t.tolist(),
            'positions'    : pos.tolist(),
            'velocities'   : vel.tolist(),
            'accelerations': acc.tolist(),
            'arc_fracs'    : arc_fracs.tolist(),
            'n_samples'    : len(pos),
        },
        'ee_path'   : {
            'positions': ee_path.tolist(),
            'arc_fracs': arc_fracs.tolist(),
            'length_m' : round(path_len, 5),
        },
    }

# ─────────────────────────────────────────────────────────────────────────────
# TIME CAP
# ─────────────────────────────────────────────────────────────────────────────

def apply_time_cap(out: Dict, requested: float) -> Dict:
    """
    Cap each arm's duration at N_arms × requested seconds.

    FIX: previously rebuilt CPs from scratch (losing step_4 modifications).
    Now resamples the EXISTING trajectory array directly — preserves any
    spline refinements made by step_4 and is faster.
    """
    arm_names = [k for k in out if k.startswith('dsr')]
    max_dur   = len(arm_names) * requested
    for name in arm_names:
        dur = float(out[name]['metadata']['duration'])
        if dur <= max_dur:
            continue
        print(f'  [{name}] ⚠  {dur:.2f}s > cap {max_dur:.2f}s — resampling')
        pos_in = np.array(out[name]['trajectory']['positions'], dtype=float)
        n_out  = max(2, int(round(max_dur * RATE_HZ)))
        s_in   = np.linspace(0., 1., len(pos_in))
        s_out  = np.linspace(0., 1., n_out)
        # Resample positions
        pos_out = np.zeros((n_out, NDOF))
        for j in range(NDOF):
            pos_out[:, j] = np.interp(s_out, s_in, pos_in[:, j])
        pos_out = np.clip(pos_out, POS_LIM[:, 0], POS_LIM[:, 1])
        # Recompute vel / acc via finite differences on new time grid
        dt      = max_dur / max(n_out - 1, 1)
        vel_out = np.gradient(pos_out, dt, axis=0)
        acc_out = np.gradient(vel_out,  dt, axis=0)
        t_out   = np.linspace(0., max_dur, n_out)
        out[name]['metadata']['duration']        = max_dur
        out[name]['metadata']['n_samples']       = n_out
        out[name]['trajectory']['time']          = t_out.tolist()
        out[name]['trajectory']['positions']     = pos_out.tolist()
        out[name]['trajectory']['velocities']    = vel_out.tolist()
        out[name]['trajectory']['accelerations'] = acc_out.tolist()
        out[name]['trajectory']['arc_fracs']     = s_out.tolist()
    return out

def main():
    print('\n' + '=' * 68)
    print(f'  STEP 12  —  4-Arm Cubic B-Spline Trajectory  '
          f'({N_SEG} segs × {N_CP_SEG} CPs, degree={DEG})')
    print('=' * 68)

    if not os.path.exists('ik_solutions.json'):
        print('\n  ❌  ik_solutions.json not found — run step_11 first'); sys.exit(1)

    with open('ik_solutions.json') as fh:
        data = json.load(fh)

    arm_names = sorted([k for k in data if k.startswith('dsr')])
    if not arm_names:
        print('\n  ❌  No arm data found'); sys.exit(1)

    requested = float(data.get('duration', 10.0))
    max_dur   = len(arm_names) * requested

    print(f'\n  Arms       : {arm_names}')
    print(f'  Requested  : {requested:.2f}s')
    print(f'  Time cap   : {max_dur:.2f}s  ({len(arm_names)} × {requested:.2f}s)')
    print(f'  Structure  : {N_SEG} segments × {N_CP_SEG} CPs × {len(arm_names)} arms\n')

    out = {'duration': requested, 'time_cap': max_dur}
    for name in arm_names:
        arm_in  = data[name]
        base    = np.array(ROBOT_BASES.get(name, [0, 0, 0]))
        start_q = np.array(arm_in['start_joints'],  dtype=float)
        end_q   = np.array(arm_in['target_joints'], dtype=float)
        out[name] = build_arm_trajectory(name, start_q, end_q, base, requested)

    out = apply_time_cap(out, requested, arm_names)

    with open('trajectories.json', 'w') as fh:
        json.dump(out, fh, indent=2)

    kb = os.path.getsize('trajectories.json') / 1024.0
    print(f'\n  ✅  Saved: trajectories.json  ({kb:.1f} KB)')
    print('  Next  →  python3 step_13.py\n')

if __name__ == '__main__':
    main()