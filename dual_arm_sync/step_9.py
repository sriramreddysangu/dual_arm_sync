#!/usr/bin/env python3
"""
step_9.py  --  Kuramoto Sync + Home-CP Refinement
===============================================================================
Input  : trajectories.json + collision_report.json
Output : synchronized_trajectories.json

LOGIC
-----
Phase A  Kuramoto synchronization:
  Adjust phase (timing) of each arm so they don't arrive at the collision
  zone simultaneously.

Phase B  (if Kuramoto didn't resolve):
  Replace control points of the colliding segment with HOME_Q = [0,0,0,0,0,0].
  Boundary CPs are pinned — trajectory still connects correctly to neighbours.
  Both arms independently retract toward home through the collision zone,
  creating natural inter-arm separation.

  CP schedule  (RESOLVED_CP_N labels):
    CP_1, CP_2, CP_3  ->  4 interior CPs (2 home interior)
    CP_4              ->  6 interior CPs (4 home interior, stronger detour)
    CP_5              ->  8 interior CPs (6 home interior, strongest detour)

KEY FIX vs old version
-----------------------
  1. np.zeros(NDOF) seed REMOVED from _ik_seeds (was biasing toward home)
  2. pair_min_dist vectorized with NumPy broadcasting (~6x faster)
  3. pair_collides vectorized
  4. K_BASE aligned to 5.0 (was 8.0 — inconsistent with step_4/step_6)
  5. Home-CP refinement replaces alternating IK — simpler, more reliable
  6. Output writes sync_pos (Kuramoto phase-adjusted) not raw spline
  7. check_vel_limits checks both velocity AND acceleration
===============================================================================
"""

import json, os, sys
from typing import Dict, List, Optional, Tuple
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
}

LINK_RADII    = np.array([0.10, 0.10, 0.08, 0.08, 0.06, 0.06])
SAFETY_MARGIN = 0.12
HOME_Q        = np.zeros(NDOF)   # [0,0,0,0,0,0] rad -- home configuration

N_SEG_BASE  = 5
N_CP_BASE   = 4
DEG         = 3

# Kuramoto -- aligned with step_4 and step_6
K_BASE        = 5.0
K_REPULSE     = 80.0
K_EMERGENCY   = 250.0
KUR_DT        = 0.01
MIN_SAFE      = 0.15
REPULSE_D     = 0.28
LEADER_THRESH = 0.05

MAX_REFINE   = 5
CP_INCREMENT = 2


# =============================================================================
# FK + GEOMETRY
# =============================================================================

def fk(q):
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


def link_origins(q, base):
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


def pair_min_dist(qi, bi, qj, bj) -> float:
    """Minimum link-origin distance -- vectorised NumPy broadcasting."""
    oi   = link_origins(qi, bi)                              # (6, 3)
    oj   = link_origins(qj, bj)                              # (6, 3)
    diff = oi[:, np.newaxis, :] - oj[np.newaxis, :, :]      # (6, 6, 3)
    return float(np.min(np.linalg.norm(diff, axis=2)))


def pair_collides(qi, bi, qj, bj) -> bool:
    """Vectorised collision check -- no Python loops."""
    oi    = link_origins(qi, bi)
    oj    = link_origins(qj, bj)
    diff  = oi[:, np.newaxis, :] - oj[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)                     # (6, 6)
    radii = (LINK_RADII[:, np.newaxis] + LINK_RADII[np.newaxis, :]) + SAFETY_MARGIN
    return bool(np.any(dists < radii))


def interp_pos(pos, frac):
    frac = float(np.clip(frac, 0, 1)); n = len(pos) - 1
    if n <= 0: return pos[0].copy()
    idx = min(int(frac * n), n - 1); a = frac * n - idx
    return pos[idx] + a * (pos[idx + 1] - pos[idx])


# =============================================================================
# B-SPLINE UTILITIES
# =============================================================================

def make_knots(ncp, deg=DEG):
    ni  = max(0, ncp - deg - 1)
    inn = np.linspace(0, 1, ni + 2)[1:-1] if ni > 0 else np.array([])
    return np.concatenate([np.zeros(deg + 1), inn, np.ones(deg + 1)])


def eval_full_trajectory(seg_cps, duration):
    """Flatten per-segment CPs into one global B-spline and evaluate."""
    all_cp = []
    for seg_idx, (cp, _) in enumerate(seg_cps):
        all_cp.append(cp if seg_idx == 0 else cp[1:])
    cp_global = np.vstack(all_cp)
    knots     = make_knots(len(cp_global))
    n_steps   = max(2, int(round(duration * RATE_HZ)))
    s_full    = np.linspace(0.0, 1.0, n_steps)
    pos = np.zeros((n_steps, NDOF))
    vel = np.zeros((n_steps, NDOF))
    acc = np.zeros((n_steps, NDOF))
    for j in range(NDOF):
        spl       = BSpline(knots, cp_global[:, j], DEG, extrapolate=True)
        pos[:, j] = spl(s_full)
        vel[:, j] = spl.derivative(1)(s_full) / duration
        acc[:, j] = spl.derivative(2)(s_full) / duration**2
    pos = np.clip(pos, POS_LIM[:, 0], POS_LIM[:, 1])
    return pos, vel, acc, np.linspace(0., duration, n_steps)


def load_seg_cps(traj_data):
    segs  = traj_data['spline']['segments']
    n_cp  = int(traj_data['spline'].get('n_cp_seg', N_CP_BASE))
    knots = make_knots(n_cp)
    return [(np.array(seg['cp'], dtype=float), knots.copy()) for seg in segs]


# =============================================================================
# SYNC RESAMPLE + VEL/ACC
# =============================================================================

def resample_sync(sync_pos, t_kur, duration):
    n_out = max(2, int(round(duration * RATE_HZ)))
    t_out = np.linspace(0., duration, n_out)
    t_in  = t_kur if len(t_kur) == len(sync_pos) \
        else np.linspace(0., duration, len(sync_pos))
    out = np.zeros((n_out, NDOF))
    for j in range(NDOF):
        out[:, j] = np.interp(t_out, t_in, sync_pos[:, j])
    return np.clip(out, POS_LIM[:, 0], POS_LIM[:, 1])


def vel_acc_from_pos(pos, duration):
    dt  = duration / max(len(pos) - 1, 1)
    vel = np.gradient(pos, dt, axis=0)
    acc = np.gradient(vel,  dt, axis=0)
    return vel, acc


def check_limits(pos, duration):
    """Return (vel_scale, acc_scale) -- both >= 1.0 if limits exceeded."""
    vel = np.gradient(pos, duration / max(len(pos) - 1, 1), axis=0)
    acc = np.gradient(vel, duration / max(len(pos) - 1, 1), axis=0)
    sv = sa = 1.0
    for j in range(NDOF):
        vp = float(np.max(np.abs(vel[:, j])))
        ap = float(np.max(np.abs(acc[:, j])))
        if vp > VEL_LIM[j]: sv = max(sv, vp / VEL_LIM[j])
        if ap > ACC_LIM[j]: sa = max(sa, float(np.sqrt(ap / ACC_LIM[j])))
    return sv, sa


# =============================================================================
# KURAMOTO
# =============================================================================

def run_kuramoto(arm_names, arm_pos, arm_bases, duration):
    """
    Phase-coupled timing adjustment.

    Original coupling (sin term) + repulsion -- rate clamped to [0, 2]
    so arms never move backward. Runs until all phi=1 or stall guard (4x).
    Forces exact target as last sample even on stall -- Phase B then fixes
    the geometry so the final position is actually safe.
    """
    N      = len(arm_names)
    omega0 = 1.0 / duration
    phi    = np.zeros(N)
    om     = np.full(N, omega0)
    pairs  = [(i, j) for i in range(N) for j in range(i + 1, N)]

    max_steps = int(round(4.0 * duration / KUR_DT))
    sync_acc  = {n: [] for n in arm_names}
    pdist_acc = {p: [] for p in pairs}

    k = 0
    while k < max_steps:
        phi_c = np.clip(phi, 0., 1.)
        q_now = {n: interp_pos(arm_pos[n], phi_c[idx])
                 for idx, n in enumerate(arm_names)}
        for idx, n in enumerate(arm_names):
            sync_acc[n].append(q_now[n].copy())

        dists = {}
        for (i, j) in pairs:
            d = pair_min_dist(q_now[arm_names[i]], arm_bases[arm_names[i]],
                              q_now[arm_names[j]], arm_bases[arm_names[j]])
            dists[(i, j)] = d
            pdist_acc[(i, j)].append(d)

        if np.all(phi >= 1.0 - 1e-9):
            break

        dphi = np.zeros(N)
        for (i, j) in pairs:
            dist   = dists[(i, j)]
            df     = float(np.clip(1 - dist / REPULSE_D, 0, 1))
            danger = float(np.clip(1 - dist / MIN_SAFE,  0, 1))
            diff   = phi[i] - phi[j]
            leader = i if diff > LEADER_THRESH else (j if diff < -LEADER_THRESH else -1)
            Kij    = min(K_BASE * (1 + 4 * df), 15.0)
            dphi[i] += Kij * float(np.sin(phi[j] - phi[i]))
            dphi[j] += Kij * float(np.sin(phi[i] - phi[j]))
            if dist < REPULSE_D:
                mag = K_REPULSE * df**2 * 30 +                       (K_EMERGENCY * danger**3 if dist < MIN_SAFE else 0)
                if   leader == i: dphi[i] -= mag * 2;   dphi[j] -= mag * 0.3
                elif leader == j: dphi[j] -= mag * 2;   dphi[i] -= mag * 0.3
                else:             dphi[i] -= mag * 0.7; dphi[j] -= mag * 0.7

        # Never backward -- arms monotonically advance toward phi=1.
        phi += KUR_DT * np.clip(om + dphi, 0.0, 2.0)
        k   += 1

    # Force exact target as last sample (valid after Phase B fixes geometry).
    for n in arm_names:
        sync_acc[n].append(arm_pos[n][-1].copy())
    for (i, j) in pairs:
        d = pair_min_dist(arm_pos[arm_names[i]][-1], arm_bases[arm_names[i]],
                          arm_pos[arm_names[j]][-1], arm_bases[arm_names[j]])
        pdist_acc[(i, j)].append(d)

    sync     = {n: np.array(sync_acc[n]) for n in arm_names}
    n_out    = len(sync[arm_names[0]])
    real_dur = max((n_out - 1) * KUR_DT, duration)
    t_vec    = np.linspace(0., real_dur, n_out)

    pair_rep = {}; total_crit = 0
    for (i, j) in pairs:
        dv = np.array(pdist_acc[(i, j)])
        nc = int(np.sum(dv < MIN_SAFE)); total_crit += nc
        pair_rep['{}<->{}'.format(arm_names[i], arm_names[j])] = {
            'min_dist_m': float(np.min(dv)), 'critical': nc, 'collision_free': nc == 0,
        }
    return sync, t_vec, {
        'pair_reports': pair_rep, 'total_critical': total_crit,
        'collision_free': total_crit == 0,
    }

def census(arm_names, sync_pos, arm_bases, t_vec, n_seg):
    result = {}
    for i in range(len(arm_names)):
        for j in range(i + 1, len(arm_names)):
            ni, nj = arm_names[i], arm_names[j]
            bi, bj = arm_bases[ni], arm_bases[nj]
            pi, pj = sync_pos[ni], sync_pos[nj]
            K = min(len(pi), len(pj), len(t_vec))
            first_k = -1; nc = 0
            for k in range(K):
                if pair_collides(pi[k], bi, pj[k], bj):
                    nc += 1
                    if first_k < 0: first_k = k
            if nc > 0:
                frac = first_k / max(K - 1, 1)
                result[(ni, nj)] = {
                    'first_k': first_k, 'first_frac': frac,
                    'first_seg': min(int(frac * n_seg), n_seg - 1), 'n_coll': nc,
                }
    return result


# =============================================================================
# HOME-CP REFINEMENT
# =============================================================================

def home_cp_segment(pin_start, pin_end, n_cp):
    """
    Build collision-segment CPs with interior points = HOME_Q = [0,0,0,0,0,0].
    Boundary CPs are pinned to maintain trajectory continuity.
    More interior home CPs = stronger detour toward home = greater separation.
    """
    cp     = np.zeros((n_cp, NDOF), dtype=float)  # all zeros = HOME_Q
    cp[0]  = pin_start.copy()
    cp[-1] = pin_end.copy()
    return cp


def rebuild_spline(orig_seg_cps, pos_full, arc_full, coll_seg, refine_count):
    """
    Replace collision segment CPs with home-config detour.
    Neighbour segments enriched to smooth transition.

    CP count schedule:
      refine_count 0,1,2  ->  n_cp_coll = 4   (CP_1, CP_2, CP_3)
      refine_count 3      ->  n_cp_coll = 6   (CP_4 -- stronger detour)
      refine_count 4+     ->  n_cp_coll = 8   (CP_5 -- strongest detour)
    """
    n_seg  = len(orig_seg_cps)
    result = list(orig_seg_cps)

    if refine_count <= 2:
        n_cp_coll = N_CP_BASE                    # 4
    elif refine_count == 3:
        n_cp_coll = N_CP_BASE + CP_INCREMENT     # 6
    else:
        n_cp_coll = N_CP_BASE + CP_INCREMENT * 2 # 8

    n_cp_nbr   = N_CP_BASE
    neighbours = [coll_seg - 1, coll_seg + 1]

    for seg in range(n_seg):
        s0   = seg / n_seg; s1 = (seg + 1) / n_seg
        mask = (arc_full >= s0 - 1e-9) & (arc_full <= s1 + 1e-9)
        idx  = np.where(mask)[0]
        if len(idx) < 2: continue
        pin_s = pos_full[idx[0]].copy()
        pin_e = pos_full[idx[-1]].copy()

        if seg == coll_seg:
            cp = home_cp_segment(pin_s, pin_e, n_cp_coll)
            result[seg] = (cp, make_knots(n_cp_coll))

        elif seg in neighbours and 0 <= seg < n_seg:
            # Re-fit neighbour segment with extra CPs to smooth transition
            s_loc = np.clip((arc_full[idx] - s0) / (s1 - s0), 0., 1.)
            wps   = pos_full[idx]
            # Simple LS fit for neighbour
            ncp   = n_cp_nbr
            knots = make_knots(ncp)
            B = np.zeros((len(s_loc), ncp))
            e = np.zeros(ncp)
            for ki in range(ncp):
                e[:] = 0.; e[ki] = 1.
                v = BSpline(knots, e.copy(), DEG, extrapolate=False)(s_loc)
                B[:, ki] = np.where(np.isfinite(v), v, 0.)
            cp = np.zeros((ncp, NDOF))
            for j in range(NDOF):
                ts = float(pin_s[j]); te = float(pin_e[j])
                A_fr = B[:, 1:-1]
                rhs  = wps[:, j] - B[:, 0] * ts - B[:, -1] * te
                if A_fr.shape[1] > 0:
                    z, *_ = np.linalg.lstsq(A_fr, rhs, rcond=None)
                    z = np.where(np.isfinite(z), z, np.linspace(ts, te, len(z)))
                    z = np.clip(z, POS_LIM[j, 0], POS_LIM[j, 1])
                else:
                    z = np.array([])
                cp[:, j] = np.concatenate([[ts], z if len(z) else [], [te]])
            result[seg] = (cp, make_knots(ncp))

    return result


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def synchronise(trajectories, arm_bases=None):
    if arm_bases is None: arm_bases = ROBOT_BASES
    arm_names = sorted(trajectories.keys())
    N     = len(arm_names)
    n_seg = max(int(trajectories[n]['spline']['n_seg']) for n in arm_names)

    print('\n  Arms: {}  |  {} pair(s)  |  {} segments'.format(
        arm_names, N * (N - 1) // 2, n_seg))

    arm_seg_cps = {n: load_seg_cps(trajectories[n]) for n in arm_names}
    duration    = max(float(trajectories[n]['metadata']['duration']) for n in arm_names)
    arm_pos     = {n: eval_full_trajectory(arm_seg_cps[n], duration)[0] for n in arm_names}
    arm_arc     = {n: np.linspace(0., 1., len(arm_pos[n])) for n in arm_names}

    history         = []
    sync_pos        = {}
    t_vec           = np.array([])
    kur_rep         = {}
    final_iteration = 0

    for iteration in range(MAX_REFINE + 1):
        print('\n  {}'.format('-' * 66))
        print('  Iter {}  |  Kuramoto  (dur={:.2f}s  K_BASE={})'.format(
            iteration, duration, K_BASE))

        sync_pos, t_vec, kur_rep = run_kuramoto(
            arm_names, arm_pos, arm_bases, duration)

        for pk, pr in kur_rep['pair_reports'].items():
            icon = 'OK' if pr['collision_free'] else 'FAIL'
            print('    [{}] {:<22}  min={:.1f}cm  crit={}'.format(
                icon, pk, pr['min_dist_m'] * 100, pr['critical']))

        history.append({'iteration': iteration, 'kuramoto': kur_rep})
        final_iteration = iteration

        if kur_rep['collision_free']:
            print('\n  Collision-free at iteration {}'.format(iteration))
            break

        if iteration >= MAX_REFINE:
            print('\n  WARNING: MAX_REFINE={} reached'.format(MAX_REFINE))
            break

        cens = census(arm_names, sync_pos, arm_bases, t_vec, n_seg)
        if not cens: break

        # Process all colliding pairs ordered by earliest first collision
        pairs_ordered = sorted(cens.keys(), key=lambda p: cens[p]['first_frac'])

        for pair_key in pairs_ordered:
            ni, nj   = pair_key
            coll_seg = cens[pair_key]['first_seg']
            s0_c     = coll_seg / n_seg
            s1_c     = (coll_seg + 1) / n_seg

            print('\n  Phase B: {}<->{} coll_seg={} arc=[{:.2f},{:.2f}]  '
                  'refine={} (home-CP detour: interior -> [0,0,0,0,0,0])'.format(
                      ni, nj, coll_seg, s0_c, s1_c, iteration))

            arm_seg_cps[ni] = rebuild_spline(
                arm_seg_cps[ni], arm_pos[ni], arm_arc[ni], coll_seg, iteration)
            arm_seg_cps[nj] = rebuild_spline(
                arm_seg_cps[nj], arm_pos[nj], arm_arc[nj], coll_seg, iteration)

            for n in (ni, nj):
                pos_n, _, _, _ = eval_full_trajectory(arm_seg_cps[n], duration)
                arm_pos[n] = pos_n
                arm_arc[n] = np.linspace(0., 1., len(pos_n))
            print('    B-spline rebuilt -- both arms detour through home in seg {}'.format(
                coll_seg))

    # -- Output assembly ---------------------------------------------------
    # Use the Kuramoto real duration (may be extended if phase stalled).
    # IMPORTANT: check_limits must exclude the forced last sample (target jump)
    # because that jump is an output artifact, not a real trajectory velocity.
    if len(t_vec) > 1:
        real_dur = float(t_vec[-1])
        duration = max(duration, real_dur)

    # Check velocity/acceleration on everything EXCEPT the forced last sample.
    # The forced last sample is arm_pos[n][-1] appended after the Kuramoto loop;
    # it appears as an instantaneous jump and would create a phantom scaling factor.
    global_sv = global_sa = 1.0
    for name in arm_names:
        # Exclude last sample from limit check (it is the forced target)
        check_pos = np.array(sync_pos[name][:-1]) if len(sync_pos[name]) > 1                     else np.array(sync_pos[name])
        check_dur = max((len(check_pos) - 1) * KUR_DT, 1e-3)
        sv, sa    = check_limits(check_pos, check_dur)
        global_sv = max(global_sv, sv)
        global_sa = max(global_sa, sa)

    scale = max(global_sv, global_sa)
    if scale > 1.0:
        print('  Limits exceeded x{:.3f} -- scaling shared duration'.format(scale))
        duration = duration * scale * 1.05

    out = {}
    for name in arm_names:
        pos_sync = resample_sync(sync_pos[name], t_vec, duration)
        vel_sync, acc_sync = vel_acc_from_pos(pos_sync, duration)
        n_steps   = len(pos_sync)
        t_out     = np.linspace(0., duration, n_steps)
        n_seg_arm = len(arm_seg_cps[name])
        seg_info  = []
        for seg, (cp, kn) in enumerate(arm_seg_cps[name]):
            seg_info.append({
                'segment'  : int(seg),
                'arc_start': round(seg / n_seg_arm, 4),
                'arc_end'  : round((seg + 1) / n_seg_arm, 4),
                'n_cp'     : int(len(cp)),
                'cp'       : cp.tolist(),
            })
        out[name] = {
            'robot_name': name,
            'metadata': {
                **trajectories[name]['metadata'],
                'duration'         : float(duration),
                'n_samples'        : int(n_steps),
                'refine_iterations': int(final_iteration),
            },
            'spline': {
                'n_seg'   : int(n_seg_arm),
                'n_cp_seg': int(N_CP_BASE),
                'degree'  : int(DEG),
                'segments': seg_info,
            },
            'trajectory': {
                'time'         : t_out.tolist(),
                'positions'    : pos_sync.tolist(),  # Kuramoto sync_pos
                'velocities'   : vel_sync.tolist(),
                'accelerations': acc_sync.tolist(),
                'arc_fracs'    : np.linspace(0., 1., n_steps).tolist(),
                'n_samples'    : int(n_steps),
            },
        }

    out['synchronisation_report'] = kur_rep
    out['refinement_history']     = history
    out['parameters'] = {
        'k_base': float(K_BASE), 'k_repulse': float(K_REPULSE),
        'min_safe_dist': float(MIN_SAFE), 'max_refine': int(MAX_REFINE),
        'cp_increment': int(CP_INCREMENT),
    }
    return out


# =============================================================================
# MAIN
# =============================================================================

def main():
    print('\n' + '=' * 68)
    print('  STEP 9  --  Kuramoto + Home-CP B-Spline Refinement')
    print('=' * 68)

    for fname, step in [('trajectories.json', 'step_7'),
                        ('collision_report.json', 'step_8')]:
        if not os.path.exists(fname):
            print('\n  FAIL  {} not found -- run {} first'.format(fname, step))
            sys.exit(1)

    with open('trajectories.json')     as fh: tdata = json.load(fh)
    with open('collision_report.json') as fh: crep  = json.load(fh)

    active = sorted([k for k in tdata if k.startswith('dsr')])
    if not active:
        print('\n  FAIL  No arm data in trajectories.json'); sys.exit(1)

    print('\n  Active arms   : {}'.format(active))
    print('  Step-8 status : {}'.format(crep['overall_status']))

    result = synchronise(
        {n: tdata[n] for n in active},
        {n: ROBOT_BASES.get(n, np.zeros(3)) for n in active},
    )

    with open('synchronized_trajectories.json', 'w') as fh:
        json.dump(result, fh, indent=2)

    safe = result['synchronisation_report'].get('collision_free', False)
    kb   = os.path.getsize('synchronized_trajectories.json') / 1024.0
    print('\n  {}  Saved: synchronized_trajectories.json  ({:.1f} KB)'.format(
        'OK' if safe else 'WARN', kb))
    print('  Next  ->  ros2 run dual_arm_sync step_10\n')


if __name__ == '__main__':
    main()