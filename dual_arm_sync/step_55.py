#!/usr/bin/env python3
"""
step_55.py  --  TIER 2: Kuramoto-Centric Home-Pull CP Insertion
================================================================
INPUT  : s54_tier1.json
OUTPUT : s55_tier2.json

PHILOSOPHY (Kabir 2019 inspired, Kuramoto-centric):
  This merges the old Tier 2 + Tier 3 into ONE loop where Kuramoto is
  checked at EVERY trial -- because Kuramoto is the core contribution.

  Mechanism (mirrors Kabir's "increase control points until feasible"):
    1. Keep the original start->target B-spline CPs.
    2. Home pivot per arm = [mid_j1, 0, 0, 0, 0, 0],
         mid_j1 = (start_j1 + target_j1) / 2   (FIXED for whole loop).
       This fixed target gives MONOTONIC convergence -- inserting more CPs
       pulls progressively harder toward one stable point (unlike a moving
       j1-at-collision target).
    3. For n_insert = 1, 2, 3, ... :
         a. Insert n_insert home-valued CPs at the collision arc.
         b. Re-evaluate the B-spline.
         c. Check geometric collision.
         d. *** RUN KURAMOTO on this trajectory (every trial). ***
         e. If geometry + Kuramoto timing together are collision-free -> DONE.
       Collision score may temporarily rise (Kabir's refinement also allows
       this); we keep best-so-far and accept the first fully-clear combo.

  WHY mid_j1 (not j1-at-collision):
    The collision arc shifts each iteration. j1-at-collision would move the
    home target every trial, breaking monotonic convergence. mid_j1 is fixed,
    so "more CPs = harder pull toward the same point", which is what makes the
    Kabir-style convergence argument hold. Kuramoto then does the fine timing.

  COLLISION CHECK uses per-link-pair thresholds (thick links have larger
  thresholds than thin links).
"""
import json, os, sys, time
import numpy as np
from scipy.interpolate import BSpline

sys.path.insert(0, os.path.dirname(__file__))
from _robot5x import (NDOF, POS_LIM, VEL_LIM, ACC_LIM, RATE_HZ, ROBOT_BASES,
                       ARM_NAMES, pair_collides, pair_min_dist, link_origins,
                       LINK_RADII, SAFETY_MARGIN)

DEG            = 3
N_SEG          = 5
N_CP_SEG       = 4
MAX_INSERT     = 8          # up to 8 home-CP insertions
MIN_SAFE       = LINK_RADII.min() * 2 + SAFETY_MARGIN
# Kuramoto params (checked every trial)
K_ATTRACT      = 5.0
K_REPULSE      = 80.0
K_EMERGENCY    = 250.0
KUR_DT         = 0.01
MAX_RESCALE    = 5.0


# ── Home pivot: mid_j1 (fixed for whole loop, per arm) ───────────────────────

def mid_j1_home(start_q, target_q):
    """Home pivot = [mid_j1, 0, 0, 0, 0, 0], mid_j1 = (start_j1+target_j1)/2."""
    home = np.zeros(NDOF)
    home[0] = (float(start_q[0]) + float(target_q[0])) / 2.0
    return home


# ── Per-link-pair collision helpers ──────────────────────────────────────────

def find_deepest_collision_arc(arm_pos_dict, bases, arm_names):
    """Find arc fraction of deepest per-link-pair penetration."""
    K = min(len(arm_pos_dict[n]) for n in arm_names)
    arc = np.linspace(0., 1., K)
    thr = (LINK_RADII[:, None] + LINK_RADII[None, :]) + SAFETY_MARGIN
    worst_pen = 0.; worst_arc = 0.5; total = 0
    for i in range(len(arm_names)):
        for j in range(i+1, len(arm_names)):
            ni, nj = arm_names[i], arm_names[j]
            for k in range(K):
                oi = link_origins(arm_pos_dict[ni][k], bases[ni])
                oj = link_origins(arm_pos_dict[nj][k], bases[nj])
                d_mat = np.linalg.norm(oi[:, None, :] - oj[None, :, :], axis=2)
                pen = float(np.max(thr - d_mat))
                if pen > 0:
                    total += 1
                    if pen > worst_pen:
                        worst_pen = pen; worst_arc = float(arc[k])
    return worst_arc, worst_pen, total


def check_collision_summary(arm_pos_dict, bases, arm_names):
    K = min(len(arm_pos_dict[n]) for n in arm_names)
    reports = {}
    for i in range(len(arm_names)):
        for j in range(i+1, len(arm_names)):
            ni, nj = arm_names[i], arm_names[j]
            nc = 0; md = float('inf')
            for k in range(K):
                d = pair_min_dist(arm_pos_dict[ni][k], bases[ni],
                                   arm_pos_dict[nj][k], bases[nj])
                if d < md: md = d
                if pair_collides(arm_pos_dict[ni][k], bases[ni],
                                  arm_pos_dict[nj][k], bases[nj]):
                    nc += 1
            reports[f'{ni}<->{nj}'] = {'min_dist_m': round(float(md), 4),
                                        'critical_steps': int(nc),
                                        'collision_free': nc == 0}
    return reports


# ── B-spline helpers ─────────────────────────────────────────────────────────

def make_knots(ncp):
    ni  = max(0, ncp - DEG - 1)
    inn = np.linspace(0, 1, ni + 2)[1:-1] if ni > 0 else np.array([])
    return np.concatenate([np.zeros(DEG + 1), inn, np.ones(DEG + 1)])


def eval_spline(cp_flat, duration):
    knots   = make_knots(len(cp_flat))
    n_steps = max(2, int(round(duration * RATE_HZ)))
    s = np.linspace(0., 1., n_steps)
    pos = np.zeros((n_steps, NDOF)); vel = np.zeros((n_steps, NDOF))
    acc = np.zeros((n_steps, NDOF))
    for j in range(NDOF):
        spl = BSpline(knots, cp_flat[:, j], DEG, extrapolate=True)
        pos[:, j] = spl(s)
        vel[:, j] = spl.derivative(1)(s) / duration
        acc[:, j] = spl.derivative(2)(s) / duration**2
    return np.clip(pos, POS_LIM[:, 0], POS_LIM[:, 1]), vel, acc


def scale_duration(cp_flat, duration):
    pos, vel, acc = eval_spline(cp_flat, duration)
    sv = sa = 1.0
    for j in range(NDOF):
        vp = float(np.max(np.abs(vel[:, j]))); ap = float(np.max(np.abs(acc[:, j])))
        if vp > VEL_LIM[j]: sv = max(sv, vp / VEL_LIM[j])
        if ap > ACC_LIM[j]: sa = max(sa, float(np.sqrt(ap / ACC_LIM[j])))
    scale = max(sv, sa)
    if scale > 1.0:
        duration = duration * scale * 1.05
        pos, vel, acc = eval_spline(cp_flat, duration)
    return pos, vel, acc, duration


def insert_home_cps(cp_base, home_q, coll_arc, n_insert):
    """Insert n_insert home-valued CPs near coll_arc. Boundaries preserved."""
    n_base = len(cp_base)
    cp_arcs = np.linspace(0., 1., n_base)
    spreads = {
        1: [0.], 2: [-0.04, +0.04], 3: [-0.05, 0., +0.05],
        4: [-0.08, -0.04, +0.04, +0.08], 5: [-0.08, -0.04, 0., +0.04, +0.08],
        6: [-0.10, -0.06, -0.02, +0.02, +0.06, +0.10],
        7: [-0.10, -0.06, -0.02, 0., +0.02, +0.06, +0.10],
        8: [-0.12, -0.085, -0.05, -0.015, +0.015, +0.05, +0.085, +0.12],
    }
    insert_arcs = np.clip(np.array([coll_arc + o for o in spreads.get(n_insert, [0.])]),
                           0.06, 0.94)
    items = [(a, 'b', cp_base[i]) for i, a in enumerate(cp_arcs)]
    for a in insert_arcs:
        items.append((a, 'h', home_q))
    items.sort(key=lambda x: (x[0], 0 if x[1] == 'b' else 1))
    return np.clip(np.array([it[2] for it in items]), POS_LIM[:, 0], POS_LIM[:, 1])


# ── Kuramoto (checked every trial) ───────────────────────────────────────────

def smooth_positions(pos, window=7):
    if window < 3 or len(pos) < window: return pos
    if window % 2 == 0: window += 1
    half = window // 2; out = pos.copy()
    for k in range(half, len(pos) - half):
        out[k] = np.mean(pos[k-half:k+half+1], axis=0)
    return out


def run_kuramoto(arm_names, arm_pos, bases, duration):
    """
    Run Kuramoto timing on the given (already CP-modified) trajectories.
    Returns (synchronized_pos_dict, collision_free_bool, final_duration,
             min_dist_m).
    """
    N = len(arm_names)
    omega0 = 1.0 / duration
    phi = np.zeros(N); om = np.full(N, omega0)
    pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    max_steps = int(round(4.0 * duration / KUR_DT))

    def interp(pos, frac):
        frac = float(np.clip(frac, 0, 1)); n = len(pos) - 1
        if n <= 0: return pos[0].copy()
        idx = min(int(frac * n), n - 1); a = frac * n - idx
        return pos[idx] + a * (pos[min(idx+1, n)] - pos[idx])

    sync = {n: [] for n in arm_names}
    k = 0
    while k < max_steps:
        phi_c = np.clip(phi, 0., 1.)
        q_now = {n: interp(arm_pos[n], phi_c[idx]) for idx, n in enumerate(arm_names)}
        for idx, n in enumerate(arm_names): sync[n].append(q_now[n].copy())
        if np.all(phi >= 1.0 - 1e-9): break
        dphi = np.zeros(N)
        for (i, j) in pairs:
            d = pair_min_dist(q_now[arm_names[i]], bases[arm_names[i]],
                               q_now[arm_names[j]], bases[arm_names[j]])
            dphi[i] += K_ATTRACT * float(np.sin(phi[j] - phi[i]))
            dphi[j] += K_ATTRACT * float(np.sin(phi[i] - phi[j]))
            if d < MIN_SAFE:
                f_coll = float(np.clip(1 - d / MIN_SAFE, 0, 1))
                mag = K_REPULSE * (f_coll ** 2) * 30
                if d < MIN_SAFE * 0.5:
                    f_d = float(np.clip(1 - d / (MIN_SAFE * 0.5), 0, 1))
                    mag += K_EMERGENCY * (f_d ** 3)
                dphi[i] -= mag * 0.7; dphi[j] -= mag * 0.7
        phi += KUR_DT * np.clip(om + dphi, 0.0, 4.0)
        k += 1
    for n in arm_names: sync[n].append(arm_pos[n][-1].copy())
    sync_arr = {n: np.array(sync[n]) for n in arm_names}

    # Smooth + resample to nominal duration, check feasibility & collision
    real_dur = max((len(sync_arr[arm_names[0]]) - 1) * KUR_DT, duration)
    t_kur = np.linspace(0., real_dur, len(sync_arr[arm_names[0]]))
    n_out = max(2, int(round(duration * RATE_HZ)))
    t_out = np.linspace(0., duration, n_out)
    refined = {}
    for name in arm_names:
        sp = smooth_positions(sync_arr[name], window=7)
        out = np.zeros((n_out, NDOF))
        for j in range(NDOF):
            out[:, j] = np.interp(t_out, t_kur, sp[:, j])
        refined[name] = np.clip(out, POS_LIM[:, 0], POS_LIM[:, 1])

    # Feasibility: check vel/acc within limits (rescale-bounded)
    feasible = True
    dt = duration / max(n_out - 1, 1)
    for name in arm_names:
        vel = np.gradient(refined[name], dt, axis=0)
        acc = np.gradient(vel, dt, axis=0)
        for j in range(NDOF):
            if (np.max(np.abs(vel[:, j])) > VEL_LIM[j] * MAX_RESCALE or
                np.max(np.abs(acc[:, j])) > ACC_LIM[j] * MAX_RESCALE):
                feasible = False; break

    # Collision check on refined trajectory
    K = min(len(refined[n]) for n in arm_names)
    total = 0; md = float('inf')
    for (i, j) in pairs:
        for k in range(K):
            d = pair_min_dist(refined[arm_names[i]][k], bases[arm_names[i]],
                               refined[arm_names[j]][k], bases[arm_names[j]])
            if d < md: md = d
            if pair_collides(refined[arm_names[i]][k], bases[arm_names[i]],
                              refined[arm_names[j]][k], bases[arm_names[j]]):
                total += 1
    return refined, (total == 0 and feasible), duration, md


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print('\n' + '='*66)
    print('  STEP 55  --  TIER 2: Kuramoto-Centric Home-Pull')
    print('='*66)
    if not os.path.exists('s54_tier1.json'):
        print('  s54_tier1.json not found'); sys.exit(1)
    with open('s54_tier1.json') as fh: t1 = json.load(fh)

    if t1.get('tier_resolved', False):
        print('  Tier 1 already resolved -- passing through unchanged.')
        out = dict(t1); out['tier'] = 2
        out['method'] = 'passthrough_from_tier_1'; out['next_tier'] = 'DONE'
        with open('s55_tier2.json', 'w') as fh: json.dump(out, fh, indent=2)
        return

    arm_names = t1.get('arm_names', ARM_NAMES)
    duration  = float(t1.get('duration', 10.0))
    bases     = {n: np.array(ROBOT_BASES.get(n, [0, 0, 0])) for n in arm_names}

    # Reconstruct straight-line base CPs and per-arm mid_j1 home
    arm_cps = {}; arm_meta = {}; arm_home = {}
    for name in arm_names:
        meta = t1[name]['metadata']; arm_meta[name] = meta
        start_q  = np.array(meta['start_joints'], dtype=float)
        target_q = np.array(meta['end_joints'], dtype=float)
        total = N_SEG * (N_CP_SEG - 1) + 1
        s = np.linspace(0., 1., total)
        arm_cps[name] = np.clip(start_q + s[:, None] * (target_q - start_q),
                                 POS_LIM[:, 0], POS_LIM[:, 1])
        arm_home[name] = mid_j1_home(start_q, target_q)

    # Determine colliding arms
    colliding = set()
    for ps, rep in t1.get('synchronisation_report', {}).get('pair_reports', {}).items():
        if not rep.get('collision_free', True):
            a, b = ps.split('<->'); colliding.add(a); colliding.add(b)
    if not colliding: colliding = set(arm_names)
    print(f'\n  Colliding arms: {sorted(colliding)}')
    for name in colliding:
        print(f'  [{name}] mid_j1 home = [{np.degrees(arm_home[name][0]):.1f}, 0,0,0,0,0]')
    print(f'  Kuramoto checked EVERY trial (core of the method)')

    t0 = time.time()
    trial_log = []
    resolved = False
    best = None      # (residual_coll, kind, pos_dict, dur, n_insert)

    for n_insert in range(1, MAX_INSERT + 1):
        # Current trajectory positions for collision-arc detection
        cur_pos = {}
        for name in arm_names:
            pos, _, _, _ = scale_duration(arm_cps[name], duration)
            cur_pos[name] = pos
        worst_arc, worst_pen, total_geo = find_deepest_collision_arc(
            cur_pos, bases, arm_names)

        # Insert n_insert home-CPs at collision arc for colliding arms
        trial_cps = {}
        for name in arm_names:
            if name in colliding:
                trial_cps[name] = insert_home_cps(arm_cps[name], arm_home[name],
                                                    worst_arc, n_insert)
            else:
                trial_cps[name] = arm_cps[name]

        # Evaluate geometry-only trajectories
        max_dur = duration; geo_pos = {}
        for name in arm_names:
            pos, _, _, dur = scale_duration(trial_cps[name], duration)
            geo_pos[name] = pos; max_dur = max(max_dur, dur)
        for name in arm_names:   # sync length
            pos = geo_pos[name]
            nout = max(2, int(round(max_dur * RATE_HZ)))
            if len(pos) != nout:
                sin = np.linspace(0, 1, len(pos)); sout = np.linspace(0, 1, nout)
                geo_pos[name] = np.clip(np.vstack([np.interp(sout, sin, pos[:, j])
                                          for j in range(NDOF)]).T,
                                         POS_LIM[:, 0], POS_LIM[:, 1])

        geo_reports = check_collision_summary(geo_pos, bases, arm_names)
        geo_coll = sum(r['critical_steps'] for r in geo_reports.values())

        # Is geometry-only already collision-free?
        geo_free = (geo_coll == 0)

        # *** RUN KURAMOTO every trial (core of the method) ***
        kur_pos, kur_free, kur_dur, kur_md = run_kuramoto(
            arm_names, geo_pos, bases, max_dur)
        kur_reports = check_collision_summary(kur_pos, bases, arm_names)
        kur_coll = sum(r['critical_steps'] for r in kur_reports.values())

        # Choose the BETTER of geometry-only vs geometry+Kuramoto.
        # Kuramoto can HURT side-swap cases (desync creates new meeting points),
        # so we never blindly take its result -- we compare and keep the best.
        if kur_coll <= geo_coll:
            trial_pos = kur_pos; trial_coll = kur_coll
            trial_dur = kur_dur; trial_tag = 'kuramoto'
        else:
            trial_pos = geo_pos; trial_coll = geo_coll
            trial_dur = max_dur; trial_tag = 'geometry'
        trial_free = (trial_coll == 0)

        trial_log.append({
            'n_insert': n_insert, 'worst_arc': round(worst_arc, 3),
            'worst_pen_cm': round(worst_pen*100, 2),
            'geo_collisions': int(geo_coll),
            'kuramoto_collisions': int(kur_coll),
            'chosen': trial_tag, 'chosen_collisions': int(trial_coll),
            'free': bool(trial_free),
        })
        print(f'  trial {n_insert}: arc={worst_arc:.3f} pen={worst_pen*100:.1f}cm '
              f'geo={geo_coll} kur={kur_coll} -> use {trial_tag}({trial_coll}) '
              f'{"FREE" if trial_free else ""}')

        # Track best across trials
        if best is None or trial_coll < best[0]:
            best = (trial_coll, trial_tag,
                    {n: trial_pos[n].copy() for n in arm_names},
                    trial_dur, n_insert)

        # Advance the base CPs for the next trial (monotonic pull toward mid_j1)
        arm_cps = trial_cps

        if trial_free:
            resolved = True
            final_pos = trial_pos; final_dur = trial_dur
            final_kind = f'home_pull_plus_{trial_tag}'
            final_ninsert = n_insert
            print(f'  ✓ RESOLVED at {n_insert} insertion(s) via {trial_tag}')
            break

    if not resolved:
        # use best-so-far
        best_coll, best_kind, best_pos, best_dur, best_ni = best
        final_pos = best_pos; final_dur = best_dur
        final_kind = f'best_effort_{best_kind}'
        final_ninsert = best_ni
        print(f'\n  Tier-2 exhausted -- best was {best_coll} collisions '
              f'({best_ni} insertions + Kuramoto)')

    t2_ms = round((time.time() - t0) * 1000, 1)
    final_reports = check_collision_summary(final_pos, bases, arm_names)
    final_coll = sum(r['critical_steps'] for r in final_reports.values())
    final_md = min(r['min_dist_m'] for r in final_reports.values())

    out = {
        'tier'          : 2,
        'method'        : 'kuramoto_centric_home_pull',
        'tier_resolved' : resolved,
        'next_tier'     : 'DONE' if resolved else 'tier_3',
        'duration'      : final_dur,
        'arm_names'     : arm_names,
        't2_time_ms'    : t2_ms,
        'insertions_used': final_ninsert,
        'resolution_kind': final_kind,
        'home_targets'  : {n: arm_home[n].tolist() for n in colliding},
        'home_rule'     : 'mid_j1 = (start_j1+target_j1)/2 per arm (j2..j6=0), Kuramoto every trial',
        'trial_log'     : trial_log,
        'pair_reports'  : final_reports,
        'final_verification': {
            'collisions_after_resample': int(final_coll),
            'min_dist_after_resample_m': round(float(final_md), 4),
            'verified_collision_free': resolved,
        },
    }
    for name in arm_names:
        pos = final_pos[name]
        dt  = final_dur / max(len(pos) - 1, 1)
        vel = np.gradient(pos, dt, axis=0); acc = np.gradient(vel, dt, axis=0)
        out[name] = {
            'robot_name': name,
            'metadata'  : {**arm_meta[name], 'duration': float(final_dur),
                            'n_samples': len(pos)},
            'spline'    : {'cp_flat': arm_cps[name].tolist()},
            'trajectory': {
                'time'         : np.linspace(0., final_dur, len(pos)).tolist(),
                'positions'    : pos.tolist(),
                'velocities'   : vel.tolist(),
                'accelerations': acc.tolist(),
                'arc_fracs'    : np.linspace(0., 1., len(pos)).tolist(),
            },
        }

    with open('s55_tier2.json', 'w') as fh: json.dump(out, fh, indent=2)
    if resolved:
        print(f'\n  ✓ TIER 2 RESOLVED  ({final_ninsert} insertions + Kuramoto)')
        print(f'  Saved: s55_tier2.json  --  next: step_57')
    else:
        print(f'\n  TIER 2 PARTIAL  (best: {final_coll} collisions)')
        print(f'  Saved: s55_tier2.json  --  next: step_56')


if __name__ == '__main__': main()