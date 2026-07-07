#!/usr/bin/env python3
"""
step_56.py  --  TIER 3: Deep Home-Pull + Kuramoto Fallback
============================================================
INPUT  : s55_tier2.json
OUTPUT : s56_tier3.json

PHILOSOPHY:
  Tier 2 exhausted its 8 home-CP insertions without fully clearing collisions.
  Tier 3 escalates with two additional mechanisms:

    1. MULTI-ARC INSERTION: Tier 2 inserted CPs around ONE arc fraction
       (the deepest-collision arc). Tier 3 finds the TOP-K collision arcs
       and inserts home-CPs at each, spreading the home-pull across multiple
       collision regions.

    2. KURAMOTO REFINEMENT: After deep CP insertion, if grazing collisions
       remain (worst_pen < 1cm), apply a final Kuramoto pass to time-separate
       them.

  Same per-arm home selection rule as Tier 2 (sign-consistent + far-from-0
  -> mid_j1; otherwise full_home).
"""
import json, os, sys, time
import numpy as np
from scipy.interpolate import BSpline

sys.path.insert(0, os.path.dirname(__file__))
from _robot5x import (NDOF, POS_LIM, VEL_LIM, ACC_LIM, RATE_HZ, ROBOT_BASES,
                       ARM_NAMES, pair_collides, pair_min_dist, link_origins,
                       LINK_RADII, SAFETY_MARGIN)

DEG       = 3
N_SEG     = 5
N_CP_SEG  = 4
MAX_TIER3_ITER = 4         # 4 multi-arc passes
FAR_FROM_0_DEG = 30.0


def home_from_j1_at_arc(arm_pos, coll_arc):
    """Home-pull target [j1_at_collision, 0,0,0,0,0] (see step_55 docs)."""
    K = len(arm_pos)
    k = int(np.clip(coll_arc, 0, 1) * (K - 1))
    home = np.zeros(NDOF)
    home[0] = float(arm_pos[k][0])
    return home, f'j1_at_coll={np.degrees(home[0]):.1f}deg'


def find_top_k_collision_arcs(arm_pos_dict, bases, arm_names, k=3):
    """Find the K arc fractions with worst per-link penetrations."""
    K = min(len(arm_pos_dict[n]) for n in arm_names)
    arc = np.linspace(0., 1., K)
    thr = (LINK_RADII[:, None] + LINK_RADII[None, :]) + SAFETY_MARGIN
    per_step_pen = np.zeros(K)
    for i in range(len(arm_names)):
        for j in range(i+1, len(arm_names)):
            ni, nj = arm_names[i], arm_names[j]
            for kk in range(K):
                oi = link_origins(arm_pos_dict[ni][kk], bases[ni])
                oj = link_origins(arm_pos_dict[nj][kk], bases[nj])
                diff = oi[:, None, :] - oj[None, :, :]
                d_mat = np.linalg.norm(diff, axis=2)
                pen = float(np.max(thr - d_mat))
                if pen > per_step_pen[kk]:
                    per_step_pen[kk] = pen
    # Pick top-K local maxima
    # Simple approach: sort positions by penetration depth, take top-k that
    # are at least 0.1 arc apart to avoid clumping at one cluster
    sorted_idx = np.argsort(-per_step_pen)
    picks = []
    for idx in sorted_idx:
        if per_step_pen[idx] <= 0: break
        a = float(arc[idx])
        if all(abs(a - p[0]) > 0.10 for p in picks):
            picks.append((a, float(per_step_pen[idx])))
            if len(picks) >= k: break
    return picks


def check_collision_summary(arm_pos_dict, bases, arm_names):
    K = min(len(arm_pos_dict[n]) for n in arm_names)
    pair_reports = {}
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
            pair_reports[f'{ni}<->{nj}'] = {
                'min_dist_m': round(float(md), 4),
                'critical_steps': int(nc),
                'collision_free': nc == 0,
            }
    return pair_reports


def make_knots(ncp):
    ni  = max(0, ncp - DEG - 1)
    inn = np.linspace(0, 1, ni + 2)[1:-1] if ni > 0 else np.array([])
    return np.concatenate([np.zeros(DEG + 1), inn, np.ones(DEG + 1)])


def eval_spline(cp_flat, duration):
    knots   = make_knots(len(cp_flat))
    n_steps = max(2, int(round(duration * RATE_HZ)))
    s = np.linspace(0., 1., n_steps)
    pos = np.zeros((n_steps, NDOF))
    vel = np.zeros((n_steps, NDOF))
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
        vp = float(np.max(np.abs(vel[:, j])))
        ap = float(np.max(np.abs(acc[:, j])))
        if vp > VEL_LIM[j]: sv = max(sv, vp / VEL_LIM[j])
        if ap > ACC_LIM[j]: sa = max(sa, float(np.sqrt(ap / ACC_LIM[j])))
    scale = max(sv, sa)
    if scale > 1.0:
        duration = duration * scale * 1.05
        pos, vel, acc = eval_spline(cp_flat, duration)
    return pos, vel, acc, duration


def insert_home_at_multiple_arcs(cp_base, home_q, arcs, n_per_arc):
    """Insert n_per_arc home-CPs at each arc in `arcs`."""
    cp_arcs = list(np.linspace(0., 1., len(cp_base)))
    items = [(a, 'base', cp_base[i]) for i, a in enumerate(cp_arcs)]
    spreads = {1: [0.], 2: [-0.04, +0.04], 3: [-0.05, 0., +0.05],
                4: [-0.07, -0.03, +0.03, +0.07]}
    offsets = spreads.get(n_per_arc, [0.])
    for arc in arcs:
        for o in offsets:
            a = np.clip(arc + o, 0.06, 0.94)
            items.append((a, 'home', home_q))
    items.sort(key=lambda x: (x[0], 0 if x[1] == 'base' else 1))
    cp_combined = np.array([it[2] for it in items])
    return np.clip(cp_combined, POS_LIM[:, 0], POS_LIM[:, 1])


def kuramoto_refinement(arm_pos, bases, arm_names, duration):
    """Quick Kuramoto pass for grazing residuals. Returns (pos_dict, success)."""
    N = len(arm_names)
    omega0 = 1.0 / duration
    phi = np.zeros(N); om = np.full(N, omega0)
    KUR_DT = 0.01
    pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    MIN_SAFE = LINK_RADII.min() * 2 + SAFETY_MARGIN
    K_REPULSE = 60.0
    def interp(pos, frac):
        frac = float(np.clip(frac, 0, 1)); n = len(pos) - 1
        if n <= 0: return pos[0].copy()
        idx = min(int(frac * n), n - 1); a = frac * n - idx
        return pos[idx] + a * (pos[min(idx+1, n)] - pos[idx])
    sync = {n: [] for n in arm_names}
    max_steps = int(round(2.5 * duration / KUR_DT))
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
            if d < MIN_SAFE:
                f_coll = float(np.clip(1 - d / MIN_SAFE, 0, 1))
                mag = K_REPULSE * (f_coll ** 2) * 20
                dphi[i] -= mag * 0.6; dphi[j] -= mag * 0.6
        phi += KUR_DT * np.clip(om + dphi, 0.0, 3.0)
        k += 1
    for n in arm_names: sync[n].append(arm_pos[n][-1].copy())
    sync_arr = {n: np.array(sync[n]) for n in arm_names}
    n_out = max(2, int(round(duration * RATE_HZ)))
    refined = {}
    t_kur = np.linspace(0., max((len(sync_arr[arm_names[0]])-1)*KUR_DT, duration),
                         len(sync_arr[arm_names[0]]))
    t_out = np.linspace(0., duration, n_out)
    for name in arm_names:
        sp = sync_arr[name]
        out = np.zeros((n_out, NDOF))
        for j in range(NDOF):
            out[:, j] = np.interp(t_out, t_kur, sp[:, j])
        refined[name] = np.clip(out, POS_LIM[:, 0], POS_LIM[:, 1])
    K = min(len(refined[n]) for n in arm_names)
    total = sum(1 for (i, j) in pairs for kk in range(K)
                 if pair_collides(refined[arm_names[i]][kk], bases[arm_names[i]],
                                   refined[arm_names[j]][kk], bases[arm_names[j]]))
    return refined, (total == 0)


def main():
    print('\n' + '='*66)
    print('  STEP 56  --  TIER 3: Deep Home-Pull + Kuramoto Fallback')
    print('='*66)
    if not os.path.exists('s55_tier2.json'):
        print('  s55_tier2.json not found'); sys.exit(1)
    with open('s55_tier2.json') as fh: t2 = json.load(fh)

    if t2.get('tier_resolved', False):
        print('  Tier 2 already resolved -- passing through.')
        out = dict(t2)
        out['tier'] = 3; out['method'] = 'passthrough_from_tier_2'
        out['next_tier'] = 'DONE'
        with open('s56_tier3.json', 'w') as fh: json.dump(out, fh, indent=2)
        return

    arm_names = t2['arm_names']
    duration  = float(t2['duration'])
    bases     = {n: np.array(ROBOT_BASES.get(n, [0, 0, 0])) for n in arm_names}
    N_arms    = len(arm_names)

    # Load tier-2's best-so-far CPs and positions
    arm_cps = {n: np.array(t2[n]['spline']['cp_flat']) for n in arm_names}
    arm_pos = {n: np.array(t2[n]['trajectory']['positions'], dtype=float)
                for n in arm_names}
    arm_meta = {n: t2[n]['metadata'] for n in arm_names}

    # Identify colliding arms from tier-2 pair_reports
    colliding_arms = set()
    for pair_str, rep in t2.get('pair_reports', {}).items():
        if not rep.get('collision_free', True):
            ni, nj = pair_str.split('<->')
            colliding_arms.add(ni); colliding_arms.add(nj)
    if not colliding_arms: colliding_arms = set(arm_names)
    print(f'\n  Colliding arms: {sorted(colliding_arms)}')

    # Home target is [j1_at_collision, 0,0,0,0,0] per arm, computed each
    # iteration from j1 at the deepest collision arc (same rule as step_55).
    arm_home = {n: np.zeros(NDOF) for n in colliding_arms}
    print(f'  Home-pull: [j1_at_collision, 0,0,0,0,0] per arm (recomputed each iter)')

    t0 = time.time()
    growth_log = []
    final_resolved = False
    best_state = None     # (best_coll, best_pos, best_cps, best_iter)

    # ── Quick check: is Tier 2's input ALREADY at grazing state? ───────────
    # If yes, skip multi-arc (which would likely make it worse) and go
    # straight to Kuramoto refinement to time-separate the grazing contacts.
    thr_check = (LINK_RADII[:, None] + LINK_RADII[None, :]) + SAFETY_MARGIN
    init_worst_pen = 0.
    K_init = min(len(arm_pos[n]) for n in arm_names)
    for i in range(len(arm_names)):
        for j in range(i+1, len(arm_names)):
            ni, nj = arm_names[i], arm_names[j]
            for k in range(K_init):
                oi = link_origins(arm_pos[ni][k], bases[ni])
                oj = link_origins(arm_pos[nj][k], bases[nj])
                d_mat = np.linalg.norm(oi[:,None,:] - oj[None,:,:], axis=2)
                pen = float(np.max(thr_check - d_mat))
                if pen > init_worst_pen: init_worst_pen = pen

    print(f'\n  Input worst penetration: {init_worst_pen*100:.2f}cm')

    if init_worst_pen < 0.01:
        print(f'  Already grazing -- skipping multi-arc, going to Kuramoto')
        best_state = (None,
                      {n: arm_pos[n].copy() for n in arm_names},
                      {n: arm_cps[n].copy() for n in arm_names},
                      -1)
        # Force exit from multi-arc loop by setting a flag
        skip_multi_arc = True
    else:
        skip_multi_arc = False

    # Multi-arc insertions: try increasing number of arcs each iteration
    for grow_iter in range(MAX_TIER3_ITER if not skip_multi_arc else 0):
        n_arcs = grow_iter + 2   # 2, 3, 4, 5 arc clusters
        n_per_arc = 3            # constant 3 home-CPs per arc cluster
        arc_picks = find_top_k_collision_arcs(arm_pos, bases, arm_names,
                                                k=n_arcs)
        if not arc_picks:
            final_resolved = True
            print(f'  iter {grow_iter}: no penetration found -- resolved')
            break
        print(f'  iter {grow_iter}: {len(arc_picks)} arc cluster(s) '
              f'{[round(a[0],2) for a in arc_picks]}  '
              f'pens {[round(a[1]*100,1) for a in arc_picks]}cm  '
              f'inserting {n_per_arc} home-CPs per arc')

        # Recompute per-arm home from j1 at the deepest collision arc
        deepest_arc = arc_picks[0][0] if arc_picks else 0.5
        for name in colliding_arms:
            h, _ = home_from_j1_at_arc(arm_pos[name], deepest_arc)
            arm_home[name] = h

        # Apply multi-arc insertion BUILDING ON top of tier-2's best CPs.
        new_cps = {}
        for name in arm_names:
            if name in colliding_arms:
                new_cps[name] = insert_home_at_multiple_arcs(
                    arm_cps[name], arm_home[name],
                    [a[0] for a in arc_picks], n_per_arc)
            else:
                new_cps[name] = arm_cps[name]

        # Re-evaluate
        max_dur = duration
        new_pos = {}
        for name in arm_names:
            pos, _, _, dur = scale_duration(new_cps[name], duration)
            new_pos[name] = pos
            max_dur = max(max_dur, dur)
        # Sync length
        for name in arm_names:
            pos = new_pos[name]
            nout = max(2, int(round(max_dur * RATE_HZ)))
            if len(pos) != nout:
                sin = np.linspace(0, 1, len(pos)); sout = np.linspace(0, 1, nout)
                p2 = np.vstack([np.interp(sout, sin, pos[:, j])
                                 for j in range(NDOF)]).T
                new_pos[name] = np.clip(p2, POS_LIM[:, 0], POS_LIM[:, 1])

        pair_reps = check_collision_summary(new_pos, bases, arm_names)
        total_coll = sum(r['critical_steps'] for r in pair_reps.values())
        min_d = min(r['min_dist_m'] for r in pair_reps.values())
        growth_log.append({
            'iter': grow_iter, 'n_arcs': n_arcs, 'n_per_arc': n_per_arc,
            'total_cps': {n: len(new_cps[n]) for n in arm_names},
            'collisions': int(total_coll),
            'min_dist_m': round(float(min_d), 4),
        })
        print(f'    after multi-arc: collisions={total_coll}  min_dist={min_d*100:.1f}cm  '
              f'n_cps={dict((n, len(new_cps[n])) for n in colliding_arms)}')

        if best_state is None or total_coll < best_state[0]:
            best_state = (int(total_coll),
                          {n: new_pos[n].copy() for n in arm_names},
                          {n: new_cps[n].copy() for n in arm_names},
                          grow_iter)

        arm_cps = new_cps
        arm_pos = new_pos
        duration = max_dur

        if total_coll == 0:
            final_resolved = True
            break

    # Use best-so-far if not resolved
    if not final_resolved and best_state is not None:
        best_coll, best_pos, best_cps, best_iter = best_state
        arm_pos = best_pos
        arm_cps = best_cps
        print(f'\n  Multi-arc phase exhausted -- best was {best_coll} '
              f'collisions (at iter {best_iter})')

    # Kuramoto refinement for grazing residuals
    if not final_resolved:
        _, _, total = 0, 0, 0
        thr = (LINK_RADII[:, None] + LINK_RADII[None, :]) + SAFETY_MARGIN
        worst_pen = 0
        for i in range(len(arm_names)):
            for j in range(i+1, len(arm_names)):
                ni, nj = arm_names[i], arm_names[j]
                K = min(len(arm_pos[ni]), len(arm_pos[nj]))
                for k in range(K):
                    oi = link_origins(arm_pos[ni][k], bases[ni])
                    oj = link_origins(arm_pos[nj][k], bases[nj])
                    d_mat = np.linalg.norm(oi[:,None,:] - oj[None,:,:], axis=2)
                    p = float(np.max(thr - d_mat))
                    if p > worst_pen: worst_pen = p
        if worst_pen < 0.01:  # < 1cm = grazing
            print(f'\n  Residual is grazing ({worst_pen*100:.2f}cm) -- '
                  f'trying Kuramoto refinement')
            refined, kur_ok = kuramoto_refinement(arm_pos, bases, arm_names, duration)
            if kur_ok:
                arm_pos = refined
                final_resolved = True
                print(f'  ✓ Kuramoto cleared the residual grazing collisions')
            else:
                print(f'  Kuramoto could not clear residual')
        else:
            print(f'\n  Residual is not grazing ({worst_pen*100:.1f}cm) -- '
                  f'no Kuramoto refinement attempted')

    t3_ms = round((time.time() - t0) * 1000, 1)
    final_pair_reps = check_collision_summary(arm_pos, bases, arm_names)
    final_coll = sum(r['critical_steps'] for r in final_pair_reps.values())
    final_md = min(r['min_dist_m'] for r in final_pair_reps.values())

    out = {
        'tier'          : 3,
        'method'        : 'multi_arc_home_pull_plus_kuramoto',
        'tier_resolved' : final_resolved,
        'next_tier'     : 'DONE' if final_resolved else 'FAIL',
        'duration'      : duration,
        'arm_names'     : arm_names,
        't3_time_ms'    : t3_ms,
        'home_targets'  : {n: arm_home[n].tolist() for n in colliding_arms},
        'growth_log'    : growth_log,
        'pair_reports'  : final_pair_reps,
        'final_verification': {
            'collisions_after_resample': int(final_coll),
            'min_dist_after_resample_m': round(float(final_md), 4),
            'verified_collision_free': final_resolved,
        },
    }
    for name in arm_names:
        pos = arm_pos[name]
        dt  = duration / max(len(pos) - 1, 1)
        vel = np.gradient(pos, dt, axis=0); acc = np.gradient(vel, dt, axis=0)
        out[name] = {
            'robot_name': name,
            'metadata'  : {**arm_meta[name], 'duration': float(duration),
                            'n_samples': len(pos),
                            'tier3_cps_used': len(arm_cps[name])},
            'spline'    : {'cp_flat': arm_cps[name].tolist()},
            'trajectory': {
                'time'         : np.linspace(0., duration, len(pos)).tolist(),
                'positions'    : pos.tolist(),
                'velocities'   : vel.tolist(),
                'accelerations': acc.tolist(),
                'arc_fracs'    : np.linspace(0., 1., len(pos)).tolist(),
            },
        }

    with open('s56_tier3.json', 'w') as fh: json.dump(out, fh, indent=2)
    if final_resolved:
        print(f'\n  ✓ TIER 3 RESOLVED  ({t3_ms}ms)')
    else:
        print(f'\n  TIER 3 EXHAUSTED  --  best was {final_coll} collisions')
    print(f'  Saved: s56_tier3.json  --  next: step_57\n')


if __name__ == '__main__': main()