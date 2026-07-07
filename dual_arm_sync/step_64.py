#!/usr/bin/env python3
"""
step_64.py  --  Surgical CP Resolver (SKAR-N Core)
====================================================
INPUT  : s62_trajectories.json + s63_collision_map.json
OUTPUT : s64_resolved.json

THE CORE ALGORITHM:
  For each colliding pair (ni, nj):
    1. Identify collision segment s_coll from s63
    2. Determine retraction strategy:
         |j1_i(s_coll) - j1_j(s_coll)| > J1_SEP_THRESH
           -> j1-preserved: CP_interior = [j1(t), 0, 0, 0, 0, 0]
         else
           -> full home:    CP_interior = [0, 0, 0, 0, 0, 0]
    3. Replace interior CPs of collision segment with retracted config
       CP count: 4 (iter 1-3) → 6 (iter 4) → 8 (iter 5) -- RESOLVED_CP_N
    4. Rebuild global spline for ni and nj
    5. CASCADE CHECK: re-scan ALL pairs -- not just (ni, nj)
       If new collisions appear, add to queue and continue
    6. Repeat until globally clean or MAX_REFINE exceeded

Paper metrics:
  - resolve_time_ms         (total surgical resolution time)
  - iterations_used         (how many CP-replacement rounds)
  - n_pairs_resolved        (how many pairs were fixed)
  - modification_magnitude  (||CP_new - CP_orig||_F per arm per segment)
  - retraction_strategy     (j1-preserved or full-home per pair)
"""

import json, os, sys, time
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.interpolate import BSpline

sys.path.insert(0, os.path.dirname(__file__))
from _robot import (NDOF, POS_LIM, ROBOT_BASES, ARM_NAMES, LINK_RADII,
                    SAFETY_MARGIN, J1_SEP_THRESH, link_origins, pair_collides)

N_SEG    = 5
N_CP_SEG = 4
DEG      = 3
MAX_REFINE = 9     # iter 0..9 = 10 total iterations
CP_INCREMENT = 2
# Expansion phases:
#   iter 0..4: single collision segment
#   iter 5..7: two segments (coll_seg + previous)
#   iter 8..9: three segments (previous + coll + next)
RATE_HZ  = 100.0


# ── Spline rebuild ────────────────────────────────────────────────────────────

def make_knots(ncp):
    ni  = max(0, ncp - DEG - 1)
    inn = np.linspace(0, 1, ni + 2)[1:-1] if ni > 0 else np.array([])
    return np.concatenate([np.zeros(DEG + 1), inn, np.ones(DEG + 1)])


def eval_from_segs(seg_cps, duration):
    """Flatten per-segment CPs into one global B-spline."""
    all_cp = []
    for seg_idx, (cp, _) in enumerate(seg_cps):
        all_cp.append(cp if seg_idx == 0 else cp[1:])
    cp_global = np.vstack(all_cp)
    knots     = make_knots(len(cp_global))
    n_steps   = max(2, int(round(duration * RATE_HZ)))
    s         = np.linspace(0., 1., n_steps)
    pos = np.zeros((n_steps, NDOF))
    for j in range(NDOF):
        spl = BSpline(knots, cp_global[:, j], DEG, extrapolate=True)
        pos[:, j] = spl(s)
    return np.clip(pos, POS_LIM[:, 0], POS_LIM[:, 1])


def load_seg_cps(traj_data) -> List[Tuple[np.ndarray, np.ndarray]]:
    segs  = traj_data['spline']['segments']
    n_cp  = int(traj_data['spline'].get('n_cp_seg', N_CP_SEG))
    knots = make_knots(n_cp)
    return [(np.array(seg['cp'], dtype=float), knots.copy()) for seg in segs]


def fit_cp_neighbour(pos_full, arc_full, seg, n_seg, ncp):
    """Fit B-spline CPs to existing trajectory for a neighbour segment."""
    s0 = seg / n_seg; s1 = (seg + 1) / n_seg
    mask = (arc_full >= s0 - 1e-9) & (arc_full <= s1 + 1e-9)
    idx  = np.where(mask)[0]
    if len(idx) < 2:
        return None
    pin_s = pos_full[idx[0]].copy(); pin_e = pos_full[idx[-1]].copy()
    s_loc = np.clip((arc_full[idx] - s0) / (s1 - s0), 0., 1.)
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
        A_fr = B[:, 1:-1]; rhs = pos_full[idx, j] - B[:, 0]*ts - B[:, -1]*te
        if A_fr.shape[1] > 0:
            z, *_ = np.linalg.lstsq(A_fr, rhs, rcond=None)
            z = np.where(np.isfinite(z), z, np.linspace(ts, te, len(z)))
            z = np.clip(z, POS_LIM[j, 0], POS_LIM[j, 1])
        else:
            z = np.array([])
        cp[:, j] = np.concatenate([[ts], z if len(z) else [], [te]])
    return cp, knots


# ── Retraction CP builder ─────────────────────────────────────────────────────

def home_target(q_original, strategy):
    """
    Compute the home-direction target for a single CP.

    strategy = 'j1_preserved': q_home = [q[0], 0, 0, 0, 0, 0]
               'full_home'   : q_home = [0, 0, 0, 0, 0, 0]

    The CP will be pulled FROM q_original TOWARD this q_home by factor alpha.
    """
    q_home = np.zeros(NDOF)
    if strategy == 'j1_preserved':
        q_home[0] = q_original[0]
    return q_home


def blend_toward_home(q_original, alpha, strategy):
    """
    Pull q_original toward home by factor alpha in [0, 1].
      alpha = 0   -> no change (return q_original)
      alpha = 1   -> fully at home pose for this strategy
      alpha = 0.5 -> halfway between original and home
    """
    q_h = home_target(q_original, strategy)
    return (1.0 - alpha) * q_original + alpha * q_h


def retract_cp(pos_full, arc_full, coll_seg, n_seg, n_cp, strategy,
                j1_vals=None, pin_last=False, alpha=1.0,
                pin_first=True, pin_last_cp=True):
    """
    Build retracted CPs for ONE segment in the collision zone.

    pin_first / pin_last_cp tell us whether THIS segment's boundary CPs
    are anchored to the original trajectory (preserves continuity with
    a non-retracted neighbour) or freed to blend toward home (when the
    next/prev segment is also being retracted).

    alpha in [0, 1] is the retraction strength. We never use alpha = 1.0
    in the schedule (max is 0.9) so the trajectory keeps some of its
    original shape even at maximum retraction.

    The j1_preserved strategy keeps q[0] (base rotation) at the trajectory
    value and pulls only q[1..5] toward zero. The full_home strategy pulls
    every joint toward zero.

    pin_last (legacy name): special-case for the very last segment of the
    whole trajectory -- forces cp[-1] to the planned target so the arm
    still reaches its goal even when the last segment is retracted.

    Returns (cp, knots) or None.
    """
    s0 = coll_seg / n_seg; s1 = (coll_seg + 1) / n_seg
    mask = (arc_full >= s0 - 1e-9) & (arc_full <= s1 + 1e-9)
    idx  = np.where(mask)[0]
    if len(idx) < 2:
        return None

    # CP arc positions (evenly spaced within this segment)
    cp_arcs = np.linspace(s0, s1, n_cp)
    cp      = np.zeros((n_cp, NDOF), dtype=float)

    for k in range(n_cp):
        # Sample the original trajectory's joint values at this CP's arc
        q_orig = np.array([float(np.interp(cp_arcs[k], arc_full, pos_full[:, j]))
                            for j in range(NDOF)])

        # Decide whether to pin or blend
        is_first_cp = (k == 0)
        is_last_cp  = (k == n_cp - 1)

        if (is_first_cp and pin_first) or (is_last_cp and pin_last_cp):
            cp[k] = q_orig   # anchored to original trajectory
        else:
            cp[k] = blend_toward_home(q_orig, alpha, strategy)

    # Last-segment-of-trajectory hard pin: the planned target must be reached
    if pin_last:
        cp[-1] = pos_full[idx[-1]].copy()

    return cp, make_knots(n_cp)


# Retraction strength schedule -- alpha in [0, 1].
# We never reach 1.0 (always keep some of the original trajectory shape).
SCHEDULE = [
    # (phase, target_segs_pattern, n_cp_per_seg, alpha,  outcome_label)
    # ────────────────────── single-segment phase ──────────────────────
    ('single_seg', [0],             4, 0.40, 'RESOLVED_CP_1'),
    ('single_seg', [0],             4, 0.65, 'RESOLVED_CP_2'),
    ('single_seg', [0],             4, 0.90, 'RESOLVED_CP_3'),
    ('single_seg', [0],             6, 0.90, 'RESOLVED_CP_4'),
    ('single_seg', [0],             8, 0.90, 'RESOLVED_CP_5'),
    # ────────────────────── two-segment phase ─────────────────────────
    ('two_seg',    [-1, 0],         4, 0.65, 'RESOLVED_2SEG_1'),
    ('two_seg',    [-1, 0],         4, 0.90, 'RESOLVED_2SEG_2'),
    ('two_seg',    [-1, 0],         6, 0.90, 'RESOLVED_2SEG_3'),
    # ────────────────────── three-segment phase ───────────────────────
    ('three_seg',  [-1, 0, +1],     4, 0.65, 'RESOLVED_3SEG_1'),
    ('three_seg',  [-1, 0, +1],     4, 0.90, 'RESOLVED_3SEG_2'),
]


def schedule_for_iteration(refine_count, coll_seg, n_seg):
    """
    Returns (target_segs, cp_count, phase, alpha) for the given iteration.

    target_segs : list of absolute segment indices to retract
    cp_count    : number of CPs per retracted segment
    phase       : 'single_seg' / 'two_seg' / 'three_seg'
    alpha       : retraction strength in [0, 1]

    Returns (None, 0, reason, 0.) if edge-segment limits prevent expansion.

    LAST-SEGMENT FAST-PATH:
      When coll_seg == n_seg - 1 (last segment), the last CP is pinned to
      the planned target (pin_last=True), so single-segment retraction has
      severely reduced freedom -- only interior CPs can move. Wasting 5
      iterations on single_seg attempts that mathematically can't fully
      retract is wasteful. We skip directly to the two_seg phase, which
      lets segment s-1 fully retract (no pin_last there) while keeping the
      planned target at the end of segment s.

      iter 0..2 -> two_seg phase with growing alpha
      iter 3..4 -> three_seg phase (or end of schedule if not possible)

      Other collision positions (coll_seg < n_seg - 1) use normal schedule.
    """
    if refine_count >= len(SCHEDULE):
        return (None, 0, 'beyond_schedule', 0.0)

    is_last_seg = (coll_seg == n_seg - 1)
    if is_last_seg:
        # Last-segment fast schedule: jump straight to multi-seg phases
        LAST_SEG_SCHEDULE = [
            ('two_seg',   [-1, 0],     4, 0.65, 'RESOLVED_LASTSEG_1'),
            ('two_seg',   [-1, 0],     4, 0.90, 'RESOLVED_LASTSEG_2'),
            ('two_seg',   [-1, 0],     6, 0.90, 'RESOLVED_LASTSEG_3'),
            ('two_seg',   [-1, 0],     8, 0.90, 'RESOLVED_LASTSEG_4'),
            ('three_seg', [-2, -1, 0], 4, 0.90, 'RESOLVED_LASTSEG_5'),  # only if s>=2
        ]
        if refine_count >= len(LAST_SEG_SCHEDULE):
            return (None, 0, 'beyond_last_seg_schedule', 0.0)
        phase, offsets, n_cp, alpha, _label = LAST_SEG_SCHEDULE[refine_count]
    else:
        phase, offsets, n_cp, alpha, _label = SCHEDULE[refine_count]

    target_segs = [coll_seg + o for o in offsets]

    # Edge-segment validity check
    if phase == 'two_seg' and target_segs[0] < 0:
        return (None, 0, 'edge_seg_no_previous', 0.0)
    if phase == 'three_seg' and (target_segs[0] < 0 or target_segs[-1] >= n_seg):
        return (None, 0, 'edge_seg_no_neighbours', 0.0)

    # Final safety check
    if any(s < 0 or s >= n_seg for s in target_segs):
        return (None, 0, 'edge_seg_no_neighbours', 0.0)

    return (target_segs, n_cp, phase, alpha)


def rebuild_spline_for_arm(orig_segs, pos_full, arc_full, coll_seg,
                            n_seg, refine_count, strategy, j1_vals=None):
    """
    Pull-toward-home retraction with adjustable strength alpha.

    Pinning rules (outermost-boundaries only):
      - The very FIRST CP of the FIRST target segment is pinned (entry into
        the retracted zone from the previous, untouched segment).
      - The very LAST CP of the LAST target segment is pinned (exit from
        the retracted zone into the next, untouched segment).
      - All OTHER CPs -- including shared boundaries BETWEEN consecutive
        target segments -- are FREE to blend toward home by factor alpha.

    Example for two_seg expansion (coll_seg = 2):
      target_segs = [1, 2]
      seg 1 CPs: [s1cp0 PINNED] [s1cp1 BLEND] [s1cp2 BLEND] [s1cp3 BLEND]
      seg 2 CPs:                [s2cp0 BLEND] [s2cp1 BLEND] [s2cp2 BLEND] [s2cp3 PINNED]
      (Note: s1cp3 and s2cp0 are the SAME spline knot. By making both BLEND
       with the same source value, they stay coherent and BOTH pull toward home.)

    The entry-side neighbour segment is re-fit so its end matches s1cp0
    (continuity); the exit-side neighbour is untouched (already starts at
    the pinned s_last_cp).

    Special case: if the last target segment is the trajectory's last segment,
    we force its final CP to the planned target (pin_last=True), regardless
    of blending. This ensures the arm always reaches its goal.

    Returns (new_segs, phase, alpha) or (None, reason, 0.0) on giveup.
    """
    target_segs, n_cp_coll, phase, alpha = schedule_for_iteration(
        refine_count, coll_seg, n_seg)
    if target_segs is None:
        return None, phase, 0.0

    result         = list(orig_segs)
    sorted_targets = sorted(target_segs)
    first_target   = sorted_targets[0]
    last_target    = sorted_targets[-1]
    is_at_traj_end = (last_target == n_seg - 1)

    # Retract each segment in the zone
    for seg in sorted_targets:
        # Pin THIS segment's first CP only if it is the very first segment
        # of the zone (it shares with the untouched previous segment).
        pin_first = (seg == first_target)
        # Pin THIS segment's last CP only if it is the very last segment
        # of the zone (it shares with the untouched next segment).
        pin_last_cp = (seg == last_target)
        # Hard pin to planned target if this is the trajectory's last segment
        pin_traj_end = (seg == last_target and is_at_traj_end)

        ret = retract_cp(pos_full, arc_full, seg, n_seg,
                          n_cp_coll, strategy, j1_vals,
                          pin_last=pin_traj_end,
                          alpha=alpha,
                          pin_first=pin_first,
                          pin_last_cp=pin_last_cp)
        if ret is not None:
            result[seg] = ret

    # Smooth the entry-side neighbour segment so it ends at the (now-blended)
    # first CP of the retracted zone. The fit_cp_neighbour function re-fits
    # this neighbour to its own trajectory positions while we trust the
    # retracted zone's first CP to provide the link.
    entry_seg = first_target - 1
    if entry_seg >= 0 and entry_seg not in target_segs:
        fit = fit_cp_neighbour(pos_full, arc_full, entry_seg, n_seg, N_CP_SEG)
        if fit is not None:
            result[entry_seg] = fit

    return result, phase, alpha


# ── Cascade collision check ───────────────────────────────────────────────────

def find_all_collisions(arm_names, arm_pos, bases, n_seg):
    """Return dict of colliding pairs with first_seg."""
    collisions = {}
    N = len(arm_names)
    for i in range(N):
        for j in range(i + 1, N):
            ni, nj = arm_names[i], arm_names[j]
            pi, pj = arm_pos[ni], arm_pos[nj]
            K = min(len(pi), len(pj))
            arcs = np.linspace(0., 1., K)
            first_k = -1
            for k in range(K):
                if pair_collides(pi[k], bases[ni], pj[k], bases[nj]):
                    first_k = k; break
            if first_k >= 0:
                frac = float(arcs[first_k])
                seg  = min(int(frac * n_seg), n_seg - 1)
                collisions[(ni, nj)] = {'first_k': first_k, 'first_frac': frac,
                                         'first_seg': seg}
    return collisions


# ── Main resolver ─────────────────────────────────────────────────────────────

def resolve(tdata, cmap) -> Dict:
    arm_names = tdata.get('arm_names', ARM_NAMES)
    n_seg     = int(tdata[arm_names[0]]['spline']['n_seg'])
    duration  = float(tdata['duration'])
    bases     = {n: np.array(ROBOT_BASES.get(n, [0, 0, 0])) for n in arm_names}

    arm_seg_cps = {n: load_seg_cps(tdata[n]) for n in arm_names}
    arm_pos     = {n: eval_from_segs(arm_seg_cps[n], duration) for n in arm_names}
    arm_arc     = {n: np.linspace(0., 1., len(arm_pos[n])) for n in arm_names}

    # Record original CPs for modification magnitude metric
    orig_cps    = {n: [cp.copy() for cp, _ in arm_seg_cps[n]] for n in arm_names}

    history = []; retraction_log = {}; iteration = 0
    iteration_snapshots = []
    # Per-pair counter for tracking schedule progression independently of
    # global iteration count. When collision moves between last and non-last
    # segments, we restart the per-pair counter so the schedule starts fresh.
    pair_phase_counter = {}     # (ni, nj) -> int (counter for current phase)
    pair_last_kind = {}         # (ni, nj) -> 'last_seg' or 'normal'
    # Snapshot iter 0 = BEFORE any modification (original B-spline)
    iteration_snapshots.append({
        'iteration': -1,
        'label': 'original',
        'positions': {n: arm_pos[n].tolist() for n in arm_names},
        'collisions_found': None,
        'first_coll_segs': {},
    })
    t0 = time.time()

    while iteration <= MAX_REFINE:
        coll = find_all_collisions(arm_names, arm_pos, bases, n_seg)
        if not coll:
            iteration_snapshots.append({
                'iteration': iteration,
                'label': f'resolved_at_iter_{iteration}',
                'positions': {n: arm_pos[n].tolist() for n in arm_names},
                'collisions_found': 0, 'first_coll_segs': {},
            })
            print(f'  Globally collision-free at iteration {iteration}'); break
        if iteration == MAX_REFINE:
            iteration_snapshots.append({
                'iteration': iteration,
                'label': f'unresolved_at_max_refine',
                'positions': {n: arm_pos[n].tolist() for n in arm_names},
                'collisions_found': len(coll),
                'first_coll_segs': {f'{k[0]}<->{k[1]}': v['first_seg'] for k, v in coll.items()},
            })
            print(f'  WARNING: MAX_REFINE={MAX_REFINE} reached -- {len(coll)} pairs unresolved')
            break

        # Process pairs ordered by earliest first collision
        edge_giveup_pairs = []   # collected if any pair hits edge-segment giveup

        for pair_key in sorted(coll.keys(), key=lambda p: coll[p]['first_frac']):
            ni, nj = pair_key
            cs = coll[pair_key]['first_seg']

            # Determine retraction strategy (j1-preserved vs full home)
            arc_mid = (cs + 0.5) / n_seg
            j1_i = float(np.interp(arc_mid, arm_arc[ni], arm_pos[ni][:, 0]))
            j1_j = float(np.interp(arc_mid, arm_arc[nj], arm_pos[nj][:, 0]))
            dj1  = abs(j1_i - j1_j)
            strategy_i = strategy_j = ('j1_preserved' if dj1 > J1_SEP_THRESH
                                                       else 'full_home')

            # Track per-pair phase counter (resets when collision moves between
            # last-segment and non-last-segment positions).
            current_kind = 'last_seg' if cs == n_seg - 1 else 'normal'
            prev_kind = pair_last_kind.get(pair_key)
            if prev_kind is None or prev_kind != current_kind:
                pair_phase_counter[pair_key] = 0   # reset on transition
                pair_last_kind[pair_key] = current_kind
            pair_iter = pair_phase_counter[pair_key]
            pair_phase_counter[pair_key] += 1

            # Look up schedule with the PER-PAIR counter
            target_segs, cp_count, phase, alpha = schedule_for_iteration(pair_iter, cs, n_seg)

            pair_str = f'{ni}<->{nj}'

            if target_segs is None:
                # GIVE UP CLEANLY: collision is at seg 0 or last seg,
                # and we've hit the expansion phase that needs neighbours.
                print(f'  [{pair_str}] iter={iteration}  seg={cs}  '
                      f'GIVEUP_EDGE_SEG ({phase}) -- cannot expand from edge')
                retraction_log[pair_str] = {
                    'iteration': iteration, 'coll_seg': cs,
                    'delta_j1_rad': round(dj1, 4),
                    'strategy': strategy_i, 'phase': phase,
                    'giveup_reason': 'edge_segment_cannot_expand',
                }
                edge_giveup_pairs.append(pair_str)
                continue

            retraction_log[pair_str] = {
                'iteration': iteration, 'coll_seg': cs,
                'delta_j1_rad': round(dj1, 4),
                'strategy': strategy_i,
                'phase': phase,
                'target_segs': target_segs,
                'cp_count': cp_count,
                'alpha': round(alpha, 3),
            }
            print(f'  [{pair_str}] iter={iteration}(pair={pair_iter})  seg={cs}  '
                  f'phase={phase}  segs={target_segs}  cp={cp_count}  alpha={alpha:.2f}  '
                  f'strategy={strategy_i}  dj1={np.degrees(dj1):.1f}deg')

            # Rebuild both arms with the same schedule
            for nm, strat in [(ni, strategy_i), (nj, strategy_j)]:
                new_segs, _phase, _alpha = rebuild_spline_for_arm(
                    arm_seg_cps[nm], arm_pos[nm], arm_arc[nm],
                    cs, n_seg, iteration, strat, arm_pos[nm][:, 0])
                if new_segs is not None:
                    arm_seg_cps[nm] = new_segs
                    arm_pos[nm]     = eval_from_segs(arm_seg_cps[nm], duration)
                    arm_arc[nm]     = np.linspace(0., 1., len(arm_pos[nm]))

        # If every colliding pair gave up on edge, stop iterating
        if edge_giveup_pairs and len(edge_giveup_pairs) == len(coll):
            print(f'  All colliding pairs at edge segments -- stopping early')
            break

        # Snapshot after iteration's CP modifications.
        first_pair = sorted(coll.keys(), key=lambda p: coll[p]['first_frac'])[0]
        first_cs   = coll[first_pair]['first_seg']
        _, snap_cp, snap_phase, snap_alpha = schedule_for_iteration(iteration, first_cs, n_seg)
        iteration_snapshots.append({
            'iteration': iteration,
            'label': f'iter_{iteration}_{snap_phase}_a{int(snap_alpha*100):03d}',
            'positions': {n: arm_pos[n].tolist() for n in arm_names},
            'collisions_found': len(coll),
            'first_coll_segs': {f'{k[0]}<->{k[1]}': v['first_seg'] for k, v in coll.items()},
            'cp_count': snap_cp,
            'phase': snap_phase,
            'alpha': round(snap_alpha, 3),
        })
        history.append({'iteration': iteration, 'n_coll_pairs': len(coll)})
        iteration += 1

    resolve_ms = round((time.time() - t0) * 1000, 1)

    # Modification magnitude: mean position deviation between original and resolved
    # trajectories. We can't compare raw CPs directly because CP count grows with
    # iterations (4 -> 6 -> 8). Comparing the EVALUATED positions is exact and
    # shape-agnostic -- it captures the actual trajectory deviation in rad.
    mod_mag = {}
    orig_pos = {n: eval_from_segs([(cp, None) for cp in orig_cps[n]], duration)
                for n in arm_names}
    for name in arm_names:
        new_pos = arm_pos[name]
        old_pos = orig_pos[name]
        K = min(len(new_pos), len(old_pos))
        mean_dev = float(np.mean(np.linalg.norm(new_pos[:K] - old_pos[:K], axis=1)))
        mod_mag[name] = round(mean_dev, 6)

    # Build output trajectories in same format as s62
    out = {'duration': duration, 'arm_names': arm_names,
           'resolve_time_ms': resolve_ms,
           'iterations_used': iteration,
           'retraction_log': retraction_log,
           'modification_magnitude': mod_mag,
           'refinement_history': history,
           'iteration_snapshots': iteration_snapshots}

    for name in arm_names:
        pos = arm_pos[name]
        arc = arm_arc[name]
        dt  = duration / max(len(pos) - 1, 1)
        vel = np.gradient(pos, dt, axis=0)
        acc = np.gradient(vel,  dt, axis=0)
        t_v = np.linspace(0., duration, len(pos))

        seg_info = []
        for seg, (cp, _) in enumerate(arm_seg_cps[name]):
            orig_cp = orig_cps[name][seg]
            # Shape may differ (CP count grew). Compare via evaluated positions
            # in the segment's arc range instead.
            n_arc = arm_pos[name].shape[0]
            arc   = np.linspace(0., 1., n_arc)
            s0, s1 = seg / n_seg, (seg + 1) / n_seg
            mask  = (arc >= s0 - 1e-9) & (arc <= s1 + 1e-9)
            if mask.any() and orig_pos[name].shape[0] >= n_arc:
                seg_mod = float(np.mean(np.linalg.norm(
                    arm_pos[name][mask] - orig_pos[name][:n_arc][mask], axis=1)))
            else:
                seg_mod = 0.
            seg_info.append({
                'segment'   : seg,
                'arc_start' : round(s0, 4),
                'arc_end'   : round(s1, 4),
                'cp'        : cp.tolist(),
                'n_cp'      : int(len(cp)),
                'cp_orig'   : orig_cp.tolist(),
                'mod_mag'   : round(seg_mod, 6),
                'modified'  : seg_mod > 1e-9,
            })

        out[name] = {
            'robot_name': name,
            'metadata': {
                **tdata[name]['metadata'],
                'resolve_iterations': iteration,
                'modification_magnitude': mod_mag[name],
            },
            'spline': {'n_seg': n_seg, 'segments': seg_info},
            'trajectory': {
                'time'         : t_v.tolist(),
                'positions'    : pos.tolist(),
                'velocities'   : vel.tolist(),
                'accelerations': acc.tolist(),
                'arc_fracs'    : arc.tolist(),
            },
        }
    return out


def main():
    print('\n' + '='*66)
    print('  STEP 64  --  Surgical CP Resolver (SKAR-N Core)')
    print('='*66)

    for fname, step in [('s62_trajectories.json', 'step_62'),
                        ('s63_collision_map.json', 'step_63')]:
        if not os.path.exists(fname):
            print(f'  {fname} not found -- run {step} first'); sys.exit(1)

    with open('s62_trajectories.json') as fh: tdata = json.load(fh)
    with open('s63_collision_map.json') as fh: cmap  = json.load(fh)

    if cmap['overall_status'] == 'SAFE':
        print('  No collision -- copying s62 directly to s64')
        import shutil; shutil.copy('s62_trajectories.json', 's64_resolved.json')
        print('  Saved: s64_resolved.json'); print('  Next : step_65\n'); return

    print(f'\n  Status: {cmap["overall_status"]}  '
          f'{cmap["n_colliding_pairs"]}/{cmap["n_pairs"]} pairs colliding')

    out = resolve(tdata, cmap)
    with open('s64_resolved.json', 'w') as fh: json.dump(out, fh, indent=2)

    kb = os.path.getsize('s64_resolved.json') / 1024.
    print(f'\n  Resolve time : {out["resolve_time_ms"]:.0f} ms')
    print(f'  Iterations   : {out["iterations_used"]}')
    print(f'  Mod magnitude: {out["modification_magnitude"]}')
    for pair, info in out['retraction_log'].items():
        if 'giveup_reason' in info:
            print(f'  [{pair}] giveup={info["giveup_reason"]}  iter={info["iteration"]}')
        else:
            print(f'  [{pair}] strategy={info["strategy"]}  '
                  f'phase={info.get("phase","?")}  cp={info.get("cp_count","?")}  '
                  f'alpha={info.get("alpha","?")}  '
                  f'segs={info.get("target_segs","?")}')
    print(f'  Saved: s64_resolved.json ({kb:.0f} KB)')
    print(f'  Next : ros2 run dual_arm_sync step_65\n')


if __name__ == '__main__': main()