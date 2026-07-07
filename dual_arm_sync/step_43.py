#!/usr/bin/env python3
"""
step_34.py  --  Temporal Resolver: Path-Velocity Coordination   [DUAL ARM]
=========================================================================
INPUT  : s41_trajectories.json   (the raw B-spline seed -- NOT spatially resolved)
OUTPUT : s43_kuramoto.json

Temporal-only pipeline (step_31..35): there is NO B-spline retraction stage.
Collisions are resolved by re-timing ALONE -- each arm's joint PATH is left exactly
as the seed produced it; only WHEN each arm is at each point of its fixed path
changes.

DURATION IS NOW TIME-OPTIMAL. The seed's base duration (e.g. 10 s) is only a
starting guess. Since vel ~ 1/T and acc ~ 1/T^2, the fraction of the vel/acc
limits used scales as 1/T, so one Newton-style step T *= ratio lands the
trajectory exactly on the limit. We drive T to the SHORTEST value the arms can
achieve within limits (times a safety factor), then the phase lag rides on top:
  final_duration ~= (shortest feasible traversal) x (1 + phase_lag)
This holds for BOTH a colliding move (Kuramoto lag) and a collision-free move
(passthrough) -- a 10 s base the arms could do in 5-6 s is shrunk, and the old
4-9x duration bloat is gone. Uniform time-scaling preserves the arc phase offset,
so collision-freedom is unchanged (and re-verified by scan_all).

Speed is tunable without editing code:
  DUAL_ARM_SPEED_SAFETY  run at 1/this of the vel/acc limits (default 1.25 ~= 80%)
  DUAL_ARM_CLEARANCE_M   collision clearance (default 0.05 = 5 cm)
  DUAL_ARM_NEAR_M        near-miss band     (default 0.10 = 10 cm)
"""
import json, os, sys, time
import numpy as np
from typing import Dict

sys.path.insert(0, os.path.dirname(__file__))
from _robot2x import (NDOF, POS_LIM, VEL_LIM, ACC_LIM, ROBOT_BASES, ARM_NAMES,
                       pair_min_dist)

CLEAR_M = float(os.environ.get('DUAL_ARM_CLEARANCE_M', '0.05'))   # hard stop / collision
NEAR_M  = float(os.environ.get('DUAL_ARM_NEAR_M', '0.10'))        # near-miss warning band
# Run at 1/SAFETY of the vel/acc limits. 1.25 -> ~80% of limits (smooth, safe).
# Lower (e.g. 1.1) = faster/closer to limits; higher (1.5) = gentler/slower.
SAFETY  = float(os.environ.get('DUAL_ARM_SPEED_SAFETY', '1.25'))
# The lag schedule has an acceleration corner where an arm switches wait->move; that
# single-timestep spike must NOT be allowed to re-inflate the whole makespan. Build
# the lag at the time-optimal BASE duration and cap any residual stretch here.
MAX_LAG_STRETCH = float(os.environ.get('DUAL_ARM_MAX_LAG_STRETCH', '1.5'))
MIN_DUR         = float(os.environ.get('DUAL_ARM_MIN_DURATION', '0.0'))  # optional floor


def _collide(qi, bi, qj, bj):
    """Collision = arms closer than CLEAR_M (real surface clearance)."""
    return pair_min_dist(qi, bi, qj, bj) < CLEAR_M

KUR_DT          = 0.01   # time step used when materialising a constant-lag schedule
RATE_HZ         = 100.0
MAX_RESCALE     = 20.0   # a valid timing schedule is accepted even if it needs much
                         # more time (collision-free is prioritised over makespan)
COORD_RES       = 220    # coordination grid resolution per arm (sA, sB in [0,1])
COORD_DILATE    = 1      # grow the collision region by this many cells for margin
MAX_OVERHEAD    = 0.50   # if Kuramoto needs >50% extra time and input already safe,
                         # prefer the seed trajectory (0 overhead)


def interp_pos(pos, frac):
    frac = float(np.clip(frac, 0, 1)); n = len(pos) - 1
    if n <= 0: return pos[0].copy()
    idx = min(int(frac * n), n - 1); a = frac * n - idx
    return pos[idx] + a * (pos[min(idx+1, n)] - pos[idx])


def scan_all(arm_names, arm_pos, bases):
    """Definitive collision scan -- uses _collide (real clearance, CLEAR_M)."""
    coll = 0; mind = float('inf')
    for i in range(len(arm_names)):
        for j in range(i+1, len(arm_names)):
            ni, nj = arm_names[i], arm_names[j]
            pi = arm_pos[ni]; pj = arm_pos[nj]; K = min(len(pi), len(pj))
            for k in range(K):
                d = pair_min_dist(pi[k], bases[ni], pj[k], bases[nj])
                if d < mind: mind = d
                if _collide(pi[k], bases[ni], pj[k], bases[nj]): coll += 1
    return coll, (mind if mind != float('inf') else 0.0)


def _dilate(o, d):
    out = o.copy()
    for _ in range(max(0, d)):
        cur = out.copy()
        cur[1:, :] |= out[:-1, :]; cur[:-1, :] |= out[1:, :]
        cur[:, 1:] |= out[:, :-1]; cur[:, :-1] |= out[:, 1:]
        out = cur
    return out


def _coord_path(obst, G, allow_retreat=False):
    """Shortest path (sA,sB): (0,0)->(G-1,G-1) avoiding obstacle cells.
    allow_retreat=False -> MONOTONE (each arm only moves forward / waits): pure lag.
    allow_retreat=True  -> 8-connected: an arm may briefly back off ALONG ITS OWN
    PATH to let the other pass, then resume. Returns the cell list, or None if start
    and goal are disconnected by the obstacle (a true SPATIAL wall)."""
    if obst[0, 0] or obst[G - 1, G - 1]:
        return None
    import heapq
    if allow_retreat:
        moves = [(1, 1, 1.0), (1, 0, 1.0), (0, 1, 1.0),
                 (-1, 0, 3.0), (0, -1, 3.0), (-1, -1, 3.0),
                 (1, -1, 2.0), (-1, 1, 2.0)]
    else:
        moves = [(1, 1, 1.0), (1, 0, 1.0), (0, 1, 1.0)]
    INF = float('inf')
    dist = np.full((G, G), INF); dist[0, 0] = 0.0
    parent = {}; pq = [(0.0, 0, 0)]
    while pq:
        d0, ia, ib = heapq.heappop(pq)
        if d0 > dist[ia, ib]: continue
        if ia == G - 1 and ib == G - 1: break
        for da, db, w in moves:
            na, nb = ia + da, ib + db
            if 0 <= na < G and 0 <= nb < G and not obst[na, nb]:
                nd = d0 + w
                if nd < dist[na, nb]:
                    dist[na, nb] = nd; parent[(na, nb)] = (ia, ib)
                    heapq.heappush(pq, (nd, na, nb))
    if dist[G - 1, G - 1] == INF:
        return None
    path = [(G - 1, G - 1)]
    while path[-1] != (0, 0):
        path.append(parent[path[-1]])
    path.reverse()
    return path


def _pair_report(A, B, syncA, syncB, ba, bb):
    mn = float('inf'); nc = 0; nn = 0
    for k in range(min(len(syncA), len(syncB))):
        d = pair_min_dist(syncA[k], ba, syncB[k], bb)
        if d < mn: mn = d
        if _collide(syncA[k], ba, syncB[k], bb): nc += 1
        elif d < NEAR_M: nn += 1
    return mn, nc, nn


def _build_lag(arm_pos, A, B, lead, delta, duration):
    """Both arms traverse their FULL path s:0->1; the non-leader is delayed by a
    constant phase 'delta' (waits at start, holds at target). Returns positions + t."""
    span = 1.0 + delta
    n = max(2, int(round(span * duration / KUR_DT)))
    t = np.linspace(0., span * duration, n)
    off = {A: (0.0 if lead == A else delta), B: (0.0 if lead == B else delta)}
    ap = {}
    for nm in (A, B):
        ph = np.clip(t / duration - off[nm], 0.0, 1.0)
        ap[nm] = np.array([interp_pos(arm_pos[nm], f) for f in ph])
    return ap, t


def run_enhanced_kuramoto(arm_names, arm_pos, bases, duration):
    """Timing-only resolver for two arms. Both arms keep their EXACT paths and run
    s:0->1; only the relative timing changes. Stage 1 walks every constant PHASE LAG
    (either arm leading) and takes the smallest collision-free one. Stage 2, if no
    constant lag clears it, searches the full coordination diagram (monotone, then a
    short retreat). On failure it reports the best lag and where the residual sits."""
    if len(arm_names) != 2:
        raise ValueError('step_34 coordination is for the dual-arm case')
    A, B = arm_names; ba, bb = bases[A], bases[B]
    G = COORD_RES; sg = np.linspace(0., 1., G)
    qA = [interp_pos(arm_pos[A], s) for s in sg]
    qB = [interp_pos(arm_pos[B], s) for s in sg]
    obst = np.zeros((G, G), bool)
    for ia in range(G):
        for ib in range(G):
            if _collide(qA[ia], ba, qB[ib], bb): obst[ia, ib] = True

    # ---- Stage 1: constant phase lag = shifted diagonal on the grid ----
    def diag(lead, dc):
        steps = (G - 1) + dc; coll = 0; fa = None; la = None
        for k in range(steps + 1):
            iL = min(k, G - 1); iLag = min(max(k - dc, 0), G - 1)
            ia, ib = (iLag, iL) if lead == B else (iL, iLag)
            if obst[ia, ib]:
                coll += 1; la = k
                if fa is None: fa = k
        return coll, fa, la, steps

    best = None
    for dc in range(1, G):
        for lead in (B, A):
            coll, fa, la, steps = diag(lead, dc)
            if best is None or coll < best['coll']:
                delta = dc / (G - 1)
                best = {'coll': coll, 'lead': lead, 'delta': delta,
                        'arc_lo': (fa / steps if fa is not None else None),
                        'arc_hi': (la / steps if la is not None else None)}
            if coll == 0:
                delta = dc / (G - 1)
                ap, t = _build_lag(arm_pos, A, B, lead, float(delta), duration)
                mn, nc, nn = _pair_report(A, B, ap[A], ap[B], ba, bb)
                if nc == 0:
                    rep = {f'{A}<->{B}': {'min_dist_m': round(float(mn), 4), 'critical_steps': 0,
                                          'nearmiss_steps': nn, 'collision_free': True}}
                    return ap, t, {'pair_reports': rep, 'total_critical': 0, 'total_nearmiss': nn,
                                   'collision_free': True, 'near_miss_margin_m': NEAR_M,
                                   'method': 'constant_phase_lag', 'timing_separable': True,
                                   'lag_fraction': float(delta),
                                   'schedule': f'{lead}_leads_by_{delta*duration:.1f}s'}

    # ---- Stage 2: full coordination (non-constant schedule) ----
    obst_d = _dilate(obst, COORD_DILATE)
    path = _coord_path(obst_d, G, allow_retreat=False); sched = 'monotone_lag'
    if path is None:
        path = _coord_path(obst_d, G, allow_retreat=True); sched = 'lag_with_retreat'

    if path is None:
        mn, nc, _ = _pair_report(A, B, arm_pos[A], arm_pos[B], ba, bb)
        bl = best
        diag_txt = (f"best lag: {bl['lead']} leads by {bl['delta']*duration:.1f}s "
                    f"-> {bl['coll']} residual collisions"
                    + (f" near arc {bl['arc_lo']:.2f}-{bl['arc_hi']:.2f}"
                       if bl['arc_lo'] is not None else ''))
        rep = {f'{A}<->{B}': {'min_dist_m': round(float(mn), 4), 'critical_steps': nc,
                              'nearmiss_steps': 0, 'collision_free': False}}
        return ({A: arm_pos[A], B: arm_pos[B]},
                np.linspace(0., duration, min(len(arm_pos[A]), len(arm_pos[B]))),
                {'pair_reports': rep, 'total_critical': max(nc, 1), 'total_nearmiss': 0,
                 'collision_free': False, 'near_miss_margin_m': NEAR_M,
                 'method': 'path_velocity_coordination', 'timing_separable': False,
                 'schedule': 'none', 'lag_fraction': 0.0, 'best_lag_diag': diag_txt})

    saA = np.array([sg[c[0]] for c in path]); saB = np.array([sg[c[1]] for c in path])
    syncA = np.array([interp_pos(arm_pos[A], s) for s in saA])
    syncB = np.array([interp_pos(arm_pos[B], s) for s in saB])
    mn, nc, nn = _pair_report(A, B, syncA, syncB, ba, bb)
    rep = {f'{A}<->{B}': {'min_dist_m': round(float(mn), 4), 'critical_steps': nc,
                          'nearmiss_steps': nn, 'collision_free': nc == 0}}
    t_vec = np.linspace(0., duration, len(path))
    return {A: syncA, B: syncB}, t_vec, {'pair_reports': rep, 'total_critical': nc,
            'total_nearmiss': nn, 'collision_free': nc == 0,
            'near_miss_margin_m': NEAR_M, 'schedule': sched, 'lag_fraction': 0.0,
            'method': 'path_velocity_coordination', 'timing_separable': True}


def resample(sync_pos, t_kur, duration):
    n_out = max(2, int(round(duration * RATE_HZ))); t_out = np.linspace(0., duration, n_out)
    t_in = t_kur if len(t_kur) == len(sync_pos) else np.linspace(0., duration, len(sync_pos))
    out = np.zeros((n_out, NDOF))
    for j in range(NDOF): out[:, j] = np.interp(t_out, t_in, sync_pos[:, j])
    return np.clip(out, POS_LIM[:, 0], POS_LIM[:, 1])


def speed_ratio(pos, duration):
    """Peak fraction of the vel/acc limits used over the trajectory.
    <1 => the arms are UNDER the limit and could go faster; >1 => too fast.
    Because vel ~ 1/duration and acc ~ 1/duration^2 (so sqrt(acc-ratio) ~ 1/duration),
    this whole quantity is ~ proportional to 1/duration, which is why one step
    duration *= ratio lands right on the limit."""
    dt = duration / max(len(pos) - 1, 1)
    vel = np.gradient(pos, dt, axis=0); acc = np.gradient(vel, dt, axis=0)
    r = 0.0
    for j in range(NDOF):
        r = max(r, float(np.max(np.abs(vel[:, j]))) / VEL_LIM[j])
        r = max(r, float(np.sqrt(max(float(np.max(np.abs(acc[:, j]))), 0.0) / ACC_LIM[j])))
    return r


def time_optimal(arm_names, pos_per_arm, t_ref, base_dur, safety):
    """Uniformly time-scale a set of arm trajectories (sharing time base t_ref) to
    the SHORTEST duration keeping every joint within VEL/ACC limits * (1/safety).
    Stretching the shared time base proportionally means a longer duration genuinely
    SLOWS the motion (unlike padding a hold at the end). Relative phase is preserved,
    so any collision-freedom of the input is preserved. Returns (kdur, {arm: pos})."""
    tspan = float(t_ref[-1]) if (len(t_ref) and t_ref[-1] > 0) else float(base_dur)
    def st(kd):
        return t_ref * (kd / tspan)
    kdur = tspan
    for _ in range(16):
        ratio = max(speed_ratio(resample(pos_per_arm[n], st(kdur), kdur), kdur)
                    for n in arm_names)
        if ratio <= 1e-9:
            break
        new = max(0.3, kdur * ratio * safety)
        done = abs(new - kdur) <= 0.005 * kdur
        kdur = new
        if done:
            break
    kpos = {n: resample(pos_per_arm[n], st(kdur), kdur) for n in arm_names}
    return kdur, kpos


def _write(out, arm_names, fname='s43_kuramoto.json'):
    with open(fname, 'w') as fh: json.dump(out, fh, indent=2)


def _write_traj(arm_names, kpos, kdur, base_dur, rdata, sync_report,
                reason, kmind, kur_ms, used_kuramoto):
    """Write s34 with resampled positions + recomputed vel/acc at duration kdur."""
    out = {'duration': float(kdur), 'arm_names': arm_names, 'kuramoto_time_ms': kur_ms,
           'final_duration_s': round(float(kdur), 4),
           'duration_overhead': round(float(kdur) / base_dur - 1.0, 4),
           'synchronisation_report': sync_report,
           'enhanced_kuramoto_used': used_kuramoto}
    for n in arm_names:
        pos = np.asarray(kpos[n]); dt = kdur / max(len(pos) - 1, 1)
        vel = np.gradient(pos, dt, axis=0); acc = np.gradient(vel, dt, axis=0)
        out[n] = {'robot_name': n,
            'metadata': {**rdata[n]['metadata'], 'duration': float(kdur)},
            'trajectory': {'time': np.linspace(0., kdur, len(pos)).tolist(),
                'positions': pos.tolist(), 'velocities': vel.tolist(),
                'accelerations': acc.tolist(),
                'arc_fracs': np.linspace(0., 1., len(pos)).tolist()}}
    out['final_verification'] = {'collisions_after_resample': 0,
        'min_dist_after_resample_m': round(float(kmind), 4),
        'verified_collision_free': True, 'reason': reason}
    _write(out, arm_names)


def _passthrough(rdata, arm_names, label, collision_free, mind, reason, kur_ms=0.):
    out = {'duration': float(rdata['duration']), 'arm_names': arm_names,
           'kuramoto_time_ms': kur_ms, 'final_duration_s': float(rdata['duration']),
           'duration_overhead': 0., 'enhanced_kuramoto_used': False,
           'synchronisation_report': {'pair_reports': {}, 'collision_free': collision_free,
               'passthrough': True, 'total_critical': 0},
           'final_verification': {'collisions_after_resample': 0 if collision_free else -1,
               'min_dist_after_resample_m': round(float(mind), 4),
               'verified_collision_free': collision_free, 'reason': reason}}
    for n in arm_names: out[n] = rdata[n]
    _write(out, arm_names)
    print(f'  Saved: s43_kuramoto.json ({label})')


def _report_clearance(arm_names, kpos, bases):
    """Print the inter-arm surface distance profile (like MATLAB): min, where it
    occurs, and the value at start / end of the motion."""
    A, B = arm_names[0], arm_names[1]
    K = min(len(kpos[A]), len(kpos[B])); step = max(1, K // 300)
    ks = list(range(0, K, step)); 
    if ks[-1] != K - 1: ks.append(K - 1)
    ds = np.array([pair_min_dist(kpos[A][k], bases[A], kpos[B][k], bases[B]) for k in ks])
    kmin = ks[int(np.argmin(ds))]
    print(f'  clearance (arm-arm surface): min {ds.min()*100:.1f}cm @ '
          f'{100*kmin/max(K-1,1):.0f}% of path | start {ds[0]*100:.1f}cm  end {ds[-1]*100:.1f}cm')
    return float(ds.min())


def _report_targets(arm_names, kpos, arm_pos):
    """Confirm each arm's final config matches the commanded target (given joints)."""
    for n in arm_names:
        tgt = np.asarray(arm_pos[n])[-1]
        err = float(np.degrees(np.max(np.abs(np.asarray(kpos[n])[-1] - tgt))))
        print(f'  [{n}] reaches target config: max joint err {err:.3f} deg')


def main():
    print('\n' + '='*66)
    print('  STEP 43  --  Kuramoto-ALONE (temporal phase lag; retraction deferred to step_44) [DUAL ARM]')
    print('='*66)
    if not os.path.exists('s41_trajectories.json'):
        print('  s41_trajectories.json not found'); sys.exit(1)
    with open('s41_trajectories.json') as fh: rdata = json.load(fh)
    arm_names = rdata.get('arm_names', ARM_NAMES); duration = float(rdata['duration'])
    bases = {n: np.array(ROBOT_BASES.get(n, [0, 0, 0])) for n in arm_names}
    arm_pos = {n: np.array(rdata[n]['trajectory']['positions'], dtype=float) for n in arm_names}

    # (0) Definitive check of the seed input
    in_coll, in_mind = scan_all(arm_names, arm_pos, bases)
    input_safe = (in_coll == 0)
    print(f'\n  Arms: {arm_names}  dur={duration:.2f}s  input_safe={input_safe} '
          f'(min_dist={in_mind*100:.1f}cm, {in_coll} colliding steps)  safety={SAFETY:.2f}')

    # ---- time-optimal CLEAN base duration (full speed, no lag corners -> no spike) ----
    L = min(len(arm_pos[n]) for n in arm_names)
    base_seed = {n: arm_pos[n][:L] for n in arm_names}
    t_base = np.linspace(0., duration, L)
    T_opt, _ = time_optimal(arm_names, base_seed, t_base, duration, SAFETY)
    T_opt = max(T_opt, MIN_DUR)

    # (1) Collision-free seed -> no lag; run the whole move at the time-optimal duration.
    if input_safe:
        st = t_base * (T_opt / float(t_base[-1]))
        kpos = {n: resample(base_seed[n], st, T_opt) for n in arm_names}
        kcoll, kmind = scan_all(arm_names, kpos, bases)
        if kcoll != 0:
            print('  Time-scaled seed unexpectedly collided -- passing seed through.')
            _passthrough(rdata, arm_names, 'passthrough SUCCESS', True, in_mind,
                         'no_collision_in_input')
            print(f'  Next : ros2 run dual_arm_sync step_44\n'); return
        rep = {'pair_reports': {}, 'collision_free': True, 'passthrough': True,
               'total_critical': 0, 'method': 'time_optimal_passthrough'}
        _write_traj(arm_names, kpos, T_opt, duration, rdata, rep,
                    'time_optimal_no_collision', kmind, 0.0, False)
        print(f'  Input collision-free -> time-optimal duration (no lag).')
        print(f'  Duration: base {duration:.1f}s -> {T_opt:.2f}s')
        _report_clearance(arm_names, kpos, bases)
        _report_targets(arm_names, kpos, arm_pos)
        print(f'  Saved: s43_kuramoto.json (time-optimal passthrough)')
        print(f'  Next : ros2 run dual_arm_sync step_44\n'); return

    # (2) Colliding seed -> resolve by phase lag BUILT AT the time-optimal base duration
    #     (do NOT time-optimise the lag schedule: its wait->move corner spikes and the
    #      optimiser would chase that spike, re-inflating the makespan to tens of seconds).
    print(f'  Path-velocity coordination: grid={COORD_RES}x{COORD_RES}  '
          f'(timing only; lag built at T_opt={T_opt:.2f}s)')
    t0 = time.time()
    sync, t_vec, kur_rep = run_enhanced_kuramoto(arm_names, arm_pos, bases, T_opt)
    kur_ms = round((time.time() - t0) * 1000, 1)
    print(f'  result: collision_free={kur_rep["collision_free"]}  '
          f'timing_separable={kur_rep.get("timing_separable")}  '
          f'schedule={kur_rep.get("schedule")}  time={kur_ms:.0f}ms')
    for pair, rep in kur_rep['pair_reports'].items():
        print(f'  {pair}: min_dist={rep["min_dist_m"]*100:.1f}cm  '
              f'critical={rep["critical_steps"]}  nearmiss={rep["nearmiss_steps"]}')

    # (3) Materialise the lag at its natural span (= (1+lag)*T_opt); cap any residual
    #     stretch so the acceleration corner can't re-inflate the whole makespan.
    feasible_kur = kur_rep['collision_free']
    if feasible_kur:
        pos_per_arm = {n: np.array(sync[n]) for n in arm_names}
        kdur = float(t_vec[-1])
        kpos = {n: resample(pos_per_arm[n], t_vec, kdur) for n in arm_names}
        ratio = max(speed_ratio(kpos[n], kdur) for n in arm_names)
        if ratio > 1.05:                          # genuine limit use beyond safety headroom
            kdur = kdur * min(ratio, MAX_LAG_STRETCH)
            st = t_vec * (kdur / float(t_vec[-1]))
            kpos = {n: resample(pos_per_arm[n], st, kdur) for n in arm_names}
        kcoll, kmind = scan_all(arm_names, kpos, bases)
        if kcoll == 0:
            _write_traj(arm_names, kpos, kdur, duration, rdata,
                        {**kur_rep, 'collision_free': True},
                        'kuramoto_synchronized', kmind, kur_ms, True)
            print(f'  Saved: s43_kuramoto.json (Kuramoto SUCCESS)')
            print(f'  Duration: base {duration:.1f}s -> {kdur:.2f}s  '
                  f'(T_opt {T_opt:.2f}s + lag; overhead vs base {(kdur/duration-1)*100:.0f}%)')
            _report_clearance(arm_names, kpos, bases)
            _report_targets(arm_names, kpos, arm_pos)
            print(f'  near-miss handled: {kur_rep["total_nearmiss"]}')
            print(f'  Next : ros2 run dual_arm_sync step_44\n'); return

    # (4) Timing alone can't skip it -> NOT a failure here. Hand the seed to step_44,
    #     which will retract both arms minimally (re-checking Kuramoto each step) and
    #     finish. We pass the ORIGINAL seed through with needs_retraction=True so step_44
    #     retracts from the optimal path, not from a half-lagged one.
    print(f'  Timing alone cannot skip this collision (input had {in_coll} colliding steps).')
    if kur_rep.get('best_lag_diag'):
        print(f'  {kur_rep["best_lag_diag"]}')
    print('  -> deferring to step_44 (retraction + Kuramoto).')
    out = {'duration': float(duration), 'arm_names': arm_names, 'kuramoto_time_ms': kur_ms,
           'final_duration_s': float(duration), 'duration_overhead': 0.0,
           'enhanced_kuramoto_used': False, 'needs_retraction': True,
           'synchronisation_report': {'pair_reports': kur_rep.get('pair_reports', {}),
               'collision_free': False, 'passthrough': True, 'total_critical': in_coll,
               'method': 'kuramoto_alone_insufficient'},
           'final_verification': {'collisions_after_resample': in_coll,
               'min_dist_after_resample_m': round(float(in_mind), 4),
               'verified_collision_free': False, 'reason': 'needs_retraction'}}
    for n in arm_names:
        out[n] = rdata[n]
    _write(out, arm_names)
    print('  Saved: s43_kuramoto.json (needs_retraction=True)')
    print('  Next : ros2 run dual_arm_sync step_44 (retraction + Kuramoto)\n')


if __name__ == '__main__': main()