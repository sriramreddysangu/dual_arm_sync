#!/usr/bin/env python3
"""
step_54.py  --  TIER 1: Kuramoto-Only Resolver
================================================
INPUT  : s52_trajectories.json + s53_collision_map.json
OUTPUT : s54_tier1.json

PHILOSOPHY:
  Cheapest fix first. Take the ORIGINAL B-spline trajectories (no geometric
  modification) and run Kuramoto phase synchronization. If timing alone
  resolves all collisions, we're done. No need for tier 2 or tier 3.

OUTPUT:
  s54_tier1.json contains:
    - 'tier_resolved': True if Kuramoto cleared everything, else False
    - 'next_tier': 'DONE' / 'tier_2' (which tier to run next)
    - 'synchronisation_report', 'duration', per-arm trajectories
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from _robot5x import (NDOF, POS_LIM, VEL_LIM, ACC_LIM, ROBOT_BASES, ARM_NAMES,
                       pair_min_dist, pair_collides, LINK_RADII, SAFETY_MARGIN)

K_ATTRACT      = 5.0
K_REPULSE_COLL = 80.0
K_EMERGENCY    = 250.0
KUR_DT         = 0.01
MIN_SAFE       = LINK_RADII.min() * 2 + SAFETY_MARGIN
RATE_HZ        = 100.0
MAX_RESCALE    = 5.0


def interp_pos(pos, frac):
    frac = float(np.clip(frac, 0, 1)); n = len(pos) - 1
    if n <= 0: return pos[0].copy()
    idx = min(int(frac * n), n - 1); a = frac * n - idx
    return pos[idx] + a * (pos[min(idx+1, n)] - pos[idx])


def smooth_positions(pos, window=7):
    if window < 3 or len(pos) < window: return pos
    if window % 2 == 0: window += 1
    half = window // 2
    smoothed = pos.copy(); n, _ = pos.shape
    for k in range(half, n - half):
        smoothed[k] = np.mean(pos[k-half:k+half+1], axis=0)
    return smoothed


def resample(sync_pos, t_kur, duration):
    n_out = max(2, int(round(duration * RATE_HZ)))
    t_out = np.linspace(0., duration, n_out)
    t_in = t_kur if len(t_kur) == len(sync_pos) else np.linspace(0., duration, len(sync_pos))
    out = np.zeros((n_out, NDOF))
    for j in range(NDOF):
        out[:, j] = np.interp(t_out, t_in, sync_pos[:, j])
    return np.clip(out, POS_LIM[:, 0], POS_LIM[:, 1])


def check_limits(pos, duration):
    dt = duration / max(len(pos) - 1, 1)
    vel = np.gradient(pos, dt, axis=0); acc = np.gradient(vel, dt, axis=0)
    sv = sa = 1.0
    for j in range(NDOF):
        vp = float(np.max(np.abs(vel[:, j]))); ap = float(np.max(np.abs(acc[:, j])))
        if vp > VEL_LIM[j]: sv = max(sv, vp / VEL_LIM[j])
        if ap > ACC_LIM[j]: sa = max(sa, float(np.sqrt(ap / ACC_LIM[j])))
    return sv, sa


def run_kuramoto(arm_names, arm_pos, bases, duration):
    N = len(arm_names)
    omega0 = 1.0 / duration
    phi = np.zeros(N); om = np.full(N, omega0)
    pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    max_steps = int(round(4.0 * duration / KUR_DT))

    sync_acc = {n: [] for n in arm_names}
    pdist_acc = {p: [] for p in pairs}
    k = 0
    while k < max_steps:
        phi_c = np.clip(phi, 0., 1.)
        q_now = {n: interp_pos(arm_pos[n], phi_c[idx])
                  for idx, n in enumerate(arm_names)}
        for idx, n in enumerate(arm_names): sync_acc[n].append(q_now[n].copy())
        dists = {}
        for (i, j) in pairs:
            d = pair_min_dist(q_now[arm_names[i]], bases[arm_names[i]],
                               q_now[arm_names[j]], bases[arm_names[j]])
            dists[(i, j)] = d
            pdist_acc[(i, j)].append(d)
        if np.all(phi >= 1.0 - 1e-9): break

        dphi = np.zeros(N)
        for (i, j) in pairs:
            dist = dists[(i, j)]
            f_coll = float(np.clip(1 - dist / MIN_SAFE, 0, 1))
            f_danger = float(np.clip(1 - dist / (MIN_SAFE * 0.5), 0, 1))
            dphi[i] += K_ATTRACT * float(np.sin(phi[j] - phi[i]))
            dphi[j] += K_ATTRACT * float(np.sin(phi[i] - phi[j]))
            if dist < MIN_SAFE:
                mag = K_REPULSE_COLL * (f_coll ** 2) * 30
                if dist < MIN_SAFE * 0.5:
                    mag += K_EMERGENCY * (f_danger ** 3)
                dphi[i] -= mag * 0.7; dphi[j] -= mag * 0.7
        phi += KUR_DT * np.clip(om + dphi, 0.0, 4.0)
        k += 1

    for n in arm_names: sync_acc[n].append(arm_pos[n][-1].copy())
    for (i, j) in pairs:
        d = pair_min_dist(arm_pos[arm_names[i]][-1], bases[arm_names[i]],
                           arm_pos[arm_names[j]][-1], bases[arm_names[j]])
        pdist_acc[(i, j)].append(d)

    sync = {n: np.array(sync_acc[n]) for n in arm_names}
    n_out = len(sync[arm_names[0]])
    real_dur = max((n_out - 1) * KUR_DT, duration)
    t_vec = np.linspace(0., real_dur, n_out)

    pair_rep = {}
    total_critical = 0
    for (i, j) in pairs:
        dv = np.array(pdist_acc[(i, j)])
        nc = int(np.sum(dv < MIN_SAFE))
        total_critical += nc
        pair_rep[f'{arm_names[i]}<->{arm_names[j]}'] = {
            'min_dist_m': round(float(np.min(dv)), 4),
            'critical_steps': nc, 'collision_free': nc == 0}

    return sync, t_vec, {
        'pair_reports': pair_rep,
        'total_critical': total_critical,
        'collision_free': total_critical == 0}


def main():
    print('\n' + '='*66)
    print('  STEP 54  --  TIER 1: Kuramoto-Only Resolver')
    print('='*66)
    for fname, step in [('s52_trajectories.json', 'step_52'),
                         ('s53_collision_map.json', 'step_53')]:
        if not os.path.exists(fname):
            print(f'  {fname} not found'); sys.exit(1)

    with open('s52_trajectories.json') as fh: tdata = json.load(fh)
    with open('s53_collision_map.json') as fh: cmap = json.load(fh)

    arm_names = tdata.get('arm_names', ARM_NAMES)
    duration  = float(tdata['duration'])
    bases     = {n: np.array(ROBOT_BASES.get(n, [0, 0, 0])) for n in arm_names}
    arm_pos = {n: np.array(tdata[n]['trajectory']['positions'], dtype=float)
                for n in arm_names}

    # If no collision detected, just pass through
    if cmap['overall_status'] == 'SAFE':
        print('  No collision in original trajectory -- no resolution needed.')
        out = {'tier_resolved': True, 'next_tier': 'DONE',
                'tier': 1, 'method': 'no_action_needed',
                'duration': duration, 'arm_names': arm_names,
                'synchronisation_report': {'collision_free': True,
                                           'pair_reports': {}}}
        for name in arm_names:
            pos = arm_pos[name]
            out[name] = {
                'robot_name': name,
                'metadata' : tdata[name]['metadata'],
                'trajectory': {
                    'time'         : np.linspace(0., duration, len(pos)).tolist(),
                    'positions'    : pos.tolist(),
                    'velocities'   : np.array(tdata[name]['trajectory']['velocities']).tolist(),
                    'accelerations': np.array(tdata[name]['trajectory']['accelerations']).tolist(),
                    'arc_fracs'    : np.linspace(0., 1., len(pos)).tolist(),
                }}
        with open('s54_tier1.json', 'w') as fh: json.dump(out, fh, indent=2)
        print(f'  Saved: s54_tier1.json (passthrough)')
        return

    print(f'\n  Collisions present -- running Kuramoto on original trajectory')
    print(f'  K_ATTRACT={K_ATTRACT} K_REPULSE_COLL={K_REPULSE_COLL} '
          f'MIN_SAFE={MIN_SAFE*100:.0f}cm')

    t0 = time.time()
    sync, t_vec, kur_rep = run_kuramoto(arm_names, arm_pos, bases, duration)
    kur_ms = round((time.time() - t0) * 1000, 1)
    print(f'\n  Kuramoto: collision_free={kur_rep["collision_free"]}  '
          f'real_dur={t_vec[-1]:.2f}s  time={kur_ms:.0f}ms')
    for pair, rep in kur_rep['pair_reports'].items():
        print(f'  {pair}: min_dist={rep["min_dist_m"]*100:.1f}cm  '
              f'critical={rep["critical_steps"]}')

    # If Kuramoto failed: write FAIL passthrough, escalate to tier 2
    if not kur_rep['collision_free']:
        print(f'\n  TIER 1 FAILED  -->  escalating to TIER 2 (3-waypoint via-J)')
        out = {'tier_resolved': False, 'next_tier': 'tier_2',
                'tier': 1, 'method': 'kuramoto_only',
                'duration': duration, 'arm_names': arm_names,
                'kuramoto_time_ms': kur_ms,
                'synchronisation_report': kur_rep}
        # passthrough original trajectory for next tier to consume
        for name in arm_names:
            out[name] = tdata[name]
        with open('s54_tier1.json', 'w') as fh: json.dump(out, fh, indent=2)
        print(f'  Saved: s54_tier1.json (escalate)')
        return

    # Kuramoto succeeded -- smooth, rescale, verify
    duration = max(duration, float(t_vec[-1]))
    pos_per_arm = {n: smooth_positions(np.array(sync[n]), window=7) for n in arm_names}
    rescale_infeasible = False
    for attempt in range(5):
        sv_g = sa_g = 1.0
        for name in arm_names:
            pos_r = resample(pos_per_arm[name], t_vec, duration)
            sv, sa = check_limits(pos_r, duration)
            sv_g = max(sv_g, sv); sa_g = max(sa_g, sa)
        scale = max(sv_g, sa_g)
        if scale <= 1.001:
            print(f'  Limits OK after attempt {attempt} (dur={duration:.2f}s)')
            break
        if scale > MAX_RESCALE:
            print(f'  Attempt {attempt}: scale={scale:.2f} > {MAX_RESCALE} -- INFEASIBLE')
            rescale_infeasible = True; break
        window = 7 + 4 * attempt
        for name in arm_names:
            pos_per_arm[name] = smooth_positions(np.array(sync[name]), window=window)
        duration = duration * min(scale, 2.0) * 1.05
        print(f'  Attempt {attempt}: scale={scale:.2f} window={window} -> dur={duration:.2f}s')
    else:
        if scale > 1.5: rescale_infeasible = True

    if rescale_infeasible:
        print(f'  TIER 1 RESCALE INFEASIBLE  -->  escalating to TIER 2')
        out = {'tier_resolved': False, 'next_tier': 'tier_2',
                'tier': 1, 'method': 'kuramoto_only',
                'duration': float(tdata['duration']), 'arm_names': arm_names,
                'kuramoto_time_ms': kur_ms,
                'synchronisation_report': {**kur_rep, 'collision_free': False}}
        for name in arm_names:
            out[name] = tdata[name]
        with open('s54_tier1.json', 'w') as fh: json.dump(out, fh, indent=2)
        return

    # Final verification on resampled output
    final_coll = 0; final_md = float('inf')
    for i in range(len(arm_names)):
        for j in range(i+1, len(arm_names)):
            ni, nj = arm_names[i], arm_names[j]
            pi = resample(pos_per_arm[ni], t_vec, duration)
            pj = resample(pos_per_arm[nj], t_vec, duration)
            K = min(len(pi), len(pj))
            for k in range(K):
                d = pair_min_dist(pi[k], bases[ni], pj[k], bases[nj])
                if d < final_md: final_md = d
                if pair_collides(pi[k], bases[ni], pj[k], bases[nj]):
                    final_coll += 1

    out = {'tier_resolved': final_coll == 0,
            'next_tier': 'DONE' if final_coll == 0 else 'tier_2',
            'tier': 1, 'method': 'kuramoto_only',
            'duration': duration, 'arm_names': arm_names,
            'final_duration_s': round(duration, 4),
            'duration_overhead': round(duration / float(tdata['duration']) - 1, 4),
            'kuramoto_time_ms': kur_ms,
            'synchronisation_report': kur_rep,
            'final_verification': {
                'collisions_after_resample': int(final_coll),
                'min_dist_after_resample_m': round(float(final_md), 4),
                'verified_collision_free': final_coll == 0,
            }}
    for name in arm_names:
        pos = resample(pos_per_arm[name], t_vec, duration)
        dt = duration / max(len(pos) - 1, 1)
        vel = np.gradient(pos, dt, axis=0); acc = np.gradient(vel, dt, axis=0)
        out[name] = {
            'robot_name': name,
            'metadata' : {**tdata[name]['metadata'], 'duration': float(duration)},
            'trajectory': {
                'time'         : np.linspace(0., duration, len(pos)).tolist(),
                'positions'    : pos.tolist(),
                'velocities'   : vel.tolist(),
                'accelerations': acc.tolist(),
                'arc_fracs'    : np.linspace(0., 1., len(pos)).tolist(),
            }}

    with open('s54_tier1.json', 'w') as fh: json.dump(out, fh, indent=2)
    if final_coll == 0:
        print(f'\n  ✓ TIER 1 RESOLVED  --  duration overhead {out["duration_overhead"]*100:.0f}%')
        print(f'  Saved: s54_tier1.json (DONE)')
    else:
        print(f'\n  TIER 1 NEAR-MISS  ({final_coll} collisions after resample)')
        print(f'  Escalating to TIER 2')


if __name__ == '__main__': main()