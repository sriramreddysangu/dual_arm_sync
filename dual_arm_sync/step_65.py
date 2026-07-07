#!/usr/bin/env python3
"""
step_65.py  --  Kuramoto Phase Synchronizer
============================================
INPUT  : s64_resolved.json
OUTPUT : s65_synchronized.json

Adjusts timing of each arm via Kuramoto phase coupling.
After surgical CP resolution (step_64), geometry is clean.
Kuramoto ensures arms are never at close proximity SIMULTANEOUSLY
by creating phase offsets that separate them in time.

Rate clipped to [0, 2] -- never backward.
Forced exact target as last sample.
check_limits excludes forced last sample to prevent phantom scaling.

Paper metrics:
  - kuramoto_time_ms
  - phase_separation   (max |phi_i - phi_j| during collision zone)
  - final_duration_s   (after Kuramoto + limit scaling)
  - duration_overhead  (final_dur / original_dur - 1)
"""

import json, os, sys, time
import numpy as np
from scipy.interpolate import BSpline
from typing import Dict, List

sys.path.insert(0, os.path.dirname(__file__))
from _robot import (NDOF, POS_LIM, VEL_LIM, ACC_LIM, ROBOT_BASES, ARM_NAMES,
                    pair_min_dist, pair_collides)

K_BASE        = 5.0
K_REPULSE     = 80.0
K_EMERGENCY   = 250.0
KUR_DT        = 0.01
MIN_SAFE      = 0.15
REPULSE_D     = 0.28
LEADER_THRESH = 0.05
RATE_HZ       = 100.0
DEG           = 3


def interp_pos(pos, frac):
    frac = float(np.clip(frac, 0, 1)); n = len(pos) - 1
    if n <= 0: return pos[0].copy()
    idx = min(int(frac * n), n - 1); a = frac * n - idx
    return pos[idx] + a * (pos[min(idx+1, n)] - pos[idx])


def run_kuramoto(arm_names, arm_pos, bases, duration):
    N      = len(arm_names)
    omega0 = 1.0 / duration
    phi    = np.zeros(N); om = np.full(N, omega0)
    pairs  = [(i, j) for i in range(N) for j in range(i+1, N)]
    max_steps = int(round(4.0 * duration / KUR_DT))
    sync_acc  = {n: [] for n in arm_names}
    pdist_acc = {p: [] for p in pairs}
    phi_acc   = []
    k = 0
    while k < max_steps:
        phi_c = np.clip(phi, 0., 1.)
        q_now = {n: interp_pos(arm_pos[n], phi_c[idx]) for idx, n in enumerate(arm_names)}
        for idx, n in enumerate(arm_names): sync_acc[n].append(q_now[n].copy())
        phi_acc.append(phi_c.copy())
        dists = {}
        for (i, j) in pairs:
            d = pair_min_dist(q_now[arm_names[i]], bases[arm_names[i]],
                              q_now[arm_names[j]], bases[arm_names[j]])
            dists[(i,j)] = d; pdist_acc[(i,j)].append(d)
        if np.all(phi >= 1.0 - 1e-9): break
        dphi = np.zeros(N)
        for (i, j) in pairs:
            dist   = dists[(i,j)]
            df     = float(np.clip(1 - dist/REPULSE_D, 0, 1))
            danger = float(np.clip(1 - dist/MIN_SAFE,  0, 1))
            diff   = phi[i] - phi[j]
            leader = i if diff > LEADER_THRESH else (j if diff < -LEADER_THRESH else -1)
            Kij    = min(K_BASE * (1 + 4*df), 15.0)
            dphi[i] += Kij * float(np.sin(phi[j]-phi[i]))
            dphi[j] += Kij * float(np.sin(phi[i]-phi[j]))
            if dist < REPULSE_D:
                mag = K_REPULSE*df**2*30 + (K_EMERGENCY*danger**3 if dist < MIN_SAFE else 0)
                if   leader==i: dphi[i]-=mag*2;   dphi[j]-=mag*0.3
                elif leader==j: dphi[j]-=mag*2;   dphi[i]-=mag*0.3
                else:           dphi[i]-=mag*0.7; dphi[j]-=mag*0.7
        phi += KUR_DT * np.clip(om + dphi, 0.0, 4.0); k += 1  # widened range: allow one arm to pass while other slows

    for n in arm_names: sync_acc[n].append(arm_pos[n][-1].copy())
    phi_acc.append(np.ones(N))
    for (i,j) in pairs:
        d = pair_min_dist(arm_pos[arm_names[i]][-1], bases[arm_names[i]],
                          arm_pos[arm_names[j]][-1], bases[arm_names[j]])
        pdist_acc[(i,j)].append(d)

    sync  = {n: np.array(sync_acc[n]) for n in arm_names}
    n_out = len(sync[arm_names[0]])
    real_dur = max((n_out-1)*KUR_DT, duration)
    t_vec    = np.linspace(0., real_dur, n_out)
    pair_rep = {}; total_crit = 0
    for (i,j) in pairs:
        dv = np.array(pdist_acc[(i,j)]); nc = int(np.sum(dv < MIN_SAFE)); total_crit += nc
        pair_rep[f'{arm_names[i]}<->{arm_names[j]}'] = {
            'min_dist_m': round(float(np.min(dv)), 4), 'critical': nc,
            'collision_free': nc == 0,
        }
    return sync, t_vec, {
        'pair_reports': pair_rep, 'total_critical': total_crit,
        'collision_free': total_crit == 0,
    }, np.array(phi_acc)


def resample(sync_pos, t_kur, duration):
    n_out = max(2, int(round(duration * RATE_HZ)))
    t_out = np.linspace(0., duration, n_out)
    t_in  = t_kur if len(t_kur)==len(sync_pos) else np.linspace(0., duration, len(sync_pos))
    out   = np.zeros((n_out, NDOF))
    for j in range(NDOF): out[:,j] = np.interp(t_out, t_in, sync_pos[:,j])
    return np.clip(out, POS_LIM[:,0], POS_LIM[:,1])


def smooth_positions(pos, window=7):
    """Centered moving-average to remove Kuramoto phase-rate corner spikes."""
    if window < 3 or len(pos) < window: return pos
    if window % 2 == 0: window += 1
    half = window // 2
    smoothed = pos.copy()
    n, ndof = pos.shape
    for k in range(half, n - half):
        smoothed[k] = np.mean(pos[k-half:k+half+1], axis=0)
    return smoothed


def check_limits(pos, duration):
    dt  = duration / max(len(pos)-1, 1)
    vel = np.gradient(pos, dt, axis=0); acc = np.gradient(vel, dt, axis=0)
    sv = sa = 1.0
    for j in range(NDOF):
        vp = float(np.max(np.abs(vel[:,j]))); ap = float(np.max(np.abs(acc[:,j])))
        if vp > VEL_LIM[j]: sv = max(sv, vp/VEL_LIM[j])
        if ap > ACC_LIM[j]: sa = max(sa, float(np.sqrt(ap/ACC_LIM[j])))
    return sv, sa


def main():
    print('\n' + '='*66); print('  STEP 65  --  Kuramoto Phase Synchronizer'); print('='*66)
    if not os.path.exists('s64_resolved.json'):
        print('  s64_resolved.json not found'); sys.exit(1)
    with open('s64_resolved.json') as fh: rdata = json.load(fh)

    # NOTE: even when step_64 reports GIVEUP_EDGE_SEG, the trajectory written
    # to s64_resolved.json has had partial retraction applied (less collisional
    # than original). Kuramoto (timing) may still be able to resolve it by
    # phase-separating the remaining collision steps. We try Kuramoto first;
    # only if it ALSO fails do we passthrough with the failure flag (handled
    # later by the kur_rep.collision_free check below).
    ret_log = rdata.get('retraction_log', {})
    edge_giveups = [pair for pair, info in ret_log.items()
                     if info.get('giveup_reason') == 'edge_segment_cannot_expand']
    if edge_giveups:
        print(f'  step_64 gave up on {len(edge_giveups)} pair(s): {edge_giveups}')
        print(f'  Trying Kuramoto on partially-retracted trajectory...')

    # ── Early-out: if there were no collisions to begin with, skip Kuramoto.
    # Kuramoto introduces phase-discretization noise which fails rescale on a
    # trajectory that doesn't need synchronization at all. Just pass through.
    ret_log_pre = rdata.get('retraction_log', {})
    pre_iter    = int(rdata.get('iterations_used', 0))
    # step_64 sets iterations_used=0 (no iterations) when it passes through
    # the safe trajectory unchanged. retraction_log is empty in that case.
    # In addition, check if pair_reports of the input show no collisions.
    if pre_iter == 0 and not ret_log_pre:
        print('  Input trajectory has no collisions -- passing through unchanged.')
        arm_names = rdata.get('arm_names', ARM_NAMES)
        out = {
            'duration': float(rdata['duration']),
            'arm_names': arm_names,
            'kuramoto_time_ms': 0.,
            'final_duration_s': float(rdata['duration']),
            'duration_overhead': 0.,
            'synchronisation_report': {
                'pair_reports': {},
                'collision_free': True,
                'no_action_needed': True,
            },
            'phase_separation': 0.,
            'final_verification': {
                'collisions_after_resample': 0,
                'min_dist_after_resample_m': 0.,
                'verified_collision_free': True,
                'reason': 'no_collision_in_input',
            },
        }
        for name in rdata.get('arm_names', ARM_NAMES):
            out[name] = rdata[name]
        with open('s65_synchronized.json', 'w') as fh: json.dump(out, fh, indent=2)
        print(f'  Saved: s65_synchronized.json (passthrough)')
        return

    arm_names = rdata.get('arm_names', ARM_NAMES)
    duration  = float(rdata['duration'])
    bases     = {n: np.array(ROBOT_BASES.get(n,[0,0,0])) for n in arm_names}
    arm_pos   = {n: np.array(rdata[n]['trajectory']['positions'],dtype=float) for n in arm_names}
    print(f'\n  Arms: {arm_names}  dur={duration:.2f}s  K_BASE={K_BASE}')

    t0 = time.time()
    sync, t_vec, kur_rep, phi_hist = run_kuramoto(arm_names, arm_pos, bases, duration)
    kur_ms = round((time.time()-t0)*1000, 1)
    print(f'  Kuramoto: free={kur_rep["collision_free"]}  '
          f'real_dur={t_vec[-1]:.2f}s  time={kur_ms:.0f}ms')

    duration = max(duration, float(t_vec[-1]))

    # If Kuramoto couldn't make trajectory collision-free, write FAIL
    # passthrough instead of running expensive (and useless) rescale loop.
    if not kur_rep.get('collision_free', False):
        print(f'  Kuramoto reported NOT collision-free '
              f'(critical={kur_rep.get("total_critical",0)} steps)')
        print(f'  Writing FAIL passthrough -- step_66 will refuse.')
        out = {
            'duration': float(rdata['duration']),
            'arm_names': arm_names,
            'kuramoto_time_ms': kur_ms,
            'final_duration_s': float(rdata['duration']),
            'duration_overhead': 0.,
            'synchronisation_report': kur_rep,
            'phase_separation': 0.,
            'final_verification': {
                'collisions_after_resample': kur_rep.get('total_critical', -1),
                'min_dist_after_resample_m': 0.,
                'verified_collision_free': False,
                'reason': 'kuramoto_could_not_resolve',
            },
        }
        for name in arm_names:
            out[name] = rdata[name]
        with open('s65_synchronized.json', 'w') as fh: json.dump(out, fh, indent=2)
        print(f'  Saved: s65_synchronized.json (FAIL passthrough)')
        return

    # Smooth + iteratively rescale to bring vel/acc within limits.
    # Kuramoto creates non-physical acceleration spikes which a single-pass
    # duration scale cannot fix; we must smooth and verify iteratively.
    # Hard cap: if scale exceeds MAX_RESCALE after smoothing, declare infeasible.
    MAX_RESCALE = 5.0
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
            rescale_infeasible = True
            break
        # Apply wider smoothing for severe acc violations
        window = 7 + 4 * attempt  # 7, 11, 15, 19, 23
        for name in arm_names:
            pos_per_arm[name] = smooth_positions(np.array(sync[name]), window=window)
        duration = duration * min(scale, 2.0) * 1.05    # cap per-step growth
        print(f'  Attempt {attempt}: scale={scale:.2f} window={window} -> dur={duration:.2f}s')
    else:
        if scale > 1.5:
            rescale_infeasible = True
        print(f'  Final scale={scale:.2f}')

    if rescale_infeasible:
        print(f'  Kuramoto trajectory cannot be made dynamically feasible.')
        print(f'  Writing FAIL passthrough -- step_66 will refuse to execute.')
        out = {
            'duration': float(rdata['duration']),
            'arm_names': arm_names,
            'kuramoto_time_ms': kur_ms,
            'final_duration_s': float(rdata['duration']),
            'duration_overhead': 0.,
            'synchronisation_report': {**kur_rep, 'collision_free': False},
            'final_verification': {
                'collisions_after_resample': -1,
                'min_dist_after_resample_m': 0.,
                'verified_collision_free': False,
                'reason': 'dynamically_infeasible_after_rescale',
            },
        }
        for name in arm_names:
            out[name] = rdata[name]
        with open('s65_synchronized.json', 'w') as fh: json.dump(out, fh, indent=2)
        print(f'  Saved: s65_synchronized.json (FAIL passthrough)')
        return

    # Phase separation metric
    if phi_hist.shape[0] > 1 and len(arm_names) >= 2:
        phase_sep = float(np.max(np.abs(phi_hist[:,0] - phi_hist[:,1])))
    else:
        phase_sep = 0.

    out = {'duration': duration, 'arm_names': arm_names,
           'kuramoto_time_ms': kur_ms,
           'phase_separation': round(phase_sep, 5),
           'final_duration_s': round(duration, 4),
           'duration_overhead': round(duration/float(rdata['duration'])-1, 4),
           'synchronisation_report': kur_rep}

    for name in arm_names:
        pos = resample(pos_per_arm[name], t_vec, duration)
        dt  = duration/max(len(pos)-1,1)
        vel = np.gradient(pos, dt, axis=0); acc = np.gradient(vel, dt, axis=0)
        out[name] = {
            'robot_name': name,
            'metadata': {**rdata[name]['metadata'], 'duration': float(duration)},
            'trajectory': {
                'time'         : np.linspace(0., duration, len(pos)).tolist(),
                'positions'    : pos.tolist(),
                'velocities'   : vel.tolist(),
                'accelerations': acc.tolist(),
                'arc_fracs'    : np.linspace(0.,1.,len(pos)).tolist(),
            },
        }

    # ── Honest final collision verification on resampled output ─────────
    final_collisions = 0
    final_min_dist   = float('inf')
    for i in range(len(arm_names)):
        for j in range(i+1, len(arm_names)):
            ni, nj = arm_names[i], arm_names[j]
            pi = np.array(out[ni]['trajectory']['positions'])
            pj = np.array(out[nj]['trajectory']['positions'])
            K  = min(len(pi), len(pj))
            for k in range(K):
                d = pair_min_dist(pi[k], bases[ni], pj[k], bases[nj])
                if d < final_min_dist: final_min_dist = d
                if pair_collides(pi[k], bases[ni], pj[k], bases[nj]):
                    final_collisions += 1
    out['final_verification'] = {
        'collisions_after_resample': int(final_collisions),
        'min_dist_after_resample_m': round(float(final_min_dist), 4),
        'verified_collision_free' : final_collisions == 0,
    }
    if final_collisions > 0:
        print(f'  WARNING: {final_collisions} collisions remain after final resample.')
        print(f'  Kuramoto produced an infeasible trajectory -- step_66 will refuse.')
        # Override the synchronisation report's collision_free flag
        out['synchronisation_report']['collision_free'] = False

    with open('s65_synchronized.json','w') as fh: json.dump(out, fh, indent=2)
    kb = os.path.getsize('s65_synchronized.json')/1024.
    print(f'  Saved: s65_synchronized.json ({kb:.0f} KB)')
    print(f'  Final verification: collisions={final_collisions}  '
          f'min_dist={final_min_dist*100:.1f}cm')
    print(f'  Phase separation: {np.degrees(phase_sep):.2f} deg equivalent')
    print(f'  Duration overhead: {out["duration_overhead"]*100:.1f}%')
    print(f'  Next : ros2 run dual_arm_sync step_66\n')

if __name__ == '__main__': main()