#!/usr/bin/env python3
"""
step_plan.py  --  Plan synchronized trajectories (replaces step_22 + 23 + 24)
===============================================================================
Input  : ik_solutions.json   (from step_21)
Output : synchronized_trajectories.json   (consumed by step_25 unchanged)

The old three-stage flow (segment-grid B-spline -> 15-pair collision scan ->
Kuramoto + alternating-IK CP refinement) is replaced by ONE call into
multi_arm_core.plan_arms, which uses a single global clamped cubic B-spline per
arm, Kuramoto phase synchronisation (temporal stage), and local Boehm
knot-insertion retraction toward (J1,0,0,0,0,0) for residual SPATIAL conflicts,
routed by collision arc-fraction.

Output schema matches what step_25 reads:
  data[name]['trajectory'] = {time, positions, velocities, accelerations, n_samples}
  data[name]['metadata']   = {duration, end_joints, target_world_pos, n_samples,
                              outcome, refine_iterations}
===============================================================================
"""
import json, os, sys
import numpy as np

from dual_arm_sync import multi_arm_core as C

NDOF = C.NDOF
RATE_HZ = C.RATE_HZ
ROBOT_BASES = C.ROBOT_BASES


def _derivatives(pos: np.ndarray, duration: float):
    n = len(pos)
    dt = duration / max(n - 1, 1)
    vel = np.gradient(pos, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    return vel, acc


def main():
    print('=' * 68)
    print('  STEP PLAN  --  single-spline + Kuramoto + knot-insertion retraction')
    print('=' * 68)

    if not os.path.exists('ik_solutions.json'):
        print('  ik_solutions.json not found -- run step_21 first'); sys.exit(1)
    with open('ik_solutions.json') as fh:
        ik = json.load(fh)

    names = sorted([k for k in ik if k.startswith('dsr')])
    if not names:
        print('  no arm data'); sys.exit(1)

    starts = {n: np.array(ik[n]['start_joints'], float) for n in names}
    targets = {n: np.array(ik[n]['target_joints'], float) for n in names}
    bases = {n: np.array(ik[n].get('base', ROBOT_BASES[n]), float) for n in names}
    target_world = {n: ik[n].get('target_pos_world',
                    ik[n].get('target_world', (C.fk(targets[n])[0] + bases[n]).tolist()))
                    for n in names}

    print('  Arms     :', names)
    print('  Planning ...')
    res = C.plan_arms(starts, targets, bases, verbose=True)

    dur = float(res['duration'])
    t_vec = np.linspace(0.0, dur, len(res['time']))

    out = {'duration': dur, 'outcome': res['outcome']}
    for n in names:
        pos = np.asarray(res['positions'][n], float)
        vel, acc = _derivatives(pos, dur)
        out[n] = {
            'robot_name': n,
            'metadata': {
                'duration': dur,
                'n_samples': len(pos),
                'start_joints': starts[n].tolist(),
                'end_joints': targets[n].tolist(),
                'target_world_pos': list(target_world[n]),
                'outcome': res['outcome'],
                'refine_iterations': int(res['rounds']),
            },
            'trajectory': {
                'time': t_vec.tolist(),
                'positions': pos.tolist(),
                'velocities': vel.tolist(),
                'accelerations': acc.tolist(),
                'n_samples': len(pos),
            },
        }

    # synchronisation summary (final pairwise clearances on the planned motion)
    pair_rep = {}
    Kp = min(len(out[n]['trajectory']['positions']) for n in names)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ni, nj = names[i], names[j]
            pi = np.asarray(out[ni]['trajectory']['positions'])
            pj = np.asarray(out[nj]['trajectory']['positions'])
            dmin = min(C.pair_min_dist(pi[k], bases[ni], pj[k], bases[nj])
                       for k in range(Kp))
            nc = sum(1 for k in range(Kp)
                     if C.pair_collides(pi[k], bases[ni], pj[k], bases[nj]))
            pair_rep[f'{ni}<->{nj}'] = {
                'min_dist_m': round(float(dmin), 5),
                'collisions': nc, 'collision_free': nc == 0}

    out['synchronisation_report'] = {
        'pair_reports': pair_rep,
        'collision_free': bool(res['collision_free']),
        'residual_collision_steps': int(res['residual_collision_steps']),
        'boundary_pairs': [list(p) for p in res['boundary_pairs']],
    }
    out['parameters'] = {
        'k_base': C.K_BASE, 'min_safe_dist': C.MIN_SAFE, 'degree': C.DEG,
        'seed_ncp': C.SEED_NCP, 'max_refine': C.MAX_REFINE,
        'boundary_lo': C.BOUNDARY_LO, 'boundary_hi': C.BOUNDARY_HI,
        'n_arms': len(names),
    }

    with open('synchronized_trajectories.json', 'w') as fh:
        json.dump(out, fh, indent=2)
    kb = os.path.getsize('synchronized_trajectories.json') / 1024.0
    icon = 'OK' if res['collision_free'] else '!!'
    print('\n  %s  outcome=%s  rounds=%d  dur=%.2fs  residual=%d'
          % (icon, res['outcome'], res['rounds'], dur, res['residual_collision_steps']))
    print('  Saved: synchronized_trajectories.json  (%.1f KB)  ->  run step_25.py' % kb)


if __name__ == '__main__':
    main()