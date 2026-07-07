#!/usr/bin/env python3
"""step_23.py -- Collision Scan (REAL MESH, deepest-link) [6 ARM]
INPUT s22_trajectories.json   OUTPUT s23_collision_map.json
Scans all 15 arm pairs on the real M1013 mesh (via _robot6x -> mesh_collision).
Per colliding timestep records the deepest link-link contact for the resolver.
Collision = real SURFACE distance < CLEAR_M (5cm; 0=touching). ASCII only."""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from _robot6x import (NDOF, ROBOT_BASES, ARM_NAMES, LINK_NAMES,
                       link_origins, deepest_link_pair, pair_min_dist)
CLEAR_M = float(os.environ.get('DUAL_ARM_CLEARANCE_M', '0.05'))

def scan_pair(ni, nj, pi, pj, bi, bj):
    K = min(len(pi), len(pj)); arc = np.linspace(0., 1., K)
    coll = np.zeros(K, bool); dmin = np.full(K, np.inf)
    detail = []; worst = 0.; first = -1
    for k in range(K):
        d = pair_min_dist(pi[k], bi, pj[k], bj); dmin[k] = d
        if d < CLEAR_M:                                   # real mesh collision
            coll[k] = True
            if first < 0: first = k
            res = deepest_link_pair(pi[k], bi, pj[k], bj)
            if res is not None:
                a, b, pen = res
                if pen > worst: worst = pen
                oi = link_origins(pi[k], bi); oj = link_origins(pj[k], bj)
                detail.append({'timestep': int(k), 'arc': float(arc[k]),
                    'link_i': int(a), 'link_j': int(b),
                    'link_i_name': LINK_NAMES[a], 'link_j_name': LINK_NAMES[b],
                    'pen_m': round(float(pen), 4),
                    'pos_link_i': oi[a].tolist(), 'pos_link_j': oj[b].tolist()})
    n_coll = int(np.sum(coll))
    return {'pair': f'{ni}<->{nj}', 'arm_i': ni, 'arm_j': nj,
            'status': 'COLLISION' if n_coll > 0 else 'SAFE',
            'steps_checked': K, 'coll_steps': n_coll,
            'global_min_dist_m': round(float(np.min(dmin)), 4),
            'first_collision_arc': round(float(arc[first]), 5) if first >= 0 else None,
            'worst_penetration_m': round(worst, 4), 'per_timestep_detail': detail}

def main():
    print('\n' + '='*66)
    print('  STEP 23  --  Collision Scan (REAL MESH, deepest-link) [6 ARM]')
    print('='*66)
    if not os.path.exists('s22_trajectories.json'):
        print('  s22_trajectories.json not found'); sys.exit(1)
    with open('s22_trajectories.json') as fh: t = json.load(fh)
    arm_names = t.get('arm_names', ARM_NAMES)
    pos = {n: np.array(t[n]['trajectory']['positions'], float) for n in arm_names}
    bases = {n: np.array(ROBOT_BASES.get(n, [0, 0, 0])) for n in arm_names}
    N = len(arm_names)
    pairs = [(arm_names[i], arm_names[j]) for i in range(N) for j in range(i+1, N)]
    print(f'\n  Arms: {arm_names}  ({len(pairs)} pairs)  clearance gate={CLEAR_M*100:.0f}cm')
    results = []; t0 = time.time()
    for ni, nj in pairs:
        r = scan_pair(ni, nj, pos[ni], pos[nj], bases[ni], bases[nj])
        if r['status'] == 'COLLISION':
            print(f'  [FAIL] {ni}<->{nj}  min={r["global_min_dist_m"]*100:.1f}cm  '
                  f'coll={r["coll_steps"]}  worst_pen={r["worst_penetration_m"]*100:.1f}cm')
        results.append(r)
    n_coll = sum(1 for r in results if r['status'] == 'COLLISION')
    out = {'overall_status': 'COLLISION' if n_coll else 'SAFE', 'n_pairs': len(pairs),
           'n_colliding_pairs': n_coll, 'scan_time_ms': round((time.time()-t0)*1000, 1), 'pairs': results}
    with open('s23_collision_map.json', 'w') as fh: json.dump(out, fh, indent=2)
    print(f'\n  Overall: {out["overall_status"]}  ({n_coll}/{len(pairs)} pairs colliding)')
    print(f'  Saved: s23_collision_map.json   Next : ros2 run dual_arm_sync step_24\n')

if __name__ == '__main__': main()