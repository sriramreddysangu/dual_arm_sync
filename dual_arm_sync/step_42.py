#!/usr/bin/env python3
"""
step_42.py  --  Collision Scan with Deepest-Link Tracking   [DUAL ARM]
============================================================
INPUT  : s41_trajectories.json
OUTPUT : s42_collision_map.json
Dual-arm version (imports _robot2x). Logic identical to the 4-arm step_42.
"""

import json, os, sys, time
from typing import Dict, List
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _robot2x import (NDOF, ROBOT_BASES, ARM_NAMES, LINK_RADII, LINK_NAMES,
                       SAFETY_MARGIN, link_origins, deepest_link_pair,
                       pair_min_dist)

WARNING_MARGIN = 0.08


def scan_pair_detailed(name_i, name_j, pos_i, pos_j, base_i, base_j, arc_fracs, n_seg):
    K = min(len(pos_i), len(pos_j), len(arc_fracs))
    coll = np.zeros(K, dtype=bool); dists = np.full(K, np.inf)
    per_timestep_detail = []
    first_k = -1; worst_pen = 0.
    for k in range(K):
        # capsule-consistent detection (same model as step_43/44/45)
        dists[k] = pair_min_dist(pos_i[k], base_i, pos_j[k], base_j)
        res = deepest_link_pair(pos_i[k], base_i, pos_j[k], base_j)
        if res is not None:
            a, b, penetration = res
            coll[k] = True
            if first_k < 0: first_k = k
            if penetration > worst_pen: worst_pen = penetration
            oi = link_origins(pos_i[k], base_i); oj = link_origins(pos_j[k], base_j)
            per_timestep_detail.append({
                'timestep': int(k), 'arc': float(arc_fracs[k]),
                'link_i': int(a), 'link_j': int(b),
                'link_i_name': LINK_NAMES[a], 'link_j_name': LINK_NAMES[b],
                'pen_m': round(float(penetration), 4),
                'pos_link_i': oi[a].tolist(), 'pos_link_j': oj[b].tolist()})
    n_coll = int(np.sum(coll)); global_min = float(np.min(dists))
    first_arc = float(arc_fracs[first_k]) if first_k >= 0 else None
    first_seg = (min(int(first_arc * n_seg), n_seg - 1) if first_arc is not None else None)
    BAR = ''
    for seg in range(n_seg):
        s0, s1 = seg / n_seg, (seg + 1) / n_seg
        mask = (arc_fracs[:K] >= s0 - 1e-9) & (arc_fracs[:K] <= s1 + 1e-9)
        nc = int(np.sum(coll[:K][mask]))
        BAR += '#' if nc > 0 else '.'
    status = 'COLLISION' if n_coll > 0 else 'SAFE'
    return {
        'pair': f'{name_i}<->{name_j}', 'arm_i': name_i, 'arm_j': name_j,
        'status': status, 'steps_checked': K, 'coll_steps': n_coll,
        'collision_fraction': round(n_coll / K, 5) if K > 0 else 0.,
        'global_min_dist_m': round(global_min, 4),
        'first_collision_arc': round(first_arc, 5) if first_arc is not None else None,
        'first_coll_seg': first_seg, 'worst_penetration_m': round(worst_pen, 4),
        'segment_bar': BAR, 'per_timestep_detail': per_timestep_detail}


def main():
    print('\n' + '='*66)
    print('  STEP 42  --  Collision Scan with Deepest-Link Tracking [DUAL ARM]')
    print('='*66)
    if not os.path.exists('s41_trajectories.json'):
        print('  s41_trajectories.json not found'); sys.exit(1)
    with open('s41_trajectories.json') as fh: tdata = json.load(fh)
    arm_names = tdata.get('arm_names', ARM_NAMES)
    n_seg     = int(tdata[arm_names[0]]['spline']['n_seg'])
    arm_pos = {n: np.array(tdata[n]['trajectory']['positions'], dtype=float) for n in arm_names}
    bases = {n: np.array(ROBOT_BASES.get(n, [0, 0, 0])) for n in arm_names}
    print(f'\n  Arms: {arm_names}  segments: {n_seg}')
    N = len(arm_names)
    pairs = [(arm_names[i], arm_names[j]) for i in range(N) for j in range(i+1, N)]
    results = []; t0 = time.time()
    for ni, nj in pairs:
        K = min(len(arm_pos[ni]), len(arm_pos[nj]))
        print(f'\n  Scanning {ni} <-> {nj}  ({K} steps)...')
        arc_common = np.linspace(0., 1., K)
        res = scan_pair_detailed(ni, nj, arm_pos[ni], arm_pos[nj], bases[ni], bases[nj], arc_common, n_seg)
        icon = '[FAIL]' if res['status'] == 'COLLISION' else '[ OK ]'
        print(f'  {icon}  {ni}<->{nj}  {res["status"]}  '
              f'min={res["global_min_dist_m"]*100:.1f}cm  '
              f'coll={res["coll_steps"]}  bar={res["segment_bar"]}')
        if res['status'] == 'COLLISION':
            print(f'         worst_pen={res["worst_penetration_m"]*100:.1f}cm  first_seg={res["first_coll_seg"]}')
            if res['per_timestep_detail']:
                link_counts = {}
                for d in res['per_timestep_detail']:
                    key = f"{d['link_i_name']}<->{d['link_j_name']}"
                    link_counts[key] = link_counts.get(key, 0) + 1
                for lp, c in sorted(link_counts.items(), key=lambda x: -x[1])[:3]:
                    print(f'         link pair {lp}: {c} steps')
        results.append(res)
    scan_ms = round((time.time() - t0) * 1000, 1)
    n_coll  = sum(1 for r in results if r['status'] == 'COLLISION')
    overall = 'COLLISION' if n_coll > 0 else 'SAFE'
    out = {'overall_status': overall, 'n_pairs': len(pairs),
           'n_colliding_pairs': n_coll, 'scan_time_ms': scan_ms, 'pairs': results}
    with open('s42_collision_map.json', 'w') as fh: json.dump(out, fh, indent=2)
    print(f'\n  Overall: {overall}  ({n_coll}/{len(pairs)} pairs colliding)')
    print(f'  Saved: s42_collision_map.json')
    print(f'  Next : ros2 run dual_arm_sync step_43\n')


if __name__ == '__main__': main()



