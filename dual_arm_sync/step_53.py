#!/usr/bin/env python3
"""
step_53.py  --  Collision Scan (Hierarchical Pipeline)
=========================================================
INPUT  : s52_trajectories.json
OUTPUT : s53_collision_map.json
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from _robot5x import (NDOF, ROBOT_BASES, ARM_NAMES, LINK_RADII, SAFETY_MARGIN,
                       link_origins, pair_min_dist)


def scan_pair(name_i, name_j, pos_i, pos_j, base_i, base_j, arc, n_seg):
    K = min(len(pos_i), len(pos_j), len(arc))
    coll = np.zeros(K, dtype=bool)
    dists = np.full(K, np.inf)
    first_k = -1; worst = 0.

    for k in range(K):
        oi = link_origins(pos_i[k], base_i)
        oj = link_origins(pos_j[k], base_j)
        diff = oi[:, None, :] - oj[None, :, :]
        d_mat = np.linalg.norm(diff, axis=2)
        thresh = (LINK_RADII[:, None] + LINK_RADII[None, :]) + SAFETY_MARGIN
        pen = thresh - d_mat
        dists[k] = float(np.min(d_mat))
        if np.any(pen > 0):
            coll[k] = True
            if first_k < 0: first_k = k
            worst = max(worst, float(np.max(pen)))

    n_coll = int(np.sum(coll))
    BAR = ''
    for seg in range(n_seg):
        s0, s1 = seg / n_seg, (seg + 1) / n_seg
        mask = (arc[:K] >= s0 - 1e-9) & (arc[:K] <= s1 + 1e-9)
        BAR += '#' if int(np.sum(coll[:K][mask])) > 0 else '.'

    return {
        'pair': f'{name_i}<->{name_j}',
        'arm_i': name_i, 'arm_j': name_j,
        'status': 'COLLISION' if n_coll > 0 else 'SAFE',
        'steps_checked': K, 'coll_steps': n_coll,
        'global_min_dist_m': round(float(np.min(dists)), 4),
        'first_collision_arc': round(float(arc[first_k]), 5) if first_k >= 0 else None,
        'first_coll_seg': (min(int(arc[first_k] * n_seg), n_seg - 1)
                           if first_k >= 0 else None),
        'worst_penetration_m': round(worst, 4),
        'segment_bar': BAR,
    }


def main():
    print('\n' + '='*66)
    print('  STEP 53  --  Collision Scan')
    print('='*66)
    if not os.path.exists('s52_trajectories.json'):
        print('  s52_trajectories.json not found'); sys.exit(1)
    with open('s52_trajectories.json') as fh: tdata = json.load(fh)

    arm_names = tdata.get('arm_names', ARM_NAMES)
    n_seg = int(tdata[arm_names[0]]['spline']['n_seg'])
    arm_pos = {n: np.array(tdata[n]['trajectory']['positions'], dtype=float)
                for n in arm_names}
    bases = {n: np.array(ROBOT_BASES.get(n, [0, 0, 0])) for n in arm_names}
    print(f'\n  Arms: {arm_names}  segments: {n_seg}')

    N = len(arm_names)
    pairs = [(arm_names[i], arm_names[j]) for i in range(N) for j in range(i+1, N)]
    results = []; t0 = time.time()

    for ni, nj in pairs:
        K = min(len(arm_pos[ni]), len(arm_pos[nj]))
        arc = np.linspace(0., 1., K)
        res = scan_pair(ni, nj, arm_pos[ni], arm_pos[nj],
                         bases[ni], bases[nj], arc, n_seg)
        icon = '[FAIL]' if res['status'] == 'COLLISION' else '[ OK ]'
        print(f'  {icon}  {ni}<->{nj}  {res["status"]}  '
              f'min={res["global_min_dist_m"]*100:.1f}cm  '
              f'coll={res["coll_steps"]}  bar={res["segment_bar"]}')
        if res['status'] == 'COLLISION':
            print(f'         worst_pen={res["worst_penetration_m"]*100:.1f}cm  '
                  f'first_seg={res["first_coll_seg"]}')
        results.append(res)

    n_coll = sum(1 for r in results if r['status'] == 'COLLISION')
    overall = 'COLLISION' if n_coll > 0 else 'SAFE'
    out = {'overall_status': overall, 'n_pairs': len(pairs),
            'n_colliding_pairs': n_coll,
            'scan_time_ms': round((time.time() - t0) * 1000, 1),
            'pairs': results}
    with open('s53_collision_map.json', 'w') as fh: json.dump(out, fh, indent=2)
    print(f'\n  Overall: {overall}  ({n_coll}/{len(pairs)} pairs colliding)')
    print(f'  Saved: s53_collision_map.json')
    print(f'  Next : ros2 run dual_arm_sync step_54\n')


if __name__ == '__main__': main()