#!/usr/bin/env python3
"""
step_63.py  --  Simultaneous Collision Scanner
================================================
INPUT  : s62_trajectories.json
OUTPUT : s63_collision_map.json

Checks all N*(N-1)/2 arm pairs simultaneously. For each pair:
  - Scans every timestep with both arms moving concurrently
  - Identifies WHICH segment has the first collision
  - Computes worst penetration depth, first collision time
  - Produces visual segment bar: .=safe #=collision ~=warning

Paper metrics written:
  - scan_time_ms          (how long collision checking took)
  - collision_fraction    (% of timesteps in collision per pair)
  - first_collision_arc   (arc fraction where collision first occurs)
  - worst_penetration_m   (deepest link overlap)
"""

import json, os, sys, time
from typing import Dict, List, Tuple
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _robot import (NDOF, ROBOT_BASES, ARM_NAMES, LINK_RADII, LINK_NAMES,
                    SAFETY_MARGIN, link_origins, pair_min_dist)

WARNING_MARGIN = 0.08   # metres extra beyond collision threshold -> warning zone


def scan_pair(name_i: str, name_j: str,
              pos_i: np.ndarray, pos_j: np.ndarray,
              base_i: np.ndarray, base_j: np.ndarray,
              arc_fracs: np.ndarray, n_seg: int) -> Dict:

    K    = min(len(pos_i), len(pos_j), len(arc_fracs))
    coll = np.zeros(K, dtype=bool)
    warn = np.zeros(K, dtype=bool)
    dists = np.full(K, np.inf)

    # Link-pair detail tracking
    lp_min   = np.full((NDOF, NDOF), np.inf)
    lp_coll  = np.zeros((NDOF, NDOF), dtype=int)
    first_k  = -1; worst_pen = 0.; worst_k = 0

    for k in range(K):
        oi = link_origins(pos_i[k], base_i)   # (6, 3)
        oj = link_origins(pos_j[k], base_j)   # (6, 3)
        diff   = oi[:, np.newaxis, :] - oj[np.newaxis, :, :]   # (6,6,3)
        d_mat  = np.linalg.norm(diff, axis=2)                   # (6,6)
        thresh = (LINK_RADII[:, np.newaxis] +
                  LINK_RADII[np.newaxis, :] + SAFETY_MARGIN)
        pen    = thresh - d_mat   # positive = penetration
        min_d  = float(np.min(d_mat))
        dists[k] = min_d

        if np.any(pen > 0):
            coll[k] = True
            if first_k < 0: first_k = k
            wp = float(np.max(pen))
            if wp > worst_pen: worst_pen = wp; worst_k = k
        elif np.any(pen > -WARNING_MARGIN):
            warn[k] = True
        lp_min   = np.minimum(lp_min,  d_mat)
        lp_coll += (pen > 0).astype(int)

    n_coll = int(np.sum(coll))
    n_warn = int(np.sum(warn))
    global_min = float(np.min(dists))
    first_arc  = float(arc_fracs[first_k]) if first_k >= 0 else None
    first_t    = None
    worst_arc  = float(arc_fracs[worst_k]) if n_coll > 0 else None

    # Map first collision to segment
    first_seg = None
    if first_arc is not None:
        first_seg = min(int(first_arc * n_seg), n_seg - 1)

    # Segment collision summary
    seg_data = []
    BAR = ''
    for seg in range(n_seg):
        s0 = seg / n_seg; s1 = (seg + 1) / n_seg
        mask = (arc_fracs[:K] >= s0 - 1e-9) & (arc_fracs[:K] <= s1 + 1e-9)
        nc   = int(np.sum(coll[:K][mask]))
        nw   = int(np.sum(warn[:K][mask]))
        pen_seg = float(thresh.max() - dists[:K][mask].min()) if mask.any() else 0.
        if nc > 0:  label = 'C'; BAR += '#'
        elif nw > 0: label = 'W'; BAR += '~'
        else:        label = 'S'; BAR += '.'
        seg_data.append({
            'segment': seg, 'status': label,
            'arc_start': round(s0, 3), 'arc_end': round(s1, 3),
            'coll_steps': nc, 'warn_steps': nw,
            'max_pen_m': round(max(0., pen_seg), 4),
        })

    # Link-pair detail
    link_pairs = []
    for a in range(NDOF):
        for b in range(NDOF):
            if lp_coll[a, b] == 0 and lp_min[a, b] > (LINK_RADII[a]+LINK_RADII[b]+SAFETY_MARGIN+WARNING_MARGIN):
                continue
            pen_m = (LINK_RADII[a]+LINK_RADII[b]+SAFETY_MARGIN) - lp_min[a, b]
            link_pairs.append({
                'arm_i_link': LINK_NAMES[a], 'arm_j_link': LINK_NAMES[b],
                'min_dist_m': round(float(lp_min[a, b]), 4),
                'clearance_m': round(-float(pen_m), 4),
                'coll_steps': int(lp_coll[a, b]),
            })
    link_pairs.sort(key=lambda x: x['coll_steps'], reverse=True)

    status = 'COLLISION' if n_coll > 0 else ('WARNING' if n_warn > 0 else 'SAFE')
    return {
        'pair': f'{name_i}<->{name_j}',
        'arm_i': name_i, 'arm_j': name_j,
        'status': status,
        'steps_checked': K,
        'coll_steps': n_coll,
        'warn_steps': n_warn,
        'collision_fraction': round(n_coll / K, 5) if K > 0 else 0.,
        'global_min_dist_m': round(global_min, 4),
        'first_collision_arc': round(first_arc, 5) if first_arc is not None else None,
        'first_coll_seg': first_seg,
        'worst_penetration_m': round(worst_pen, 4),
        'worst_pen_arc': round(worst_arc, 5) if worst_arc is not None else None,
        'segment_bar': BAR,
        'segments': seg_data,
        'link_pairs': link_pairs[:8],   # top 8 most colliding pairs
    }


def main():
    print('\n' + '='*66)
    print('  STEP 63  --  Simultaneous Collision Scanner')
    print('='*66)

    if not os.path.exists('s62_trajectories.json'):
        print('  s62_trajectories.json not found'); sys.exit(1)
    with open('s62_trajectories.json') as fh: tdata = json.load(fh)

    arm_names = tdata.get('arm_names', ARM_NAMES)
    n_seg     = int(tdata[arm_names[0]]['spline']['n_seg'])
    print(f'\n  Arms: {arm_names}  segments: {n_seg}')

    arm_pos  = {n: np.array(tdata[n]['trajectory']['positions'], dtype=float)
                for n in arm_names}
    arm_arcs = {n: np.array(tdata[n]['trajectory']['arc_fracs'], dtype=float)
                for n in arm_names}
    bases    = {n: np.array(ROBOT_BASES.get(n, [0, 0, 0])) for n in arm_names}

    N      = len(arm_names)
    pairs  = [(arm_names[i], arm_names[j]) for i in range(N) for j in range(i+1, N)]
    results = []
    t0     = time.time()

    for ni, nj in pairs:
        K = min(len(arm_pos[ni]), len(arm_pos[nj]))
        print(f'\n  Scanning {ni} <-> {nj}  ({K} steps)...')
        arc_common = np.linspace(0., 1., K)
        res = scan_pair(ni, nj, arm_pos[ni], arm_pos[nj],
                        bases[ni], bases[nj], arc_common, n_seg)
        icon = {'COLLISION': '[FAIL]', 'WARNING': '[WARN]', 'SAFE': '[ OK ]'}[res['status']]
        print(f'  {icon}  {ni}<->{nj}  {res["status"]:9s}  '
              f'min={res["global_min_dist_m"]*100:.1f}cm  '
              f'coll={res["coll_steps"]}  seg_bar={res["segment_bar"]}')
        if res['status'] == 'COLLISION':
            print(f'         first_coll arc={res["first_collision_arc"]:.3f}  '
                  f'seg={res["first_coll_seg"]}  '
                  f'worst_pen={res["worst_penetration_m"]*100:.1f}cm')
        results.append(res)

    scan_ms = round((time.time() - t0) * 1000, 1)
    n_coll  = sum(1 for r in results if r['status'] == 'COLLISION')
    overall = 'COLLISION' if n_coll > 0 else 'SAFE'

    out = {
        'overall_status': overall,
        'n_pairs': len(pairs),
        'n_colliding_pairs': n_coll,
        'scan_time_ms': scan_ms,
        'pairs': results,
    }
    with open('s63_collision_map.json', 'w') as fh: json.dump(out, fh, indent=2)

    print(f'\n  {"="*66}')
    print(f'  Overall: {overall}  ({n_coll}/{len(pairs)} pairs colliding)')
    print(f'  Scan time: {scan_ms:.0f} ms')
    print(f'  Saved: s63_collision_map.json')
    print(f'  Next : ros2 run dual_arm_sync step_64\n')


if __name__ == '__main__': main()