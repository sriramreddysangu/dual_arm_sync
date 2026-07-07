#!/usr/bin/env python3
"""
step_8.py  --  Simultaneous Collision Check
===============================================================================
Input  : trajectories.json
Output : collision_report.json

LOGIC
-----
Both arms move simultaneously.
At every timestep check all 36 link-sphere pairs.
Map each collision back to the exact spline segment (0-4) it belongs to.

Output per pair:
  - Which segments have collision (segment index 0-4)
  - Per-link detail: which link of arm-i hits which link of arm-j
  - First collision time + arc-fraction + segment
  - Worst penetration depth + location
  - Clean segment status bar: ░░█░░
===============================================================================
"""

import json, os, sys
from typing import Dict, List
import numpy as np

_PI   = np.pi
_PI_2 = np.pi / 2.0
L1, L2, L3, L4 = 0.1525, 0.6200, 0.5590, 0.1210
A = 0.0345

DH = np.array([
    [0.0,    0.0,  0.0,    L1],
    [-_PI_2, 0.0, -_PI_2,  A ],
    [0.0,    L2,   _PI_2,  0.0],
    [_PI_2,  0.0,  0.0,    L3],
    [-_PI_2, 0.0,  0.0,    0.0],
    [_PI_2,  0.0,  0.0,    L4],
], dtype=float)

NDOF = 6
ROBOT_BASES: Dict[str, np.ndarray] = {
    'dsr01': np.array([0.0,  0.5, 0.0]),
    'dsr02': np.array([0.0, -0.5, 0.0]),
}

LINK_RADII  = np.array([0.10, 0.10, 0.08, 0.08, 0.06, 0.06])
LINK_NAMES  = ['base', 'shoulder', 'upper_arm', 'forearm', 'wrist1', 'wrist2']
SAFETY_MAR  = 0.12
WARNING_MAR = 0.20


# ─────────────────────────────────────────────────────────────────────────────
# JSON ENCODER  — numpy types are not JSON-serializable by default
# ─────────────────────────────────────────────────────────────────────────────

class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):    return bool(obj)
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


# ─────────────────────────────────────────────────────────────────────────────
# FK — LINK ORIGINS
# ─────────────────────────────────────────────────────────────────────────────

def link_origins(q: np.ndarray, base: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    o = np.zeros((NDOF, 3))
    for i in range(NDOF):
        al, a, to, d = DH[i]
        th = q[i] + to
        ct, st = np.cos(th), np.sin(th)
        ca, sa = np.cos(al),  np.sin(al)
        T = T @ np.array([
            [ct,    -st,    0.,  a    ],
            [st*ca,  ct*ca, -sa, -sa*d],
            [st*sa,  ct*sa,  ca,  ca*d],
            [0.,     0.,    0.,  1.   ],
        ])
        o[i] = T[:3, 3] + base
    return o


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE TIMESTEP: CHECK ALL 36 LINK PAIRS
# ─────────────────────────────────────────────────────────────────────────────

def check_timestep(oi: np.ndarray, oj: np.ndarray) -> List[Dict]:
    events = []
    for li in range(NDOF):
        for lj in range(NDOF):
            dist = float(np.linalg.norm(oi[li] - oj[lj]))
            thr  = float(LINK_RADII[li] + LINK_RADII[lj] + SAFETY_MAR)
            wrn  = float(LINK_RADII[li] + LINK_RADII[lj] + WARNING_MAR)
            if dist < wrn:
                events.append({
                    'li'           : int(li),
                    'lj'           : int(lj),
                    'link_name_i'  : LINK_NAMES[li],
                    'link_name_j'  : LINK_NAMES[lj],
                    'distance_m'   : round(dist, 5),
                    'threshold_m'  : round(thr, 5),
                    'penetration_m': round(float(max(thr - dist, 0.0)), 5),
                    'collision'    : bool(dist < thr),
                    'warning'      : bool(thr <= dist < wrn),
                })
    return events


# ─────────────────────────────────────────────────────────────────────────────
# FULL SCAN: ONE ARM PAIR
# ─────────────────────────────────────────────────────────────────────────────

def scan_pair(name_i: str, name_j: str,
              pos_i: np.ndarray, pos_j: np.ndarray,
              t_vec: np.ndarray, arc_i: np.ndarray,
              base_i: np.ndarray, base_j: np.ndarray,
              n_seg: int) -> Dict:

    N = min(len(pos_i), len(pos_j), len(t_vec))

    seg_n_coll  = [0]   * n_seg
    seg_n_warn  = [0]   * n_seg
    seg_max_pen = [0.0] * n_seg

    lp_min  = {(li, lj): float('inf') for li in range(NDOF) for lj in range(NDOF)}
    lp_nc   = {(li, lj): 0            for li in range(NDOF) for lj in range(NDOF)}
    lp_nw   = {(li, lj): 0            for li in range(NDOF) for lj in range(NDOF)}
    lp_maxp = {(li, lj): 0.0          for li in range(NDOF) for lj in range(NDOF)}

    n_coll_total = 0
    n_warn_total = 0
    first_coll_k = -1
    worst_pen    = 0.0
    worst_k = worst_li = worst_lj = -1
    global_min   = float('inf')
    timeline     = []

    for k in range(N):
        oi  = link_origins(pos_i[k], base_i)
        oj  = link_origins(pos_j[k], base_j)
        evs = check_timestep(oi, oj)

        any_c = any_w = False

        for e in evs:
            li, lj = e['li'], e['lj']
            d = e['distance_m']; p = e['penetration_m']
            lp_min[(li, lj)]  = min(lp_min[(li, lj)], d)
            global_min        = min(global_min, d)
            if e['collision']:
                lp_nc[(li, lj)]   += 1
                lp_maxp[(li, lj)]  = max(lp_maxp[(li, lj)], p)
                any_c = True
                if p > worst_pen:
                    worst_pen = p; worst_k = k
                    worst_li  = li; worst_lj = lj
            if e['warning']:
                lp_nw[(li, lj)] += 1; any_w = True

        arc_s = float(arc_i[k]) if k < len(arc_i) else k / max(N - 1, 1)
        seg   = int(min(int(arc_s * n_seg), n_seg - 1))

        if any_c:
            n_coll_total    += 1
            seg_n_coll[seg] += 1
            seg_max_pen[seg] = max(seg_max_pen[seg], worst_pen)
            if first_coll_k < 0: first_coll_k = k
        if any_w:
            n_warn_total    += 1
            seg_n_warn[seg] += 1

        if evs:
            timeline.append({
                'step'     : int(k),
                'time_s'   : round(float(t_vec[k]), 4),
                'arc_frac' : round(arc_s, 4),
                'segment'  : seg,
                'collision': bool(any_c),
                'warning'  : bool(any_w),
                'events'   : evs,
            })

    # Per-segment summary
    segment_summary = []
    for seg in range(n_seg):
        s0    = seg / n_seg; s1 = (seg + 1) / n_seg
        _coll = bool(seg_n_coll[seg] > 0)
        _warn = bool(seg_n_warn[seg] > 0 and not _coll)
        segment_summary.append({
            'segment'      : int(seg),
            'arc_start'    : round(s0, 4),
            'arc_end'      : round(s1, 4),
            'arc_mid'      : round((s0 + s1) / 2, 4),
            'n_coll_steps' : int(seg_n_coll[seg]),
            'n_warn_steps' : int(seg_n_warn[seg]),
            'max_pen_m'    : round(float(seg_max_pen[seg]), 5),
            'collision'    : _coll,
            'warning'      : _warn,
            'status'       : ('COLLISION' if _coll else 'WARNING' if _warn else 'SAFE'),
        })

    # Per link-pair summary
    link_summary = []
    for li in range(NDOF):
        for lj in range(NDOF):
            mn  = lp_min[(li, lj)]
            thr = float(LINK_RADII[li] + LINK_RADII[lj] + SAFETY_MAR)
            wrn = float(LINK_RADII[li] + LINK_RADII[lj] + WARNING_MAR)
            if mn < wrn:
                _nc = int(lp_nc[(li, lj)])
                _nw = int(lp_nw[(li, lj)])
                link_summary.append({
                    'arm_i_link'       : '{}.{} (link_{})'.format(name_i, LINK_NAMES[li], li),
                    'arm_j_link'       : '{}.{} (link_{})'.format(name_j, LINK_NAMES[lj], lj),
                    'min_distance_m'   : round(float(mn), 5),
                    'clearance_m'      : round(float(mn - thr), 5),
                    'max_penetration_m': round(float(lp_maxp[(li, lj)]), 5),
                    'n_collision_steps': _nc,
                    'n_warning_steps'  : _nw,
                    'status'           : ('COLLISION' if _nc > 0
                                          else 'WARNING' if _nw > 0 else 'CLOSE'),
                })
    link_summary.sort(key=lambda x: x['clearance_m'])

    # Worst penetration detail
    worst_info = None
    if worst_k >= 0:
        arc_wk = float(arc_i[worst_k]) if worst_k < len(arc_i) \
            else worst_k / max(N - 1, 1)
        worst_info = {
            'arm_i_link'   : '{}.{} (link_{})'.format(name_i, LINK_NAMES[worst_li], worst_li),
            'arm_j_link'   : '{}.{} (link_{})'.format(name_j, LINK_NAMES[worst_lj], worst_lj),
            'time_s'       : round(float(t_vec[worst_k]), 4),
            'arc_frac'     : round(arc_wk, 4),
            'segment'      : int(min(int(arc_wk * n_seg), n_seg - 1)),
            'penetration_m': round(float(worst_pen), 5),
        }

    coll_segs  = [s for s in segment_summary if s['collision']]
    frac_fc    = first_coll_k / max(N - 1, 1) if first_coll_k >= 0 else None
    overall    = ('COLLISION' if n_coll_total > 0
                  else 'WARNING' if n_warn_total > 0 else 'SAFE')

    return {
        'pair'                  : '{} <-> {}'.format(name_i, name_j),
        'arm_i'                 : name_i,
        'arm_j'                 : name_j,
        'overall_status'        : overall,
        'collision_free'        : bool(n_coll_total == 0),
        'n_steps_checked'       : int(N),
        'n_collision_steps'     : int(n_coll_total),
        'n_warning_steps'       : int(n_warn_total),
        'collision_pct'         : round(n_coll_total / max(N, 1) * 100, 2),
        'global_min_dist_m'     : round(float(global_min), 5) if global_min < 1e9 else None,
        'first_collision_time_s': round(float(t_vec[first_coll_k]), 4) if first_coll_k >= 0 else None,
        'first_collision_frac'  : round(float(frac_fc), 4) if frac_fc is not None else None,
        'first_collision_seg'   : int(min(int(frac_fc * n_seg), n_seg - 1)) if frac_fc is not None else None,
        'worst_penetration'     : worst_info,
        'segment_summary'       : segment_summary,
        'colliding_segments'    : coll_segs,
        'link_pair_summary'     : link_summary,
        'timeline'              : timeline,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PRINT REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_report(report: Dict):
    bar  = '=' * 68
    thin = '-' * 68
    print('\n{}\n  STEP 8  --  COLLISION REPORT (simultaneous motion)\n{}'.format(bar, bar))

    n_seg = report['n_seg']

    for pr in report['pair_reports']:
        status = pr['overall_status']
        icon   = 'OK' if status == 'SAFE' else ('WARN' if status == 'WARNING' else 'FAIL')
        print('\n  [{}]  {}  ->  {}'.format(icon, pr['pair'], status))
        print('      Steps checked    : {}'.format(pr['n_steps_checked']))
        print('      Collision steps  : {} ({:.2f}%)'.format(
            pr['n_collision_steps'], pr['collision_pct']))
        print('      Warning  steps   : {}'.format(pr['n_warning_steps']))
        gm = pr['global_min_dist_m']
        if gm is not None:
            print('      Global min dist  : {:.2f}cm'.format(gm * 100))

        if pr['first_collision_frac'] is not None:
            print('      First collision  : t={:.3f}s  arc={:.3f}  segment={}'.format(
                pr['first_collision_time_s'],
                pr['first_collision_frac'],
                pr['first_collision_seg']))

        wp = pr['worst_penetration']
        if wp:
            print('      Worst penetration: {:.2f}cm  @ t={:.3f}s  seg={}'.format(
                wp['penetration_m'] * 100, wp['time_s'], wp['segment']))
            print('        {}'.format(wp['arm_i_link']))
            print('        <-> {}'.format(wp['arm_j_link']))

        bar_str = ''
        labels  = []
        for ss in pr['segment_summary']:
            sym = '#' if ss['collision'] else ('~' if ss['warning'] else '.')
            bar_str += sym
            labels.append('seg{}:{}'.format(ss['segment'], ss['status'][0]))
        print('\n      SEGMENTS [{}]:  {}  (#=COLLISION  ~=WARNING  .=SAFE)'.format(
            n_seg, bar_str))
        print('      Labels:  {}'.format(' | '.join(labels)))

        for ss in pr['segment_summary']:
            if ss['collision']:
                print('      [FAIL] seg{} arc=[{:.2f},{:.2f}]  coll_steps={}  max_pen={:.2f}cm'.format(
                    ss['segment'], ss['arc_start'], ss['arc_end'],
                    ss['n_coll_steps'], ss['max_pen_m'] * 100))

        lps = pr['link_pair_summary']
        if lps:
            print('\n      LINK-PAIR DETAIL:')
            hdr = '      {:<34} {:<34} {:>8} {:>10} {:>5}'.format(
                'Arm-i link', 'Arm-j link', 'MinDist', 'Clearance', 'Coll')
            print(hdr)
            print('      ' + thin)
            for lp in lps[:8]:
                print('      {:<34} {:<34} {:>6.2f}cm {:>8.2f}cm {:>5}'.format(
                    lp['arm_i_link'], lp['arm_j_link'],
                    lp['min_distance_m'] * 100, lp['clearance_m'] * 100,
                    lp['n_collision_steps']))

    any_c = any(p['n_collision_steps'] > 0 for p in report['pair_reports'])
    any_w = any(p['n_warning_steps']   > 0 for p in report['pair_reports'])
    print('\n' + bar)
    if any_c:
        print('  COLLISION -> step_9 will resolve via Kuramoto + alternating IK')
    elif any_w:
        print('  WARNING   -> step_9 Kuramoto timing should resolve')
    else:
        print('  ALL CLEAR')
    print(bar + '\n')


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print('\n' + '=' * 68)
    print('  STEP 8  --  Simultaneous Collision Check')
    print('=' * 68)

    if not os.path.exists('trajectories.json'):
        print('\n  FAIL  trajectories.json not found -- run step_7 first')
        sys.exit(1)

    with open('trajectories.json') as fh:
        tdata = json.load(fh)

    arm_names = sorted([k for k in tdata if k.startswith('dsr')])
    if not arm_names:
        print('\n  FAIL  No arm data in trajectories.json')
        sys.exit(1)

    n_seg = int(tdata[arm_names[0]]['spline']['n_seg'])
    print('\n  Arms    : {}'.format(arm_names))
    print('  Segments: {} per arm'.format(n_seg))

    max_n   = max(len(tdata[n]['trajectory']['positions']) for n in arm_names)
    max_dur = max(float(tdata[n]['metadata']['duration'])  for n in arm_names)
    t_ref   = np.linspace(0.0, max_dur, max_n)

    arm_pos: Dict[str, np.ndarray] = {}
    arm_arc: Dict[str, np.ndarray] = {}
    for name in arm_names:
        pos_r = np.array(tdata[name]['trajectory']['positions'], dtype=float)
        arc_r = np.array(tdata[name]['trajectory'].get(
            'arc_fracs', np.linspace(0, 1, len(pos_r))), dtype=float)
        if len(pos_r) == max_n:
            arm_pos[name] = pos_r
            arm_arc[name] = arc_r
        else:
            s_in = np.linspace(0, 1, len(pos_r))
            s_out = np.linspace(0, 1, max_n)
            r = np.zeros((max_n, NDOF))
            for j in range(NDOF):
                r[:, j] = np.interp(s_out, s_in, pos_r[:, j])
            arm_pos[name] = r
            arm_arc[name] = np.interp(s_out, s_in, arc_r)

    pairs = [(arm_names[i], arm_names[j])
             for i in range(len(arm_names))
             for j in range(i + 1, len(arm_names))]

    pair_reports = []
    for (ni, nj) in pairs:
        print('\n  Scanning {} <-> {}  ({} steps simultaneously) ...'.format(ni, nj, max_n))
        bi = ROBOT_BASES.get(ni, np.zeros(3))
        bj = ROBOT_BASES.get(nj, np.zeros(3))
        pr = scan_pair(ni, nj, arm_pos[ni], arm_pos[nj],
                       t_ref, arm_arc[ni], bi, bj, n_seg)
        gm = pr['global_min_dist_m']
        print('  -> {}  coll_steps={}  {}'.format(
            pr['overall_status'], pr['n_collision_steps'],
            'min_dist={:.2f}cm'.format(gm * 100) if gm is not None else ''))
        pair_reports.append(pr)

    # Single-arm mode: no pairs to scan → always SAFE
    if not pairs:
        print('\n  Single-arm mode: no pairs to check -> SAFE')

    any_c = any(p['n_collision_steps'] > 0 for p in pair_reports)
    any_w = any(p['n_warning_steps']   > 0 for p in pair_reports)

    report = {
        'overall_status' : 'COLLISION' if any_c else 'WARNING' if any_w else 'SAFE',
        'collision_free' : bool(not any_c),
        'n_arms'         : int(len(arm_names)),
        'arm_names'      : arm_names,
        'n_seg'          : int(n_seg),
        'scan_params': {
            'link_radii_m'    : LINK_RADII.tolist(),
            'link_names'      : LINK_NAMES,
            'safety_margin_m' : float(SAFETY_MAR),
            'warning_margin_m': float(WARNING_MAR),
            'n_timesteps'     : int(max_n),
            'duration_s'      : float(max_dur),
        },
        'pair_reports': pair_reports,
    }

    print_report(report)

    with open('collision_report.json', 'w') as fh:
        json.dump(report, fh, indent=2, cls=_NpEncoder)

    kb = os.path.getsize('collision_report.json') / 1024.0
    print('  OK  Saved: collision_report.json  ({:.1f} KB)'.format(kb))
    print('  Next  ->  ros2 run dual_arm_sync step_9\n')


if __name__ == '__main__':
    main()