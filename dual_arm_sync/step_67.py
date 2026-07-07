#!/usr/bin/env python3
"""
step_67.py  --  Paper Metrics Collector
=========================================
INPUT  : s61_ik.json + s62_trajectories.json + s63_collision_map.json
         + s64_resolved.json + s65_synchronized.json + s66_execution.json
OUTPUT : s67_metrics.json

Computes all metrics needed for the paper's results table:
  PLANNING STAGE:
    ik_time_ms, bspline_time_ms, resolve_time_ms, kuramoto_time_ms
    total_plan_time_ms
  QUALITY:
    surgical_mod_magnitude    -- ||CP_new - CP_orig||_F (modified arms only)
    unmodified_arms_unchanged -- bool: non-colliding arms untouched
    optimality_ratio          -- path length vs direct EE line
    duration_overhead_pct
  SAFETY:
    collision_free_after_resolve  -- bool
    min_inter_arm_dist_m
    coll_fraction_before          -- from step_63
    coll_fraction_after           -- zero if resolved
  EXECUTION:
    ee_error_mm, joint_error_deg  -- actual hardware error
    success
"""

import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _robot import ARM_NAMES, ROBOT_BASES, NDOF, fk_world, pair_min_dist


def _load(fname):
    if not os.path.exists(fname): return None
    with open(fname) as fh: return json.load(fh)


def main():
    print('\n' + '='*66); print('  STEP 67  --  Paper Metrics Collector'); print('='*66)

    ik   = _load('s61_ik.json')
    trj  = _load('s62_trajectories.json')
    cmap = _load('s63_collision_map.json')
    res  = _load('s64_resolved.json')
    sync = _load('s65_synchronized.json')
    exc  = _load('s66_execution.json')

    if not all([ik, trj, cmap, res, sync]):
        print('  Some input files missing -- run steps 61-66 first'); sys.exit(1)

    arm_names = ik.get('arm_names', ARM_NAMES)
    bases     = {n: np.array(ROBOT_BASES.get(n,[0,0,0])) for n in arm_names}

    # ── Timing ────────────────────────────────────────────────────────────────
    ik_ms  = float(ik.get('ik_total_time_ms', 0))
    bs_ms  = float(trj.get('total_bspline_time_ms', 0))
    res_ms = float(res.get('resolve_time_ms', 0))
    kur_ms = float(sync.get('kuramoto_time_ms', 0))
    total_ms = ik_ms + bs_ms + res_ms + kur_ms

    # ── Surgical modification ─────────────────────────────────────────────────
    mod_mag = res.get('modification_magnitude', {})
    # Which arms were involved in collisions?
    coll_arms = set()
    for pair_info in cmap.get('pairs', []):
        if pair_info['status'] == 'COLLISION':
            coll_arms.add(pair_info['arm_i']); coll_arms.add(pair_info['arm_j'])
    safe_arms = set(arm_names) - coll_arms

    # Non-colliding arms must have zero modification magnitude
    unmod_clean = all(abs(mod_mag.get(n, 0.)) < 1e-9 for n in safe_arms)

    # ── Optimality ratio ──────────────────────────────────────────────────────
    opt_ratios = [float(trj[n]['metadata'].get('optimality_ratio', 1.)) for n in arm_names]
    mean_opt   = float(np.mean(opt_ratios))

    # ── Duration overhead ─────────────────────────────────────────────────────
    dur_orig   = float(ik.get('duration', 10.))
    dur_final  = float(sync.get('final_duration_s', dur_orig))
    dur_oh_pct = round((dur_final / dur_orig - 1) * 100, 2)

    # ── Safety ────────────────────────────────────────────────────────────────
    coll_before = max((p['collision_fraction'] for p in cmap.get('pairs', [])
                       if p['status']=='COLLISION'), default=0.)
    kur_rep = sync.get('synchronisation_report', {})
    cf_after = kur_rep.get('collision_free', True)
    all_pair_min = [pr.get('min_dist_m', 1.) for pr in kur_rep.get('pair_reports',{}).values()]
    global_min_m = min(all_pair_min) if all_pair_min else 1.

    # ── Execution ─────────────────────────────────────────────────────────────
    exec_ok = False; ee_errs = []; jt_errs = []
    if exc:
        exec_ok = exc.get('execution', {}).get('success', False)
        for name in arm_names:
            v = exc.get('verification', {}).get(name, {})
            if v.get('ee_error_mm') is not None:
                ee_errs.append(float(v['ee_error_mm']))
                jt_errs.append(float(v['joint_error_deg']))

    metrics = {
        'arm_names' : arm_names,
        'planning': {
            'ik_time_ms'       : round(ik_ms, 1),
            'bspline_time_ms'  : round(bs_ms, 1),
            'resolve_time_ms'  : round(res_ms, 1),
            'kuramoto_time_ms' : round(kur_ms, 1),
            'total_plan_ms'    : round(total_ms, 1),
        },
        'quality': {
            'surgical_mod_mag_sum'   : {n: round(float(mod_mag.get(n, 0.)), 5) for n in arm_names},
            'unmodified_arms_intact' : bool(unmod_clean),
            'mean_optimality_ratio'  : round(mean_opt, 4),
            'duration_overhead_pct'  : dur_oh_pct,
            'iterations_used'        : int(res.get('iterations_used', 0)),
            'retraction_alpha_max'   : max(
                (v.get('alpha', 0.) for v in res.get('retraction_log', {}).values()
                 if isinstance(v, dict)), default=0.),
            'retraction_phase_final' : next(
                (v.get('phase', '?') for v in reversed(list(res.get('retraction_log', {}).values()))
                 if isinstance(v, dict) and 'phase' in v), '?'),
        },
        'safety': {
            'collision_fraction_before' : round(float(coll_before), 5),
            'collision_free_after'      : bool(cf_after),
            'min_inter_arm_dist_m'      : round(float(global_min_m), 4),
            'n_colliding_pairs_before'  : int(cmap.get('n_colliding_pairs', 0)),
            'n_colliding_pairs_after'   : 0 if cf_after else '?',
        },
        'execution': {
            'success'              : bool(exec_ok),
            'mean_ee_error_mm'     : round(float(np.mean(ee_errs)), 3) if ee_errs else None,
            'max_ee_error_mm'      : round(float(np.max(ee_errs)),  3) if ee_errs else None,
            'mean_joint_error_deg' : round(float(np.mean(jt_errs)), 3) if jt_errs else None,
            'per_arm': {n: exc['verification'].get(n, {}) for n in arm_names} if exc else {},
        },
    }

    with open('s67_metrics.json', 'w') as fh: json.dump(metrics, fh, indent=2)

    print(f'\n  PLANNING TIMES')
    p = metrics['planning']
    print(f'    IK={p["ik_time_ms"]:.0f}ms  B-spline={p["bspline_time_ms"]:.0f}ms  '
          f'Resolve={p["resolve_time_ms"]:.0f}ms  Kuramoto={p["kuramoto_time_ms"]:.0f}ms')
    print(f'    Total={p["total_plan_ms"]:.0f}ms')
    q = metrics['quality']
    print(f'\n  QUALITY')
    print(f'    Mod magnitude (per arm): {q["surgical_mod_mag_sum"]}')
    print(f'    Unmodified arms intact : {q["unmodified_arms_intact"]}')
    print(f'    Optimality ratio       : {q["mean_optimality_ratio"]:.4f}')
    print(f'    Duration overhead      : {q["duration_overhead_pct"]:.1f}%')
    s = metrics['safety']
    print(f'\n  SAFETY')
    print(f'    Collision before: {s["collision_fraction_before"]*100:.1f}%  '
          f'After: {"SAFE" if s["collision_free_after"] else "FAIL"}')
    print(f'    Min inter-arm dist: {s["min_inter_arm_dist_m"]*100:.1f}cm')
    if exc:
        e = metrics['execution']
        print(f'\n  EXECUTION  success={e["success"]}')
        if e['mean_ee_error_mm'] is not None:
            print(f'    Mean EE error: {e["mean_ee_error_mm"]:.2f}mm  '
                  f'Max: {e["max_ee_error_mm"]:.2f}mm')
    print(f'\n  Saved: s67_metrics.json\n')

if __name__ == '__main__': main()