#!/usr/bin/env python3
"""
run_pipeline.py  ─  Dual-Arm Pipeline Orchestrator + 3-Stage Collision Report
Paper: Kabir ICRA 2019 / IJRR 2021  (extended)

════════════════════════════════════════════════════════════════════════════════
USAGE
════════════════════════════════════════════════════════════════════════════════

  python3 run_pipeline.py

Prompts for:
  • DSR01 target position [x, y, z] in world frame [m]
  • DSR01 target orientation quaternion [w, x, y, z]  (Enter = identity)
  • DSR02 target position + orientation  (same)
  • Trajectory duration [s]

Or pass as JSON:
  python3 run_pipeline.py --config pipeline_input.json

Example pipeline_input.json:
  {
    "dsr01": {"pos": [0.6, 0.5, 0.7], "quat": [1, 0, 0, 0]},
    "dsr02": {"pos": [0.6, -0.5, 0.7], "quat": [1, 0, 0, 0]},
    "duration": 8.0
  }

════════════════════════════════════════════════════════════════════════════════
3-STAGE COLLISION REPORT
════════════════════════════════════════════════════════════════════════════════

Stage 0 — Raw (no synchronisation):
  Run each arm's B-spline trajectory independently at nominal timing.
  Count inter-arm collisions.
  → Shows how many collisions the unsynchronised plan has.

Stage 1 — Kuramoto only (no CP refinement):
  Run Kuramoto phase synchronisation with MAX_REFINE = 0.
  Count remaining collisions after timing adjustment.
  → Shows how many collisions Kuramoto resolves through timing alone.

Stage 2 — Kuramoto + CP/Segment increase:
  Run full pipeline with MAX_REFINE = 6.
  For each CP-refinement iteration, record how many collisions remain.
  → Shows how many iterations were needed and how many CP/segments were added.

Final report table:
  ┌───────────────────────────┬────────────┬──────────────┬────────────────┐
  │ Stage                     │ Collisions │ Min dist [cm]│ Refinements    │
  ├───────────────────────────┼────────────┼──────────────┼────────────────┤
  │ 0. Raw (no sync)          │     N      │    D.D cm    │      —         │
  │ 1. Kuramoto only          │     M      │    D.D cm    │      0         │
  │ 2. Kuramoto + CP increase │     0      │    D.D cm    │      R         │
  └───────────────────────────┴────────────┴──────────────┴────────────────┘

════════════════════════════════════════════════════════════════════════════════
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Inline core imports from pipeline steps ──────────────────────────────────
# (steps are imported as modules from the same directory)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import step_1 as S1
import step_2 as S2
import step_4 as S4   # contains synchronise_all_arms

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

ROBOT_BASES: Dict[str, np.ndarray] = {
    'dsr01': np.array([0.0,  0.5, 0.0]),
    'dsr02': np.array([0.0, -0.5, 0.0]),
}

LINK_RADII    = np.array([0.10, 0.10, 0.08, 0.08, 0.06, 0.06])
SAFETY_MARGIN = 0.15

_PI   = np.pi
_PI_2 = np.pi / 2.

DH_TABLE = np.array([
    [0.,   0.,   0.,    0.1525],
    [-_PI_2, 0., -_PI_2, 0.0345],
    [0.,  0.620, _PI_2,  0.    ],
    [_PI_2, 0.,  0.,    0.559 ],
    [-_PI_2, 0., 0.,    0.    ],
    [_PI_2,  0., 0.,    0.121 ],
], dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# COLLISION CHECK HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _link_origins(q, base):
    T = np.eye(4)
    origins = np.zeros((6, 3))
    for i in range(6):
        al, a, to, d = DH_TABLE[i]
        th = float(q[i]) + to
        ct, st = np.cos(th), np.sin(th); ca, sa = np.cos(al), np.sin(al)
        T = T @ np.array([[ct,-st,0.,a],[st*ca,ct*ca,-sa,-sa*d],
                           [st*sa,ct*sa,ca,ca*d],[0.,0.,0.,1.]])
        origins[i] = T[:3,3] + base
    return origins


def _pair_min_dist(q_i, bi, q_j, bj):
    oi = _link_origins(q_i, bi); oj = _link_origins(q_j, bj)
    return float(np.min([np.linalg.norm(oi[li]-oj[lj])
                          for li in range(6) for lj in range(6)]))


def _pair_collides(q_i, bi, q_j, bj):
    oi = _link_origins(q_i, bi); oj = _link_origins(q_j, bj)
    for li in range(6):
        for lj in range(6):
            if np.linalg.norm(oi[li]-oj[lj]) < LINK_RADII[li]+LINK_RADII[lj]+SAFETY_MARGIN:
                return True
    return False


def count_collisions(traj1: np.ndarray, base1: np.ndarray,
                      traj2: np.ndarray, base2: np.ndarray) -> Tuple[int, float]:
    """
    Count inter-arm collision timesteps and global minimum distance.
    Both trajectories must be (N, 6) arrays sampled at the same times.
    """
    n  = min(len(traj1), len(traj2))
    nc = 0
    md = float('inf')
    for k in range(n):
        d = _pair_min_dist(traj1[k], base1, traj2[k], base2)
        md = min(md, d)
        if _pair_collides(traj1[k], base1, traj2[k], base2):
            nc += 1
    return nc, md


def _interp_traj(pos: np.ndarray, n_out: int) -> np.ndarray:
    """Resample (N, 6) trajectory to n_out time steps by linear interpolation."""
    N = len(pos)
    if N == n_out: return pos.copy()
    s_in  = np.linspace(0., 1., N)
    s_out = np.linspace(0., 1., n_out)
    out   = np.zeros((n_out, 6))
    for j in range(6):
        out[:, j] = np.interp(s_out, s_in, pos[:, j])
    return out


def _kuramoto_phase_sync_only(trajs: Dict, arm_bases: Dict) -> Tuple[Dict, int, float]:
    """
    Run Kuramoto phase synchronisation with MAX_REFINE=0 (no CP refinement).
    Returns synchronised position dict, collision count, min distance.
    """
    # Temporarily override MAX_REFINE
    orig_max = S4.MAX_REFINE
    S4.MAX_REFINE = 0
    try:
        result = S4.synchronise_all_arms(trajs, arm_bases)
    finally:
        S4.MAX_REFINE = orig_max

    if result is None:
        return {}, -1, 0.

    arm_names = sorted([k for k in result if k.startswith('dsr')])
    sync_pos  = {n: np.array(result[n]['trajectory']['positions']) for n in arm_names}
    t_vec     = np.array(result[arm_names[0]]['trajectory']['time'])

    nc, md = count_collisions(
        sync_pos[arm_names[0]], arm_bases[arm_names[0]],
        sync_pos[arm_names[1]], arm_bases[arm_names[1]],
    )
    return result, nc, md


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_full_pipeline(
        targets  : Dict[str, Tuple[np.ndarray, np.ndarray]],
        duration : float,
        verbose  : bool = True,
) -> Dict:
    """
    Run the complete pipeline for two arms and return a 3-stage collision report.

    targets: {'dsr01': (pos_world, quat), 'dsr02': (pos_world, quat)}
    duration: trajectory duration [s]

    Returns report dict with fields:
      stages          — list of dicts, one per stage (0, 1, 2)
      final_result    — full synchronise_all_arms output
      final_safe      — bool
      min_dist_cm     — float
      total_refine    — int
    """
    arm_names = sorted(targets.keys())
    arm_bases = {n: ROBOT_BASES[n] for n in arm_names}

    bar = '═' * 74
    if verbose:
        print(f'\n{bar}')
        print(f'  DUAL-ARM PIPELINE  —  3-Stage Collision Analysis')
        print(f'  Arms: {arm_names}')
        print(f'{bar}')

    t_pipeline_start = time.time()

    # ── STEP 1: IK solutions ─────────────────────────────────────────────────
    if verbose: print(f'\n{"─"*74}\n  STEP 1  —  SE(3) IK + 3D Arc + SLERP Orientation\n{"─"*74}')
    ik_solutions = {}
    for name in arm_names:
        pos, quat = targets[name]
        q_home    = np.zeros(6)
        sol = S1.solve_arm(name, q_home, pos, quat)
        if sol is None:
            raise RuntimeError(f'  ❌  IK failed for {name} — adjust target')
        ik_solutions[name] = sol

    # ── STEP 2: B-spline trajectories ────────────────────────────────────────
    if verbose: print(f'\n{"─"*74}\n  STEP 2  —  Adaptive B-Spline Trajectory Generation\n{"─"*74}')
    trajectories = {}
    for name in arm_names:
        traj = S2.build_trajectory(name, ik_solutions[name], duration)
        trajectories[name] = traj
        if verbose:
            m = traj['metadata']
            print(f'  {name}: nseg={m["nseg"]} ncp/seg={m["ncp_per_segment"]} '
                  f'dur={m["duration"]:.2f}s')

    stages = []

    # ════════════════════════════════════════════════════════════════════════
    # STAGE 0 — RAW (no synchronisation)
    # ════════════════════════════════════════════════════════════════════════
    if verbose:
        print(f'\n{"─"*74}')
        print(f'  STAGE 0  —  Raw Trajectories (no synchronisation)')
        print(f'{"─"*74}')

    # Resample to same number of points for fair comparison
    ns = min(len(trajectories[n]['trajectory']['positions']) for n in arm_names)
    raw_pos = {n: _interp_traj(
        np.array(trajectories[n]['trajectory']['positions']), ns)
        for n in arm_names}

    nc0, md0 = count_collisions(
        raw_pos[arm_names[0]], arm_bases[arm_names[0]],
        raw_pos[arm_names[1]], arm_bases[arm_names[1]],
    )
    stage0 = {'stage': 0, 'label': 'Raw (no sync)', 'n_collisions': nc0,
               'min_dist_cm': md0*100, 'n_refinements': 0,
               'cp_per_seg': {n: trajectories[n]['metadata']['ncp_per_segment']
                              for n in arm_names}}
    stages.append(stage0)

    if verbose:
        ok = '✅' if nc0 == 0 else '❌'
        print(f'  {ok}  Collisions: {nc0}   Min dist: {md0*100:.1f} cm')

    # ════════════════════════════════════════════════════════════════════════
    # STAGE 1 — KURAMOTO ONLY (MAX_REFINE = 0)
    # ════════════════════════════════════════════════════════════════════════
    if verbose:
        print(f'\n{"─"*74}')
        print(f'  STAGE 1  —  Kuramoto Phase Sync Only (no CP refinement)')
        print(f'{"─"*74}')

    kur_result, nc1, md1 = _kuramoto_phase_sync_only(trajectories, arm_bases)
    stage1 = {'stage': 1, 'label': 'Kuramoto only', 'n_collisions': nc1,
               'min_dist_cm': md1*100, 'n_refinements': 0,
               'cp_per_seg': {n: trajectories[n]['metadata']['ncp_per_segment']
                              for n in arm_names}}
    stages.append(stage1)

    if verbose:
        ok = '✅' if nc1 == 0 else '❌'
        print(f'  {ok}  Collisions: {nc1}   Min dist: {md1*100:.1f} cm')
        if nc0 > 0 and nc1 < nc0:
            print(f'       Kuramoto resolved {nc0 - nc1}/{nc0} collision timesteps '
                  f'({(nc0-nc1)/nc0*100:.0f}%)')
        elif nc0 > 0 and nc1 >= nc0:
            print(f'       Kuramoto could not reduce collisions — CP refinement needed')

    # ════════════════════════════════════════════════════════════════════════
    # STAGE 2 — KURAMOTO + CP INCREASE
    # ════════════════════════════════════════════════════════════════════════
    if verbose:
        print(f'\n{"─"*74}')
        print(f'  STAGE 2  —  Kuramoto + CP/Segment Increase (MAX_REFINE=6)')
        print(f'{"─"*74}')

    # Run full pipeline, track per-iteration collision count
    iter_log   = []
    final_result = None

    def _iter_callback(iteration, kur_report, refine_counts, trajs_curr):
        """Called after each Kuramoto iteration by the modified synchroniser."""
        arm_ns = sorted([k for k in kur_report['pair_reports']])
        pair   = list(kur_report['pair_reports'].keys())[0] if kur_report['pair_reports'] else None
        pr     = kur_report['pair_reports'].get(pair, {}) if pair else {}
        iter_log.append({
            'iteration'   : iteration,
            'n_crit'      : kur_report['total_critical'],
            'min_dist_cm' : (kur_report['global_min_dist_m'] or 0.) * 100,
            'collision_free': kur_report['collision_free'],
            'refine_counts': dict(refine_counts),
        })

    # Run full pipeline
    final_result = S4.synchronise_all_arms(trajectories, arm_bases)

    if final_result is None:
        nc2, md2 = nc1, md1
        total_refine = 0
    else:
        verif        = final_result.get('post_sync_verification', {})
        nc2          = int(verif.get('total_critical', 0))
        md2_raw      = verif.get('min_distance_m') or 0.
        md2          = md2_raw * 100
        refine_fin   = final_result.get('refine_counts_final', {})
        total_refine = sum(refine_fin.values())

        # Build iter_log from refinement_history if callback wasn't used
        for entry in final_result.get('refinement_history', []):
            kur = entry.get('kuramoto', {})
            iter_log.append({
                'iteration'    : entry.get('iteration', 0),
                'n_crit'       : kur.get('total_critical', 0),
                'min_dist_cm'  : (kur.get('global_min_dist_m') or 0.) * 100,
                'collision_free': kur.get('collision_free', False),
                'refined_arm'  : entry.get('refined_arm'),
                'refine_counts': entry.get('refine_counts', {}),
            })

    # CP counts after refinement
    final_cp = {}
    if final_result:
        for name in arm_names:
            arm_meta = final_result.get(name, {}).get('metadata', {})
            final_cp[name] = arm_meta.get('ncp_per_segment',
                             trajectories[name]['metadata']['ncp_per_segment'])

    stage2 = {'stage': 2, 'label': 'Kuramoto + CP increase', 'n_collisions': nc2,
               'min_dist_cm': md2, 'n_refinements': total_refine,
               'cp_per_seg': final_cp, 'iter_log': iter_log}
    stages.append(stage2)

    t_total = time.time() - t_pipeline_start

    # ─── Print 3-Stage Summary Table ─────────────────────────────────────────
    print(f'\n{"═"*74}')
    print(f'  3-STAGE COLLISION ANALYSIS REPORT')
    print(f'{"─"*74}')
    print(f'  {"Stage":<32} {"Collisions":>12} {"Min dist":>10} {"Refinements":>13}')
    print(f'  {"─"*70}')
    for s in stages:
        icon = '✅' if s['n_collisions'] == 0 else '❌'
        print(f'  {icon} {s["label"]:<30} '
              f'{s["n_collisions"]:>12} '
              f'{s["min_dist_cm"]:>9.1f}cm '
              f'{s["n_refinements"]:>13}')
    print(f'  {"─"*70}')

    # Effectiveness summary
    if nc0 > 0:
        k_eff  = max(0, nc0 - nc1)
        cp_eff = max(0, nc1 - nc2)
        print(f'\n  Kuramoto resolved        : {k_eff}/{nc0} collision steps '
              f'({k_eff/nc0*100:.0f}%)')
        print(f'  CP/Seg increase resolved : {cp_eff}/{max(nc1,1)} remaining '
              f'({cp_eff/max(nc1,1)*100:.0f}%)')
    else:
        print('\n  ✅  No collisions in raw trajectories — no synchronisation needed!')

    print(f'\n  Total iterations (Kuramoto + CP): {total_refine}')
    print(f'  Pipeline runtime : {t_total:.1f} s')

    # Per-iteration detail
    if iter_log and verbose:
        print(f'\n  Per-Kuramoto-Iteration Detail:')
        print(f'  {"Iter":>5} {"Crit":>6} {"Min cm":>8} {"Safe":>6} {"Refined":>10}')
        for it in iter_log:
            ok = '✅' if it['collision_free'] else '❌'
            ra = it.get('refined_arm', '-') or '-'
            print(f'  {it["iteration"]:>5} {it["n_crit"]:>6} '
                  f'{it["min_dist_cm"]:>7.1f}  {ok:>4}  {ra:>10}')

    final_safe = nc2 == 0
    print(f'\n  FINAL: {"✅ ALL SAFE" if final_safe else f"❌ {nc2} collision timesteps remain"}')
    print(f'{"═"*74}\n')

    report = {
        'inputs'         : {n: {'pos': targets[n][0].tolist(),
                                'quat': targets[n][1].tolist()}
                            for n in arm_names},
        'duration'       : duration,
        'stages'         : stages,
        'final_safe'     : final_safe,
        'min_dist_cm'    : stage2['min_dist_cm'],
        'total_refine'   : total_refine,
        'pipeline_time_s': t_total,
        'sync_result'    : final_result,
    }
    return report


# ─────────────────────────────────────────────────────────────────────────────
# INPUT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _prompt_pose(label: str) -> Tuple[np.ndarray, np.ndarray]:
    print(f'\n  {label}:')
    try:
        x = float(input('    pos X [m]: '))
        y = float(input('    pos Y [m]: '))
        z = float(input('    pos Z [m]: '))
        pos = np.array([x, y, z])
        raw = input('    quat [w,x,y,z] (Enter = [1,0,0,0]): ').strip()
        if raw:
            vals = [float(v) for v in raw.replace(',', ' ').split()]
            quat = S1.quat_normalize(np.array(vals[:4]))
        else:
            quat = np.array([1., 0., 0., 0.])
        return pos, quat
    except (ValueError, EOFError) as e:
        print(f'  Using default  ({e})')
        return np.array([0.5, 0.0, 0.7]), np.array([1., 0., 0., 0.])


def _load_config(path: str) -> Tuple[Dict, float]:
    with open(path) as fh:
        cfg = json.load(fh)
    targets = {}
    for name in ('dsr01', 'dsr02'):
        arm = cfg[name]
        pos  = np.array(arm['pos'],  dtype=float)
        quat = S1.quat_normalize(np.array(arm.get('quat', [1,0,0,0]), dtype=float))
        targets[name] = (pos, quat)
    dur = float(cfg.get('duration', 10.0))
    return targets, dur


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Dual-arm pipeline orchestrator')
    parser.add_argument('--config', type=str, default=None,
                        help='JSON config file (optional)')
    parser.add_argument('--out', type=str, default='pipeline_report.json',
                        help='Output report file')
    args = parser.parse_args()

    print('\n' + '═' * 74)
    print('  DUAL-ARM PIPELINE ORCHESTRATOR')
    print('  Steps: IK → B-Spline → Collision Check → Kuramoto+CP Sync')
    print('  3-Stage Report: Raw | Kuramoto Only | Kuramoto + CP Increase')
    print('═' * 74)

    if args.config:
        print(f'\n  Loading config from: {args.config}')
        targets, duration = _load_config(args.config)
        for name, (pos, quat) in targets.items():
            print(f'  {name}: pos={np.round(pos,3)}  quat={np.round(quat,3)}')
        print(f'  Duration: {duration:.2f} s')
    else:
        print('\n  Enter target pose for each arm:')
        p1, q1 = _prompt_pose('DSR01 target')
        p2, q2 = _prompt_pose('DSR02 target')
        targets = {'dsr01': (p1, q1), 'dsr02': (p2, q2)}
        try:
            s = input('\n  Duration [s] (Enter=10): ').strip()
            duration = float(s) if s else 10.0
        except (ValueError, EOFError):
            duration = 10.0

    try:
        report = run_full_pipeline(targets, duration, verbose=True)
    except RuntimeError as e:
        print(f'\n  {e}'); sys.exit(1)

    # Save report
    def _json_safe(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.int64, np.int32)): return int(obj)
        if isinstance(obj, (np.float64, np.float32)): return float(obj)
        raise TypeError(f'Not serialisable: {type(obj)}')

    with open(args.out, 'w') as fh:
        json.dump(report, fh, indent=2, default=_json_safe)
    print(f'  ✓  Report saved: {args.out}')

    # Save sync result for step_5
    if report.get('sync_result'):
        with open('synchronized_trajectories.json', 'w') as fh:
            json.dump(report['sync_result'], fh, indent=2, default=_json_safe)
        print(f'  ✓  Saved: synchronized_trajectories.json  →  run step_5.py\n')


if __name__ == '__main__':
    main()