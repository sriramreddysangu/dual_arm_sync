#!/usr/bin/env python3
"""
kuramoto_sync.py
================
Stage 4 — Kuramoto phase synchronization with adaptive spatial refinement.

This stage combines two complementary mechanisms:

  TEMPORAL power — Kuramoto coupling
  ───────────────────────────────────
    Each arm's trajectory is parameterised by a phase φ_i ∈ [0, 2π]:

        dφ_i/dt = ω_i + (K/N) Σ_{j≠i} sin(φ_j − φ_i)

    ω_i = 2π / T_i  (natural frequency),  K = coupling strength.
    Collision-aware initial offsets φ_i(0) from collision_checker spread
    arms in phase → Kuramoto coupling then pulls them back into sync while
    preserving the relative offset that avoids overlap.

  SPATIAL power — Targeted seed refinement
  ─────────────────────────────────────────
    If collisions remain after global Kuramoto (penetration > REFINE_TRIGGER):
      1. Identify conflicting arms  (from collision_result.json)
      2. For each conflicting arm:
           N_cp += CP_INCREMENT  (adds 2 extra control points)
           N_seg = max(N_seg, ceil(N_cp / CP_PER_SEG))
      3. Re-fit B-spline seed with new N_seg / N_cp  (via trajectory_generation)
      4. Re-evaluate dense trajectory
      5. Run local Kuramoto on conflicting-arm sub-group (faster convergence)
      6. Re-check collisions
      7. Repeat up to MAX_REFINE_ITER times

    Arms NOT involved in collision are untouched → minimal disruption.
    Each iteration logs N_seg / N_cp / max penetration for traceability.

Convergence criteria:
  Phase: circular std-dev < CONV_THRESHOLD  (0.05 rad)
  Collision: max penetration < COLLISION_TOL  (2 mm)

Output: synchronized_trajectories.json
  {
    "arm_ids":              [...],
    "converged":            bool,
    "final_spread_rad":     float,
    "collision_free":       bool,
    "refinement_iterations": int,
    "refinement_log":       [{...}, ...],   ← per-iteration summary
    "trajectories": {
      "arm_id": {
        "phase_history":     [...],
        "time_history":      [...],
        "n_seg_final":       int,
        "n_cp_final":        int,
        "trajectory_points": [{"time":t, "joints":[...], "time_eff":t_eff}]
      }
    }
  }

Usage:
    ros2 run dual_arm_sync kuramoto_sync
"""

import json
import numpy as np
import warnings
from typing import Dict, List, Optional, Set, Tuple

warnings.filterwarnings('ignore')

try:
    from dual_arm_sync.ik_solver import (
        COLLISION_TOL, ARM_REGISTRY, RobotBases,
        forward_kinematics, LINK_SPHERES, N_SPHERES,
    )
    from dual_arm_sync.trajectory_generation import refit_arm_trajectory
    from dual_arm_sync.collision_checker import check_trajectories, _resolve_bases
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from ik_solver import (
        COLLISION_TOL, ARM_REGISTRY, RobotBases,
        forward_kinematics, LINK_SPHERES, N_SPHERES,
    )
    from trajectory_generation import refit_arm_trajectory
    from collision_checker import check_trajectories, _resolve_bases


# ============================================================================
# CONFIGURATION
# ============================================================================

# Kuramoto parameters
K_GLOBAL        = 5.0    # coupling strength — global sync pass
K_LOCAL         = 8.0    # coupling strength — local (conflicting-arm) pass
SYNC_DT         = 0.01   # integration timestep  (s)
MAX_SYNC_TIME   = 30.0   # maximum Kuramoto integration time  (s)
CONV_THRESHOLD  = 0.05   # phase spread below this → converged  (rad)
OUTPUT_STEPS    = 200    # dense output steps per arm

# Adaptive refinement parameters
MAX_REFINE_ITER = 3      # maximum refinement iterations
REFINE_TRIGGER  = 0.005  # trigger refinement if max_pen > this  (5 mm)
CP_INCREMENT    = 2      # N_cp increase per conflicting arm per iteration
CP_PER_SEG      = 4      # control points per segment guideline
N_CP_MAX_REFINE = 24     # hard cap on N_cp during refinement
N_SEG_MAX_REFINE = 12    # hard cap on N_seg during refinement


# ============================================================================
# TRAJECTORY INTERPOLATION
# ============================================================================

def build_arrays(trajectory_points: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert trajectory_points list → (t_arr, j_arr) for interpolation."""
    t_arr = np.array([p['time']   for p in trajectory_points])
    j_arr = np.array([p['joints'] for p in trajectory_points])
    return t_arr, j_arr


def interpolate_at(t_arr: np.ndarray,
                   j_arr: np.ndarray,
                   t_q:   float) -> np.ndarray:
    """Linear interpolation of joints at time t_q.  Clamps at boundaries."""
    t_q = float(np.clip(t_q, t_arr[0], t_arr[-1]))
    idx = np.searchsorted(t_arr, t_q)
    if idx == 0:
        return j_arr[0].copy()
    if idx >= len(t_arr):
        return j_arr[-1].copy()
    alpha = (t_q - t_arr[idx-1]) / (t_arr[idx] - t_arr[idx-1] + 1e-12)
    return j_arr[idx-1] + alpha * (j_arr[idx] - j_arr[idx-1])


# ============================================================================
# KURAMOTO OSCILLATOR
# ============================================================================

def phase_spread(phases: np.ndarray) -> float:
    """
    Circular standard deviation — synchronization quality.
    0 = perfect sync,  π = maximally de-synced.
    """
    if len(phases) <= 1:
        return 0.0
    R = abs(np.mean(np.exp(1j * phases)))
    return float(np.sqrt(-2.0 * np.log(max(R, 1e-12))))


def _kuramoto_step(phases: np.ndarray,
                   omegas: np.ndarray,
                   K:      float,
                   dt:     float) -> np.ndarray:
    """
    Single Euler step:  dφ_i/dt = ω_i + (K/N) Σ_{j≠i} sin(φ_j − φ_i)
    """
    N    = len(phases)
    dphi = omegas.copy()
    for i in range(N):
        dphi[i] += (K / N) * sum(
            np.sin(phases[j] - phases[i]) for j in range(N) if j != i
        )
    return phases + dphi * dt


def run_kuramoto(arm_ids:    List[str],
                 durations:  Dict[str, float],
                 offsets:    Dict[str, float],
                 K:          float  = K_GLOBAL,
                 dt:         float  = SYNC_DT,
                 max_time:   float  = MAX_SYNC_TIME,
                 label:      str    = 'global'
                 ) -> Tuple[Dict[str, np.ndarray], bool, float]:
    """
    Integrate Kuramoto ODE for the specified arm_ids until convergence.

    Used for both global sync (all arms) and local sync (conflicting arms).

    Args:
        arm_ids   : arms to include  (subset for local sync)
        durations : {arm_id: T_i}   trajectory durations
        offsets   : {arm_id: Δt_i}  initial time offsets (→ phase offsets)
        K         : coupling strength
        dt        : timestep
        max_time  : integration time limit
        label     : 'global' or 'local' — for print only

    Returns:
        phase_histories : {arm_id: np.array of φ at each step}
        converged       : bool
        final_spread    : float  (rad)
    """
    N      = len(arm_ids)
    omegas = np.array([2.0 * np.pi / durations[a] for a in arm_ids])
    phases = np.array([2.0 * np.pi * offsets.get(a, 0.0) / durations[a]
                       for a in arm_ids])

    n_max    = int(max_time / dt)
    history  = {a: [] for a in arm_ids}
    log_every = max(1, n_max // 8)

    print(f'\n  [Kuramoto {label}]  arms={arm_ids}  N={N}  K={K}  dt={dt}')
    print(f'  ω  : {dict(zip(arm_ids, np.round(omegas, 4)))}')
    print(f'  φ₀ : {dict(zip(arm_ids, np.round(phases, 4)))}')
    print(f'  Δt : {offsets}')

    converged = False
    for step in range(n_max):
        for i, a in enumerate(arm_ids):
            history[a].append(float(phases[i]))

        spread = phase_spread(phases)
        if step % log_every == 0:
            print(f'    step={step:5d}  t={step*dt:.2f}s'
                  f'  spread={spread:.4f}rad'
                  f'  φ={np.round(phases % (2*np.pi), 3)}')

        if spread < CONV_THRESHOLD and step > 5:
            converged = True
            print(f'\n  ✓  [{label}] Converged  step={step}'
                  f'  t={step*dt:.2f}s  spread={spread:.5f}rad')
            break

        phases = _kuramoto_step(phases, omegas, K, dt)

    if not converged:
        print(f'\n  ⚠  [{label}] Not converged within {max_time}s'
              f'  spread={phase_spread(phases):.4f}rad')

    return ({a: np.array(history[a]) for a in arm_ids},
            converged, phase_spread(phases))


def phase_to_traj(arm_id:        str,
                   phase_hist:    np.ndarray,
                   t_arr:         np.ndarray,
                   j_arr:         np.ndarray,
                   duration:      float,
                   dt:            float = SYNC_DT,
                   out_steps:     int   = OUTPUT_STEPS) -> Dict:
    """
    Apply the converged Kuramoto phase offset to the ORIGINAL dense trajectory.

    Bug that this replaces
    ──────────────────────
    The old version sub-sampled the Kuramoto TRANSIENT (phase_hist has only
    N_converged_steps elements, often just 7 for small offsets).  Output was
    7 waypoints over 0.06 s — the arm barely moved in Gazebo.

    Correct behaviour
    ──────────────────
    1. Extract converged phase φ_conv from the stable tail of phase_hist.
    2. Compute time offset  t_off = φ_conv / (2π) × T
    3. Resample the ORIGINAL dense trajectory (t_arr / j_arr) at `out_steps`
       evenly-spaced wall-clock times, applying t_off as a temporal delay:
         • For t_wall < t_off: arm holds at θ(0)  (start position)
         • For t_wall ≥ t_off: arm plays at rate  T_traj / (T − t_off)
           so it completes the FULL trajectory within the wall-clock window.
    4. Output is always `out_steps` points over `duration` seconds.

    For the typical case (t_off ≪ T, e.g. 0.006 s / 10 s), the output is
    virtually identical to the original trajectory — which is exactly right:
    the arms are already safe and merely need very minor timing adjustment.
    """
    # ── Converged phase → time offset ──────────────────────────────────────
    stable      = min(30, max(1, len(phase_hist)))
    phi_stable  = float(np.mean(phase_hist[-stable:]))
    phi_conv    = phi_stable % (2.0 * np.pi)
    t_off       = phi_conv / (2.0 * np.pi) * duration   # seconds of delay

    T_traj  = float(t_arr[-1] - t_arr[0])               # original traj duration
    t_wall  = np.linspace(0.0, duration, out_steps)
    t_eff_arr = np.empty(out_steps)

    for i, tw in enumerate(t_wall):
        if t_off > 1e-6 and tw < t_off:
            # Hold at start
            t_eff_arr[i] = t_arr[0]
        else:
            # Play at adjusted speed so the full trajectory fits in the remaining time
            play_elapsed  = tw - t_off
            play_window   = duration - t_off
            if play_window > 1e-6:
                frac = play_elapsed / play_window
            else:
                frac = 1.0
            t_eff_arr[i] = t_arr[0] + frac * T_traj

    t_eff_arr = np.clip(t_eff_arr, t_arr[0], t_arr[-1])

    points = [{'time':     float(tw),
               'time_eff': float(te),
               'joints':   interpolate_at(t_arr, j_arr, te).tolist()}
              for tw, te in zip(t_wall, t_eff_arr)]

    return {
        'phase_history':     phase_hist.tolist(),
        'time_history':      t_eff_arr.tolist(),
        'trajectory_points': points,
        't_offset_applied':  float(t_off),
    }


# ============================================================================
# ADAPTIVE REFINEMENT LOOP
# ============================================================================

def _check_in_memory(trajs:     Dict,
                      arm_bases: Dict[str, np.ndarray]) -> Dict:
    """
    Run collision check in memory (no file I/O) for the refinement loop.
    Returns same dict structure as collision_checker.check_trajectories().
    """
    return check_trajectories(trajs, arm_bases)


def _increment_params(n_seg_cur: int, n_cp_cur: int, iteration: int
                      ) -> Tuple[int, int]:
    """
    Increase N_cp by CP_INCREMENT; adjust N_seg to keep segment density
    reasonable.  Hard-capped at N_CP_MAX_REFINE / N_SEG_MAX_REFINE.
    """
    n_cp  = min(N_CP_MAX_REFINE, n_cp_cur + CP_INCREMENT)
    n_seg = min(N_SEG_MAX_REFINE,
                max(n_seg_cur, int(np.ceil(n_cp / CP_PER_SEG))))
    return n_seg, n_cp


def adaptive_refinement_loop(trajs_init:    Dict[str, Dict],
                               ik_data:       Dict,
                               arm_bases:     Dict[str, np.ndarray],
                               durations:     Dict[str, float],
                               offsets_init:  Dict[str, float]
                               ) -> Tuple[Dict, Dict, List[Dict]]:
    """
    Global Kuramoto → collision check → targeted refinement → local Kuramoto
    → repeat up to MAX_REFINE_ITER times.

    Args:
        trajs_init   : initial trajectories from trajectory_generation
        ik_data      : raw ik_solutions dict (needed for refit)
        arm_bases    : {arm_id: base_position}
        durations    : {arm_id: T_i}
        offsets_init : initial time offsets from collision_checker

    Returns:
        final_trajs  : {arm_id: trajectory dict with trajectory_points}
        sync_results : {arm_id: phase_to_traj output}
        refine_log   : list of per-iteration dicts
    """
    arm_ids    = list(trajs_init.keys())
    trajs      = {a: dict(t) for a, t in trajs_init.items()}
    offsets    = dict(offsets_init)
    refine_log: List[Dict] = []

    # Track current N_seg / N_cp per arm (for refinement increments)
    n_seg_cur  = {a: trajs[a]['n_seg'] for a in arm_ids}
    n_cp_cur   = {a: trajs[a]['n_cp']  for a in arm_ids}

    print(f'\n{"="*80}')
    print('ADAPTIVE REFINEMENT LOOP')
    print(f'{"="*80}')
    print(f'  Max iterations    : {MAX_REFINE_ITER}')
    print(f'  Refine trigger    : {REFINE_TRIGGER*1000:.1f} mm penetration')
    print(f'  CP increment      : +{CP_INCREMENT} per conflicting arm')
    print(f'  Initial N_seg/N_cp: '
          f'{dict((a, (n_seg_cur[a], n_cp_cur[a])) for a in arm_ids)}')

    # ----------------------------------------------------------------
    # Iteration 0  — global Kuramoto pass
    # ----------------------------------------------------------------
    print(f'\n{"─"*70}')
    print('ITERATION 0  —  Global Kuramoto')
    print(f'{"─"*70}')

    phase_hists, converged, spread = run_kuramoto(
        arm_ids, durations, offsets,
        K=K_GLOBAL, dt=SYNC_DT, max_time=MAX_SYNC_TIME,
        label='global-0',
    )

    # Build synced trajectories from phase histories
    sync_trajs: Dict[str, Dict] = {}
    arr_cache:  Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for a in arm_ids:
        t_arr, j_arr = build_arrays(trajs[a]['trajectory_points'])
        arr_cache[a] = (t_arr, j_arr)
        sync_trajs[a] = phase_to_traj(
            a, phase_hists[a], t_arr, j_arr, durations[a]
        )

    # In-memory collision check on synchronized output
    coll = _check_in_memory(sync_trajs, arm_bases)
    max_pen = coll['max_penetration_m']

    log0 = {
        'iteration':              0,
        'phase':                  'global_kuramoto',
        'max_pen_mm':             round(max_pen * 1000, 3),
        'safe':                   coll['safe'],
        'collision_free':         coll['safe'],
        'collision_free_summary': coll.get('collision_free_summary',
                                   'COLLISION-FREE' if coll['safe'] else
                                   f'COLLISION max_pen={max_pen*1000:.2f}mm'),
        'converged':              converged,
        'spread_rad':             round(spread, 5),
        'conflicting':            coll['conflicting_arms'],
        'n_seg':                  dict(n_seg_cur),
        'n_cp':                   dict(n_cp_cur),
    }
    refine_log.append(log0)
    print(f'\n  [0] max_pen={max_pen*1000:.2f}mm  safe={coll["safe"]}'
          f'  converged={converged}  spread={spread:.4f}rad')

    # ----------------------------------------------------------------
    # Refinement iterations 1 … MAX_REFINE_ITER
    # ----------------------------------------------------------------
    for it in range(1, MAX_REFINE_ITER + 1):
        # Check whether we need to refine
        if coll['safe'] or max_pen <= REFINE_TRIGGER:
            print(f'\n  ✓  Collision resolved before iteration {it}'
                  f'   max_pen={max_pen*1000:.2f}mm ≤ {REFINE_TRIGGER*1000:.1f}mm')
            break

        conflicting: List[str] = coll['conflicting_arms']
        if not conflicting:
            print(f'\n  ⚠  No conflicting arms identified — stopping')
            break

        print(f'\n{"─"*70}')
        print(f'ITERATION {it}  —  Spatial refinement + local Kuramoto')
        print(f'{"─"*70}')
        print(f'  Conflicting arms : {conflicting}')
        print(f'  max penetration  : {max_pen*1000:.2f} mm')

        # ---- Increase N_cp/N_seg for conflicting arms only ----
        for a in conflicting:
            old_seg, old_cp = n_seg_cur[a], n_cp_cur[a]
            n_seg_cur[a], n_cp_cur[a] = _increment_params(
                n_seg_cur[a], n_cp_cur[a], it
            )
            print(f'  [{a}]  N_seg: {old_seg}→{n_seg_cur[a]}'
                  f'   N_cp: {old_cp}→{n_cp_cur[a]}')

        # ---- Re-fit B-spline seed for each conflicting arm ----
        for a in conflicting:
            arm_ik = ik_data.get(a, {})
            # Inject current_joints / optimal_joints from original ik_data
            arm_ik['arm_id'] = a
            refitted = refit_arm_trajectory(
                arm_ik, n_seg_cur[a], n_cp_cur[a], durations[a]
            )
            if refitted:
                trajs[a]          = refitted
                t_arr, j_arr      = build_arrays(refitted['trajectory_points'])
                arr_cache[a]      = (t_arr, j_arr)
                print(f'  [{a}]  seed refit  OK  '
                      f'steps={len(refitted["trajectory_points"])}')
            else:
                print(f'  [{a}]  ⚠  seed refit FAILED — keeping previous')

        # ---- Local Kuramoto: sync conflicting arms among themselves ----
        # Non-conflicting arms keep their phase from the global pass.
        # We only re-integrate the conflicting sub-group with updated offsets
        # so they converge relative to one another faster (K_LOCAL > K_GLOBAL).
        local_offsets = {a: 0.0 for a in conflicting}
        for a in conflicting:
            # Seed local offset from the collision checker suggestion
            local_offsets[a] = offsets.get(a, 0.0)

        local_hists, local_conv, local_spread = run_kuramoto(
            conflicting,
            durations,
            local_offsets,
            K        = K_LOCAL,
            dt       = SYNC_DT,
            max_time = MAX_SYNC_TIME / 2,
            label    = f'local-{it}',
        )

        # Update phase histories: conflicting arms get local history,
        # non-conflicting arms keep global history from previous iteration
        for a in arm_ids:
            t_arr, j_arr = arr_cache[a]
            if a in conflicting:
                sync_trajs[a] = phase_to_traj(
                    a, local_hists[a], t_arr, j_arr, durations[a]
                )
            else:
                sync_trajs[a] = phase_to_traj(
                    a, phase_hists[a], t_arr, j_arr, durations[a]
                )

        # ---- Re-check collisions ----
        coll    = _check_in_memory(sync_trajs, arm_bases)
        max_pen = coll['max_penetration_m']

        log_i = {
            'iteration':              it,
            'phase':                  'spatial_refine + local_kuramoto',
            'max_pen_mm':             round(max_pen * 1000, 3),
            'safe':                   coll['safe'],
            'collision_free':         coll['safe'],
            'collision_free_summary': coll.get('collision_free_summary',
                                       'COLLISION-FREE' if coll['safe'] else
                                       f'COLLISION max_pen={max_pen*1000:.2f}mm'),
            'local_converged':        local_conv,
            'local_spread':           round(local_spread, 5),
            'conflicting':            coll['conflicting_arms'],
            'n_seg':                  dict(n_seg_cur),
            'n_cp':                   dict(n_cp_cur),
        }
        refine_log.append(log_i)
        print(f'\n  [{it}] max_pen={max_pen*1000:.2f}mm  safe={coll["safe"]}'
              f'  local_converged={local_conv}'
              f'  local_spread={local_spread:.4f}rad')

        # Update offsets for next iteration from new collision check
        offsets = coll['time_offsets']

        # Update phase_hists for non-conflicting arms (re-use global)
        # Conflicting arms: use local histories for traceability
        for a in conflicting:
            phase_hists[a] = local_hists[a]

    # ---- Final result ----
    final_collision_free = coll['safe'] or max_pen <= COLLISION_TOL

    return trajs, sync_trajs, refine_log, converged, spread, final_collision_free


# ============================================================================
# PIPELINE RUNNER
# ============================================================================

def run(traj_file:      str = 'trajectories.json',
        ik_file:        str = 'ik_solutions.json',
        collision_file: str = 'collision_result.json',
        output_file:    str = 'synchronized_trajectories.json') -> Dict:
    """
    Load trajectories + IK data + collision offsets →
    adaptive Kuramoto + refinement loop →
    write synchronized_trajectories.json.
    """
    print('\n' + '=' * 80)
    print('KURAMOTO SYNC  [Global + Adaptive Spatial Refinement]')
    print('=' * 80)

    # ---- Load trajectories ----
    with open(traj_file) as f:
        traj_data = json.load(f)

    trajs    = traj_data.get('trajectories', traj_data)
    arm_ids  = traj_data.get('arm_ids', list(trajs.keys()))
    duration = float(traj_data.get('duration', 10.0))
    durations = {a: float(trajs[a].get('duration', duration)) for a in arm_ids}

    print(f'\n  Arms     : {arm_ids}')
    print(f'  Duration : {duration} s')
    print(f'  N_seg    : {dict((a, trajs[a]["n_seg"]) for a in arm_ids)}')
    print(f'  N_cp     : {dict((a, trajs[a]["n_cp"])  for a in arm_ids)}')

    # ---- Load IK solutions (for refit) ----
    ik_data: Dict = {}
    try:
        with open(ik_file) as f:
            ik_raw = json.load(f)
        for a in arm_ids:
            if a in ik_raw:
                ik_data[a] = ik_raw[a]
        print(f'\n  IK data loaded : {list(ik_data.keys())}')
    except FileNotFoundError:
        print(f'\n  ⚠  {ik_file} not found — refinement refit disabled')

    # ---- Load collision offsets ----
    time_offsets: Dict[str, float] = {a: 0.0 for a in arm_ids}
    initial_conflicting: List[str] = []
    try:
        with open(collision_file) as f:
            coll_init = json.load(f)
        time_offsets.update(coll_init.get('time_offsets', {}))
        initial_conflicting = coll_init.get('conflicting_arms', [])
        safe_init           = coll_init.get('safe', True)
        print(f'\n  Initial collision : safe={safe_init}'
              f'  conflicting={initial_conflicting}')
        if any(v > 0 for v in time_offsets.values()):
            print(f'  Phase offsets    : {time_offsets}')
        else:
            print('  Phase offsets    : none (all arms start in phase)')
    except FileNotFoundError:
        print(f'\n  ⚠  {collision_file} not found — no initial offsets')

    # ---- Resolve arm bases ----
    arm_bases = _resolve_bases(arm_ids)

    # ---- Adaptive refinement loop ----
    final_trajs, sync_trajs, refine_log, global_conv, global_spread, \
        final_cf = adaptive_refinement_loop(
            trajs_init   = trajs,
            ik_data      = ik_data,
            arm_bases    = arm_bases,
            durations    = durations,
            offsets_init = time_offsets,
        )

    # ---- Refinement summary ----
    print(f'\n{"="*80}')
    print('REFINEMENT SUMMARY')
    print(f'{"="*80}')
    for log in refine_log:
        print(f'  iter={log["iteration"]}  '
              f'max_pen={log["max_pen_mm"]:.2f}mm  '
              f'safe={log["safe"]}  '
              f'phase={log["phase"]}')
        print(f'    N_seg={log["n_seg"]}  N_cp={log["n_cp"]}')

    print(f'\n  global_converged  : {global_conv}')
    print(f'  global_spread     : {global_spread:.5f} rad')
    print(f'  final_collision_free : {final_cf}')
    print(f'  refinement_iters  : {len(refine_log)-1}')

    # ---- Assemble output ----
    traj_output: Dict[str, Dict] = {}
    for a in arm_ids:
        sync = sync_trajs[a]
        traj_output[a] = {
            'phase_history':     sync['phase_history'],
            'time_history':      sync['time_history'],
            'trajectory_points': sync['trajectory_points'],
            'n_seg_final':       final_trajs[a]['n_seg'],
            'n_cp_final':        final_trajs[a]['n_cp'],
            'method':            final_trajs[a].get('method', 'kabir_2019'),
        }
        print(f'  {a}: steps={len(sync["trajectory_points"])}'
              f'  N_seg={final_trajs[a]["n_seg"]}'
              f'  N_cp={final_trajs[a]["n_cp"]}')

    output = {
        'arm_ids':               arm_ids,
        'converged':             global_conv,
        'final_spread_rad':      float(global_spread),
        'collision_free':        final_cf,
        'refinement_iterations': len(refine_log) - 1,
        'refinement_log':        refine_log,
        'k_global':              K_GLOBAL,
        'k_local':               K_LOCAL,
        'dt':                    SYNC_DT,
        'original_duration':     duration,
        'time_offsets_used':     time_offsets,
        'trajectories':          traj_output,
    }

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f'\n✓  {output_file}  written')
    print('Next → ros2 run dual_arm_sync gazebo_executor')
    return output


# ============================================================================
# ROS2 ENTRY
# ============================================================================

def main(args=None):
    try:
        import rclpy; rclpy.init(args=args)
    except Exception:
        pass
    try:
        run()
    except FileNotFoundError as e:
        print(f'✗  {e}')
        print('   Run trajectory_generation and collision_checker first')
    except Exception:
        import traceback; traceback.print_exc()
    try:
        import rclpy; rclpy.shutdown()
    except Exception:
        pass


if __name__ == '__main__':
    main()