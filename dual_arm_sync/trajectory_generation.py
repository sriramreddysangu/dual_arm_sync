#!/usr/bin/env python3
"""
trajectory_generation.py
=========================
Stage 2 — Cartesian-Space Trajectory Generation  (per-arm independent)

════════════════════════════════════════════════════════════════════
PAPER METHOD  (Kabir et al. 2019, §IV-C)  — adapted for independent arms

The paper generates synchronized configuration-space trajectories by:
  1. Cartesian minimum-jerk path:  s(τ) = 10τ³ − 15τ⁴ + 6τ⁵
  2. IK at each Cartesian sample  (we reuse chained IK from Stage 1)
  3. B-spline fit through joint-space waypoints
  4. Duration scaling if limits violated
  5. Cross-arm NLP coupling  ← REPLACED by Kuramoto (Stage 4)

WHAT WE DO DIFFERENTLY FROM THE PAPER
  Paper:  one central NLP couples all arms together during trajectory
          optimisation → collision-avoiding trajectories directly.

  Our approach (matches paper §IV-C except step 5):
    Step 1  Cartesian minimum-jerk path  ✓ same
    Step 2  IK at each sample            ✓ same (done in Stage 1 — reused)
    Step 3  B-spline fit                 ✓ same
    Step 4  Limit scaling                ✓ same
    Step 5  Cross-arm coupling           ✗ REPLACED
              → arms planned independently here
              → Kuramoto oscillator (Stage 4) handles synchronisation
              → Adaptive control-point insertion (Stage 4) handles collision

WHY WE REUSE STAGE-1 IK SAMPLES INSTEAD OF RE-SOLVING
  dual_arm_ik_solver.py (Stage 1) already solved branch-continuous
  chained IK at N curvature-adaptive Cartesian samples using SLERP
  orientation interpolation and lexicographic branch selection.

  Re-solving IK here (e.g. with fresh SLSQP at every waypoint) would:
    • Introduce arbitrary ±π branch flips  (SLSQP has no memory)
    • Inflate B-spline velocity 10–100x   (large gradient at flip)
    • Force 10x duration scaling           (seen in earlier version)

  Instead we apply minimum-jerk TIME parameterisation to the already-
  solved spatial samples.  The IK is Cartesian-space–correct; only the
  time stamps change.

MINIMUM-JERK TIME PARAMETERISATION  (§IV-C, eq. 1)
  Stage 1 stores N joint solutions at UNIFORM spatial parameters
  s_i = i/(N-1) ∈ [0, 1].

  We invert  s(τ) = 10τ³ − 15τ⁴ + 6τ⁵  to find τ_i = t_i/T such that
  s(τ_i) = s_i.  This makes the Cartesian-space velocity profile follow
  the minimum-jerk curve:
    • Zero velocity and acceleration at start and end
    • Peak velocity at s ≈ 0.5 (midpoint of path)

  Near s=0 or s=1 (slow phase): waypoints are dense in time.
  Near s=0.5      (fast phase):  waypoints are sparse in time.
  The B-spline fitted through non-uniform (t_i, q_i) produces smooth
  joint trajectories WITHOUT the velocity spikes that caused 10x scaling.
════════════════════════════════════════════════════════════════════

Input  :  ik_solutions.json        (Stage 1 — dual_arm_ik_solver)
Output :  trajectories.json        (Stage 3 — collision_checker)

Output format (consumed by collision_checker AND kuramoto_sync AND gazebo_executor):
  {
    "arm_ids":  ["dsr01", "dsr02"],
    "duration": T_max,
    "method":   "...",
    "dsr01": {
      "trajectory_points": [{"time": t, "joints": [...]}, ...],  ← collision_checker
      "time":            [...],   ← kuramoto_synchronization
      "positions":       [...],   ← kuramoto_synchronization / gazebo_executor
      "velocities":      [...],
      "accelerations":   [...],
      "num_samples":     1000,
      "joint_waypoints": [...],   ← kuramoto control-point refinement
      "waypoint_times":  [...],
      "duration":        T,
      "metadata":        {...}
    },
    "dsr02": { ... }
  }

Usage:
    ros2 run dual_arm_sync trajectory_generation
"""

import json
import os
import sys

import numpy as np
from scipy.interpolate import splrep, splev

# ── Foundation imports (Standard DH, correct limits) ──────────────────────────
try:
    from dual_arm_sync.ik_solver import (
        forward_kinematics,    # (joints) → (pos, rot, frames) — Standard DH
        RobotBases,            # DSR01_BASE, DSR02_BASE
        JOINT_LIMITS,          # (6, 2)  [lower, upper] rad
        JOINT_VEL_MAX,         # (6,)    rad/s
    )
except ImportError:
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(__file__))
    from ik_solver import (
        forward_kinematics, RobotBases, JOINT_LIMITS, JOINT_VEL_MAX,
    )


# ============================================================================
# CONSTANTS
# ============================================================================

# Acceleration limit  (conservative: v_max / 0.5 s)
# ik_solver.py does not define JOINT_ACC_MAX so we derive it here.
JOINT_ACC_MAX = JOINT_VEL_MAX * 2.0   # rad/s²

N_EVAL = 1000   # output trajectory evaluation samples


# ============================================================================
# SECTION 1 — Minimum-Jerk Time Parameterisation
# ============================================================================
#
# Forward:   s(τ) = 10τ³ − 15τ⁴ + 6τ⁵       τ = t/T ∈ [0,1]
# Inverse:   given s_i → find τ_i via dense table lookup  (fast, accurate)
#
# Pre-compute once at import time.
_TAU_TABLE = np.linspace(0.0, 1.0, 200_000)
_S_TABLE   = (10 * _TAU_TABLE**3
              - 15 * _TAU_TABLE**4
              + 6  * _TAU_TABLE**5)


def _minjerk_tau(s: np.ndarray) -> np.ndarray:
    """s ∈ [0,1] → τ ∈ [0,1]  via pre-computed table + linear interp."""
    return np.interp(np.clip(s, 0.0, 1.0), _S_TABLE, _TAU_TABLE)


def s_to_time(s_values: np.ndarray, T: float) -> np.ndarray:
    """
    Map uniform spatial parameters s_i → absolute times t_i = τ_i · T.

    Guarantees strict monotonicity (handles floating-point ties from
    duplicate s values produced by SLERP at near-zero curvature paths).
    """
    tau = _minjerk_tau(np.asarray(s_values, dtype=float))
    t   = tau * T
    eps = 1e-8
    for i in range(1, len(t)):
        if t[i] <= t[i - 1] + eps:
            t[i] = t[i - 1] + eps
    return t


# ============================================================================
# SECTION 2 — Cubic B-Spline Fitting
# ============================================================================

def fit_bspline(joint_wps: np.ndarray,
                t_wps:     np.ndarray,
                T:         float,
                n_eval:    int = N_EVAL) -> dict:
    """
    Fit interpolating cubic B-spline through (t_wps[i], joint_wps[i]).

    t_wps  : (N,)   absolute times — non-uniform (min-jerk parameterised)
    joint_wps: (N,6) joint-space waypoints
    T      : float  total duration (== t_wps[-1])
    n_eval : int    number of uniform evaluation points

    Normalises t to [0, 1] internally for numerical stability.

    Returns dict with:
        time            (n_eval,)
        positions       (n_eval, 6)   [rad]
        velocities      (n_eval, 6)   [rad/s]
        accelerations   (n_eval, 6)   [rad/s²]
        joint_waypoints (N, 6)
        waypoint_times  (N,)
        num_samples     int
    """
    n_way = len(joint_wps)
    if n_way < 2:
        raise ValueError(f"Need ≥ 2 waypoints, got {n_way}")

    t_n    = t_wps / T                        # normalise → [0, 1]
    t_ev_n = np.linspace(0.0, 1.0, n_eval)
    t_ev   = t_ev_n * T

    pos = np.zeros((n_eval, 6))
    vel = np.zeros((n_eval, 6))
    acc = np.zeros((n_eval, 6))

    for j in range(6):
        k   = min(3, n_way - 1)              # cubic where possible
        tck = splrep(t_n, joint_wps[:, j], k=k, s=0)
        pos[:, j] = splev(t_ev_n, tck, der=0)
        vel[:, j] = splev(t_ev_n, tck, der=1) / T          # chain rule
        acc[:, j] = splev(t_ev_n, tck, der=2) / (T ** 2)

    return {
        'time':            t_ev,
        'positions':       pos,
        'velocities':      vel,
        'accelerations':   acc,
        'joint_waypoints': joint_wps.copy(),
        'waypoint_times':  t_wps.copy(),
        'num_samples':     n_eval,
    }


# ============================================================================
# SECTION 3 — Joint-Limit Check & Duration Scaling
# ============================================================================

def scale_duration_if_needed(traj:      dict,
                               joint_wps: np.ndarray,
                               t_wps:     np.ndarray,
                               T:         float) -> dict:
    """
    Check peak velocity and acceleration against hardware limits.
    If violated, scale T by the exact factor and re-fit once.

    Scaling T → k·T divides velocities by k and accelerations by k².
    One scaling pass is always sufficient.

    Uses JOINT_VEL_MAX and JOINT_ACC_MAX from ik_solver (Standard DH,
    correct for M1013).  Does NOT use constants.py.
    """
    v_scale = 1.0
    a_scale = 1.0
    for j in range(6):
        v_peak = float(np.max(np.abs(traj['velocities'][:, j])))
        a_peak = float(np.max(np.abs(traj['accelerations'][:, j])))
        if v_peak > JOINT_VEL_MAX[j]:
            v_scale = max(v_scale, v_peak / JOINT_VEL_MAX[j])
        if a_peak > JOINT_ACC_MAX[j]:
            a_scale = max(a_scale, np.sqrt(a_peak / JOINT_ACC_MAX[j]))

    scale = max(v_scale, a_scale)
    if scale > 1.0:
        T_new = T * scale * 1.05          # 5 % headroom
        t_new = t_wps * (T_new / T)
        print(f"    ⚠  Limit violation — scaling  {T:.3f}s → {T_new:.3f}s"
              f"  (vel×{v_scale:.2f}  acc×{a_scale:.2f})")
        return fit_bspline(joint_wps, t_new, T_new)

    print(f"    ✓  All limits satisfied  (T = {T:.3f}s unchanged)")
    return traj


# ============================================================================
# SECTION 4 — Per-Arm Trajectory Builder
# ============================================================================

def build_arm_trajectory(robot_name:  str,
                          joint_wps:  np.ndarray,
                          s_values:   np.ndarray,
                          q_start:    np.ndarray,
                          q_end:      np.ndarray,
                          p_start:    np.ndarray,
                          p_end:      np.ndarray,
                          base:       np.ndarray,
                          T:          float) -> dict:
    """
    Paper §IV-C per-arm pipeline — no cross-arm coupling.

    Parameters
    ----------
    robot_name : 'dsr01' or 'dsr02'
    joint_wps  : (N, 6)  chained IK solutions at s_values  (from Stage 1)
    s_values   : (N,)    uniform spatial parameters ∈ [0, 1]
    q_start    : (6,)    exact start joints (Gazebo live state)
    q_end      : (6,)    exact target joints (lexicographic optimal from Stage 1)
    p_start    : (3,)    world-frame start EE position
    p_end      : (3,)    world-frame target EE position
    base       : (3,)    robot base in world frame
    T          : float   requested duration [s]

    Steps (paper §IV-C)
    -------------------
    ① Anchor start/end to exact Stage-1 joint solutions (no drift)
    ② s_i (uniform) → t_i (non-uniform, min-jerk)  — paper eq. 1
    ③ Cubic B-spline fit through (t_i, q_i)
    ④ Evaluate at 1000 uniform time steps
    ⑤ Check velocity/acceleration limits; scale T if violated
    ⑥ Report EE accuracy and velocity utilisation
    """
    N = len(joint_wps)
    print(f"\n  ── {robot_name.upper()} ──")
    dist_mm = float(np.linalg.norm(p_end - p_start)) * 1000
    dq_deg  = float(np.max(np.degrees(np.abs(q_end - q_start))))
    print(f"  Cartesian path    : {np.round(p_start, 4)}  →  {np.round(p_end, 4)}")
    print(f"  Path length       : {dist_mm:.1f} mm")
    print(f"  Chained IK pts    : {N}  (from Stage 1 — NOT re-solved here)")
    print(f"  Max joint Δ       : {dq_deg:.1f}°")

    # ① Anchor exact start/end (prevent drift from intermediate IK errors)
    wps = joint_wps.copy()
    wps[0]  = q_start
    wps[-1] = q_end

    # ② Minimum-jerk time parameterisation  (paper §IV-C eq. 1)
    #    Uniform s_i → non-uniform t_i.  Near s=0/1 (slow): dense time.
    #    Near s=0.5 (fast): sparse time.  This is the paper's key equation.
    t_wps = s_to_time(np.asarray(s_values, dtype=float), T)
    print(f"  Time range        : [{t_wps[0]:.4f}, {t_wps[-1]:.4f}] s"
          f"  (min-jerk, non-uniform)")

    # ③④ Fit B-spline and evaluate
    traj = fit_bspline(wps, t_wps, T)

    # ⑤ Scale if limits violated
    traj = scale_duration_if_needed(traj, wps, t_wps, T)

    final_T = float(traj['time'][-1])

    # ⑥ EE end-point accuracy (uses Standard DH forward_kinematics)
    p_local, *_ = forward_kinematics(traj['positions'][-1])
    err_mm = float(np.linalg.norm(p_local + base - p_end)) * 1000
    print(f"  EE end error      : {err_mm:.2f} mm")

    # Velocity utilisation
    v_util = np.max(np.abs(traj['velocities']), axis=0) / JOINT_VEL_MAX * 100
    print(f"  Vel utilisation % : {np.round(v_util, 1)}")
    print(f"  Final duration    : {final_T:.3f}s")

    # Attach metadata
    traj['robot_name'] = robot_name
    traj['metadata'] = {
        'robot_name':        robot_name,
        'duration':          final_T,
        'num_samples':       N_EVAL,
        'num_waypoints':     N,
        'start_joints':      q_start.tolist(),
        'end_joints':        q_end.tolist(),
        'start_world_pos':   p_start.tolist(),
        'target_world_pos':  p_end.tolist(),
        'planning_method':   'cartesian_minjerk_bspline_independent',
        'ee_error_mm':       round(err_mm, 3),
    }
    return traj


# ============================================================================
# SECTION 5 — Output Format
# ============================================================================

def make_arm_output(traj: dict) -> dict:
    """
    Convert internal trajectory dict to the JSON output consumed by all
    downstream stages.  Produces BOTH data formats in one dict.

    trajectory_points  →  collision_checker.py  (reads this exact key)
                          gazebo_executor.py    (primary format)
    time / positions   →  kuramoto_synchronization.py (flat arrays)
    joint_waypoints    →  kuramoto_synchronization.py (control-point refinement)
    """
    t   = traj['time']          # (N_EVAL,)
    pos = traj['positions']     # (N_EVAL, 6)
    vel = traj['velocities']
    acc = traj['accelerations']

    # Format A: trajectory_points  (collision_checker, gazebo_executor)
    traj_pts = [
        {'time': float(t[i]), 'joints': pos[i].tolist()}
        for i in range(len(t))
    ]

    return {
        # ── Format A ─────────────────────────────────────────────────────────
        'trajectory_points': traj_pts,

        # ── Format B (flat arrays) ────────────────────────────────────────────
        'time':            t.tolist(),
        'positions':       pos.tolist(),
        'velocities':      vel.tolist(),
        'accelerations':   acc.tolist(),
        'num_samples':     traj['num_samples'],

        # ── Waypoints for Kuramoto control-point refinement ───────────────────
        'joint_waypoints': traj['joint_waypoints'].tolist(),
        'waypoint_times':  traj['waypoint_times'].tolist(),

        # ── Metadata ─────────────────────────────────────────────────────────
        'metadata': traj['metadata'],
        'duration': float(traj['time'][-1]),
    }


# ============================================================================
# SECTION 6 — Helpers
# ============================================================================

def _extract_path_samples(sol: dict, arm_id: str):
    """
    Extract (joint_wps, s_values) from Stage-1 IK solution.

    Stage 1 (dual_arm_ik_solver.py) stores chained IK solutions in:
      sol['ik_path_samples']['joint_solutions']  (N×6)
      sol['ik_path_samples']['s_values']         (N,) uniform in [0,1]

    Falls back to 2-point (start→end) if Stage 1 failed.
    """
    ps = sol.get('ik_path_samples')
    if ps is not None and ps.get('n_success', 0) >= 2:
        js = np.array(ps['joint_solutions'], dtype=float)   # (N, 6)
        sv = np.array(ps['s_values'],        dtype=float)   # (N,)
        print(f"   {arm_id}: {ps['n_success']}/{ps['n_samples']} IK samples  "
              f"curvature_N={ps.get('curvature_n_adaptive', ps['n_samples'])}")
        return js, sv

    # 2-point fallback — minimum viable trajectory (no intermediate waypoints)
    print(f"   {arm_id}: ⚠  no ik_path_samples — using 2-point fallback")
    q_s = np.array(sol['current_joints'], dtype=float)
    q_e = np.array(sol['optimal_joints'],  dtype=float)
    return np.vstack([q_s, q_e]), np.array([0.0, 1.0])


def _pad_to_duration(traj: dict, T_new: float) -> dict:
    """
    Proportionally stretch waypoint times so both arms share the same duration.
    Called when one arm finishes before the other after limit scaling.
    """
    T_old = float(traj['time'][-1])
    if T_old >= T_new - 1e-9:
        return traj
    wps   = traj['joint_waypoints']
    t_wps = traj['waypoint_times'] * (T_new / T_old)
    updated = fit_bspline(wps, t_wps, T_new)
    updated['robot_name'] = traj['robot_name']
    updated['metadata']   = {**traj['metadata'], 'duration': T_new}
    return updated


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 80)
    print("TRAJECTORY GENERATION  —  Cartesian Min-Jerk B-Spline  (Paper §IV-C)")
    print("=" * 80)
    print("  Paper method   : Cartesian min-jerk path + IK + B-spline  (§IV-C)")
    print("  Independence   : Each arm planned separately — no cross-arm NLP")
    print("  IK source      : ik_path_samples from dual_arm_ik_solver (Stage 1)")
    print("  Time param     : s(τ) = 10τ³ − 15τ⁴ + 6τ⁵  inverted → t_i")
    print("  Synchronisation: deferred to kuramoto_synchronization (Stage 4)")
    print("  Input          : ik_solutions.json")
    print("  Output         : trajectories.json")
    print("=" * 80)

    # ── Load Stage-1 output ───────────────────────────────────────────────────
    if not os.path.exists('ik_solutions.json'):
        print("\n❌  ik_solutions.json not found")
        print("    Run: ros2 run dual_arm_sync dual_arm_ik_solver")
        return

    with open('ik_solutions.json') as f:
        ik = json.load(f)

    sol1 = ik.get('dsr01')
    sol2 = ik.get('dsr02')
    if sol1 is None or sol2 is None:
        print("\n❌  ik_solutions.json missing 'dsr01' or 'dsr02'")
        return

    T_req  = float(ik.get('duration', 10.0))
    method = ik.get('method', 'unknown')
    print(f"\n✓  ik_solutions.json loaded")
    print(f"   IK method    : {method}")
    print(f"   Requested T  : {T_req:.1f}s")

    base1 = RobotBases.DSR01_BASE    # [0,  0.5, 0]
    base2 = RobotBases.DSR02_BASE    # [0, -0.5, 0]

    # ── Extract chained IK samples ────────────────────────────────────────────
    js1, sv1 = _extract_path_samples(sol1, 'dsr01')
    js2, sv2 = _extract_path_samples(sol2, 'dsr02')

    # ── Per-arm trajectory generation  (FULLY INDEPENDENT) ───────────────────
    print("\n" + "─" * 80)
    print("Building trajectories  (each arm independent — paper §IV-C without central NLP)")
    print("─" * 80)

    traj1 = build_arm_trajectory(
        robot_name = 'dsr01',
        joint_wps  = js1,
        s_values   = sv1,
        q_start    = np.array(sol1['current_joints'],   dtype=float),
        q_end      = np.array(sol1['optimal_joints'],   dtype=float),
        p_start    = np.array(sol1['current_world_pos'], dtype=float),
        p_end      = np.array(sol1['target_world_pos'],  dtype=float),
        base       = base1,
        T          = T_req,
    )

    traj2 = build_arm_trajectory(
        robot_name = 'dsr02',
        joint_wps  = js2,
        s_values   = sv2,
        q_start    = np.array(sol2['current_joints'],   dtype=float),
        q_end      = np.array(sol2['optimal_joints'],   dtype=float),
        p_start    = np.array(sol2['current_world_pos'], dtype=float),
        p_end      = np.array(sol2['target_world_pos'],  dtype=float),
        base       = base2,
        T          = T_req,
    )

    # ── Align to common duration  ─────────────────────────────────────────────
    # Arms are independent; the slower one sets the common duration.
    # Kuramoto will handle synchronisation — we just need the same time axis.
    T1    = float(traj1['time'][-1])
    T2    = float(traj2['time'][-1])
    T_max = max(T1, T2)
    print(f"\n  Duration alignment : dsr01={T1:.3f}s  dsr02={T2:.3f}s  → common={T_max:.3f}s")
    traj1 = _pad_to_duration(traj1, T_max)
    traj2 = _pad_to_duration(traj2, T_max)

    # ── Save trajectories.json ────────────────────────────────────────────────
    #
    # NOTE on top-level structure:
    #   collision_checker.py does  data.get('trajectories', data)
    #   → if no 'trajectories' key, it treats the whole dict as {arm_id: arm_data}
    #   → so flat layout {arm_ids, duration, dsr01:{...}, dsr02:{...}} works correctly
    #
    output = {
        'arm_ids':  ['dsr01', 'dsr02'],
        'duration': T_max,
        'method':   'cartesian_minjerk_bspline_independent',
        'dsr01':    make_arm_output(traj1),
        'dsr02':    make_arm_output(traj2),
    }

    with open('trajectories.json', 'w') as f:
        json.dump(output, f, indent=2)

    kb = os.path.getsize('trajectories.json') / 1024
    print(f"\n{'─' * 80}")
    print(f"✓  Saved: trajectories.json  ({kb:.0f} KB)")
    print(f"   dsr01 : {N_EVAL} samples  T={traj1['time'][-1]:.3f}s  "
          f"waypoints={len(traj1['joint_waypoints'])}")
    print(f"   dsr02 : {N_EVAL} samples  T={traj2['time'][-1]:.3f}s  "
          f"waypoints={len(traj2['joint_waypoints'])}")
    print(f"\n{'=' * 80}")
    print("Next step: ros2 run dual_arm_sync collision_checker")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()