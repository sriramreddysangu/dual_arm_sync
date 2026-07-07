#!/usr/bin/env python3
"""
dual_arm_ik_solver.py
=====================
Stage 1 — takes (x, y, z, roll, pitch, yaw) target for each arm,
           samples the Cartesian path with curvature-adaptive N,
           runs chained IK at every sample, writes ik_solutions.json.

Paper: Kabir et al. 2019  Section IV-C  "Generating Approximate Solution"

  Step 1 — User provides 6-DOF target  (position + RPY orientation)
  Step 2 — Curvature-adaptive path sampling  (paper §IV-C):
              - Straight-line Cartesian path  start → target
              - Orientation path: SLERP  R_start → R_target  (s ∈ [0,1])
              - Compute curvature κ(s) = ‖d²P/ds²‖  at dense evaluation
              - Integrate curvature variation  K_total = ∫ κ(s) ds
              - N_samples = clip(N_BASE + round(K_total / κ_per_sample),
                                 N_MIN, N_MAX)
                → straight paths get fewer samples; curved paths get more
  Step 3 — Chained IK at each sample:
              - Initial guess  = previous sample's joints (branch continuity)
              - Orientation  = SLERP(R_start, R_target, s_i)  for each sample
              - Selection rule = lexicographic  P1→P2→P3→P4
  Step 4 — Write ik_solutions.json

Output: ik_solutions.json
  {
    "arm_ids":  [...],
    "duration": 10.0,
    "method":   "paper_path_sampling_orient_adaptive",
    "dsr01": {
      "current_joints":  [...],
      "optimal_joints":  [...],
      "target_orient_rpy": [roll, pitch, yaw],
      "ik_path_samples": {
        "joint_solutions": [[...], ...],   ← (N×6) → trajectory_generation
        "success_mask":    [...],
        "s_values":        [...],
        "n_samples":       int,
        "n_success":       int,
        "orient_matrices": [[[...]], ...], ← (N×3×3) SLERP'd orientations
        "orient_errors_rad": [...],        ← per-sample orientation error
        "curvature_n_adaptive": int        ← N chosen by curvature analysis
      }
    },
    ...
  }

Usage:
    ros2 run dual_arm_sync dual_arm_ik_solver
"""

import numpy as np

# ROS2 imports — guarded so the module can be imported standalone (tests, CI)
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    _ROS2_AVAILABLE = True
except ImportError:
    rclpy = None
    Node = object
    JointState = None
    _ROS2_AVAILABLE = False
import json
from typing import Dict, List, Optional, Tuple

try:
    from dual_arm_sync.ik_solver import (
        forward_kinematics, solve_ik_numerical,
        select_optimal_solution, select_chained_solution,
        ARM_REGISTRY, RobotBases, IK_POS_TOL, JOINT_LIMITS,
        rpy_to_matrix, matrix_to_rpy, slerp,
        interpolate_orientations, orientation_error, get_target_rotation,
        _normalize_joints_to_current,
    )
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from ik_solver import (
        forward_kinematics, solve_ik_numerical,
        select_optimal_solution, select_chained_solution,
        ARM_REGISTRY, RobotBases, IK_POS_TOL, JOINT_LIMITS,
        rpy_to_matrix, matrix_to_rpy, slerp,
        interpolate_orientations, orientation_error, get_target_rotation,
        _normalize_joints_to_current,
    )


# ============================================================================
# CONFIGURATION
# ============================================================================

# Curvature-adaptive sampling  (paper §IV-C)
N_BASE          = 25    # baseline path samples (was 10 — too coarse for branch continuity)
N_MIN           = 20    # absolute minimum
N_MAX           = 60    # absolute maximum
KAPPA_PER_STEP  = 0.05  # 1 extra sample per this many rad/m of integrated curvature
N_DENSE_EVAL    = 500   # dense points for curvature integration

CHAIN_JUMP_WARN_RAD = 1.0   # warn if consecutive IK solutions jump > this [rad]
DEFAULT_DURATION_S  = 10.0

# Arm base positions — add more arms here (up to 10)
ARM_CONFIG: Dict[str, np.ndarray] = {
    'dsr01': RobotBases.DSR01_BASE,
    'dsr02': RobotBases.DSR02_BASE,
    # 'dsr03': np.array([0.0,  1.5, 0.0]),
}


# ============================================================================
# CURVATURE-ADAPTIVE SAMPLING  (paper §IV-C)
# ============================================================================

def compute_adaptive_n(start_pos: np.ndarray,
                        end_pos:   np.ndarray,
                        R_start:   np.ndarray,
                        R_end:     np.ndarray,
                        verbose:   bool = True) -> int:
    """
    Paper §IV-C: choose N proportional to path complexity.

    Curvature measure κ(s) combines:
      - Positional curvature: rate of change of path tangent direction
        (for a straight line κ_pos = 0; curves give κ_pos > 0)
      - Orientational curvature: ‖dR/ds‖_F  (geodesic rate of rotation)
        from the SLERP path between R_start and R_end

    For a pure straight-line translation with no orientation change,
    this returns N_BASE.  The more the orientation rotates or the more
    the Cartesian path bends, the larger N becomes (up to N_MAX).

    Args:
        start_pos : (3,)  world frame
        end_pos   : (3,)  world frame
        R_start   : (3,3) start orientation
        R_end     : (3,3) end orientation

    Returns:
        N_samples : int  ∈ [N_MIN, N_MAX]
    """
    s  = np.linspace(0.0, 1.0, N_DENSE_EVAL)
    ds = 1.0 / (N_DENSE_EVAL - 1)

    # ---- Positional curvature ----
    # pos(s) is linear → tangent is constant → κ_pos = 0 for straight line.
    # This captures any non-linearity if the user extends to curved paths later.
    pos_path = start_pos[None, :] + np.outer(s, end_pos - start_pos)  # (N,3)
    tangent  = np.gradient(pos_path, ds, axis=0)
    t_norm   = np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-12
    t_unit   = tangent / t_norm
    dt_unit  = np.gradient(t_unit, ds, axis=0)
    kappa_pos = np.linalg.norm(dt_unit, axis=1)   # (N,)

    # ---- Orientational curvature  ‖dR/ds‖_F ----
    kappa_ori = np.zeros(N_DENSE_EVAL)
    q0 = _rot_to_quat(R_start)
    q1 = _rot_to_quat(R_end)
    if np.dot(q0, q1) < 0:
        q1 = -q1
    dot   = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    theta = np.arccos(dot)   # total rotation angle (rad)
    # ‖dR/ds‖ ≈ θ_total for SLERP (constant angular velocity)
    kappa_ori[:] = theta

    # ---- Integrated curvature variation ----
    kappa_total = kappa_pos + kappa_ori
    K_total     = float(np.trapezoid(kappa_total, s)
                       if hasattr(np, 'trapezoid') else np.trapz(kappa_total, s))

    n_extra  = int(round(K_total / KAPPA_PER_STEP))
    n_samples = int(np.clip(N_BASE + n_extra, N_MIN, N_MAX))

    if verbose:
        path_len_mm = float(np.linalg.norm(end_pos - start_pos)) * 1000
        ori_deg     = np.degrees(theta)
        print(f'    curvature adaptive N:'
              f'  path={path_len_mm:.1f} mm'
              f'  ΔR={ori_deg:.1f}°'
              f'  K_total={K_total:.3f}'
              f'  N_base={N_BASE} + N_extra={n_extra}'
              f'  → N={n_samples}')

    return n_samples


def _rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → unit quaternion [w,x,y,z] (local helper)."""
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return np.array([0.25/s, (R[2,1]-R[1,2])*s,
                         (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s])
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        return np.array([(R[2,1]-R[1,2])/s, 0.25*s,
                         (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s])
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        return np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s,
                         0.25*s, (R[1,2]+R[2,1])/s])
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        return np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s,
                         (R[1,2]+R[2,1])/s, 0.25*s])


# ============================================================================
# BRANCH-JUMP SMOOTHING
# ============================================================================

def _smooth_branch_jumps(joint_solutions: np.ndarray,
                          success_mask:   List[bool]) -> np.ndarray:
    """
    Post-process a joint trajectory to eliminate residual branch jumps.

    For each consecutive valid sample pair (i-1, i), normalize sample i so
    that each joint angle is the equivalent angle (mod 2π) closest to the
    PREVIOUS sample — preventing oscillation between +π and −π branches.

    Called AFTER the chained IK loop so that normalization is applied to
    the FULL collected sequence, not just IK-call-by-call.

    Args:
        joint_solutions : (N, 6) — raw IK output (may have branch jumps)
        success_mask    : length-N bool list

    Returns:
        smoothed (N, 6) — each sample normalised to its predecessor
    """
    out  = joint_solutions.copy()
    prev = None
    for i in range(len(out)):
        if not success_mask[i]:
            continue
        if prev is None:
            prev = out[i].copy()
            continue
        out[i] = _normalize_joints_to_current(out[i], prev)
        prev   = out[i].copy()
    return out




def chained_ik_along_path(start_world:   np.ndarray,
                           end_world:     np.ndarray,
                           R_start:       np.ndarray,
                           R_end:         np.ndarray,
                           start_joints:  np.ndarray,
                           end_joints:    np.ndarray,
                           base:          np.ndarray,
                           arm_id:        str,
                           n_samples:     int) -> Dict:
    """
    Paper §IV-C chained IK with full 6-DOF targets (position + SLERP orientation).

    Key fixes applied here
    ──────────────────────
    1. s=0 anchored to start_joints  — no IK solved at start, eliminates
       the 400°+ branch-jump that occurred because IK at the home position
       finds J4=−180° (high manipulability) even though the arm is at J4=0°.

    2. s=1 anchored to normalized end_joints  — normalize to prev_joints so
       the arm arrives on the same branch it was tracking (e.g. J4=+180° not
       J4=−180° when coming from J4=+90°).

    3. Post-processing via _smooth_branch_jumps()  — after the loop, the
       entire collected sequence is re-normalised sample-by-sample.  This
       fixes mid-path branch flips that the per-step logic can miss.
    """
    dist_mm = float(np.linalg.norm(end_world - start_world)) * 1000
    print(f'\n  [{arm_id}]  path = {dist_mm:.1f} mm   N_samples = {n_samples}')
    print(f'    pos start : {np.round(start_world, 4)}')
    print(f'    pos end   : {np.round(end_world,   4)}')
    r0, p0, y0 = [np.degrees(v) for v in _rpy_from_matrix(R_start)]
    r1, p1, y1 = [np.degrees(v) for v in _rpy_from_matrix(R_end)]
    print(f'    ori start : roll={r0:.1f}°  pitch={p0:.1f}°  yaw={y0:.1f}°')
    print(f'    ori end   : roll={r1:.1f}°  pitch={p1:.1f}°  yaw={y1:.1f}°')

    s_values   = np.linspace(0.0, 1.0, n_samples)
    other_arms = ARM_REGISTRY.get_other_arms(arm_id)

    joint_solutions = np.zeros((n_samples, 6))
    success_mask    = [False] * n_samples
    orient_matrices = []
    orient_errors   = []
    prev_joints     = start_joints.copy()
    n_failed        = 0

    for i, sv in enumerate(s_values):
        R_i = slerp(R_start, R_end, float(sv))

        # ── Anchor s=0: use start_joints exactly (no IK) ──────────────────
        if sv < 1e-9:
            joint_solutions[i] = start_joints.copy()
            success_mask[i]    = True
            _, fk_rot0, _      = forward_kinematics(start_joints)
            orient_matrices.append(R_start.tolist())
            orient_errors.append(float(orientation_error(fk_rot0, R_i)))
            prev_joints = start_joints.copy()
            continue

        # ── Anchor s=1: use normalized end_joints (no IK) ─────────────────
        if sv > 1.0 - 1e-9 and n_samples > 1:
            best = _normalize_joints_to_current(end_joints, prev_joints)
            joint_solutions[i] = best
            success_mask[i]    = True
            _, fk_rot1, _      = forward_kinematics(best)
            orient_matrices.append(R_end.tolist())
            orient_errors.append(float(orientation_error(fk_rot1, R_end)))
            prev_joints = best.copy()
            continue

        # ── Intermediate: solve IK ─────────────────────────────────────────
        pos_w  = start_world + sv * (end_world - start_world)
        pos_lo = pos_w - base

        sols = solve_ik_numerical(
            pos_lo,
            target_orient  = R_i,
            initial_guess  = prev_joints,
            n_restarts     = 6,
            orient_weight  = 0.5,
        )

        if not sols:
            joint_solutions[i] = prev_joints.copy()
            orient_matrices.append(R_i.tolist())
            orient_errors.append(999.0)
            n_failed += 1
            continue

        best = select_chained_solution(
            sols, prev_joints,
            arm_id=arm_id, base=base, other_arms=other_arms,
        )
        if best is None:
            raw  = min(sols, key=lambda s: np.linalg.norm(s - prev_joints))
            best = _normalize_joints_to_current(raw, prev_joints)

        jump = float(np.linalg.norm(best - prev_joints))
        if jump > CHAIN_JUMP_WARN_RAD:
            print(f'    ⚠  [{arm_id}] branch jump {np.degrees(jump):.1f}°'
                  f'  at s={sv:.2f}')

        _, fk_rot, _ = forward_kinematics(best)
        ori_err      = float(orientation_error(fk_rot, R_i))

        joint_solutions[i] = best
        success_mask[i]    = True
        orient_matrices.append(R_i.tolist())
        orient_errors.append(ori_err)
        prev_joints = best.copy()

    # ── Post-process: smooth residual branch jumps ─────────────────────────
    joint_solutions = _smooth_branch_jumps(joint_solutions, success_mask)

    n_ok = sum(success_mask)
    print(f'    IK solved : {n_ok}/{n_samples}   failed : {n_failed}')
    if n_ok >= 2:
        js_ok     = joint_solutions[[bool(m) for m in success_mask]]
        excursion = np.degrees(js_ok.max(0) - js_ok.min(0))
        print(f'    excursion (deg) : {np.round(excursion, 1)}')
        valid_errs = [e for e in orient_errors if e < 100]
        if valid_errs:
            print(f'    mean ori error  : {np.degrees(np.mean(valid_errs)):.2f}°')

    return {
        's_values':              s_values.tolist(),
        'joint_solutions':       joint_solutions.tolist(),
        'success_mask':          success_mask,
        'n_samples':             n_samples,
        'n_success':             n_ok,
        'orient_matrices':       orient_matrices,
        'orient_errors_rad':     orient_errors,
        'curvature_n_adaptive':  n_samples,
        'start_world':           start_world.tolist(),
        'end_world':             end_world.tolist(),
    }




def _rpy_from_matrix(R: np.ndarray) -> Tuple[float, float, float]:
    """RPY from rotation matrix (local helper — avoids import loop)."""
    pitch = np.arctan2(-R[2,0], np.sqrt(R[0,0]**2 + R[1,0]**2))
    if abs(np.cos(pitch)) < 1e-8:
        return 0.0, float(pitch), np.arctan2(-R[1,2], R[1,1])
    return (float(np.arctan2(R[2,1], R[2,2])),
            float(pitch),
            float(np.arctan2(R[1,0], R[0,0])))


# ============================================================================
# ENDPOINT IK  (6-DOF)
# ============================================================================

def solve_endpoint_ik(arm_id:         str,
                       target_world:   np.ndarray,
                       R_target:       np.ndarray,
                       current_joints: np.ndarray,
                       base:           np.ndarray) -> Optional[Dict]:
    """
    Solve IK for the 6-DOF target endpoint (position + orientation).
    Uses geodesic orientation error in cost function.
    """
    target_local = target_world - base
    try:
        ee_lo, ee_rot, _ = forward_kinematics(current_joints)
    except Exception:
        ee_lo, ee_rot = np.zeros(3), np.eye(3)
    ee_world = ee_lo + base

    print(f'\n{"="*80}')
    print(f'ENDPOINT IK  —  {arm_id.upper()}')
    print(f'{"="*80}')
    print(f'  Current joints (deg) : {np.round(np.degrees(current_joints), 2)}')
    print(f'  Current EE  (world)  : {np.round(ee_world, 4)}')
    r_c, p_c, y_c = [np.degrees(v) for v in _rpy_from_matrix(ee_rot)]
    print(f'  Current ori  (deg)   : roll={r_c:.1f}°  pitch={p_c:.1f}°  yaw={y_c:.1f}°')
    print(f'  Target pos  (world)  : {np.round(target_world, 4)}')
    r_t, p_t, y_t = [np.degrees(v) for v in _rpy_from_matrix(R_target)]
    print(f'  Target ori  (deg)    : roll={r_t:.1f}°  pitch={p_t:.1f}°  yaw={y_t:.1f}°')

    solutions = solve_ik_numerical(
        target_local,
        target_orient  = R_target,
        initial_guess  = current_joints,
        n_restarts     = 10,
        orient_weight  = 0.5,
    )

    if not solutions:
        print('  ✗  No IK solutions found')
        return None

    print(f'  ✓  {len(solutions)} solution(s)')

    result = select_optimal_solution(
        solutions, current_joints,
        arm_id=arm_id, base=base, verbose=True,
    )
    if result is None:
        return None

    opt_joints, info = result
    _, fk_rot, _     = forward_kinematics(opt_joints)
    fk_world         = forward_kinematics(opt_joints)[0] + base
    pos_err          = float(np.linalg.norm(fk_world - target_world))
    ori_err          = float(orientation_error(fk_rot, R_target))

    print(f'  Optimal joints (deg) : {np.round(np.degrees(opt_joints), 2)}')
    print(f'  Position error       : {pos_err*1000:.3f} mm')
    print(f'  Orientation error    : {np.degrees(ori_err):.3f}°')
    print(f'  Collision-free       : {info["collision_free"]}')
    print(f'{"="*80}')

    rpy_t = _rpy_from_matrix(R_target)
    rpy_c = _rpy_from_matrix(ee_rot)

    return {
        'current_joints':      current_joints.tolist(),
        'current_joints_deg':  np.degrees(current_joints).tolist(),
        'current_world_pos':   ee_world.tolist(),
        'current_orient_rpy':  list(rpy_c),
        'optimal_joints':      opt_joints.tolist(),
        'optimal_joints_deg':  np.degrees(opt_joints).tolist(),
        'optimal_world_pos':   fk_world.tolist(),
        'target_world_pos':    target_world.tolist(),
        'target_orient_rpy':   list(rpy_t),
        'target_orient_matrix': R_target.tolist(),
        'position_error_m':    pos_err,
        'orientation_error_rad': ori_err,
        'ik_info':             info,
    }


# ============================================================================
# FULL SOLVE (endpoint + adaptive path sampling)
# ============================================================================

def solve_arm(arm_id:         str,
              target_world:   np.ndarray,
              R_target:       np.ndarray,
              current_joints: np.ndarray,
              base:           np.ndarray) -> Optional[Dict]:
    """
    Complete paper method for one arm (with orientation):
      1. Endpoint IK with 6-DOF target  → optimal_joints
      2. Curvature-adaptive N selection
      3. Chained IK along SLERP-augmented path  → ik_path_samples
    """
    endpoint = solve_endpoint_ik(
        arm_id, target_world, R_target, current_joints, base
    )
    if endpoint is None:
        return None

    start_world  = np.array(endpoint['current_world_pos'])
    try:
        _, R_start, _ = forward_kinematics(current_joints)
    except Exception:
        R_start = np.eye(3)

    print(f'\n{"="*80}')
    print(f'PATH SAMPLING  —  {arm_id.upper()}  [paper §IV-C + SLERP orient]')
    print(f'{"="*80}')

    # Curvature-adaptive N
    n_samples = compute_adaptive_n(
        start_world, target_world, R_start, R_target, verbose=True
    )

    path_samples = chained_ik_along_path(
        start_world  = start_world,
        end_world    = target_world,
        R_start      = R_start,
        R_end        = R_target,
        start_joints = current_joints,
        end_joints   = np.array(endpoint['optimal_joints']),
        base         = base,
        arm_id       = arm_id,
        n_samples    = n_samples,
    )

    result = dict(endpoint)
    result['arm_id']         = arm_id
    result['ik_path_samples'] = path_samples
    return result


# ============================================================================
# ORIENTATION INPUT PARSING
# ============================================================================

def parse_orientation_input(arm_id: str) -> np.ndarray:
    """
    Interactive orientation input for one arm.

    Options:
      1. Press Enter at Roll prompt → auto-orientation from target position
      2. Enter roll value, then Enter at Pitch/Yaw → default Pitch=0, Yaw=0
      3. Enter all three values explicitly

    Returns: (3,3) rotation matrix, or None (= use auto heuristic)
    """
    def _read_deg(prompt: str, default: float = 0.0) -> float:
        """Read a degree value; return default if user just presses Enter."""
        s = input(prompt).strip()
        if not s:
            return default
        return float(s)

    print(f'\n  {arm_id.upper()} orientation (press Enter on any field for default):')
    roll_str = input('    Roll  (deg, Enter=auto) : ').strip()

    if not roll_str:
        print('    → Auto-orientation (computed from target position)')
        return None   # None signals "use auto" to solve_arm

    roll  = np.radians(float(roll_str))
    pitch = np.radians(_read_deg('    Pitch (deg, Enter=0)  : ', 0.0))
    yaw   = np.radians(_read_deg('    Yaw   (deg, Enter=0)  : ', 0.0))
    R = rpy_to_matrix(roll, pitch, yaw)
    print(f'    → R  roll={np.degrees(roll):.1f}°  '
          f'pitch={np.degrees(pitch):.1f}°  yaw={np.degrees(yaw):.1f}°')
    return R


# ============================================================================
# ROS2 NODE
# ============================================================================

class DualArmIKSolver(Node):
    """
    ROS2 node — subscribes to joint states, accepts 6-DOF targets
    interactively, runs curvature-adaptive path-sampling IK,
    writes ik_solutions.json.
    """

    def __init__(self):
        super().__init__('dual_arm_ik_solver')

        self.robots: Dict[str, Dict] = {}
        for arm_id, base in ARM_CONFIG.items():
            self.robots[arm_id] = {
                'base':            np.asarray(base, dtype=float),
                'current_joints':  np.zeros(6),
                'joints_received': False,
            }
            ARM_REGISTRY.register(arm_id, base, np.zeros(6))
            self.create_subscription(
                JointState,
                f'/{arm_id}/gz/joint_states',
                lambda msg, aid=arm_id: self._joint_cb(msg, aid),
                10,
            )

        self.get_logger().info('=' * 80)
        self.get_logger().info(
            'DUAL ARM IK SOLVER  [Paper §IV-C + Orient + Adaptive N]')
        self.get_logger().info(f'Arms   : {list(self.robots.keys())}')
        self.get_logger().info(f'N_range: [{N_MIN}, {N_MAX}]  (curvature-adaptive)')
        self.get_logger().info('=' * 80)

    def _joint_cb(self, msg: JointState, arm_id: str):
        if len(msg.position) < 6:
            return
        jmap = {n: i for i, n in enumerate(msg.name)}
        if all(f'joint_{k}' in jmap for k in range(1, 7)):
            joints = np.array([msg.position[jmap[f'joint_{k}']]
                               for k in range(1, 7)])
        else:
            joints = np.array(msg.position[:6])
        self.robots[arm_id]['current_joints'] = joints
        ARM_REGISTRY.update_joints(arm_id, joints)
        if not self.robots[arm_id]['joints_received']:
            self.robots[arm_id]['joints_received'] = True
            self.get_logger().info(f'✓  {arm_id} joint states received')

    def all_ready(self) -> bool:
        return all(r['joints_received'] for r in self.robots.values())

    def solve_all(self, targets: Dict[str, Dict]) -> Optional[Dict]:
        """
        targets = {arm_id: {'pos': np.array([x,y,z]),
                             'orient': np.ndarray(3,3) or None}}
        """
        rclpy.spin_once(self, timeout_sec=0.05)
        results: Dict[str, Dict] = {}
        for arm_id, tgt in targets.items():
            robot  = self.robots[arm_id]
            base   = robot['base']
            joints = robot['current_joints'].copy()
            pos_w  = tgt['pos']
            R_tgt  = tgt['orient']

            if R_tgt is None:
                # auto: compute from target local position
                pos_local = pos_w - base
                R_tgt = get_target_rotation(pos_local)
                print(f'  [{arm_id}] auto-orient from position')

            result = solve_arm(arm_id, pos_w, R_tgt, joints, base)
            if result is None:
                print(f'\n✗  IK failed for {arm_id}')
                return None
            results[arm_id] = result
        return results


# ============================================================================
# MAIN
# ============================================================================

def main(args=None):
    import time
    if not _ROS2_AVAILABLE:
        print('[WARN] rclpy not available — running in standalone mode (no ROS2 publishing)')
    else:
        rclpy.init(args=args)

    print('\n' + '=' * 80)
    print('DUAL ARM IK SOLVER  [Kabir 2019 + Orientation + Adaptive N]')
    print(f'Arms    : {list(ARM_CONFIG.keys())}')
    print(f'N range : [{N_MIN}, {N_MAX}]  (curvature-adaptive)')
    print('=' * 80)

    node = DualArmIKSolver()

    print('\nWaiting for joint states...')
    t0 = time.time()
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.all_ready():
            print('✓  All arms ready\n')
            break
        if time.time() - t0 > 15.0:
            print('✗  Timeout waiting for joint states')
            try:
                node.destroy_node()
            except Exception:
                pass
            try:
                if rclpy.ok():
                    rclpy.shutdown()
            except Exception:
                pass
            return

    arm_ids = list(ARM_CONFIG.keys())

    while rclpy.ok():
        try:
            print('\n' + '=' * 80)
            print('ENTER TARGET POSES  (world frame)')
            print('  Position   : x, y, z in metres')
            print('  Orientation: roll, pitch, yaw in degrees  (or Enter = auto)')
            print('=' * 80)

            targets: Dict[str, Dict] = {}
            for arm_id in arm_ids:
                print(f'\n  {arm_id.upper()} position:')
                x = float(input('    X (m) : '))
                y = float(input('    Y (m) : '))
                z = float(input('    Z (m) : '))
                R = parse_orientation_input(arm_id)
                targets[arm_id] = {'pos': np.array([x, y, z]), 'orient': R}

            print('\n' + '=' * 80)
            print('SOLVING  —  endpoint IK + adaptive path sampling')
            print('=' * 80)

            results = node.solve_all(targets)
            if results is None:
                print('\n✗  IK failed — check targets and retry')
                continue

            output = {
                'method':   'paper_path_sampling_orient_adaptive',
                'n_arms':   len(results),
                'arm_ids':  list(results.keys()),
                'duration': DEFAULT_DURATION_S,
            }
            for arm_id, r in results.items():
                output[arm_id] = r

            with open('ik_solutions.json', 'w') as f:
                json.dump(output, f, indent=2)

            print('\n✓  ik_solutions.json  written')
            for arm_id, r in results.items():
                ns = r['ik_path_samples']['n_success']
                N  = r['ik_path_samples']['n_samples']
                Na = r['ik_path_samples']['curvature_n_adaptive']
                print(f'  {arm_id} : {ns}/{N} solved'
                      f'   adaptive_N={Na}'
                      f'   collision_free={r["ik_info"]["collision_free"]}')

            print('\nNext → ros2 run dual_arm_sync trajectory_generation')

            if input('\nSolve another set? (y/n) : ').strip().lower() != 'y':
                break

        except KeyboardInterrupt:
            break
        except ValueError as e:
            print(f'\n✗  Invalid input: {e}')
        except Exception:
            import traceback
            traceback.print_exc()

    if _ROS2_AVAILABLE:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()