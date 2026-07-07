#!/usr/bin/env python3
"""
local_deformation.py
Local Deformation — Geometric Waypoint Finder Only

Role in pipeline (called only when Kuramoto fails to resolve collision):
  - Find FIRST collision timestep in the two trajectories
  - Identify worst-penetrating link pair
  - Perturb end-effector away from collision, solve IK
  - Write deformation_waypoints.json with:
      * which arm was deformed
      * original start joints
      * deformed (collision-free) waypoint joints
      * original end joints
      * collision_idx (for duration split in trajectory_generation)

This module does NOT generate trajectories.
Trajectory generation is handled by trajectory_generation.py (two-segment mode).

Input:  trajectories.json
Output: deformation_waypoints.json

Usage:
    ros2 run dual_arm_sync local_deformation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import os

try:
    from dual_arm_sync.constants import DHParameters, JointLimits
    from dual_arm_sync.ik_solver import (
        forward_kinematics,
        solve_ik_numerical,
        select_optimal_solution,
        RobotBases,
    )
except ImportError:
    print("WARNING: Could not import dual_arm_sync modules - running in test mode.")


# ============================================================================
# CONFIGURATION
# ============================================================================

class DeformationConfig:
    PERTURBATION_MAGNITUDE    = 0.12
    PERTURBATION_SCALE_FACTOR = 1.5
    MAX_PERTURBATION_ATTEMPTS = 6
    IK_POSITION_TOLERANCE     = 0.015   # 15 mm
    SAFETY_MARGIN             = 0.15    # 15 cm — must match collision_checker.py
    LINK_RADII = np.array([0.10, 0.10, 0.08, 0.08, 0.06, 0.06])

    SEARCH_DIRECTIONS = [
        np.array([ 1.,  0.,  0.]),
        np.array([-1.,  0.,  0.]),
        np.array([ 0.,  1.,  0.]),
        np.array([ 0., -1.,  0.]),
        np.array([ 0.,  0.,  1.]),
        np.array([ 0.,  0., -1.]),
        np.array([ 1.,  0.,  1.]) / np.sqrt(2.),
        np.array([-1.,  0.,  1.]) / np.sqrt(2.),
        np.array([ 0.,  1.,  1.]) / np.sqrt(2.),
        np.array([ 0., -1.,  1.]) / np.sqrt(2.),
    ]


# ============================================================================
# FK — ALL LINK POSITIONS IN WORLD FRAME
# ============================================================================

def compute_all_link_positions(joint_angles: np.ndarray,
                               robot_base:   np.ndarray) -> np.ndarray:
    """Return (6,3) world-frame link tip positions. Mirrors collision_checker.py."""
    dh_params = DHParameters.get_dh_params(joint_angles)
    T = np.eye(4)
    link_positions = []
    for i in range(6):
        alpha, a, theta, d = dh_params[i]
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        T_i = np.array([
            [ct,    -st,     0.,    a    ],
            [st*ca,  ct*ca, -sa,   -sa*d ],
            [st*sa,  ct*sa,  ca,    ca*d ],
            [0.,     0.,     0.,    1.   ],
        ])
        T = T @ T_i
        link_positions.append(T[:3, 3] + robot_base)
    return np.array(link_positions)


# ============================================================================
# COLLISION CHECK
# ============================================================================

def check_pair_collision(links1: np.ndarray,
                         links2: np.ndarray) -> Tuple[bool, List[Tuple]]:
    """Check all 36 link-pair distances. Returns (collision_exists, colliding_pairs)."""
    colliding_pairs = []
    for i in range(6):
        for j in range(6):
            dist   = float(np.linalg.norm(links1[i] - links2[j]))
            thresh = float(
                DeformationConfig.LINK_RADII[i]
                + DeformationConfig.LINK_RADII[j]
                + DeformationConfig.SAFETY_MARGIN
            )
            if dist < thresh:
                colliding_pairs.append((i, j, dist, thresh))
    return len(colliding_pairs) > 0, colliding_pairs


# ============================================================================
# FIND FIRST COLLISION TIMESTEP
# ============================================================================

def find_first_collision_timestep(traj1: Dict,
                                  traj2: Dict
                                  ) -> Tuple[Optional[int], Optional[float]]:
    """
    Scan both trajectories sample-by-sample and return (idx, time) of
    first collision, or (None, None) if clear.
    """
    pos1     = np.array(traj1['trajectory']['positions'])
    pos2     = np.array(traj2['trajectory']['positions'])
    time_vec = np.array(traj1['trajectory']['time'])

    base1 = RobotBases.get_base_position(traj1['robot_name'])
    base2 = RobotBases.get_base_position(traj2['robot_name'])

    for idx in range(min(len(pos1), len(pos2))):
        links1 = compute_all_link_positions(pos1[idx], base1)
        links2 = compute_all_link_positions(pos2[idx], base2)
        collides, _ = check_pair_collision(links1, links2)
        if collides:
            t = float(time_vec[idx]) if idx < len(time_vec) else float(idx) * 0.01
            return idx, t

    return None, None


# ============================================================================
# REPULSION DIRECTION
# ============================================================================

def compute_repulsion_direction(colliding_link_pos: np.ndarray,
                                other_link_pos:     np.ndarray) -> np.ndarray:
    """Unit vector FROM other_link TOWARD colliding_link. Falls back to +Z."""
    diff = colliding_link_pos - other_link_pos
    norm = np.linalg.norm(diff)
    if norm < 1e-6:
        return np.array([0., 0., 1.])
    return diff / norm


# ============================================================================
# FIND COLLISION-FREE DEFORMED WAYPOINT
# ============================================================================

def find_deformed_waypoint(joints_arm:       np.ndarray,
                           joints_other:     np.ndarray,
                           robot_name:       str,
                           other_robot_name: str,
                           colliding_pairs:  List[Tuple],
                           verbose:          bool = True) -> Optional[np.ndarray]:
    """
    Find a collision-free joint configuration by perturbing the end-effector.

    Args:
        joints_arm       : (6,) joints of arm being deformed at collision instant
        joints_other     : (6,) joints of other arm at same instant
        robot_name       : arm being deformed
        other_robot_name : other arm
        colliding_pairs  : pre-computed by caller from world-frame link positions
        verbose          : print progress

    Returns:
        collision-free (6,) joint config, or None
    """
    # Base positions for evaluating IK candidates
    base_arm   = RobotBases.get_base_position(robot_name)
    base_other = RobotBases.get_base_position(other_robot_name)

    # End-effector in LOCAL frame
    ee_local, _ = forward_kinematics(joints_arm)

    # Worst-penetrating pair
    worst_pair     = min(colliding_pairs, key=lambda p: p[2] / p[3])
    link_idx_arm   = worst_pair[0]
    link_idx_other = worst_pair[1]

    # World-frame positions for repulsion direction only
    links_arm_w   = compute_all_link_positions(joints_arm,   base_arm)
    links_other_w = compute_all_link_positions(joints_other, base_other)

    repulsion_dir = compute_repulsion_direction(
        links_arm_w[link_idx_arm],
        links_other_w[link_idx_other]
    )

    if verbose:
        print(f"    [LocalDeform] arm={robot_name}")
        print(f"      Worst pair  : arm_link{link_idx_arm} <-> other_link{link_idx_other}")
        print(f"      Dist/thresh : {worst_pair[2]*100:.1f}cm / {worst_pair[3]*100:.1f}cm")
        print(f"      Repulsion   : {np.round(repulsion_dir, 3)}")

    directions = [repulsion_dir] + list(DeformationConfig.SEARCH_DIRECTIONS)
    magnitude  = DeformationConfig.PERTURBATION_MAGNITUDE

    for attempt in range(DeformationConfig.MAX_PERTURBATION_ATTEMPTS):
        for direction in directions:
            new_ee_local = ee_local + direction * magnitude

            solutions = solve_ik_numerical(new_ee_local, joints_arm)
            if not solutions:
                continue

            result = select_optimal_solution(solutions, joints_arm, verbose=False)
            if result is None:
                continue

            candidate_joints, _ = result

            # FK accuracy check
            fk_pos, _ = forward_kinematics(candidate_joints)
            if np.linalg.norm(fk_pos - new_ee_local) > DeformationConfig.IK_POSITION_TOLERANCE:
                continue

            # Collision-free check against other arm
            links_cand         = compute_all_link_positions(candidate_joints, base_arm)
            links_other_static = compute_all_link_positions(joints_other,     base_other)

            still_collides, _ = check_pair_collision(links_cand, links_other_static)
            if still_collides:
                continue

            min_clr = min(
                np.linalg.norm(links_cand[i] - links_other_static[j])
                for i in range(6) for j in range(6)
            )
            if verbose:
                print(f"      OK attempt {attempt+1} mag={magnitude*100:.1f}cm "
                      f"dir={np.round(direction,2)} clr={min_clr*100:.1f}cm")
            return candidate_joints

        magnitude *= DeformationConfig.PERTURBATION_SCALE_FACTOR
        if verbose:
            print(f"      Scaling to {magnitude*100:.1f}cm (attempt {attempt+1})")

    if verbose:
        print(f"      FAIL: No collision-free waypoint for {robot_name}")
    return None


# ============================================================================
# TOP-LEVEL: RUN LOCAL DEFORMATION
# ============================================================================

def run_local_deformation(traj1: Dict,
                          traj2: Dict,
                          verbose: bool = True) -> Optional[Dict]:
    """
    Find the first collision, decide which arm to deform (the one that
    has moved further in joint space), find its collision-free waypoint,
    and return a deformation_waypoints dict.

    Returns None if no collision exists or deformation fails.

    The returned dict contains everything trajectory_generation needs
    to rebuild the deformed arm's trajectory in two-segment mode:
    {
        'deformed_arm'   : 'dsr01' or 'dsr02',
        'static_arm'     : the other one,
        'collision_idx'  : int,
        'collision_time' : float,
        'collision_frac' : float  (collision_idx / total_samples),
        'start_joints'   : [6 floats, radians],
        'deformed_joints': [6 floats, radians],
        'end_joints'     : [6 floats, radians],
        'total_duration' : float
    }
    """
    print("\n" + "=" * 70)
    print("LOCAL DEFORMATION - FINDING COLLISION-FREE WAYPOINT")
    print("=" * 70)

    # Find first collision
    coll_idx, coll_time = find_first_collision_timestep(traj1, traj2)

    if coll_idx is None:
        print("  No collision found - deformation not needed.")
        return None

    print(f"  First collision: idx={coll_idx}, t={coll_time:.3f}s")

    pos1 = np.array(traj1['trajectory']['positions'])
    pos2 = np.array(traj2['trajectory']['positions'])
    idx2 = min(coll_idx, len(pos2) - 1)

    # Decide which arm to deform - the one further in joint space from start
    motion1 = float(np.linalg.norm(pos1[coll_idx] - pos1[0]))
    motion2 = float(np.linalg.norm(pos2[idx2]     - pos2[0]))
    deform_arm1 = (motion1 >= motion2)

    if deform_arm1:
        traj_deform = traj1
        traj_static = traj2
        arm_deform  = traj1['robot_name']
        arm_static  = traj2['robot_name']
        col_idx_d   = coll_idx
    else:
        traj_deform = traj2
        traj_static = traj1
        arm_deform  = traj2['robot_name']
        arm_static  = traj1['robot_name']
        col_idx_d   = idx2

    print(f"  Motion: {traj1['robot_name']}={motion1:.4f} rad, "
          f"{traj2['robot_name']}={motion2:.4f} rad")
    print(f"  Deforming: {arm_deform}")

    # Extract joint configs at collision instant
    pos_d   = np.array(traj_deform['trajectory']['positions'])
    pos_s   = np.array(traj_static['trajectory']['positions'])
    time_d  = np.array(traj_deform['trajectory']['time'])

    joints_at_col       = pos_d[col_idx_d]
    joints_other_at_col = pos_s[min(col_idx_d, len(pos_s) - 1)]

    # Compute colliding pairs (in deform_trajectory scope)
    base_d = RobotBases.get_base_position(arm_deform)
    base_s = RobotBases.get_base_position(arm_static)

    links_d = compute_all_link_positions(joints_at_col,       base_d)
    links_s = compute_all_link_positions(joints_other_at_col, base_s)

    has_col, colliding_pairs = check_pair_collision(links_d, links_s)

    if not has_col:
        print("  No collision pairs found at collision index - deformation skipped.")
        return None

    # Find the deformed waypoint
    deformed_joints = find_deformed_waypoint(
        joints_arm       = joints_at_col,
        joints_other     = joints_other_at_col,
        robot_name       = arm_deform,
        other_robot_name = arm_static,
        colliding_pairs  = colliding_pairs,
        verbose          = verbose,
    )

    if deformed_joints is None:
        print("  FAIL: Could not find collision-free waypoint.")
        return None

    total_duration = float(time_d[-1])
    collision_frac = col_idx_d / max(len(pos_d) - 1, 1)

    result = {
        'deformed_arm':    arm_deform,
        'static_arm':      arm_static,
        'collision_idx':   int(col_idx_d),
        'collision_time':  float(coll_time),
        'collision_frac':  float(collision_frac),
        'start_joints':    pos_d[0].tolist(),
        'deformed_joints': deformed_joints.tolist(),
        'end_joints':      pos_d[-1].tolist(),
        'total_duration':  total_duration,
        'deformed_joints_deg': np.degrees(deformed_joints).tolist(),
    }

    print(f"\n  Deformed waypoint found:")
    print(f"    Arm        : {arm_deform}")
    print(f"    Collision  : idx={col_idx_d}, frac={collision_frac:.3f}")
    print(f"    Joints(deg): {np.round(np.degrees(deformed_joints), 2).tolist()}")

    return result


# ============================================================================
# STANDALONE MAIN
# ============================================================================

def main():
    """
    Standalone: reads trajectories.json, writes deformation_waypoints.json.
    In the pipeline this is called by results_step1.py programmatically.
    """
    print("\n" + "=" * 70)
    print("LOCAL DEFORMATION - STANDALONE")
    print("Input:  trajectories.json")
    print("Output: deformation_waypoints.json")
    print("=" * 70)

    if not os.path.exists('trajectories.json'):
        print("ERROR: trajectories.json not found.")
        return

    with open('trajectories.json', 'r') as f:
        raw = json.load(f)

    def _load(d):
        return {
            'robot_name': d['robot_name'],
            'metadata':   d.get('metadata', {}),
            'trajectory': {
                'time':          np.array(d['trajectory']['time']),
                'positions':     np.array(d['trajectory']['positions']),
                'velocities':    np.array(d['trajectory']['velocities']),
                'accelerations': np.array(d['trajectory']['accelerations']),
                'num_samples':   int(d['trajectory']['num_samples']),
            },
        }

    traj1 = _load(raw['dsr01'])
    traj2 = _load(raw['dsr02'])

    result = run_local_deformation(traj1, traj2, verbose=True)

    if result is None:
        print("\nNo deformation waypoints produced.")
        return

    with open('deformation_waypoints.json', 'w') as f:
        json.dump(result, f, indent=2)

    size_kb = os.path.getsize('deformation_waypoints.json') / 1024
    print(f"\nSaved: deformation_waypoints.json ({size_kb:.1f} KB)")
    print("\nNext: ros2 run dual_arm_sync trajectory_generation  (two-segment mode)")


if __name__ == '__main__':
    main()