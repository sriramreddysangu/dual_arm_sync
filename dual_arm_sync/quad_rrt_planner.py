#!/usr/bin/env python3
"""
quad_rrt_planner.py
RRT-Connect planner for 4-arm system.

Strategy:
  - Plan each arm individually treating all OTHER arms as static obstacles
    at their START positions (conservative but fast).
  - After all 4 paths are found, verify SIMULTANEOUS execution collision-free.
  - Re-plan with new seeds if simultaneous check fails (up to MAX_ATTEMPTS).

Reads:  ik_solutions.json
Writes: rrt_trajectories.json  (same schema as dual-arm version)

Usage:
    ros2 run dual_arm_sync quad_rrt_planner
"""

import numpy as np
import json
import time
from typing import List, Tuple, Optional, Dict
import sys

try:
    try:
        from dual_arm_sync.quad_arm_config import (
            ARM_NAMES, BASE_POSITIONS, COLLISION_PAIRS,
            N_ARMS, MIN_SAFE_DISTANCE, LINK_RADII,
            JOINT_POSITION_LIMITS,
        )
    except ImportError:
        from quad_arm_config import (
            ARM_NAMES, BASE_POSITIONS, COLLISION_PAIRS,
            N_ARMS, MIN_SAFE_DISTANCE, LINK_RADII,
            JOINT_POSITION_LIMITS,
        )
except ImportError as e:
    print(f"ERROR: {e}")
    sys.exit(1)

# ── DH / FK ───────────────────────────────────────────────────────────────────

DH_PARAMS = [
    (0.0,   0.1555, np.pi/2,  0.0     ),
    (0.409, 0.0,    0.0,      0.0     ),
    (0.0,   0.0,    -np.pi/2, np.pi/2 ),
    (0.0,   0.3995, np.pi/2,  0.0     ),
    (0.0,   0.0,    -np.pi/2, 0.0     ),
    (0.0,   0.082,  0.0,      0.0     ),
]

JOINT_LIMITS = JOINT_POSITION_LIMITS

# ── RRT parameters ────────────────────────────────────────────────────────────
STEP_SIZE       = 0.15    # rad
MAX_ITER        = 5000
GOAL_BIAS       = 0.15
INTERP_SAMPLES  = 1000
DURATION        = 10.0
MAX_ATTEMPTS    = 5
COLLISION_THRESH = MIN_SAFE_DISTANCE


# ── FK / link positions ───────────────────────────────────────────────────────

def _dh_mat(a, d, alpha, theta):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [ 0,  sa,     ca,    d   ],
        [ 0,  0,      0,     1   ],
    ])


def get_link_positions(joints: np.ndarray, base: np.ndarray) -> List[np.ndarray]:
    T   = np.eye(4)
    pts = [base.copy()]
    for i, (a, d, alpha, t0) in enumerate(DH_PARAMS):
        T = T @ _dh_mat(a, d, alpha, joints[i] + t0)
        pts.append(T[:3, 3] + base)
    return pts   # 7 points: base + 6 links


def min_dist_pair(joints_a: np.ndarray, base_a: np.ndarray,
                   joints_b: np.ndarray, base_b: np.ndarray) -> float:
    la = get_link_positions(joints_a, base_a)
    lb = get_link_positions(joints_b, base_b)
    return float(min(
        np.linalg.norm(la[i] - lb[j])
        for i in range(1, len(la))
        for j in range(1, len(lb))
    ))


# ── Collision checks ──────────────────────────────────────────────────────────

def joints_valid(joints: np.ndarray) -> bool:
    return bool(np.all(joints >= JOINT_LIMITS[:, 0]) and
                np.all(joints <= JOINT_LIMITS[:, 1]))


def config_collision_free(joints_moving: np.ndarray,
                            base_moving: np.ndarray,
                            static_configs: List[Tuple[np.ndarray, np.ndarray]]) -> bool:
    """Check moving config against a list of (joints, base) static obstacles."""
    for j_s, b_s in static_configs:
        if min_dist_pair(joints_moving, base_moving, j_s, b_s) < COLLISION_THRESH:
            return False
    return True


def check_simultaneous(trajs: Dict[str, np.ndarray]) -> Tuple[bool, int, float]:
    """Check all pairs across synchronized trajectories."""
    names = list(trajs.keys())
    n_steps  = min(len(trajs[n]) for n in names)
    violations = 0
    min_d      = np.inf

    for name_a, name_b in COLLISION_PAIRS:
        if name_a not in trajs or name_b not in trajs:
            continue
        for k in range(n_steps):
            d = min_dist_pair(trajs[name_a][k], BASE_POSITIONS[name_a],
                               trajs[name_b][k], BASE_POSITIONS[name_b])
            if d < min_d:
                min_d = d
            if d < COLLISION_THRESH:
                violations += 1

    return violations == 0, violations, float(min_d)


# ── RRT-Connect ───────────────────────────────────────────────────────────────

class RRTTree:
    def __init__(self, root: np.ndarray):
        self.nodes   = [root.copy()]
        self.parents = [-1]

    def add(self, cfg: np.ndarray, parent: int) -> int:
        self.nodes.append(cfg.copy())
        self.parents.append(parent)
        return len(self.nodes) - 1

    def nearest(self, cfg: np.ndarray) -> Tuple[int, np.ndarray]:
        dists = [np.linalg.norm(n - cfg) for n in self.nodes]
        idx   = int(np.argmin(dists))
        return idx, self.nodes[idx]

    def path(self, idx: int) -> List[np.ndarray]:
        p = []
        while idx != -1:
            p.append(self.nodes[idx])
            idx = self.parents[idx]
        return list(reversed(p))


def steer(frm: np.ndarray, to: np.ndarray) -> np.ndarray:
    diff = to - frm
    d    = np.linalg.norm(diff)
    return to.copy() if d < STEP_SIZE else frm + diff / d * STEP_SIZE


def extend(tree: RRTTree, target: np.ndarray,
           base: np.ndarray,
           obstacles: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[str, int]:
    near_idx, near = tree.nearest(target)
    new_cfg        = steer(near, target)

    if not joints_valid(new_cfg):
        return 'trapped', -1
    if not config_collision_free(new_cfg, base, obstacles):
        return 'trapped', -1

    new_idx = tree.add(new_cfg, near_idx)
    status  = 'reached' if np.linalg.norm(new_cfg - target) < STEP_SIZE else 'advanced'
    return status, new_idx


def connect(tree: RRTTree, target: np.ndarray,
            base: np.ndarray,
            obstacles: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[str, int]:
    while True:
        status, idx = extend(tree, target, base, obstacles)
        if status != 'advanced':
            return status, idx


def plan_rrt(start: np.ndarray, goal: np.ndarray,
              base: np.ndarray,
              obstacles: List[Tuple[np.ndarray, np.ndarray]],
              robot_name: str = 'robot',
              seed: Optional[int] = None) -> Optional[List[np.ndarray]]:
    if seed is not None:
        np.random.seed(seed)

    tree_a = RRTTree(start)
    tree_b = RRTTree(goal)

    for it in range(MAX_ITER):
        rand = goal.copy() if np.random.random() < GOAL_BIAS else \
               np.random.uniform(JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])

        status_a, new_idx_a = extend(tree_a, rand, base, obstacles)
        if status_a == 'trapped':
            continue

        new_node_a = tree_a.nodes[new_idx_a]
        status_b, new_idx_b = connect(tree_b, new_node_a, base, obstacles)

        if status_b == 'reached':
            path_a = tree_a.path(new_idx_a)
            path_b = tree_b.path(new_idx_b)
            path_b.reverse()

            if np.linalg.norm(path_a[0] - start) > 0.01:
                path_a.reverse()
            if np.linalg.norm(path_b[-1] - goal) > 0.01:
                path_b.reverse()

            print(f"    ✓ {robot_name}: path found in {it+1} iterations")
            return path_a + path_b

        tree_a, tree_b = tree_b, tree_a

    print(f"    ✗ {robot_name}: no path in {MAX_ITER} iterations")
    return None


# ── Smoothing & interpolation ─────────────────────────────────────────────────

def smooth(path: List[np.ndarray],
           base: np.ndarray,
           obstacles: List[Tuple[np.ndarray, np.ndarray]],
           attempts: int = 200) -> List[np.ndarray]:
    if len(path) <= 2:
        return path
    smoothed = [p.copy() for p in path]
    for _ in range(attempts):
        if len(smoothed) <= 2:
            break
        i = np.random.randint(0, len(smoothed) - 2)
        j = np.random.randint(i + 1, len(smoothed))
        if j - i <= 1:
            continue
        frm, to = smoothed[i], smoothed[j]
        n_steps = max(2, int(np.linalg.norm(to - frm) / (STEP_SIZE * 0.5)))
        ok = True
        for t in np.linspace(0, 1, n_steps):
            interp = frm + t * (to - frm)
            if not joints_valid(interp) or \
               not config_collision_free(interp, base, obstacles):
                ok = False
                break
        if ok:
            smoothed = smoothed[:i+1] + smoothed[j:]
    return smoothed


def interpolate(waypoints: List[np.ndarray],
                n_samples: int = INTERP_SAMPLES,
                duration: float = DURATION) -> Tuple[np.ndarray, np.ndarray]:
    if len(waypoints) < 2:
        return np.array([waypoints[0]] * n_samples), \
               np.linspace(0, duration, n_samples)

    dists = [0.0]
    for i in range(1, len(waypoints)):
        dists.append(dists[-1] + np.linalg.norm(waypoints[i] - waypoints[i-1]))
    total   = dists[-1]
    targets = np.linspace(0, total, n_samples)
    wp_arr  = np.array(waypoints)

    joints_out = np.zeros((n_samples, 6))
    for si, td in enumerate(targets):
        seg = np.clip(np.searchsorted(dists, td, side='right') - 1,
                      0, len(waypoints) - 2)
        seg_len = dists[seg+1] - dists[seg]
        t = 0.0 if seg_len < 1e-12 else (td - dists[seg]) / seg_len
        joints_out[si] = wp_arr[seg] + t * (wp_arr[seg+1] - wp_arr[seg])

    return joints_out, np.linspace(0, duration, n_samples)


# ── Main planner ──────────────────────────────────────────────────────────────

def plan_all_arms(starts: Dict[str, np.ndarray],
                   goals:  Dict[str, np.ndarray]) -> Tuple[
                       Optional[Dict[str, np.ndarray]],
                       Optional[np.ndarray],
                       dict]:
    """
    Plan all 4 arms with simultaneous verification.
    Returns (trajs_dict, time_vec, metadata) or (None, None, meta) on failure.
    """
    print(f"\n{'='*70}")
    print(f"  RRT-CONNECT PLANNER  ({N_ARMS} arms)")
    print(f"{'='*70}")
    print(f"  Max iterations  : {MAX_ITER}")
    print(f"  Step size       : {STEP_SIZE} rad")
    print(f"  Collision thresh: {COLLISION_THRESH*100:.0f}cm")
    print(f"  Max attempts    : {MAX_ATTEMPTS}")

    t_start    = time.time()
    all_logs   = []

    for attempt in range(MAX_ATTEMPTS):
        seed = attempt * 42 + 7
        print(f"\n--- Attempt {attempt+1}/{MAX_ATTEMPTS}  (seed={seed}) ---")

        paths     = {}
        plan_fail = False

        for i, name in enumerate(ARM_NAMES):
            print(f"  [{i+1}/{N_ARMS}] Planning {name}...")
            # Static obstacles = all OTHER arms at their START positions
            obstacles = [
                (starts[other], BASE_POSITIONS[other])
                for other in ARM_NAMES if other != name
            ]
            path = plan_rrt(starts[name], goals[name],
                             BASE_POSITIONS[name],
                             obstacles,
                             robot_name=name,
                             seed=seed + i * 1000)
            if path is None:
                plan_fail = True
                break
            paths[name] = path

        if plan_fail:
            all_logs.append({'attempt': attempt+1, 'result': 'plan_failed'})
            continue

        # Smooth all paths
        print("\n  Smoothing...")
        for i, name in enumerate(ARM_NAMES):
            obstacles = [
                (starts[other], BASE_POSITIONS[other])
                for other in ARM_NAMES if other != name
            ]
            paths[name] = smooth(paths[name], BASE_POSITIONS[name], obstacles)
        sizes = {n: len(paths[n]) for n in paths}
        print(f"  Smoothed waypoints: {sizes}")

        # Interpolate
        trajs = {}
        time_vec = None
        for name in ARM_NAMES:
            traj, tvec = interpolate(paths[name], INTERP_SAMPLES, DURATION)
            trajs[name]  = traj
            if time_vec is None:
                time_vec = tvec

        # Simultaneous check
        print("\n  🔍 Verifying simultaneous execution...")
        is_safe, violations, min_d = check_simultaneous(trajs)
        print(f"  Violations: {violations}   Min dist: {min_d*100:.1f}cm")

        if is_safe:
            elapsed = round(time.time() - t_start, 3)
            print(f"\n  ✓ SIMULTANEOUS CHECK PASSED  (attempt {attempt+1})")
            print(f"  Min arm-to-arm distance: {min_d*100:.1f}cm")
            all_logs.append({
                'attempt': attempt+1, 'result': 'success',
                'violations': 0, 'min_dist_cm': round(min_d*100, 2),
            })
            meta = {
                'planning_time_s':    elapsed,
                'attempts_used':      attempt + 1,
                'simultaneous_check': 'PASSED',
                'min_distance_cm':    round(min_d * 100, 2),
                'violations':         0,
                'duration':           DURATION,
                'samples':            INTERP_SAMPLES,
                'attempt_logs':       all_logs,
            }
            return trajs, time_vec, meta

        else:
            print(f"  ⚠ Check FAILED → replanning...")
            all_logs.append({
                'attempt': attempt+1, 'result': 'simultaneous_fail',
                'violations': violations, 'min_dist_cm': round(min_d*100, 2),
            })

    # All attempts failed
    elapsed = round(time.time() - t_start, 3)
    meta = {
        'planning_time_s':    elapsed,
        'attempts_used':      MAX_ATTEMPTS,
        'simultaneous_check': 'FAILED_ALL',
        'attempt_logs':       all_logs,
    }
    return None, None, meta


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*70}")
    print("  QUAD-ARM RRT-CONNECT PLANNER")
    print(f"{'='*70}")

    try:
        with open('ik_solutions.json') as f:
            ik = json.load(f)
    except FileNotFoundError:
        print("  ✗ ik_solutions.json not found")
        print("    Run: ros2 run dual_arm_sync quad_arm_ik_solver")
        return

    starts, goals = {}, {}
    for name in ARM_NAMES:
        if name not in ik:
            print(f"  ✗ {name} missing from ik_solutions.json")
            return
        starts[name] = np.array(ik[name]['current_joints'])
        goals[name]  = np.array(ik[name]['optimal_joints'])

    print("\n  Start/Goal joints (deg):")
    for name in ARM_NAMES:
        s = np.round(np.degrees(starts[name]), 1)
        g = np.round(np.degrees(goals[name]), 1)
        print(f"    {name}: {s} → {g}")

    trajs, time_vec, meta = plan_all_arms(starts, goals)

    if trajs is None:
        print(f"\n  ✗ RRT FAILED after {MAX_ATTEMPTS} attempts")
        print("  Suggestions:")
        print("    • Choose targets further apart")
        print("    • Increase MAX_ATTEMPTS / MAX_ITER at top of file")
        with open('rrt_trajectories.json', 'w') as f:
            json.dump({'success': False, 'metadata': meta}, f, indent=2)
        return

    # Save
    output = {
        name: {
            'joint_positions': trajs[name].tolist(),
            'time':            time_vec.tolist(),
            'start':           starts[name].tolist(),
            'goal':            goals[name].tolist(),
        }
        for name in ARM_NAMES
    }
    output['success']  = True
    output['metadata'] = meta

    with open('rrt_trajectories.json', 'w') as f:
        json.dump(output, f, indent=2)

    import os
    size_kb = os.path.getsize('rrt_trajectories.json') / 1024
    print(f"\n{'='*70}")
    print(f"  ✓ RRT SUCCESS")
    print(f"{'='*70}")
    print(f"  Planning time : {meta['planning_time_s']:.2f}s")
    print(f"  Attempts used : {meta['attempts_used']}/{MAX_ATTEMPTS}")
    print(f"  Min dist      : {meta['min_distance_cm']:.1f}cm")
    print(f"  Output        : rrt_trajectories.json ({size_kb:.1f} KB)")
    print(f"\n  Next: ros2 run dual_arm_sync quad_gazebo_executor")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()