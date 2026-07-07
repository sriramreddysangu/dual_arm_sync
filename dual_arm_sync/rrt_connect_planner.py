#!/usr/bin/env python3
"""
rrt_connect_planner.py
RRT-Connect Planner for Dual-Arm Collision Avoidance

FIXED:
  - Post-planning SIMULTANEOUS collision verification (both robots moving together)
  - Re-plans if simultaneous check fails (up to MAX_REPLAN_ATTEMPTS)
  - Reports actual verified safety, not just per-robot construction guarantee
  - Improved smoothing that preserves collision-avoidance margins

Input:  ik_solutions.json  (or warm start from synchronized_trajectories.json)
Output: rrt_trajectories.json
"""

import numpy as np
import json
import time
from typing import List, Tuple, Optional

# ============================================================================
# DH / FK (same as IK solver — must match exactly)
# ============================================================================

DH_PARAMS = [
    (0.0,    0.1555,  np.pi/2,  0.0),
    (0.409,  0.0,     0.0,      0.0),
    (0.0,    0.0,    -np.pi/2,  np.pi/2),
    (0.0,    0.3995,  np.pi/2,  0.0),
    (0.0,    0.0,    -np.pi/2,  0.0),
    (0.0,    0.082,   0.0,      0.0),
]

JOINT_LIMITS = np.array([
    (-np.pi,          np.pi),
    (-np.pi/2,        np.pi/2),
    (-np.pi*150/180,  np.pi*150/180),
    (-np.pi,          np.pi),
    (-np.pi/2,        np.pi/2),
    (-np.pi,          np.pi),
])

# Robot bases in world frame
DSR01_BASE = np.array([0.0,  0.5, 0.0])
DSR02_BASE = np.array([0.0, -0.5, 0.0])

# Safety parameters
COLLISION_THRESHOLD = 0.15   # 15 cm
STEP_SIZE          = 0.15    # rad
MAX_ITER           = 5000
GOAL_BIAS          = 0.15
INTERP_SAMPLES     = 1000
DURATION           = 10.0

MAX_REPLAN_ATTEMPTS = 5       # how many times to retry if simultaneous check fails


# ============================================================================
# FORWARD KINEMATICS
# ============================================================================

def dh_matrix(a, d, alpha, theta):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,      sa,     ca,    d],
        [0,       0,      0,    1],
    ])


def forward_kinematics(joints: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    for i, (a, d, alpha, to) in enumerate(DH_PARAMS):
        T = T @ dh_matrix(a, d, alpha, joints[i] + to)
    return T


def get_link_positions(joints: np.ndarray, base: np.ndarray) -> List[np.ndarray]:
    """Return world-frame positions of each link (0=base, 1-6=links, 7=EE)"""
    T = np.eye(4)
    positions = [base.copy()]
    for i, (a, d, alpha, to) in enumerate(DH_PARAMS):
        T = T @ dh_matrix(a, d, alpha, joints[i] + to)
        p = T[:3, 3] + base
        positions.append(p)
    return positions


def min_distance_between_arms(joints1: np.ndarray, joints2: np.ndarray) -> Tuple[float, str, str]:
    """
    Compute minimum distance between any pair of links from the two robots.
    Returns (min_dist_m, link1_name, link2_name)
    """
    links1 = get_link_positions(joints1, DSR01_BASE)
    links2 = get_link_positions(joints2, DSR02_BASE)

    min_dist = np.inf
    min_pair = ('', '')

    link_names = ['Base', 'Link1', 'Link2', 'Link3', 'Link4', 'Link5', 'EE']

    for i in range(1, len(links1)):
        for j in range(1, len(links2)):
            dist = np.linalg.norm(links1[i] - links2[j])
            if dist < min_dist:
                min_dist = dist
                min_pair = (f'dsr01 {link_names[min(i, len(link_names)-1)]}',
                            f'dsr02 {link_names[min(j, len(link_names)-1)]}')

    return min_dist, min_pair[0], min_pair[1]


# ============================================================================
# COLLISION CHECKING
# ============================================================================

def is_joint_config_valid(joints: np.ndarray) -> bool:
    """Check joint limits"""
    return np.all(joints >= JOINT_LIMITS[:, 0]) and np.all(joints <= JOINT_LIMITS[:, 1])


def is_collision_free_single(joints_moving: np.ndarray,
                              joints_static: np.ndarray,
                              base_moving: np.ndarray,
                              base_static: np.ndarray) -> bool:
    """
    Check if moving robot at joints_moving collides with static robot.
    """
    links_m = get_link_positions(joints_moving, base_moving)
    links_s = get_link_positions(joints_static, base_static)
    for p1 in links_m[1:]:
        for p2 in links_s[1:]:
            if np.linalg.norm(p1 - p2) < COLLISION_THRESHOLD:
                return False
    return True


def check_path_simultaneous(path1: List[np.ndarray],
                              path2: List[np.ndarray]) -> Tuple[bool, int, float]:
    """
    Verify two synchronized paths are collision-free when executed simultaneously.

    Both paths must be same length (synchronized in time).

    Returns:
        (is_safe, num_violations, min_distance_m)
    """
    assert len(path1) == len(path2), "Paths must be same length for simultaneous check"

    violations = 0
    min_dist = np.inf

    for j1, j2 in zip(path1, path2):
        dist, _, _ = min_distance_between_arms(j1, j2)
        if dist < min_dist:
            min_dist = dist
        if dist < COLLISION_THRESHOLD:
            violations += 1

    is_safe = (violations == 0)
    return is_safe, violations, min_dist


# ============================================================================
# RRT-CONNECT PLANNER (single robot)
# ============================================================================

class RRTTree:
    def __init__(self, root: np.ndarray):
        self.nodes = [root.copy()]
        self.parents = [-1]

    def add(self, config: np.ndarray, parent_idx: int):
        self.nodes.append(config.copy())
        self.parents.append(parent_idx)
        return len(self.nodes) - 1

    def nearest(self, config: np.ndarray) -> Tuple[int, np.ndarray]:
        dists = [np.linalg.norm(n - config) for n in self.nodes]
        idx = int(np.argmin(dists))
        return idx, self.nodes[idx]

    def path_to_root(self, idx: int) -> List[np.ndarray]:
        path = []
        while idx != -1:
            path.append(self.nodes[idx])
            idx = self.parents[idx]
        return list(reversed(path))


def steer(from_config: np.ndarray, to_config: np.ndarray) -> np.ndarray:
    """Move from_config toward to_config by STEP_SIZE"""
    diff = to_config - from_config
    dist = np.linalg.norm(diff)
    if dist < STEP_SIZE:
        return to_config.copy()
    return from_config + (diff / dist) * STEP_SIZE


def extend(tree: RRTTree, target: np.ndarray,
           other_joints: np.ndarray, other_base: np.ndarray,
           my_base: np.ndarray) -> Tuple[str, int]:
    """
    Try to extend tree toward target.
    Returns ('reached'|'advanced'|'trapped', new_node_idx)
    """
    near_idx, near = tree.nearest(target)
    new_config = steer(near, target)

    if not is_joint_config_valid(new_config):
        return 'trapped', -1
    if not is_collision_free_single(new_config, other_joints, my_base, other_base):
        return 'trapped', -1

    new_idx = tree.add(new_config, near_idx)

    if np.linalg.norm(new_config - target) < STEP_SIZE:
        return 'reached', new_idx
    return 'advanced', new_idx


def connect(tree: RRTTree, target: np.ndarray,
            other_joints: np.ndarray, other_base: np.ndarray,
            my_base: np.ndarray) -> Tuple[str, int]:
    """Keep extending until target is reached or trapped"""
    while True:
        status, idx = extend(tree, target, other_joints, other_base, my_base)
        if status != 'advanced':
            return status, idx


def plan_rrt_connect(start: np.ndarray, goal: np.ndarray,
                     other_joints_static: np.ndarray,
                     my_base: np.ndarray, other_base: np.ndarray,
                     robot_name: str = 'Robot',
                     seed: Optional[int] = None) -> Optional[List[np.ndarray]]:
    """
    Bidirectional RRT-Connect from start to goal.

    other_joints_static: snapshot of the other robot's joints
                         (used for per-step collision avoidance against static pose)

    Returns list of joint configs or None if failed.
    """
    if seed is not None:
        np.random.seed(seed)

    tree_a = RRTTree(start)
    tree_b = RRTTree(goal)

    for iteration in range(MAX_ITER):
        # Random sample with goal bias
        if np.random.random() < GOAL_BIAS:
            rand_config = goal.copy()
        else:
            rand_config = np.random.uniform(JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])

        # Extend tree_a
        status_a, new_idx_a = extend(tree_a, rand_config,
                                     other_joints_static, other_base, my_base)
        if status_a == 'trapped':
            continue

        # Connect tree_b toward new node in tree_a
        new_node_a = tree_a.nodes[new_idx_a]
        status_b, new_idx_b = connect(tree_b, new_node_a,
                                      other_joints_static, other_base, my_base)

        if status_b == 'reached':
            # Build path: a_root → new_node_a then reverse(b_root → new_node_b)
            path_a = tree_a.path_to_root(new_idx_a)
            path_b = tree_b.path_to_root(new_idx_b)
            path_b.reverse()

            # Ensure start/goal are correct
            if np.linalg.norm(path_a[0] - start) > 0.01:
                path_a.reverse()
            if np.linalg.norm(path_b[-1] - goal) > 0.01:
                path_b.reverse()

            full_path = path_a + path_b

            if iteration % 100 == 0 or iteration < 50:
                print(f'  Path found in {iteration+1} iterations')
            else:
                print(f'  Path found in {iteration+1} iterations')
            return full_path

        # Swap trees every iteration (bidirectional)
        tree_a, tree_b = tree_b, tree_a

    print(f'  ❌ No path found in {MAX_ITER} iterations for {robot_name}')
    return None


# ============================================================================
# PATH SMOOTHING
# ============================================================================

def smooth_path(path: List[np.ndarray],
                other_joints_static: np.ndarray,
                my_base: np.ndarray, other_base: np.ndarray,
                max_attempts: int = 200) -> List[np.ndarray]:
    """
    Shortcut smoothing: try to directly connect random pairs of waypoints.
    """
    if len(path) <= 2:
        return path

    smoothed = [p.copy() for p in path]

    for _ in range(max_attempts):
        if len(smoothed) <= 2:
            break

        i = np.random.randint(0, len(smoothed) - 2)
        j = np.random.randint(i + 1, len(smoothed))

        if j - i <= 1:
            continue

        # Check if direct path from i to j is collision-free
        from_c = smoothed[i]
        to_c   = smoothed[j]
        dist   = np.linalg.norm(to_c - from_c)
        steps  = max(2, int(dist / (STEP_SIZE * 0.5)))

        ok = True
        for t in np.linspace(0, 1, steps):
            interp = from_c + t * (to_c - from_c)
            if not is_joint_config_valid(interp):
                ok = False
                break
            if not is_collision_free_single(interp, other_joints_static, my_base, other_base):
                ok = False
                break

        if ok:
            smoothed = smoothed[:i+1] + smoothed[j:]

    return smoothed


# ============================================================================
# PATH INTERPOLATION
# ============================================================================

def interpolate_path(waypoints: List[np.ndarray],
                     n_samples: int = INTERP_SAMPLES,
                     duration: float = DURATION) -> Tuple[np.ndarray, np.ndarray]:
    """
    Linearly interpolate a waypoint list to n_samples uniformly in time.
    Returns (joint_positions [n, 6], time_vector [n])
    """
    if len(waypoints) < 2:
        repeated = np.array([waypoints[0]] * n_samples)
        return repeated, np.linspace(0, duration, n_samples)

    # Compute cumulative arc-length along path
    dists = [0.0]
    for i in range(1, len(waypoints)):
        dists.append(dists[-1] + np.linalg.norm(waypoints[i] - waypoints[i-1]))

    total = dists[-1]
    target_dists = np.linspace(0, total, n_samples)

    joints_out = np.zeros((n_samples, 6))
    wp_arr = np.array(waypoints)

    for s_idx, td in enumerate(target_dists):
        # Find segment
        seg = np.searchsorted(dists, td, side='right') - 1
        seg = np.clip(seg, 0, len(waypoints) - 2)

        seg_len = dists[seg+1] - dists[seg]
        if seg_len < 1e-12:
            t = 0.0
        else:
            t = (td - dists[seg]) / seg_len

        joints_out[s_idx] = wp_arr[seg] + t * (wp_arr[seg+1] - wp_arr[seg])

    time_vec = np.linspace(0, duration, n_samples)
    return joints_out, time_vec


# ============================================================================
# SIMULTANEOUS COLLISION-AWARE REPLANNING
# ============================================================================

def plan_with_simultaneous_verification(
    start1: np.ndarray, goal1: np.ndarray,
    start2: np.ndarray, goal2: np.ndarray,
    max_attempts: int = MAX_REPLAN_ATTEMPTS
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], dict]:
    """
    Plan both robots and verify simultaneous collision-freeness.
    Re-plans with different random seeds if the simultaneous check fails.

    Returns:
        (traj1, traj2, time_vector, metadata)
        traj1/traj2 are None if all attempts failed.
    """

    print(f'\n{"=" * 80}')
    print('RRT-CONNECT PLANNING WITH SIMULTANEOUS VERIFICATION')
    print('=' * 80)
    print(f'Max planning attempts: {max_attempts}')
    print(f'Max iterations per attempt: {MAX_ITER}')
    print(f'Step size: {STEP_SIZE} rad')
    print(f'Collision threshold: {COLLISION_THRESHOLD*100:.0f}cm')
    print(f'Simultaneous check: ✓ ENABLED')
    print()

    t_start = time.time()
    attempt_logs = []

    for attempt in range(max_attempts):
        seed = attempt * 42 + 7  # deterministic but varied seeds

        print(f'--- Attempt {attempt+1}/{max_attempts} (seed={seed}) ---')

        # ---- Plan DSR01 (using DSR02 goal as static obstacle — conservative) ----
        print(f'[1/2] Planning DSR01...')
        path1 = plan_rrt_connect(
            start1, goal1,
            other_joints_static=start2,   # treat DSR02 as at its start
            my_base=DSR01_BASE, other_base=DSR02_BASE,
            robot_name='DSR01', seed=seed
        )

        if path1 is None:
            print(f'  ❌ DSR01 planning failed, trying next attempt...\n')
            attempt_logs.append({'attempt': attempt+1, 'result': 'dsr01_failed'})
            continue

        print(f'✓ DSR01 raw path: {len(path1)} waypoints')

        # ---- Plan DSR02 (using DSR01 goal as static obstacle — conservative) ----
        print(f'[2/2] Planning DSR02...')
        path2 = plan_rrt_connect(
            start2, goal2,
            other_joints_static=start1,   # treat DSR01 as at its start
            my_base=DSR02_BASE, other_base=DSR01_BASE,
            robot_name='DSR02', seed=seed + 1000
        )

        if path2 is None:
            print(f'  ❌ DSR02 planning failed, trying next attempt...\n')
            attempt_logs.append({'attempt': attempt+1, 'result': 'dsr02_failed'})
            continue

        print(f'✓ DSR02 raw path: {len(path2)} waypoints')

        # ---- Smooth ----
        print('\nSmoothing paths...')
        path1 = smooth_path(path1, start2, DSR01_BASE, DSR02_BASE)
        path2 = smooth_path(path2, start1, DSR02_BASE, DSR01_BASE)
        print(f'✓ Smoothed: DSR01={len(path1)}, DSR02={len(path2)} waypoints')

        # ---- Interpolate ----
        print('\nInterpolating to uniform time grid...')
        traj1, time1 = interpolate_path(path1, INTERP_SAMPLES, DURATION)
        traj2, time2 = interpolate_path(path2, INTERP_SAMPLES, DURATION)

        # ---- SIMULTANEOUS COLLISION CHECK ----
        print('\n🔍 Verifying simultaneous execution...')
        is_safe, violations, min_dist = check_path_simultaneous(traj1, traj2)

        print(f'   Violations:   {violations}')
        print(f'   Min distance: {min_dist*100:.1f}cm (threshold: {COLLISION_THRESHOLD*100:.0f}cm)')

        if is_safe:
            planning_time = time.time() - t_start
            print(f'\n✓ SIMULTANEOUS VERIFICATION PASSED (attempt {attempt+1})')
            print(f'  Min arm-to-arm distance: {min_dist*100:.1f}cm')

            metadata = {
                'planning_time': round(planning_time, 3),
                'attempts_used': attempt + 1,
                'warm_start_used': False,
                'dsr01_waypoints_raw': len(path1),
                'dsr02_waypoints_raw': len(path2),
                'simultaneous_check': 'PASSED',
                'min_distance_cm': round(min_dist * 100, 2),
                'collision_violations': 0,
                'duration': DURATION,
                'samples': INTERP_SAMPLES,
            }

            attempt_logs.append({
                'attempt': attempt+1, 'result': 'success',
                'violations': 0, 'min_dist_cm': round(min_dist*100, 2)
            })
            metadata['attempt_logs'] = attempt_logs

            return traj1, traj2, time1, metadata

        else:
            print(f'  ⚠  SIMULTANEOUS CHECK FAILED: {violations} violations, '
                  f'min dist {min_dist*100:.1f}cm → replanning...\n')
            attempt_logs.append({
                'attempt': attempt+1, 'result': 'simultaneous_fail',
                'violations': violations, 'min_dist_cm': round(min_dist*100, 2)
            })

    # All attempts exhausted
    planning_time = time.time() - t_start
    print(f'\n❌ All {max_attempts} attempts failed simultaneous verification')
    metadata = {
        'planning_time': round(planning_time, 3),
        'attempts_used': max_attempts,
        'simultaneous_check': 'FAILED_ALL',
        'attempt_logs': attempt_logs,
    }
    return None, None, None, metadata


# ============================================================================
# MAIN
# ============================================================================

def main():
    print('\n' + '=' * 80)
    print('RRT-CONNECT PLANNER')
    print('=' * 80)

    # ---- Load IK solutions ----
    try:
        with open('ik_solutions.json', 'r') as f:
            ik_data = json.load(f)
    except FileNotFoundError:
        print('❌ ik_solutions.json not found')
        print('   Run: ros2 run dual_arm_sync dual_arm_ik_solver')
        return

    start1 = np.array(ik_data['dsr01']['current_joints'])
    goal1  = np.array(ik_data['dsr01']['optimal_joints'])
    start2 = np.array(ik_data['dsr02']['current_joints'])
    goal2  = np.array(ik_data['dsr02']['optimal_joints'])

    print(f'\n✓ Loaded start and goal from ik_solutions.json')
    print(f'  DSR01: {np.degrees(start1).round(1)} → {np.degrees(goal1).round(1)} deg')
    print(f'  DSR02: {np.degrees(start2).round(1)} → {np.degrees(goal2).round(1)} deg')

    # ---- Try warm start from Kuramoto ----
    warm_start_used = False
    try:
        with open('synchronized_trajectories.json', 'r') as f:
            sync_data = json.load(f)
        # Use midpoint of Kuramoto trajectory as intermediate waypoints (future enhancement)
        print('\n  [Warm start] synchronized_trajectories.json found (not used for now)')
    except Exception:
        pass

    # ---- Plan with simultaneous verification ----
    traj1, traj2, time_vec, metadata = plan_with_simultaneous_verification(
        start1, goal1, start2, goal2
    )

    # ---- Save ----
    if traj1 is None:
        print('\n❌ RRT-Connect planning FAILED after all attempts')
        print('   Suggestions:')
        print('   • Choose different target positions (further apart)')
        print('   • Increase MAX_REPLAN_ATTEMPTS in script')
        print('   • Increase MAX_ITER for more thorough search')

        # Save failure report
        failure_report = {
            'success': False,
            'metadata': metadata,
        }
        with open('rrt_trajectories.json', 'w') as f:
            json.dump(failure_report, f, indent=2)
        print('   Failure report saved to rrt_trajectories.json')
        return

    output = {
        'dsr01': {
            'joint_positions': traj1.tolist(),
            'time': time_vec.tolist(),
            'start': start1.tolist(),
            'goal':  goal1.tolist(),
        },
        'dsr02': {
            'joint_positions': traj2.tolist(),
            'time': time_vec.tolist(),
            'start': start2.tolist(),
            'goal':  goal2.tolist(),
        },
        'success': True,
        'metadata': metadata,
    }

    with open('rrt_trajectories.json', 'w') as f:
        json.dump(output, f, indent=2)

    # ---- Summary ----
    print(f'\n{"=" * 80}')
    print('✓ RRT SUCCESS')
    print('=' * 80)
    print(f'Planning time    : {metadata["planning_time"]:.2f}s')
    print(f'Attempts used    : {metadata["attempts_used"]}/{MAX_REPLAN_ATTEMPTS}')
    print(f'Simultaneous ✓   : {metadata["simultaneous_check"]}')
    print(f'Min arm distance : {metadata["min_distance_cm"]:.1f}cm')
    print(f'Warm start       : {warm_start_used}')
    print(f'Output           : rrt_trajectories.json')
    print(f'Duration         : {DURATION}s')
    print(f'Samples          : {INTERP_SAMPLES}')
    print('=' * 80)
    print('\nNext step: ros2 run dual_arm_sync gazebo_executor')


if __name__ == '__main__':
    main()