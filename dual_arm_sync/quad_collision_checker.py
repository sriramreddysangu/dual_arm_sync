#!/usr/bin/env python3
"""
quad_collision_checker.py
Collision detection for 4-arm system.

Checks all N*(N-1)/2 = 6 pairs during SIMULTANEOUS motion.

Reads:  trajectories.json
Writes: collision_report.json

Usage:
    ros2 run dual_arm_sync quad_collision_checker
"""

import numpy as np
from typing import Dict, List, Tuple
import json
import os
import sys

try:
    from dual_arm_sync.constants import DHParameters
    try:
        from dual_arm_sync.quad_arm_config import (
            ARM_NAMES, BASE_POSITIONS, COLLISION_PAIRS,
            MIN_SAFE_DISTANCE, SAFETY_MARGIN, LINK_RADII, N_ARMS,
        )
    except ImportError:
        from quad_arm_config import (
            ARM_NAMES, BASE_POSITIONS, COLLISION_PAIRS,
            MIN_SAFE_DISTANCE, SAFETY_MARGIN, LINK_RADII, N_ARMS,
        )
except ImportError as e:
    print(f"ERROR: Cannot import required modules - {e}")
    sys.exit(1)


# ── FK helper ─────────────────────────────────────────────────────────────────

def _link_positions(joints: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Return (6,3) array of link world positions."""
    dh = DHParameters.get_dh_params(joints)
    T  = np.eye(4)
    pts = []
    for i in range(6):
        alpha, a, theta, d = dh[i]
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        T = T @ np.array([
            [ct, -st,  0,  a],
            [st*ca,  ct*ca, -sa, -sa*d],
            [st*sa,  ct*sa,  ca,  ca*d],
            [0, 0, 0, 1],
        ])
        pts.append(T[:3, 3] + base)
    return np.array(pts)


def _min_distance(joints_a: np.ndarray, base_a: np.ndarray,
                   joints_b: np.ndarray, base_b: np.ndarray) -> Tuple[float, int, int]:
    """Minimum capsule distance between two arms. Returns (dist, link_i, link_j)."""
    links_a = _link_positions(joints_a, base_a)
    links_b = _link_positions(joints_b, base_b)
    min_d, mi, mj = np.inf, 0, 0
    for i in range(6):
        for j in range(6):
            # clearance = euclidean dist - sum of radii - safety margin
            clearance = (np.linalg.norm(links_a[i] - links_b[j])
                         - LINK_RADII[i] - LINK_RADII[j] - SAFETY_MARGIN)
            if clearance < min_d:
                min_d, mi, mj = clearance, i, j
    return min_d, mi, mj


# ── Per-pair simultaneous check ───────────────────────────────────────────────

def check_pair_simultaneous(name_a: str, traj_a: np.ndarray,
                              name_b: str, traj_b: np.ndarray,
                              time_vec: np.ndarray) -> Dict:
    """
    Check one arm-pair during simultaneous execution.
    Both trajectory arrays must be same length (same time grid).
    """
    base_a = BASE_POSITIONS[name_a]
    base_b = BASE_POSITIONS[name_b]

    n = min(len(traj_a), len(traj_b))
    min_distances = []
    violations    = []

    for k in range(n):
        d, li, lj = _min_distance(traj_a[k], base_a, traj_b[k], base_b)
        min_distances.append(d)
        if d < 0:   # negative clearance = collision
            violations.append({
                'time':   float(time_vec[k]),
                'clearance_m': float(d),
                'link_a': li,
                'link_b': lj,
            })

    overall_min = float(min(min_distances))
    has_collision = len(violations) > 0

    return {
        'pair':            f"{name_a}↔{name_b}",
        'arm_a':           name_a,
        'arm_b':           name_b,
        'has_collision':   has_collision,
        'num_violations':  len(violations),
        'min_clearance_m': overall_min,
        'min_clearance_cm': round(overall_min * 100, 2),
        'violations':      violations[:10],   # cap to 10 for JSON size
    }


# ── Main collision check ──────────────────────────────────────────────────────

def check_all_pairs(trajectories: Dict) -> Dict:
    """
    Check all 6 arm-pairs. Returns full collision report.
    All trajectories are re-sampled to the same length (min common length).
    """
    print(f"\n{'='*70}")
    print(f"  QUAD-ARM COLLISION CHECK  ({len(COLLISION_PAIRS)} pairs)")
    print(f"{'='*70}")

    # Extract position arrays and time vectors
    pos = {}
    tvecs = {}
    for name in ARM_NAMES:
        if name not in trajectories:
            continue
        t = trajectories[name]['trajectory']
        pos[name]   = np.array(t['positions'])
        tvecs[name] = np.array(t['time'])

    # Common length
    common_len = min(len(pos[n]) for n in pos)
    # Resample all to common length if needed
    resampled = {}
    time_vec  = None
    for name in pos:
        if len(pos[name]) != common_len:
            idx = np.linspace(0, len(pos[name]) - 1, common_len).astype(int)
            resampled[name] = pos[name][idx]
        else:
            resampled[name] = pos[name]
        if time_vec is None:
            time_vec = tvecs[name][:common_len]

    # Check each pair
    pair_results = []
    overall_collision = False

    for name_a, name_b in COLLISION_PAIRS:
        if name_a not in resampled or name_b not in resampled:
            print(f"  ⚠ Skipping {name_a}↔{name_b}: trajectory missing")
            continue

        result = check_pair_simultaneous(
            name_a, resampled[name_a],
            name_b, resampled[name_b],
            time_vec,
        )
        pair_results.append(result)

        status = "✗ COLLISION" if result['has_collision'] else "✓ clear"
        print(f"  {name_a}↔{name_b}: {status}"
              f"  min_clearance={result['min_clearance_cm']:.1f}cm"
              f"  violations={result['num_violations']}")

        if result['has_collision']:
            overall_collision = True

    print(f"\n{'='*70}")
    if overall_collision:
        n_col = sum(1 for r in pair_results if r['has_collision'])
        print(f"  ✗ COLLISION DETECTED in {n_col}/{len(pair_results)} pairs")
    else:
        min_cl = min(r['min_clearance_cm'] for r in pair_results) if pair_results else 0
        print(f"  ✓ ALL CLEAR  — min clearance across all pairs: {min_cl:.1f}cm")
    print(f"{'='*70}\n")

    return {
        'collision_detected': overall_collision,
        'n_pairs_checked':    len(pair_results),
        'pair_results':       pair_results,
        'trajectories':       trajectories,
        'summary': {
            'total_pairs':     len(COLLISION_PAIRS),
            'pairs_with_collision': sum(1 for r in pair_results if r['has_collision']),
            'min_clearance_cm': min(
                (r['min_clearance_cm'] for r in pair_results), default=0
            ),
        },
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*70}")
    print("  QUAD-ARM COLLISION CHECKER")
    print(f"{'='*70}")
    print("  Input : trajectories.json")
    print("  Output: collision_report.json")

    try:
        with open('trajectories.json') as f:
            traj_data = json.load(f)
    except FileNotFoundError:
        print("  ✗ trajectories.json not found")
        print("    Run: ros2 run dual_arm_sync quad_trajectory_generation")
        return

    report = check_all_pairs(traj_data)

    with open('collision_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)

    size_kb = os.path.getsize('collision_report.json') / 1024
    print(f"  ✓ Saved: collision_report.json ({size_kb:.1f} KB)")

    if report['collision_detected']:
        print("\n  ⚠ Collision detected. Run Kuramoto or RRT to resolve:")
        print("    ros2 run dual_arm_sync quad_kuramoto_synchronization")
    else:
        print("\n  ✓ Safe. Proceed to executor:")
        print("    ros2 run dual_arm_sync quad_gazebo_executor")

    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()