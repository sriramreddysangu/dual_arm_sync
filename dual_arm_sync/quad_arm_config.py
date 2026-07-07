#!/usr/bin/env python3
"""
quad_arm_config.py
Central configuration for 4-arm system.

Layout (2x2 grid, 1m spacing):
    dsr01  dsr02
    dsr03  dsr04

World positions (x, y, z):
  dsr01: ( 0.0,  0.5, 0.0)  white
  dsr02: ( 0.0, -0.5, 0.0)  blue
  dsr03: ( 1.0,  0.5, 0.0)  blue
  dsr04: ( 1.0, -0.5, 0.0)  white

To scale to more arms later:
  - Add entries to BASE_POSITIONS / ARM_NAMES / ARM_COLORS
  - Update the launch file (dual_gazebo.launch.py or a new one)
  - Update IK solver RobotBases
"""

import numpy as np

# ── Core count ────────────────────────────────────────────────────────────────
N_ARMS = 4

# ── Per-arm names (must match ROS namespace / topic prefix) ───────────────────
ARM_NAMES = ['dsr01', 'dsr02', 'dsr03', 'dsr04']

# ── Colors (only 'white' and 'blue' are confirmed safe in xacro) ──────────────
ARM_COLORS = ['white', 'blue', 'blue', 'white']

# ── Base positions in world frame [x, y, z] ───────────────────────────────────
BASE_POSITIONS = {
    'dsr01': np.array([ 0.0,  0.5, 0.0]),
    'dsr02': np.array([ 0.0, -0.5, 0.0]),
    'dsr03': np.array([ 1.0,  0.5, 0.0]),
    'dsr04': np.array([ 1.0, -0.5, 0.0]),
}

# ── Robot model ───────────────────────────────────────────────────────────────
MODEL = 'm1013'

# ── Safety / planning constants ───────────────────────────────────────────────
MIN_SAFE_DISTANCE   = 0.15   # 15 cm hard limit
SAFETY_MARGIN       = 0.15   # collision checker padding
LINK_RADII          = np.array([0.10, 0.10, 0.08, 0.08, 0.06, 0.06])

# ── Joint limits (Doosan M1013) ───────────────────────────────────────────────
JOINT_POSITION_LIMITS = np.array([
    [-6.283,  6.283],   # Joint 1
    [-1.650,  1.650],   # Joint 2
    [-2.792,  2.792],   # Joint 3
    [-6.283,  6.283],   # Joint 4
    [-6.283,  6.283],   # Joint 5
    [-6.283,  6.283],   # Joint 6
])

JOINT_VELOCITY_LIMITS = np.array([
    2.0944, 2.0944, 3.1416, 3.9270, 3.9270, 3.9270
])

JOINT_ACCELERATION_LIMITS = np.array([3.0, 3.0, 4.0, 5.0, 5.0, 6.0])

# ── Workspace bounds (per-arm, in LOCAL frame) ────────────────────────────────
WORKSPACE = {
    'x': (-0.8,  0.8),
    'y': (-1.0,  1.0),
    'z': ( 0.0,  1.5),
    'radius_min': 0.15,
    'radius_max': 1.30,
}

# ── Collision pairs (all unique pairs of arms) ────────────────────────────────
# Pre-computed: N*(N-1)/2 = 6 pairs for N=4
COLLISION_PAIRS = [
    ('dsr01', 'dsr02'),
    ('dsr01', 'dsr03'),
    ('dsr01', 'dsr04'),
    ('dsr02', 'dsr03'),
    ('dsr02', 'dsr04'),
    ('dsr03', 'dsr04'),
]


def get_base(name: str) -> np.ndarray:
    """Return world-frame base position for named arm."""
    return BASE_POSITIONS[name].copy()


def arm_index(name: str) -> int:
    """Return 0-based index of arm in ARM_NAMES list."""
    return ARM_NAMES.index(name)


def print_config():
    print(f"\n{'='*60}")
    print(f"  QUAD-ARM CONFIG  (N={N_ARMS})")
    print(f"{'='*60}")
    for name in ARM_NAMES:
        b = BASE_POSITIONS[name]
        print(f"  {name}  pos=({b[0]:+.1f},{b[1]:+.1f},{b[2]:.1f})"
              f"  color={ARM_COLORS[arm_index(name)]}")
    print(f"\n  Collision pairs: {len(COLLISION_PAIRS)}")
    for p in COLLISION_PAIRS:
        print(f"    {p[0]} <-> {p[1]}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    print_config()