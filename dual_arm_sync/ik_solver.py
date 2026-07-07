#!/usr/bin/env python3
"""
ik_solver.py
============
Foundation module — shared by every stage of the pipeline.

  Doosan M1013  (6-DOF serial manipulator)
  DH parameters, joint limits, forward kinematics
  Numerical IK  (L-BFGS-B, multiple restarts)
    ↳ full 6-DOF: position + orientation in cost
  Orientation utilities: RPY ↔ rotation matrix, SLERP, path interpolation
  Sphere collision model  (12 spheres / arm)
  Yoshikawa manipulability index
  Joint-limit margin
  Lexicographic IK solution selection  (paper eq. 7)
  Global ArmRegistry  (N-arm workspace state)

Pipeline:
  ik_solver → dual_arm_ik_solver → trajectory_generation
                                 → collision_checker
                                 → kuramoto_sync
"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple


# ============================================================================
# ROBOT GEOMETRY  —  Doosan M1013
# ============================================================================

L1 = 0.1525    # base  → shoulder   (m)
L2 = 0.620     # upper arm
L3 = 0.559     # forearm
L4 = 0.121     # wrist → EE
A  = 0.0345    # lateral shoulder offset

# ============================================================================
# DH PARAMETERS  —  Doosan M1013  (Standard DH convention)
#
# Convention: Standard (Denavit–Hartenberg), NOT Modified DH.
#
#   T_{i-1,i} = Rot(z_{i-1}, θ_i) · Trans(z_{i-1}, d_i)
#              · Trans(x_i, a_i)  · Rot(x_i, α_i)
#
#   Table columns: [a,  alpha,  d,  theta_offset]
#
#   Verified against Gazebo/RViz:
#     FK(home=[0,0,0,0,0,0]) → local EE = [0.6545, 0, −0.5275]
#     World EE dsr01 = base + local = [0.6545, 0.5, −0.5275]  ← matches terminal
#
# NOTE: constants.py uses Modified DH with [alpha, a, theta_offset, d].
# The physical link lengths are identical; only the matrix convention differs.
# This file uses Standard DH to match the Gazebo URDF joint reference frames.
# ============================================================================

DH_PARAMS = np.array([
    # [ a,         alpha,       d,   theta_offset ]
    [0,    np.pi/2,  L1,  0],   # Joint 1: base rotation
    [L2,   0,        0,   0],   # Joint 2: shoulder pitch
    [A,    np.pi/2,  0,   0],   # Joint 3: elbow pitch
    [0,   -np.pi/2,  L3,  0],   # Joint 4: wrist 1 roll
    [0,    np.pi/2,  0,   0],   # Joint 5: wrist 2 pitch
    [0,    0,        L4,  0],   # Joint 6: wrist 3 roll
], dtype=float)

# Joint limits  [lower, upper]  radians  (from constants.py JointLimits)
JOINT_LIMITS = np.array([
    [-2*np.pi,  2*np.pi],   # J1  ±360°
    [-1.6493,   1.6493 ],   # J2  ±94.5°  (LIMITED)
    [-2.7925,   2.7925 ],   # J3  ±160°   (LIMITED)
    [-2*np.pi,  2*np.pi],   # J4  ±360°
    [-2*np.pi,  2*np.pi],   # J5  ±360°
    [-2*np.pi,  2*np.pi],   # J6  ±360°
], dtype=float)

JOINT_VEL_MAX = np.array([2.09, 2.09, 3.14, 3.93, 3.93, 3.93])  # rad/s

# IK tolerances
IK_POS_TOL    = 0.002   # 2 mm position
IK_ORI_WEIGHT = 0.3     # orientation weight in IK cost (lower → favour position)

# Collision: penetration ≤ this → collision-free
COLLISION_TOL = 0.002

# Lexicographic tie band: within 5% of best range → "tied"
LEX_TIE_FRAC = 0.05


# ============================================================================
# ROBOT BASE POSITIONS
# ============================================================================

class RobotBases:
    """World-frame base positions for the 2-arm cell."""
    DSR01_BASE = np.array([0.0,  0.5, 0.0])
    DSR02_BASE = np.array([0.0, -0.5, 0.0])


# ============================================================================
# ARM REGISTRY  (global N-arm workspace state)
# ============================================================================

class ArmRegistry:
    """
    Tracks base position + live joint state for every arm in the cell.

    Other arms ARE the only collision hazards (no static obstacles).
    When arm-i selects an IK solution, it queries every arm j≠i through
    this registry for their current sphere positions.

    Scales to 10 arms without any architectural changes.
    """

    def __init__(self):
        self._arms: Dict[str, Dict] = {}

    def register(self, arm_id: str, base: np.ndarray, joints: np.ndarray):
        self._arms[arm_id] = {
            'base':   np.asarray(base,   dtype=float),
            'joints': np.asarray(joints, dtype=float),
        }

    def update_joints(self, arm_id: str, joints: np.ndarray):
        """Called on every joint-state callback to keep registry live."""
        if arm_id in self._arms:
            self._arms[arm_id]['joints'] = np.asarray(joints, dtype=float)

    def get_other_arms(self, arm_id: str) -> List[Dict]:
        """Return [{base, joints}, ...] for every arm except arm_id."""
        return [v for k, v in self._arms.items() if k != arm_id]

    def __len__(self):
        return len(self._arms)

    def __contains__(self, arm_id):
        return arm_id in self._arms


ARM_REGISTRY = ArmRegistry()


# ============================================================================
# ORIENTATION UTILITIES
# ============================================================================

def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Euler RPY (intrinsic Z-Y-X) → rotation matrix (3×3).

    Convention: yaw(Z) → pitch(Y) → roll(X)  (robotics standard)
    Input in radians.
    """
    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)
    return np.array([
        [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [-sp,    cp*sr,              cp*cr            ],
    ])


def matrix_to_rpy(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Rotation matrix → RPY (roll, pitch, yaw) in radians.
    Returns (roll, pitch, yaw).
    """
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
    if abs(np.cos(pitch)) < 1e-8:
        roll = 0.0
        yaw  = np.arctan2(-R[1, 2], R[1, 1]) if R[2, 0] < 0 else \
               np.arctan2( R[1, 2], R[1, 1])
    else:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw  = np.arctan2(R[1, 0], R[0, 0])
    return float(roll), float(pitch), float(yaw)


def rotation_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → unit quaternion [w, x, y, z]."""
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2,1] - R[1,2]) * s
        y = (R[0,2] - R[2,0]) * s
        z = (R[1,0] - R[0,1]) * s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        w = (R[2,1] - R[1,2]) / s
        x = 0.25 * s
        y = (R[0,1] + R[1,0]) / s
        z = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        w = (R[0,2] - R[2,0]) / s
        x = (R[0,1] + R[1,0]) / s
        y = 0.25 * s
        z = (R[1,2] + R[2,1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        w = (R[1,0] - R[0,1]) / s
        x = (R[0,2] + R[2,0]) / s
        y = (R[1,2] + R[2,1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / (np.linalg.norm(q) + 1e-12)


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """Unit quaternion [w, x, y, z] → rotation matrix (3×3)."""
    w, x, y, z = q / (np.linalg.norm(q) + 1e-12)
    return np.array([
        [1-2*(y*y+z*z),  2*(x*y-z*w),    2*(x*z+y*w)  ],
        [2*(x*y+z*w),    1-2*(x*x+z*z),  2*(y*z-x*w)  ],
        [2*(x*z-y*w),    2*(y*z+x*w),    1-2*(x*x+y*y)],
    ])


def slerp(R0: np.ndarray, R1: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical linear interpolation between two rotation matrices.

    Args:
        R0 : (3,3)  start rotation
        R1 : (3,3)  end rotation
        t  : float  interpolation parameter  [0, 1]

    Returns:
        R_t : (3,3)  interpolated rotation
    """
    q0 = rotation_to_quaternion(R0)
    q1 = rotation_to_quaternion(R1)

    # Ensure shortest path
    if np.dot(q0, q1) < 0:
        q1 = -q1

    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    theta = np.arccos(dot)

    if abs(theta) < 1e-8:
        qt = q0 + t * (q1 - q0)
    else:
        s0 = np.sin((1 - t) * theta) / np.sin(theta)
        s1 = np.sin(      t  * theta) / np.sin(theta)
        qt = s0 * q0 + s1 * q1

    return quaternion_to_matrix(qt / (np.linalg.norm(qt) + 1e-12))


def interpolate_orientations(R_start: np.ndarray,
                               R_end:   np.ndarray,
                               n:       int) -> List[np.ndarray]:
    """
    SLERP interpolation of n orientations from R_start to R_end.

    Returns list of n rotation matrices.
    """
    s_vals = np.linspace(0.0, 1.0, n)
    return [slerp(R_start, R_end, float(s)) for s in s_vals]


def orientation_error(R_actual: np.ndarray, R_target: np.ndarray) -> float:
    """
    Geodesic orientation error  ‖log(R_target^T · R_actual)‖  [rad].
    Uses rotation angle of the error matrix — rotationally invariant.
    """
    R_err = R_target.T @ R_actual
    trace = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(trace))


def get_target_rotation(target_local: np.ndarray) -> np.ndarray:
    """
    Auto-orientation fallback (used when user does not supply orientation):
      z < 0.6 m  →  EE points DOWN  (table-top work)
      z ≥ 0.6 m  →  EE points TOWARD target
    """
    if target_local[2] < 0.6:
        return np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)
    z_ax = target_local / (np.linalg.norm(target_local) + 1e-12)
    x_ax = np.cross([0, 0, 1], z_ax)
    xn   = np.linalg.norm(x_ax)
    x_ax = x_ax / xn if xn > 1e-6 else np.array([1., 0., 0.])
    return np.column_stack([x_ax, np.cross(z_ax, x_ax), z_ax])


# ============================================================================
# SPHERE COLLISION MODEL  (Doosan M1013 — 11 spheres, MDH-aware)
# ============================================================================
#
# (link_frame_index, z_offset_along_z_axis, radius_m)
LINK_SPHERES: List[Tuple[int, float, float]] = [
    (0, 0.00, 0.080),   # base body
    (0, 0.08, 0.070),   # shoulder lower
    (1, 0.10, 0.070),   # upper arm proximal
    (1, 0.30, 0.070),   # upper arm mid
    (1, 0.50, 0.060),   # upper arm distal
    (2, 0.00, 0.060),   # elbow
    (3, 0.10, 0.060),   # forearm proximal
    (3, 0.30, 0.060),   # forearm mid
    (3, 0.50, 0.055),   # forearm distal
    (4, 0.00, 0.055),   # wrist 1
    (5, 0.00, 0.050),   # wrist 2
    (6, 0.00, 0.045),   # end-effector
]
N_SPHERES = len(LINK_SPHERES)

# Self-collision pairs: skip frames within 2 hops of each other.
# Threshold is 2 (not 1) because frames 3 and 5 (forearm-distal / wrist)
# are separated by |3-5|=2 but are directly connected through the wrist
# joint — they physically overlap by design in the M1013 DH geometry.
SELF_COLLISION_PAIRS: List[Tuple[int, int]] = [
    (i, j)
    for i in range(N_SPHERES)
    for j in range(i + 2, N_SPHERES)
    if abs(LINK_SPHERES[i][0] - LINK_SPHERES[j][0]) > 2
]


# ============================================================================
# FORWARD KINEMATICS  (Modified DH — Craig convention)
# ============================================================================

def _std_dh_matrix(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """
    Standard Denavit–Hartenberg transform matrix T_{i-1,i}.

    T = Rot(z, θ) · Trans(z, d) · Trans(x, a) · Rot(x, α)

    Produces:
      [cos(θ),  −sin(θ)·cos(α),   sin(θ)·sin(α),   a·cos(θ)]
      [sin(θ),   cos(θ)·cos(α),  −cos(θ)·sin(α),   a·sin(θ)]
      [0,        sin(α),           cos(α),           d       ]
      [0,        0,                0,                1       ]

    Columns: [a, alpha, d, theta_offset] (Standard DH — NOT Modified DH).
    Verified: FK(home=[0,0,0,0,0,0]) → EE local = [0.6545, 0, −0.5275].
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha),  np.sin(alpha)
    return np.array([
        [ct,  -st*ca,   st*sa,  a*ct],
        [st,   ct*ca,  -ct*sa,  a*st],
        [0,    sa,      ca,     d   ],
        [0,    0,       0,      1   ],
    ])


def forward_kinematics(joints: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Doosan M1013 forward kinematics — Standard DH convention.

    theta_i = DH_PARAMS[i, 3] + joints[i]   (theta_offset + joint_angle)
    DH_PARAMS columns: [a, alpha, d, theta_offset]

    At home [0,0,0,0,0,0]:
        EE local position : [0.6545, 0, −0.5275]  ← verified vs Gazebo terminal
        (DSR01 world = base + local = [0.6545, 0.5, −0.5275])

    Returns:
        ee_pos    : (3,)    EE position in robot-local frame
        ee_rot    : (3,3)   EE rotation matrix
        T_frames  : list[7] of (4,4)
                    frames[0] = identity (base)
                    frames[i] = cumulative transform through joint i
    """
    T = np.eye(4)
    frames = [T.copy()]
    for i in range(6):
        a, alpha, d, theta_off = DH_PARAMS[i]
        theta = theta_off + joints[i]
        T = T @ _std_dh_matrix(a, alpha, d, theta)
        frames.append(T.copy())
    return T[:3, 3].copy(), T[:3, :3].copy(), frames


# ============================================================================
# SPHERE CENTRES
# ============================================================================

def get_sphere_centres(joints: np.ndarray,
                       base:   np.ndarray) -> np.ndarray:
    """All N_SPHERES sphere centres in WORLD frame.  Shape (N_SPHERES, 3)."""
    _, _, frames = forward_kinematics(joints)
    centres = np.zeros((N_SPHERES, 3))
    for k, (fi, zo, _) in enumerate(LINK_SPHERES):
        T = frames[fi]
        centres[k] = T[:3, 3] + zo * T[:3, 2] + base
    return centres


# ============================================================================
# COLLISION SCORES
# ============================================================================

def self_collision_score(joints: np.ndarray) -> float:
    """Summed penetration depth for non-adjacent link sphere pairs (local frame)."""
    _, _, frames = forward_kinematics(joints)
    centres = np.zeros((N_SPHERES, 3))
    for k, (fi, zo, _) in enumerate(LINK_SPHERES):
        T = frames[fi]
        centres[k] = T[:3, 3] + zo * T[:3, 2]
    total = 0.0
    for i, j in SELF_COLLISION_PAIRS:
        d = np.linalg.norm(centres[i] - centres[j])
        total += max(0.0, LINK_SPHERES[i][2] + LINK_SPHERES[j][2] - d)
    return total
    return total


def inter_arm_collision_score(joints:      np.ndarray,
                               base:       np.ndarray,
                               other_arms: List[Dict]) -> float:
    """Summed penetration depth vs ALL other arms (world frame)."""
    if not other_arms:
        return 0.0
    my_c = get_sphere_centres(joints, base)
    total = 0.0
    for arm in other_arms:
        oc = get_sphere_centres(arm['joints'], arm['base'])
        for i in range(N_SPHERES):
            for j in range(N_SPHERES):
                d = np.linalg.norm(my_c[i] - oc[j])
                total += max(0.0, LINK_SPHERES[i][2] + LINK_SPHERES[j][2] - d)
    return total


def total_collision_score(joints:      np.ndarray,
                           base:       np.ndarray,
                           other_arms: List[Dict]) -> float:
    return self_collision_score(joints) + \
           inter_arm_collision_score(joints, base, other_arms)


# ============================================================================
# JACOBIAN + MANIPULABILITY
# ============================================================================

def compute_jacobian(joints: np.ndarray) -> np.ndarray:
    """Geometric Jacobian  J ∈ ℝ^{6×6}."""
    _, _, frames = forward_kinematics(joints)
    p_ee = frames[6][:3, 3]
    J = np.zeros((6, 6))
    for i in range(6):
        z = frames[i][:3, 2]
        p = frames[i][:3, 3]
        J[:3, i] = np.cross(z, p_ee - p)
        J[3:, i] = z
    return J


def manipulability(joints: np.ndarray) -> float:
    """Yoshikawa index  w = √det(J·Jᵀ).   0 at singularity."""
    try:
        J = compute_jacobian(joints)
        return float(np.sqrt(max(0.0, np.linalg.det(J @ J.T))))
    except Exception:
        return 0.0


# ============================================================================
# JOINT-LIMIT MARGIN
# ============================================================================

def joint_limit_margin(joints: np.ndarray) -> float:
    """
    Minimum normalised distance to any joint boundary.
    Range [0, 0.5].  0 = at limit,  0.5 = at midpoint.
    """
    margins = []
    for j in range(6):
        lo, hi = JOINT_LIMITS[j]
        margins.append(min(joints[j] - lo, hi - joints[j]) / (hi - lo))
    return float(min(margins))


# ============================================================================
# JOINT ANGLE NORMALIZATION  (prevent unnecessary multi-revolution sweeps)
# ============================================================================

def _normalize_joints_to_current(solution:       np.ndarray,
                                   current_joints: np.ndarray) -> np.ndarray:
    """
    Per-joint: find the kinematically equivalent angle (solution[j] + k·2π)
    that is closest to current_joints[j], subject to JOINT_LIMITS.

    Why this matters
    ────────────────
    The numerical IK solver searches in [-2π, 2π].  J1 = -360° and J1 = 0°
    produce identical FK.  But interpolating from J1=0° → J1=-360° sweeps
    a full circle, passing through every orientation.

    Tiebreaker rule
    ───────────────
    When two equivalent angles are EQUIDISTANT from current_joints[j]
    (e.g. J4=+π and J4=−π are both π away from J4=0), we prefer the one
    with smaller absolute value (closer to 0).  This deterministically
    resolves the ±π ambiguity: from J4=0° the solver always picks +180°
    or −180° consistently on the same branch, preventing oscillation.
    """
    sol = solution.copy()
    for j in range(6):
        lo, hi = JOINT_LIMITS[j]
        base   = sol[j]
        best   = base
        bdist  = abs(base - current_joints[j])
        for k in range(-3, 4):          # ±3 full turns is always enough
            cand = base + k * 2.0 * np.pi
            if lo - 1e-9 <= cand <= hi + 1e-9:
                cand   = float(np.clip(cand, lo, hi))
                d      = abs(cand - current_joints[j])
                closer = d < bdist - 1e-9
                # Tiebreaker: among equidistant candidates prefer |cand| smaller
                tiebreak = (abs(d - bdist) < 1e-9) and (abs(cand) < abs(best) - 1e-9)
                if closer or tiebreak:
                    bdist = d
                    best  = cand
        sol[j] = best
    return sol




def _tier(values: np.ndarray, higher_better: bool) -> np.ndarray:
    rng = values.max() - values.min()
    if rng < 1e-12:
        return np.zeros(len(values), dtype=int)
    threshold = LEX_TIE_FRAC * rng
    best = values.max() if higher_better else values.min()
    in_band = (values >= best - threshold) if higher_better \
              else (values <= best + threshold)
    return np.where(in_band, 0, 1).astype(int)


def select_optimal_solution(solutions:      List[np.ndarray],
                             current_joints: np.ndarray,
                             arm_id:         Optional[str]        = None,
                             base:           Optional[np.ndarray] = None,
                             other_arms:     Optional[List[Dict]] = None,
                             verbose:        bool = True
                             ) -> Optional[Tuple[np.ndarray, Dict]]:
    """
    Lexicographic IK solution selection — paper eq. 7:

      P1 : min total collision  (self + all other arms)
      P2 : max manipulability   (Yoshikawa √det(JJᵀ))
      P3 : max joint-limit margin
      P4 : min displacement from current joints  (tiebreaker)
    """
    if not solutions:
        return None

    if other_arms is None:
        other_arms = ARM_REGISTRY.get_other_arms(arm_id) if arm_id else []
    if base is None:
        base = ARM_REGISTRY._arms[arm_id]['base'] \
               if (arm_id and arm_id in ARM_REGISTRY) else np.zeros(3)

    # ── Normalize each solution to minimize displacement from current_joints ──
    # FK / collision scores are 2π-periodic so values are unchanged.
    # Joint-limit margin and displacement both improve with normalization.
    norm_sols = [_normalize_joints_to_current(s, current_joints) for s in solutions]

    N     = len(norm_sols)
    col_s = np.array([total_collision_score(s, base, other_arms) for s in norm_sols])
    man_s = np.array([manipulability(s)                          for s in norm_sols])
    lim_s = np.array([joint_limit_margin(s)                      for s in norm_sols])
    dis_s = np.array([np.linalg.norm(s - current_joints)         for s in norm_sols])

    if verbose:
        print(f'\n  [Lex]  candidates={N}  other_arms={len(other_arms)}')
        print(f"  {'':>4} {'Collision':>10} {'Manip':>10} {'Margin':>8} {'Disp°':>8}")
        for k in range(N):
            tag = '✓' if col_s[k] <= COLLISION_TOL else '✗'
            print(f"  [{k}]{tag} {col_s[k]:>10.4f} {man_s[k]:>10.4f} "
                  f"{lim_s[k]:>8.3f} {np.degrees(dis_s[k]):>8.1f}")

    cf_mask = col_s <= COLLISION_TOL
    active  = np.where(cf_mask)[0] if cf_mask.any() else np.arange(N)
    if verbose and not cf_mask.any():
        print('  ⚠  All in collision — selecting least-bad')

    t1    = _tier(col_s[active], higher_better=False)
    tied1 = active[t1 == 0]
    t2    = _tier(man_s[tied1], higher_better=True)
    tied2 = tied1[t2 == 0]
    t3    = _tier(lim_s[tied2], higher_better=True)
    tied3 = tied2[t3 == 0]
    winner = int(tied3[np.argmin(dis_s[tied3])])

    if verbose:
        print(f'  P1→{tied1.tolist()}  P2→{tied2.tolist()}  '
              f'P3→{tied3.tolist()}  winner=[{winner}]')

    sol  = norm_sols[winner]   # normalized: no unnecessary full rotations
    info = {
        'collision_score':  float(col_s[winner]),
        'self_collision':   float(self_collision_score(sol)),
        'inter_collision':  float(inter_arm_collision_score(sol, base, other_arms)),
        'manipulability':   float(man_s[winner]),
        'limit_margin':     float(lim_s[winner]),
        'displacement':     float(dis_s[winner]),
        'collision_free':   bool(col_s[winner] <= COLLISION_TOL),
        'num_solutions':    N,
        'num_free':         int(cf_mask.sum()),
        'n_other_arms':     len(other_arms),
        'cost':             float(col_s[winner]),
    }
    return sol, info


def select_chained_solution(solutions:   List[np.ndarray],
                             prev_joints: np.ndarray,
                             arm_id:      Optional[str]        = None,
                             base:        Optional[np.ndarray] = None,
                             other_arms:  Optional[List[Dict]] = None
                             ) -> Optional[np.ndarray]:
    """
    Lexicographic selection for IK chain along path.
    Displacement measured vs prev_joints → keeps θ_j(s) on one IK branch.
    """
    result = select_optimal_solution(
        solutions, prev_joints,
        arm_id=arm_id, base=base, other_arms=other_arms,
        verbose=False,
    )
    return result[0] if result else None


# ============================================================================
# NUMERICAL IK SOLVER  (position + orientation)
# ============================================================================

def _ik_cost(joints:   np.ndarray,
             tgt_pos:  np.ndarray,
             tgt_rot:  np.ndarray,
             w_orient: float) -> float:
    """
    IK cost = position error  +  w_orient × geodesic orientation error.

    Position error  : ‖p_FK − p_target‖  (metres)
    Orientation error: geodesic angle  arccos((tr(R_err)−1)/2)  (radians)
    """
    try:
        pos, rot, _ = forward_kinematics(joints)
    except Exception:
        return 1e6
    pos_err = float(np.linalg.norm(pos - tgt_pos))
    if tgt_rot is not None:
        ori_err = orientation_error(rot, tgt_rot)
        return pos_err + w_orient * ori_err
    return pos_err


def solve_ik_numerical(target_local:  np.ndarray,
                        target_orient: Optional[np.ndarray] = None,
                        initial_guess: Optional[np.ndarray] = None,
                        n_restarts:    int   = 8,
                        pos_tol:       float = IK_POS_TOL,
                        orient_weight: float = IK_ORI_WEIGHT
                        ) -> List[np.ndarray]:
    """
    Numerical IK — L-BFGS-B with joint-limit bounds and multiple random restarts.

    Args:
        target_local  : (3,)  target position in robot-local frame
        target_orient : (3,3) target rotation matrix  (None → auto heuristic)
        initial_guess : (6,)  warm start for first restart
        n_restarts    : number of random restarts
        pos_tol       : position convergence threshold (m)
        orient_weight : weight on geodesic orientation error in cost

    Returns:
        List of distinct joint solutions satisfying pos_tol.
    """
    if target_orient is None:
        target_orient = get_target_rotation(target_local)

    bounds    = [(JOINT_LIMITS[j, 0], JOINT_LIMITS[j, 1]) for j in range(6)]
    solutions = []

    for i in range(n_restarts):
        if i == 0 and initial_guess is not None:
            x0 = initial_guess.copy()
        else:
            x0 = np.array([np.random.uniform(lo, hi) for lo, hi in bounds])

        res = minimize(
            _ik_cost, x0,
            args=(target_local, target_orient, orient_weight),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 600, 'ftol': 1e-10, 'gtol': 1e-8},
        )
        if res.fun >= pos_tol + orient_weight * np.pi:
            continue
        try:
            fk_pos, fk_rot, _ = forward_kinematics(res.x)
        except Exception:
            continue
        if np.linalg.norm(fk_pos - target_local) > pos_tol:
            continue
        # Keep only distinct solutions (joint-space distance > 0.05 rad)
        if not any(np.linalg.norm(res.x - s) < 0.05 for s in solutions):
            solutions.append(res.x.copy())

    return solutions


# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================

def compute_cost(joints: np.ndarray, current_joints: np.ndarray) -> float:
    """Legacy scalar cost."""
    col  = self_collision_score(joints)
    man  = manipulability(joints)
    disp = float(np.linalg.norm(joints - current_joints))
    return col * 10.0 + max(0.0, 0.1 - man) * 2.0 + disp