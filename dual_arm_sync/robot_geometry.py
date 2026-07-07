"""
robot_geometry.py  —  Doosan M1013 Physical Geometry
9-Frame DH Table + Capsule Collision Model

Shared by step_3, step_4, step_5, step_6.
All IK/FK for motion planning uses step_1's 6-frame DH (see fk6 / fk6_world).
The 9-frame model is used ONLY for collision geometry.

6-FRAME DH  (step_1 IK/FK — authoritative for all joint-space computations)
═══════════════════════════════════════════════════════════════════════════════
  Row │ α(i-1)   │ a(i-1) │ θ_offset │ d(i)
  ────┼──────────┼────────┼──────────┼──────
   0  │  0       │  0     │  0       │  L1=0.1525
   1  │ -π/2     │  0     │ -π/2     │  A=0.0345
   2  │  0       │  L2    │  π/2     │  0
   3  │  π/2     │  0     │  0       │  L3=0.5590
   4  │ -π/2     │  0     │  0       │  0
   5  │  π/2     │  0     │  0       │  L4=0.1210

9-FRAME DH  (collision geometry — physical link path)
═══════════════════════════════════════════════════════════════════════════════
  Row │ α(i-1)   │ a(i-1) │ θ_offset │ d(i)   │ Joint var
  ────┼──────────┼────────┼──────────┼────────┼──────────
   0  │  0       │  0     │  0       │  L1    │  q[0]
   1  │ -π/2     │  0     │ -π/2     │  A1    │  q[1]
   2  │  0       │  L2    │  0       │  0     │  fixed
   3  │  0       │  0     │  π/2     │ -A2    │  q[2]
   4  │  π/2     │  0     │  π/2     │  L3    │  q[3]
   5  │  0       │  A3    │  0       │  L4    │  fixed
   6  │  0       │  0     │ -π/2     │  L5    │  fixed
   7  │ -π/2     │  0     │  0       │ -A3    │  q[4]
   8  │  π/2     │  0     │  0       │  L6    │  q[5]
"""

import numpy as np
from typing import Dict, List, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# PHYSICAL LINK LENGTHS  (Doosan M1013)
# ─────────────────────────────────────────────────────────────────────────────

_PI   = np.pi
_PI_2 = np.pi / 2.

# 6-frame IK constants (step_1 authoritative values)
L1_6 = 0.1525
L2_6 = 0.6200
L3_6 = 0.5590
L4_6 = 0.1210
A_6  = 0.0345

# 9-frame physical link constants
L1 = 0.1525
L2 = 0.620
L3 = 0.22
L4 = 0.195
L5 = 0.14
L6 = 0.121
A1 = 0.21
A2 = 0.1755
A3 = 0.16

# ─────────────────────────────────────────────────────────────────────────────
# 6-FRAME DH TABLE  (authoritative for IK/FK — matches step_1 exactly)
# ─────────────────────────────────────────────────────────────────────────────

DH6 = np.array([
    [0.0,    0.0,   0.0,    L1_6],
    [-_PI_2, 0.0,  -_PI_2,  A_6 ],
    [0.0,    L2_6,  _PI_2,  0.0 ],
    [_PI_2,  0.0,   0.0,    L3_6],
    [-_PI_2, 0.0,   0.0,    0.0 ],
    [_PI_2,  0.0,   0.0,    L4_6],
], dtype=float)

# ─────────────────────────────────────────────────────────────────────────────
# 9-FRAME DH TABLE  [α(i-1), a(i-1), θ_offset, d(i)]
# ─────────────────────────────────────────────────────────────────────────────

DH9 = np.array([
    [  0.0,    0.0,   0.0,    L1  ],   # row 0: q[0]
    [ -_PI_2,  0.0,  -_PI_2,  A1  ],   # row 1: q[1]  offset -π/2
    [  0.0,    L2,    0.0,    0.0 ],   # row 2: FIXED  θ=0
    [  0.0,    0.0,   _PI_2, -A2  ],   # row 3: q[2]  offset +π/2
    [  _PI_2,  0.0,   _PI_2,  L3  ],   # row 4: q[3]  offset +π/2
    [  0.0,    A3,    0.0,    L4  ],   # row 5: FIXED  θ=0
    [  0.0,    0.0,  -_PI_2,  L5  ],   # row 6: FIXED  θ=-π/2
    [ -_PI_2,  0.0,   0.0,   -A3  ],   # row 7: q[4]
    [  _PI_2,  0.0,   0.0,    L6  ],   # row 8: q[5]
], dtype=float)

# Maps each DH9 row to joint index (0-5), -1 = fixed frame
JOINT_VAR = np.array([0, 1, -1, 2, 3, -1, -1, 4, 5], dtype=int)

N_FRAMES = 9
NDOF     = 6

# ─────────────────────────────────────────────────────────────────────────────
# JOINT LIMITS
# ─────────────────────────────────────────────────────────────────────────────

POS_LIM = np.array([
    [-2*_PI,  2*_PI ],
    [-1.6493, 1.6493],
    [-2.7925, 2.7925],
    [-2*_PI,  2*_PI ],
    [-2*_PI,  2*_PI ],
    [-2*_PI,  2*_PI ],
], dtype=float)

VEL_LIM = np.array([2.094, 2.094, 3.141, 3.927, 3.927, 3.927])
ACC_LIM = np.array([8.0,   8.0,   8.0,  12.0,  12.0,  12.0 ])

# ─────────────────────────────────────────────────────────────────────────────
# CAPSULE RADII  (diameter 10cm→6cm linear taper, 9 segments)
# ─────────────────────────────────────────────────────────────────────────────

CAPSULE_RADII = np.array([
    0.0500,   # L1  — 10.0cm diameter
    0.0475,   # A1  —  9.5cm
    0.0450,   # L2  —  9.0cm
    0.0425,   # A2  —  8.5cm
    0.0400,   # L3  —  8.0cm
    0.0375,   # L4  —  7.5cm
    0.0350,   # A3  —  7.0cm
    0.0325,   # L5  —  6.5cm
    0.0300,   # L6  —  6.0cm
])

SEG_NAMES = ['L1', 'A1', 'L2', 'A2', 'L3', 'L4', 'A3', 'L5', 'L6']

# Non-adjacent segment pairs for self-collision
SELF_PAIRS = [
    (0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),
    (1,3),(1,4),(1,5),(1,6),(1,7),(1,8),
    (2,4),(2,5),(2,6),(2,7),(2,8),
    (3,5),(3,6),(3,7),(3,8),
    (4,6),(4,7),(4,8),
    (5,7),(5,8),
    (6,8),
]

INTER_ARM_MARGIN = 0.02   # [m]  safety buffer for inter-arm capsule collision
WARNING_MARGIN   = 0.06   # [m]  warning zone (outside this = OK, inside = warn)
SELF_MARGIN      = 0.003  # [m]

# ─────────────────────────────────────────────────────────────────────────────
# DH TRANSFORM HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _dh_mat(al: float, a: float, th: float, d: float) -> np.ndarray:
    ct, st = np.cos(th), np.sin(th)
    ca, sa = np.cos(al), np.sin(al)
    return np.array([
        [ct,    -st,    0.,  a    ],
        [st*ca,  ct*ca, -sa, -sa*d],
        [st*sa,  ct*sa,  ca,  ca*d],
        [0.,     0.,    0.,  1.   ],
    ])

# ─────────────────────────────────────────────────────────────────────────────
# 6-FRAME FK  (authoritative — matches step_1 exactly)
# ─────────────────────────────────────────────────────────────────────────────

def fk6(q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """6-frame FK.  Returns (pos_local_3, T_4x4_local)."""
    T = np.eye(4)
    for i in range(6):
        al, a, to, d = DH6[i]
        T = T @ _dh_mat(al, a, q[i] + to, d)
    return T[:3, 3].copy(), T.copy()


def fk6_world(q: np.ndarray, base: np.ndarray) -> np.ndarray:
    """World-frame EE position using 6-frame DH."""
    pos, _ = fk6(q)
    return pos + np.asarray(base)


# ─────────────────────────────────────────────────────────────────────────────
# 9-FRAME FK  (collision geometry only)
# ─────────────────────────────────────────────────────────────────────────────

def joint_origins_9(q: np.ndarray, base: np.ndarray) -> np.ndarray:
    """
    Returns (10, 3) world-frame points: origins[0] = base,
    origins[k+1] = end of DH9 row k (world frame).
    """
    T = np.eye(4)
    origins = np.zeros((10, 3))
    origins[0] = np.asarray(base).copy()
    b = np.asarray(base)
    for row in range(N_FRAMES):
        al, a, th_off, d = DH9[row]
        jv = JOINT_VAR[row]
        th = th_off + (float(q[jv]) if jv >= 0 else 0.0)
        T  = T @ _dh_mat(al, a, th, d)
        origins[row + 1] = T[:3, 3] + b
    return origins


def fk_ee_9(q: np.ndarray, base: np.ndarray) -> np.ndarray:
    """World-frame EE from 9-frame DH (collision geometry only)."""
    return joint_origins_9(q, base)[-1]


def get_capsules_9(q: np.ndarray,
                   base: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """Returns 9 capsule tuples (p1, p2, radius)."""
    o = joint_origins_9(q, base)
    return [(o[i], o[i+1], CAPSULE_RADII[i]) for i in range(9)]


# ─────────────────────────────────────────────────────────────────────────────
# SEGMENT-SEGMENT DISTANCE  (Eberly 2001 — numerically robust)
# ─────────────────────────────────────────────────────────────────────────────

def seg_seg_dist(p1: np.ndarray, p2: np.ndarray,
                 q1: np.ndarray, q2: np.ndarray) -> float:
    """Minimum distance between two finite line segments."""
    d1 = p2 - p1;  d2 = q2 - q1;  r = p1 - q1
    a  = float(np.dot(d1, d1))
    e  = float(np.dot(d2, d2))
    f  = float(np.dot(d2, r))
    EPS = 1e-10
    if a <= EPS and e <= EPS:
        return float(np.linalg.norm(p1 - q1))
    if a <= EPS:
        s = 0.0;  t = float(np.clip(f / e, 0., 1.))
    else:
        c = float(np.dot(d1, r))
        if e <= EPS:
            t = 0.0;  s = float(np.clip(-c / a, 0., 1.))
        else:
            b     = float(np.dot(d1, d2))
            denom = a * e - b * b
            s     = (float(np.clip((b*f - c*e) / denom, 0., 1.))
                     if abs(denom) > EPS else 0.)
            t = (b * s + f) / e
            if t < 0.:
                t = 0.;  s = float(np.clip(-c / a, 0., 1.))
            elif t > 1.:
                t = 1.;  s = float(np.clip((b - c) / a, 0., 1.))
    return float(np.linalg.norm((p1 + s*d1) - (q1 + t*d2)))


# ─────────────────────────────────────────────────────────────────────────────
# INTER-ARM COLLISION  (81 capsule-pair checks)
# ─────────────────────────────────────────────────────────────────────────────

def pair_min_dist_9(q_i: np.ndarray, base_i: np.ndarray,
                    q_j: np.ndarray, base_j: np.ndarray) -> float:
    """
    Minimum capsule-surface distance between two arms.
    Negative value indicates interpenetration.
    """
    oi = joint_origins_9(q_i, base_i)
    oj = joint_origins_9(q_j, base_j)
    mn = float('inf')
    for i in range(9):
        ri = CAPSULE_RADII[i]
        for j in range(9):
            rj = CAPSULE_RADII[j]
            d  = seg_seg_dist(oi[i], oi[i+1], oj[j], oj[j+1]) - ri - rj
            if d < mn:
                mn = d
    return mn


def pair_collides_9(q_i: np.ndarray, base_i: np.ndarray,
                    q_j: np.ndarray, base_j: np.ndarray,
                    margin: float = INTER_ARM_MARGIN) -> bool:
    """True if any capsule pair is within margin distance."""
    oi = joint_origins_9(q_i, base_i)
    oj = joint_origins_9(q_j, base_j)
    for i in range(9):
        ri = CAPSULE_RADII[i]
        for j in range(9):
            rj = CAPSULE_RADII[j]
            if seg_seg_dist(oi[i], oi[i+1], oj[j], oj[j+1]) < ri + rj + margin:
                return True
    return False


def pair_min_dist_9_detail(q_i: np.ndarray, base_i: np.ndarray,
                            q_j: np.ndarray, base_j: np.ndarray,
                            margin: float = INTER_ARM_MARGIN
                            ) -> List[dict]:
    """
    Returns per-capsule-pair details for collision report.
    Only entries within WARNING_MARGIN are included.
    """
    oi = joint_origins_9(q_i, base_i)
    oj = joint_origins_9(q_j, base_j)
    events = []
    for i in range(9):
        ri = CAPSULE_RADII[i]
        for j in range(9):
            rj = CAPSULE_RADII[j]
            d   = seg_seg_dist(oi[i], oi[i+1], oj[j], oj[j+1])
            thr = ri + rj + margin
            wrn = ri + rj + WARNING_MARGIN
            if d < wrn:
                events.append({
                    'seg_i'        : i,
                    'seg_j'        : j,
                    'name_i'       : SEG_NAMES[i],
                    'name_j'       : SEG_NAMES[j],
                    'distance_m'   : round(d, 5),
                    'threshold_m'  : round(thr, 5),
                    'penetration_m': round(max(thr - d, 0.0), 5),
                    'collision'    : d < thr,
                    'warning'      : thr <= d < wrn,
                })
    return events


# ─────────────────────────────────────────────────────────────────────────────
# SELF-COLLISION  (per-arm)
# ─────────────────────────────────────────────────────────────────────────────

def self_collides_9(q: np.ndarray, base: np.ndarray) -> bool:
    caps = get_capsules_9(q, base)
    for (i, j) in SELF_PAIRS:
        p1, p2, ri = caps[i]
        q1, q2, rj = caps[j]
        if seg_seg_dist(p1, p2, q1, q2) < ri + rj + SELF_MARGIN:
            return True
    return False


def min_self_clearance_9(q: np.ndarray,
                          base: np.ndarray) -> Tuple[float, str]:
    caps   = get_capsules_9(q, base)
    mn_d   = float('inf');  mn_name = ''
    for (i, j) in SELF_PAIRS:
        p1, p2, ri = caps[i]
        q1, q2, rj = caps[j]
        d = seg_seg_dist(p1, p2, q1, q2) - ri - rj
        if d < mn_d:
            mn_d = d;  mn_name = f'{SEG_NAMES[i]}↔{SEG_NAMES[j]}'
    return float(mn_d), mn_name