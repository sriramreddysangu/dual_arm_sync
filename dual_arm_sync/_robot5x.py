#!/usr/bin/env python3
"""
_robot5x.py  --  Shared constants for step_51..57 (Hierarchical resolver pipeline)
==================================================================================
Tiered collision resolution (the architecture you described):
  tier 1 : Kuramoto phase lag           -- cheap re-timing, tried FIRST
  tier 2 : local B-spline retraction     -- pull the colliding pair toward J
  tier 3 : full detour via J pivot        -- J = [mid_j1, 0,0,0,0,0] (arm folded in)
J is produced by compute_J_pivot(); retraction runs ONLY when phase lag alone
cannot skip the collision.

Collision is REAL-MESH (FCL) via mesh_collision.py -- the same geometry MATLAB
and Gazebo use -- with the simple joint-origin model kept only as a fallback.
Every step_5X file that imports pair_min_dist / pair_collides / deepest_link_pair
from here gets mesh automatically. ASCII only.
"""
import numpy as np
from typing import Dict

_PI   = np.pi
_PI_2 = np.pi / 2.0
L1, L2, L3, L4 = 0.1525, 0.6200, 0.5590, 0.1210
A = 0.0345

DH = np.array([
    [0.0,    0.0,  0.0,    L1],
    [-_PI_2, 0.0, -_PI_2,  A ],
    [0.0,    L2,   _PI_2,  0.0],
    [_PI_2,  0.0,  0.0,    L3],
    [-_PI_2, 0.0,  0.0,    0.0],
    [_PI_2,  0.0,  0.0,    L4],
], dtype=float)

NDOF    = 6
RATE_HZ = 100.0

POS_LIM = np.array([
    [-2*_PI,  2*_PI ], [-1.6493, 1.6493], [-2.7925, 2.7925],
    [-2*_PI,  2*_PI ], [-2*_PI,  2*_PI ], [-2*_PI,  2*_PI ],
], dtype=float)

VEL_LIM = np.array([2.094, 2.094, 3.140, 3.927, 3.927, 3.927])
ACC_LIM = np.array([8.0,   8.0,   8.0,  12.0,  12.0,  12.0])

ROBOT_BASES: Dict[str, np.ndarray] = {
    'dsr01': np.array([0.0,  0.5, 0.0]),
    'dsr02': np.array([0.0, -0.5, 0.0]),
}
ARM_NAMES     = ['dsr01', 'dsr02']
LINK_RADII    = np.array([0.10, 0.10, 0.08, 0.08, 0.06, 0.06])
LINK_NAMES    = ['base', 'shoulder', 'upper_arm', 'forearm', 'wrist1', 'wrist2']
SAFETY_MARGIN = 0.12
HOME_Q        = np.zeros(NDOF)

# DH-to-controller joint reorder (for joint-state reading)
DH_TO_CMD = np.array([0, 1, 3, 4, 2, 5])


# ===========================================================================
# FORWARD KINEMATICS
# ===========================================================================
def fk(q):
    """Return (ee_pos_local, T_4x4)."""
    T = np.eye(4)
    for i in range(NDOF):
        al, a, to, d = DH[i]; th = q[i] + to
        ct, st = np.cos(th), np.sin(th); ca, sa = np.cos(al), np.sin(al)
        T = T @ np.array([
            [ct,    -st,    0.,  a    ],
            [st*ca,  ct*ca, -sa, -sa*d],
            [st*sa,  ct*sa,  ca,  ca*d],
            [0.,     0.,     0.,  1.  ],
        ])
    return T[:3, 3].copy(), T


def fk_world(q, base):
    return fk(q)[0] + np.asarray(base)


def link_origins(q, base):
    """World-frame origin of each of the 6 links."""
    T = np.eye(4); o = np.zeros((NDOF, 3))
    for i in range(NDOF):
        al, a, to, d = DH[i]; th = q[i] + to
        ct, st = np.cos(th), np.sin(th); ca, sa = np.cos(al), np.sin(al)
        T = T @ np.array([
            [ct,    -st,    0.,  a    ],
            [st*ca,  ct*ca, -sa, -sa*d],
            [st*sa,  ct*sa,  ca,  ca*d],
            [0.,     0.,     0.,  1.  ],
        ])
        o[i] = T[:3, 3] + np.asarray(base)
    return o


# ===========================================================================
# COLLISION  (joint-origin fallback; overridden by real mesh at bottom)
# ===========================================================================
def pair_min_dist(qi, bi, qj, bj):
    oi = link_origins(qi, bi); oj = link_origins(qj, bj)
    diff = oi[:, None, :] - oj[None, :, :]
    return float(np.min(np.linalg.norm(diff, axis=2)))


def pair_collides(qi, bi, qj, bj, margin=SAFETY_MARGIN):
    oi = link_origins(qi, bi); oj = link_origins(qj, bj)
    diff = oi[:, None, :] - oj[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    radii = (LINK_RADII[:, None] + LINK_RADII[None, :]) + margin
    return bool(np.any(dists < radii))


def deepest_link_pair(qi, bi, qj, bj, margin=SAFETY_MARGIN):
    """(link_i, link_j, penetration_m) of the worst link pair, or None if clear.
    Fallback version on joint origins; mesh version (below) is exact."""
    oi = link_origins(qi, bi); oj = link_origins(qj, bj)
    diff = oi[:, None, :] - oj[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    thr = (LINK_RADII[:, None] + LINK_RADII[None, :]) + margin
    pen = thr - dists
    idx = np.unravel_index(int(np.argmax(pen)), pen.shape)
    if pen[idx] <= 0:
        return None
    return (int(idx[0]), int(idx[1]), float(pen[idx]))


# ===========================================================================
# J PIVOT  (tier-2 / tier-3 retraction target = arm folded inward, safe)
# ===========================================================================
def compute_J_pivot(start_q, target_q):
    """J = midpoint waypoint for a tier-2 / tier-3 detour.
       J[0]    = (start[0] + target[0]) / 2   -> j1 partway rotated
       J[1..5] = 0                            -> arm folded inward (vertical column,
                                                  collision-proof regardless of j1)"""
    J = np.zeros(NDOF)
    J[0] = (start_q[0] + target_q[0]) / 2.0
    return J


# ===========================================================================
# REAL-MESH COLLISION OVERRIDE  (FCL)
# ===========================================================================
# When mesh_collision.py + python-fcl are importable, the three cross-arm queries
# above are replaced by exact convex-mesh (GJK) tests on the real M1013 geometry.
# The tiered resolver then decides collisions on the true arm shape (roll- and
# shape-accurate), and the retraction toward J only fires on genuine overlaps.
USE_MESH = True
if USE_MESH:
    try:
        import mesh_collision as _mc
        if _mc.available():
            pair_collides     = _mc.pair_collides       # noqa: F811
            pair_min_dist     = _mc.pair_min_dist        # noqa: F811
            deepest_link_pair = _mc.deepest_link_pair    # noqa: F811
            print("[_robot5x] REAL-MESH collision active (FCL)")
        else:
            print("[_robot5x] mesh_collision unavailable -> joint-origin fallback")
    except Exception as _e:
        print("[_robot5x] mesh import failed (%s) -> joint-origin fallback" % _e)