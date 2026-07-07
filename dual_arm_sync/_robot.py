#!/usr/bin/env python3
"""
_robot.py  --  Shared robot constants for SKAR-N (steps 61-80)
Imported by all pipeline files. Never run directly.
"""

import numpy as np
from typing import Dict

# ── Doosan M1013 Modified DH ──────────────────────────────────────────────────
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
ARM_NAMES = ['dsr01', 'dsr02']

LINK_RADII    = np.array([0.10, 0.10, 0.08, 0.08, 0.06, 0.06])
LINK_NAMES    = ['base', 'shoulder', 'upper_arm', 'forearm', 'wrist1', 'wrist2']
SAFETY_MARGIN = 0.12   # metres — added to sum of link radii

HOME_Q        = np.zeros(NDOF)
J1_SEP_THRESH = 0.30   # rad — minimum |j1_i - j1_j| to use j1-preserved retraction

# Gazebo ↔ DH joint reordering
# Gazebo publishes/commands: [joint_1, joint_2, joint_4, joint_5, joint_3, joint_6]
# DH order:                  [j1,      j2,      j3,      j4,      j5,      j6     ]
DH_TO_CMD = np.array([0, 1, 3, 4, 2, 5])   # q_cmd = q_dh[DH_TO_CMD]  (send to controller)
# Reading by joint NAME (jmap) gives DH order directly -- no index reorder needed.


# ── FK utilities ─────────────────────────────────────────────────────────────

def fk(q: np.ndarray):
    """Returns (pos_local, T_4x4). Add ROBOT_BASES[name] for world position."""
    T = np.eye(4)
    for i in range(NDOF):
        al, a, to, d = DH[i]; th = q[i] + to
        ct, st = np.cos(th), np.sin(th); ca, sa = np.cos(al), np.sin(al)
        T = T @ np.array([
            [ct,    -st,    0.,  a    ],
            [st*ca,  ct*ca, -sa, -sa*d],
            [st*sa,  ct*sa,  ca,  ca*d],
            [0.,     0.,    0.,  1.   ],
        ])
    return T[:3, 3].copy(), T


def fk_world(q: np.ndarray, base: np.ndarray) -> np.ndarray:
    return fk(q)[0] + base


def link_origins(q: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Returns world positions of all 6 joint frames. Shape (6, 3)."""
    T = np.eye(4); o = np.zeros((NDOF, 3))
    for i in range(NDOF):
        al, a, to, d = DH[i]; th = q[i] + to
        ct, st = np.cos(th), np.sin(th); ca, sa = np.cos(al), np.sin(al)
        T = T @ np.array([
            [ct,    -st,    0.,  a    ],
            [st*ca,  ct*ca, -sa, -sa*d],
            [st*sa,  ct*sa,  ca,  ca*d],
            [0.,     0.,    0.,  1.   ],
        ])
        o[i] = T[:3, 3] + base
    return o


def pair_min_dist(qi, bi, qj, bj) -> float:
    oi   = link_origins(qi, bi)
    oj   = link_origins(qj, bj)
    diff = oi[:, np.newaxis, :] - oj[np.newaxis, :, :]
    return float(np.min(np.linalg.norm(diff, axis=2)))


def pair_collides(qi, bi, qj, bj) -> bool:
    oi    = link_origins(qi, bi)
    oj    = link_origins(qj, bj)
    diff  = oi[:, np.newaxis, :] - oj[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    radii = (LINK_RADII[:, np.newaxis] + LINK_RADII[np.newaxis, :]) + SAFETY_MARGIN
    return bool(np.any(dists < radii))


def jacobian(q: np.ndarray) -> np.ndarray:
    """Geometric Jacobian (6×6): [angular vel; linear vel] per joint."""
    T = np.eye(4); z = np.zeros((NDOF, 3)); p = np.zeros((NDOF, 3))
    for i in range(NDOF):
        al, a, to, d = DH[i]; th = q[i] + to
        ct, st = np.cos(th), np.sin(th); ca, sa = np.cos(al), np.sin(al)
        T = T @ np.array([
            [ct,    -st,    0.,  a    ],
            [st*ca,  ct*ca, -sa, -sa*d],
            [st*sa,  ct*sa,  ca,  ca*d],
            [0.,     0.,    0.,  1.   ],
        ])
        z[i] = T[:3, 2]; p[i] = T[:3, 3]
    ee = p[-1]
    J = np.zeros((6, NDOF))
    for i in range(NDOF):
        J[:3, i] = z[i]
        J[3:, i] = np.cross(z[i], ee - p[i])
    return J


def manipulability(q: np.ndarray) -> float:
    J = jacobian(q)
    JJT = J @ J.T
    val = float(np.sqrt(max(np.linalg.det(JJT), 0.0)))
    return float(np.tanh(val / 0.05))