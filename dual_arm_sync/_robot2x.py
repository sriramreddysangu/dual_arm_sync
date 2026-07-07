#!/usr/bin/env python3
"""
_robot2x.py  --  Shared robot module for the DUAL-ARM Lagrangian pipeline
===============================================================================
Reconstructed counterpart of _robot4x, scoped to 2 arms (dsr01, dsr02).
Every step_4X file imports its constants + geometry from here, so switching a
step file from 4-arm to 2-arm is a single import change:

    from _robot4x import (...)        ->     from _robot2x import (...)

For the 6-arm scale-up, use _robot6x.py (identical except ARM_NAMES / bases).

Provides EVERYTHING the step_40..46 files import:
  constants : DH NDOF POS_LIM VEL_LIM ACC_LIM RATE_HZ ROBOT_BASES ARM_NAMES
              LINK_RADII LINK_NAMES SAFETY_MARGIN NEAR_MISS_MARGIN COMFORT_DIST
              DH_TO_CMD  L1 L2 L3 L4 A
  geometry  : fk fk_world link_origins pair_min_dist pair_collides
              deepest_link_pair truncated_jacobian_linear

Collision model:
  CAPSULE = True  -> samples points ALONG each link (swept-sphere). Catches the
                    mid-link crossings the joint-centre-only model is blind to
                    (a 27 cm blind spot was measured on the M1013). deepest_link_pair
                    still returns LINK indices 0..5 so the Lagrangian solver in
                    step_43 (truncated Jacobian per link) works unchanged.
  CAPSULE = False -> original 6-joint-centre model (matches old _robot4x exactly).

  ** NEW at bottom of file: USE_MESH=True overrides pair_collides / pair_min_dist /
     deepest_link_pair with the REAL collision meshes (FCL) from mesh_collision.py,
     the same geometry MATLAB/Gazebo use. Capsule stays as the fallback. **
ASCII only.
===============================================================================
"""
import numpy as np

# ---------------------------------------------------------------------------
# DOOSAN M1013 CONSTANTS
# ---------------------------------------------------------------------------
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

# ---- DUAL ARM layout (this is the only thing that differs from _robot6x) ----
ARM_NAMES = ['dsr01', 'dsr02']
ROBOT_BASES = {
    'dsr01': np.array([0.0,  0.5, 0.0]),
    'dsr02': np.array([0.0, -0.5, 0.0]),
}

LINK_RADII    = np.array([0.10, 0.10, 0.08, 0.08, 0.06, 0.06])
LINK_NAMES    = ['base', 'shoulder', 'upper_arm', 'forearm', 'wrist1', 'wrist2']
SAFETY_MARGIN = 0.12
NEAR_MISS_MARGIN = 0.08      # warning band beyond the collision threshold (step_44)
COMFORT_DIST     = 0.30      # "comfortable" clearance target (step_44)

# Gazebo publishes joint_states as [j1, j2, j4, j5, j3, j6] -> the controller
# command array uses that SAME order. DH_TO_CMD reorders a DH-order vector
# q=[q1..q6] into command order: cmd = q[DH_TO_CMD].
# VERIFY against your controllers.yaml `joints:` list; if it is sequential
# joint_1..joint_6 then set DH_TO_CMD = np.arange(6).
DH_TO_CMD = np.array([0, 1, 3, 4, 2, 5])

# ---------------------------------------------------------------------------
# Collision-model switch
# ---------------------------------------------------------------------------
CAPSULE  = True       # along-link sampling (Gazebo-valid). False = origins only.
LINK_SUB = 3          # samples inserted between consecutive joint centres


# ===========================================================================
# FORWARD KINEMATICS
# ===========================================================================
def _chain(q, base=None):
    T = np.eye(4); origins = np.zeros((NDOF, 3))
    for i in range(NDOF):
        al, a, to, d = DH[i]; th = q[i] + to
        ct, st = np.cos(th), np.sin(th); ca, sa = np.cos(al), np.sin(al)
        T = T @ np.array([[ct, -st, 0., a],
                          [st*ca, ct*ca, -sa, -sa*d],
                          [st*sa, ct*sa, ca, ca*d],
                          [0., 0., 0., 1.]])
        origins[i] = T[:3, 3] + (base if base is not None else 0.0)
    return T, origins


def fk(q):
    """Return (ee_pos_local, T_4x4)."""
    T, _ = _chain(q)
    return T[:3, 3].copy(), T


def fk_world(q, base):
    return fk(q)[0] + np.asarray(base)


def link_origins(q, base):
    """World-frame origin of each of the 6 links."""
    _, o = _chain(q, np.asarray(base))
    return o


# ===========================================================================
# COLLISION GEOMETRY  (EXACT segment-to-segment capsule distance)
# ===========================================================================
def _seg_seg_dist(p1, q1, p2, q2):
    """Exact minimum distance between segment p1->q1 and segment p2->q2."""
    d1 = q1 - p1; d2 = q2 - p2; r = p1 - p2
    a = float(d1 @ d1); e = float(d2 @ d2); f = float(d2 @ r)
    if a <= 1e-12 and e <= 1e-12:
        return float(np.linalg.norm(p1 - p2))
    if a <= 1e-12:
        s = 0.0; t = float(np.clip(f / e, 0.0, 1.0))
    else:
        c = float(d1 @ r)
        if e <= 1e-12:
            t = 0.0; s = float(np.clip(-c / a, 0.0, 1.0))
        else:
            bdot = float(d1 @ d2); den = a * e - bdot * bdot
            s = float(np.clip((bdot * f - c * e) / den, 0.0, 1.0)) if den > 1e-12 else 0.0
            t = (bdot * s + f) / e
            if t < 0.0:
                t = 0.0; s = float(np.clip(-c / a, 0.0, 1.0))
            elif t > 1.0:
                t = 1.0; s = float(np.clip((bdot - c) / a, 0.0, 1.0))
    cp1 = p1 + d1 * s; cp2 = p2 + d2 * t
    return float(np.linalg.norm(cp1 - cp2))


# ---------------------------------------------------------------------------
# 9-FRAME collision kinematics (Modified-DH).  Captures the M1013 link offsets
# (A1,A2,A3) that a straight joint-to-joint model misses, so the capsule chain
# matches the real link shape. Used ONLY for collision geometry; IK/EE still use
# the 6-DOF fk() above (which matches Gazebo to <1mm; the 9-frame agrees to 4mm).
# ---------------------------------------------------------------------------
_L1, _L2, _L3, _L4, _L5, _L6 = 0.1525, 0.620, 0.22, 0.195, 0.14, 0.121
_A1, _A2, _A3 = 0.21, 0.1755, 0.16
_PI_2 = np.pi / 2.0
_DH9 = np.array([
    [  0.0,    0.0,    0.0,   _L1 ],   # 0: q[0]
    [-_PI_2,   0.0,  -_PI_2,  _A1 ],   # 1: q[1]
    [  0.0,   _L2,     0.0,    0.0 ],   # 2: FIXED
    [  0.0,    0.0,   _PI_2, -_A2 ],   # 3: q[2]
    [ _PI_2,   0.0,   _PI_2,  _L3 ],   # 4: q[3]
    [  0.0,   _A3,     0.0,   _L4 ],   # 5: FIXED
    [  0.0,    0.0,  -_PI_2,  _L5 ],   # 6: FIXED
    [-_PI_2,   0.0,    0.0,  -_A3 ],   # 7: q[4]
    [ _PI_2,   0.0,    0.0,   _L6 ],   # 8: q[5]
], dtype=float)
_ACT9 = [0, 1, 3, 4, 7, 8]            # actuated rows -> q[0..5]
# map each of the 9 capsule segments to a 6-DOF link index (for naming/reporting)
SEG_TO_LINK = [0, 1, 1, 2, 3, 3, 4, 4, 5]
LINK9_RADII = np.array([LINK_RADII[SEG_TO_LINK[s]] for s in range(9)])

def _mdh_T(al, a, th, d):
    ca, sa, ct, st = np.cos(al), np.sin(al), np.cos(th), np.sin(th)
    return np.array([[ct, -st, 0.0, a],
                     [st * ca, ct * ca, -sa, -sa * d],
                     [st * sa, ct * sa, ca, ca * d],
                     [0.0, 0.0, 0.0, 1.0]])

def collision_origins(q, base):
    """10 world-frame origins of the 9-frame chain (accurate link shape)."""
    T = np.eye(4); T[:3, 3] = base
    pts = [T[:3, 3].copy()]
    for i in range(9):
        al, a, th_off, d = _DH9[i]
        th = th_off + (q[_ACT9.index(i)] if i in _ACT9 else 0.0)
        T = T @ _mdh_T(al, a, th, d)
        pts.append(T[:3, 3].copy())
    return np.array(pts)


def _seg_seg_matrix(P1, Q1, P2, Q2):
    """Vectorized: distance matrix [Ni,Nj] between two sets of segments."""
    d1 = Q1 - P1; d2 = Q2 - P2
    r = P1[:, None, :] - P2[None, :, :]
    a = np.sum(d1 * d1, axis=1)[:, None]
    e = np.sum(d2 * d2, axis=1)[None, :]
    f = np.einsum('jk,ijk->ij', d2, r)
    c = np.einsum('ik,ijk->ij', d1, r)
    b = np.einsum('ik,jk->ij', d1, d2)
    den = a * e - b * b
    sden = np.where(den > 1e-12, den, 1.0)
    sa = np.where(a > 1e-12, a, 1.0); se = np.where(e > 1e-12, e, 1.0)
    s0 = np.where(den > 1e-12, np.clip((b * f - c * e) / sden, 0.0, 1.0), 0.0)
    t = (b * s0 + f) / se
    s = np.where(t < 0, np.clip(-c / sa, 0.0, 1.0),
                 np.where(t > 1, np.clip((b - c) / sa, 0.0, 1.0), s0))
    t = np.clip(t, 0.0, 1.0)
    cp1 = P1[:, None, :] + d1[:, None, :] * s[:, :, None]
    cp2 = P2[None, :, :] + d2[None, :, :] * t[:, :, None]
    return np.linalg.norm(cp1 - cp2, axis=2)


def _arm_caps(q, base):
    o = collision_origins(q, base)
    return o, o[:9], o[1:10]


def caps_min_dist(ci, cj):
    """Min centre-line distance from precomputed caps (oi,P1,Q1),(oj,P2,Q2)."""
    oi, P1, Q1 = ci; oj, P2, Q2 = cj
    gap = _aabb_gap(oi, oj)
    if gap > 1.5:
        return gap
    return float(_seg_seg_matrix(P1, Q1, P2, Q2).min())


def caps_collide(ci, cj, margin=SAFETY_MARGIN):
    oi, P1, Q1 = ci; oj, P2, Q2 = cj
    if _aabb_gap(oi, oj) > 2 * LINK9_RADII.max() + margin:
        return False
    M = _seg_seg_matrix(P1, Q1, P2, Q2)
    thr = LINK9_RADII[:, None] + LINK9_RADII[None, :] + margin
    return bool((M < thr).any())


def link_segments(q, base):
    """9 capsules (p, q, radius, seg_index) along the 9-frame chain."""
    o = collision_origins(q, base)
    if not CAPSULE:
        return [(o[i], o[i], LINK9_RADII[min(i, 8)], i) for i in range(len(o))]
    return [(o[s], o[s + 1], LINK9_RADII[s], s) for s in range(9)]


def _aabb_gap(oi, oj):
    """Positive lower-bound on the gap between two arms' link-origin boxes."""
    lo = np.maximum(oi.min(0), oj.min(0)); hi = np.minimum(oi.max(0), oj.max(0))
    d = lo - hi
    return float(np.linalg.norm(np.maximum(d, 0.0)))   # 0 if boxes overlap


def pair_min_dist(qi, bi, qj, bj):
    """Minimum centre-line distance between the two arms' capsules (metres)."""
    oi, P1, Q1 = _arm_caps(qi, bi); oj, P2, Q2 = _arm_caps(qj, bj)
    gap = _aabb_gap(oi, oj)
    if gap > 1.5:
        return gap
    return float(_seg_seg_matrix(P1, Q1, P2, Q2).min())


def pair_collides(qi, bi, qj, bj, margin=SAFETY_MARGIN):
    oi, P1, Q1 = _arm_caps(qi, bi); oj, P2, Q2 = _arm_caps(qj, bj)
    if _aabb_gap(oi, oj) > 2 * LINK9_RADII.max() + margin:
        return False
    M = _seg_seg_matrix(P1, Q1, P2, Q2)
    thr = LINK9_RADII[:, None] + LINK9_RADII[None, :] + margin
    return bool((M < thr).any())


def deepest_link_pair(qi, bi, qj, bj, margin=SAFETY_MARGIN):
    """(link_i, link_j, penetration_m) mapped to 6-DOF link indices, or None."""
    oi, P1, Q1 = _arm_caps(qi, bi); oj, P2, Q2 = _arm_caps(qj, bj)
    if _aabb_gap(oi, oj) > 2 * LINK9_RADII.max() + margin:
        return None
    M = _seg_seg_matrix(P1, Q1, P2, Q2)
    pen = (LINK9_RADII[:, None] + LINK9_RADII[None, :] + margin) - M
    idx = np.unravel_index(int(np.argmax(pen)), pen.shape)
    if pen[idx] <= 0:
        return None
    return (SEG_TO_LINK[idx[0]], SEG_TO_LINK[idx[1]], float(pen[idx]))


# ===========================================================================
# TRUNCATED LINEAR JACOBIAN  (only joints upstream of `link_idx` contribute)
# ===========================================================================
def truncated_jacobian_linear(q, link_idx, base, eps=1e-6):
    """(3, NDOF) linear-velocity Jacobian of link `link_idx`'s origin.
    Columns for joints downstream of link_idx are zero (an upstream link cannot
    be moved by a downstream joint). Used by step_43 to map a Cartesian
    displacement to a joint perturbation via damped least squares."""
    J = np.zeros((3, NDOF))
    o0 = link_origins(q, base)[link_idx]
    for j in range(link_idx + 1):
        dq = q.copy(); dq[j] += eps
        oj = link_origins(dq, base)[link_idx]
        J[:, j] = (oj - o0) / eps
    return J


# ===========================================================================
# REAL-MESH COLLISION OVERRIDE  (FCL)  -- added for the mesh-accurate pipeline
# ===========================================================================
# When USE_MESH=True and mesh_collision.py + python-fcl are importable, the three
# cross-arm collision queries below are replaced by exact convex-mesh (GJK) tests
# on the real M1013 collision geometry (10 parts/arm) -- the same meshes Gazebo
# and MATLAB use. This closes the capsule under-detection (roll-blind + shape-
# blind). Everything downstream (step_33 scan, step_34 Kuramoto, step_35) uses
# these names unchanged, so no other file needs editing.
#
# The STEP 31 endpoint CSP intentionally keeps caps_collide (capsule) so step_31's
# branch selection is unchanged; only the path-collision queries switch to mesh.
USE_MESH = True
if USE_MESH:
    try:
        import mesh_collision as _mc
        if _mc.available():
            pair_collides     = _mc.pair_collides       # noqa: F811
            pair_min_dist     = _mc.pair_min_dist        # noqa: F811
            deepest_link_pair = _mc.deepest_link_pair    # noqa: F811
            print("[_robot2x] REAL-MESH collision active (FCL)")
        else:
            print("[_robot2x] mesh_collision unavailable -> capsule model")
    except Exception as _e:
        print("[_robot2x] mesh import failed (%s) -> capsule model" % _e)