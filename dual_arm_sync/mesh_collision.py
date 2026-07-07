#!/usr/bin/env python3
"""
mesh_collision.py  --  REAL-MESH collision for the dual-arm Doosan pipeline
===========================================================================
SELF-CONTAINED: the M1013 collision geometry (10 convex parts / arm) is baked
into this file as a base64 blob -- no external data file needed. Physically
accurate cross-arm collision via FCL, the same geometry MATLAB's checkCollision
uses and the same meshes Gazebo loads. Replaces the roll-blind, shape-blind
9-frame capsule model in _robot2x.

Drop-in: same functions step_33/34/35 already call --
    pair_collides(qi, bi, qj, bj, margin=SAFETY_MARGIN) -> bool
    pair_min_dist(qi, bi, qj, bj)                       -> float (surface gap, m)
    deepest_link_pair(qi, bi, qj, bj, margin=...)       -> (li, lj, penetration) | None

Requires:  pip install python-fcl scipy    (numpy already present)
ASCII only.
===========================================================================
"""
import os, io, gzip, base64
import numpy as np

try:
    import fcl
    from scipy.spatial import ConvexHull
    _FCL_OK = True
except Exception as _e:
    _FCL_OK = False
    _IMPORT_ERR = _e

NDOF = 6
# Collision gate: two arms "collide" when their real SURFACE distance drops below
# this (0 = touching = hard stop). 5 cm by default, shared with step_33/34/35's
# CLEAR_M via the same env var so the whole pipeline uses one threshold.
# (Was 0.12 m, a capsule-era centreline margin -- too large for true surface gaps.)
SAFETY_MARGIN = float(os.environ.get('DUAL_ARM_CLEARANCE_M', '0.05'))

_JX = np.array([[0, 0, 0.1525], [0, 0.0345, 0], [0.62, 0, 0],
                [0, -0.559, 0], [0, 0, 0], [0, -0.121, 0]], float)
_JR = np.array([[0, 0, 0], [0, -1.571, -1.571], [0, 0, 1.571],
                [1.571, 0, 0], [-1.571, 0, 0], [1.571, 0, 0]], float)


def _rpy(r, p, y):
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                     [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                     [-sp,   cp*sr,            cp*cr]])


def _jT(x, r, rz):
    M = np.eye(4); M[:3, :3] = _rpy(*r); M[:3, 3] = x
    c, s = np.cos(rz), np.sin(rz)
    Rz = np.eye(4); Rz[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return M @ Rz


def fk_frames(q, base):
    """7 world transforms: F[0]=base(link_0), F[1..6]=link_1..link_6."""
    F = [np.eye(4)]; F[0][:3, 3] = np.asarray(base, float)
    T = F[0].copy()
    for i in range(NDOF):
        T = T @ _jT(_JX[i], _JR[i], q[i]); F.append(T.copy())
    return F


_BLOB = (
    "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"
    "k4SA/PcY/J/xb4tp7n+//g9+jMfGjTwAAA=="
)


def _clamp_link(k):
    return min(int(k), 5)

_BVH = None; _LINKIDX = None; _CENT = None; _RAD = None; _REPORT_LINK = None


def _load():
    global _BVH, _LINKIDX, _CENT, _RAD, _REPORT_LINK
    if _BVH is not None:
        return
    if not _FCL_OK:
        raise ImportError("mesh_collision needs python-fcl + scipy: %r" % (_IMPORT_ERR,))
    d = np.load(io.BytesIO(gzip.decompress(base64.b64decode(_BLOB))))
    _LINKIDX = d['linkidx'].astype(int)
    counts = d['counts'].astype(int)
    verts = d['verts'].astype(float)
    parts = []; off = 0
    for c in counts:
        parts.append(verts[off:off+c]); off += c
    _BVH = []; _CENT = np.zeros((len(parts), 3)); _RAD = np.zeros(len(parts))
    for i, V in enumerate(parts):
        h = ConvexHull(V)
        hv = h.vertices
        remap = {o: n for n, o in enumerate(hv)}
        HV = V[hv]
        faces = []
        for t in h.simplices:
            faces += [3, remap[t[0]], remap[t[1]], remap[t[2]]]
        # fcl.Convex uses GJK for convex-convex distance -- ~30x faster than BVHModel
        _BVH.append(fcl.Convex(HV, len(h.simplices), np.array(faces, dtype=np.int64)))
        _CENT[i] = V.mean(0)
        _RAD[i] = np.linalg.norm(V - _CENT[i], axis=1).max()
    _REPORT_LINK = np.array([_clamp_link(k) for k in _LINKIDX])


_OBJ_CACHE = {}
_CACHE_MAX = 4000        # step_34 grid has ~2*COORD_RES unique configs; cap keeps RAM bounded


def _arm_objs(q, base):
    """Posed FCL objects + world centroids for one arm, cached by (q, base) so the
    step_34 coordination grid (each config reused COORD_RES times) doesn't rebuild
    FK + collision objects on every cell -- the difference between ~3 min and a few
    seconds for the grid."""
    _load()
    q = np.asarray(q, float); base = np.asarray(base, float)
    key = (q.tobytes(), base.tobytes())
    hit = _OBJ_CACHE.get(key)
    if hit is not None:
        return hit
    F = fk_frames(q, base)
    objs = []; cw = np.zeros((len(_BVH), 3))
    for p in range(len(_BVH)):
        T = F[_LINKIDX[p]]
        objs.append(fcl.CollisionObject(_BVH[p], fcl.Transform(T[:3, :3], T[:3, 3])))
        cw[p] = T[:3, :3] @ _CENT[p] + T[:3, 3]
    if len(_OBJ_CACHE) > _CACHE_MAX:
        _OBJ_CACHE.clear()
    _OBJ_CACHE[key] = (objs, cw)
    return objs, cw


def pair_min_dist(qi, bi, qj, bj):
    """Exact minimum surface-to-surface distance between the two arms (m).
    0.0 when the meshes interpenetrate."""
    OA, cA = _arm_objs(qi, bi); OB, cB = _arm_objs(qj, bj)
    dreq = fcl.DistanceRequest(); dmin = np.inf
    for a in range(len(OA)):
        gaps = np.linalg.norm(cA[a] - cB, axis=1) - _RAD[a] - _RAD
        for b in range(len(OB)):
            if gaps[b] > dmin:
                continue
            res = fcl.DistanceResult()
            dd = fcl.distance(OA[a], OB[b], dreq, res)
            if dd < dmin:
                dmin = max(dd, 0.0)
                if dmin <= 0.0:
                    return 0.0
    return float(dmin)


def pair_collides(qi, bi, qj, bj, margin=SAFETY_MARGIN):
    """True if any cross-arm mesh pair is within `margin` of contact
    (margin=0 -> true geometric interpenetration)."""
    OA, cA = _arm_objs(qi, bi); OB, cB = _arm_objs(qj, bj)
    creq = fcl.CollisionRequest(); dreq = fcl.DistanceRequest()
    for a in range(len(OA)):
        gaps = np.linalg.norm(cA[a] - cB, axis=1) - _RAD[a] - _RAD
        for b in range(len(OB)):
            if gaps[b] > margin:
                continue
            if margin <= 0.0:
                if fcl.collide(OA[a], OB[b], creq, fcl.CollisionResult()):
                    return True
            else:
                res = fcl.DistanceResult()
                if fcl.distance(OA[a], OB[b], dreq, res) < margin:
                    return True
    return False


def deepest_link_pair(qi, bi, qj, bj, margin=SAFETY_MARGIN):
    """(link_i, link_j, penetration_m) of the worst cross-arm pair, mapped to
    6-DOF link indices (0..5), or None if all pairs clear of `margin`."""
    OA, cA = _arm_objs(qi, bi); OB, cB = _arm_objs(qj, bj)
    dreq = fcl.DistanceRequest(); worst = None; worst_pen = 0.0
    for a in range(len(OA)):
        gaps = np.linalg.norm(cA[a] - cB, axis=1) - _RAD[a] - _RAD
        for b in range(len(OB)):
            if gaps[b] > margin:
                continue
            res = fcl.DistanceResult()
            dd = fcl.distance(OA[a], OB[b], dreq, res)
            pen = margin - max(dd, 0.0)
            if pen > worst_pen:
                worst_pen = pen
                worst = (int(_REPORT_LINK[a]), int(_REPORT_LINK[b]), float(pen))
    return worst


def available():
    try:
        _load(); return True
    except Exception:
        return False


if __name__ == '__main__':
    b1 = np.array([0, 0.5, 0.]); b2 = np.array([0, -0.5, 0.]); home = np.zeros(6)
    qa = np.radians([-56.81, 53.62, 30.00, -20.21, 88.39, 0])
    qb = np.radians([44.16, 28.38, 52.28, -0.26, 88.34, 0])
    print("FCL available:", available())
    for f in [0.0, 0.5, 0.7, 0.79, 1.0]:
        qA = home + f*(qa-home); qB = home + f*(qb-home)
        print("  f=%.2f  collide=%s  gap=%.1f cm  deepest=%s" % (
            f, pair_collides(qA, b1, qB, b2, margin=0.0),
            pair_min_dist(qA, b1, qB, b2)*100,
            deepest_link_pair(qA, b1, qB, b2)))