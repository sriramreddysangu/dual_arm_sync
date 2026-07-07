#!/usr/bin/env python3
"""
multi_arm_core.py  --  Core trajectory engine for 6-arm Doosan M1013 planning
===============================================================================
This module replaces the old N_SEG=5 x N_CP_SEG=4 segment-grid machinery with
ONE global clamped cubic B-spline per arm.  Collisions are resolved by LOCAL
knot insertion (Boehm) + retraction toward (J1, 0,0,0,0,0), with an
arc-fraction router that hands boundary collisions to Kuramoto instead of
geometry.  Kuramoto phase synchronisation is the primary (temporal) stage.

Design (agreed):
  * Path        : single clamped cubic B-spline, C2 by construction.
  * Seed        : straight line in joint space, quintic (min-jerk) timing.
  * Resolution  : timing first (Kuramoto), geometry second (knot insertion),
                  and only in the interior 0.15 < s < 0.85.  Boundary
                  collisions (s<=0.15 or s>=0.85) go back to Kuramoto / staging
                  because deflection authority vanishes at a clamped end.
  * Retraction  : pull the LOCAL control points toward (J1, 0,...,0).  This is
                  an approximating pull (convex-hull bend), NOT interpolation.

Constants, bases, link model and IK are taken verbatim from step_21..step_26.
ASCII only.
===============================================================================
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# ROBOT CONSTANTS (Doosan M1013) -- from step_21..26
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

POS_LIM = np.array([
    [-2*_PI,  2*_PI ], [-1.6493, 1.6493], [-2.7925, 2.7925],
    [-2*_PI,  2*_PI ], [-2*_PI,  2*_PI ], [-2*_PI,  2*_PI ],
], dtype=float)

VEL_LIM = np.array([2.094, 2.094, 3.140, 3.927, 3.927, 3.927])
ACC_LIM = np.array([8.0,   8.0,   8.0,  12.0,  12.0,  12.0])
NDOF    = 6
RATE_HZ = 100.0
DEG     = 3

ROBOT_BASES: Dict[str, np.ndarray] = {
    'dsr01': np.array([ 0.0,  0.5, 0.0]),
    'dsr02': np.array([ 0.0, -0.5, 0.0]),
    'dsr03': np.array([ 1.0,  0.5, 0.0]),
    'dsr04': np.array([ 1.0, -0.5, 0.0]),
    'dsr05': np.array([-1.0,  0.5, 0.0]),
    'dsr06': np.array([-1.0, -0.5, 0.0]),
}
ROBOT_NAMES = ['dsr01', 'dsr02', 'dsr03', 'dsr04', 'dsr05', 'dsr06']

LINK_RADII    = np.array([0.10, 0.10, 0.08, 0.08, 0.06, 0.06])
LINK_NAMES    = ['base', 'shoulder', 'upper_arm', 'forearm', 'wrist1', 'wrist2']
SAFETY_MARGIN = 0.12

# Quintic / timing
MIN_DURATION = 5.0
VMARGIN      = 0.95        # use 95% of limits when sizing duration

# Resolution router / budgets
BOUNDARY_LO  = 0.08      # below this arc-fraction -> staging retraction
BOUNDARY_HI  = 0.92      # above this -> staging retraction
MAX_REFINE   = 6
SEED_NCP     = 8          # control points in the straight-line seed

# Kuramoto
K_BASE        = 8.0       # 6-arm tuned value (per memory)
K_REPULSE     = 80.0
K_EMERGENCY   = 250.0
KUR_DT        = 0.01
MIN_SAFE      = 0.15
REPULSE_D     = 0.28
LEADER_THRESH = 0.05
KUR_RATE_MAX  = 2.0       # phase rate clamp (monotonic, never backward)


# ===========================================================================
# FORWARD KINEMATICS + GEOMETRY
# ===========================================================================
def fk(q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (ee_pos_local, T_4x4)."""
    T = np.eye(4)
    for i in range(NDOF):
        al, a, to, d = DH[i]; th = q[i] + to
        ct, st = np.cos(th), np.sin(th); ca, sa = np.cos(al), np.sin(al)
        T = T @ np.array([[ct, -st, 0., a],
                          [st*ca, ct*ca, -sa, -sa*d],
                          [st*sa, ct*sa, ca, ca*d],
                          [0., 0., 0., 1.]])
    return T[:3, 3].copy(), T


def fk_world(q: np.ndarray, base: np.ndarray) -> np.ndarray:
    return fk(q)[0] + base


def link_origins(q: np.ndarray, base: np.ndarray) -> np.ndarray:
    """World-frame origin of each of the 6 links."""
    T = np.eye(4); o = np.zeros((NDOF, 3))
    for i in range(NDOF):
        al, a, to, d = DH[i]; th = q[i] + to
        ct, st = np.cos(th), np.sin(th); ca, sa = np.cos(al), np.sin(al)
        T = T @ np.array([[ct, -st, 0., a],
                          [st*ca, ct*ca, -sa, -sa*d],
                          [st*sa, ct*sa, ca, ca*d],
                          [0., 0., 0., 1.]])
        o[i] = T[:3, 3] + base
    return o


# Number of extra sample points inserted ALONG each link (between consecutive
# joint centres). The 6-origins-only model is blind to the long links between
# joints (M1013 links are 0.5-0.6 m), so two forearms can cross while the joint
# centres stay >25 cm apart. Sampling along the links = swept-sphere / capsule
# collision model. LINK_SUB=3 -> 16 points/arm (vs 6). Raise for more fidelity.
LINK_SUB = 3


def link_points(q: np.ndarray, base: np.ndarray, n_sub: int = LINK_SUB):
    """World points sampled along the kinematic chain + their sphere radii.
    Approximates each link as a capsule (sphere swept along the segment)."""
    o = link_origins(q, base)                      # (NDOF,3) joint centres
    pts = [o[0]]; rad = [LINK_RADII[0]]
    for i in range(NDOF - 1):
        for t in np.linspace(0.0, 1.0, n_sub + 1)[1:]:
            pts.append((1.0 - t) * o[i] + t * o[i + 1])
            rad.append(max(LINK_RADII[i], LINK_RADII[i + 1]))   # conservative
    return np.asarray(pts), np.asarray(rad)


def _pair_grids(qi, bi, qj, bj):
    pi, ri = link_points(qi, bi); pj, rj = link_points(qj, bj)
    diff = pi[:, None, :] - pj[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    return dist, ri, rj, pi, pj


def pair_min_dist(qi, bi, qj, bj) -> float:
    """Surface-to-surface min distance along the links (capsule model)."""
    dist, ri, rj, _, _ = _pair_grids(qi, bi, qj, bj)
    return float(np.min(dist - ri[:, None] - rj[None, :]))


def closest_link_pair(qi, bi, qj, bj):
    """Return (surface_dist, pi_idx, pj_idx, unit_sep) where unit_sep points j -> i."""
    dist, ri, rj, pi, pj = _pair_grids(qi, bi, qj, bj)
    surf = dist - ri[:, None] - rj[None, :]
    a, b = np.unravel_index(int(np.argmin(surf)), surf.shape)
    sep = pi[a] - pj[b]; n = np.linalg.norm(sep)
    unit = sep / n if n > 1e-9 else np.array([0., 0., 1.])
    return float(surf[a, b]), int(a), int(b), unit


def pair_collides(qi, bi, qj, bj, margin=SAFETY_MARGIN) -> bool:
    dist, ri, rj, _, _ = _pair_grids(qi, bi, qj, bj)
    return bool(np.any(dist < ri[:, None] + rj[None, :] + margin))


def pair_clearance(qi, bi, qj, bj, margin=SAFETY_MARGIN) -> float:
    """Signed margin to collision (capsule model). Positive = safe."""
    dist, ri, rj, _, _ = _pair_grids(qi, bi, qj, bj)
    return float(np.min(dist - ri[:, None] - rj[None, :] - margin))


def pair_status(qi, bi, qj, bj, margin=SAFETY_MARGIN):
    """Single pass -> (surface_min_distance, collides_bool). Used in Kuramoto."""
    dist, ri, rj, _, _ = _pair_grids(qi, bi, qj, bj)
    surf = dist - ri[:, None] - rj[None, :]
    return float(np.min(surf)), bool(np.any(surf < margin))


# ===========================================================================
# JOINT WRAP HELPERS (from step_26 -- prevents accumulated-angle false fails)
# ===========================================================================
def normalize_joints(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    for j in range(NDOF):
        lo, hi = POS_LIM[j]; span = hi - lo
        out[j] = lo + (q[j] - lo) % span
        out[j] = float(np.clip(out[j], lo, hi))
    return out


def angular_error(q_read, q_target) -> np.ndarray:
    diff = q_read - q_target
    return np.abs(np.arctan2(np.sin(diff), np.cos(diff)))


# ===========================================================================
# QUINTIC (MINIMUM-JERK) TIMING  --  smootherstep 6u^5 - 15u^4 + 10u^3
# ===========================================================================
def smootherstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u*u*u*(u*(u*6.0 - 15.0) + 10.0)


def smootherstep_d1(u):
    u = np.clip(u, 0.0, 1.0)
    return 30.0*u*u*(u*(u - 2.0) + 1.0)        # 30 u^2 (1-u)^2


def duration_from_limits(q_start: np.ndarray, q_target: np.ndarray) -> float:
    """
    Exact minimum-jerk duration so the quintic peaks graze the joint limits.
      |qdot|_peak = 1.875 |dq| / T          (peak of s' at tau=0.5)
      |qddot|_peak = 5.7735 |dq| / T^2      (10/sqrt(3), peak of s'')
    """
    dq = np.abs(q_target - q_start)
    T = MIN_DURATION
    for j in range(NDOF):
        if dq[j] < 1e-9:
            continue
        t_v = 1.875 * dq[j] / (VEL_LIM[j] * VMARGIN)
        t_a = float(np.sqrt(5.7735 * dq[j] / (ACC_LIM[j] * VMARGIN)))
        T = max(T, t_v, t_a)
    return float(T)


# ===========================================================================
# CLAMPED CUBIC B-SPLINE  +  BOEHM KNOT INSERTION
# ===========================================================================
def clamped_knots(n_cp: int, deg: int = DEG) -> np.ndarray:
    n_inner = max(0, n_cp - deg - 1)
    inner = np.linspace(0, 1, n_inner + 2)[1:-1] if n_inner > 0 else np.array([])
    return np.concatenate([np.zeros(deg + 1), inner, np.ones(deg + 1)])


def greville(U: np.ndarray, deg: int = DEG) -> np.ndarray:
    """Greville abscissae: control point i most influences the curve near g_i."""
    n_cp = len(U) - deg - 1
    return np.array([np.mean(U[i+1:i+1+deg]) for i in range(n_cp)])


def boehm_insert(U: np.ndarray, P: np.ndarray, u: float, deg: int = DEG):
    """
    Insert a single knot u (Boehm's algorithm). Returns (U_new, P_new).
    The curve is unchanged; one control point is added; C2 is preserved
    because the inserted interior knot has multiplicity 1.
    """
    U = np.asarray(U, float); P = np.asarray(P, float)
    n = len(P) - 1
    # find span k such that U[k] <= u < U[k+1]
    k = int(np.searchsorted(U, u, side='right') - 1)
    k = min(max(k, deg), n)
    Q = np.zeros((len(P) + 1, P.shape[1]))
    Q[:k-deg+1] = P[:k-deg+1]
    Q[k+1:] = P[k:]
    for i in range(k - deg + 1, k + 1):
        denom = U[i + deg] - U[i]
        a = (u - U[i]) / denom if denom > 1e-12 else 0.0
        Q[i] = (1.0 - a) * P[i - 1] + a * P[i]
    U_new = np.insert(U, k + 1, u)
    return U_new, Q


class Spline:
    """One arm's joint-space trajectory: a clamped cubic B-spline over s in [0,1]."""
    def __init__(self, U: np.ndarray, P: np.ndarray, deg: int = DEG):
        self.U = np.asarray(U, float)
        self.P = np.asarray(P, float)         # (n_cp, NDOF)
        self.deg = deg

    @property
    def n_cp(self):
        return self.P.shape[0]

    def copy(self):
        return Spline(self.U.copy(), self.P.copy(), self.deg)

    def eval(self, s, der: int = 0) -> np.ndarray:
        s = np.atleast_1d(np.asarray(s, float))
        out = np.zeros((len(s), NDOF))
        for j in range(NDOF):
            spl = BSpline(self.U, self.P[:, j], self.deg, extrapolate=True)
            out[:, j] = spl.derivative(der)(s) if der else spl(s)
        return out

    def insert(self, u: float):
        self.U, self.P = boehm_insert(self.U, self.P, float(u), self.deg)
        return self

    def greville(self):
        return greville(self.U, self.deg)


def seed_spline(q_start: np.ndarray, q_target: np.ndarray,
                n_cp: int = SEED_NCP) -> Spline:
    """Straight line in joint space: control points evenly spaced start->target."""
    s = np.linspace(0.0, 1.0, n_cp)
    P = q_start[None, :] + s[:, None] * (q_target - q_start)[None, :]
    P = np.clip(P, POS_LIM[:, 0], POS_LIM[:, 1])
    return Spline(clamped_knots(n_cp), P)


def sample_minjerk(spl: Spline, n_steps: int) -> np.ndarray:
    """Sample the path at quintic (min-jerk) arc spacing -> positions array."""
    tau = np.linspace(0.0, 1.0, n_steps)
    s = smootherstep(tau)
    pos = spl.eval(s)
    return np.clip(pos, POS_LIM[:, 0], POS_LIM[:, 1])


# ===========================================================================
# LOCAL RETRACTION RESOLUTION (interior collisions only)
# ===========================================================================
def retract_config(q_window: np.ndarray) -> np.ndarray:
    """(J1, 0,0,0,0,0): keep base angle, fold the rest straight up."""
    R = np.zeros(NDOF)
    R[0] = q_window[0]
    return np.clip(R, POS_LIM[:, 0], POS_LIM[:, 1])


def _ensure_local_cps(spl: Spline, s0: float, s1: float, n_insert: int) -> Spline:
    """Insert n_insert interior knots spread across (s0,s1) for local authority."""
    us = np.linspace(s0, s1, n_insert + 2)[1:-1]
    for u in us:
        spl.insert(float(u))
    return spl


def _pull_local(spl: Spline, s0: float, s1: float, R: np.ndarray,
                strength: float) -> Spline:
    """
    Move control points whose Greville abscissa lies in (s0,s1) toward R.
    A Hann taper keeps the window-edge CPs near nominal so the curve rejoins
    the original path; CPs outside the window are untouched (locality).
    """
    g = spl.greville()
    width = max(s1 - s0, 1e-6)
    for i in range(spl.n_cp):
        if s0 < g[i] < s1:
            t = (g[i] - s0) / width            # 0..1 across window
            w = strength * 0.5 * (1.0 - np.cos(2.0 * np.pi * t))   # Hann
            spl.P[i] = (1.0 - w) * spl.P[i] + w * R
    spl.P = np.clip(spl.P, POS_LIM[:, 0], POS_LIM[:, 1])
    return spl


def _window_clear(spl_i, spl_j, bi, bj, s0, s1, n=40) -> bool:
    ss = np.linspace(max(0.0, s0), min(1.0, s1), n)
    Pi = spl_i.eval(ss); Pj = spl_j.eval(ss)
    for k in range(n):
        if pair_collides(Pi[k], bi, Pj[k], bj):
            return False
    return True


def resolve_pair_local(spl_i: Spline, spl_j: Spline, bi, bj,
                       win_i: Tuple[float, float], win_j: Tuple[float, float],
                       max_rounds: int = MAX_REFINE):
    """
    Local knot-insertion retraction sized to the WHOLE collision span.
    win_i/win_j are the arc-fraction ranges where each arm is in collision.
    Both arms fold toward their own (J1,0,0,0,0,0) across that span, with the
    window + pull escalating until the padded span clears (or budget runs out).
    Returns (spl_i, spl_j, ok, rounds_used).
    """
    spl_i = spl_i.copy(); spl_j = spl_j.copy()
    ci = 0.5 * (win_i[0] + win_i[1]); cj = 0.5 * (win_j[0] + win_j[1])
    base_half_i = max(0.10, 0.5 * (win_i[1] - win_i[0]) + 0.05)
    base_half_j = max(0.10, 0.5 * (win_j[1] - win_j[0]) + 0.05)
    for r in range(max_rounds):
        grow = 1.0 + 0.30 * r
        s0i, s1i = max(0.0, ci - base_half_i * grow), min(1.0, ci + base_half_i * grow)
        s0j, s1j = max(0.0, cj - base_half_j * grow), min(1.0, cj + base_half_j * grow)
        n_ins = 3 + r
        _ensure_local_cps(spl_i, s0i, s1i, n_ins)
        _ensure_local_cps(spl_j, s0j, s1j, n_ins)
        Ri = retract_config(spl_i.eval([ci])[0])
        Rj = retract_config(spl_j.eval([cj])[0])
        pull = min(0.95, 0.45 + 0.14 * r)
        _pull_local(spl_i, s0i, s1i, Ri, pull)
        _pull_local(spl_j, s0j, s1j, Rj, pull)
        pad = 0.06
        if _window_clear(spl_i, spl_j, bi, bj,
                         min(s0i, s0j) - pad, max(s1i, s1j) + pad, n=60):
            return spl_i, spl_j, True, r + 1
    return spl_i, spl_j, False, max_rounds


# ===========================================================================
# KURAMOTO  --  min-jerk base rate + repulsion lag, monotonic phase
# ===========================================================================
def run_kuramoto(arm_names: List[str], splines: Dict[str, Spline],
                 bases: Dict[str, np.ndarray], T_nom: float):
    """
    Phase-coupled timing.  Each arm's progress phi advances at the shared
    min-jerk rate (smootherstep') plus inter-arm coupling/repulsion that
    introduces lag near collisions.  Phase is clamped monotonic (>=0).
    Returns (sync_pos, phi_hist, t_vec, report).
    """
    N = len(arm_names)
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    max_steps = int(round(4.0 * T_nom / KUR_DT))   # stall guard
    phi = np.zeros(N)

    sync_acc = {n: [] for n in arm_names}
    phi_acc = []
    pdist_acc = {p: [] for p in pairs}
    pcoll_acc = {p: 0 for p in pairs}

    # precompute path samples for fast phase interpolation
    grid = np.linspace(0.0, 1.0, 400)
    pathP = {n: splines[n].eval(grid) for n in arm_names}

    def at(n, ph):
        ph = float(np.clip(ph, 0.0, 1.0))
        idx = min(int(ph * (len(grid) - 1)), len(grid) - 2)
        a = ph * (len(grid) - 1) - idx
        return pathP[n][idx] + a * (pathP[n][idx + 1] - pathP[n][idx])

    k = 0
    while k < max_steps:
        tau = (k * KUR_DT) / T_nom
        omega = smootherstep_d1(tau) / T_nom        # shared min-jerk rate
        q_now = {n: at(n, phi[idx]) for idx, n in enumerate(arm_names)}
        for idx, n in enumerate(arm_names):
            sync_acc[n].append(q_now[n].copy())
        phi_acc.append(phi.copy())

        ds = {}
        for (i, j) in pairs:
            d, col = pair_status(q_now[arm_names[i]], bases[arm_names[i]],
                                 q_now[arm_names[j]], bases[arm_names[j]])
            ds[(i, j)] = d
            pdist_acc[(i, j)].append(d)
            if col:
                pcoll_acc[(i, j)] += 1

        if np.all(phi >= 1.0 - 1e-9):
            break

        dp = np.zeros(N)
        for (i, j) in pairs:
            dist = ds[(i, j)]
            df = float(np.clip(1 - dist / REPULSE_D, 0, 1))
            danger = float(np.clip(1 - dist / MIN_SAFE, 0, 1))
            diff = phi[i] - phi[j]
            leader = i if diff > LEADER_THRESH else (j if diff < -LEADER_THRESH else -1)
            Kij = min(K_BASE * (1 + 4 * df), 15.0)
            dp[i] += Kij * float(np.sin(phi[j] - phi[i]))
            dp[j] += Kij * float(np.sin(phi[i] - phi[j]))
            if dist < REPULSE_D:
                mag = K_REPULSE * df**2 * 30 + (K_EMERGENCY * danger**3 if dist < MIN_SAFE else 0)
                if leader == i:   dp[i] -= mag * 2;   dp[j] -= mag * 0.3
                elif leader == j: dp[j] -= mag * 2;   dp[i] -= mag * 0.3
                else:             dp[i] -= mag * 0.7; dp[j] -= mag * 0.7

        # monotonic: never backward
        phi = phi + KUR_DT * np.clip(omega + dp, 0.0, KUR_RATE_MAX)
        k += 1

    # force exact target as the last sample
    for idx, n in enumerate(arm_names):
        sync_acc[n].append(splines[n].P[-1].copy())
    phi_acc.append(np.ones(N))
    for (i, j) in pairs:
        pdist_acc[(i, j)].append(
            pair_min_dist(splines[arm_names[i]].P[-1], bases[arm_names[i]],
                          splines[arm_names[j]].P[-1], bases[arm_names[j]]))

    sync = {n: np.array(sync_acc[n]) for n in arm_names}
    phi_hist = np.array(phi_acc)
    n_out = len(phi_hist)
    t_vec = np.linspace(0.0, max((n_out - 1) * KUR_DT, T_nom), n_out)

    rep = {}
    total_coll = 0
    for (i, j) in pairs:
        dv = np.array(pdist_acc[(i, j)])
        nc = pcoll_acc[(i, j)]; total_coll += nc
        rep[(arm_names[i], arm_names[j])] = {
            'min_dist_m': float(np.min(dv)), 'collisions': nc, 'collision_free': nc == 0}
    return sync, phi_hist, t_vec, {'pairs': rep, 'collision_free': total_coll == 0}


def path_collision_span(spl_i: Spline, spl_j: Spline, bi, bj, n: int = 200):
    """
    Synchronous sweep: both arms at the SAME parameter s in [0,1].  Returns the
    arc-fraction span [s_first, s_last] where the paths collide, or None.
    A span -> SPATIAL conflict (retract geometry).  None while the timed motion
    still collides -> purely TEMPORAL conflict (Kuramoto's job).
    """
    ss = np.linspace(0.0, 1.0, n)
    Pi = spl_i.eval(ss); Pj = spl_j.eval(ss)
    hits = [k for k in range(n) if pair_collides(Pi[k], bi, Pj[k], bj)]
    if not hits:
        return None
    return (float(ss[hits[0]]), float(ss[hits[-1]]))


def census(arm_names, sync_pos, phi_hist, bases):
    """Find colliding pairs and the per-arm path-arc SPAN (first..last colliding
    phase of each arm) over the synchronised motion."""
    out = {}
    for i in range(len(arm_names)):
        for j in range(i + 1, len(arm_names)):
            ni, nj = arm_names[i], arm_names[j]
            bi, bj = bases[ni], bases[nj]
            pi, pj = sync_pos[ni], sync_pos[nj]
            K = min(len(pi), len(pj), len(phi_hist))
            hits = [k for k in range(K) if pair_collides(pi[k], bi, pj[k], bj)]
            if hits:
                f, l = hits[0], hits[-1]
                out[(ni, nj)] = {
                    'first_k': f, 'n': len(hits),
                    'win_i': (float(phi_hist[f, i]), float(phi_hist[l, i])),
                    'win_j': (float(phi_hist[f, j]), float(phi_hist[l, j])),
                }
    return out


# ===========================================================================
# ORCHESTRATOR
# ===========================================================================
def plan_arms(starts: Dict[str, np.ndarray], targets: Dict[str, np.ndarray],
              bases: Dict[str, np.ndarray] = None, verbose: bool = True) -> Dict:
    """
    Full pipeline for an arbitrary set of arms.
      seed -> simultaneous check -> Kuramoto -> interior knot-insertion
      retraction for residual pairs -> re-time -> repeat.
    Returns a dict with per-arm Spline, sampled positions, duration, outcome.
    """
    if bases is None:
        bases = ROBOT_BASES
    names = sorted(starts.keys())
    splines = {n: seed_spline(normalize_joints(starts[n]), targets[n]) for n in names}

    T = max(duration_from_limits(starts[n], targets[n]) for n in names)
    T0 = T
    n_steps = max(2, int(round(T * RATE_HZ)))

    def log(*a):
        if verbose: print(*a)

    # 1. nominal min-jerk simultaneous check
    pos = {n: sample_minjerk(splines[n], n_steps) for n in names}
    if not _any_collision(names, pos, bases):
        log("  SAFE_NO_COLL  (T=%.2fs)" % T)
        return _package(names, splines, bases, T, "SAFE_NO_COLL", 0)

    # 2..N. Kuramoto + interior retraction, iterate
    boundary_pairs = set()
    for it in range(MAX_REFINE + 1):
        sync, phi_hist, t_vec, krep = run_kuramoto(names, splines, bases, T)
        if krep['collision_free']:
            log("  RESOLVED_KUR" if it == 0 else "  RESOLVED_CP_%d" % it)
            return _package(names, splines, bases, T,
                            "RESOLVED_KUR" if it == 0 else "RESOLVED_CP_%d" % it,
                            it, sync=sync, t_vec=t_vec)
        if it >= MAX_REFINE:
            break
        cens = census(names, sync, phi_hist, bases)
        if not cens:
            break
        ordered = sorted(cens.keys(), key=lambda p: cens[p]['first_k'])
        did_geo = False
        for (ni, nj) in ordered:
            # SPATIAL vs TEMPORAL: synchronous sweep on the current paths.
            span = path_collision_span(splines[ni], splines[nj], bases[ni], bases[nj])
            if span is None:
                # paths clear at equal s -> conflict is purely from timing;
                # leave it to Kuramoto (it will keep staggering next pass).
                log("  iter %d  %s<->%s  TEMPORAL only -> Kuramoto" % (it, ni, nj))
                continue
            wi = wj = span                      # synchronous -> same arc on both
            c = 0.5 * (span[0] + span[1])
            interior = BOUNDARY_LO <= c <= BOUNDARY_HI
            if not interior:
                wi = wj = (float(np.clip(span[0], 0.10, 0.90)),
                           float(np.clip(span[1], 0.10, 0.90)))
                boundary_pairs.add((ni, nj))
            tag = "interior" if interior else "boundary stage"
            spl_i, spl_j, ok, rr = resolve_pair_local(
                splines[ni], splines[nj], bases[ni], bases[nj], wi, wj)
            splines[ni], splines[nj] = spl_i, spl_j
            did_geo = True
            log("  iter %d  %s<->%s  span=[%.2f,%.2f] %s rounds=%d %s"
                % (it, ni, nj, span[0], span[1], tag, rr,
                   "cleared" if ok else "partial"))
        if not did_geo:
            # all residual conflicts are temporal; give Kuramoto a stronger pass
            # by lengthening the horizon (more room to stagger), then retry.
            if T < 2.5 * T0:
                T *= 1.20
                log("  all-temporal -> extend horizon T=%.2fs" % T)

    sync, phi_hist, t_vec, krep = run_kuramoto(names, splines, bases, T)
    outcome = "UNRESOLVED" if not krep['collision_free'] else "RESOLVED_CP_%d" % MAX_REFINE
    log("  %s  (boundary_pairs=%s)" % (outcome, sorted(boundary_pairs)))
    return _package(names, splines, bases, T, outcome, MAX_REFINE,
                    sync=sync, t_vec=t_vec, boundary=sorted(boundary_pairs))


def _any_collision(names, pos, bases) -> bool:
    K = min(len(pos[n]) for n in names)
    for k in range(K):
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if pair_collides(pos[names[i]][k], bases[names[i]],
                                 pos[names[j]][k], bases[names[j]]):
                    return True
    return False


def _package(names, splines, bases, T, outcome, rounds,
             sync=None, t_vec=None, boundary=None):
    n_steps = max(2, int(round(T * RATE_HZ)))
    if sync is None:
        sync = {n: sample_minjerk(splines[n], n_steps) for n in names}
        t_vec = np.linspace(0.0, T, n_steps)
    # resample every arm to a common step count on a shared clock
    Nmax = max(len(sync[n]) for n in names)
    pos = {}
    for n in names:
        p = sync[n]
        if len(p) != Nmax:
            s_in = np.linspace(0, 1, len(p)); s_out = np.linspace(0, 1, Nmax)
            r = np.zeros((Nmax, NDOF))
            for j in range(NDOF):
                r[:, j] = np.interp(s_out, s_in, p[:, j])
            p = r
        pos[n] = np.clip(p, POS_LIM[:, 0], POS_LIM[:, 1])
    dur = float(t_vec[-1]) if t_vec is not None else T

    # final duration scaling: ensure vel/accel limits hold (exclude forced
    # last sample, which is an instantaneous jump to the exact target).
    scale = 1.0
    for n in names:
        p = pos[n][:-1] if len(pos[n]) > 2 else pos[n]
        dt = dur / max(len(p) - 1, 1)
        vel = np.gradient(p, dt, axis=0); acc = np.gradient(vel, dt, axis=0)
        for j in range(NDOF):
            vp = float(np.max(np.abs(vel[:, j]))); ap = float(np.max(np.abs(acc[:, j])))
            if vp > VEL_LIM[j]: scale = max(scale, vp / VEL_LIM[j])
            if ap > ACC_LIM[j]: scale = max(scale, float(np.sqrt(ap / ACC_LIM[j])))
    dur *= scale * 1.05

    # final verification on the packaged (resampled) motion across all pairs
    Kp = Nmax
    resid = 0
    for k in range(Kp):
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                if pair_collides(pos[names[a]][k], bases[names[a]],
                                 pos[names[b]][k], bases[names[b]]):
                    resid += 1
                    break
    return {
        'outcome': outcome, 'rounds': rounds, 'duration': dur,
        'splines': splines, 'positions': pos,
        'time': np.linspace(0.0, dur, Nmax),
        'boundary_pairs': boundary or [],
        'residual_collision_steps': resid,
        'collision_free': resid == 0,
        'names': names, 'bases': bases,
    }


# ===========================================================================
# INVERSE KINEMATICS  (compact port of step_21 -- SE(3), multi-seed, scored)
# ===========================================================================
def quat_to_rot(q):
    q = np.asarray(q, float); n = np.linalg.norm(q)
    w, x, y, z = q / n if n > 1e-12 else np.array([1., 0., 0., 0.])
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
                     [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]])


def rot_err(Rg, Rt):
    R = Rt @ Rg.T
    return float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def _ik_seeds(current, tgt):
    px, py, pz = tgt
    cl = lambda q: np.clip(q, POS_LIM[:, 0], POS_LIM[:, 1])
    seeds = [cl(current.copy())]
    for j in range(NDOF):
        for d in (0.3, -0.3, 0.6, -0.6):
            s = current.copy(); s[j] += d; seeds.append(cl(s))
    t1 = float(np.arctan2(py, px)); rh = float(np.hypot(px, py))
    re = float(np.sqrt(max(rh**2 - A**2, 0.))); h = float(pz - L1)
    c3 = float(np.clip((re**2 + h**2 - L2**2 - L3**2) / (2*L2*L3), -1, 1))
    for sgn in (1., -1.):
        th3 = sgn * float(np.arccos(c3)); q3 = th3 - _PI_2
        th2 = float(np.arctan2(h, re)) - float(np.arctan2(L3*np.sin(th3), L2+L3*np.cos(th3)))
        q2 = th2 + _PI_2
        for q5 in (0., _PI_2, -_PI_2):
            for t in (t1, t1+_PI_2, t1-_PI_2):
                seeds.append(cl(np.array([t, q2, q3, 0., q5, 0.])))
    uniq = []
    for s in seeds:
        if all(np.linalg.norm(s - u) > 0.05 for u in uniq):
            uniq.append(s)
    return uniq


def solve_ik(tloc, trot, current, w_pos=1.0, w_rot=0.15,
             tol_pos=0.010, tol_rot=0.05, max_iter=600) -> List[np.ndarray]:
    bds = [(POS_LIM[i, 0], POS_LIM[i, 1]) for i in range(NDOF)]

    def obj(q):
        p, T = fk(q)
        return w_pos*float(np.sum((p-tloc)**2)) + w_rot*float(rot_err(T[:3, :3], trot)**2)

    sols = []
    for seed in _ik_seeds(current, tloc):
        res = minimize(obj, seed, method='SLSQP', bounds=bds,
                       options={'maxiter': max_iter, 'ftol': 1e-10})
        if not res.success:
            continue
        q = np.clip(res.x, POS_LIM[:, 0], POS_LIM[:, 1]); p, T = fk(q)
        if np.linalg.norm(p - tloc) >= tol_pos:
            continue
        if rot_err(T[:3, :3], trot) >= tol_rot:
            continue
        if all(np.linalg.norm(q - s) > 0.12 for s in sols):
            sols.append(q)
    return sols


def find_best_targets(arms: Dict, starts: Dict[str, np.ndarray]) -> Optional[Dict]:
    """
    arms[name] = {'base','tloc','trot'} ; returns {name: target_q} chosen to
    maximise minimum inter-arm clearance (clearance-primary, time-secondary).
    """
    names = sorted(arms.keys())
    allsol = {}; best = {}
    for n in names:
        sols = solve_ik(arms[n]['tloc'], arms[n]['trot'], starts[n])
        if not sols:                       # position-only fallback
            bds = [(POS_LIM[i, 0], POS_LIM[i, 1]) for i in range(NDOF)]
            tl = arms[n]['tloc']
            for seed in _ik_seeds(starts[n], tl):
                res = minimize(lambda q, t=tl: float(np.sum((fk(q)[0]-t)**2)),
                               seed, method='SLSQP', bounds=bds,
                               options={'maxiter': 600, 'ftol': 1e-10})
                q = np.clip(res.x, POS_LIM[:, 0], POS_LIM[:, 1])
                if np.linalg.norm(fk(q)[0]-tl) < 0.010:
                    sols.append(q); break
        if not sols:
            return None
        allsol[n] = sols
        best[n] = min(sols, key=lambda q, s=starts[n]: float(np.max(np.abs(q-s)/VEL_LIM)))
    for n in names:
        b = arms[n]['base']; sq = starts[n]
        others = [(best[m], arms[m]['base']) for m in names if m != n]

        def sc(q):
            clr = 1.0 if not others else min(
                float(np.tanh(pair_min_dist(q, b, oq, ob)/0.25)) for oq, ob in others)
            cost = float(np.sum(((q - sq)/VEL_LIM)**2))
            return 3.0*clr + 2.0*float(np.exp(-cost/25.0))
        best[n] = allsol[n][int(np.argmax([sc(q) for q in allsol[n]]))]
    return best