#!/usr/bin/env python3
"""
step_4.py  —  Kuramoto Sync + Alternating IK + B-Spline Refinement
═══════════════════════════════════════════════════════════════════════════════
Input  : trajectories.json + collision_report.json
Output : synchronized_trajectories.json

LOGIC
─────
Phase A — Kuramoto synchronization:
  Adjust phase (timing) of each arm so they don't arrive at the collision
  zone simultaneously. No shape change yet.

Phase B — if Kuramoto didn't resolve:
  For the colliding segment:
    1. Take alternating IK waypoints along that segment's arc
       arm-i maximizes distance from arm-j, arm-j maximizes distance from arm-i
    2. Rebuild B-spline with:
         - Original control points kept for all NON-colliding segments
         - Collision segment: new CPs from alternating IK
         - Neighbouring segments: add extra control points to smooth transition
         - Each retry increases CPs + segments near collision zone
    3. Re-run Kuramoto check
    4. Repeat until collision-free or MAX_REFINE exhausted
═══════════════════════════════════════════════════════════════════════════════
"""

import json, os, sys
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import BSpline

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

ROBOT_BASES: Dict[str, np.ndarray] = {
    'dsr01': np.array([0.0,  0.5, 0.0]),
    'dsr02': np.array([0.0, -0.5, 0.0]),
}

LINK_RADII    = np.array([0.10, 0.10, 0.08, 0.08, 0.06, 0.06])
SAFETY_MARGIN = 0.12
LINK_NAMES    = ['base', 'shoulder', 'upper_arm', 'forearm', 'wrist1', 'wrist2']

# Fixed base spline structure from step_2
N_SEG_BASE    = 5
N_CP_BASE     = 4
DEG           = 3

# Kuramoto
K_BASE      = 4.0                                         # for changing k values, change line 135 in step_6
K_REPULSE   = 80.0
K_EMERGENCY = 250.0
KUR_DT      = 0.01
MIN_SAFE    = 0.15
REPULSE_D   = 0.28
LEADER_THRESH = 0.05

# Refinement
MAX_REFINE      = 6
CP_INCREMENT    = 2    # extra CPs added per retry in collision + neighbour segments
SEG_INCREMENT   = 1    # extra segments added near collision per retry
IK_TOL_POS      = 0.015
IK_MAX_ITER     = 300


# ─────────────────────────────────────────────────────────────────────────────
# FK + GEOMETRY
# ─────────────────────────────────────────────────────────────────────────────

def fk(q):
    T = np.eye(4)
    for i in range(NDOF):
        al, a, to, d = DH[i]
        th = q[i] + to
        ct, st = np.cos(th), np.sin(th)
        ca, sa = np.cos(al),  np.sin(al)
        T = T @ np.array([
            [ct,    -st,    0.,  a    ],
            [st*ca,  ct*ca, -sa, -sa*d],
            [st*sa,  ct*sa,  ca,  ca*d],
            [0.,     0.,    0.,  1.   ],
        ])
    return T[:3, 3].copy(), T


def fk_pos(q, base):
    return fk(q)[0] + base


def link_origins(q, base):
    T = np.eye(4); o = np.zeros((NDOF, 3))
    for i in range(NDOF):
        al, a, to, d = DH[i]
        th = q[i] + to
        ct, st = np.cos(th), np.sin(th)
        ca, sa = np.cos(al),  np.sin(al)
        T = T @ np.array([
            [ct,    -st,    0.,  a    ],
            [st*ca,  ct*ca, -sa, -sa*d],
            [st*sa,  ct*sa,  ca,  ca*d],
            [0.,     0.,    0.,  1.   ],
        ])
        o[i] = T[:3, 3] + base
    return o


def pair_min_dist(qi, bi, qj, bj):
    """Minimum link-origin distance — vectorised broadcasting."""
    oi = link_origins(qi, bi)  # (NDOF,3)
    oj = link_origins(qj, bj)  # (NDOF,3)
    diff = oi[:, np.newaxis, :] - oj[np.newaxis, :, :]  # (NDOF,NDOF,3)
    return float(np.min(np.linalg.norm(diff, axis=2)))


def pair_collides(qi, bi, qj, bj):
    oi = link_origins(qi, bi); oj = link_origins(qj, bj)
    for a in range(NDOF):
        for b in range(NDOF):
            if np.linalg.norm(oi[a]-oj[b]) < LINK_RADII[a]+LINK_RADII[b]+SAFETY_MARGIN:
                return True
    return False


def interp_pos(pos, frac):
    frac = float(np.clip(frac, 0, 1)); n = len(pos)-1
    if n <= 0: return pos[0].copy()
    idx = min(int(frac*n), n-1); a = frac*n - idx
    return pos[idx] + a*(pos[idx+1]-pos[idx])


# ─────────────────────────────────────────────────────────────────────────────
# B-SPLINE UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def make_knots(ncp, deg=DEG):
    ni  = max(0, ncp-deg-1)
    inn = np.linspace(0,1,ni+2)[1:-1] if ni>0 else np.array([])
    return np.concatenate([np.zeros(deg+1), inn, np.ones(deg+1)])


def eval_seg(cp_j, s_loc, knots):
    """Evaluate one joint's B-spline over s_loc."""
    spl = BSpline(knots, cp_j, DEG, extrapolate=True)
    return spl(s_loc)


def fit_cp_to_waypoints(waypoints: np.ndarray,
                         s_vals   : np.ndarray,
                         ncp      : int,
                         pin_start: np.ndarray,
                         pin_end  : np.ndarray) -> np.ndarray:
    """
    Fit B-spline control points to waypoints via least squares.
    Pin first and last CP to pin_start / pin_end (boundary continuity).
    Returns cp of shape (ncp, NDOF).
    """
    ncp    = max(DEG + 2, ncp)
    knots  = make_knots(ncp)

    # Design matrix
    B = np.zeros((len(s_vals), ncp))
    e = np.zeros(ncp)
    for k in range(ncp):
        e[:] = 0.; e[k] = 1.
        v = BSpline(knots, e.copy(), DEG, extrapolate=False)(s_vals)
        B[:, k] = np.where(np.isfinite(v), v, 0.)

    cp = np.zeros((ncp, NDOF))
    for j in range(NDOF):
        th_s = float(pin_start[j]); th_e = float(pin_end[j])
        A_fr = B[:, 1:-1]
        rhs  = waypoints[:, j] - B[:, 0]*th_s - B[:, -1]*th_e
        if A_fr.shape[1] > 0:
            z, *_ = np.linalg.lstsq(A_fr, rhs, rcond=None)
            z = np.where(np.isfinite(z), z, np.linspace(th_s, th_e, len(z)))
            z = np.clip(z, POS_LIM[j, 0], POS_LIM[j, 1])
        else:
            z = np.array([])
        cp[:, j] = np.concatenate([[th_s], z if len(z) else [], [th_e]])
    return cp


def eval_full_trajectory(seg_cps: List[Tuple[np.ndarray, np.ndarray]],
                          duration: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Flatten per-segment CPs into one global CP sequence and evaluate
    as a SINGLE global clamped cubic B-spline.

    Segments share boundary CPs (de-duplicated when flattening).
    Single-spline evaluation eliminates stitching artifacts and oscillation.
    """
    # De-duplicate shared boundary CPs: keep first CP of each seg + last CP of last seg
    all_cp = []
    for seg_idx, (cp, _) in enumerate(seg_cps):
        if seg_idx == 0:
            all_cp.append(cp)          # keep all CPs of first segment
        else:
            all_cp.append(cp[1:])      # skip first CP (shared with previous seg)
    cp_global = np.vstack(all_cp)      # (total_unique, NDOF)

    n_global = len(cp_global)
    knots    = make_knots(n_global)
    n_steps  = max(2, int(round(duration * RATE_HZ)))
    s_full   = np.linspace(0.0, 1.0, n_steps)

    pos = np.zeros((n_steps, NDOF))
    vel = np.zeros((n_steps, NDOF))
    acc = np.zeros((n_steps, NDOF))

    for j in range(NDOF):
        spl       = BSpline(knots, cp_global[:, j], DEG, extrapolate=True)
        pos[:, j] = spl(s_full)
        vel[:, j] = spl.derivative(1)(s_full) / duration
        acc[:, j] = spl.derivative(2)(s_full) / duration**2

    pos = np.clip(pos, POS_LIM[:, 0], POS_LIM[:, 1])
    t   = np.linspace(0., duration, n_steps)
    return pos, vel, acc, t


def scale_if_needed(seg_cps, duration):
    pos, vel, acc, t = eval_full_trajectory(seg_cps, duration)
    sv = sa = 1.0
    for j in range(NDOF):
        vp = float(np.max(np.abs(vel[:,j])))
        ap = float(np.max(np.abs(acc[:,j])))
        if vp > VEL_LIM[j]: sv = max(sv, vp/VEL_LIM[j])
        if ap > ACC_LIM[j]: sa = max(sa, float(np.sqrt(ap/ACC_LIM[j])))
    scale = max(sv, sa)
    if scale > 1.0:
        duration = duration * scale * 1.05
        pos, vel, acc, t = eval_full_trajectory(seg_cps, duration)
    return pos, vel, acc, t, duration


# ─────────────────────────────────────────────────────────────────────────────
# LOAD SPLINE FROM TRAJECTORIES.JSON
# ─────────────────────────────────────────────────────────────────────────────

def load_seg_cps(traj_data: Dict) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Load per-segment control points from trajectories.json spline structure.
    Returns list of (cp (ncp,NDOF), knots) per segment.
    """
    segs  = traj_data['spline']['segments']
    n_cp  = int(traj_data['spline']['n_cp_seg'])
    knots = make_knots(n_cp)
    result = []
    for seg_info in segs:
        cp = np.array(seg_info['cp'], dtype=float)   # (n_cp, NDOF)
        result.append((cp, knots.copy()))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# PHASE A — KURAMOTO
# ─────────────────────────────────────────────────────────────────────────────

def run_kuramoto(arm_names, arm_pos, arm_bases, duration):
    N       = len(arm_names)
    n_steps = max(2, int(duration / KUR_DT))
    t_vec   = np.linspace(0., duration, n_steps)
    omega0  = 1.0 / duration
    phi = np.zeros(N); om = np.full(N, omega0)

    sync   = {n: np.zeros((n_steps, NDOF)) for n in arm_names}
    pairs  = [(i,j) for i in range(N) for j in range(i+1,N)]
    pdists = {p: np.zeros(n_steps) for p in pairs}

    for k in range(n_steps):
        phi = np.clip(phi, 0., 1.)
        q_now = {n: interp_pos(arm_pos[n], phi[idx])
                 for idx, n in enumerate(arm_names)}
        for idx, n in enumerate(arm_names): sync[n][k] = q_now[n]

        dists = {}
        for (i,j) in pairs:
            d = pair_min_dist(q_now[arm_names[i]], arm_bases[arm_names[i]],
                               q_now[arm_names[j]], arm_bases[arm_names[j]])
            dists[(i,j)] = d; pdists[(i,j)][k] = d

        dphi = np.zeros(N)
        for (i,j) in pairs:
            dist  = dists[(i,j)]
            df    = float(np.clip(1-dist/REPULSE_D, 0, 1))
            danger= float(np.clip(1-dist/MIN_SAFE, 0, 1))
            diff  = phi[i]-phi[j]
            leader= i if diff>LEADER_THRESH else (j if diff<-LEADER_THRESH else -1)
            Kij   = min(K_BASE*(1+4*df), 15.0)
            dphi[i] += Kij*float(np.sin(phi[j]-phi[i]))
            dphi[j] += Kij*float(np.sin(phi[i]-phi[j]))
            if dist < REPULSE_D:
                mag = K_REPULSE*df**2*30 + (K_EMERGENCY*danger**3 if dist<MIN_SAFE else 0)
                if   leader==i: dphi[i]-=mag*2; dphi[j]-=mag*0.3
                elif leader==j: dphi[j]-=mag*2; dphi[i]-=mag*0.3
                else:           dphi[i]-=mag*0.7; dphi[j]-=mag*0.7

        phi += KUR_DT * np.clip(om + dphi, -2., 2.)

    pair_rep = {}; total_crit = 0
    for (i,j) in pairs:
        dv = pdists[(i,j)]; nc = int(np.sum(dv < MIN_SAFE)); total_crit += nc
        pair_rep[f'{arm_names[i]}↔{arm_names[j]}'] = {
            'min_dist_m': float(np.min(dv)), 'critical': nc, 'collision_free': nc==0
        }
    return sync, t_vec, {'pair_reports': pair_rep,
                          'total_critical': total_crit,
                          'collision_free': total_crit==0}


# ─────────────────────────────────────────────────────────────────────────────
# CENSUS — WHICH SEGMENTS STILL COLLIDE POST-KURAMOTO
# ─────────────────────────────────────────────────────────────────────────────

def census(arm_names, sync_pos, arm_bases, t_vec, n_seg):
    """
    Returns dict: {(ni,nj): {'first_k', 'first_frac', 'first_seg', 'n_coll'}}
    """
    result = {}
    for i in range(len(arm_names)):
        for j in range(i+1, len(arm_names)):
            ni, nj = arm_names[i], arm_names[j]
            bi, bj = arm_bases[ni], arm_bases[nj]
            pi, pj = sync_pos[ni], sync_pos[nj]
            K = min(len(pi), len(pj), len(t_vec))
            first_k = -1; nc = 0
            for k in range(K):
                if pair_collides(pi[k], bi, pj[k], bj):
                    nc += 1
                    if first_k < 0: first_k = k
            if nc > 0:
                frac = first_k / max(K-1, 1)
                result[(ni,nj)] = {
                    'first_k'   : first_k,
                    'first_frac': frac,
                    'first_seg' : min(int(frac * n_seg), n_seg-1),
                    'n_coll'    : nc,
                }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# ALTERNATING IK — GENERATE WAYPOINTS THAT MAXIMIZE CLEARANCE
# ─────────────────────────────────────────────────────────────────────────────

def _ik_seeds(current, tgt_local):
    px, py, pz = tgt_local
    cl = lambda q: np.clip(q, POS_LIM[:,0], POS_LIM[:,1])
    # FIX: removed np.zeros(NDOF) seed — biases toward home config
    seeds = [cl(current.copy())]
    for j in range(NDOF):
        for d in (0.3,-0.3,0.6,-0.6):
            s=current.copy(); s[j]+=d; seeds.append(cl(s))
    t1=float(np.arctan2(py,px)); rh=float(np.hypot(px,py))
    re=float(np.sqrt(max(rh**2-A**2,0.))); h=float(pz-L1)
    c3=float(np.clip((re**2+h**2-L2**2-L3**2)/(2*L2*L3),-1,1))
    for sgn in(1.,-1.):
        th3=sgn*float(np.arccos(c3)); q3=th3-_PI_2
        th2=float(np.arctan2(h,re))-float(np.arctan2(L3*np.sin(th3),L2+L3*np.cos(th3)))
        q2=th2+_PI_2
        for q5 in(0.,_PI_2,-_PI_2):
            for t in(t1,t1+_PI_2,t1-_PI_2): seeds.append(cl(np.array([t,q2,q3,0.,q5,0.])))
    uniq=[]
    for s in seeds:
        if all(np.linalg.norm(s-u)>0.05 for u in uniq): uniq.append(s)
    return uniq


def ik_away_multi(tgt_local:  np.ndarray,
                  current_q:  np.ndarray,
                  this_base:  np.ndarray,
                  all_others: List[Tuple[np.ndarray, np.ndarray]]
                  ) -> Optional[np.ndarray]:
    """
    Position-only IK maximising minimum clearance from ALL other arms.

    FIX: original ik_away() only avoided a single arm (the pair partner).
    When arm01 collides with arm02 AND arm03, resolving arm01↔arm02
    could move arm01 into arm03.  This function takes all_others =
    [(q_k, base_k), ...] for every other arm and selects the config that
    maximises the WORST-CASE clearance to any arm:
        argmax_q  min_k( pair_min_dist(q, this_base, q_k, base_k) )
    """
    bds = [(POS_LIM[i, 0], POS_LIM[i, 1]) for i in range(NDOF)]
    valid: List[np.ndarray] = []
    for seed in _ik_seeds(current_q, tgt_local):
        def obj(q, _t=tgt_local):
            p, _ = fk(q)
            return float(np.sum((p - _t) ** 2))
        res = minimize(obj, seed, method='SLSQP', bounds=bds,
                       options={'maxiter': IK_MAX_ITER, 'ftol': 1e-9})
        if not res.success:
            continue
        q = np.clip(res.x, POS_LIM[:, 0], POS_LIM[:, 1])
        if np.linalg.norm(fk(q)[0] - tgt_local) < IK_TOL_POS:
            valid.append(q)
    if not valid:
        return None
    def _min_clear(q):
        if not all_others:
            return 1.0
        return min(pair_min_dist(q, this_base, oq, ob) for oq, ob in all_others)
    return max(valid, key=_min_clear)


def alternating_ik_for_segment(arm_names:     List[str],
                                 ni:            str,
                                 nj:            str,
                                 arm_pos:       Dict[str, np.ndarray],
                                 arm_arc:       Dict[str, np.ndarray],
                                 arm_bases:     Dict[str, np.ndarray],
                                 seg_arc_start: float,
                                 seg_arc_end:   float,
                                 n_waypoints:   int
                                 ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Sample n_waypoints along [seg_arc_start, seg_arc_end].

    FIX: N-arm-aware. Each arm's IK now avoids ALL other arms, not just the
    single pair partner:
      - arm_i: avoids arm_j (current guess) + every bystander arm
      - arm_j: avoids UPDATED arm_i         + every bystander arm
    This prevents a config chosen to avoid arm_j from colliding with arm_k.
    """
    pos_i   = arm_pos[ni];   pos_j   = arm_pos[nj]
    base_i  = arm_bases[ni]; base_j  = arm_bases[nj]
    arc_i   = arm_arc[ni];   N_j     = len(pos_j)
    other_names = [n for n in arm_names if n != ni and n != nj]

    mask = (arc_i >= seg_arc_start - 1e-6) & (arc_i <= seg_arc_end + 1e-6)
    idx  = np.where(mask)[0]
    if len(idx) < 2:
        return None, None

    sample_idx = np.unique(np.linspace(0, len(idx) - 1, n_waypoints, dtype=int))
    wps_i: List[np.ndarray] = []
    wps_j: List[np.ndarray] = []
    prev_qi = pos_i[idx[0]].copy()
    prev_qj = pos_j[min(idx[0], N_j - 1)].copy()

    for si in sample_idx:
        k  = idx[si]
        kj = min(k, N_j - 1)
        ee_i = fk(pos_i[k])[0]
        ee_j = fk(pos_j[kj])[0]

        # arm-i: avoid arm-j (current guess) + all bystander arms at step k
        others_i = [(prev_qj, base_j)] + [
            (arm_pos[n][min(k, len(arm_pos[n]) - 1)], arm_bases[n])
            for n in other_names
        ]
        qi_new = ik_away_multi(ee_i, prev_qi, base_i, others_i)
        if qi_new is None:
            qi_new = pos_i[k].copy()

        # arm-j: avoid UPDATED arm-i + all bystander arms at step k
        others_j = [(qi_new, base_i)] + [
            (arm_pos[n][min(k, len(arm_pos[n]) - 1)], arm_bases[n])
            for n in other_names
        ]
        qj_new = ik_away_multi(ee_j, prev_qj, base_j, others_j)
        if qj_new is None:
            qj_new = pos_j[kj].copy()

        wps_i.append(qi_new)
        wps_j.append(qj_new)
        prev_qi = qi_new
        prev_qj = qj_new

    return np.array(wps_i), np.array(wps_j)


# ─────────────────────────────────────────────────────────────────────────────
# REBUILD B-SPLINE: OLD CPs KEPT + COLLISION SEG REPLACED + NEIGHBOURS ENRICHED
# ─────────────────────────────────────────────────────────────────────────────

def rebuild_spline(orig_seg_cps : List[Tuple[np.ndarray, np.ndarray]],
                    pos_full    : np.ndarray,
                    arc_full    : np.ndarray,
                    coll_seg    : int,
                    new_wps     : np.ndarray,
                    refine_count: int,
                    n_seg_base  : int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Rebuild the per-segment CP list:
      - Non-collision, non-neighbour segments: keep original CPs unchanged
      - Collision segment: fit new CPs from alternating IK waypoints
        with N_CP_BASE + CP_INCREMENT × refine_count CPs
      - Neighbour segments (±1): add CP_INCREMENT extra CPs to smooth transition

    Returns updated list of (cp, knots) per segment.
    """
    n_seg   = len(orig_seg_cps)
    result  = list(orig_seg_cps)   # start with all original

    extra_cp  = CP_INCREMENT * (refine_count + 1)
    n_cp_coll = N_CP_BASE + extra_cp
    n_cp_nbr  = N_CP_BASE + max(0, extra_cp - CP_INCREMENT)

    neighbours = [coll_seg - 1, coll_seg + 1]

    for seg in range(n_seg):
        s0   = seg / n_seg; s1 = (seg+1) / n_seg
        mask = (arc_full >= s0-1e-9) & (arc_full <= s1+1e-9)
        idx  = np.where(mask)[0]
        if len(idx) < 2: continue

        # Boundary pins — ensure continuity with adjacent segments
        pin_s = pos_full[idx[0]].copy()
        pin_e = pos_full[idx[-1]].copy()

        if seg == coll_seg:
            # Replace with alternating IK waypoints
            s_wps = np.linspace(0., 1., len(new_wps))
            cp    = fit_cp_to_waypoints(new_wps, s_wps, n_cp_coll, pin_s, pin_e)
            result[seg] = (cp, make_knots(n_cp_coll))

        elif seg in neighbours and 0 <= seg < n_seg:
            # Enrich neighbours with extra CPs (smooth transition)
            s_loc = np.clip((arc_full[idx]-s0)/(s1-s0), 0., 1.)
            wps   = pos_full[idx]
            cp    = fit_cp_to_waypoints(wps, s_loc, n_cp_nbr, pin_s, pin_e)
            result[seg] = (cp, make_knots(n_cp_nbr))
        # else: keep original

    return result


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def synchronise(trajectories: Dict, arm_bases: Dict = None) -> Dict:
    if arm_bases is None: arm_bases = ROBOT_BASES
    arm_names = sorted(trajectories.keys())
    N = len(arm_names)
    n_seg = N_SEG_BASE

    print(f'\n  Arms: {arm_names}  |  {N*(N-1)//2} pair(s)  |  {n_seg} segments')

    # Load spline CPs per arm
    arm_seg_cps: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {
        n: load_seg_cps(trajectories[n]) for n in arm_names
    }
    duration = max(float(trajectories[n]['metadata']['duration']) for n in arm_names)

    # Build pos arrays from spline CPs
    def make_pos(name):
        pos, _, _, _ = eval_full_trajectory(arm_seg_cps[name], duration)
        return pos

    arm_pos = {n: make_pos(n) for n in arm_names}
    arm_arc = {n: np.linspace(0., 1., len(arm_pos[n])) for n in arm_names}

    history = []
    sync_pos = {}; t_vec = np.array([]); kur_rep = {}

    for iteration in range(MAX_REFINE + 1):
        print(f'\n  {"─"*66}')
        print(f'  Iter {iteration}  |  Kuramoto  (dur={duration:.2f}s)')

        sync_pos, t_vec, kur_rep = run_kuramoto(
            arm_names, arm_pos, arm_bases, duration)

        for pk, pr in kur_rep['pair_reports'].items():
            icon = '✅' if pr['collision_free'] else '❌'
            print(f'    {icon} {pk:<22}  '
                  f'min={pr["min_dist_m"]*100:.1f}cm  crit={pr["critical"]}')

        history.append({'iteration': iteration, 'kuramoto': kur_rep})

        if kur_rep['collision_free']:
            print(f'\n  ✅  Collision-free at iteration {iteration}')
            break

        if iteration >= MAX_REFINE:
            print(f'\n  ⚠   MAX_REFINE={MAX_REFINE} reached')
            break

        # Census
        cens = census(arm_names, sync_pos, arm_bases, t_vec, n_seg)
        if not cens: break

        # FIX C: process ALL colliding pairs per iteration ordered by urgency,
        # and for each pair pass all arm positions so each arm avoids ALL others.
        pairs_ordered = sorted(cens.keys(), key=lambda p: cens[p]['first_frac'])
        n_wp = N_CP_BASE + CP_INCREMENT * (iteration + 1)

        for pair_key in pairs_ordered:
            info     = cens[pair_key]
            ni, nj   = pair_key
            coll_seg = info['first_seg']
            bi, bj   = arm_bases[ni], arm_bases[nj]
            s0_c = coll_seg / n_seg
            s1_c = (coll_seg + 1) / n_seg

            print(f'\n  Phase B: {ni}↔{nj}  '
                  f'coll_seg={coll_seg} arc=[{s0_c:.2f},{s1_c:.2f}]  '
                  f'n_wp={n_wp}  iter={iteration}')

            # Alternating IK — all arm positions passed; each arm avoids ALL others
            wps_i, wps_j = alternating_ik_for_segment(
                arm_names, ni, nj,
                arm_pos, arm_arc, arm_bases,
                s0_c, s1_c, n_wp)

            if wps_i is not None:
                print(f'    Alternating IK: {len(wps_i)} waypoints generated')
                arm_seg_cps[ni] = rebuild_spline(
                    arm_seg_cps[ni], arm_pos[ni], arm_arc[ni],
                    coll_seg, wps_i, iteration, n_seg)
                arm_seg_cps[nj] = rebuild_spline(
                    arm_seg_cps[nj], arm_pos[nj], arm_arc[nj],
                    coll_seg, wps_j, iteration, n_seg)
                pos_i, _, _, _ = eval_full_trajectory(arm_seg_cps[ni], duration)
                pos_j, _, _, _ = eval_full_trajectory(arm_seg_cps[nj], duration)
                arm_pos[ni] = pos_i; arm_arc[ni] = np.linspace(0,1,len(pos_i))
                arm_pos[nj] = pos_j; arm_arc[nj] = np.linspace(0,1,len(pos_j))
                print(f'    B-spline rebuilt with updated control points')
            else:
                print(f'    ⚠  {ni}↔{nj}: alternating IK produced no valid configs — skipping')

    # ── Assemble output ────────────────────────────────────────────────────────
    out = {}
    for name in arm_names:
        pos_f, vel_f, acc_f, t_f, dur_f = scale_if_needed(arm_seg_cps[name], duration)
        n_seg_arm = len(arm_seg_cps[name])
        seg_info  = []
        for seg, (cp, kn) in enumerate(arm_seg_cps[name]):
            seg_info.append({
                'segment'  : seg,
                'arc_start': round(seg/n_seg_arm, 4),
                'arc_end'  : round((seg+1)/n_seg_arm, 4),
                'n_cp'     : len(cp),
                'cp'       : cp.tolist(),
            })
        out[name] = {
            'robot_name': name,
            'metadata'  : {
                **trajectories[name]['metadata'],
                'duration'        : float(dur_f),
                'n_samples'       : len(pos_f),
                'refine_iterations': iteration,
            },
            'spline': {
                'n_seg'   : n_seg_arm,
                'degree'  : DEG,
                'segments': seg_info,
            },
            'trajectory': {
                'time'        : t_f.tolist(),
                'positions'   : pos_f.tolist(),
                'velocities'  : vel_f.tolist(),
                'accelerations': acc_f.tolist(),
                'n_samples'   : len(pos_f),
            },
        }

    out['synchronisation_report'] = kur_rep
    out['refinement_history']     = history
    out['parameters'] = {
        'k_base': K_BASE, 'k_repulse': K_REPULSE, 'min_safe_dist': MIN_SAFE,
        'max_refine': MAX_REFINE, 'cp_increment': CP_INCREMENT,
        'seg_increment': SEG_INCREMENT,
    }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print('\n' + '=' * 68)
    print('  STEP 4  —  Kuramoto + Alternating IK + B-Spline Refinement')
    print('=' * 68)

    for fname, step in [('trajectories.json','step_2'), ('collision_report.json','step_3')]:
        if not os.path.exists(fname):
            print(f'\n  ❌  {fname} not found — run {step} first'); sys.exit(1)

    with open('trajectories.json')     as fh: tdata = json.load(fh)
    with open('collision_report.json') as fh: crep  = json.load(fh)

    active = sorted([k for k in tdata if k.startswith('dsr')])
    if not active: print('\n  ❌  No arm data'); sys.exit(1)

    print(f'\n  Active arms   : {active}')
    print(f'  Step-3 status : {crep["overall_status"]}')

    result = synchronise(
        {n: tdata[n] for n in active},
        {n: ROBOT_BASES[n] for n in active},
    )

    with open('synchronized_trajectories.json', 'w') as fh:
        json.dump(result, fh, indent=2)

    safe = result['synchronisation_report'].get('collision_free', False)
    kb   = os.path.getsize('synchronized_trajectories.json') / 1024.0
    print(f'\n  {"✅" if safe else "⚠ "}  Saved: synchronized_trajectories.json  ({kb:.1f} KB)')
    print('  Next  →  python3 step_5.py\n')


if __name__ == '__main__':
    main()