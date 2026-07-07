#!/usr/bin/env python3
"""
step_14.py  —  Kuramoto Sync + Alternating IK + B-Spline Refinement  (4-arm)
═══════════════════════════════════════════════════════════════════════════════
Input  : trajectories.json + collision_report.json
Output : synchronized_trajectories.json

Handles 4 arms (dsr01–dsr04) and all C(4,2)=6 pairs.

LOGIC  (identical to step_4, generalised)
─────
Phase A — Kuramoto synchronization over all 6 pairs simultaneously.
Phase B — For remaining collisions, per-pair alternating IK + B-spline CP
           refinement.  Each colliding pair is processed in priority order
           (earliest first-collision first).  The clearance IK maximises
           distance from ALL other arms, not just the colliding partner.
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
    'dsr03': np.array([1.0,  0.5, 0.0]),
    'dsr04': np.array([1.0, -0.5, 0.0]),
}

LINK_RADII    = np.array([0.10, 0.10, 0.08, 0.08, 0.06, 0.06])
SAFETY_MARGIN = 0.12

N_SEG_BASE = 5
N_CP_BASE  = 4
DEG        = 3

# Kuramoto
K_BASE      = 5.0             # for changing k values, change line 96 in step_16
K_REPULSE   = 80.0
K_EMERGENCY = 250.0
KUR_DT      = 0.01
MIN_SAFE    = 0.15
REPULSE_D   = 0.28
LEADER_THRESH = 0.05

# Refinement
MAX_REFINE   = 6
CP_INCREMENT = 2
SEG_INCREMENT = 1
IK_TOL_POS   = 0.015
IK_MAX_ITER  = 300

# ─────────────────────────────────────────────────────────────────────────────
# FK + GEOMETRY
# ─────────────────────────────────────────────────────────────────────────────

def fk(q):
    T = np.eye(4)
    for i in range(NDOF):
        al, a, to, d = DH[i]; th = q[i] + to
        ct, st = np.cos(th), np.sin(th); ca, sa = np.cos(al), np.sin(al)
        T = T @ np.array([[ct,-st,0.,a],[st*ca,ct*ca,-sa,-sa*d],
                           [st*sa,ct*sa,ca,ca*d],[0.,0.,0.,1.]])
    return T[:3, 3].copy(), T

def link_origins(q, base):
    T = np.eye(4); o = np.zeros((NDOF, 3))
    for i in range(NDOF):
        al, a, to, d = DH[i]; th = q[i] + to
        ct, st = np.cos(th), np.sin(th); ca, sa = np.cos(al), np.sin(al)
        T = T @ np.array([[ct,-st,0.,a],[st*ca,ct*ca,-sa,-sa*d],
                           [st*sa,ct*sa,ca,ca*d],[0.,0.,0.,1.]])
        o[i] = T[:3, 3] + base
    return o

def arm_min_dist(qi, bi, qj, bj):
    oi = link_origins(qi, bi); oj = link_origins(qj, bj)
    return float(np.min([np.linalg.norm(oi[a]-oj[b])
                         for a in range(NDOF) for b in range(NDOF)]))

def min_dist_to_all(q, base, others: List[Tuple[np.ndarray, np.ndarray]]) -> float:
    """Minimum distance from arm (q,base) to ALL other arms."""
    if not others: return 1.0
    return min(arm_min_dist(q, base, oq, ob) for oq, ob in others)

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

def fit_cp_to_waypoints(waypoints, s_vals, ncp, pin_start, pin_end):
    ncp   = max(DEG+2, ncp); knots = make_knots(ncp)
    B     = np.zeros((len(s_vals), ncp)); e = np.zeros(ncp)
    for k in range(ncp):
        e[:] = 0.; e[k] = 1.
        v = BSpline(knots, e.copy(), DEG, extrapolate=False)(s_vals)
        B[:, k] = np.where(np.isfinite(v), v, 0.)
    cp = np.zeros((ncp, NDOF))
    for j in range(NDOF):
        ts = float(pin_start[j]); te = float(pin_end[j])
        Af = B[:, 1:-1]; rhs = waypoints[:, j] - B[:,0]*ts - B[:,-1]*te
        if Af.shape[1] > 0:
            z, *_ = np.linalg.lstsq(Af, rhs, rcond=None)
            z = np.where(np.isfinite(z), z, np.linspace(ts,te,len(z)))
            z = np.clip(z, POS_LIM[j,0], POS_LIM[j,1])
        else:
            z = np.array([])
        cp[:, j] = np.concatenate([[ts], z if len(z) else [], [te]])
    return cp

def eval_full_trajectory(seg_cps, duration):
    all_cp = []
    for seg_idx, (cp, _) in enumerate(seg_cps):
        all_cp.append(cp if seg_idx == 0 else cp[1:])
    cp_global = np.vstack(all_cp)
    n_global  = len(cp_global); knots = make_knots(n_global)
    n_steps   = max(2, int(round(duration * RATE_HZ)))
    s_full    = np.linspace(0.0, 1.0, n_steps)
    pos = np.zeros((n_steps, NDOF)); vel = np.zeros_like(pos); acc = np.zeros_like(pos)
    for j in range(NDOF):
        spl       = BSpline(knots, cp_global[:,j], DEG, extrapolate=True)
        pos[:,j]  = spl(s_full)
        vel[:,j]  = spl.derivative(1)(s_full) / duration
        acc[:,j]  = spl.derivative(2)(s_full) / duration**2
    pos = np.clip(pos, POS_LIM[:,0], POS_LIM[:,1])
    t   = np.linspace(0., duration, n_steps)
    return pos, vel, acc, t

def scale_if_needed(seg_cps, duration):
    pos, vel, acc, t = eval_full_trajectory(seg_cps, duration)
    sv = sa = 1.0
    for j in range(NDOF):
        vp = float(np.max(np.abs(vel[:,j]))); ap = float(np.max(np.abs(acc[:,j])))
        if vp > VEL_LIM[j]: sv = max(sv, vp/VEL_LIM[j])
        if ap > ACC_LIM[j]: sa = max(sa, float(np.sqrt(ap/ACC_LIM[j])))
    sc = max(sv, sa)
    if sc > 1.0:
        duration = duration * sc * 1.05
        pos, vel, acc, t = eval_full_trajectory(seg_cps, duration)
    return pos, vel, acc, t, duration

def load_seg_cps(traj_data):
    segs  = traj_data['spline']['segments']
    n_cp  = int(traj_data['spline']['n_cp_seg'])
    knots = make_knots(n_cp)
    return [(np.array(seg_info['cp'], dtype=float), knots.copy()) for seg_info in segs]

# ─────────────────────────────────────────────────────────────────────────────
# PHASE A — KURAMOTO  (N-arm generalised)
# ─────────────────────────────────────────────────────────────────────────────

def run_kuramoto(arm_names, arm_pos, arm_bases, duration):
    N       = len(arm_names)
    n_steps = max(2, int(duration / KUR_DT))
    omega0  = 1.0 / duration
    phi     = np.zeros(N); om = np.full(N, omega0)
    pairs   = [(i,j) for i in range(N) for j in range(i+1,N)]
    pdists  = {p: np.zeros(n_steps) for p in pairs}
    sync    = {n: np.zeros((n_steps, NDOF)) for n in arm_names}

    for k in range(n_steps):
        phi = np.clip(phi, 0., 1.)
        qn  = {n: interp_pos(arm_pos[n], phi[idx]) for idx,n in enumerate(arm_names)}
        for idx, n in enumerate(arm_names): sync[n][k] = qn[n]
        ds = {}
        for (i,j) in pairs:
            d = arm_min_dist(qn[arm_names[i]], arm_bases[arm_names[i]],
                              qn[arm_names[j]], arm_bases[arm_names[j]])
            ds[(i,j)] = d; pdists[(i,j)][k] = d
        dp = np.zeros(N)
        for (i,j) in pairs:
            dist = ds[(i,j)]; df = float(np.clip(1-dist/REPULSE_D,0,1))
            danger = float(np.clip(1-dist/MIN_SAFE,0,1))
            diff   = phi[i]-phi[j]
            leader = i if diff>LEADER_THRESH else (j if diff<-LEADER_THRESH else -1)
            Kij    = min(K_BASE*(1+4*df), 15.0)
            dp[i] += Kij*float(np.sin(phi[j]-phi[i]))
            dp[j] += Kij*float(np.sin(phi[i]-phi[j]))
            if dist < REPULSE_D:
                mag = K_REPULSE*df**2*30 + (K_EMERGENCY*danger**3 if dist<MIN_SAFE else 0)
                if   leader==i: dp[i]-=mag*2; dp[j]-=mag*0.3
                elif leader==j: dp[j]-=mag*2; dp[i]-=mag*0.3
                else:           dp[i]-=mag*0.7; dp[j]-=mag*0.7
        phi += KUR_DT * np.clip(om + dp, -2., 2.)

    pr = {}; tc = 0
    for (i,j) in pairs:
        dv = pdists[(i,j)]; nc = int(np.sum(dv < MIN_SAFE)); tc += nc
        pr[f'{arm_names[i]}↔{arm_names[j]}'] = {
            'min_dist_m': float(np.min(dv)), 'critical': nc, 'collision_free': nc==0
        }
    return sync, {'pair_reports': pr, 'total_critical': tc, 'collision_free': tc==0}

# ─────────────────────────────────────────────────────────────────────────────
# CENSUS — COLLIDING PAIRS POST-KURAMOTO
# ─────────────────────────────────────────────────────────────────────────────

def census(arm_names, sync_pos, arm_bases, t_vec, n_seg):
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
                    'first_k': first_k, 'first_frac': frac,
                    'first_seg': min(int(frac * n_seg), n_seg-1),
                    'n_coll': nc,
                }
    return result

# ─────────────────────────────────────────────────────────────────────────────
# ALTERNATING IK — CLEARANCE FROM ALL OTHER ARMS
# ─────────────────────────────────────────────────────────────────────────────

def _ik_seeds(current, tgt_local):
    px, py, pz = tgt_local
    cl = lambda q: np.clip(q, POS_LIM[:,0], POS_LIM[:,1])
    # FIX: removed np.zeros(NDOF) seed — biases solver to home config;
    # always seed from robot's actual current state instead.
    seeds = [cl(current.copy())]
    for j in range(NDOF):
        for d in (0.3,-0.3,0.6,-0.6):
            s=current.copy(); s[j]+=d; seeds.append(cl(s))
    t1=float(np.arctan2(py,px)); rh=float(np.hypot(px,py))
    re=float(np.sqrt(max(rh**2-A**2,0.))); h=float(pz-L1)
    c3=float(np.clip((re**2+h**2-L2**2-L3**2)/(2*L2*L3),-1,1))
    for sgn in (1.,-1.):
        th3=sgn*float(np.arccos(c3)); q3=th3-_PI_2
        th2=float(np.arctan2(h,re))-float(np.arctan2(L3*np.sin(th3),L2+L3*np.cos(th3)))
        q2=th2+_PI_2
        for q5 in (0.,_PI_2,-_PI_2):
            for t in (t1,t1+_PI_2,t1-_PI_2): seeds.append(cl(np.array([t,q2,q3,0.,q5,0.])))
    uniq = []
    for s in seeds:
        if all(np.linalg.norm(s-u)>0.05 for u in uniq): uniq.append(s)
    return uniq

def ik_away_multi(tgt_local, current_q, this_base,
                  all_others: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[np.ndarray]:
    """
    Position-only IK that maximises minimum clearance from ALL other arms.
    `all_others` = list of (q, base) for every arm that is NOT this one.
    """
    bds = [(POS_LIM[i,0], POS_LIM[i,1]) for i in range(NDOF)]
    valid = []
    for seed in _ik_seeds(current_q, tgt_local):
        def obj(q, _t=tgt_local):
            p, _ = fk(q); return float(np.sum((p - _t)**2))
        res = minimize(obj, seed, method='SLSQP', bounds=bds,
                       options={'maxiter': IK_MAX_ITER, 'ftol': 1e-9})
        if not res.success: continue
        q = np.clip(res.x, POS_LIM[:,0], POS_LIM[:,1])
        if np.linalg.norm(fk(q)[0] - tgt_local) < IK_TOL_POS:
            valid.append(q)
    if not valid: return None
    return max(valid, key=lambda q: min_dist_to_all(q, this_base, all_others))

def alternating_ik_for_pair(pos_i, pos_j, arc_i,
                              base_i, base_j,
                              all_arm_pos, all_arm_bases, arm_names_excl_pair,
                              seg_arc_start, seg_arc_end,
                              n_waypoints) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Alternating IK for the colliding pair (i, j).
    Each arm's IK maximises clearance from ALL other arms (not just the partner).

    arm_names_excl_pair: list of names of arms other than ni and nj
    all_arm_pos / all_arm_bases: dicts for all arms
    """
    N_j  = len(pos_j)
    mask = (arc_i >= seg_arc_start - 1e-6) & (arc_i <= seg_arc_end + 1e-6)
    idx  = np.where(mask)[0]
    if len(idx) < 2: return None, None

    sample_idx = np.unique(np.linspace(0, len(idx)-1, n_waypoints, dtype=int))
    wps_i = []; wps_j = []; prev_qi = pos_i[idx[0]].copy(); prev_qj = pos_j[min(idx[0],N_j-1)].copy()

    for si in sample_idx:
        k = idx[si]; kj = min(k, N_j-1)
        ee_i = fk(pos_i[k])[0]; ee_j = fk(pos_j[kj])[0]

        # Others for arm i = arm j (current) + any other arms
        others_for_i = [(prev_qj, base_j)] + [
            (all_arm_pos[n][min(k, len(all_arm_pos[n])-1)], all_arm_bases[n])
            for n in arm_names_excl_pair
        ]
        qi_new = ik_away_multi(ee_i, prev_qi, base_i, others_for_i)
        if qi_new is None: qi_new = pos_i[k].copy()

        # Others for arm j = updated arm i + any other arms
        others_for_j = [(qi_new, base_i)] + [
            (all_arm_pos[n][min(k, len(all_arm_pos[n])-1)], all_arm_bases[n])
            for n in arm_names_excl_pair
        ]
        qj_new = ik_away_multi(ee_j, prev_qj, base_j, others_for_j)
        if qj_new is None: qj_new = pos_j[kj].copy()

        wps_i.append(qi_new); wps_j.append(qj_new)
        prev_qi = qi_new; prev_qj = qj_new

    return np.array(wps_i), np.array(wps_j)

# ─────────────────────────────────────────────────────────────────────────────
# REBUILD B-SPLINE
# ─────────────────────────────────────────────────────────────────────────────

def rebuild_spline(orig_seg_cps, pos_full, arc_full, coll_seg, new_wps, refine_count, n_seg):
    result    = list(orig_seg_cps)
    extra_cp  = CP_INCREMENT * (refine_count + 1)
    n_cp_coll = N_CP_BASE + extra_cp
    n_cp_nbr  = N_CP_BASE + max(0, extra_cp - CP_INCREMENT)
    neighbours = [coll_seg - 1, coll_seg + 1]
    for seg in range(n_seg):
        s0 = seg/n_seg; s1 = (seg+1)/n_seg
        mask = (arc_full >= s0-1e-9) & (arc_full <= s1+1e-9)
        idx  = np.where(mask)[0]
        if len(idx) < 2: continue
        pin_s = pos_full[idx[0]].copy(); pin_e = pos_full[idx[-1]].copy()
        if seg == coll_seg:
            s_wps = np.linspace(0.,1.,len(new_wps))
            cp    = fit_cp_to_waypoints(new_wps, s_wps, n_cp_coll, pin_s, pin_e)
            result[seg] = (cp, make_knots(n_cp_coll))
        elif seg in neighbours and 0 <= seg < n_seg:
            s_loc = np.clip((arc_full[idx]-s0)/(s1-s0), 0., 1.)
            cp    = fit_cp_to_waypoints(pos_full[idx], s_loc, n_cp_nbr, pin_s, pin_e)
            result[seg] = (cp, make_knots(n_cp_nbr))
    return result

# ─────────────────────────────────────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def synchronise(trajectories: Dict, arm_bases: Dict = None) -> Dict:
    if arm_bases is None: arm_bases = ROBOT_BASES
    arm_names = sorted(trajectories.keys())
    N         = len(arm_names)
    n_seg     = N_SEG_BASE
    n_pairs   = N * (N-1) // 2

    print(f'\n  Arms: {arm_names}  |  {n_pairs} pair(s)  |  {n_seg} segments')

    arm_seg_cps = {n: load_seg_cps(trajectories[n]) for n in arm_names}
    duration    = max(float(trajectories[n]['metadata']['duration']) for n in arm_names)

    def make_pos(name):
        pos, _, _, _ = eval_full_trajectory(arm_seg_cps[name], duration); return pos

    arm_pos = {n: make_pos(n) for n in arm_names}
    arm_arc = {n: np.linspace(0., 1., len(arm_pos[n])) for n in arm_names}

    history = []; sync_pos = {}; kur_rep = {}

    for iteration in range(MAX_REFINE + 1):
        print(f'\n  {"─"*66}')
        print(f'  Iter {iteration}  |  Kuramoto  (dur={duration:.2f}s)')

        sync_pos, kur_rep = run_kuramoto(arm_names, arm_pos, arm_bases, duration)

        for pk, pr in kur_rep['pair_reports'].items():
            icon = '✅' if pr['collision_free'] else '❌'
            print(f'    {icon} {pk:<28}  '
                  f'min={pr["min_dist_m"]*100:.1f}cm  crit={pr["critical"]}')

        history.append({'iteration': iteration, 'kuramoto': kur_rep})

        if kur_rep['collision_free']:
            print(f'\n  ✅  All {n_pairs} pairs collision-free at iteration {iteration}')
            break

        if iteration >= MAX_REFINE:
            print(f'\n  ⚠   MAX_REFINE={MAX_REFINE} reached'); break

        # Census — find all still-colliding pairs in post-Kuramoto positions
        # FIX: was called twice; first call incorrectly passed a joint array
        # (sync_pos[arm_names[0]]) as t_vec, giving K=NDOF=6 steps instead of
        # the full trajectory length. Only the corrected second call is kept.
        t_dummy = np.linspace(0., 1., len(arm_pos[arm_names[0]]))
        cens    = census(arm_names, sync_pos, arm_bases, t_dummy, n_seg)
        if not cens: break

        # FIX: process ALL colliding pairs each iteration ordered by urgency.
        # Previously only the single most urgent pair was fixed per iteration.
        pairs_ordered = sorted(cens.keys(), key=lambda p: cens[p]['first_frac'])
        n_wp = N_CP_BASE + CP_INCREMENT * (iteration + 1)

        for pair_key in pairs_ordered:
            info     = cens[pair_key]
            ni, nj   = pair_key
            coll_seg = info['first_seg']
            bi, bj   = arm_bases[ni], arm_bases[nj]
            s0_c     = coll_seg / n_seg
            s1_c     = (coll_seg + 1) / n_seg
            others_names = [n for n in arm_names if n != ni and n != nj]

            print(f'\n  Phase B: {ni}↔{nj}  '
                  f'coll_seg={coll_seg} arc=[{s0_c:.2f},{s1_c:.2f}]  n_wp={n_wp}')

            wps_i, wps_j = alternating_ik_for_pair(
                arm_pos[ni], arm_pos[nj], arm_arc[ni],
                bi, bj,
                arm_pos, arm_bases, others_names,
                s0_c, s1_c, n_wp)

            if wps_i is not None:
                print(f'    Alternating IK: {len(wps_i)} waypoints generated')
                arm_seg_cps[ni] = rebuild_spline(
                    arm_seg_cps[ni], arm_pos[ni], arm_arc[ni], coll_seg, wps_i, iteration, n_seg)
                arm_seg_cps[nj] = rebuild_spline(
                    arm_seg_cps[nj], arm_pos[nj], arm_arc[nj], coll_seg, wps_j, iteration, n_seg)
                pos_i, _, _, _ = eval_full_trajectory(arm_seg_cps[ni], duration)
                pos_j, _, _, _ = eval_full_trajectory(arm_seg_cps[nj], duration)
                # Update immediately so next pair sees latest positions
                arm_pos[ni] = pos_i; arm_arc[ni] = np.linspace(0,1,len(pos_i))
                arm_pos[nj] = pos_j; arm_arc[nj] = np.linspace(0,1,len(pos_j))
            else:
                print(f'    ⚠  {ni}↔{nj}: alternating IK no valid configs — skipping')

    # ── Assemble output ────────────────────────────────────────────────────
    out = {}
    for name in arm_names:
        pos_f, vel_f, acc_f, t_f, dur_f = scale_if_needed(arm_seg_cps[name], duration)
        n_seg_arm = len(arm_seg_cps[name]); seg_info = []
        for seg, (cp, kn) in enumerate(arm_seg_cps[name]):
            seg_info.append({
                'segment'  : seg,
                'arc_start': round(seg/n_seg_arm, 4),
                'arc_end'  : round((seg+1)/n_seg_arm, 4),
                'n_cp'     : len(cp), 'cp': cp.tolist(),
            })
        out[name] = {
            'robot_name': name,
            'metadata'  : {
                **trajectories[name]['metadata'],
                'duration'          : float(dur_f),
                'n_samples'         : len(pos_f),
                'refine_iterations' : iteration,
            },
            'spline'    : {'n_seg': n_seg_arm, 'degree': DEG, 'segments': seg_info},
            'trajectory': {
                'time'          : t_f.tolist(),
                'positions'     : pos_f.tolist(),
                'velocities'    : vel_f.tolist(),
                'accelerations' : acc_f.tolist(),
                'n_samples'     : len(pos_f),
            },
        }

    out['synchronisation_report'] = kur_rep
    out['refinement_history']     = history
    out['parameters'] = {
        'k_base': K_BASE, 'k_repulse': K_REPULSE, 'min_safe_dist': MIN_SAFE,
        'max_refine': MAX_REFINE, 'cp_increment': CP_INCREMENT,
        'n_arms': len(arm_names), 'n_pairs': n_pairs,
    }
    return out

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print('\n' + '=' * 68)
    print('  STEP 14  —  Kuramoto + Alternating IK + B-Spline Refinement (4-arm)')
    print('=' * 68)

    for fname, step in [('trajectories.json','step_12'), ('collision_report.json','step_13')]:
        if not os.path.exists(fname):
            print(f'\n  ❌  {fname} not found — run {step} first'); sys.exit(1)

    with open('trajectories.json') as fh:     tdata = json.load(fh)
    with open('collision_report.json') as fh: crep  = json.load(fh)

    active = sorted([k for k in tdata if k.startswith('dsr')])
    if not active: print('\n  ❌  No arm data'); sys.exit(1)

    print(f'\n  Active arms   : {active}')
    print(f'  Step-13 status: {crep["overall_status"]}')
    print(f'  Pairs         : {crep.get("n_pairs", len(active)*(len(active)-1)//2)}')

    result = synchronise(
        {n: tdata[n] for n in active},
        {n: ROBOT_BASES[n] for n in active},
    )

    with open('synchronized_trajectories.json', 'w') as fh:
        json.dump(result, fh, indent=2)

    safe = result['synchronisation_report'].get('collision_free', False)
    kb   = os.path.getsize('synchronized_trajectories.json') / 1024.0
    print(f'\n  {"✅" if safe else "⚠ "}  Saved: synchronized_trajectories.json  ({kb:.1f} KB)')
    print('  Next  →  python3 step_15.py\n')

if __name__ == '__main__':
    main()