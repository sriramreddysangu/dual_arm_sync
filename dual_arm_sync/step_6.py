#!/usr/bin/env python3
"""
step_6.py  —  Automated Gazebo Execution Benchmark
═══════════════════════════════════════════════════════════════════════════════
PURPOSE
───────
Automated N-trial Gazebo benchmark for evaluating the dual-arm synchronized
trajectory generation pipeline. Each trial is fully executed on real Gazebo
hardware and verified — results are suitable for paper submission tables and
statistical analysis.

TRIAL FLOW (strictly sequential)
─────────────────────────────────
  Trial t:
    1. Read current Gazebo joint states → use as start config
       (Trial 1: initial Gazebo state. Trial t+1: verified final state of trial t)
    2. Generate random target pose (pos + orientation) per arm
    3. Run full pipeline:
         IK → cubic B-spline (5 seg × 4 CP) → collision check →
         Kuramoto sync + alternating IK CP refinement (up to 5 rounds)
    4. Execute synchronized trajectory in Gazebo at 100 Hz (lock-step)
    5. Hold final pose 1s
    6. Read fresh joint states from Gazebo:
         ≡ ros2 topic echo /dsr01/gz/joint_states --once
         ≡ ros2 topic echo /dsr02/gz/joint_states --once
    7. Verify: joint error + EE position error vs planned target
    8. Verified Gazebo JS becomes start config for next trial

OUTCOME CLASSIFICATION (9 mutually exclusive)
──────────────────────────────────────────────
  FAIL_IK        : IK found no valid solution
  SAFE_NO_COLL   : Trajectory has zero inter-arm collision — trivially safe
  RESOLVED_KUR   : Collision resolved by Kuramoto phase synchronization only
  RESOLVED_CP_1  : Resolved after 1 alternating IK + CP refinement round
  RESOLVED_CP_2  : Resolved after 2 rounds
  RESOLVED_CP_3  : Resolved after 3 rounds
  RESOLVED_CP_4  : Resolved after 4 rounds
  RESOLVED_CP_5  : Resolved after 5 rounds
  UNRESOLVED     : Still colliding after all MAX_REFINE rounds

PAPER METRICS RECORDED PER TRIAL
──────────────────────────────────
  Inputs        : start joints, target pos+quat per arm
  Planning      : IK solve time, B-spline build time, collision check time,
                  Kuramoto iterations, CP refinement rounds, total plan time
  Trajectory    : duration, EE path length, max joint velocity, collision steps
  Execution     : trajectory execution time (wall clock)
  Verification  : joint error (deg), EE position error (mm), pass/fail
  Outcome       : one of 9 classes above

OUTPUTS
───────
  trial_results.json  — per-trial full detail (for analysis)
  trial_summary.json  — aggregate statistics (for paper tables)
  benchmark_report.txt — human-readable paper-ready summary

Usage:
  ros2 run dual_arm_sync step_6
  ros2 run dual_arm_sync step_6 --trials 200
  ros2 run dual_arm_sync step_6 --trials 100 --seed 42
  ros2 run dual_arm_sync step_6 --trials 200 --seed 0 --duration 10.0
═══════════════════════════════════════════════════════════════════════════════
"""

import argparse, json, os, sys, time, threading, datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import BSpline

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float64MultiArray
    from sensor_msgs.msg import JointState
except ImportError:
    print('[step_6] ❌  rclpy not found — ROS2 required for Gazebo execution')
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# ROBOT CONSTANTS  (Doosan M1013)
# ─────────────────────────────────────────────────────────────────────────────

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

ROBOT_BASES: Dict[str, np.ndarray] = {
    'dsr01': np.array([0.0,  0.5, 0.0]),
    'dsr02': np.array([0.0, -0.5, 0.0]),
}
ROBOT_NAMES = ['dsr01', 'dsr02']
LINK_RADII  = np.array([0.10, 0.10, 0.08, 0.08, 0.06, 0.06])
LINK_NAMES  = ['base', 'shoulder', 'upper_arm', 'forearm', 'wrist1', 'wrist2']

CTRL_TOPIC = '/{arm}/gz/dsr_position_controller/commands'
JS_TOPICS  = ['/{arm}/gz/joint_states', '/{arm}/joint_states']

# ── Pipeline parameters ────────────────────────────────────────────────────────
SAFETY_MARGIN = 0.12     # m  — collision sphere margin
WARNING_MARGIN = 0.20    # m  — warning zone
N_SEG      = 5           # B-spline segments
N_CP_SEG   = 4           # control points per segment
DEG        = 3           # cubic
RATE_HZ    = 100.0       # control frequency Hz
MAX_REFINE = 5           # max CP refinement rounds
CP_INC     = 2           # extra CPs per refinement round

IK_TOL_POS  = 0.010     # m
IK_TOL_ROT  = 0.05      # rad
IK_MAX_ITER = 600
IK_FTOL     = 1e-10
IK_UNIQ     = 0.12
W_POS = 1.0; W_ROT = 0.15

K_BASE=5.0; K_REPULSE=80.0; K_EMERGENCY=250.0                                              # line 135 for k change 
KUR_DT=0.01; MIN_SAFE=0.15; REPULSE_D=0.28; LEADER_THRESH=0.05

# ── Random target workspace ────────────────────────────────────────────────────
REACH_MIN=0.40; REACH_MAX=1.10; Z_MIN=0.10; Z_MAX=1.00

# ── Execution / verification tolerances ───────────────────────────────────────
HOLD_AFTER     = 0.8     # s  — brief hold after trajectory; settle_wait runs concurrently
                          #      Total post-traj overhead ≈ 0.8 + settle(≤2.0) ≤ 2.8s
VER_TIMEOUT    = 5.0     # s  — timeout for fresh Gazebo JS read
SETTLE_TIMEOUT = 2.0     # s  — max settle wait (reduced — Gazebo controller settles fast)
SETTLE_VEL_THR = 0.008   # rad/s — tighter: robot is settled
SETTLE_DT      = 0.04    # s  — polling interval
JOINT_VER_TOL  = 0.15    # rad (8.6deg)
EE_VER_TOL_MM  = 25.0    # mm

OUTCOMES = [
    'FAIL_IK', 'SAFE_NO_COLL', 'RESOLVED_KUR',
    'RESOLVED_CP_1','RESOLVED_CP_2','RESOLVED_CP_3',
    'RESOLVED_CP_4','RESOLVED_CP_5','UNRESOLVED',
]


# ─────────────────────────────────────────────────────────────────────────────
# FK + GEOMETRY
# ─────────────────────────────────────────────────────────────────────────────

def fk(q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    T = np.eye(4)
    for i in range(NDOF):
        al,a,to,d = DH[i]; th = q[i]+to
        ct,st = np.cos(th),np.sin(th); ca,sa = np.cos(al),np.sin(al)
        T = T @ np.array([[ct,-st,0.,a],[st*ca,ct*ca,-sa,-sa*d],
                           [st*sa,ct*sa,ca,ca*d],[0.,0.,0.,1.]])
    return T[:3,3].copy(), T


def fk_pos(q: np.ndarray, base: np.ndarray) -> np.ndarray:
    return fk(q)[0] + base


def link_origins(q: np.ndarray, base: np.ndarray) -> np.ndarray:
    T = np.eye(4); o = np.zeros((NDOF,3))
    for i in range(NDOF):
        al,a,to,d = DH[i]; th = q[i]+to
        ct,st = np.cos(th),np.sin(th); ca,sa = np.cos(al),np.sin(al)
        T = T @ np.array([[ct,-st,0.,a],[st*ca,ct*ca,-sa,-sa*d],
                           [st*sa,ct*sa,ca,ca*d],[0.,0.,0.,1.]])
        o[i] = T[:3,3]+base
    return o


def rot_err(Ra: np.ndarray, Rb: np.ndarray) -> float:
    R = Rb @ Ra.T
    return float(np.arccos(np.clip((np.trace(R)-1)/2,-1,1)))


def pair_min_dist(qi, bi, qj, bj) -> float:
    """Minimum link-origin distance — vectorised broadcasting."""
    oi = link_origins(qi, bi)  # (NDOF,3)
    oj = link_origins(qj, bj)  # (NDOF,3)
    diff = oi[:, np.newaxis, :] - oj[np.newaxis, :, :]  # (NDOF,NDOF,3)
    return float(np.min(np.linalg.norm(diff, axis=2)))


def pair_collides(qi,bi,qj,bj) -> bool:
    oi=link_origins(qi,bi); oj=link_origins(qj,bj)
    for a in range(NDOF):
        for b in range(NDOF):
            if np.linalg.norm(oi[a]-oj[b]) < LINK_RADII[a]+LINK_RADII[b]+SAFETY_MARGIN:
                return True
    return False


def interp_pos(pos: np.ndarray, frac: float) -> np.ndarray:
    frac=float(np.clip(frac,0,1)); n=len(pos)-1
    if n<=0: return pos[0].copy()
    idx=min(int(frac*n),n-1); a=frac*n-idx
    return pos[idx]+a*(pos[idx+1]-pos[idx])




# ─────────────────────────────────────────────────────────────────────────────
# JOINT NORMALISATION + ADAPTIVE DURATION  (ported from step_26)
# ─────────────────────────────────────────────────────────────────────────────

VEL_USE_FRAC  = 0.60   # use 60% of velocity limit for duration calculation
MIN_DURATION  = 8.0    # floor: never plan below 8 s regardless of joint delta


def normalize_joints(q: np.ndarray) -> np.ndarray:
    """Wrap Gazebo-accumulated angles back into POS_LIM range."""
    q_out = q.copy()
    for j in range(NDOF):
        lo, hi = POS_LIM[j]
        span   = hi - lo
        q_out[j] = lo + (q[j] - lo) % span
        q_out[j] = float(np.clip(q_out[j], lo, hi))
    return q_out


def angular_error(q_read: np.ndarray, q_target: np.ndarray) -> np.ndarray:
    """Wrap-aware minimum angular distance per joint, result in [0, π]."""
    diff = q_read - q_target
    return np.abs(np.arctan2(np.sin(diff), np.cos(diff)))


def adaptive_duration(start_qs: Dict[str, np.ndarray],
                      target_qs: Dict[str, np.ndarray],
                      requested: float) -> float:
    """
    Compute minimum trajectory duration so no joint exceeds VEL_USE_FRAC×VEL_LIM.
    Returns max(needed × 1.3 safety, MIN_DURATION, requested).
    Prevents the controller from being asked to track faster than it can handle.
    """
    max_needed = MIN_DURATION
    for n in ROBOT_NAMES:
        sq = start_qs.get(n, np.zeros(NDOF))
        eq = target_qs.get(n, np.zeros(NDOF))
        disp     = angular_error(sq, eq)
        t_needed = float(np.max(disp / (VEL_LIM * VEL_USE_FRAC)))
        max_needed = max(max_needed, t_needed)
    return max(max_needed * 1.3, MIN_DURATION, requested)


# ─────────────────────────────────────────────────────────────────────────────
# QUATERNION + RANDOM TARGET GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def quat_norm(q: np.ndarray) -> np.ndarray:
    n=np.linalg.norm(q); return q/n if n>1e-12 else np.array([1.,0.,0.,0.])


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    w,x,y,z=quat_norm(q)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)  ],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)  ],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
    ])


def random_quat(rng: np.random.Generator) -> np.ndarray:
    """Uniformly distributed random unit quaternion (Shoemake 1992)."""
    u1,u2,u3 = rng.random(),rng.random(),rng.random()
    return quat_norm(np.array([
        np.sqrt(1-u1)*np.sin(2*_PI*u2), np.sqrt(1-u1)*np.cos(2*_PI*u2),
        np.sqrt(u1)  *np.sin(2*_PI*u3), np.sqrt(u1)  *np.cos(2*_PI*u3),
    ]))


def random_target(rng: np.random.Generator,
                   base: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample a uniformly random reachable target in the arm's workspace.
    Returns (pos_world, quaternion).
    Workspace: spherical shell r ∈ [REACH_MIN, REACH_MAX], z ∈ [Z_MIN, Z_MAX].
    """
    for _ in range(300):
        r     = rng.uniform(REACH_MIN, REACH_MAX)
        theta = rng.uniform(0, 2*_PI)
        z     = rng.uniform(Z_MIN, Z_MAX)
        r_xy  = np.sqrt(max(r**2 - z**2, 0.))
        x_loc = r_xy * np.cos(theta)
        y_loc = r_xy * np.sin(theta)
        # Reachability: xy radius must exceed base offset A
        if np.hypot(x_loc, y_loc) > A + 0.05:
            return np.array([x_loc, y_loc, z]) + base, random_quat(rng)
    # Deterministic fallback
    return base + np.array([0.6, 0., 0.5]), np.array([1.,0.,0.,0.])


# ─────────────────────────────────────────────────────────────────────────────
# IK
# ─────────────────────────────────────────────────────────────────────────────

def _ik_seeds(current: np.ndarray, tgt: np.ndarray) -> List[np.ndarray]:
    px,py,pz=tgt; cl=lambda q:np.clip(q,POS_LIM[:,0],POS_LIM[:,1])
    # FIX A: removed np.zeros(NDOF) from initial seeds list —
    # biases solver toward home config, producing slow solutions.
    seeds=[cl(current.copy())]
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
            for t in (t1,t1+_PI_2,t1-_PI_2):
                seeds.append(cl(np.array([t,q2,q3,0.,q5,0.])))
    # FIX B: removed duplicate np.zeros(NDOF) appended after geometric seeds.
    # The geometric seeds already give global workspace coverage.
    uniq: List[np.ndarray]=[]
    for s in seeds:
        if all(np.linalg.norm(s-u)>0.05 for u in uniq): uniq.append(s)
    return uniq


def solve_ik(tloc: np.ndarray, trot: np.ndarray,
             cur: np.ndarray) -> List[np.ndarray]:
    bds=[(POS_LIM[i,0],POS_LIM[i,1]) for i in range(NDOF)]
    def obj(q):
        p,T=fk(q)
        return W_POS*float(np.sum((p-tloc)**2))+W_ROT*float(rot_err(T[:3,:3],trot)**2)
    sols: List[np.ndarray]=[]
    for seed in _ik_seeds(cur,tloc):
        res=minimize(obj,seed,method='SLSQP',bounds=bds,
                     options={'maxiter':IK_MAX_ITER,'ftol':IK_FTOL})
        if not res.success: continue
        q=np.clip(res.x,POS_LIM[:,0],POS_LIM[:,1]); p,T=fk(q)
        if np.linalg.norm(p-tloc)>=IK_TOL_POS or rot_err(T[:3,:3],trot)>=IK_TOL_ROT: continue
        if all(np.linalg.norm(q-s)>IK_UNIQ for s in sols): sols.append(q)
    return sols


def find_best_configs(arms_data: Dict,
                       start_qs: Dict[str,np.ndarray]) -> Optional[Dict[str,np.ndarray]]:
    """
    Round 1: SE(3) IK for each arm from its current start config.
    Round 2: Re-score solutions by inter-arm clearance (dominant weight).
    """
    an=sorted(arms_data.keys()); all_sols:Dict[str,List]={}; best:Dict[str,np.ndarray]={}

    for name in an:
        d=arms_data[name]; sq=start_qs.get(name,np.zeros(NDOF))
        sols=solve_ik(d['tloc'],d['trot'],sq)
        if not sols:
            # Position-only fallback
            bds=[(POS_LIM[i,0],POS_LIM[i,1]) for i in range(NDOF)]
            def po(q,tl=d['tloc']): p,_=fk(q); return float(np.sum((p-tl)**2))
            for seed in _ik_seeds(sq,d['tloc']):
                res=minimize(po,seed,method='SLSQP',bounds=bds,
                             options={'maxiter':IK_MAX_ITER,'ftol':IK_FTOL})
                q=np.clip(res.x,POS_LIM[:,0],POS_LIM[:,1])
                if np.linalg.norm(fk(q)[0]-d['tloc'])<IK_TOL_POS: sols.append(q); break
        if not sols: return None
        all_sols[name]=sols
        # FIX Round-1: use exact bottleneck time, not Euclidean norm.
        # min_motion_time = max(|Δq_i|/VEL_LIM_i) correctly weights by velocity limit.
        best[name]=min(sols, key=lambda q, s=sq: float(np.max(np.abs(q-s)/VEL_LIM)))

    # Re-score: clearance (primary) + velocity-weighted sq diff from start (secondary)
    for name in an:
        base    = arms_data[name]['base']
        start_q = start_qs.get(name, np.zeros(NDOF))
        others  = [(best[n],arms_data[n]['base']) for n in an if n!=name]
        def _score(q, sq=start_q, b=base, oth=others):
            # FIX Round-2: added velocity-weighted squared difference term.
            # sq_vel_cost = sum((Δq_i/VEL_LIM_i)^2) — the 'difference square'
            # that selects the config reachable in minimum time from start.
            clr = sum(float(np.tanh(pair_min_dist(q,b,oq,ob)/0.25)) for oq,ob in oth) if oth else 1.0
            cost = float(np.sum(((q - sq) / VEL_LIM) ** 2))
            time_sc = float(np.exp(-cost / 25.0))  # T_REF=5s → T_REF^2=25
            return 3.0*clr + 2.0*time_sc
        if all_sols[name]:
            scores = [_score(q) for q in all_sols[name]]
            best[name] = all_sols[name][int(np.argmax(scores))]
    return best


# ─────────────────────────────────────────────────────────────────────────────
# B-SPLINE
# ─────────────────────────────────────────────────────────────────────────────

def make_knots(ncp: int, deg: int=DEG) -> np.ndarray:
    ni=max(0,ncp-deg-1); inn=np.linspace(0,1,ni+2)[1:-1] if ni>0 else np.array([])
    return np.concatenate([np.zeros(deg+1),inn,np.ones(deg+1)])


def build_cp(sq: np.ndarray, eq: np.ndarray,
             n_seg: int=N_SEG, n_cp: int=N_CP_SEG) -> np.ndarray:
    """Linear seed — monotone CPs → no oscillation in joint space."""
    total=n_seg*(n_cp-1)+1; s=np.linspace(0.,1.,total)
    cpg=sq[np.newaxis,:]+s[:,np.newaxis]*(eq-sq)[np.newaxis,:]
    cpg=np.clip(cpg,POS_LIM[:,0],POS_LIM[:,1])
    cps=np.zeros((n_seg,n_cp,NDOF))
    for seg in range(n_seg): i0=seg*(n_cp-1); cps[seg]=cpg[i0:i0+n_cp]
    return cps


def seg_list_from_cp(cps: np.ndarray, n_seg: int=N_SEG,
                      n_cp: int=N_CP_SEG) -> List[Tuple[np.ndarray,np.ndarray]]:
    kn=make_knots(n_cp); return [(cps[s].copy(),kn.copy()) for s in range(n_seg)]


def eval_global(sl: List[Tuple[np.ndarray,np.ndarray]],
                duration: float) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """
    Single global clamped cubic B-spline.
    Flatten segment CPs (de-duplicate shared boundaries) → one spline.
    Eliminates stitching artifacts and joint-space oscillation.
    """
    all_cp: List[np.ndarray]=[]
    for si,(cp,_) in enumerate(sl): all_cp.append(cp if si==0 else cp[1:])
    cpg=np.vstack(all_cp); kn=make_knots(len(cpg))
    ns=max(2,int(round(duration*RATE_HZ))); sf=np.linspace(0.,1.,ns)
    pos=np.zeros((ns,NDOF)); vel=pos.copy(); acc=pos.copy()
    for j in range(NDOF):
        spl=BSpline(kn,cpg[:,j],DEG,extrapolate=True)
        pos[:,j]=spl(sf); vel[:,j]=spl.derivative(1)(sf)/duration
        acc[:,j]=spl.derivative(2)(sf)/duration**2
    return np.clip(pos,POS_LIM[:,0],POS_LIM[:,1]),vel,acc,np.linspace(0.,duration,ns)


def scale_duration(sl: List[Tuple[np.ndarray,np.ndarray]],
                    duration: float) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,float]:
    pos,vel,acc,t=eval_global(sl,duration); sv=sa=1.0
    for j in range(NDOF):
        vp=float(np.max(np.abs(vel[:,j]))); ap=float(np.max(np.abs(acc[:,j])))
        if vp>VEL_LIM[j]: sv=max(sv,vp/VEL_LIM[j])
        if ap>ACC_LIM[j]: sa=max(sa,float(np.sqrt(ap/ACC_LIM[j])))
    sc=max(sv,sa)
    if sc>1.0: duration=duration*sc*1.05; pos,vel,acc,t=eval_global(sl,duration)
    return pos,vel,acc,t,duration


# ─────────────────────────────────────────────────────────────────────────────
# COLLISION + KURAMOTO
# ─────────────────────────────────────────────────────────────────────────────

def count_collisions(pi,pj,bi,bj) -> Tuple[int,Optional[float]]:
    N=min(len(pi),len(pj)); nc=0; fk=-1
    for k in range(N):
        if pair_collides(pi[k],bi,pj[k],bj): nc+=1; fk=k if fk<0 else fk
    return nc, (fk/max(N-1,1) if fk>=0 else None)


def first_coll_seg_idx(pi,pj,bi,bj,n_seg) -> Tuple[Optional[int],Optional[float]]:
    N=min(len(pi),len(pj))
    for k in range(N):
        if pair_collides(pi[k],bi,pj[k],bj):
            frac=k/max(N-1,1); return min(int(frac*n_seg),n_seg-1),frac
    return None,None


def run_kuramoto(arm_names,arm_pos,arm_bases,duration) -> Tuple[Dict,Dict]:
    N=len(arm_names); ns=max(2,int(duration/KUR_DT)); omega0=1./duration
    phi=np.zeros(N); om=np.full(N,omega0)
    pairs=[(i,j) for i in range(N) for j in range(i+1,N)]
    pdists={p:np.zeros(ns) for p in pairs}
    sync={n:np.zeros((ns,NDOF)) for n in arm_names}

    for k in range(ns):
        phi=np.clip(phi,0.,1.)
        qn={n:interp_pos(arm_pos[n],phi[idx]) for idx,n in enumerate(arm_names)}
        for idx,n in enumerate(arm_names): sync[n][k]=qn[n]
        ds={}
        for (i,j) in pairs:
            d=pair_min_dist(qn[arm_names[i]],arm_bases[arm_names[i]],
                             qn[arm_names[j]],arm_bases[arm_names[j]])
            ds[(i,j)]=d; pdists[(i,j)][k]=d
        dp=np.zeros(N)
        for (i,j) in pairs:
            dist=ds[(i,j)]; df=float(np.clip(1-dist/REPULSE_D,0,1))
            danger=float(np.clip(1-dist/MIN_SAFE,0,1))
            diff=phi[i]-phi[j]
            leader=i if diff>LEADER_THRESH else (j if diff<-LEADER_THRESH else -1)
            Kij=min(K_BASE*(1+4*df),15.)
            dp[i]+=Kij*float(np.sin(phi[j]-phi[i])); dp[j]+=Kij*float(np.sin(phi[i]-phi[j]))
            if dist<REPULSE_D:
                mag=K_REPULSE*df**2*30+(K_EMERGENCY*danger**3 if dist<MIN_SAFE else 0)
                if   leader==i: dp[i]-=mag*2;dp[j]-=mag*0.3
                elif leader==j: dp[j]-=mag*2;dp[i]-=mag*0.3
                else:           dp[i]-=mag*0.7;dp[j]-=mag*0.7
        phi+=KUR_DT*np.clip(om+dp,-2.,2.)

    pr={}; tc=0
    for (i,j) in pairs:
        dv=pdists[(i,j)]; nc=int(np.sum(dv<MIN_SAFE)); tc+=nc
        pr[f'{arm_names[i]}↔{arm_names[j]}']={
            'min_dist_m':float(np.min(dv)),'critical':nc,'collision_free':nc==0}
    return sync,{'pair_reports':pr,'total_critical':tc,'collision_free':tc==0}


# ─────────────────────────────────────────────────────────────────────────────
# ALTERNATING IK + B-SPLINE CP REBUILD
# ─────────────────────────────────────────────────────────────────────────────

def _ik_away(tloc, cur, oq, ob, tb) -> Optional[np.ndarray]:
    """
    Position-only IK that maximises inter-arm clearance.

    WHY POSITION-ONLY (not SE(3)):
    ─────────────────────────────
    During the alternating-IK refinement step the arm must find a joint
    configuration that (a) places the EE at the required Cartesian waypoint
    and (b) keeps the two arms as far apart as possible.

    Adding an orientation constraint to (a) dramatically shrinks the
    feasible solution space: the SLSQP must satisfy BOTH position AND
    rotation simultaneously while the clearance objective is pushing joints
    away.  In practice this means many seeds fail, only 1-2 solutions
    survive, and the clearance maximisation has almost nothing to choose
    from.

    With position-only IK the feasible set is much larger (any joint config
    that puts the EE at the correct XYZ qualifies).  The collision-avoidance
    objective then has a rich pool of candidates and reliably selects the one
    with maximum inter-arm distance.

    The small orientation error introduced is acceptable because:
      • Orientation is already matched at the IK endpoints (step 1 uses SE(3))
      • The alternating-IK waypoints are interior to the segment, not the
        trajectory endpoints
      • The B-spline naturally interpolates orientation smoothly between the
        correctly-oriented endpoint CPs
    """
    bds  = [(POS_LIM[i,0], POS_LIM[i,1]) for i in range(NDOF)]
    tol  = 0.012          # 12 mm — slightly tighter than before
    valid: List[np.ndarray] = []

    for seed in _ik_seeds(cur, tloc):
        # Pure position objective — no orientation term
        def obj(q, _t=tloc):
            p, _ = fk(q)
            return float(np.sum((p - _t) ** 2))

        res = minimize(obj, seed, method='SLSQP', bounds=bds,
                       options={'maxiter': 300, 'ftol': 1e-9})
        if not res.success:
            continue
        q = np.clip(res.x, POS_LIM[:, 0], POS_LIM[:, 1])
        if np.linalg.norm(fk(q)[0] - tloc) < tol:
            valid.append(q)

    if not valid:
        return None
    # Among all position-feasible configs, pick the one furthest from the other arm
    return max(valid, key=lambda q: pair_min_dist(q, tb, oq, ob))


def alternating_ik(pi, pj, arc, bi, bj, s0, s1, n_wp) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Position-only alternating IK along the collision segment.

    n_wp = number of waypoints to sample inside [s0, s1].
    arm-i goes first (away from arm-j), then arm-j goes away from new arm-i.

    The position-only IK (see _ik_away) gives a much richer feasible set,
    so the max-clearance selection consistently finds a good config.
    """
    Nj   = len(pj)
    mask = (arc >= s0 - 1e-6) & (arc <= s1 + 1e-6)
    idx  = np.where(mask)[0]
    if len(idx) < 2:
        return None, None

    # Denser sampling inside the collision window
    samp = np.unique(np.linspace(0, len(idx) - 1, n_wp, dtype=int))
    wps_i: List[np.ndarray] = []
    wps_j: List[np.ndarray] = []
    pqi = pi[idx[0]].copy()
    pqj = pj[min(idx[0], Nj - 1)].copy()

    for si in samp:
        k  = idx[si]; kj = min(k, Nj - 1)
        # Use FK position at current trajectory sample as the target EE position
        ee_i = fk(pi[k])[0]
        ee_j = fk(pj[kj])[0]

        # arm-i: position-only IK maximising dist from arm-j
        qi_new = _ik_away(ee_i, pqi, pqj, bj, bi)
        if qi_new is None:
            qi_new = pi[k].copy()   # fallback: keep original

        # arm-j: position-only IK maximising dist from updated arm-i
        qj_new = _ik_away(ee_j, pqj, qi_new, bi, bj)
        if qj_new is None:
            qj_new = pj[kj].copy()

        wps_i.append(qi_new)
        wps_j.append(qj_new)
        pqi = qi_new; pqj = qj_new

    return np.array(wps_i), np.array(wps_j)


def _fit_cp(wps,sv,ncp,ps,pe):
    ncp=max(DEG+2,ncp); kn=make_knots(ncp)
    B=np.zeros((len(sv),ncp)); e=np.zeros(ncp)
    for k in range(ncp):
        e[:]=0.; e[k]=1.
        v=BSpline(kn,e.copy(),DEG,extrapolate=False)(sv)
        B[:,k]=np.where(np.isfinite(v),v,0.)
    cp=np.zeros((ncp,NDOF))
    for j in range(NDOF):
        ts=float(ps[j]); te=float(pe[j]); Af=B[:,1:-1]; rhs=wps[:,j]-B[:,0]*ts-B[:,-1]*te
        if Af.shape[1]>0:
            z,*_=np.linalg.lstsq(Af,rhs,rcond=None)
            z=np.where(np.isfinite(z),z,np.linspace(ts,te,len(z)))
            z=np.clip(z,POS_LIM[j,0],POS_LIM[j,1])
        else: z=np.array([])
        cp[:,j]=np.concatenate([[ts],z if len(z) else [],[te]])
    return cp


def rebuild_spline(orig_sl,pos_full,arc,coll_seg,new_wps,refine_count) -> List:
    """
    Keep original CPs for non-collision segments.
    Collision segment: replace with alternating IK waypoints (more CPs).
    Neighbour segments ±1: add extra CPs to smooth transition.
    """
    result=list(orig_sl); n_seg=len(orig_sl)
    extra=CP_INC*(refine_count+1)
    ncp_c=N_CP_SEG+extra; ncp_n=N_CP_SEG+max(0,extra-CP_INC)
    for seg in range(n_seg):
        s0=seg/n_seg; s1=(seg+1)/n_seg
        mask=(arc>=s0-1e-9)&(arc<=s1+1e-9); idx=np.where(mask)[0]
        if len(idx)<2: continue
        ps=pos_full[idx[0]].copy(); pe=pos_full[idx[-1]].copy()
        if seg==coll_seg:
            sw=np.linspace(0.,1.,len(new_wps)); cp=_fit_cp(new_wps,sw,ncp_c,ps,pe)
            result[seg]=(cp,make_knots(ncp_c))
        elif seg in[coll_seg-1,coll_seg+1] and 0<=seg<n_seg:
            sl=np.clip((arc[idx]-s0)/(s1-s0),0.,1.); cp=_fit_cp(pos_full[idx],sl,ncp_n,ps,pe)
            result[seg]=(cp,make_knots(ncp_n))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# PLANNING PIPELINE  (with fine-grained timing for paper)
# ─────────────────────────────────────────────────────────────────────────────

def plan(arms_data: Dict,
          start_qs: Dict[str,np.ndarray],
          duration: float) -> Dict:
    """
    Full planning pipeline with per-stage timing.
    Returns dict with outcome, arm_pos, and all metrics needed for paper.
    """
    timings: Dict[str,float] = {}
    result: Dict = {
        'outcome'           : None,
        'cp_rounds'         : 0,
        'had_collision'     : False,
        'n_coll_raw'        : 0,          # collision steps before Kuramoto
        'n_coll_final'      : 0,          # collision steps after Kuramoto+CPs
        'arm_pos'           : None,
        'duration'          : duration,
        'target_joints'     : {},
        'start_joints'      : {n: start_qs.get(n,np.zeros(NDOF)).tolist() for n in ROBOT_NAMES},
        'inter_arm_clear_cm': None,       # clearance at IK target configs
        'kur_min_dist_cm'   : None,       # Kuramoto min inter-arm distance
        'ee_path_length_m'  : {},         # EE path length per arm
        'timings'           : timings,
        'plan_time_s'       : None,       # total planning time
    }

    t_total = time.time()

    # ── IK ────────────────────────────────────────────────────────────────────
    t0 = time.time()
    best = find_best_configs(arms_data, start_qs)
    timings['ik_s'] = round(time.time()-t0, 4)

    if best is None:
        result['outcome']    = 'FAIL_IK'
        result['plan_time_s']= round(time.time()-t_total, 3)
        return result

    for n in ROBOT_NAMES:
        result['target_joints'][n] = best[n].tolist()

    d_ik = pair_min_dist(best['dsr01'],ROBOT_BASES['dsr01'],
                          best['dsr02'],ROBOT_BASES['dsr02'])
    result['inter_arm_clear_cm'] = round(d_ik*100, 2)

    # ── Adaptive duration + B-spline trajectories ─────────────────────────────
    t0 = time.time()
    # FIX: compute minimum safe duration from actual joint displacements;
    # prevents the controller from tracking a trajectory it cannot follow.
    duration = adaptive_duration(start_qs, best, duration)

    arm_sl:  Dict[str,List] = {}
    arm_pos: Dict[str,np.ndarray] = {}

    for n in ROBOT_NAMES:
        cp  = build_cp(start_qs.get(n,np.zeros(NDOF)), best[n])
        sl  = seg_list_from_cp(cp)
        pos,_,_,_,duration = scale_duration(sl, duration)
        arm_sl[n]  = sl
        arm_pos[n] = pos

    result['duration'] = duration
    arc = np.linspace(0., 1., len(arm_pos['dsr01']))

    # EE path lengths
    for n in ROBOT_NAMES:
        base = ROBOT_BASES[n]
        ee   = np.array([fk_pos(arm_pos[n][k],base) for k in range(len(arm_pos[n]))])
        plen = float(np.sum(np.linalg.norm(np.diff(ee,axis=0),axis=1))) if len(ee)>1 else 0.
        result['ee_path_length_m'][n] = round(plen, 4)

    timings['bspline_s'] = round(time.time()-t0, 4)

    # ── Collision check ───────────────────────────────────────────────────────
    t0 = time.time()
    bi=ROBOT_BASES['dsr01']; bj=ROBOT_BASES['dsr02']
    nc_raw, _ = count_collisions(arm_pos['dsr01'],arm_pos['dsr02'],bi,bj)
    result['had_collision'] = nc_raw > 0
    result['n_coll_raw']    = nc_raw
    timings['collision_check_s'] = round(time.time()-t0, 4)

    if not result['had_collision']:
        result['outcome']    = 'SAFE_NO_COLL'
        result['arm_pos']    = arm_pos
        result['plan_time_s']= round(time.time()-t_total, 3)
        return result

    # ── Kuramoto + alternating IK refinement ──────────────────────────────────
    t0 = time.time(); resolved=False

    for iteration in range(MAX_REFINE+1):
        sync_pos, kur_rep = run_kuramoto(ROBOT_NAMES, arm_pos, ROBOT_BASES, duration)
        md=kur_rep['pair_reports'].get('dsr01↔dsr02',{}).get('min_dist_m',0.)
        result['kur_min_dist_cm']=round(md*100, 2)

        # Check synchronized positions
        nc_sync,_ = count_collisions(sync_pos['dsr01'],sync_pos['dsr02'],bi,bj)
        result['n_coll_final'] = nc_sync

        if nc_sync == 0:
            resolved=True
            result['arm_pos']   = {n:sync_pos[n] for n in ROBOT_NAMES}
            result['outcome']   = 'RESOLVED_KUR' if iteration==0 else f'RESOLVED_CP_{iteration}'
            result['cp_rounds'] = iteration
            break

        if iteration >= MAX_REFINE: break

        cs,_ = first_coll_seg_idx(arm_pos['dsr01'],arm_pos['dsr02'],bi,bj,N_SEG)
        if cs is None: break
        s0=cs/N_SEG; s1=(cs+1)/N_SEG
        # Scale waypoints and CPs aggressively with iteration count
        # iter 0→6wp, iter 1→8wp, iter 2→10wp, iter 3→12wp, iter 4→14wp
        n_wp = N_CP_SEG + CP_INC * (iteration + 2)
        wps_i,wps_j=alternating_ik(arm_pos['dsr01'],arm_pos['dsr02'],arc,bi,bj,s0,s1,n_wp)
        if wps_i is not None:
            arm_sl['dsr01']=rebuild_spline(arm_sl['dsr01'],arm_pos['dsr01'],arc,cs,wps_i,iteration)
            arm_sl['dsr02']=rebuild_spline(arm_sl['dsr02'],arm_pos['dsr02'],arc,cs,wps_j,iteration)
            pi,_,_,_=eval_global(arm_sl['dsr01'],duration)
            pj,_,_,_=eval_global(arm_sl['dsr02'],duration)
            arm_pos['dsr01']=pi; arm_pos['dsr02']=pj
            arc=np.linspace(0.,1.,len(pi))

    if not resolved:
        result['outcome']   = 'UNRESOLVED'
        result['arm_pos']   = arm_pos
        result['cp_rounds'] = MAX_REFINE

    timings['kuramoto_refinement_s'] = round(time.time()-t0, 4)
    result['plan_time_s'] = round(time.time()-t_total, 3)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# ROS2 NODE
# ─────────────────────────────────────────────────────────────────────────────

class BenchmarkNode(Node):
    """
    Handles Gazebo communication:
      - Publish joint commands at 100 Hz to dsr_position_controller
      - Subscribe to /dsr01/gz/joint_states and /dsr02/gz/joint_states
      - Fresh joint state reads (≡ ros2 topic echo --once per arm)
    """

    def __init__(self):
        super().__init__('step_6_benchmark')
        self._pubs = {
            n: self.create_publisher(Float64MultiArray,CTRL_TOPIC.format(arm=n),10)
            for n in ROBOT_NAMES
        }
        self._lock = threading.Lock()
        self._js:    Dict[str,Optional[np.ndarray]] = {n:None for n in ROBOT_NAMES}
        self._js_t:  Dict[str,float] = {n:0. for n in ROBOT_NAMES}

        for n in ROBOT_NAMES:
            for t in JS_TOPICS:
                self.create_subscription(JointState,t.format(arm=n),
                                         lambda msg,nm=n:self._js_cb(msg,nm),10)
        self.get_logger().info('BenchmarkNode ready')

    def _js_cb(self,msg: JointState,name: str):
        if len(msg.position)<NDOF: return
        jmap={nm:i for i,nm in enumerate(msg.name)}
        keys=[f'joint_{k}' for k in range(1,NDOF+1)]
        q=(np.array([msg.position[jmap[k]] for k in keys])
           if all(k in jmap for k in keys) else np.array(msg.position[:NDOF]))
        with self._lock:
            self._js[name]=q.astype(float); self._js_t[name]=time.time()

    def get_joints(self,name: str) -> Optional[np.ndarray]:
        with self._lock: return self._js[name].copy() if self._js[name] is not None else None

    def wait_for_js(self,timeout: float=30.) -> bool:
        t0=time.time()
        while time.time()-t0<timeout:
            rclpy.spin_once(self,timeout_sec=0.05)
            with self._lock:
                if all(self._js[n] is not None for n in ROBOT_NAMES): return True
        return False

    def publish_all(self,cmds: Dict[str,np.ndarray]):
        """Publish to ALL arms in one call — guarantees lock-step timing."""
        msg=Float64MultiArray()
        for n in ROBOT_NAMES:
            msg.data=[float(v) for v in cmds[n]]; self._pubs[n].publish(msg)

    def execute_trajectory(self, arm_pos: Dict[str,np.ndarray], duration: float) -> Dict:
        """
        Execute synchronized trajectory in Gazebo at 100 Hz.
        Both arms published at identical timestep = true lock-step synchronization.

        TIMING DESIGN (10 s trajectory stays ≈ 10 s total):
        ─────────────────────────────────────────────────────
        • Trajectory loop: exactly duration × RATE_HZ steps  → duration s
        • Hold + settle:   HOLD_AFTER (0.8 s) + settle_wait (≤ 2 s)
          Total execution wall time ≈ duration + 2.8 s at most.

        The settle_wait polls joint velocity at 40 ms intervals while
        ALSO publishing the hold command — so the robot is being actively
        driven to the final position during settling.
        """
        dt_ns = int(1e9 / RATE_HZ)
        n_out = max(2, int(round(duration * RATE_HZ)))

        # Resample all arms to exactly n_out steps
        arm_rs: Dict[str, np.ndarray] = {}
        for n in ROBOT_NAMES:
            pos = arm_pos[n]
            if len(pos) == n_out:
                arm_rs[n] = pos
            else:
                si = np.linspace(0, 1, len(pos)); so = np.linspace(0, 1, n_out)
                r  = np.zeros((n_out, NDOF))
                for j in range(NDOF): r[:, j] = np.interp(so, si, pos[:, j])
                arm_rs[n] = r

        t_exec = time.time()
        for k in range(n_out):
            t0n = time.monotonic_ns()
            self.publish_all({n: arm_rs[n][min(k, len(arm_rs[n]) - 1)] for n in ROBOT_NAMES})
            rclpy.spin_once(self, timeout_sec=0.)
            rem = dt_ns - (time.monotonic_ns() - t0n)
            if rem > 0: time.sleep(rem * 1e-9)
        exec_time = time.time() - t_exec

        # Hold final pose and settle — combined: hold publishes continuously,
        # settle_wait checks velocity and also keeps publishing.
        final_cmd = {n: arm_rs[n][-1] for n in ROBOT_NAMES}

        # Brief blocking hold first so controller catches up
        t_hold = time.time()
        while time.time() - t_hold < HOLD_AFTER:
            self.publish_all(final_cmd)
            rclpy.spin_once(self, timeout_sec=0.)
            time.sleep(0.02)

        # Settle-wait: keep holding while checking velocity convergence
        settle_time = self._wait_for_settle(final_cmd)

        return {
            'execution_time_s': round(exec_time, 3),
            'settle_time_s'   : round(HOLD_AFTER + settle_time, 3),
            'n_steps'         : n_out,
            'duration_s'      : round(duration, 3),
        }

    def _wait_for_settle(self, final_cmd: Dict[str, np.ndarray]) -> float:
        """
        Poll joint velocity while continuously publishing the hold command.
        Returns time spent in this function (not counting HOLD_AFTER).

        Early exit when ALL joints on BOTH arms have |Δq/Δt| < SETTLE_VEL_THR.
        Two consecutive passing checks required to avoid noise-triggered early exit.
        """
        t0  = time.time()
        prev_q: Dict[str, Optional[np.ndarray]] = {n: None for n in ROBOT_NAMES}
        pass_count = 0      # require 2 consecutive checks below threshold

        while time.time() - t0 < SETTLE_TIMEOUT:
            # Keep driving to final pose while waiting
            self.publish_all(final_cmd)
            rclpy.spin_once(self, timeout_sec=SETTLE_DT)
            time.sleep(SETTLE_DT)

            cur_q: Dict[str, Optional[np.ndarray]] = {}
            with self._lock:
                for n in ROBOT_NAMES:
                    cur_q[n] = self._js[n].copy() if self._js[n] is not None else None

            settled = True
            for n in ROBOT_NAMES:
                if cur_q[n] is None or prev_q[n] is None:
                    settled = False; break
                vel = np.max(np.abs(cur_q[n] - prev_q[n])) / SETTLE_DT
                if vel > SETTLE_VEL_THR:
                    settled = False

            prev_q = cur_q
            if settled:
                pass_count += 1
                if pass_count >= 2:   # confirmed settled
                    break
            else:
                pass_count = 0        # reset on any non-settled sample

        return round(time.time() - t0, 3)

    def read_fresh_js(self,timeout: float=VER_TIMEOUT) -> Dict[str,Optional[np.ndarray]]:
        """
        Flush stored JS and wait for brand-new messages from Gazebo.
        Equivalent to: ros2 topic echo /dsr0X/gz/joint_states --once (per arm)

        Returns the freshly-received joint positions after the robot has settled.
        """
        with self._lock:
            for n in ROBOT_NAMES: self._js[n]=None  # flush buffer
        t0=time.time()
        while time.time()-t0<timeout:
            rclpy.spin_once(self,timeout_sec=0.05)
            with self._lock:
                if all(self._js[n] is not None for n in ROBOT_NAMES):
                    return {n:self._js[n].copy() for n in ROBOT_NAMES}
        # Timeout — return whatever we have
        with self._lock:
            return {n:self._js[n].copy() if self._js[n] is not None else None
                    for n in ROBOT_NAMES}


# ─────────────────────────────────────────────────────────────────────────────
# VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def verify(read_js: Dict[str,Optional[np.ndarray]],
            target_joints: Dict[str,List],
            arms_data: Dict) -> Dict:
    """
    Compare Gazebo-measured joint states vs planned targets.
    Records joint error (deg) and EE position error (mm) per arm.
    """
    ver={'passed':True,'per_arm':{}}
    for n in ROBOT_NAMES:
        q_r=read_js.get(n); q_t=np.array(target_joints.get(n,np.zeros(NDOF).tolist()))
        base=ROBOT_BASES[n]
        if q_r is None:
            ver['per_arm'][n]={'js_received':False,'max_joint_err_deg':None,
                                'ee_err_mm':None,'joint_ok':False,'ee_ok':False}
            ver['passed']=False; continue
        j_err=np.abs(q_r-q_t); max_je=float(np.max(j_err))
        j_ok=max_je<=JOINT_VER_TOL
        ee_r=fk_pos(q_r,base)
        ee_t=np.array(arms_data[n].get('target_world',ee_r))
        ee_err=float(np.linalg.norm(ee_r-ee_t)*1000); ee_ok=ee_err<=EE_VER_TOL_MM
        if not j_ok or not ee_ok: ver['passed']=False
        ver['per_arm'][n]={
            'js_received'      : True,
            'read_joints_rad'  : q_r.tolist(),
            'read_joints_deg'  : [round(float(np.degrees(v)),3) for v in q_r],
            'target_joints_rad': q_t.tolist(),
            'target_joints_deg': [round(float(np.degrees(v)),3) for v in q_t],
            'joint_err_deg'    : [round(float(np.degrees(e)),3) for e in j_err],
            'max_joint_err_deg': round(float(np.degrees(max_je)),3),
            'ee_read_world'    : ee_r.tolist(),
            'ee_target_world'  : ee_t.tolist(),
            'ee_err_mm'        : round(ee_err,3),
            'joint_ok'         : j_ok,
            'ee_ok'            : ee_ok,
        }
    return ver


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_stats(results: List[Dict]) -> Dict:
    n=len(results); stats={}
    for oc in OUTCOMES:
        g=[r for r in results if r['outcome']==oc]; ng=len(g)
        def _mean(vals): return round(float(np.mean(vals)),3) if vals else None
        def _std(vals):  return round(float(np.std(vals)),3)  if len(vals)>1 else None
        def _med(vals):  return round(float(np.median(vals)),3) if vals else None

        plan_t  =[r['plan_time_s']  for r in g if r.get('plan_time_s')]
        ik_t    =[r['timings'].get('ik_s',0) for r in g if r.get('timings')]
        kur_t   =[r['timings'].get('kuramoto_refinement_s',0) for r in g if r.get('timings')]
        exec_t  =[r.get('execution',{}).get('execution_time_s') for r in g
                  if r.get('execution',{}).get('execution_time_s')]
        total_t =[r['total_time_s'] for r in g if r.get('total_time_s')]
        vo      =[r.get('verification',{}).get('passed',False) for r in g
                  if not r.get('verification',{}).get('skipped')]
        cl      =[r['inter_arm_clear_cm'] for r in g if r.get('inter_arm_clear_cm')]
        je_vals =[]
        ee_vals =[]
        for r in g:
            for nm in ROBOT_NAMES:
                pa=r.get('verification',{}).get('per_arm',{}).get(nm,{})
                if pa.get('js_received'):
                    if pa.get('max_joint_err_deg') is not None: je_vals.append(pa['max_joint_err_deg'])
                    if pa.get('ee_err_mm')          is not None: ee_vals.append(pa['ee_err_mm'])
        cp_r    =[r['cp_rounds'] for r in g]
        nc_raw  =[r['n_coll_raw'] for r in g if r.get('n_coll_raw',0)>0]

        stats[oc]={
            'count'               : ng,
            'percent'             : round(100*ng/max(n,1),2),
            'plan_time_mean_s'    : _mean(plan_t),
            'plan_time_std_s'     : _std(plan_t),
            'plan_time_median_s'  : _med(plan_t),
            'ik_time_mean_s'      : _mean(ik_t),
            'kuramoto_time_mean_s': _mean(kur_t),
            'exec_time_mean_s'    : _mean(exec_t),
            'total_time_mean_s'   : _mean(total_t),
            'verify_pass_pct'     : round(100*sum(vo)/max(len(vo),1),1) if vo else None,
            'avg_clear_cm'        : _mean(cl),
            'avg_joint_err_deg'   : _mean(je_vals),
            'std_joint_err_deg'   : _std(je_vals),
            'avg_ee_err_mm'       : _mean(ee_vals),
            'std_ee_err_mm'       : _std(ee_vals),
            'avg_cp_rounds'       : _mean(cp_r),
            'avg_coll_steps_when_had': _mean(nc_raw),
        }
    stats['_total']=n; stats['_seed']=None; stats['_duration_s']=None
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# PRINT TRIAL DETAIL
# ─────────────────────────────────────────────────────────────────────────────

def print_trial(tid: int, n_total: int, rec: Dict):
    thin='─'*68
    print(f'\n{thin}')
    print(f'  TRIAL {tid:>4}/{n_total}  │  Outcome: {rec["outcome"]}')
    print(f'{thin}')

    # Inputs — start config + target
    for n in ROBOT_NAMES:
        tgt = rec['targets'].get(n,{})
        sq  = rec.get('start_joints',{}).get(n,[])
        sq_d= [round(float(np.degrees(v)),2) for v in sq] if sq else []
        print(f'  [{n}] start  (deg): {sq_d}')
        print(f'  [{n}] target pos  : {[round(v,4) for v in tgt.get("pos",[])]}  m')
        print(f'  [{n}] target quat : {[round(v,4) for v in tgt.get("quat",[])]}')

    # Planning breakdown
    t=rec.get('timings',{}); pt=rec.get('plan_time_s','—')
    print(f'\n  PLANNING  (total={pt}s)')
    print(f'    IK           : {t.get("ik_s","—")}s')
    print(f'    B-spline     : {t.get("bspline_s","—")}s')
    print(f'    Collision chk: {t.get("collision_check_s","—")}s')
    print(f'    Kur+refine   : {t.get("kuramoto_refinement_s","—")}s')
    print(f'    Had collision: {rec.get("had_collision")}  ({rec.get("n_coll_raw",0)} raw steps)')
    if rec.get('n_coll_final',0)>0:
        print(f'    Final coll   : {rec["n_coll_final"]} steps after Kuramoto+CPs')
    print(f'    CP rounds    : {rec.get("cp_rounds",0)}')
    print(f'    Arm clearance: {rec.get("inter_arm_clear_cm","—")}cm  (at IK configs)')
    if rec.get('kur_min_dist_cm'):
        print(f'    Kur min dist : {rec["kur_min_dist_cm"]}cm')
    for n in ROBOT_NAMES:
        plen=rec.get('ee_path_length_m',{}).get(n)
        if plen: print(f'    [{n}] EE path : {plen*100:.1f}cm')

    # Execution
    ex=rec.get('execution',{})
    if ex.get('skipped'):
        print(f'\n  EXECUTION : SKIPPED  ({ex.get("reason","")})')
    else:
        print(f'\n  EXECUTION')
        print(f'    Duration    : {ex.get("duration_s","—")}s  (trajectory)')
        print(f'    Wall time   : {ex.get("execution_time_s","—")}s')
        print(f'    Settle wait : {ex.get("settle_time_s","—")}s')
        print(f'    Steps sent  : {ex.get("n_steps","—")}')

    # Verification
    ver=rec.get('verification',{})
    if ver.get('skipped'):
        print(f'\n  VERIFICATION: SKIPPED')
    else:
        vok='✅ PASSED' if ver.get('passed') else '❌ FAILED'
        print(f'\n  VERIFICATION: {vok}')
        for n in ROBOT_NAMES:
            pa=ver.get('per_arm',{}).get(n,{})
            if not pa.get('js_received'):
                print(f'  [{n}] ❌  No JS received from Gazebo'); continue
            jok='✅' if pa.get('joint_ok') else '❌'
            eok='✅' if pa.get('ee_ok')    else '❌'
            print(f'  [{n}] JS ✅  '
                  f'joint_err={pa.get("max_joint_err_deg","—")}°{jok}  '
                  f'ee_err={pa.get("ee_err_mm","—")}mm{eok}')
            print(f'  [{n}] read  (deg): {pa.get("read_joints_deg",[])}')
            print(f'  [{n}] target(deg): {pa.get("target_joints_deg",[])}')

    # Next trial start
    nxt=rec.get('next_start_joints',{})
    if nxt:
        print(f'\n  NEXT TRIAL START:')
        for n in ROBOT_NAMES:
            q=nxt.get(n,[])
            if q: print(f'  [{n}] {[round(float(np.degrees(v)),2) for v in q]} deg')

    print(f'\n  Total time: {rec.get("total_time_s","—")}s')


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLE  (console)
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(stats: Dict, n_trials: int, seed: int, duration: float):
    bar='═'*72; thin='─'*72
    print(f'\n{bar}')
    print(f'  STEP 6  —  BENCHMARK SUMMARY')
    print(f'  Trials: {n_trials}  |  Seed: {seed}  |  Duration: {duration}s/trial')
    print(f'{bar}')

    # Main outcome table
    print(f'\n  {"Outcome":<20} {"N":>5} {"Pct":>6}  '
          f'{"Plan(s)":>9}  {"Exec(s)":>9}  '
          f'{"JointErr°":>10}  {"EEErr mm":>9}  {"Verify%":>8}')
    print(f'  {thin}')

    for oc in OUTCOMES:
        s  =stats.get(oc,{}); n=s.get('count',0); pct=s.get('percent',0.)
        pt =f'{s["plan_time_mean_s"]:.2f}' if s.get('plan_time_mean_s') else '    —   '
        et =f'{s["exec_time_mean_s"]:.2f}' if s.get('exec_time_mean_s') else '    —   '
        je =f'{s["avg_joint_err_deg"]:.2f}' if s.get('avg_joint_err_deg') else '    —   '
        ee =f'{s["avg_ee_err_mm"]:.2f}'    if s.get('avg_ee_err_mm')    else '    —   '
        vr =f'{s["verify_pass_pct"]:.0f}%' if s.get('verify_pass_pct') is not None else '   —  '
        icon='❌' if oc=='FAIL_IK' else '⚠ ' if oc=='UNRESOLVED' else '✅'
        bar_v='█'*max(0,int(pct/2))
        print(f'  {icon} {oc:<19} {n:>5}  {pct:>5.1f}%  '
              f'{pt:>9}  {et:>9}  '
              f'{je:>10}  {ee:>9}  {vr:>8}  {bar_v}')

    # Grouped counts
    nf=stats.get('FAIL_IK',{}).get('count',0)
    ns=stats.get('SAFE_NO_COLL',{}).get('count',0)
    nk=stats.get('RESOLVED_KUR',{}).get('count',0)
    nc=sum(stats.get(f'RESOLVED_CP_{i}',{}).get('count',0) for i in range(1,6))
    nu=stats.get('UNRESOLVED',{}).get('count',0)
    nr=ns+nk+nc

    print(f'\n  {thin}  GROUPED RESULTS')
    for label,count in [
        ('IK Failed',                   nf),
        ('Trivially safe (no collision)',ns),
        ('Resolved — Kuramoto only',     nk),
        ('Resolved — Kuramoto + CPs',    nc),
        ('Unresolved',                   nu),
    ]:
        pct=100*count/max(n_trials,1)
        print(f'  {label:<40}: {count:>4}  ({pct:5.1f}%)')
    print(f'  {"─"*55}')
    print(f'  {"Total resolved (safe + resolved)":<40}: {nr:>4}  ({100*nr/max(n_trials,1):5.1f}%)')

    # Global timing stats
    all_plan=[r for r in [stats.get(oc,{}).get('plan_time_mean_s') for oc in OUTCOMES] if r]
    print(f'\n  {thin}  TIMING')
    # Show per-outcome plan time
    for oc in OUTCOMES:
        s=stats.get(oc,{}); n=s.get('count',0)
        if n==0: continue
        pt=s.get('plan_time_mean_s'); ps=s.get('plan_time_std_s')
        if pt: print(f'  {oc:<20}: plan {pt:.2f}s ± {ps if ps else "—"}s  '
                      f'ik={s.get("ik_time_mean_s","—")}s  '
                      f'kur={s.get("kuramoto_time_mean_s","—")}s')

    print(f'\n{bar}\n')


# ─────────────────────────────────────────────────────────────────────────────
# PAPER-READY TEXT REPORT
# ─────────────────────────────────────────────────────────────────────────────

def write_paper_report(stats: Dict, results: List[Dict],
                        n_trials: int, seed: int, duration: float,
                        start_time: str) -> str:
    """
    Generate a structured text report suitable for paper submission.
    Saved to benchmark_report.txt.
    """
    lines = []
    sep = '=' * 70

    lines += [
        sep,
        'DUAL-ARM SYNCHRONIZED TRAJECTORY GENERATION — BENCHMARK REPORT',
        sep,
        f'Date/Time   : {start_time}',
        f'Trials      : {n_trials}',
        f'Random seed : {seed}',
        f'Duration/trial: {duration}s',
        f'Robot       : Doosan M1013 × 2 (dsr01 + dsr02)',
        f'Base sep.   : 1.0 m (y = +0.5 and -0.5)',
        f'Control rate: {RATE_HZ} Hz',
        f'B-spline    : {N_SEG} segments × {N_CP_SEG} CPs, degree {DEG}',
        f'Safety mar. : {SAFETY_MARGIN*100:.0f} cm (sphere model)',
        f'Max refine  : {MAX_REFINE} rounds',
        '',
        '── OUTCOME DISTRIBUTION ' + '─'*46,
    ]

    for oc in OUTCOMES:
        s=stats.get(oc,{}); n=s.get('count',0); pct=s.get('percent',0.)
        lines.append(f'  {oc:<20}: {n:>4}  ({pct:5.1f}%)')

    nf=stats.get('FAIL_IK',{}).get('count',0)
    ns=stats.get('SAFE_NO_COLL',{}).get('count',0)
    nk=stats.get('RESOLVED_KUR',{}).get('count',0)
    nc=sum(stats.get(f'RESOLVED_CP_{i}',{}).get('count',0) for i in range(1,6))
    nu=stats.get('UNRESOLVED',{}).get('count',0)
    nr=ns+nk+nc

    lines += [
        '',
        f'  IK failure rate      : {100*nf/max(n_trials,1):.1f}%',
        f'  No collision rate    : {100*ns/max(n_trials,1):.1f}%',
        f'  Resolved (Kur only) : {100*nk/max(n_trials,1):.1f}%',
        f'  Resolved (Kur+CPs)  : {100*nc/max(n_trials,1):.1f}%',
        f'  Unresolved rate      : {100*nu/max(n_trials,1):.1f}%',
        f'  Overall success rate : {100*nr/max(n_trials,1):.1f}%',
        '',
        '── PLANNING TIME (seconds) ' + '─'*43,
    ]

    for oc in OUTCOMES:
        s=stats.get(oc,{}); n=s.get('count',0)
        if n==0: continue
        pt=s.get('plan_time_mean_s'); ps=s.get('plan_time_std_s')
        pm=s.get('plan_time_median_s')
        ikt=s.get('ik_time_mean_s'); kt=s.get('kuramoto_time_mean_s')
        lines.append(f'  {oc:<20}  mean={pt}s  std={ps}s  med={pm}s  '
                      f'(ik={ikt}s  kur={kt}s)')

    lines += ['', '── VERIFICATION ACCURACY ' + '─'*45]
    for oc in OUTCOMES:
        s=stats.get(oc,{}); n=s.get('count',0)
        if n==0 or s.get('verify_pass_pct') is None: continue
        je=s.get('avg_joint_err_deg'); jes=s.get('std_joint_err_deg')
        ee=s.get('avg_ee_err_mm');     ees=s.get('std_ee_err_mm')
        vp=s.get('verify_pass_pct')
        lines.append(f'  {oc:<20}  verify={vp}%  '
                      f'joint_err={je}±{jes}°  ee_err={ee}±{ees}mm')

    lines += ['', '── COLLISION STATISTICS ' + '─'*46]
    # Count how many had collision initially
    had_c = sum(1 for r in results if r.get('had_collision'))
    lines.append(f'  Trials with initial collision  : {had_c}/{n_trials} ({100*had_c/max(n_trials,1):.1f}%)')
    # Average collision steps
    coll_steps=[r['n_coll_raw'] for r in results if r.get('n_coll_raw',0)>0]
    if coll_steps:
        lines.append(f'  Avg collision steps (when had): {np.mean(coll_steps):.1f} ± {np.std(coll_steps):.1f}')
    # Clearance at IK configs
    clears=[r['inter_arm_clear_cm'] for r in results if r.get('inter_arm_clear_cm')]
    if clears:
        lines.append(f'  Inter-arm clearance at configs: {np.mean(clears):.1f} ± {np.std(clears):.1f} cm')

    lines += ['', sep, 'END OF REPORT', sep, '']
    text='\n'.join(lines)
    with open('benchmark_report.txt','w') as fh: fh.write(text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(args=None):
    parser=argparse.ArgumentParser(
        description='Dual-arm Gazebo benchmark for paper evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--trials',  type=int,   default=100,  help='Number of trials')
    parser.add_argument('--seed',    type=int,   default=0,    help='Random seed')
    parser.add_argument('--duration',type=float, default=10.0, help='Trajectory duration per trial (s)')
    ap=parser.parse_args()

    n_trials=ap.trials; rng=np.random.default_rng(ap.seed); duration=ap.duration
    start_time=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print('\n'+'='*72)
    print(f'  STEP 6  —  Gazebo Benchmark  ({n_trials} trials  seed={ap.seed}  dur={duration}s)')
    print('='*72)
    print(f'\n  Trial flow:')
    print(f'    Read JS → plan → execute Gazebo 100Hz → hold 1s → fresh JS read')
    print(f'    → verify → use verified JS as START for next trial\n')
    print(f'  Outcomes tracked: {OUTCOMES}\n')

    rclpy.init(args=args)
    node=BenchmarkNode()

    print('  Waiting for Gazebo joint states ...')
    if not node.wait_for_js(30.):
        print('  ❌  Timeout — check Gazebo and dsr_position_controller')
        node.destroy_node(); rclpy.shutdown(); sys.exit(1)
    print('  ✅  Gazebo ready\n')

    results: List[Dict]=[]
    counts={oc:0 for oc in OUTCOMES}
    t_bench=time.time()

    # Trial 1 start: read current Gazebo joint states
    current_start_qs: Dict[str,np.ndarray]={}
    for n in ROBOT_NAMES:
        q=node.get_joints(n)
        current_start_qs[n]=q if q is not None else np.zeros(NDOF)
    print(f'  Initial joint states:')
    for n in ROBOT_NAMES:
        print(f'    [{n}] {[round(float(np.degrees(v)),2) for v in current_start_qs[n]]} deg')
    print()

    for tid in range(1, n_trials+1):
        t_trial=time.time()

        # Generate random targets
        arms_data: Dict={}; targets: Dict={}
        for n in ROBOT_NAMES:
            base=ROBOT_BASES[n]; pw,quat=random_target(rng,base)
            arms_data[n]={'base':base,'target_world':pw,'tloc':pw-base,'trot':quat_to_rot(quat)}
            targets[n]={'pos':pw.tolist(),'quat':quat.tolist()}

        # Plan
        plan_res=plan(arms_data, current_start_qs, duration)

        rec: Dict={
            'trial_id'          : tid,
            'outcome'           : plan_res['outcome'],
            'had_collision'     : plan_res['had_collision'],
            'n_coll_raw'        : plan_res['n_coll_raw'],
            'n_coll_final'      : plan_res['n_coll_final'],
            'cp_rounds'         : plan_res['cp_rounds'],
            'plan_time_s'       : plan_res['plan_time_s'],
            'timings'           : plan_res['timings'],
            'inter_arm_clear_cm': plan_res['inter_arm_clear_cm'],
            'kur_min_dist_cm'   : plan_res['kur_min_dist_cm'],
            'ee_path_length_m'  : plan_res['ee_path_length_m'],
            'targets'           : targets,
            'start_joints'      : {n:current_start_qs[n].tolist() for n in ROBOT_NAMES},
            'target_joints'     : plan_res['target_joints'],
            'execution'         : {},
            'verification'      : {},
            'next_start_joints' : {},
            'total_time_s'      : None,
        }

        # Execute + verify (only skip if IK totally failed)
        if plan_res['outcome']!='FAIL_IK' and plan_res['arm_pos'] is not None:
            # Execute in Gazebo
            ex_res=node.execute_trajectory(plan_res['arm_pos'], plan_res['duration'])
            rec['execution']={**ex_res}

            # Read fresh JS — ≡ ros2 topic echo --once per arm
            read_js=node.read_fresh_js(timeout=VER_TIMEOUT)

            # Verify
            rec['verification']=verify(read_js, plan_res['target_joints'], arms_data)

            # Update start config for next trial from verified Gazebo JS
            for n in ROBOT_NAMES:
                q=read_js.get(n)
                # Use actual Gazebo reading; fallback to planned target if JS timed out
                current_start_qs[n] = q.copy() if q is not None else \
                                       np.array(plan_res['target_joints'].get(n,np.zeros(NDOF)))
            rec['next_start_joints']={n:current_start_qs[n].tolist() for n in ROBOT_NAMES}

        else:
            # IK failed — arms didn't move, keep current_start_qs unchanged
            rec['execution']={'skipped':True,'reason':plan_res['outcome']}
            rec['verification']={'passed':False,'skipped':True}
            rec['next_start_joints']={n:current_start_qs[n].tolist() for n in ROBOT_NAMES}

        rec['total_time_s']=round(time.time()-t_trial,3)
        results.append(rec); counts[plan_res['outcome']]+=1

        # Print trial
        print_trial(tid, n_trials, rec)

        # Progress line every 10 trials
        if tid%10==0 or tid==n_trials:
            el=time.time()-t_bench; eta=el/tid*(n_trials-tid)
            parts=[
                f'{oc.replace("RESOLVED_","R_CP_").replace("SAFE_NO_COLL","SAFE").replace("FAIL_IK","FAIL").replace("UNRESOLVED","UNRES").replace("R_CP_KUR","KUR")}:{counts[oc]}'
                for oc in OUTCOMES if counts[oc]>0
            ]
            print(f'\n  ── {tid}/{n_trials}  elapsed={el:.0f}s  eta={eta:.0f}s  |  '+'  '.join(parts))

    # ── Final stats + outputs ──────────────────────────────────────────────────
    stats=compute_stats(results)
    stats['_seed']=ap.seed; stats['_duration_s']=duration
    print_summary(stats, n_trials, ap.seed, duration)

    # Write paper report
    report_text=write_paper_report(stats,results,n_trials,ap.seed,duration,start_time)

    # Save JSON files
    with open('trial_results.json','w') as fh:
        json.dump({'seed':ap.seed,'n_trials':n_trials,
                   'duration_s':duration,'results':results},fh,indent=2)
    with open('trial_summary.json','w') as fh:
        json.dump({'seed':ap.seed,'n_trials':n_trials,
                   'duration_s':duration,'stats':stats},fh,indent=2)

    kb1=os.path.getsize('trial_results.json')/1024.
    kb2=os.path.getsize('trial_summary.json')/1024.
    kb3=os.path.getsize('benchmark_report.txt')/1024.
    print(f'  ✅  trial_results.json   ({kb1:.1f} KB)')
    print(f'  ✅  trial_summary.json   ({kb2:.1f} KB)')
    print(f'  ✅  benchmark_report.txt ({kb3:.1f} KB)\n')

    # Print paper report to console as well
    print(report_text)

    node.destroy_node(); rclpy.shutdown()


if __name__=='__main__':
    main()