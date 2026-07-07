#!/usr/bin/env python3
"""
step_16.py  —  Automated Gazebo Execution Benchmark  (4-arm)
═══════════════════════════════════════════════════════════════════════════════
PURPOSE
───────
Automated N-trial Gazebo benchmark for the 4-arm synchronized trajectory
generation pipeline. Evaluates all C(4,2)=6 inter-arm pairs.

ARM LAYOUT  (matches multi_arm_gazebo.launch.py)
─────────────────────────────────────────────────
  dsr01  x=0.0  y=+0.5      dsr03  x=1.0  y=+0.5
  dsr02  x=0.0  y=-0.5      dsr04  x=1.0  y=-0.5

TRIAL FLOW
──────────
1. Read Gazebo joint states → start config (4 arms)
2. Generate random target pose per arm
3. Run full pipeline: IK → B-spline → collision check (6 pairs) →
   Kuramoto sync + alternating IK CP refinement (≤5 rounds)
4. Execute synchronized trajectory at 100 Hz lock-step (4 arms)
5. Hold + settle → read fresh JS → verify → use as next start

OUTCOMES  (9 classes, same as step_6)
──────────────────────────────────────
FAIL_IK / SAFE_NO_COLL / RESOLVED_KUR /
RESOLVED_CP_1..5 / UNRESOLVED

Usage:
  ros2 run dual_arm_sync step_16
  ros2 run dual_arm_sync step_16 --trials 100 --seed 42 --duration 10.0
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
    print('[step_16] ❌  rclpy not found — ROS2 required'); sys.exit(1)

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

# Matches multi_arm_gazebo.launch.py exactly
ROBOT_BASES: Dict[str, np.ndarray] = {
    'dsr01': np.array([0.0,  0.5, 0.0]),
    'dsr02': np.array([0.0, -0.5, 0.0]),
    'dsr03': np.array([1.0,  0.5, 0.0]),
    'dsr04': np.array([1.0, -0.5, 0.0]),
}
ROBOT_NAMES = ['dsr01', 'dsr02', 'dsr03', 'dsr04']

LINK_RADII    = np.array([0.10, 0.10, 0.08, 0.08, 0.06, 0.06])
SAFETY_MARGIN = 0.12
CTRL_TOPIC    = '/{arm}/gz/dsr_position_controller/commands'
JS_TOPICS     = ['/{arm}/gz/joint_states', '/{arm}/joint_states']

# Pipeline params
N_SEG      = 5;  N_CP_SEG  = 4;  DEG       = 3
RATE_HZ    = 100.0; MAX_REFINE = 5; CP_INC  = 2
IK_TOL_POS = 0.010; IK_TOL_ROT = 0.05; IK_MAX_ITER = 600
IK_FTOL    = 1e-10; IK_UNIQ = 0.12; W_POS = 1.0; W_ROT = 0.15

K_BASE=5.0; K_REPULSE=80.0; K_EMERGENCY=250.0                                       # line 96 for k change
KUR_DT=0.01; MIN_SAFE=0.15; REPULSE_D=0.28; LEADER_THRESH=0.05

REACH_MIN=0.40; REACH_MAX=1.10; Z_MIN=0.10; Z_MAX=1.00

HOLD_AFTER=0.8; VER_TIMEOUT=5.0; SETTLE_TIMEOUT=2.0
SETTLE_VEL_THR=0.008; SETTLE_DT=0.04
JOINT_VER_TOL=0.15; EE_VER_TOL_MM  = 25.0

OUTCOMES = ['FAIL_IK','SAFE_NO_COLL','RESOLVED_KUR',
            'RESOLVED_CP_1','RESOLVED_CP_2','RESOLVED_CP_3',
            'RESOLVED_CP_4','RESOLVED_CP_5','UNRESOLVED']

# ─────────────────────────────────────────────────────────────────────────────
# FK + GEOMETRY
# ─────────────────────────────────────────────────────────────────────────────

def fk(q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    T = np.eye(4)
    for i in range(NDOF):
        al,a,to,d=DH[i]; th=q[i]+to; ct,st=np.cos(th),np.sin(th); ca,sa=np.cos(al),np.sin(al)
        T=T@np.array([[ct,-st,0.,a],[st*ca,ct*ca,-sa,-sa*d],
                       [st*sa,ct*sa,ca,ca*d],[0.,0.,0.,1.]])
    return T[:3,3].copy(), T

def fk_pos(q, base): return fk(q)[0] + base

def link_origins(q, base):
    T=np.eye(4); o=np.zeros((NDOF,3))
    for i in range(NDOF):
        al,a,to,d=DH[i]; th=q[i]+to; ct,st=np.cos(th),np.sin(th); ca,sa=np.cos(al),np.sin(al)
        T=T@np.array([[ct,-st,0.,a],[st*ca,ct*ca,-sa,-sa*d],
                       [st*sa,ct*sa,ca,ca*d],[0.,0.,0.,1.]])
        o[i]=T[:3,3]+base
    return o

def rot_err(Ra, Rb):
    R=Rb@Ra.T; return float(np.arccos(np.clip((np.trace(R)-1)/2,-1,1)))

def pair_min_dist(qi, bi, qj, bj):
    """Minimum link-origin distance — vectorised broadcasting."""
    oi = link_origins(qi, bi)  # (NDOF,3)
    oj = link_origins(qj, bj)  # (NDOF,3)
    diff = oi[:, np.newaxis, :] - oj[np.newaxis, :, :]  # (NDOF,NDOF,3)
    return float(np.min(np.linalg.norm(diff, axis=2)))

def pair_collides(qi,bi,qj,bj):
    oi=link_origins(qi,bi); oj=link_origins(qj,bj)
    for a in range(NDOF):
        for b in range(NDOF):
            if np.linalg.norm(oi[a]-oj[b])<LINK_RADII[a]+LINK_RADII[b]+SAFETY_MARGIN: return True
    return False

def interp_pos(pos, frac):
    frac=float(np.clip(frac,0,1)); n=len(pos)-1
    if n<=0: return pos[0].copy()
    idx=min(int(frac*n),n-1); a=frac*n-idx; return pos[idx]+a*(pos[idx+1]-pos[idx])

def all_pairs_collision_free(arm_pos_dict, arm_bases):
    """True only if ALL C(N,2) pairs are collision-free at this timestep."""
    names = list(arm_pos_dict.keys())
    for i in range(len(names)):
        for j in range(i+1,len(names)):
            if pair_collides(arm_pos_dict[names[i]], arm_bases[names[i]],
                              arm_pos_dict[names[j]], arm_bases[names[j]]):
                return False
    return True

def count_all_collisions(arm_pos_seqs, arm_bases):
    """Count timesteps where ANY pair collides."""
    names = list(arm_pos_seqs.keys())
    K = min(len(arm_pos_seqs[n]) for n in names); nc = 0
    for k in range(K):
        qk = {n: arm_pos_seqs[n][k] for n in names}
        if not all_pairs_collision_free(qk, arm_bases): nc += 1
    return nc

# ─────────────────────────────────────────────────────────────────────────────
# QUATERNION + RANDOM TARGET
# ─────────────────────────────────────────────────────────────────────────────

def quat_norm(q): n=np.linalg.norm(q); return q/n if n>1e-12 else np.array([1.,0.,0.,0.])

def quat_to_rot(q):
    w,x,y,z=quat_norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])

def random_quat(rng):
    u1,u2,u3=rng.random(),rng.random(),rng.random()
    return quat_norm(np.array([np.sqrt(1-u1)*np.sin(2*_PI*u2),
                                np.sqrt(1-u1)*np.cos(2*_PI*u2),
                                np.sqrt(u1)*np.sin(2*_PI*u3),
                                np.sqrt(u1)*np.cos(2*_PI*u3)]))

def random_target(rng, base):
    for _ in range(300):
        r=rng.uniform(REACH_MIN,REACH_MAX); theta=rng.uniform(0,2*_PI); z=rng.uniform(Z_MIN,Z_MAX)
        r_xy=np.sqrt(max(r**2-z**2,0.)); x_loc=r_xy*np.cos(theta); y_loc=r_xy*np.sin(theta)
        if np.hypot(x_loc,y_loc)>A+0.05:
            return np.array([x_loc,y_loc,z])+base, random_quat(rng)
    return base+np.array([0.6,0.,0.5]), np.array([1.,0.,0.,0.])

# ─────────────────────────────────────────────────────────────────────────────
# IK
# ─────────────────────────────────────────────────────────────────────────────

def _ik_seeds(current, tgt):
    px,py,pz=tgt; cl=lambda q:np.clip(q,POS_LIM[:,0],POS_LIM[:,1])
    # FIX: removed np.zeros(NDOF) seed (biases toward home config)
    seeds=[cl(current.copy())]
    for j in range(NDOF):
        for d in (0.3,-0.3,0.6,-0.6): s=current.copy(); s[j]+=d; seeds.append(cl(s))
    t1=float(np.arctan2(py,px)); rh=float(np.hypot(px,py))
    re=float(np.sqrt(max(rh**2-A**2,0.))); h=float(pz-L1)
    c3=float(np.clip((re**2+h**2-L2**2-L3**2)/(2*L2*L3),-1,1))
    for sgn in (1.,-1.):
        th3=sgn*float(np.arccos(c3)); q3=th3-_PI_2
        th2=float(np.arctan2(h,re))-float(np.arctan2(L3*np.sin(th3),L2+L3*np.cos(th3)))
        q2=th2+_PI_2
        for q5 in (0.,_PI_2,-_PI_2):
            for t in (t1,t1+_PI_2,t1-_PI_2): seeds.append(cl(np.array([t,q2,q3,0.,q5,0.])))
    uniq: List[np.ndarray]=[]
    for s in seeds:
        if all(np.linalg.norm(s-u)>0.05 for u in uniq): uniq.append(s)
    return uniq

def solve_ik(tloc, trot, cur) -> List[np.ndarray]:
    bds=[(POS_LIM[i,0],POS_LIM[i,1]) for i in range(NDOF)]
    def obj(q): p,T=fk(q); return W_POS*float(np.sum((p-tloc)**2))+W_ROT*float(rot_err(T[:3,:3],trot)**2)
    sols=[]
    for seed in _ik_seeds(cur,tloc):
        res=minimize(obj,seed,method='SLSQP',bounds=bds,options={'maxiter':IK_MAX_ITER,'ftol':IK_FTOL})
        if not res.success: continue
        q=np.clip(res.x,POS_LIM[:,0],POS_LIM[:,1]); p,T=fk(q)
        if np.linalg.norm(p-tloc)>=IK_TOL_POS or rot_err(T[:3,:3],trot)>=IK_TOL_ROT: continue
        if all(np.linalg.norm(q-s)>IK_UNIQ for s in sols): sols.append(q)
    return sols

def find_best_configs(arms_data, start_qs) -> Optional[Dict[str,np.ndarray]]:
    """
    Round 1: SE(3) IK per arm.
    Round 2: re-score maximising clearance from ALL other arms.
    """
    an=sorted(arms_data.keys()); all_sols={}; best={}
    for name in an:
        d=arms_data[name]; sq=start_qs.get(name,np.zeros(NDOF))
        sols=solve_ik(d['tloc'],d['trot'],sq)
        if not sols:
            bds=[(POS_LIM[i,0],POS_LIM[i,1]) for i in range(NDOF)]
            def po(q,tl=d['tloc']): p,_=fk(q); return float(np.sum((p-tl)**2))
            for seed in _ik_seeds(sq,d['tloc']):
                res=minimize(po,seed,method='SLSQP',bounds=bds,options={'maxiter':IK_MAX_ITER,'ftol':IK_FTOL})
                q=np.clip(res.x,POS_LIM[:,0],POS_LIM[:,1])
                if np.linalg.norm(fk(q)[0]-d['tloc'])<IK_TOL_POS: sols.append(q); break
        if not sols: return None
        all_sols[name]=sols
        # FIX Round-1: bottleneck time = max(|Δq_i|/VEL_LIM_i), not Euclidean norm
        best[name]=min(sols, key=lambda q, s=sq: float(np.max(np.abs(q - s) / VEL_LIM)))

    # Re-score: clearance (primary) + velocity-weighted sq-diff from start (secondary)
    # FIX: was clearance-only; now adds time score so configs reachable faster score higher
    for name in an:
        base    = arms_data[name]['base']
        sq_ref  = start_qs.get(name, np.zeros(NDOF))
        others  = [(best[n],arms_data[n]['base']) for n in an if n!=name]
        def _sc(q, b=base, oth=others, sq=sq_ref):
            clr = min(float(np.tanh(pair_min_dist(q,b,oq,ob)/0.25)) for oq,ob in oth) if oth else 1.0
            cost = float(np.sum(((q - sq) / VEL_LIM) ** 2))
            time_sc = float(np.exp(-cost / 25.0))
            return 3.0*clr + 2.0*time_sc
        scores=[_sc(q) for q in all_sols[name]]
        best[name]=all_sols[name][int(np.argmax(scores))]
    return best



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
# B-SPLINE
# ─────────────────────────────────────────────────────────────────────────────

def make_knots(ncp, deg=DEG):
    ni=max(0,ncp-deg-1); inn=np.linspace(0,1,ni+2)[1:-1] if ni>0 else np.array([])
    return np.concatenate([np.zeros(deg+1),inn,np.ones(deg+1)])

def build_cp(sq, eq, n_seg=N_SEG, n_cp=N_CP_SEG):
    total=n_seg*(n_cp-1)+1; s=np.linspace(0.,1.,total)
    cpg=sq[np.newaxis,:]+s[:,np.newaxis]*(eq-sq)[np.newaxis,:]
    cpg=np.clip(cpg,POS_LIM[:,0],POS_LIM[:,1])
    cps=np.zeros((n_seg,n_cp,NDOF))
    for seg in range(n_seg): i0=seg*(n_cp-1); cps[seg]=cpg[i0:i0+n_cp]
    return cps

def seg_list_from_cp(cps, n_seg=N_SEG, n_cp=N_CP_SEG):
    kn=make_knots(n_cp); return [(cps[s].copy(),kn.copy()) for s in range(n_seg)]

def eval_global(sl, duration):
    all_cp=[]
    for si,(cp,_) in enumerate(sl): all_cp.append(cp if si==0 else cp[1:])
    cpg=np.vstack(all_cp); kn=make_knots(len(cpg))
    ns=max(2,int(round(duration*RATE_HZ))); sf=np.linspace(0.,1.,ns)
    pos=np.zeros((ns,NDOF)); vel=pos.copy(); acc=pos.copy()
    for j in range(NDOF):
        spl=BSpline(kn,cpg[:,j],DEG,extrapolate=True)
        pos[:,j]=spl(sf); vel[:,j]=spl.derivative(1)(sf)/duration; acc[:,j]=spl.derivative(2)(sf)/duration**2
    return np.clip(pos,POS_LIM[:,0],POS_LIM[:,1]),vel,acc,np.linspace(0.,duration,ns)

def scale_duration(sl, duration):
    pos,vel,acc,t=eval_global(sl,duration); sv=sa=1.0
    for j in range(NDOF):
        vp=float(np.max(np.abs(vel[:,j]))); ap=float(np.max(np.abs(acc[:,j])))
        if vp>VEL_LIM[j]: sv=max(sv,vp/VEL_LIM[j])
        if ap>ACC_LIM[j]: sa=max(sa,float(np.sqrt(ap/ACC_LIM[j])))
    sc=max(sv,sa)
    if sc>1.0: duration=duration*sc*1.05; pos,vel,acc,t=eval_global(sl,duration)
    return pos,vel,acc,t,duration

# ─────────────────────────────────────────────────────────────────────────────
# COLLISION + KURAMOTO  (N-arm)
# ─────────────────────────────────────────────────────────────────────────────

def count_collisions_all_pairs(arm_pos, arm_bases):
    """Count timesteps with ANY collision across all C(N,2) pairs."""
    names=list(arm_pos.keys()); K=min(len(arm_pos[n]) for n in names); nc=0; first=None
    for k in range(K):
        qk={n:arm_pos[n][k] for n in names}
        if not all_pairs_collision_free(qk, arm_bases):
            nc+=1
            if first is None: first=k
    return nc, (first/max(K-1,1) if first is not None else None)

def first_coll_seg_for_pair(pi,pj,bi,bj,n_seg):
    N=min(len(pi),len(pj))
    for k in range(N):
        if pair_collides(pi[k],bi,pj[k],bj):
            frac=k/max(N-1,1); return min(int(frac*n_seg),n_seg-1),frac
    return None,None

def run_kuramoto(arm_names, arm_pos, arm_bases, duration):
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
            danger=float(np.clip(1-dist/MIN_SAFE,0,1)); diff=phi[i]-phi[j]
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
        pr[f'{arm_names[i]}↔{arm_names[j]}']={'min_dist_m':float(np.min(dv)),'critical':nc,'collision_free':nc==0}
    return sync,{'pair_reports':pr,'total_critical':tc,'collision_free':tc==0}

# ─────────────────────────────────────────────────────────────────────────────
# ALTERNATING IK + B-SPLINE CP REBUILD  (clearance from ALL arms)
# ─────────────────────────────────────────────────────────────────────────────

def _ik_away_multi(tloc, cur, this_base, all_others):
    """Position-only IK; maximises min distance to ALL other arms."""
    bds=[(POS_LIM[i,0],POS_LIM[i,1]) for i in range(NDOF)]; valid=[]
    for seed in _ik_seeds(cur,tloc):
        def obj(q,_t=tloc): p,_=fk(q); return float(np.sum((p-_t)**2))
        res=minimize(obj,seed,method='SLSQP',bounds=bds,options={'maxiter':300,'ftol':1e-9})
        if not res.success: continue
        q=np.clip(res.x,POS_LIM[:,0],POS_LIM[:,1])
        if np.linalg.norm(fk(q)[0]-tloc)<0.012: valid.append(q)
    if not valid: return None
    return max(valid, key=lambda q: min(pair_min_dist(q,this_base,oq,ob) for oq,ob in all_others) if all_others else 1.0)

def alternating_ik(arm_names, pair_ni, pair_nj, arm_pos, arm_arc, arm_bases, s0, s1, n_wp):
    pi=arm_pos[pair_ni]; pj=arm_pos[pair_nj]
    bi=arm_bases[pair_ni]; bj=arm_bases[pair_nj]
    arc=arm_arc[pair_ni]; Nj=len(pj)
    mask=(arc>=s0-1e-6)&(arc<=s1+1e-6); idx=np.where(mask)[0]
    if len(idx)<2: return None,None
    samp=np.unique(np.linspace(0,len(idx)-1,n_wp,dtype=int))
    wps_i=[]; wps_j=[]; pqi=pi[idx[0]].copy(); pqj=pj[min(idx[0],Nj-1)].copy()
    other_names=[n for n in arm_names if n!=pair_ni and n!=pair_nj]
    for si in samp:
        k=idx[si]; kj=min(k,Nj-1); ee_i=fk(pi[k])[0]; ee_j=fk(pj[kj])[0]
        others_for_i=[(pqj,bj)]+[(arm_pos[n][min(k,len(arm_pos[n])-1)],arm_bases[n]) for n in other_names]
        qi_new=_ik_away_multi(ee_i,pqi,bi,others_for_i)
        if qi_new is None: qi_new=pi[k].copy()
        others_for_j=[(qi_new,bi)]+[(arm_pos[n][min(k,len(arm_pos[n])-1)],arm_bases[n]) for n in other_names]
        qj_new=_ik_away_multi(ee_j,pqj,bj,others_for_j)
        if qj_new is None: qj_new=pj[kj].copy()
        wps_i.append(qi_new); wps_j.append(qj_new); pqi=qi_new; pqj=qj_new
    return np.array(wps_i),np.array(wps_j)

def _fit_cp(wps,sv,ncp,ps,pe):
    ncp=max(DEG+2,ncp); kn=make_knots(ncp)
    B=np.zeros((len(sv),ncp)); e=np.zeros(ncp)
    for k in range(ncp):
        e[:]=0.; e[k]=1.; v=BSpline(kn,e.copy(),DEG,extrapolate=False)(sv)
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

def rebuild_spline(orig_sl, pos_full, arc, coll_seg, new_wps, refine_count):
    result=list(orig_sl); n_seg=len(orig_sl)
    extra=CP_INC*(refine_count+1); ncp_c=N_CP_SEG+extra; ncp_n=N_CP_SEG+max(0,extra-CP_INC)
    for seg in range(n_seg):
        s0=seg/n_seg; s1=(seg+1)/n_seg
        mask=(arc>=s0-1e-9)&(arc<=s1+1e-9); idx=np.where(mask)[0]
        if len(idx)<2: continue
        ps=pos_full[idx[0]].copy(); pe=pos_full[idx[-1]].copy()
        if seg==coll_seg:
            sw=np.linspace(0.,1.,len(new_wps)); cp=_fit_cp(new_wps,sw,ncp_c,ps,pe)
            result[seg]=(cp,make_knots(ncp_c))
        elif seg in [coll_seg-1,coll_seg+1] and 0<=seg<n_seg:
            sl=np.clip((arc[idx]-s0)/(s1-s0),0.,1.); cp=_fit_cp(pos_full[idx],sl,ncp_n,ps,pe)
            result[seg]=(cp,make_knots(ncp_n))
    return result

# ─────────────────────────────────────────────────────────────────────────────
# PLANNING PIPELINE  (4 arms, 6 pairs)
# ─────────────────────────────────────────────────────────────────────────────

def plan(arms_data, start_qs, duration) -> Dict:
    timings={}; t_total=time.time()
    result={
        'outcome':None,'cp_rounds':0,'had_collision':False,
        'n_coll_raw':0,'n_coll_final':0,
        'arm_pos':None,'duration':duration,
        'target_joints':{},
        'start_joints':{n:start_qs.get(n,np.zeros(NDOF)).tolist() for n in ROBOT_NAMES},
        'inter_arm_clear_cm':None,'kur_min_dist_cm':None,
        'ee_path_length_m':{},'timings':timings,'plan_time_s':None,
    }

    # ── IK ────────────────────────────────────────────────────────────────
    t0=time.time()
    best=find_best_configs(arms_data, start_qs)
    timings['ik_s']=round(time.time()-t0,4)
    if best is None:
        result['outcome']='FAIL_IK'; result['plan_time_s']=round(time.time()-t_total,3); return result
    for n in ROBOT_NAMES: result['target_joints'][n]=best[n].tolist()
    # Min clearance across all 6 pairs at IK configs
    dists=[pair_min_dist(best[ROBOT_NAMES[i]],ROBOT_BASES[ROBOT_NAMES[i]],
                          best[ROBOT_NAMES[j]],ROBOT_BASES[ROBOT_NAMES[j]])
           for i in range(len(ROBOT_NAMES)) for j in range(i+1,len(ROBOT_NAMES))]
    result['inter_arm_clear_cm']=round(min(dists)*100,2)

    # ── Adaptive duration + B-spline trajectories ─────────────────────────
    t0=time.time()
    # FIX: scale duration by actual joint displacements before building splines
    duration = adaptive_duration(start_qs, best, duration)
    arm_sl={}; arm_pos={}
    for n in ROBOT_NAMES:
        cp=build_cp(start_qs.get(n,np.zeros(NDOF)),best[n])
        sl=seg_list_from_cp(cp); pos,_,_,_,duration=scale_duration(sl,duration)
        arm_sl[n]=sl; arm_pos[n]=pos
    result['duration']=duration
    arc={n:np.linspace(0.,1.,len(arm_pos[n])) for n in ROBOT_NAMES}
    for n in ROBOT_NAMES:
        base=ROBOT_BASES[n]; ee=np.array([fk_pos(arm_pos[n][k],base) for k in range(len(arm_pos[n]))])
        plen=float(np.sum(np.linalg.norm(np.diff(ee,axis=0),axis=1))) if len(ee)>1 else 0.
        result['ee_path_length_m'][n]=round(plen,4)
    timings['bspline_s']=round(time.time()-t0,4)

    # ── Collision check (all 6 pairs) ─────────────────────────────────────
    t0=time.time()
    nc_raw,_=count_collisions_all_pairs(arm_pos,ROBOT_BASES)
    result['had_collision']=nc_raw>0; result['n_coll_raw']=nc_raw
    timings['collision_check_s']=round(time.time()-t0,4)
    if not result['had_collision']:
        result['outcome']='SAFE_NO_COLL'; result['arm_pos']=arm_pos
        result['plan_time_s']=round(time.time()-t_total,3); return result

    # ── Kuramoto + alternating IK refinement ─────────────────────────────
    t0=time.time(); resolved=False
    for iteration in range(MAX_REFINE+1):
        sync_pos,kur_rep=run_kuramoto(ROBOT_NAMES,arm_pos,ROBOT_BASES,duration)
        # Global min dist across all pairs
        all_mins=[pr['min_dist_m'] for pr in kur_rep['pair_reports'].values()]
        result['kur_min_dist_cm']=round(min(all_mins)*100,2)
        nc_sync,_=count_collisions_all_pairs(sync_pos,ROBOT_BASES)
        result['n_coll_final']=nc_sync
        if nc_sync==0:
            resolved=True; result['arm_pos']={n:sync_pos[n] for n in ROBOT_NAMES}
            result['outcome']='RESOLVED_KUR' if iteration==0 else f'RESOLVED_CP_{iteration}'
            result['cp_rounds']=iteration; break
        if iteration>=MAX_REFINE: break

        # FIX: collect ALL colliding pairs this iteration, sort by urgency,
        # and address every one — not just the single most urgent.
        colliding_pairs = []
        n_wp=N_CP_SEG+CP_INC*(iteration+2)
        for i in range(len(ROBOT_NAMES)):
            for j in range(i+1,len(ROBOT_NAMES)):
                ni,nj=ROBOT_NAMES[i],ROBOT_NAMES[j]
                cs,frac=first_coll_seg_for_pair(arm_pos[ni],arm_pos[nj],
                                                 ROBOT_BASES[ni],ROBOT_BASES[nj],N_SEG)
                if cs is not None:
                    colliding_pairs.append((frac, ni, nj, cs))
        if not colliding_pairs: break
        colliding_pairs.sort()   # most urgent (earliest frac) first

        for frac, ni, nj, coll_seg in colliding_pairs:
            s0=coll_seg/N_SEG; s1=(coll_seg+1)/N_SEG
            wps_i,wps_j=alternating_ik(ROBOT_NAMES,ni,nj,arm_pos,arc,ROBOT_BASES,s0,s1,n_wp)
            if wps_i is not None:
                arm_sl[ni]=rebuild_spline(arm_sl[ni],arm_pos[ni],arc[ni],coll_seg,wps_i,iteration)
                arm_sl[nj]=rebuild_spline(arm_sl[nj],arm_pos[nj],arc[nj],coll_seg,wps_j,iteration)
                pi,_,_,_=eval_global(arm_sl[ni],duration); pj,_,_,_=eval_global(arm_sl[nj],duration)
                # Update immediately so next pair uses latest positions
                arm_pos[ni]=pi; arc[ni]=np.linspace(0.,1.,len(pi))
                arm_pos[nj]=pj; arc[nj]=np.linspace(0.,1.,len(pj))

    if not resolved:
        result['outcome']='UNRESOLVED'; result['arm_pos']=arm_pos; result['cp_rounds']=MAX_REFINE
    timings['kuramoto_refinement_s']=round(time.time()-t0,4)
    result['plan_time_s']=round(time.time()-t_total,3)
    return result

# ─────────────────────────────────────────────────────────────────────────────
# ROS2 NODE
# ─────────────────────────────────────────────────────────────────────────────

class BenchmarkNode(Node):
    def __init__(self):
        super().__init__('step_16_benchmark')
        self._pubs={n:self.create_publisher(Float64MultiArray,CTRL_TOPIC.format(arm=n),10)
                    for n in ROBOT_NAMES}
        self._lock=threading.Lock()
        self._js:  Dict[str,Optional[np.ndarray]]={n:None for n in ROBOT_NAMES}
        self._js_t:Dict[str,float]={n:0. for n in ROBOT_NAMES}
        for n in ROBOT_NAMES:
            for t in JS_TOPICS:
                self.create_subscription(JointState,t.format(arm=n),
                                         lambda msg,nm=n:self._js_cb(msg,nm),10)
        self.get_logger().info('BenchmarkNode (4-arm) ready')

    def _js_cb(self,msg,name):
        if len(msg.position)<NDOF: return
        jmap={nm:i for i,nm in enumerate(msg.name)}
        keys=[f'joint_{k}' for k in range(1,NDOF+1)]
        q=(np.array([msg.position[jmap[k]] for k in keys])
           if all(k in jmap for k in keys) else np.array(msg.position[:NDOF]))
        with self._lock: self._js[name]=q.astype(float); self._js_t[name]=time.time()

    def get_joints(self,name):
        with self._lock: return self._js[name].copy() if self._js[name] is not None else None

    def wait_for_js(self,timeout=40.):
        t0=time.time()
        while time.time()-t0<timeout:
            rclpy.spin_once(self,timeout_sec=0.05)
            with self._lock:
                if all(self._js[n] is not None for n in ROBOT_NAMES): return True
        return False

    def publish_all(self,cmds):
        msg=Float64MultiArray()
        for n in ROBOT_NAMES: msg.data=[float(v) for v in cmds[n]]; self._pubs[n].publish(msg)

    def execute_trajectory(self,arm_pos,duration):
        dt_ns=int(1e9/RATE_HZ)
        n_out=max(2,int(round(duration*RATE_HZ)))
        arm_rs={}
        for n in ROBOT_NAMES:
            pos=arm_pos[n]
            if len(pos)==n_out: arm_rs[n]=pos
            else:
                si=np.linspace(0,1,len(pos)); so=np.linspace(0,1,n_out)
                r=np.zeros((n_out,NDOF))
                for j in range(NDOF): r[:,j]=np.interp(so,si,pos[:,j])
                arm_rs[n]=r
        t_exec=time.time()
        for k in range(n_out):
            t0n=time.monotonic_ns()
            self.publish_all({n:arm_rs[n][min(k,len(arm_rs[n])-1)] for n in ROBOT_NAMES})
            rclpy.spin_once(self,timeout_sec=0.)
            rem=dt_ns-(time.monotonic_ns()-t0n)
            if rem>0: time.sleep(rem*1e-9)
        exec_time=time.time()-t_exec
        final_cmd={n:arm_rs[n][-1] for n in ROBOT_NAMES}
        t_hold=time.time()
        while time.time()-t_hold<HOLD_AFTER:
            self.publish_all(final_cmd); rclpy.spin_once(self,timeout_sec=0.); time.sleep(0.02)
        settle_time=self._wait_for_settle(final_cmd)
        return {'execution_time_s':round(exec_time,3),'settle_time_s':round(HOLD_AFTER+settle_time,3),
                'n_steps':n_out,'duration_s':round(duration,3)}

    def _wait_for_settle(self,final_cmd):
        t0=time.time(); prev_q={n:None for n in ROBOT_NAMES}; pass_count=0
        while time.time()-t0<SETTLE_TIMEOUT:
            self.publish_all(final_cmd); rclpy.spin_once(self,timeout_sec=SETTLE_DT); time.sleep(SETTLE_DT)
            with self._lock: cur_q={n:self._js[n].copy() if self._js[n] is not None else None for n in ROBOT_NAMES}
            settled=True
            for n in ROBOT_NAMES:
                if cur_q[n] is None or prev_q[n] is None: settled=False; break
                if np.max(np.abs(cur_q[n]-prev_q[n]))/SETTLE_DT>SETTLE_VEL_THR: settled=False
            prev_q=cur_q
            if settled: pass_count+=1
            else: pass_count=0
            if pass_count>=2: break
        return round(time.time()-t0,3)

    def read_fresh_js(self,timeout=VER_TIMEOUT):
        with self._lock:
            for n in ROBOT_NAMES: self._js[n]=None
        t0=time.time()
        while time.time()-t0<timeout:
            rclpy.spin_once(self,timeout_sec=0.05)
            with self._lock:
                if all(self._js[n] is not None for n in ROBOT_NAMES):
                    return {n:self._js[n].copy() for n in ROBOT_NAMES}
        with self._lock:
            return {n:self._js[n].copy() if self._js[n] is not None else None for n in ROBOT_NAMES}

# ─────────────────────────────────────────────────────────────────────────────
# VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def verify(read_js, target_joints, arms_data):
    ver={'passed':True,'per_arm':{}}
    for n in ROBOT_NAMES:
        q_r=read_js.get(n); q_t=np.array(target_joints.get(n,np.zeros(NDOF).tolist()))
        base=ROBOT_BASES[n]
        if q_r is None:
            ver['per_arm'][n]={'js_received':False,'max_joint_err_deg':None,
                               'ee_err_mm':None,'joint_ok':False,'ee_ok':False}
            ver['passed']=False; continue
        j_err=np.abs(q_r-q_t); max_je=float(np.max(j_err)); j_ok=max_je<=JOINT_VER_TOL
        ee_r=fk_pos(q_r,base); ee_t=np.array(arms_data[n].get('target_world',ee_r))
        ee_err=float(np.linalg.norm(ee_r-ee_t)*1000); ee_ok=ee_err<=EE_VER_TOL_MM
        if not j_ok or not ee_ok: ver['passed']=False
        ver['per_arm'][n]={
            'js_received':True,
            'read_joints_deg'   :[round(float(np.degrees(v)),3) for v in q_r],
            'target_joints_deg' :[round(float(np.degrees(v)),3) for v in q_t],
            'max_joint_err_deg' :round(float(np.degrees(max_je)),3),
            'ee_err_mm'         :round(ee_err,3),
            'joint_ok':j_ok,'ee_ok':ee_ok,
        }
    return ver

# ─────────────────────────────────────────────────────────────────────────────
# STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_stats(results):
    n=len(results); stats={}
    def _mean(v): return round(float(np.mean(v)),3) if v else None
    def _std(v):  return round(float(np.std(v)),3)  if len(v)>1 else None
    for oc in OUTCOMES:
        g=[r for r in results if r['outcome']==oc]; ng=len(g)
        plan_t=[r['plan_time_s'] for r in g if r.get('plan_time_s')]
        ik_t  =[r['timings'].get('ik_s',0) for r in g if r.get('timings')]
        kur_t =[r['timings'].get('kuramoto_refinement_s',0) for r in g if r.get('timings')]
        exec_t=[r.get('execution',{}).get('execution_time_s') for r in g
                if r.get('execution',{}).get('execution_time_s')]
        total_t=[r['total_time_s'] for r in g if r.get('total_time_s')]
        vo=[r.get('verification',{}).get('passed',False) for r in g
            if not r.get('verification',{}).get('skipped')]
        cl=[r['inter_arm_clear_cm'] for r in g if r.get('inter_arm_clear_cm')]
        je_vals=[]; ee_vals=[]
        for r in g:
            for nm in ROBOT_NAMES:
                pa=r.get('verification',{}).get('per_arm',{}).get(nm,{})
                if pa.get('js_received'):
                    if pa.get('max_joint_err_deg') is not None: je_vals.append(pa['max_joint_err_deg'])
                    if pa.get('ee_err_mm')          is not None: ee_vals.append(pa['ee_err_mm'])
        stats[oc]={
            'count':ng,'percent':round(100*ng/max(n,1),2),
            'plan_time_mean_s':_mean(plan_t),'plan_time_std_s':_std(plan_t),
            'ik_time_mean_s':_mean(ik_t),'kuramoto_time_mean_s':_mean(kur_t),
            'exec_time_mean_s':_mean(exec_t),'total_time_mean_s':_mean(total_t),
            'verify_pass_pct':round(100*sum(vo)/max(len(vo),1),1) if vo else None,
            'avg_clear_cm':_mean(cl),'avg_joint_err_deg':_mean(je_vals),
            'avg_ee_err_mm':_mean(ee_vals),'avg_cp_rounds':_mean([r['cp_rounds'] for r in g]),
        }
    stats['_total']=n
    return stats

# ─────────────────────────────────────────────────────────────────────────────
# PRINT SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(stats, n_trials, seed, duration):
    bar='═'*72; thin='─'*72
    print(f'\n{bar}')
    print(f'  STEP 16  —  4-ARM BENCHMARK SUMMARY')
    print(f'  Trials: {n_trials}  |  Seed: {seed}  |  Duration: {duration}s  |  Arms: 4  |  Pairs: 6')
    print(f'{bar}')
    print(f'\n  {"Outcome":<20} {"N":>5} {"Pct":>6}  {"Plan(s)":>9}  {"JointErr°":>10}  {"Verify%":>8}')
    print(f'  {thin}')
    for oc in OUTCOMES:
        s=stats.get(oc,{}); n=s.get('count',0); pct=s.get('percent',0.)
        pt=f'{s["plan_time_mean_s"]:.2f}' if s.get('plan_time_mean_s') else '    —   '
        je=f'{s["avg_joint_err_deg"]:.2f}' if s.get('avg_joint_err_deg') else '    —   '
        vr=f'{s["verify_pass_pct"]:.0f}%'  if s.get('verify_pass_pct') is not None else '   —  '
        icon='❌' if oc=='FAIL_IK' else '⚠ ' if oc=='UNRESOLVED' else '✅'
        print(f'  {icon} {oc:<19} {n:>5}  {pct:>5.1f}%  {pt:>9}  {je:>10}  {vr:>8}')
    nf=stats.get('FAIL_IK',{}).get('count',0)
    ns=stats.get('SAFE_NO_COLL',{}).get('count',0)
    nk=stats.get('RESOLVED_KUR',{}).get('count',0)
    nc=sum(stats.get(f'RESOLVED_CP_{i}',{}).get('count',0) for i in range(1,6))
    nu=stats.get('UNRESOLVED',{}).get('count',0)
    nr=ns+nk+nc
    print(f'\n  {"─"*55}  GROUPED')
    for label,count in [('IK Failed',nf),('Trivially safe',ns),
                         ('Resolved—Kuramoto only',nk),('Resolved—Kuramoto+CPs',nc),('Unresolved',nu)]:
        print(f'  {label:<40}: {count:>4}  ({100*count/max(n_trials,1):5.1f}%)')
    print(f'  {"Total resolved":<40}: {nr:>4}  ({100*nr/max(n_trials,1):5.1f}%)')
    print(f'\n{bar}\n')

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(args=None):
    parser=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--trials',  type=int,   default=100)
    parser.add_argument('--seed',    type=int,   default=0)
    parser.add_argument('--duration',type=float, default=10.0)
    ap=parser.parse_args()
    n_trials=ap.trials; rng=np.random.default_rng(ap.seed); duration=ap.duration

    print('\n'+'='*72)
    print(f'  STEP 16  —  4-Arm Gazebo Benchmark  ({n_trials} trials  seed={ap.seed}  dur={duration}s)')
    print(f'  Arms: {ROBOT_NAMES}')
    for n,b in ROBOT_BASES.items(): print(f'    {n}  base=({b[0]:.2f},{b[1]:+.2f},{b[2]:.2f})')
    print('='*72)

    rclpy.init(args=args); node=BenchmarkNode()
    print('\n  Waiting for Gazebo joint states from all 4 arms ...')
    if not node.wait_for_js(40.):
        print('  ❌  Timeout — check Gazebo and controllers')
        node.destroy_node(); rclpy.shutdown(); sys.exit(1)
    print('  ✅  All 4 arms ready\n')

    results=[]; counts={oc:0 for oc in OUTCOMES}; t_bench=time.time()
    current_start_qs={}
    for n in ROBOT_NAMES:
        q=node.get_joints(n); current_start_qs[n]=q if q is not None else np.zeros(NDOF)

    for tid in range(1, n_trials+1):
        t_trial=time.time()
        arms_data={}; targets={}
        for n in ROBOT_NAMES:
            base=ROBOT_BASES[n]; pw,quat=random_target(rng,base)
            arms_data[n]={'base':base,'target_world':pw,'tloc':pw-base,'trot':quat_to_rot(quat)}
            targets[n]={'pos':pw.tolist(),'quat':quat.tolist()}

        plan_res=plan(arms_data, current_start_qs, duration)
        rec={
            'trial_id':tid,'outcome':plan_res['outcome'],
            'had_collision':plan_res['had_collision'],
            'n_coll_raw':plan_res['n_coll_raw'],'n_coll_final':plan_res['n_coll_final'],
            'cp_rounds':plan_res['cp_rounds'],'plan_time_s':plan_res['plan_time_s'],
            'timings':plan_res['timings'],
            'inter_arm_clear_cm':plan_res['inter_arm_clear_cm'],
            'kur_min_dist_cm':plan_res['kur_min_dist_cm'],
            'ee_path_length_m':plan_res['ee_path_length_m'],
            'targets':targets,
            'start_joints':{n:current_start_qs[n].tolist() for n in ROBOT_NAMES},
            'target_joints':plan_res['target_joints'],
            'execution':{},'verification':{},'next_start_joints':{},'total_time_s':None,
        }

        if plan_res['outcome']!='FAIL_IK' and plan_res['arm_pos'] is not None:
            ex_res=node.execute_trajectory(plan_res['arm_pos'], plan_res['duration'])
            rec['execution']={**ex_res}
            read_js=node.read_fresh_js(timeout=VER_TIMEOUT)
            rec['verification']=verify(read_js, plan_res['target_joints'], arms_data)
            for n in ROBOT_NAMES:
                q=read_js.get(n)
                current_start_qs[n]=q.copy() if q is not None else \
                                     np.array(plan_res['target_joints'].get(n,np.zeros(NDOF)))
            rec['next_start_joints']={n:current_start_qs[n].tolist() for n in ROBOT_NAMES}
        else:
            rec['execution']={'skipped':True,'reason':plan_res['outcome']}
            rec['verification']={'passed':False,'skipped':True}
            rec['next_start_joints']={n:current_start_qs[n].tolist() for n in ROBOT_NAMES}

        rec['total_time_s']=round(time.time()-t_trial,3)
        results.append(rec); counts[plan_res['outcome']]+=1

        # Per-trial print
        oc=plan_res['outcome']; icon='❌' if oc=='FAIL_IK' else '⚠ ' if oc=='UNRESOLVED' else '✅'
        vok=rec['verification'].get('passed','—')
        print(f'  T{tid:>4}/{n_trials}  {icon} {oc:<20}  '
              f'plan={plan_res["plan_time_s"]:.2f}s  '
              f'coll={plan_res["n_coll_raw"]}raw→{plan_res["n_coll_final"]}fin  '
              f'ver={"✅" if vok else "❌" if vok is False else "—"}')

        if tid%10==0 or tid==n_trials:
            el=time.time()-t_bench; eta=el/tid*(n_trials-tid)
            parts=[f'{oc}:{counts[oc]}' for oc in OUTCOMES if counts[oc]>0]
            print(f'\n  ── {tid}/{n_trials}  elapsed={el:.0f}s  eta={eta:.0f}s  |  '+' '.join(parts)+'\n')

    # ── Final outputs ──────────────────────────────────────────────────────
    stats=compute_stats(results); stats['_seed']=ap.seed; stats['_duration_s']=duration
    print_summary(stats, n_trials, ap.seed, duration)

    with open('trial_results.json','w') as fh:
        json.dump({'seed':ap.seed,'n_trials':n_trials,'n_arms':4,
                   'duration_s':duration,'results':results},fh,indent=2)
    with open('trial_summary.json','w') as fh:
        json.dump({'seed':ap.seed,'n_trials':n_trials,'n_arms':4,
                   'duration_s':duration,'stats':stats},fh,indent=2)

    kb1=os.path.getsize('trial_results.json')/1024.
    kb2=os.path.getsize('trial_summary.json')/1024.
    print(f'  ✅  trial_results.json  ({kb1:.1f} KB)')
    print(f'  ✅  trial_summary.json  ({kb2:.1f} KB)\n')

    node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()