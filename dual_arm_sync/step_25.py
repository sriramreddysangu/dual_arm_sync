#!/usr/bin/env python3
"""step_25.py -- Retraction + Kuramoto (fallback for step_24)   [6 ARM]
INPUT s24_resolved.json, s22_trajectories.json   OUTPUT s25_synchronized.json
Same architecture as step_44 (Lagrangian), scoped to N arms and MULTIPLE residual
pairs. For each pair step_24 could not skip by timing, retract BOTH arms minimally
(gentle PULL_GAIN, stop the instant a phase lag can finish -> stays near optimal),
then re-run the coordination lag over ALL arms to finish. Resolves several persisting
collisions, pair by pair. Real-mesh throughout (_robot6x). ASCII only."""
import json, os, sys, time
import numpy as np
from scipy.interpolate import BSpline
sys.path.insert(0, os.path.dirname(__file__))
from _robot6x import (NDOF, POS_LIM, VEL_LIM, ACC_LIM, RATE_HZ, ROBOT_BASES,
                       ARM_NAMES, pair_min_dist, pair_collides)
CLEAR_M = float(os.environ.get('DUAL_ARM_CLEARANCE_M', '0.05'))
SAFETY  = float(os.environ.get('DUAL_ARM_SPEED_SAFETY', '1.25'))
MIN_DUR = float(os.environ.get('DUAL_ARM_MIN_DURATION', '0.0'))
GRID    = int(os.environ.get('DUAL_ARM_COORD_GRID', '48'))
MAX_LAG = float(os.environ.get('DUAL_ARM_MAX_LAG', '0.6'))   # cap per-arm phase lag
# retraction
DEG=3; N_SEG=5; N_CP_SEG=4; MAX_ITER=60; SPAN_PAD=0.10; RESOLVE_NS=220; RESOLVE_BUFFER=0.03
PULL_GAIN=float(os.environ.get('DUAL_ARM_PULL_GAIN','0.15'))

def interp(path,frac):
    frac=float(np.clip(frac,0,1)); n=len(path)-1
    if n<=0: return path[0].copy()
    idx=min(int(frac*n),n-1); a=frac*n-idx; return path[idx]+a*(path[min(idx+1,n)]-path[idx])
# ---- coordination (from step_24) ----
def build_obstacle(pi,pj,bi,bj,G):
    sa=np.linspace(0,1,G); qi=[interp(pi,s) for s in sa]; qj=[interp(pj,s) for s in sa]
    ob=np.zeros((G,G),bool)
    for a in range(G):
        for b in range(G): ob[a,b]=pair_collides(qi[a],bi,qj[b],bj,margin=CLEAR_M+0.03)
    return ob
def clear_with_lag(ob,dc,G):
    for k in range(G+abs(dc)):
        a=min(max(k,0),G-1); b=min(max(k-dc,0),G-1)
        if ob[a,b]: return False
    return True
def scan_mesh(P,names,bases,pairs):
    cs=0; mind=np.inf
    for ni,nj in pairs:
        K=min(len(P[ni]),len(P[nj]))
        for k in range(0,K,2):
            d=pair_min_dist(P[ni][k],bases[ni],P[nj][k],bases[nj])
            if d<mind: mind=d
            if d<CLEAR_M: cs+=1
    return cs,(mind if mind!=np.inf else 0.0)
def apply_lags(paths,names,delay,duration):
    span=1.0+max(delay.values()); n=max(2,int(round(span*duration*RATE_HZ))); tau=np.linspace(0,span,n); out={}
    for nm in names:
        ph=np.clip(tau-delay[nm],0,1); out[nm]=np.array([interp(paths[nm],f) for f in ph])
    return out, span*duration
def resample(pos,duration):
    n_out=max(2,int(round(duration*RATE_HZ))); t_out=np.linspace(0,duration,n_out)
    t_in=np.linspace(0,duration,len(pos)); out=np.zeros((n_out,NDOF))
    for j in range(NDOF): out[:,j]=np.interp(t_out,t_in,pos[:,j])
    return np.clip(out,POS_LIM[:,0],POS_LIM[:,1])
def speed_ratio(pos,duration):
    dt=duration/max(len(pos)-1,1); vel=np.gradient(pos,dt,axis=0); acc=np.gradient(vel,dt,axis=0); r=0.0
    for j in range(NDOF):
        r=max(r,float(np.max(np.abs(vel[:,j])))/VEL_LIM[j]); r=max(r,float(np.sqrt(max(float(np.max(np.abs(acc[:,j]))),0.0)/ACC_LIM[j])))
    return r
def time_optimal(names,pp,duration,safety):
    kdur=duration
    for _ in range(16):
        ratio=max(speed_ratio(resample(pp[n],kdur),kdur) for n in names)
        if ratio<=1e-9: break
        new=max(0.3,kdur*ratio*safety); done=abs(new-kdur)<=0.005*kdur; kdur=new
        if done: break
    return kdur,{n:resample(pp[n],kdur) for n in names}
def schedule_lags(paths,coll_pairs,bases,duration,names):
    G=GRID; ob={}
    for (ni,nj) in coll_pairs: ob[(ni,nj)]=build_obstacle(paths[ni],paths[nj],bases[ni],bases[nj],G)
    order=list(names); delay={n:0.0 for n in order}
    for idx,k in enumerate(order):
        need=0.0
        for j in order[:idx]:
            key=(j,k) if (j,k) in ob else ((k,j) if (k,j) in ob else None)
            if key is None: continue
            o=ob[key]; base_dc=int(round((delay.get(k,0.0)-delay[j])*G))
            cap=int(MAX_LAG*G); got=False
            for extra in range(0,cap+1):
                dc=base_dc-extra if key==(k,j) else base_dc+extra
                if clear_with_lag(o,dc,G): need=max(need,extra/G); got=True; break
            # if no lag within cap clears it, don't balloon -- leave it for retraction
        delay[k]=min(delay.get(k,0.0)+need, MAX_LAG)
    sched,sdur=apply_lags(paths,names,delay,duration)
    chosen={n:resample(sched[n],sdur) for n in names}
    residual=[]
    for ni,nj in coll_pairs:
        K=min(len(chosen[ni]),len(chosen[nj]))
        if any(pair_min_dist(chosen[ni][k],bases[ni],chosen[nj][k],bases[nj])<CLEAR_M for k in range(0,K,2)):
            residual.append([ni,nj])
    return chosen,residual,sdur,delay
# ---- retraction (from step_44) ----
def make_knots(ncp):
    ni=max(0,ncp-DEG-1); inn=np.linspace(0,1,ni+2)[1:-1] if ni>0 else np.array([])
    return np.concatenate([np.zeros(DEG+1),inn,np.ones(DEG+1)])
def greville(kn): return np.array([np.mean(kn[i+1:i+DEG+1]) for i in range(len(kn)-DEG-1)])
def global_cp_from_segs(td):
    n_seg=int(td['spline']['n_seg']); n_cp=int(td['spline']['n_cp_seg']); total=n_seg*(n_cp-1)+1
    cps=np.array(td['trajectory']['positions']); idx=np.linspace(0,len(cps)-1,total).astype(int); return cps[idx].copy()
def eval_cp(cp,duration):
    kn=make_knots(len(cp)); ns=max(2,int(round(duration*RATE_HZ))); spl=BSpline(kn,cp,DEG,axis=0,extrapolate=True)
    return np.clip(spl(np.linspace(0,1,ns)),POS_LIM[:,0],POS_LIM[:,1])
def eval_cp_n(cp,n):
    kn=make_knots(len(cp)); spl=BSpline(kn,cp,DEG,axis=0,extrapolate=True)
    return np.clip(spl(np.linspace(0,1,n)),POS_LIM[:,0],POS_LIM[:,1])
def scan_pair(pi,pj,bi,bj,margin=CLEAR_M):
    cs=0; ks=[]
    for k in range(min(len(pi),len(pj))):
        if pair_collides(pi[k],bi,pj[k],bj,margin=margin): cs+=1; ks.append(k)
    return cs,ks
def lag_resolvable(pi,pj,bi,bj,G=30,margin=None):
    m=CLEAR_M+RESOLVE_BUFFER if margin is None else margin
    idx=np.linspace(0,len(pi)-1,G).astype(int); qi=pi[idx]; qj=pj[idx]; ob=np.zeros((G,G),bool)
    for a in range(G):
        for b in range(G): ob[a,b]=pair_collides(qi[a],bi,qj[b],bj,margin=m)
    if ob[0,0] or ob[G-1,G-1]: return False
    for dc in range(G):
        for lead in (0,1):
            ok=True
            for k in range(G+dc):
                iL=min(k,G-1); iLag=min(max(k-dc,0),G-1); a,b=(iLag,iL) if lead else (iL,iLag)
                if ob[a,b]: ok=False; break
            if ok: return True
    return False
def resolve_pair(cp_i,cp_j,bi,bj,duration):
    cp_i=cp_i.copy(); cp_j=cp_j.copy(); kn=make_knots(len(cp_i)); grev=greville(kn); iters=0; lag_stop=False
    for it in range(MAX_ITER):
        iters=it+1; pi=eval_cp_n(cp_i,RESOLVE_NS); pj=eval_cp_n(cp_j,RESOLVE_NS)
        cs,ks=scan_pair(pi,pj,bi,bj,CLEAR_M+RESOLVE_BUFFER)
        if cs==0: iters=it; break
        if lag_resolvable(pi,pj,bi,bj): iters=it; lag_stop=True; break
        K=len(pi); arc=np.linspace(0,1,K); lo=max(arc[ks[0]]-SPAN_PAD,0); hi=min(arc[ks[-1]]+SPAN_PAD,1)
        for cp in (cp_i,cp_j):
            for c in range(1,len(cp)-1):
                g=grev[c]
                if lo<=g<=hi:
                    w=0.5-0.5*np.cos(2*np.pi*(g-lo)/max(hi-lo,1e-9)); rt=cp[c].copy(); rt[1:]=0
                    cp[c]=np.clip(cp[c]+PULL_GAIN*w*(rt-cp[c]),POS_LIM[:,0],POS_LIM[:,1])
    return cp_i,cp_j,{'iterations':iters,'lag_will_finish':lag_stop}
def _write(names,chosen,kdur,base_dur,meta,label,resolved):
    out={'duration':float(kdur),'arm_names':names,'final_duration_s':round(float(kdur),4),
         'duration_overhead':round(float(kdur)/base_dur-1,4),
         'synchronisation_report':{'collision_free':resolved,'method':'retraction_plus_kuramoto'},
         'final_verification':{'verified_collision_free':resolved}}
    for n in names:
        pos=np.asarray(chosen[n]); dt=kdur/max(len(pos)-1,1); vel=np.gradient(pos,dt,axis=0); acc=np.gradient(vel,dt,axis=0)
        out[n]={'robot_name':n,'metadata':{**meta[n],'duration':float(kdur)},
                'trajectory':{'time':np.linspace(0.,kdur,len(pos)).tolist(),'positions':pos.tolist(),
                    'velocities':vel.tolist(),'accelerations':acc.tolist(),'arc_fracs':np.linspace(0.,1.,len(pos)).tolist()}}
    with open('s25_synchronized.json','w') as fh: json.dump(out,fh,indent=2)
    print(f'  Saved: s25_synchronized.json ({label})')
def retract_once(cp_i,cp_j,bi,bj):
    kn=make_knots(len(cp_i)); grev=greville(kn)
    pi=eval_cp_n(cp_i,RESOLVE_NS); pj=eval_cp_n(cp_j,RESOLVE_NS)
    cs,ks=scan_pair(pi,pj,bi,bj,CLEAR_M+RESOLVE_BUFFER)
    if cs==0: return cp_i,cp_j,True
    cp_i=cp_i.copy(); cp_j=cp_j.copy()
    K=len(pi); arc=np.linspace(0,1,K); lo=max(arc[ks[0]]-SPAN_PAD,0); hi=min(arc[ks[-1]]+SPAN_PAD,1)
    for cp in (cp_i,cp_j):
        for c in range(1,len(cp)-1):
            g=grev[c]
            if lo<=g<=hi:
                w=0.5-0.5*np.cos(2*np.pi*(g-lo)/max(hi-lo,1e-9)); rt=cp[c].copy(); rt[1:]=0
                cp[c]=np.clip(cp[c]+PULL_GAIN*w*(rt-cp[c]),POS_LIM[:,0],POS_LIM[:,1])
    return cp_i,cp_j,False

def main():
    print('\n'+'='*66); print('  STEP 25  --  Retraction + Kuramoto (fallback for step_24) [6 ARM]'); print('='*66)
    if not os.path.exists('s24_resolved.json'): print('  s24_resolved.json not found'); sys.exit(1)
    s24=json.load(open('s24_resolved.json')); names=s24['arm_names']
    bases={n:np.array(ROBOT_BASES.get(n,[0,0,0])) for n in names}
    if not s24.get('needs_retraction',False):
        out=dict(s24); out['final_verification']={'verified_collision_free':True,'reason':'kuramoto_alone_in_step_24'}
        json.dump(out,open('s25_synchronized.json','w'),indent=2)
        print('  step_24 resolved all by timing -> passthrough.'); print('  Saved: s25_synchronized.json'); print('  Next : step_26\n'); return
    residual0=[tuple(p) for p in s24.get('residual_pairs',[])]
    print(f'  {len(residual0)} residual pair(s) need retraction: {residual0}')
    s22=json.load(open('s22_trajectories.json')); duration=float(s22['duration'])
    cp={n:global_cp_from_segs(s22[n]) for n in names}
    MAX_ROUNDS=10; t0=time.time(); chosen=None; residual=residual0; sdur=duration
    residual_set={tuple(sorted(p)) for p in residual0}; prev=None; stall=0; conflict=[]
    for rnd in range(MAX_ROUNDS):
        paths={n:eval_cp(cp[n],duration) for n in names}
        coll=[]
        for i in range(len(names)):
            for j in range(i+1,len(names)):
                ni,nj=names[i],names[j]; K=min(len(paths[ni]),len(paths[nj]))
                if any(pair_min_dist(paths[ni][k],bases[ni],paths[nj][k],bases[nj])<CLEAR_M for k in range(0,K,3)):
                    coll.append((ni,nj))
        if not coll:
            chosen={n:resample(paths[n],duration) for n in names}; residual=[]; sdur=duration; break
        chosen,residual,sdur,delay=schedule_lags(paths,coll,bases,duration,names)
        if not residual: break
        # a NEW pair (retraction pushed an arm into a third arm) is a 3-way conflict:
        new=[p for p in residual if tuple(sorted(p)) not in residual_set]
        if new: conflict=new
        nc=len(residual)
        if prev is not None and nc>=prev: stall+=1
        else: stall=0
        prev=nc
        if stall>=2: break                          # not converging -> stop, report
        for (ni,nj) in residual:                     # retract only what still collides, one step
            cp[ni],cp[nj],_=retract_once(cp[ni],cp[nj],bases[ni],bases[nj])
    print(f'  interleaved retraction+coordination: {rnd+1} rounds, {(time.time()-t0)*1000:.0f}ms, residual={residual}')
    if conflict:
        print(f'  NOTE: retraction of {residual0} pushed an arm into a THIRD arm -> new conflict {conflict}')
        print(f'        (3-way workspace conflict -- retraction alone cannot separate these).')
    if not residual:
        kdur,chosen=time_optimal(names,chosen,sdur,SAFETY); kdur=max(kdur,MIN_DUR)
    else:
        kdur=sdur                                    # unresolved: keep schedule duration, no inflation
    meta={n:s22[n]['metadata'] for n in names}
    if not residual:
        _write(names,chosen,kdur,duration,meta,'retraction + Kuramoto SUCCESS',True)
        print(f'  ALL pairs resolved. Duration {duration:.1f}s -> {kdur:.2f}s'); print('  Next : step_26\n')
    else:
        _write(names,chosen,kdur,duration,meta,f'UNRESOLVED ({residual})',False)
        print(f'  Still colliding: {residual} (deep swap / target overlap) -> re-run step_21 IK.\n')
if __name__=='__main__': main()