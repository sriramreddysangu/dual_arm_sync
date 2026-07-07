#!/usr/bin/env python3
"""step_24.py -- Kuramoto-ALONE temporal resolver (multi-collision) [6 ARM]
INPUT s22_trajectories.json, s23_collision_map.json  OUTPUT s24_resolved.json
Coordination-diagram + priority lag on REAL MESH: build the obstacle grid ONCE per
colliding pair, then each arm lags just enough to clear higher-priority arms. Resolves
several simultaneous collisions; whatever timing can't clear -> needs_retraction for
step_25. Fast (grid once per pair, not per timestep). ASCII only."""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from _robot6x import (NDOF, POS_LIM, VEL_LIM, ACC_LIM, RATE_HZ, ROBOT_BASES,
                       ARM_NAMES, pair_min_dist, pair_collides)
CLEAR_M = float(os.environ.get('DUAL_ARM_CLEARANCE_M', '0.05'))
SAFETY  = float(os.environ.get('DUAL_ARM_SPEED_SAFETY', '1.25'))
MIN_DUR = float(os.environ.get('DUAL_ARM_MIN_DURATION', '0.0'))
GRID    = int(os.environ.get('DUAL_ARM_COORD_GRID', '48'))

def interp(path, frac):
    frac=float(np.clip(frac,0,1)); n=len(path)-1
    if n<=0: return path[0].copy()
    idx=min(int(frac*n),n-1); a=frac*n-idx
    return path[idx]+a*(path[min(idx+1,n)]-path[idx])

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

def scan_mesh(P,names,bases,pairs=None):
    if pairs is None:
        pairs=[(names[i],names[j]) for i in range(len(names)) for j in range(i+1,len(names))]
    cs=0; mind=np.inf
    for ni,nj in pairs:
        K=min(len(P[ni]),len(P[nj]))
        for k in range(0,K,2):                       # subsample x2 for speed
            d=pair_min_dist(P[ni][k],bases[ni],P[nj][k],bases[nj])
            if d<mind: mind=d
            if d<CLEAR_M: cs+=1
    return cs,(mind if mind!=np.inf else 0.0)

def apply_lags(paths,names,delay,duration):
    span=1.0+max(delay.values()); n=max(2,int(round(span*duration*RATE_HZ))); tau=np.linspace(0,span,n)
    out={}
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
        r=max(r,float(np.max(np.abs(vel[:,j])))/VEL_LIM[j])
        r=max(r,float(np.sqrt(max(float(np.max(np.abs(acc[:,j]))),0.0)/ACC_LIM[j])))
    return r

def time_optimal(names,pos_per_arm,duration,safety):
    kdur=duration
    for _ in range(16):
        ratio=max(speed_ratio(resample(pos_per_arm[n],kdur),kdur) for n in names)
        if ratio<=1e-9: break
        new=max(0.3,kdur*ratio*safety); done=abs(new-kdur)<=0.005*kdur; kdur=new
        if done: break
    return kdur,{n:resample(pos_per_arm[n],kdur) for n in names}

def _write(arm_names,chosen,kdur,base_dur,tin,needs_retraction,residual_pairs,label):
    out={'duration':float(kdur),'arm_names':arm_names,'final_duration_s':round(float(kdur),4),
         'duration_overhead':round(float(kdur)/base_dur-1,4),'needs_retraction':bool(needs_retraction),
         'residual_pairs':residual_pairs,
         'synchronisation_report':{'collision_free':not needs_retraction,'method':'coordination_lag'}}
    for n in arm_names:
        pos=np.asarray(chosen[n]); dt=kdur/max(len(pos)-1,1)
        vel=np.gradient(pos,dt,axis=0); acc=np.gradient(vel,dt,axis=0)
        out[n]={'robot_name':n,'metadata':{**tin[n]['metadata'],'duration':float(kdur)},
                'spline':tin[n].get('spline',{}),
                'trajectory':{'time':np.linspace(0.,kdur,len(pos)).tolist(),'positions':pos.tolist(),
                    'velocities':vel.tolist(),'accelerations':acc.tolist(),
                    'arc_fracs':np.linspace(0.,1.,len(pos)).tolist()}}
    with open('s24_resolved.json','w') as fh: json.dump(out,fh,indent=2)
    print(f'  Saved: s24_resolved.json ({label})')

def main():
    print('\n'+'='*66); print('  STEP 24  --  Kuramoto-ALONE temporal resolver (multi-collision) [6 ARM]'); print('='*66)
    if not os.path.exists('s22_trajectories.json'): print('  s22_trajectories.json not found'); sys.exit(1)
    with open('s22_trajectories.json') as fh: tj=json.load(fh)
    arm_names=tj.get('arm_names',ARM_NAMES); duration=float(tj['duration'])
    bases={n:np.array(ROBOT_BASES.get(n,[0,0,0])) for n in arm_names}
    paths={n:np.array(tj[n]['trajectory']['positions'],float) for n in arm_names}
    cmap={}
    if os.path.exists('s23_collision_map.json'):
        with open('s23_collision_map.json') as fh: cmap=json.load(fh)
    coll_pairs=[(p['arm_i'],p['arm_j']) for p in cmap.get('pairs',[]) if p.get('status')=='COLLISION']
    print(f'\n  Arms: {arm_names}  dur={duration:.2f}s  {len(coll_pairs)} colliding pairs')
    if not coll_pairs:
        chosen={n:resample(paths[n],duration) for n in arm_names}
        kdur,chosen=time_optimal(arm_names,chosen,duration,SAFETY); kdur=max(kdur,MIN_DUR)
        _write(arm_names,chosen,kdur,duration,tj,False,[],'no collision'); 
        print(f'  No collision. Duration {duration:.1f}s -> {kdur:.2f}s'); print('  Next : step_25\n'); return
    t0=time.time(); G=GRID; ob={}
    for (ni,nj) in coll_pairs: ob[(ni,nj)]=build_obstacle(paths[ni],paths[nj],bases[ni],bases[nj],G)
    print(f'  built {len(ob)} coordination grids ({G}x{G}) in {(time.time()-t0)*1000:.0f}ms')
    order=list(arm_names); delay={n:0.0 for n in order}
    for idx,k in enumerate(order):
        need=0.0
        for j in order[:idx]:
            key=(j,k) if (j,k) in ob else ((k,j) if (k,j) in ob else None)
            if key is None: continue
            o=ob[key]; base_dc=int(round((delay.get(k,0.0)-delay[j])*G))
            for extra in range(0,G):
                dc=base_dc-extra if key==(k,j) else base_dc+extra
                if clear_with_lag(o,dc,G): need=max(need,extra/G); break
        delay[k]=delay.get(k,0.0)+need
    sched,sdur=apply_lags(paths,arm_names,delay,duration)
    chosen={n:resample(sched[n],sdur) for n in arm_names}
    cs,mind=scan_mesh(chosen,arm_names,bases,coll_pairs)
    residual=[]
    for ni,nj in coll_pairs:
        K=min(len(chosen[ni]),len(chosen[nj]))
        if any(pair_min_dist(chosen[ni][k],bases[ni],chosen[nj][k],bases[nj])<CLEAR_M for k in range(0,K,2)):
            residual.append([ni,nj])
    lagstr=" ".join(f"{n}+{delay[n]:.2f}" for n in arm_names if delay[n]>0) or "none"
    print(f'  lag schedule: {lagstr}')
    print(f'  after timing (mesh): {cs} coll steps, min={mind*100:.1f}cm, residual={residual}')
    if cs==0:
        kdur,chosen=time_optimal(arm_names,chosen,sdur,SAFETY); kdur=max(kdur,MIN_DUR)
        _write(arm_names,chosen,kdur,duration,tj,False,[],'Kuramoto SUCCESS (timing alone)')
        print(f'  ALL {len(coll_pairs)} pair(s) cleared by TIMING. Duration {duration:.1f}s -> {kdur:.2f}s')
        print('  Next : step_25\n'); return
    _write(arm_names,chosen,sdur,duration,tj,True,residual,f'timing partial ({len(residual)} need retraction)')
    print(f'  Timing left {len(residual)} pair(s) -> step_25 retraction.'); print('  Next : step_25\n')

if __name__=='__main__': main()