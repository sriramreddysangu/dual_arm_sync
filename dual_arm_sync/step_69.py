#!/usr/bin/env python3
"""step_69.py -- Baseline Comparison: Sequential vs SKAR-N"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _robot import NDOF, POS_LIM, ROBOT_BASES, ARM_NAMES, pair_collides

def simultaneous_collisions(arm_names, arm_pos, bases):
    N=len(arm_names); nc=0
    K=min(len(arm_pos[n]) for n in arm_names)
    for i in range(N):
        for j in range(i+1,N):
            for k in range(K):
                if pair_collides(arm_pos[arm_names[i]][k],bases[arm_names[i]],
                                 arm_pos[arm_names[j]][k],bases[arm_names[j]]): nc+=1
    return nc

def main():
    print('\nSTEP 69 -- Baseline Comparison')
    for f in ['s62_trajectories.json','s64_resolved.json','s65_synchronized.json']:
        if not os.path.exists(f): print(f'  {f} not found'); sys.exit(1)
    with open('s62_trajectories.json') as f: orig = json.load(f)
    with open('s65_synchronized.json') as f: sync = json.load(f)
    arm_names=orig.get('arm_names',ARM_NAMES)
    bases={n:np.array(ROBOT_BASES.get(n,[0,0,0])) for n in arm_names}
    pos_orig={n:np.array(orig[n]['trajectory']['positions'],dtype=float) for n in arm_names}
    pos_sync={n:np.array(sync[n]['trajectory']['positions'],dtype=float) for n in arm_names}

    # Baseline A: naive simultaneous (no avoidance)
    t0=time.time(); nc_naive=simultaneous_collisions(arm_names,pos_orig,bases); t_naive=round((time.time()-t0)*1000,1)
    # SKAR-N result
    t0=time.time(); nc_skar=simultaneous_collisions(arm_names,pos_sync,bases); t_skar=round((time.time()-t0)*1000,1)
    # Sequential: arm1 fixed, arm2 unconstrained (just use orig both -- simulate by time-shifting arm2)
    pos_seq={arm_names[0]:pos_orig[arm_names[0]]}
    dur=float(orig['duration']); n=len(pos_orig[arm_names[0]])
    shift=int(n*0.15)  # 15% phase shift
    pos_seq[arm_names[1]]=np.roll(pos_orig[arm_names[1]], shift, axis=0) if len(arm_names)>1 else None
    if len(arm_names)>1 and pos_seq[arm_names[1]] is not None:
        nc_seq=simultaneous_collisions(arm_names,pos_seq,bases)
    else: nc_seq=nc_naive

    out={'naive_coll_steps':nc_naive,'sequential_coll_steps':nc_seq,'skar_n_coll_steps':nc_skar,
         'skar_n_resolved':nc_skar==0,'collision_reduction_pct':round((1-nc_skar/max(nc_naive,1))*100,1)}
    with open('s69_baseline.json','w') as f: json.dump(out,f,indent=2)
    print(f'  Naive: {nc_naive} coll_steps')
    print(f'  Sequential (15% shift): {nc_seq} coll_steps')
    print(f'  SKAR-N: {nc_skar} coll_steps  ({"CLEAN" if nc_skar==0 else "UNRESOLVED"})')
    print(f'  Reduction: {out["collision_reduction_pct"]:.0f}%')
    print('  Saved: s69_baseline.json')

if __name__=='__main__': main()