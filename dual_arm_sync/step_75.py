#!/usr/bin/env python3
"""step_75.py -- Collision Map Visualizer: min dist over time"""
import json, os, sys
import numpy as np
try: import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; _PLOT=True
except ImportError: _PLOT=False

sys.path.insert(0, os.path.dirname(__file__))
from _robot import ROBOT_BASES, ARM_NAMES, pair_min_dist, SAFETY_MARGIN, LINK_RADII

def get_dist_trace(pos1, pos2, b1, b2):
    K=min(len(pos1),len(pos2))
    return [pair_min_dist(pos1[k],b1,pos2[k],b2) for k in range(K)]

def main():
    print('\nSTEP 75 -- Collision Map Visualizer')
    if not _PLOT: print('  matplotlib not available'); sys.exit(0)
    sources=[('s62_trajectories.json','Before (original)','#F44336'),
             ('s65_synchronized.json','After (SKAR-N)','#4CAF50')]
    loaded={}
    for fname,label,_ in sources:
        if os.path.exists(fname):
            with open(fname) as f: loaded[label]=(json.load(f),_)
    if not loaded: sys.exit(1)
    arm_names=list(loaded.values())[0][0].get('arm_names',ARM_NAMES)
    bases={n:np.array(ROBOT_BASES.get(n,[0,0,0])) for n in arm_names}
    if len(arm_names)<2: print('  Need at least 2 arms'); sys.exit(0)
    ni,nj=arm_names[0],arm_names[1]
    fig,ax=plt.subplots(figsize=(12,5))
    thresh=(LINK_RADII.min()*2+SAFETY_MARGIN)*100
    ax.axhline(y=thresh,color='red',lw=2,ls='--',label=f'Safety threshold ({thresh:.0f}cm)')
    ax.axhline(y=(thresh+8),color='orange',lw=1,ls=':',label='Warning zone')
    for label,(data,color) in loaded.items():
        pos={n:np.array(data[n]['trajectory']['positions']) for n in arm_names}
        dur=float(data.get('duration',10.))
        K=min(len(pos[ni]),len(pos[nj])); t=np.linspace(0,dur,K)
        dists=[pair_min_dist(pos[ni][k],bases[ni],pos[nj][k],bases[nj])*100 for k in range(K)]
        ax.plot(t,dists,color=color,lw=2,label=label)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Min inter-arm distance (cm)')
    ax.set_title(f'Inter-arm Distance: {ni} ↔ {nj}'); ax.legend(); ax.grid(True,alpha=0.3)
    plt.tight_layout(); plt.savefig('s75_collision_map.png',dpi=150,bbox_inches='tight')
    plt.close(); print('  Saved: s75_collision_map.png')

if __name__=='__main__': main()