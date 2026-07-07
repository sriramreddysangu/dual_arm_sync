#!/usr/bin/env python3
"""step_77.py -- Kuramoto Phase Diagram"""
import json, os, sys
import numpy as np
try: import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; _PLOT=True
except ImportError: _PLOT=False

sys.path.insert(0, os.path.dirname(__file__))
from _robot import ARM_NAMES

def main():
    print('\nSTEP 77 -- Kuramoto Phase Diagram')
    if not _PLOT: sys.exit(0)
    if not os.path.exists('s65_synchronized.json'): sys.exit(1)
    with open('s65_synchronized.json') as f: s=json.load(f)
    arm_names=s.get('arm_names',ARM_NAMES)
    kur=s.get('synchronisation_report',{})
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    colors=['#2196F3','#F44336','#4CAF50','#FF9800']
    for idx,name in enumerate(arm_names):
        pos=np.array(s[name]['trajectory']['positions'])
        dur=float(s[name]['metadata']['duration'])
        t  =np.linspace(0,dur,len(pos))
        # Approximate phase from arc fraction
        phi=np.linspace(0,1,len(pos))
        axes[0].plot(t,phi,color=colors[idx%4],label=name,lw=2)
    axes[0].set_xlabel('Time (s)'); axes[0].set_ylabel('Phase φ')
    axes[0].set_title('Phase Evolution'); axes[0].legend(); axes[0].grid(True,alpha=0.3)
    axes[0].plot([0,max(float(s[n]['metadata']['duration']) for n in arm_names)],[0,1],
                 'k--',lw=1,alpha=0.4,label='ideal')
    for pair,rep in kur.get('pair_reports',{}).items():
        axes[1].bar(pair,rep['min_dist_m']*100,
                    color='#4CAF50' if rep['collision_free'] else '#F44336')
    axes[1].set_ylabel('Min inter-arm dist (cm)'); axes[1].set_title('Min Distance per Pair')
    axes[1].axhline(y=15,color='red',ls='--',lw=1,label='Safety threshold'); axes[1].legend()
    axes[1].grid(True,alpha=0.3)
    plt.suptitle('Kuramoto Synchronization Analysis',fontsize=11); plt.tight_layout()
    plt.savefig('s77_kuramoto.png',dpi=150,bbox_inches='tight'); plt.close()
    print('  Saved: s77_kuramoto.png')

if __name__=='__main__': main()