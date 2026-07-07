#!/usr/bin/env python3
"""step_74.py -- Trajectory Visualizer: joint profiles, EE paths, vel/acc"""
import json, os, sys
import numpy as np
try: import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; _PLOT=True
except ImportError: _PLOT=False

sys.path.insert(0, os.path.dirname(__file__))
from _robot import NDOF, ROBOT_BASES, ARM_NAMES, fk_world

def main():
    print('\nSTEP 74 -- Trajectory Visualizer')
    if not _PLOT: print('  matplotlib not available -- skipping'); sys.exit(0)
    sources=[('s62_trajectories.json','original'),('s64_resolved.json','resolved'),
             ('s65_synchronized.json','synchronized')]
    loaded={}
    for fname,label in sources:
        if os.path.exists(fname):
            with open(fname) as f: loaded[label]=json.load(f)
    if not loaded: print('  No trajectory files found'); sys.exit(1)

    arm_names=list(loaded.values())[0].get('arm_names',ARM_NAMES)
    joint_names=[f'J{i+1}' for i in range(NDOF)]
    colors={'original':'#2196F3','resolved':'#FF9800','synchronized':'#4CAF50'}

    for name in arm_names:
        fig, axes = plt.subplots(3,1,figsize=(12,10))
        for label,data in loaded.items():
            if name not in data: continue
            pos=np.array(data[name]['trajectory']['positions'])
            t  =np.array(data[name]['trajectory']['time']) if 'time' in data[name]['trajectory'] \
                else np.linspace(0,data[name]['metadata']['duration'],len(pos))
            vel=np.gradient(pos,t[1]-t[0] if len(t)>1 else 1.,axis=0)
            base=np.array(ROBOT_BASES.get(name,[0,0,0]))
            ee=[fk_world(pos[k],base) for k in range(len(pos))]
            for j in range(NDOF):
                axes[0].plot(t,np.degrees(pos[:,j]),alpha=0.7,lw=1.5 if label=='synchronized' else 1,
                             color=colors.get(label,'gray'),label=f'{label} J{j+1}' if j==0 else '')
                axes[1].plot(t,np.degrees(vel[:,j]),alpha=0.7,lw=1.,color=colors.get(label,'gray'))
            axes[2].plot([p[0] for p in ee],[p[2] for p in ee],
                         color=colors.get(label,'gray'),label=label,lw=2 if label=='synchronized' else 1)
        axes[0].set_ylabel('Joint Angle (deg)'); axes[0].set_title(f'{name} -- Joint Profiles')
        axes[0].legend(fontsize=8); axes[0].grid(True,alpha=0.3)
        axes[1].set_ylabel('Joint Vel (deg/s)'); axes[1].set_title('Velocities'); axes[1].grid(True,alpha=0.3)
        axes[2].set_xlabel('X (m)'); axes[2].set_ylabel('Z (m)')
        axes[2].set_title('EE Path (XZ plane)'); axes[2].legend(); axes[2].grid(True,alpha=0.3)
        axes[2].set_aspect('equal')
        plt.tight_layout()
        fname_out=f's74_{name}_traj.png'; plt.savefig(fname_out,dpi=150,bbox_inches='tight')
        plt.close(); print(f'  Saved: {fname_out}')

if __name__=='__main__': main()