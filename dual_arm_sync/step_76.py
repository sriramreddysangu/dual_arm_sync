#!/usr/bin/env python3
"""step_76.py -- CP Modification Visualizer: before vs after per segment"""
import json, os, sys
import numpy as np
try: import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; _PLOT=True
except ImportError: _PLOT=False

sys.path.insert(0, os.path.dirname(__file__))
from _robot import NDOF, ARM_NAMES

def main():
    print('\nSTEP 76 -- CP Modification Visualizer')
    if not _PLOT: sys.exit(0)
    if not os.path.exists('s64_resolved.json'): sys.exit(1)
    with open('s64_resolved.json') as f: res=json.load(f)
    arm_names=res.get('arm_names',ARM_NAMES)
    joint_names=[f'J{i+1}' for i in range(NDOF)]
    for name in arm_names:
        if name not in res: continue
        segs=res[name]['spline']['segments']
        n_seg=len(segs)
        fig,axes=plt.subplots(n_seg,NDOF,figsize=(14,2.5*n_seg))
        if n_seg==1: axes=[axes]
        for seg_idx,seg in enumerate(segs):
            cp_new=np.array(seg['cp']); cp_old=np.array(seg.get('cp_orig',seg['cp']))
            mod=seg.get('modified',False)
            # CP indices normalised to [0,1] so plots are comparable when CP count differs
            x_new = np.linspace(0., 1., len(cp_new))
            x_old = np.linspace(0., 1., len(cp_old))
            for j in range(NDOF):
                ax=axes[seg_idx][j]
                ax.plot(x_old, np.degrees(cp_old[:,j]),'b--o',ms=4,lw=1.5,label='orig')
                ax.plot(x_new, np.degrees(cp_new[:,j]),'r-s',ms=4,lw=1.5,label='SKAR-N')
                if seg_idx==0: ax.set_title(joint_names[j],fontsize=9)
                if j==0:
                    n_cp_label = f'{len(cp_new)}cp' if mod else 'orig'
                    ax.set_ylabel(f'Seg {seg_idx}\n{"[MOD]" if mod else "[OK]"}\n{n_cp_label}',fontsize=8)
                ax.grid(True,alpha=0.3); ax.tick_params(labelsize=7)
                if seg_idx==0 and j==0: ax.legend(fontsize=7)
        plt.suptitle(f'{name} -- CP Modification per Segment',fontsize=11)
        plt.tight_layout()
        fname=f's76_{name}_cp.png'; plt.savefig(fname,dpi=150,bbox_inches='tight')
        plt.close(); print(f'  Saved: {fname}')

if __name__=='__main__': main()