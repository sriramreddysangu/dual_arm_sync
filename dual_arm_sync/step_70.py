#!/usr/bin/env python3
"""step_70.py -- Path Quality Analyzer: deviation from original per segment"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _robot import NDOF, ROBOT_BASES, ARM_NAMES

def main():
    print('\nSTEP 70 -- Path Quality Analyzer')
    for f in ['s62_trajectories.json','s64_resolved.json']:
        if not os.path.exists(f): print(f'  {f} not found'); sys.exit(1)
    with open('s62_trajectories.json') as f: orig=json.load(f)
    with open('s64_resolved.json')     as f: res =json.load(f)
    arm_names=orig.get('arm_names',ARM_NAMES)
    per_arm={}
    for name in arm_names:
        pos_o=np.array(orig[name]['trajectory']['positions'],dtype=float)
        pos_r=np.array(res[name]['trajectory']['positions'],dtype=float)
        n=min(len(pos_o),len(pos_r))
        pos_o=pos_o[:n]; pos_r=pos_r[:n]
        n_seg=int(orig[name]['spline']['n_seg'])
        seg_dev=[]; total_dev=float(np.mean(np.linalg.norm(pos_r-pos_o,axis=1)))
        for seg in range(n_seg):
            s0=seg/n_seg; s1=(seg+1)/n_seg
            arc=np.linspace(0,1,n)
            mask=(arc>=s0-1e-9)&(arc<=s1+1e-9)
            if mask.sum()>0:
                d=float(np.mean(np.linalg.norm(pos_r[mask]-pos_o[mask],axis=1)))
            else: d=0.
            seg_dev.append(round(d,6))
        segs_modified=[i for i,d in enumerate(seg_dev) if d>1e-9]
        per_arm[name]={'seg_deviation_rad':seg_dev,'total_mean_dev_rad':round(total_dev,6),
                       'segments_modified':segs_modified,
                       'surgical':len(segs_modified)<n_seg}
        print(f'  [{name}] total_dev={np.degrees(total_dev):.3f}deg  '
              f'segs_modified={segs_modified}  surgical={per_arm[name]["surgical"]}')
    out={'arm_names':arm_names,'per_arm':per_arm}
    with open('s70_quality.json','w') as f: json.dump(out,f,indent=2)
    print('  Saved: s70_quality.json')

if __name__=='__main__': main()