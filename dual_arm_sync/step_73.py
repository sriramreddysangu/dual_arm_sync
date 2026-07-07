#!/usr/bin/env python3
"""
step_73.py -- Lemma 3 Numerical Verifier
Numerically verifies the separation guarantee:
Given |j1_i - j1_j| > theta, what is the minimum inter-arm clearance
when both arms are in retracted pose [j1, 0,0,0,0,0]?
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _robot import ROBOT_BASES, ARM_NAMES, pair_min_dist, J1_SEP_THRESH

def main():
    print('\nSTEP 73 -- Lemma 3: Separation Guarantee Verifier')
    arm_names=ARM_NAMES; bases={n:ROBOT_BASES[n] for n in arm_names}
    theta_vals=np.linspace(0, np.pi, 37)
    results=[]
    for theta in theta_vals:
        # arm_i: j1=0, arm_j: j1=theta
        qi=np.array([0.,0.,0.,0.,0.,0.])
        qj=np.array([theta,0.,0.,0.,0.,0.])
        d=pair_min_dist(qi,bases[arm_names[0]],qj,bases[arm_names[1]])
        results.append({'theta_deg':round(np.degrees(theta),1),'min_dist_cm':round(d*100,2)})
    # Find threshold where clearance exceeds safety margin
    safe_thresh=next((r['theta_deg'] for r in results if r['min_dist_cm']>0), None)
    out={'lemma3_data':results,
         'j1_sep_threshold_used_deg':round(np.degrees(J1_SEP_THRESH),1),
         'min_clearance_at_threshold_cm':
             next((r['min_dist_cm'] for r in results
                   if abs(r['theta_deg']-np.degrees(J1_SEP_THRESH))<3), None)}
    with open('s73_lemma3.json','w') as f: json.dump(out,f,indent=2)
    print(f'  Theta range: 0 to 180 deg  ({len(results)} points)')
    print(f'  At J1_SEP_THRESH={np.degrees(J1_SEP_THRESH):.1f}deg: '
          f'clearance={out["min_clearance_at_threshold_cm"]}cm')
    print('  Saved: s73_lemma3.json')

if __name__=='__main__': main()