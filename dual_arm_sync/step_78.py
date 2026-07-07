#!/usr/bin/env python3
"""step_78.py -- Hardware Validator: extended verification with statistics"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _robot import NDOF, ARM_NAMES

def main():
    print('\nSTEP 78 -- Hardware Validator')
    results=[]
    for i in range(1,6):  # Look for multiple execution results
        fname=f's66_execution_{i}.json' if i>1 else 's66_execution.json'
        if os.path.exists(fname):
            with open(fname) as f: results.append(json.load(f))
    if not results:
        print('  No execution results found (run step_66 first)'); sys.exit(1)
    arm_names=results[0].get('arm_names',ARM_NAMES)
    ee_errs={n:[] for n in arm_names}; jt_errs={n:[] for n in arm_names}; successes=[]
    for r in results:
        successes.append(r.get('execution',{}).get('success',False))
        for name in arm_names:
            v=r.get('verification',{}).get(name,{})
            if v.get('ee_error_mm') is not None:
                ee_errs[name].append(float(v['ee_error_mm']))
                jt_errs[name].append(float(v['joint_error_deg']))
    out={'n_trials':len(results),'success_rate':round(sum(successes)/len(successes),3),
         'per_arm':{}}
    for name in arm_names:
        if ee_errs[name]:
            out['per_arm'][name]={
                'mean_ee_mm':round(float(np.mean(ee_errs[name])),3),
                'std_ee_mm' :round(float(np.std(ee_errs[name])),3),
                'max_ee_mm' :round(float(np.max(ee_errs[name])),3),
                'mean_jt_deg':round(float(np.mean(jt_errs[name])),3)}
    with open('s78_hardware.json','w') as f: json.dump(out,f,indent=2)
    print(f'  Trials: {len(results)}  Success: {out["success_rate"]*100:.0f}%')
    for n,d in out['per_arm'].items():
        print(f'  [{n}] EE: {d["mean_ee_mm"]:.2f}±{d["std_ee_mm"]:.2f}mm  max={d["max_ee_mm"]:.2f}mm')
    print('  Saved: s78_hardware.json')

if __name__=='__main__': main()