#!/usr/bin/env python3
"""step_72.py -- Kuramoto Convergence Analyzer"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _robot import ARM_NAMES

def main():
    print('\nSTEP 72 -- Kuramoto Convergence Analyzer')
    if not os.path.exists('s65_synchronized.json'): print('s65_synchronized.json not found'); sys.exit(1)
    with open('s65_synchronized.json') as f: s=json.load(f)
    kur=s.get('synchronisation_report',{}); arm_names=s.get('arm_names',ARM_NAMES)
    out={'collision_free':kur.get('collision_free',True),
         'phase_separation':s.get('phase_separation',0.),
         'duration_overhead_pct':s.get('duration_overhead',0.)*100,
         'pair_reports':kur.get('pair_reports',{})}
    for pair,rep in kur.get('pair_reports',{}).items():
        print(f'  {pair}: min_dist={rep["min_dist_m"]*100:.1f}cm  '
              f'critical={rep["critical"]}  free={rep["collision_free"]}')
    print(f'  Phase separation: {np.degrees(s.get("phase_separation",0.)):.2f} deg-equiv')
    print(f'  Duration overhead: {out["duration_overhead_pct"]:.1f}%')
    with open('s72_kuramoto.json','w') as f: json.dump(out,f,indent=2)
    print('  Saved: s72_kuramoto.json')

if __name__=='__main__': main()