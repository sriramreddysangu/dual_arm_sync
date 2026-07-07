#!/usr/bin/env python3
"""step_71.py -- Retraction Strategy Comparator: j1-preserved vs full home"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _robot import NDOF, ROBOT_BASES, ARM_NAMES, pair_min_dist, J1_SEP_THRESH

def apply_strategy(cp_segs, coll_seg, strategy, j1_arc, n_cp=4):
    import copy; result=copy.deepcopy(cp_segs)
    cp=np.zeros((n_cp,NDOF)); cp[0]=cp_segs[coll_seg][0][0]; cp[-1]=cp_segs[coll_seg][0][-1]
    if strategy=='j1_preserved' and j1_arc is not None:
        ints=np.linspace(0,1,n_cp-2+2)[1:-1]
        for k,t in enumerate(ints): cp[k+1,0]=float(np.interp(t,np.linspace(0,1,len(j1_arc)),j1_arc))
    result[coll_seg]=(cp,None)
    return result

def main():
    print('\nSTEP 71 -- Retraction Strategy Comparator')
    if not os.path.exists('s64_resolved.json'): print('s64_resolved.json not found'); sys.exit(1)
    with open('s64_resolved.json') as f: res=json.load(f)
    ret_log=res.get('retraction_log',{})
    if not ret_log: print('  No collision pairs found -- no comparison needed'); sys.exit(0)
    arm_names=res.get('arm_names',ARM_NAMES)
    out={'arm_names':arm_names,'pairs':{}}
    for pair_str,info in ret_log.items():
        dj1=info.get('delta_j1_rad',0)
        strategy_used=info.get('strategy','full_home')
        out['pairs'][pair_str]={
            'delta_j1_deg':round(np.degrees(dj1),2),'strategy_used':strategy_used,
            'j1_sep_thresh_deg':round(np.degrees(J1_SEP_THRESH),1),
            'threshold_met':dj1>J1_SEP_THRESH,'coll_seg':info.get('coll_seg'),
            'phase'        :info.get('phase','?'),
            'alpha_final'  :info.get('alpha','?'),
            'cp_count'     :info.get('cp_count','?'),
            'target_segs'  :info.get('target_segs',[]),
        }
        print(f'  {pair_str}: delta_j1={np.degrees(dj1):.1f}deg  '
              f'threshold_met={dj1>J1_SEP_THRESH}  used={strategy_used}  '
              f'phase={info.get("phase","?")}  alpha={info.get("alpha","?")}')
    with open('s71_retraction.json','w') as f: json.dump(out,f,indent=2)
    print('  Saved: s71_retraction.json')

if __name__=='__main__': main()