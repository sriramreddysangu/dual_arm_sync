#!/usr/bin/env python3
"""step_79.py -- Experiment Automator: runs full 61-67 pipeline N times"""
import argparse, json, os, subprocess, sys, time, shutil
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _robot import NDOF, POS_LIM, ROBOT_BASES, ARM_NAMES

def run_steps(step_dir, steps):
    for step in steps:
        script=os.path.join(step_dir,f'{step}.py')
        r=subprocess.run([sys.executable,script],capture_output=True,text=True)
        if r.returncode!=0: return False, step, r.stderr[-300:]
    return True, None, None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--trials',type=int,default=10)
    ap.add_argument('--seed',  type=int,default=0)
    ap.add_argument('--outdir',type=str,default='experiment_results')
    args=ap.parse_args()
    rng=np.random.default_rng(args.seed)
    step_dir=os.path.dirname(__file__)
    os.makedirs(args.outdir, exist_ok=True)
    summary=[]

    for trial in range(args.trials):
        t0=time.time()
        trial_dir=os.path.join(args.outdir,f'trial_{trial:04d}'); os.makedirs(trial_dir,exist_ok=True)
        # Generate random targets -- step_61 reads s61_ik.json if present
        targets={'duration':rng.uniform(8.,14.),'arm_names':ARM_NAMES}
        for n in ARM_NAMES:
            sq=np.clip(rng.uniform(POS_LIM[:,0],POS_LIM[:,1],NDOF), POS_LIM[:,0], POS_LIM[:,1])
            eq=np.clip(rng.uniform(POS_LIM[:,0],POS_LIM[:,1],NDOF), POS_LIM[:,0], POS_LIM[:,1])
            targets[n]={'start_joints':sq.tolist(),'target_joints':eq.tolist(),
                        'base':ROBOT_BASES[n].tolist(),'target_ee_world':[0,0,0]}
        with open('s61_ik.json','w') as f: json.dump(targets,f)
        ok,failed_step,err=run_steps(step_dir,['step_62','step_63','step_64','step_65','step_67'])
        t_ms=round((time.time()-t0)*1000,1)
        entry={'trial':trial,'ok':ok,'time_ms':t_ms}
        if not ok: entry['failed_step']=failed_step
        if os.path.exists('s67_metrics.json'):
            with open('s67_metrics.json') as f: m=json.load(f)
            entry['resolve_ms']=m['planning']['resolve_time_ms']
            entry['cf']=m['safety']['collision_free_after']
        summary.append(entry)
        for fname in ['s61_ik.json','s62_trajectories.json','s63_collision_map.json',
                      's64_resolved.json','s65_synchronized.json','s67_metrics.json']:
            if os.path.exists(fname): shutil.copy(fname,os.path.join(trial_dir,fname))
        print(f'  Trial {trial:4d}: {"OK" if ok else "FAIL"}  {t_ms:.0f}ms')

    n_ok=sum(1 for s in summary if s['ok'])
    out={'trials':args.trials,'seed':args.seed,'success_rate':round(n_ok/args.trials,3),'per_trial':summary}
    with open(os.path.join(args.outdir,'s79_experiment.json'),'w') as f: json.dump(out,f,indent=2)
    print(f'\nSuccess rate: {n_ok}/{args.trials} = {out["success_rate"]*100:.0f}%')
    print(f'Saved: {args.outdir}/s79_experiment.json')

if __name__=='__main__': main()