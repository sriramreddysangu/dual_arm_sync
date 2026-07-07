#!/usr/bin/env python3
"""step_68.py -- N-Trial Benchmark Runner"""
import argparse, json, os, subprocess, sys, time, random
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _robot import NDOF, POS_LIM, VEL_LIM, ROBOT_BASES, ARM_NAMES, fk_world

REACH_MIN=0.40; REACH_MAX=1.10; Z_MIN=0.10; Z_MAX=1.00

def random_target(rng, base):
    for _ in range(300):
        r   = rng.uniform(REACH_MIN, REACH_MAX)
        th  = rng.uniform(0, 2*np.pi)
        z   = rng.uniform(Z_MIN, Z_MAX)
        rxy = np.sqrt(max(r**2 - z**2, 0.0))
        if rxy < 0.08: continue
        return (base + np.array([rxy*np.cos(th), rxy*np.sin(th), z])).tolist()
    return (base + np.array([0.6, 0.0, 0.6])).tolist()

def run_pipeline(ik_data):
    with open('s61_ik.json','w') as f: json.dump(ik_data, f)
    results = {}
    for step in ['step_62','step_63','step_64','step_65']:
        t0 = time.time()
        r  = subprocess.run([sys.executable, f'{os.path.dirname(__file__)}/{step}.py'],
                            capture_output=True, text=True)
        results[step] = {'time_ms': round((time.time()-t0)*1000,1), 'ok': r.returncode==0}
        if r.returncode != 0: results['error'] = r.stderr[-500:]; break
    return results

def classify_outcome():
    if not os.path.exists('s64_resolved.json'): return 'FAIL_RESOLVE'
    with open('s64_resolved.json') as f: d=json.load(f)

    # Check for edge-segment giveup BEFORE Kuramoto
    ret_log = d.get('retraction_log',{})
    if any(v.get('giveup_reason')=='edge_segment_cannot_expand' for v in ret_log.values()):
        return 'GIVEUP_EDGE_SEG'

    if not os.path.exists('s65_synchronized.json'): return 'FAIL_KURAMOTO'
    with open('s65_synchronized.json') as f: s=json.load(f)
    if not s.get('synchronisation_report',{}).get('collision_free',False):
        return 'UNRESOLVED'

    it = d.get('iterations_used',0)
    if it == 0: return 'SAFE_NO_COLL'

    # Look at the retraction_log to see which phase was used
    # (depends on whether last-segment fast-path kicked in)
    ret_log = d.get('retraction_log', {})
    phases_seen = [info.get('phase','') for info in ret_log.values() if isinstance(info, dict)]
    used_lastseg_fastpath = (phases_seen and phases_seen[0] in ('two_seg','three_seg'))

    if used_lastseg_fastpath:
        if it <= 4: return f'RESOLVED_LASTSEG_{it}'
        if it == 5: return 'RESOLVED_LASTSEG_5'
        return f'RESOLVED_LASTSEG_{it}'

    # Normal schedule:
    #   iter 1..5 = single_seg phase  -> RESOLVED_CP_1..5
    #   iter 6..8 = two_seg phase     -> RESOLVED_2SEG_1..3
    #   iter 9..10 = three_seg phase  -> RESOLVED_3SEG_1..2
    if it <= 5:  return f'RESOLVED_CP_{it}'
    if it <= 8:  return f'RESOLVED_2SEG_{it - 5}'
    if it <= 10: return f'RESOLVED_3SEG_{it - 8}'
    return f'RESOLVED_{it}'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int,   default=100)
    ap.add_argument('--seed',   type=int,   default=42)
    ap.add_argument('--duration',type=float,default=10.0)
    args = ap.parse_args()
    rng  = np.random.default_rng(args.seed)
    outcomes = {}; trials_data = []; t0_total = time.time()

    print(f'\nBenchmark: {args.trials} trials  seed={args.seed}  dur={args.duration}s')
    for trial in range(args.trials):
        start_qs = {n: np.clip(rng.uniform(-1.,1.,NDOF), POS_LIM[:,0], POS_LIM[:,1]) for n in ARM_NAMES}
        targets  = {n: random_target(rng, ROBOT_BASES[n]) for n in ARM_NAMES}
        ik_data  = {'duration': args.duration, 'arm_names': ARM_NAMES}
        for n in ARM_NAMES:
            ik_data[n] = {
                'start_joints' : start_qs[n].tolist(),
                'target_joints': np.clip(rng.uniform(POS_LIM[:,0],POS_LIM[:,1],NDOF), POS_LIM[:,0], POS_LIM[:,1]).tolist(),
                'target_ee_world': targets[n], 'base': ROBOT_BASES[n].tolist(),
            }
        t_trial = time.time()
        run_pipeline(ik_data)
        outcome = classify_outcome()
        t_ms = round((time.time()-t_trial)*1000,1)
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        trials_data.append({'trial': trial, 'outcome': outcome, 'time_ms': t_ms})
        if trial % 10 == 0: print(f'  {trial}/{args.trials}  {outcome}')

    total_ms = round((time.time()-t0_total)*1000,1)
    out = {'trials': args.trials, 'seed': args.seed, 'total_time_ms': total_ms,
           'outcomes': outcomes, 'per_trial': trials_data}
    with open('s68_benchmark.json','w') as f: json.dump(out, f, indent=2)
    print(f'\nOutcomes: {outcomes}')
    print(f'Saved: s68_benchmark.json')

if __name__=='__main__': main()