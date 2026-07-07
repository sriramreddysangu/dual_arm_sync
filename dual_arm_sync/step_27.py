#!/usr/bin/env python3
"""
step_27.py  --  Monte-Carlo Benchmark Harness  [6 ARM]
======================================================
Runs N random target-set trials through the FULL planning pipeline headlessly
(no Gazebo, no typing) and reports the statistics for the paper:

  IK -> B-spline seed -> collision scan -> step_24 per-pair retraction (B-spline)
     -> step_25 Kuramoto phase-lag -> verdict

For every trial it records whether the raw seed collided, whether step_24
(B-spline retraction) cleared it, whether step_25 (Kuramoto) was needed and
whether it cleared the residual, the operation-time inflation (final/nominal),
and the mean B-spline modification. Aggregates are printed in a paper-ready block
and saved to step_27_results.csv (per trial) + step_27_summary.json.

Usage:
  ros2 run dual_arm_sync step_27                # 500 trials, seed 0
  ros2 run dual_arm_sync step_27 --trials 200 --seed 7
  python3 step_27.py --trials 500              # also runs standalone
"""
import json, os, sys, time, argparse, io, contextlib
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import step_21 as S21
import step_22 as S22
import step_24 as S24
import step_25 as S25
from _robot6x import (NDOF, VEL_LIM, ACC_LIM, ROBOT_BASES, ARM_NAMES)

NOMINAL_DUR    = 10.0
DEFAULT_TRIALS = 500
REACH_MIN, REACH_MAX = 0.35, 0.85     # target shell radius (m) in the arm base frame
EL_MIN, EL_MAX = np.radians(5), np.radians(80)
IK_RETRY       = 4                     # regenerate targets if an arm has no IK


def random_target_local(rng):
    r  = rng.uniform(REACH_MIN, REACH_MAX)
    az = rng.uniform(-np.pi, np.pi)
    el = rng.uniform(EL_MIN, EL_MAX)
    return np.array([r*np.cos(el)*np.cos(az), r*np.cos(el)*np.sin(az), 0.20 + r*np.sin(el)])


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def solve_ik(names, bases, rng):
    """Return per-arm target joint configs, or None if IK fails for any arm."""
    for _ in range(IK_RETRY):
        arms_data = {}
        for n in names:
            arms_data[n] = {'target_local': random_target_local(rng), 'target_rot': None,
                            'start_q': np.zeros(NDOF), 'base': np.array(bases[n]), 'name': n}
        best = quiet(S21.greedy, arms_data)
        if best is not None:
            return best, arms_data
    return None, None


def rescale_duration(cp, duration, names):
    scale = 1.0
    for n in names:
        pos = S24.eval_cp(cp[n], duration); dt = duration / max(len(pos) - 1, 1)
        vel = np.gradient(pos, dt, axis=0); acc = np.gradient(vel, dt, axis=0)
        for j in range(NDOF):
            vp = float(np.max(np.abs(vel[:, j]))); ap = float(np.max(np.abs(acc[:, j])))
            if vp > VEL_LIM[j]: scale = max(scale, vp / VEL_LIM[j])
            if ap > ACC_LIM[j]: scale = max(scale, float(np.sqrt(ap / ACC_LIM[j])))
    return duration * scale * S24.VEL_SAFETY if scale > 1.0 else duration


def kuramoto_best(paths, duration, names, bases, in_cs, in_mind):
    """Replicate step_25's bounded-lag sweep with the step_24 result as baseline."""
    final_cs, final_mind, used = in_cs, in_mind, False
    for (kl, ds, cap) in S25.LAG_SWEEP:
        sync, _ = S25.kuramoto(paths, duration, names, bases, kl, ds, cap)
        sr = {n: S25.resample(sync[n], duration) for n in names}
        cs, mind = S25.scan_count(sr, names, bases)
        if cs < final_cs or (cs == final_cs and used and mind > final_mind):
            final_cs, final_mind, used = cs, mind, True
        if final_cs == 0:
            break
    return final_cs, final_mind, used


def run_trial(rng, names, bases):
    best, arms_data = solve_ik(names, bases, rng)
    if best is None:
        return {'ik_fail': True}
    cp = {n: S22.seed_cp(arms_data[n]['start_q'], best[n]) for n in names}
    seed_pos = {n: S24.eval_cp(cp[n], NOMINAL_DUR) for n in names}
    seed_cs, seed_mind = S25.scan_count(seed_pos, names, bases)
    rec = {'ik_fail': False, 'seed_coll_steps': int(seed_cs),
           'seed_min_cm': round(seed_mind * 100, 2), 'nominal_dur': NOMINAL_DUR}

    if seed_cs == 0:
        rec.update({'category': 'safe', 'final_dur': NOMINAL_DUR, 'op_increase_pct': 0.0,
                    'mod': 0.0, 'final_coll': 0, 'resolved': True})
        return rec

    # ---- step_24: per-pair B-spline retraction ----
    orig = {n: seed_pos[n] for n in names}
    cp24, residual, _ = quiet(S24.resolve, cp, NOMINAL_DUR, names, bases)
    dur24 = rescale_duration(cp24, NOMINAL_DUR, names)
    pos24 = {n: S24.eval_cp(cp24[n], NOMINAL_DUR) for n in names}
    mod = float(np.mean([np.mean(np.linalg.norm(pos24[n] - orig[n], axis=1)) for n in names]))
    op_pct = round((dur24 / NOMINAL_DUR - 1) * 100, 1)

    if residual == 0:
        rec.update({'category': 'retraction', 'final_dur': round(dur24, 3),
                    'op_increase_pct': op_pct, 'mod': round(mod, 4),
                    'final_coll': 0, 'resolved': True})
        return rec

    # ---- step_25: Kuramoto phase-lag on the residual ----
    paths24 = {n: S24.eval_cp(cp24[n], dur24) for n in names}
    in_cs, in_mind = S25.scan_count(paths24, names, bases)
    fcs, fmind, used = kuramoto_best(paths24, dur24, names, bases, in_cs, in_mind)
    rec.update({'final_dur': round(dur24, 3), 'op_increase_pct': op_pct,
                'mod': round(mod, 4), 'final_coll': int(fcs)})
    if fcs == 0:
        rec.update({'category': 'kuramoto', 'resolved': True})
    else:
        rec.update({'category': 'unresolved', 'resolved': False})
    return rec


def pct(x, p):
    return round(float(np.percentile(x, p)), 1) if len(x) else 0.0


def summarize(recs, n_total, seed, secs):
    ik_fail = sum(1 for r in recs if r.get('ik_fail'))
    valid = [r for r in recs if not r.get('ik_fail')]
    nv = len(valid)
    cats = {c: [r for r in valid if r['category'] == c]
            for c in ('safe', 'retraction', 'kuramoto', 'unresolved')}
    colliding = nv - len(cats['safe'])
    resolved = len(cats['safe']) + len(cats['retraction']) + len(cats['kuramoto'])

    def opstats(rs):
        v = [r['op_increase_pct'] for r in rs]
        return {'n': len(rs), 'mean_op_inc_pct': round(float(np.mean(v)), 1) if v else 0.0,
                'median_op_inc_pct': pct(v, 50), 'p95_op_inc_pct': pct(v, 95),
                'max_op_inc_pct': round(float(np.max(v)), 1) if v else 0.0}

    summary = {
        'trials_requested': n_total, 'seed': seed, 'runtime_s': round(secs, 1),
        'ik_fail': ik_fail, 'valid_trials': nv,
        'seed_collision_rate_pct': round(100.0 * colliding / nv, 1) if nv else 0.0,
        'counts': {c: len(cats[c]) for c in cats},
        'share_of_colliding_pct': {
            'retraction_only': round(100.0 * len(cats['retraction']) / colliding, 1) if colliding else 0.0,
            'kuramoto_needed': round(100.0 * len(cats['kuramoto']) / colliding, 1) if colliding else 0.0,
            'unresolved':      round(100.0 * len(cats['unresolved']) / colliding, 1) if colliding else 0.0},
        'overall_success_pct': round(100.0 * resolved / nv, 1) if nv else 0.0,
        'op_time': {'retraction': opstats(cats['retraction']),
                    'kuramoto':   opstats(cats['kuramoto']),
                    'all_resolved_collisions': opstats(cats['retraction'] + cats['kuramoto'])},
        'bspline_mod': {
            'mean': round(float(np.mean([r['mod'] for r in cats['retraction'] + cats['kuramoto']])), 4)
                    if (cats['retraction'] + cats['kuramoto']) else 0.0,
            'max': round(float(np.max([r['mod'] for r in cats['retraction'] + cats['kuramoto']])), 4)
                   if (cats['retraction'] + cats['kuramoto']) else 0.0}}
    return summary, cats, colliding, resolved, nv


def print_report(s, cats, colliding, resolved, nv):
    L = '=' * 70
    print('\n' + L); print('  STEP 27  --  BENCHMARK SUMMARY  (paste-ready for paper)'); print(L)
    print(f'  trials valid / requested : {nv} / {s["trials_requested"]}   '
          f'(IK fails: {s["ik_fail"]})   runtime: {s["runtime_s"]}s')
    print(f'  raw-seed collision rate  : {s["seed_collision_rate_pct"]}%  '
          f'({colliding}/{nv} target-sets collided)')
    print(f'  overall success          : {s["overall_success_pct"]}%  ({resolved}/{nv} executable)')
    print('\n  Resolution breakdown (of the %d colliding sets):' % colliding)
    print('  ----------------------------------------------------------------')
    print('   stage                      count    %colliding   op-time increase')
    print('  ----------------------------------------------------------------')
    r = s['op_time']['retraction']; k = s['op_time']['kuramoto']
    sh = s['share_of_colliding_pct']
    print(f'   B-spline retraction(s24)  {len(cats["retraction"]):5d}     {sh["retraction_only"]:6.1f}%   '
          f'mean {r["mean_op_inc_pct"]:5.1f}%  med {r["median_op_inc_pct"]:5.1f}%  max {r["max_op_inc_pct"]:5.1f}%')
    print(f'   + Kuramoto phase-lag(s25) {len(cats["kuramoto"]):5d}     {sh["kuramoto_needed"]:6.1f}%   '
          f'mean {k["mean_op_inc_pct"]:5.1f}%  med {k["median_op_inc_pct"]:5.1f}%  max {k["max_op_inc_pct"]:5.1f}%')
    print(f'   unresolved (spatial)      {len(cats["unresolved"]):5d}     {sh["unresolved"]:6.1f}%        --')
    print('  ----------------------------------------------------------------')
    a = s['op_time']['all_resolved_collisions']
    print(f'  all resolved collisions   : mean op-time +{a["mean_op_inc_pct"]}%  '
          f'median +{a["median_op_inc_pct"]}%  p95 +{a["p95_op_inc_pct"]}%  max +{a["max_op_inc_pct"]}%')
    print(f'  B-spline modification     : mean {s["bspline_mod"]["mean"]} rad  max {s["bspline_mod"]["max"]} rad')
    print(L)
    print('  Saved: step_27_results.csv  (per-trial)   step_27_summary.json  (aggregate)\n')


def main(args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=DEFAULT_TRIALS)
    ap.add_argument('--seed', type=int, default=0)
    a, _ = ap.parse_known_args(args if args is not None else sys.argv[1:])

    names = list(ARM_NAMES)
    bases = {n: np.array(ROBOT_BASES[n]) for n in names}
    rng = np.random.default_rng(a.seed)

    print('\n' + '=' * 70)
    print(f'  STEP 27  --  Monte-Carlo Benchmark: {a.trials} random trials (seed {a.seed})')
    print('  Pipeline: IK -> seed -> scan -> step_24 retraction -> step_25 Kuramoto')
    print('=' * 70)

    recs = []; t0 = time.time()
    for i in range(a.trials):
        recs.append(run_trial(rng, names, bases))
        if (i + 1) % 25 == 0 or (i + 1) == a.trials:
            nv = sum(1 for r in recs if not r.get('ik_fail'))
            res = sum(1 for r in recs if r.get('resolved'))
            print(f'  ...{i+1}/{a.trials}  (valid {nv}, resolved {res}, '
                  f'elapsed {time.time()-t0:.0f}s)')

    summary, cats, colliding, resolved, nv = summarize(recs, a.trials, a.seed, time.time() - t0)

    # per-trial CSV
    cols = ['trial', 'ik_fail', 'category', 'resolved', 'seed_coll_steps', 'seed_min_cm',
            'final_dur', 'op_increase_pct', 'mod', 'final_coll']
    with open('step_27_results.csv', 'w') as fh:
        fh.write(','.join(cols) + '\n')
        for i, r in enumerate(recs):
            row = [i] + [r.get(c, '') for c in cols[1:]]
            fh.write(','.join(str(v) for v in row) + '\n')
    with open('step_27_summary.json', 'w') as fh:
        json.dump(summary, fh, indent=2)

    print_report(summary, cats, colliding, resolved, nv)


if __name__ == '__main__':
    main()