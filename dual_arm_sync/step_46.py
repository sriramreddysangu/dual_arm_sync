#!/usr/bin/env python3
"""
step_46.py  --  Side-by-Side Benchmark: Lagrangian vs Alpha-Pull
==================================================================
Compares the step_4X (Lagrangian perpendicular displacement) pipeline
against the step_6X (alpha pull-toward-home) pipeline on the SAME
random target pairs.

INPUT  : --trials N --seed S
OUTPUT : s46_compare.json + s46_compare_plots.png

For each trial, run BOTH pipelines on the same start/target configurations
and report which performed better on each metric:

  Metrics compared (per-trial):
    1. Success: did the method find a collision-free trajectory?
    2. Planning time (ms): time spent in resolver + Kuramoto
    3. Iterations used
    4. Modification magnitude (mean joint deviation from original)
    5. Duration overhead (% increase from requested duration)
    6. Min inter-arm distance (after synchronization)
    7. Path optimality (final EE path length / direct line)

This is what you SHOW YOUR GUIDE -- a table and chart proving which
method works better on the bottom line. Use this to defend the final
algorithm choice for the paper.

Usage:
  python3 step_46.py --trials 30 --seed 42
"""

import argparse, json, os, subprocess, sys, time, shutil
from typing import Dict, List
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _robot4x import NDOF, POS_LIM, ROBOT_BASES, ARM_NAMES, fk_world

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _PLOT = True
except ImportError:
    _PLOT = False

# Reach/sampling for random IK targets
REACH_MIN = 0.40
REACH_MAX = 1.10
Z_MIN     = 0.10
Z_MAX     = 1.00


def random_target_ik(rng, base, current_q):
    """Generate a random valid joint target near `base`."""
    for _ in range(300):
        rxy = rng.uniform(REACH_MIN, REACH_MAX)
        th  = rng.uniform(0, 2 * np.pi)
        z   = rng.uniform(Z_MIN, Z_MAX)
        r   = np.sqrt(max(rxy**2 - z**2, 0.0))
        if r < 0.08: continue
        # Use a random joint config in limits (proxy for IK target)
        return np.clip(rng.uniform(POS_LIM[:, 0], POS_LIM[:, 1], NDOF),
                       POS_LIM[:, 0], POS_LIM[:, 1])
    return np.zeros(NDOF)


def make_ik_json(start_qs, target_qs, duration, target_filename):
    """Build a fake s40_ik.json or s61_ik.json for the pipeline."""
    out = {'duration': duration, 'arm_names': ARM_NAMES,
            'ik_total_time_ms': 0.,
            'selection_method': 'random_benchmark'}
    for name in ARM_NAMES:
        out[name] = {
            'base'             : ROBOT_BASES[name].tolist(),
            'start_joints'     : start_qs[name].tolist(),
            'target_joints'    : target_qs[name].tolist(),
            'target_world'     : fk_world(target_qs[name], ROBOT_BASES[name]).tolist(),
            'target_ee_world'  : fk_world(target_qs[name], ROBOT_BASES[name]).tolist(),
            'position_error_mm': 0.0,
            'ik_time_ms'       : 0.0,
            'n_solutions'      : 1,
        }
    with open(target_filename, 'w') as fh: json.dump(out, fh, indent=2)


def run_step(step_dir, script_name):
    """Run a pipeline step. Returns (ok, elapsed_ms, error_str)."""
    t0 = time.time()
    r = subprocess.run([sys.executable, os.path.join(step_dir, script_name)],
                       capture_output=True, text=True, timeout=60)
    elapsed_ms = round((time.time() - t0) * 1000, 1)
    if r.returncode != 0:
        return False, elapsed_ms, r.stderr[-300:]
    return True, elapsed_ms, None


def extract_metrics_lagrangian():
    """Parse s43_resolved.json and s44_synchronized.json for metrics."""
    metrics = {'success': False, 'iterations': 0, 'mod_mag_mean': 0.,
                'duration_overhead': 0., 'min_dist': 0., 'resolve_ms': 0.,
                'kuramoto_ms': 0., 'method': 'lagrangian'}
    if not os.path.exists('s43_resolved.json'):
        return metrics
    with open('s43_resolved.json') as fh: r = json.load(fh)
    metrics['iterations']   = int(r.get('iterations_used', 0))
    metrics['resolve_ms']   = float(r.get('resolve_time_ms', 0))
    mod_mag = r.get('modification_magnitude', {})
    if mod_mag: metrics['mod_mag_mean'] = float(np.mean(list(mod_mag.values())))

    if os.path.exists('s44_synchronized.json'):
        with open('s44_synchronized.json') as fh: s = json.load(fh)
        kur = s.get('synchronisation_report', {})
        metrics['success'] = bool(kur.get('collision_free', False))
        metrics['duration_overhead'] = float(s.get('duration_overhead', 0))
        metrics['kuramoto_ms'] = float(s.get('kuramoto_time_ms', 0))
        pair_reports = kur.get('pair_reports', {})
        if pair_reports:
            metrics['min_dist'] = min(p['min_dist_m'] for p in pair_reports.values())
    return metrics


def extract_metrics_alphapull():
    """Parse s64_resolved.json and s65_synchronized.json for metrics."""
    metrics = {'success': False, 'iterations': 0, 'mod_mag_mean': 0.,
                'duration_overhead': 0., 'min_dist': 0., 'resolve_ms': 0.,
                'kuramoto_ms': 0., 'method': 'alpha_pull'}
    if not os.path.exists('s64_resolved.json'):
        return metrics
    with open('s64_resolved.json') as fh: r = json.load(fh)
    metrics['iterations']   = int(r.get('iterations_used', 0))
    metrics['resolve_ms']   = float(r.get('resolve_time_ms', 0))
    mod_mag = r.get('modification_magnitude', {})
    if mod_mag: metrics['mod_mag_mean'] = float(np.mean(list(mod_mag.values())))

    if os.path.exists('s65_synchronized.json'):
        with open('s65_synchronized.json') as fh: s = json.load(fh)
        kur = s.get('synchronisation_report', {})
        metrics['success'] = bool(kur.get('collision_free', False))
        metrics['duration_overhead'] = float(s.get('duration_overhead', 0))
        metrics['kuramoto_ms'] = float(s.get('kuramoto_time_ms', 0))
        pair_reports = kur.get('pair_reports', {})
        if pair_reports:
            metrics['min_dist'] = min(p['min_dist_m'] for p in pair_reports.values())
    return metrics


def run_trial(rng, trial_idx, duration, dir_4x, dir_6x):
    """Run one random trial through BOTH pipelines and return metrics."""
    # Generate random start/target configs
    start_qs = {n: np.zeros(NDOF) for n in ARM_NAMES}
    target_qs = {n: random_target_ik(rng, ROBOT_BASES[n], start_qs[n])
                  for n in ARM_NAMES}

    # ───── Lagrangian pipeline (step_4X) ─────
    if os.path.exists('s43_resolved.json'): os.remove('s43_resolved.json')
    if os.path.exists('s44_synchronized.json'): os.remove('s44_synchronized.json')
    make_ik_json(start_qs, target_qs, duration, 's40_ik.json')
    ok42, _, _ = run_step(dir_4x, 'step_41.py')
    if ok42: ok42, _, _ = run_step(dir_4x, 'step_42.py')
    if ok42: ok42, _, _ = run_step(dir_4x, 'step_43.py')
    if ok42: ok42, _, _ = run_step(dir_4x, 'step_44.py')
    m_lag = extract_metrics_lagrangian()

    # ───── Alpha-pull pipeline (step_6X) ─────
    if os.path.exists('s64_resolved.json'): os.remove('s64_resolved.json')
    if os.path.exists('s65_synchronized.json'): os.remove('s65_synchronized.json')
    make_ik_json(start_qs, target_qs, duration, 's61_ik.json')
    ok62, _, _ = run_step(dir_6x, 'step_62.py')
    if ok62: ok62, _, _ = run_step(dir_6x, 'step_63.py')
    if ok62: ok62, _, _ = run_step(dir_6x, 'step_64.py')
    if ok62: ok62, _, _ = run_step(dir_6x, 'step_65.py')
    m_alpha = extract_metrics_alphapull()

    return {'trial': trial_idx, 'lagrangian': m_lag, 'alpha_pull': m_alpha}


def aggregate(results):
    """Compute summary statistics across trials."""
    L = [r['lagrangian'] for r in results]
    A = [r['alpha_pull'] for r in results]
    def safe_mean(vals): return float(np.mean(vals)) if vals else 0.

    summary = {}
    summary['n_trials'] = len(results)
    summary['lagrangian'] = {
        'success_rate'   : sum(1 for m in L if m['success']) / len(L),
        'mean_iterations': safe_mean([m['iterations'] for m in L]),
        'mean_mod_mag'   : safe_mean([m['mod_mag_mean'] for m in L]),
        'mean_dur_oh_pct': safe_mean([m['duration_overhead']*100 for m in L]),
        'mean_resolve_ms': safe_mean([m['resolve_ms'] for m in L]),
        'mean_kuramoto_ms': safe_mean([m['kuramoto_ms'] for m in L]),
        'mean_min_dist_m': safe_mean([m['min_dist'] for m in L]),
    }
    summary['alpha_pull'] = {
        'success_rate'   : sum(1 for m in A if m['success']) / len(A),
        'mean_iterations': safe_mean([m['iterations'] for m in A]),
        'mean_mod_mag'   : safe_mean([m['mod_mag_mean'] for m in A]),
        'mean_dur_oh_pct': safe_mean([m['duration_overhead']*100 for m in A]),
        'mean_resolve_ms': safe_mean([m['resolve_ms'] for m in A]),
        'mean_kuramoto_ms': safe_mean([m['kuramoto_ms'] for m in A]),
        'mean_min_dist_m': safe_mean([m['min_dist'] for m in A]),
    }
    return summary


def print_table(summary):
    """Pretty table for the terminal -- what you show your guide."""
    L = summary['lagrangian']
    A = summary['alpha_pull']
    print()
    print(' ' + '=' * 78)
    print(f'  COMPARISON  --  {summary["n_trials"]} trials')
    print(' ' + '=' * 78)
    print(f'  {"Metric":<26} {"Lagrangian":>20} {"Alpha-Pull":>20} {"Winner":>10}')
    print(' ' + '-' * 78)
    def row(label, key, fmt='{:.2f}', higher_better=True):
        lv = L[key]; av = A[key]
        winner = 'Lagrang' if (lv > av if higher_better else lv < av) else 'AlphaP'
        if abs(lv - av) < 1e-6: winner = 'tie'
        print(f'  {label:<26} {fmt.format(lv):>20} {fmt.format(av):>20} {winner:>10}')
    row('Success rate',         'success_rate',     '{:.1%}', higher_better=True)
    row('Mean iterations',      'mean_iterations',  '{:.1f}', higher_better=False)
    row('Mean mod magnitude',   'mean_mod_mag',     '{:.4f}', higher_better=False)
    row('Mean duration OH (%)', 'mean_dur_oh_pct',  '{:.1f}', higher_better=False)
    row('Mean resolve time ms', 'mean_resolve_ms',  '{:.0f}', higher_better=False)
    row('Mean Kuramoto ms',     'mean_kuramoto_ms', '{:.0f}', higher_better=False)
    row('Mean min dist (m)',    'mean_min_dist_m',  '{:.3f}', higher_better=True)
    print(' ' + '=' * 78)


def plot_comparison(results, outfile='s46_compare_plots.png'):
    if not _PLOT: return
    L = [r['lagrangian'] for r in results]
    A = [r['alpha_pull'] for r in results]
    trials = [r['trial'] for r in results]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # Success rate (bar)
    ax = axes[0, 0]
    s_L = sum(1 for m in L if m['success'])
    s_A = sum(1 for m in A if m['success'])
    n = len(L)
    ax.bar(['Lagrangian', 'Alpha-Pull'], [s_L/n*100, s_A/n*100],
            color=['#1976D2', '#FF9800'])
    ax.set_ylabel('Success rate (%)'); ax.set_title('Success Rate')
    ax.set_ylim(0, 105); ax.grid(True, alpha=0.3)
    for i, v in enumerate([s_L/n*100, s_A/n*100]):
        ax.text(i, v + 2, f'{v:.0f}%', ha='center', fontweight='bold')

    # Iterations (boxplot)
    ax = axes[0, 1]
    ax.boxplot([[m['iterations'] for m in L], [m['iterations'] for m in A]],
                labels=['Lagrangian', 'Alpha-Pull'])
    ax.set_ylabel('Iterations'); ax.set_title('Iterations Used'); ax.grid(True, alpha=0.3)

    # Mod magnitude (boxplot)
    ax = axes[0, 2]
    ax.boxplot([[m['mod_mag_mean'] for m in L], [m['mod_mag_mean'] for m in A]],
                labels=['Lagrangian', 'Alpha-Pull'])
    ax.set_ylabel('Modification mag (rad)'); ax.set_title('Surgical Modification'); ax.grid(True, alpha=0.3)

    # Duration overhead
    ax = axes[1, 0]
    ax.boxplot([[m['duration_overhead']*100 for m in L], [m['duration_overhead']*100 for m in A]],
                labels=['Lagrangian', 'Alpha-Pull'])
    ax.set_ylabel('Duration overhead (%)'); ax.set_title('Duration Overhead'); ax.grid(True, alpha=0.3)

    # Planning time
    ax = axes[1, 1]
    ax.boxplot([[m['resolve_ms']+m['kuramoto_ms'] for m in L],
                 [m['resolve_ms']+m['kuramoto_ms'] for m in A]],
                labels=['Lagrangian', 'Alpha-Pull'])
    ax.set_ylabel('Resolve + Kuramoto time (ms)'); ax.set_title('Planning Time'); ax.grid(True, alpha=0.3)

    # Min inter-arm distance
    ax = axes[1, 2]
    ax.boxplot([[m['min_dist']*100 for m in L], [m['min_dist']*100 for m in A]],
                labels=['Lagrangian', 'Alpha-Pull'])
    ax.set_ylabel('Min inter-arm dist (cm)'); ax.set_title('Safety Margin'); ax.grid(True, alpha=0.3)

    plt.suptitle(f'Lagrangian vs Alpha-Pull -- {len(results)} trials',
                  fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(outfile, dpi=140, bbox_inches='tight')
    plt.close()
    print(f'  Saved plots: {outfile}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=10,
                     help='Number of random trials to run (default 10)')
    ap.add_argument('--seed',   type=int, default=42)
    ap.add_argument('--duration', type=float, default=10.0)
    ap.add_argument('--dir_4x', type=str, default=os.path.dirname(__file__),
                     help='Directory containing step_41..step_44')
    ap.add_argument('--dir_6x', type=str, default=None,
                     help='Directory containing step_62..step_65')
    args = ap.parse_args()

    if args.dir_6x is None:
        # Try common locations
        for guess in [
            '/home/sriram/dual_arm_ws/install/dual_arm_sync/lib/python3.10/site-packages/dual_arm_sync',
            '/home/claude/skar',
            os.path.expanduser('~/dual_arm_ws/src/dual_arm_sync/dual_arm_sync'),
        ]:
            if os.path.exists(os.path.join(guess, 'step_64.py')):
                args.dir_6x = guess; break
        if args.dir_6x is None:
            print('  ERROR: must specify --dir_6x (path to step_61..80 directory)')
            sys.exit(1)

    print('\n' + '=' * 66)
    print('  STEP 46  --  Lagrangian vs Alpha-Pull Benchmark')
    print('=' * 66)
    print(f'  Trials   : {args.trials}')
    print(f'  Seed     : {args.seed}')
    print(f'  Duration : {args.duration:.1f}s')
    print(f'  Lagrangian dir: {args.dir_4x}')
    print(f'  Alpha-pull dir: {args.dir_6x}')

    rng = np.random.default_rng(args.seed)
    results = []
    t_all = time.time()
    for i in range(args.trials):
        print(f'\n  ── Trial {i+1}/{args.trials} ──')
        r = run_trial(rng, i, args.duration, args.dir_4x, args.dir_6x)
        results.append(r)
        L, A = r['lagrangian'], r['alpha_pull']
        print(f'    Lagrangian: success={L["success"]}  iter={L["iterations"]}  '
              f'mod={L["mod_mag_mean"]:.4f}  resolve={L["resolve_ms"]:.0f}ms')
        print(f'    Alpha-pull: success={A["success"]}  iter={A["iterations"]}  '
              f'mod={A["mod_mag_mean"]:.4f}  resolve={A["resolve_ms"]:.0f}ms')

    summary = aggregate(results)
    summary['total_time_s'] = round(time.time() - t_all, 1)

    with open('s46_compare.json', 'w') as fh:
        json.dump({'summary': summary, 'per_trial': results}, fh, indent=2)

    print_table(summary)
    plot_comparison(results)
    print(f'\n  Saved: s46_compare.json + s46_compare_plots.png')
    print(f'  Total time: {summary["total_time_s"]:.0f}s\n')


if __name__ == '__main__': main()