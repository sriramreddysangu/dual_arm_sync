#!/usr/bin/env python3
"""
quad_arm_results.py — 4-Arm Pipeline Trial Runner

Per-trial flow:
    IK -> patch_start -> Trajectory -> Collision?
    |-- Clear       -> Execute in Gazebo -> save_pos -> next
    |-- Kuramoto    -> (if safe) Execute  -> save_pos -> next
    `-- (if unsafe) -> skip Gazebo        -> next

KEY: Robot never returns home between trials.
  After each execution the final joint config is saved to
  current_joints_state_quad.json (radians, matching ik_solutions.json format).
  Before trajectory_generation runs, ik_solutions.json current_joints is
  overwritten with the saved joints so the trajectory starts from the real
  robot position, not zeros (home).

Usage:
    python3 quad_arm_results.py --trials 50
    python3 quad_arm_results.py --trials 100 --seed 42
    python3 quad_arm_results.py --trials 50  --no-exec
    python3 quad_arm_results.py --trials 100 --start 21   # resume
"""

import argparse
import json
import math
import os
import re
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# -- Config -------------------------------------------------------------------

try:
    from dual_arm_sync.quad_arm_config import ARM_NAMES, BASE_POSITIONS, N_ARMS
except ImportError:
    try:
        from quad_arm_config import ARM_NAMES, BASE_POSITIONS, N_ARMS
    except ImportError:
        ARM_NAMES = ['dsr01', 'dsr02', 'dsr03', 'dsr04']
        N_ARMS    = 4
        BASE_POSITIONS = {
            'dsr01': np.array([ 0.0,  0.5, 0.0]),
            'dsr02': np.array([ 0.0, -0.5, 0.0]),
            'dsr03': np.array([ 1.0,  0.5, 0.0]),
            'dsr04': np.array([ 1.0, -0.5, 0.0]),
        }

PKG        = 'dual_arm_sync'
STATE_FILE = 'current_joints_state_quad.json'   # separate from dual-arm state

X_MIN, X_MAX = -0.6,  1.6
Y_MIN, Y_MAX = -0.5,  0.5
Z_MIN, Z_MAX =  0.4,  1.1
MIN_SEP      = 0.30

T_IK       = 50
T_TRAJ     = 45
T_COLL     = 90
T_KURAMOTO = 150
T_EXECUTOR = 20

PIPELINE_FILES = [
    'ik_solutions.json',
    'trajectories.json',
    'collision_report.json',
    'synchronized_trajectories.json',
]

# -- Colors -------------------------------------------------------------------

G, R, Y, C, B, D, RS = (
    '\033[92m', '\033[91m', '\033[93m',
    '\033[96m', '\033[1m', '\033[2m', '\033[0m'
)

def col(t, c): return f"{c}{t}{RS}"
def ok(t):     return col(f"OK {t}", G)
def fail(t):   return col(f"FAIL {t}", R)
def warn(t):   return col(f"WARN {t}", Y)
def info(t):   return col(f"  {t}", C)

DIV = '=' * 76
SEP = '-' * 76

# -- ROS2 helpers -------------------------------------------------------------

def run_node(node, stdin=None, timeout=60):
    cmd      = ['ros2', 'run', PKG, node]
    log_path = Path(f'/tmp/ros2_{node}.log')
    log_path.unlink(missing_ok=True)
    t0 = time.time()
    try:
        with open(log_path, 'w') as lf:
            r = subprocess.run(cmd, input=stdin, stdout=lf, stderr=lf,
                               text=True, timeout=timeout)
        elapsed = round(time.time() - t0, 3)
        out = log_path.read_text(errors='replace') if log_path.exists() else ''
        return r.returncode == 0, out, '', elapsed
    except subprocess.TimeoutExpired:
        return False, '', f'TIMEOUT after {timeout}s', round(time.time() - t0, 3)
    except Exception as e:
        return False, '', str(e), round(time.time() - t0, 3)


def execute_trajectory():
    log  = Path('/tmp/quad_executor.log')
    log.unlink(missing_ok=True)
    cmd  = ['ros2', 'run', PKG, 'quad_gazebo_executor', '--auto']
    t0   = time.time()
    lf   = open(log, 'w')
    proc = subprocess.Popen(cmd, stdout=lf, stderr=lf)
    try:
        deadline = t0 + T_EXECUTOR
        while time.time() < deadline:
            time.sleep(0.5)
            ret = proc.poll()
            if ret is not None:
                lf.flush()
                content = log.read_text(errors='replace')
                return 'TRAJECTORY EXECUTION COMPLETE' in content, round(time.time()-t0, 3)
            lf.flush()
    finally:
        lf.close()
    try:
        proc.terminate(); proc.wait(timeout=3)
    except Exception:
        pass
    content = log.read_text(errors='replace') if log.exists() else ''
    return 'TRAJECTORY EXECUTION COMPLETE' in content, round(time.time()-t0, 3)

# -- State management ---------------------------------------------------------

def save_joint_state():
    """
    Save each arm's final joint config to STATE_FILE after successful execution.

    IMPORTANT: ik_solutions.json stores optimal_joints in RADIANS (raw numpy
    from scipy IK solver). Save as-is with NO unit conversion.
    If you called math.radians() here it would double-convert and store
    near-zero values, making every trial think the arms are still at home.
    """
    try:
        d     = json.load(open('ik_solutions.json'))
        state = {}
        for name in ARM_NAMES:
            # optimal_joints = RADIANS already. Store directly.
            state[name] = d[name]['optimal_joints']
        json.dump(state, open(STATE_FILE, 'w'), indent=2)
        for name in ARM_NAMES:
            deg = [round(math.degrees(v), 2) for v in state[name]]
            print(info(f"State saved {name}: {deg} deg"))
    except Exception as e:
        print(warn(f"save_joint_state failed: {e}"))


def patch_ik_current_joints():
    """
    CRITICAL FIX: called after IK solver writes ik_solutions.json,
    before trajectory_generation reads it.

    THE PROBLEM:
      IK solver always writes current_joints = [0,0,0,0,0,0] (home).
      trajectory_generation reads current_joints as trajectory START.
      Result: every trajectory begins from home regardless of where arms are.

    THE FIX:
      Overwrite current_joints in ik_solutions.json with the joints from
      STATE_FILE (the real last position). trajectory_generation then builds
      the path from real_last_pos to new_target. No home detour.

    FIRST TRIAL: STATE_FILE does not exist -> do nothing (zeros = home correct).
    """
    if not Path(STATE_FILE).exists():
        print(info("No state file - arms start from home (trial 1 correct)"))
        return

    try:
        state = json.load(open(STATE_FILE))
        ik    = json.load(open('ik_solutions.json'))

        for name in ARM_NAMES:
            if name not in state:
                continue
            j   = state[name]            # radians, saved correctly
            opt = ik[name]['optimal_joints']   # radians

            ik[name]['current_joints']     = j
            ik[name]['current_joints_deg'] = [math.degrees(v) for v in j]
            ik[name]['displacement']       = [o - c for o, c in zip(opt, j)]
            ik[name]['displacement_deg']   = [math.degrees(o - c) for o, c in zip(opt, j)]

            deg = [round(math.degrees(v), 2) for v in j]
            print(info(f"Traj start patched {name} from: {deg} deg"))

        json.dump(ik, open('ik_solutions.json', 'w'), indent=2)

    except Exception as e:
        print(warn(f"patch_ik_current_joints failed: {e}"))

# -- JSON readers -------------------------------------------------------------

def read_ik():
    try:
        d = json.load(open('ik_solutions.json'))
        return all(n in d and 'optimal_joints' in d[n] for n in ARM_NAMES)
    except Exception:
        return False


def read_collision():
    try:
        d = json.load(open('collision_report.json'))
        return True, bool(d.get('collision_detected', True)), d.get('summary', {})
    except Exception:
        return False, True, {}


def read_kuramoto():
    try:
        d   = json.load(open('synchronized_trajectories.json'))
        ver = d.get('post_sync_verification', {})
        return True, not ver.get('has_collision', True), {
            'violations': ver.get('num_collision_points', 0),
            'min_cm':     round(ver.get('min_distance', 0) * 100, 2),
        }
    except Exception:
        return False, False, {}

# -- Cleanup ------------------------------------------------------------------

def cleanup():
    for f in PIPELINE_FILES:
        Path(f).unlink(missing_ok=True)

# -- Target generator ---------------------------------------------------------

def generate_targets(rng):
    bases = [BASE_POSITIONS[n] for n in ARM_NAMES]
    for _ in range(500):
        targets = []
        valid   = True
        for i in range(N_ARMS):
            bx, by = bases[i][0], bases[i][1]
            x = float(np.clip(bx + rng.uniform(-0.5, 0.5), X_MIN, X_MAX))
            y = float(np.clip(by + rng.uniform(-0.3, 0.3), Y_MIN, Y_MAX))
            z = float(rng.uniform(Z_MIN, Z_MAX))
            targets.append([round(x, 3), round(y, 3), round(z, 3)])

        for i in range(len(targets)):
            for j in range(i+1, len(targets)):
                sep = math.sqrt(sum((a-b)**2 for a, b in zip(targets[i], targets[j])))
                if sep < MIN_SEP:
                    valid = False
                    break
            if not valid:
                break

        if valid:
            return {ARM_NAMES[i]: {'x': targets[i][0],
                                    'y': targets[i][1],
                                    'z': targets[i][2]}
                    for i in range(N_ARMS)}

    return {
        'dsr01': {'x':  0.3, 'y':  0.3, 'z': 0.7},
        'dsr02': {'x':  0.0, 'y': -0.3, 'z': 0.7},
        'dsr03': {'x':  1.3, 'y':  0.3, 'z': 0.7},
        'dsr04': {'x':  1.0, 'y': -0.3, 'z': 0.7},
    }

# -- Trial runner -------------------------------------------------------------

def close_trial(trial, t0, method):
    trial['method']  = method
    trial['total_s'] = round(time.time() - t0, 3)
    return trial


def do_gazebo(trial, t0, method, no_exec):
    if no_exec:
        trial['steps']['exec'] = {'status': 'skipped', 'time': 0.0}
        trial['executed'] = True
        return close_trial(trial, t0, method)

    success, elapsed = execute_trajectory()
    trial['steps']['exec'] = {
        'status': 'ok' if success else 'failed',
        'time':   elapsed,
    }
    trial['executed'] = success

    if success:
        print(ok(f"Gazebo execution complete [{elapsed:.2f}s]"))
        save_joint_state()   # save where arms are now, for next trial's patch
    else:
        print(fail(f"Execution failed [{elapsed:.2f}s]"))

    return close_trial(trial, t0, method)


def run_trial(num, total, targets, no_exec):
    print(f"\n{DIV}")
    print(col(f"  TRIAL {num}/{total}", B))
    for name in ARM_NAMES:
        t = targets[name]
        print(f"    {name}: x={t['x']:+.3f} y={t['y']:+.3f} z={t['z']:.3f}")
    print(DIV)

    trial = {
        'num': num, 'targets': targets,
        'steps': {}, 'method': None,
        'executed': False, 'total_s': 0.0,
    }
    t0 = time.time()

    # 1. IK
    print("\n[1] IK Solver")
    ik_lines = ''
    for name in ARM_NAMES:
        t = targets[name]
        ik_lines += f"{t['x']} {t['y']} {t['z']}\n"
    ik_lines += 'n\n'

    ok_p, _, _, elapsed = run_node('quad_arm_ik_solver', stdin=ik_lines, timeout=T_IK)
    ik_ok = ok_p and read_ik()
    trial['steps']['ik'] = {'status': 'ok' if ik_ok else 'failed', 'time': elapsed}

    if not ik_ok:
        print(fail(f"IK failed [{elapsed:.2f}s]"))
        return close_trial(trial, t0, 'failed')
    print(ok(f"IK solved [{elapsed:.2f}s]"))

    # PATCH: fix current_joints so trajectory starts from real arm position
    patch_ik_current_joints()

    # 2. Trajectory
    print("\n[2] Trajectory Generation")
    ok_p, _, _, elapsed = run_node('quad_trajectory_generation', timeout=T_TRAJ)
    traj_ok = ok_p and Path('trajectories.json').exists()
    trial['steps']['traj'] = {'status': 'ok' if traj_ok else 'failed', 'time': elapsed}

    if not traj_ok:
        print(fail(f"Trajectory generation failed [{elapsed:.2f}s]"))
        return close_trial(trial, t0, 'failed')
    print(ok(f"Trajectories generated [{elapsed:.2f}s]"))

    # 3. Collision check
    print("\n[3] Collision Checker")
    ok_p, _, _, elapsed = run_node('quad_collision_checker', timeout=T_COLL)
    c_ok, has_col, c_det = read_collision()

    if not ok_p or not c_ok:
        print(fail(f"Collision checker error [{elapsed:.2f}s]"))
        trial['steps']['coll'] = {'status': 'error', 'time': elapsed}
        return close_trial(trial, t0, 'failed')

    det_str = f"pairs_col={c_det.get('pairs_with_collision','?')} min={c_det.get('min_clearance_cm','?')}cm"
    trial['steps']['coll'] = {
        'status': 'collision' if has_col else 'clear',
        'time': elapsed, 'detail': det_str,
    }

    if not has_col:
        print(ok(f"No collision - clear [{elapsed:.2f}s]"))
        return do_gazebo(trial, t0, 'direct', no_exec)

    print(warn(f"Collision detected - {det_str} [{elapsed:.2f}s]"))

    # 4. Kuramoto
    print("\n[4] Kuramoto Synchronization")
    ok_p, _, _, elapsed = run_node('quad_kuramoto_synchronization', timeout=T_KURAMOTO)
    k_ok, k_safe, k_det = read_kuramoto()
    k_status = 'safe' if (k_ok and k_safe) else ('unsafe' if k_ok else 'failed')
    k_detail = f"viol={k_det.get('violations','?')} min={k_det.get('min_cm','?')}cm"
    trial['steps']['kur'] = {'status': k_status, 'time': elapsed, 'detail': k_detail}

    if k_ok and k_safe:
        print(ok(f"Kuramoto resolved - {k_detail} [{elapsed:.2f}s]"))
        return do_gazebo(trial, t0, 'kuramoto', no_exec)

    print(fail(f"Kuramoto insufficient - {k_detail} [{elapsed:.2f}s]"))
    return close_trial(trial, t0, 'failed')

# -- Report -------------------------------------------------------------------

def print_report(results, elapsed, outdir):
    n       = len(results)
    direct  = sum(1 for r in results if r['method'] == 'direct')
    kur     = sum(1 for r in results if r['method'] == 'kuramoto')
    failed  = sum(1 for r in results if r['method'] == 'failed')
    exec_ok = sum(1 for r in results if r['executed'])

    lines = []
    def p(line=''):
        print(line)
        lines.append(line)

    p(f"\n{DIV}")
    p(col("  FINAL REPORT - QUAD-ARM TRIAL RUNNER", B))
    p(f"  Date   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    p(f"  Trials : {n}   Duration: {elapsed/60:.1f}min")
    p(DIV)
    p(f"\n  Direct B-spline  : {direct:4d}  ({direct/n*100:5.1f}%)")
    p(f"  Kuramoto resolved: {kur:4d}  ({kur/n*100:5.1f}%)")
    p(f"  Failed           : {failed:4d}  ({failed/n*100:5.1f}%)")
    p(f"  Executed in Gazebo: {exec_ok:4d}  ({exec_ok/n*100:5.1f}%)")

    tt = [r['total_s'] for r in results]
    p(f"\n  Per-trial: avg={np.mean(tt):.2f}s  max={max(tt):.2f}s")
    p(f"\n{DIV}\n")

    json_path = outdir / 'results.json'
    txt_path  = outdir / 'report.txt'

    with open(json_path, 'w') as f:
        json.dump({'summary': {
            'total': n, 'direct': direct, 'kuramoto': kur,
            'failed': failed, 'executed': exec_ok,
        }, 'trials': results}, f, indent=2, default=str)

    with open(txt_path, 'w') as f:
        f.write('\n'.join(re.sub(r'\033\[[0-9;]*m', '', ln) for ln in lines))

    print(ok(f"Results -> {json_path}"))
    print(ok(f"Report  -> {txt_path}"))

# -- Main ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='4-Arm Pipeline Trial Runner')
    ap.add_argument('--trials',  type=int, default=50)
    ap.add_argument('--seed',    type=int, default=None)
    ap.add_argument('--start',   type=int, default=1)
    ap.add_argument('--no-exec', action='store_true')
    args = ap.parse_args()

    ts     = datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir = Path(f"trial_results_{ts}")
    outdir.mkdir(exist_ok=True)
    rng    = np.random.default_rng(args.seed)

    print(f"\n{DIV}")
    print(col("  QUAD-ARM PIPELINE TRIAL RUNNER", B))
    print(DIV)
    print(f"  Arms  : {N_ARMS}  {ARM_NAMES}")
    print(f"  Trials: {args.trials}  Seed: {args.seed or 'random'}")
    print(f"  Output: {outdir}")
    print(f"  Gazebo: {'DISABLED' if args.no_exec else 'ENABLED'}")

    if Path(STATE_FILE).exists():
        try:
            s = json.load(open(STATE_FILE))
            print(f"\n  Resuming from saved state:")
            for name in ARM_NAMES:
                if name in s:
                    deg = [round(math.degrees(v), 2) for v in s[name]]
                    print(f"    {name} (deg): {deg}")
        except Exception:
            pass
    else:
        print(f"\n  No saved state - all arms start from home")

    print(f"\n  Flow: IK -> patch_start -> Traj -> Collision -> Gazebo -> save_pos -> next")
    print(f"{DIV}\n")

    all_results = []
    interrupted = False

    def _sig(sig, frame):
        nonlocal interrupted
        interrupted = True
        print(f"\n{warn('Interrupted - saving...')}")
    signal.signal(signal.SIGINT, _sig)

    t_start = time.time()

    for i in range(args.start, args.trials + 1):
        if interrupted:
            break
        cleanup()
        targets = generate_targets(rng)
        result  = run_trial(i, args.trials, targets, args.no_exec)
        all_results.append(result)
        with open(outdir / 'results_partial.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

    if all_results:
        print_report(all_results, time.time() - t_start, outdir)
    else:
        print("No trials completed.")


if __name__ == '__main__':
    main()