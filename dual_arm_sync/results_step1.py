#!/usr/bin/env python3
"""
results_step1.py  —  Dual-Arm Pipeline Trial Runner

Uses a persistent gazebo_executor daemon so the ROS2 node (and its publishers)
stay alive across all trials. This prevents the arm from freezing between trials
because there is no publisher teardown/restart.

Communication with daemon:
  Trial runner writes  exec_trigger.json  → daemon picks it up and executes
  Daemon writes        exec_done.json     → trial runner reads result

Per-trial flow:
    IK → patch_start → Trajectory → Collision?
         ├── Clear    → [trigger daemon] → save_pos → next
         ├── Kuramoto → [trigger daemon] → save_pos → next
         └── RRT      → [trigger daemon] → save_pos → next
                         (all fail) → next  [no Gazebo]

Usage:
    python3 results_step1.py --trials 50
    python3 results_step1.py --trials 50 --seed 42
    python3 results_step1.py --trials 50 --no-exec   # planning only
    python3 results_step1.py --trials 50 --start 11  # resume
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

# ─────────────────────────────────────────────────────────────────────────────
# WORKSPACE LIMITS
# ─────────────────────────────────────────────────────────────────────────────

X_MIN, X_MAX = -0.6,  0.6
Y_MIN, Y_MAX = -0.5,  0.5
Z_MIN, Z_MAX =  0.4,  1.1
MIN_SEP      = 0.25

# ─────────────────────────────────────────────────────────────────────────────
# TIMEOUTS (seconds)
# ─────────────────────────────────────────────────────────────────────────────

T_IK         = 40
T_TRAJ       = 45
T_COLL       = 60
T_KURAMOTO   = 120
T_RRT        = 300
T_EXEC_TRAJ  = 15    # time for the 10s trajectory itself
T_EXEC_TOTAL = 30    # max total wait for daemon to signal done (traj + overhead)
T_DAEMON_READY = 30  # max wait for daemon to start up

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

PKG          = 'dual_arm_sync'
STATE_FILE   = 'current_joints_state.json'
TRIGGER_FILE = 'exec_trigger.json'
DONE_FILE    = 'exec_done.json'

PIPELINE_FILES = [
    'ik_solutions.json',
    'trajectories.json',
    'collision_report.json',
    'synchronized_trajectories.json',
    'rrt_trajectories.json',
    # STATE_FILE, TRIGGER_FILE, DONE_FILE intentionally NOT here
]

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR HELPERS
# ─────────────────────────────────────────────────────────────────────────────

G  = '\033[92m'; R = '\033[91m'; Y = '\033[93m'
C  = '\033[96m'; B = '\033[1m';  D = '\033[2m'; RS = '\033[0m'

def col(t, c):  return f"{c}{t}{RS}"
def ok(t):      return col(f"✓ {t}", G)
def fail(t):    return col(f"✗ {t}", R)
def warn(t):    return col(f"⚠ {t}", Y)
def info(t):    return col(f"  {t}", C)

DIV  = '═' * 80
SEP  = '─' * 80
SSEP = '  ' + '─'*22 + ' ' + '─'*12 + '  ' + '─'*7 + '  ' + '─'*30

STATUS_COL = {
    'ok': G, 'safe': G, 'clear': G,
    'failed': R, 'error': R,
    'collision': Y, 'unsafe': Y,
    'skipped': D,
}

def _status(s):
    return col(f"[{s.upper()[:10]:^10}]", STATUS_COL.get(s, C))

# ─────────────────────────────────────────────────────────────────────────────
# DAEMON MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

_daemon_proc = None
_daemon_log  = None

def start_daemon():
    """
    Launch gazebo_executor --daemon in the background.
    It stays alive for the entire run, keeping publishers warm.
    Returns True once the daemon signals it is ready.
    """
    global _daemon_proc, _daemon_log

    # Clean up stale trigger/done files
    for f in (TRIGGER_FILE, DONE_FILE):
        try: Path(f).unlink(missing_ok=True)
        except Exception: pass

    log_path = Path('/tmp/gazebo_daemon.log')
    log_path.unlink(missing_ok=True)

    print(info("Starting gazebo_executor daemon..."))
    _daemon_log  = open(log_path, 'w')
    _daemon_proc = subprocess.Popen(
        ['ros2', 'run', PKG, 'gazebo_executor', '--daemon'],
        stdout=_daemon_log, stderr=_daemon_log
    )

    # Wait until daemon prints "Waiting for next trigger" = fully ready
    deadline = time.time() + T_DAEMON_READY
    while time.time() < deadline:
        time.sleep(0.3)
        if _daemon_proc.poll() is not None:
            print(fail("Daemon exited unexpectedly"))
            return False
        try:
            content = log_path.read_text(errors='replace')
            if 'Waiting for next trigger' in content or 'Waiting for trigger' in content:
                print(ok("Gazebo executor daemon is ready"))
                return True
            if 'Timeout waiting for joint states' in content:
                print(fail("Daemon: timeout waiting for joint states — is Gazebo running?"))
                return False
        except Exception:
            pass

    print(warn("Daemon startup timeout — proceeding anyway"))
    return True


def stop_daemon():
    """Shut down the daemon cleanly at end of run."""
    global _daemon_proc, _daemon_log
    if _daemon_proc is not None:
        try:
            _daemon_proc.terminate()
            _daemon_proc.wait(timeout=5)
        except Exception:
            try: _daemon_proc.kill()
            except Exception: pass
        _daemon_proc = None
    if _daemon_log is not None:
        try: _daemon_log.close()
        except Exception: pass
        _daemon_log = None


def daemon_alive():
    """Check if daemon process is still running."""
    return _daemon_proc is not None and _daemon_proc.poll() is None


def trigger_execution(trial_num):
    """
    Write trigger file → daemon executes trajectory → wait for done file.
    Returns (success, elapsed_s, source_label).
    """
    # Make sure there's no stale done file
    try: Path(DONE_FILE).unlink(missing_ok=True)
    except Exception: pass

    # Write trigger
    json.dump({'trial': trial_num}, open(TRIGGER_FILE, 'w'))

    t0 = time.time()
    deadline = t0 + T_EXEC_TOTAL

    while time.time() < deadline:
        time.sleep(0.2)

        # Check daemon is still alive
        if not daemon_alive():
            return False, round(time.time() - t0, 3), '?'

        # Check for done file
        if Path(DONE_FILE).exists():
            elapsed = round(time.time() - t0, 3)
            try:
                result = json.load(open(DONE_FILE))
                # Clean up done file
                Path(DONE_FILE).unlink(missing_ok=True)
                return result.get('success', False), elapsed, result.get('source', '?')
            except Exception:
                return False, elapsed, '?'

    # Timeout — clean up trigger if daemon didn't pick it up
    try: Path(TRIGGER_FILE).unlink(missing_ok=True)
    except Exception: pass
    return False, round(time.time() - t0, 3), '?'

# ─────────────────────────────────────────────────────────────────────────────
# ROS2 NODE RUNNER  (for planning nodes — not executor)
# ─────────────────────────────────────────────────────────────────────────────

def run_node(node, stdin=None, timeout=60):
    """Run a ROS2 node via log file (not capture_output). Returns (success, log, elapsed)."""
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
        return r.returncode == 0, out, elapsed
    except subprocess.TimeoutExpired:
        return False, '', round(time.time() - t0, 3)
    except Exception:
        return False, '', round(time.time() - t0, 3)

# ─────────────────────────────────────────────────────────────────────────────
# STATE MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def save_joint_state():
    """
    Save robot's final joint configuration after successful execution.
    ik_solutions.json stores optimal_joints in RADIANS — save as-is, no conversion.
    """
    try:
        d = json.load(open('ik_solutions.json'))
        # optimal_joints is in RADIANS (raw numpy from scipy IK solver)
        j1 = d['dsr01']['optimal_joints']
        j2 = d['dsr02']['optimal_joints']
        json.dump({'dsr01': j1, 'dsr02': j2}, open(STATE_FILE, 'w'), indent=2)
        d1 = [round(math.degrees(v), 2) for v in j1]
        d2 = [round(math.degrees(v), 2) for v in j2]
        print(info(f"  State saved — DSR01: {d1} deg"))
        print(info(f"  State saved — DSR02: {d2} deg"))
    except Exception as e:
        print(warn(f"  save_joint_state failed: {e}"))


def patch_ik_current_joints():
    """
    Fix the trajectory start point.

    BUG in IK solver: always writes current_joints = zeros (home) into ik_solutions.json.
    FIX: overwrite with the real last position from STATE_FILE before trajectory_generation runs.
    STATE_FILE stores joints in RADIANS — write directly, no conversion needed.

    First trial: STATE_FILE doesn't exist → do nothing (zeros = home is correct).
    """
    if not Path(STATE_FILE).exists():
        print(info("  No state file — starting from home (trial 1 correct)"))
        return
    try:
        state = json.load(open(STATE_FILE))
        ik    = json.load(open('ik_solutions.json'))

        j1 = state['dsr01']   # radians
        j2 = state['dsr02']   # radians

        ik['dsr01']['current_joints']     = j1
        ik['dsr01']['current_joints_deg'] = [math.degrees(v) for v in j1]
        ik['dsr02']['current_joints']     = j2
        ik['dsr02']['current_joints_deg'] = [math.degrees(v) for v in j2]

        opt1 = ik['dsr01']['optimal_joints']
        opt2 = ik['dsr02']['optimal_joints']
        ik['dsr01']['displacement']     = [o-c for o,c in zip(opt1, j1)]
        ik['dsr01']['displacement_deg'] = [math.degrees(o-c) for o,c in zip(opt1, j1)]
        ik['dsr02']['displacement']     = [o-c for o,c in zip(opt2, j2)]
        ik['dsr02']['displacement_deg'] = [math.degrees(o-c) for o,c in zip(opt2, j2)]

        json.dump(ik, open('ik_solutions.json', 'w'), indent=2)

        d1 = [round(math.degrees(v), 2) for v in j1]
        d2 = [round(math.degrees(v), 2) for v in j2]
        print(info(f"  Traj start patched — DSR01 from: {d1} deg"))
        print(info(f"  Traj start patched — DSR02 from: {d2} deg"))

    except Exception as e:
        print(warn(f"  patch_ik_current_joints failed: {e}"))

# ─────────────────────────────────────────────────────────────────────────────
# JSON READERS
# ─────────────────────────────────────────────────────────────────────────────

def read_ik():
    try:
        d = json.load(open('ik_solutions.json'))
        return ('dsr01' in d and 'optimal_joints' in d['dsr01'] and
                'dsr02' in d and 'optimal_joints' in d['dsr02'])
    except Exception:
        return False


def read_collision():
    try:
        d   = json.load(open('collision_report.json'))
        sim = d.get('simultaneous_motion', {})
        det = {
            'pts':    sim.get('num_collision_points', 0),
            'min_cm': round((sim.get('min_distance') or 0) * 100, 1),
        }
        return True, bool(d.get('collision_detected', True)), det
    except Exception:
        return False, True, {}


def read_kuramoto():
    try:
        d   = json.load(open('synchronized_trajectories.json'))
        ver = d.get('post_sync_verification', {})
        det = {
            'violations': ver.get('num_collision_points', 0),
            'min_cm':     round(ver.get('min_distance', 0) * 100, 1),
        }
        return True, not ver.get('has_collision', True), det
    except Exception:
        return False, False, {}


def read_rrt():
    try:
        d    = json.load(open('rrt_trajectories.json'))
        meta = d.get('metadata', {})
        chk  = meta.get('simultaneous_check', 'UNKNOWN')
        det  = {'check': chk, 'min_cm': meta.get('min_distance_cm', 0),
                'attempts': meta.get('attempts_used', 0)}
        return True, d.get('success', False) and chk == 'PASSED', det
    except Exception:
        return False, False, {}

# ─────────────────────────────────────────────────────────────────────────────
# CLEANUP
# ─────────────────────────────────────────────────────────────────────────────

def cleanup():
    for f in PIPELINE_FILES:
        try: Path(f).unlink(missing_ok=True)
        except Exception: pass

# ─────────────────────────────────────────────────────────────────────────────
# TARGET GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

SCENARIOS = ('corners', 'center', 'cross', 'parallel', 'wide')

def generate_targets(rng):
    for _ in range(500):
        sc = rng.choice(SCENARIOS)

        if sc == 'corners':
            xs = rng.uniform(-0.4, 0.4)
            p1 = [xs + rng.uniform(-0.08, 0.08), rng.uniform( 0.05,  0.45), rng.uniform(Z_MIN, Z_MAX)]
            p2 = [xs + rng.uniform(-0.08, 0.08), rng.uniform(-0.45, -0.05), rng.uniform(Z_MIN, Z_MAX)]
        elif sc == 'center':
            p1 = [rng.uniform(-0.25, 0.25), rng.uniform( 0.00,  0.35), rng.uniform(Z_MIN, 0.9)]
            p2 = [rng.uniform(-0.25, 0.25), rng.uniform(-0.35,  0.00), rng.uniform(Z_MIN, 0.9)]
        elif sc == 'cross':
            p1 = [rng.uniform( 0.05,  0.55), rng.uniform( 0.00,  0.45), rng.uniform(Z_MIN, Z_MAX)]
            p2 = [rng.uniform(-0.55, -0.05), rng.uniform(-0.45,  0.00), rng.uniform(Z_MIN, Z_MAX)]
        elif sc == 'parallel':
            xc = rng.uniform(-0.35, 0.35)
            p1 = [xc + rng.uniform(-0.04, 0.04), rng.uniform( 0.10,  0.45), rng.uniform(Z_MIN, Z_MAX)]
            p2 = [xc + rng.uniform(-0.04, 0.04), rng.uniform(-0.45, -0.10), rng.uniform(Z_MIN, Z_MAX)]
        else:
            p1 = [rng.uniform(-0.60, -0.15), rng.uniform( 0.10,  0.45), rng.uniform(Z_MIN, Z_MAX)]
            p2 = [rng.uniform( 0.15,  0.60), rng.uniform(-0.45, -0.10), rng.uniform(Z_MIN, Z_MAX)]

        p1 = [round(float(v), 3) for v in p1]
        p2 = [round(float(v), 3) for v in p2]

        def valid(p):
            return X_MIN <= p[0] <= X_MAX and Y_MIN <= p[1] <= Y_MAX and Z_MIN <= p[2] <= Z_MAX

        sep = math.sqrt(sum((a-b)**2 for a,b in zip(p1, p2)))
        if valid(p1) and valid(p2) and sep >= MIN_SEP:
            return {'dsr01': {'x': p1[0], 'y': p1[1], 'z': p1[2]},
                    'dsr02': {'x': p2[0], 'y': p2[1], 'z': p2[2]},
                    'scenario': sc, 'sep_m': round(sep, 3)}

    return {'dsr01': {'x':  0.4, 'y':  0.3, 'z': 0.7},
            'dsr02': {'x': -0.4, 'y': -0.3, 'z': 0.7},
            'scenario': 'fallback', 'sep_m': 1.02}

# ─────────────────────────────────────────────────────────────────────────────
# PRINT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def print_trial_header(num, total, targets):
    d1, d2, sc = targets['dsr01'], targets['dsr02'], targets['scenario']
    print(f"\n{DIV}")
    print(col(f"  TRIAL {num}/{total}  │  {sc.upper()}  │  Sep {targets['sep_m']}m", B))
    print(f"  DSR01 target → x={d1['x']:+.3f}  y={d1['y']:+.3f}  z={d1['z']:.3f}")
    print(f"  DSR02 target → x={d2['x']:+.3f}  y={d2['y']:+.3f}  z={d2['z']:.3f}")
    print(DIV)


def print_step(label, status, t, detail=''):
    print(f"  {label:<22} {_status(status)}  {t:>6.2f}s  {detail}")


def print_trial_summary(trial):
    meth   = trial['method']
    mcol   = {'direct': G, 'kuramoto': G, 'rrt': G, 'failed': R}.get(meth, Y)
    mlabel = {
        'direct':   '✓ DIRECT B-SPLINE (no collision)',
        'kuramoto': '✓ RESOLVED by Kuramoto',
        'rrt':      '✓ RESOLVED by RRT-Connect',
        'failed':   '✗ FAILED — no safe path found',
    }.get(meth, meth)

    print(f"\n{SEP}")
    print(col(f"  RESULT  : {mlabel}", mcol))
    print(f"  Gazebo  : {col('YES ✓', G) if trial['executed'] else col('NO ✗', R)}")
    print(f"\n  {'Step':<22} {'Status':^12}  {'Time':>7}  Details")
    print(SSEP)

    LABELS = {'ik': 'IK Solver', 'traj': 'Trajectory Gen',
              'coll': 'Collision Check', 'kur': 'Kuramoto Sync',
              'rrt': 'RRT-Connect', 'exec': 'Gazebo Executor'}
    for key in ('ik', 'traj', 'coll', 'kur', 'rrt', 'exec'):
        s = trial['steps'].get(key)
        if s:
            print_step(LABELS[key], s['status'], s['time'], s.get('detail', ''))

    total_s_str = f"{trial['total_s']:.2f}s"
    print(f"\n  Total: {col(total_s_str, B)}")
    print(SEP)

# ─────────────────────────────────────────────────────────────────────────────
# TRIAL EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def close_trial(trial, t0, method):
    trial['method']  = method
    trial['total_s'] = round(time.time() - t0, 3)
    print_trial_summary(trial)
    return trial


def do_gazebo(trial, t0, method, no_exec):
    lbl = {'direct':   'Executing in Gazebo (direct B-spline)',
           'kuramoto': 'Executing in Gazebo (Kuramoto synchronized)',
           'rrt':      'Executing in Gazebo (RRT-Connect)'}[method]
    print(f"\n[E] {lbl}")

    if no_exec:
        print(info("[--no-exec] Skipping Gazebo execution"))
        trial['steps']['exec'] = {'status': 'skipped', 'time': 0.0}
        trial['executed'] = True
        return close_trial(trial, t0, method)

    success, elapsed, source = trigger_execution(trial['num'])

    trial['steps']['exec'] = {
        'status': 'ok' if success else 'failed',
        'time':   elapsed,
        'detail': f"src={source}",
    }

    if success:
        print(ok(f"Gazebo execution complete  [{elapsed:.2f}s]"))
        trial['executed'] = True
        save_joint_state()
    else:
        print(fail(f"Gazebo execution failed  [{elapsed:.2f}s]"))
        trial['executed'] = False

    return close_trial(trial, t0, method)


def run_trial(num, total, targets, no_exec):
    d1 = targets['dsr01']
    d2 = targets['dsr02']
    print_trial_header(num, total, targets)

    trial = {
        'num': num, 'scenario': targets['scenario'], 'sep_m': targets['sep_m'],
        'targets': targets, 'steps': {}, 'method': None,
        'executed': False, 'total_s': 0.0, 'error': None,
    }
    t0 = time.time()

    # ── 1. IK ────────────────────────────────────────────────────────────────
    print(f"\n[1] IK Solver")
    ik_in = (f"{d1['x']}\n{d1['y']}\n{d1['z']}\n"
             f"{d2['x']}\n{d2['y']}\n{d2['z']}\nn\n")
    ok_p, _, t = run_node('dual_arm_ik_solver', stdin=ik_in, timeout=T_IK)
    ik_ok = ok_p and read_ik()
    trial['steps']['ik'] = {'status': 'ok' if ik_ok else 'failed', 'time': t}

    if not ik_ok:
        print(fail(f"IK failed  [{t:.2f}s]"))
        trial['error'] = 'ik_failed'
        return close_trial(trial, t0, 'failed')

    print(ok(f"IK solved  [{t:.2f}s]"))
    patch_ik_current_joints()

    # ── 2. Trajectory ─────────────────────────────────────────────────────────
    print(f"\n[2] Trajectory Generation")
    ok_p, _, t = run_node('trajectory_generation', timeout=T_TRAJ)
    traj_ok = ok_p and Path('trajectories.json').exists()
    trial['steps']['traj'] = {'status': 'ok' if traj_ok else 'failed', 'time': t}

    if not traj_ok:
        print(fail(f"Trajectory generation failed  [{t:.2f}s]"))
        trial['error'] = 'trajectory_failed'
        return close_trial(trial, t0, 'failed')

    print(ok(f"Trajectories generated  [{t:.2f}s]"))

    # ── 3. Collision ──────────────────────────────────────────────────────────
    print(f"\n[3] Collision Checker")
    ok_p, _, t = run_node('collision_checker', timeout=T_COLL)
    c_ok, has_col, c_det = read_collision()

    if not ok_p or not c_ok:
        print(fail(f"Collision checker error  [{t:.2f}s]"))
        trial['steps']['coll'] = {'status': 'error', 'time': t}
        trial['error'] = 'collision_checker_error'
        return close_trial(trial, t0, 'failed')

    det_str = f"pts={c_det['pts']}  min={c_det['min_cm']}cm"
    trial['steps']['coll'] = {'status': 'collision' if has_col else 'clear',
                               'time': t, 'detail': det_str}

    if not has_col:
        print(ok(f"No collision — clear  [{t:.2f}s]"))
        return do_gazebo(trial, t0, 'direct', no_exec)

    print(warn(f"Collision detected — {det_str}  [{t:.2f}s]"))

    # ── 4. Kuramoto ───────────────────────────────────────────────────────────
    print(f"\n[4] Kuramoto Synchronization")
    ok_p, _, t = run_node('kuramoto_synchronization', timeout=T_KURAMOTO)
    k_ok, k_safe, k_det = read_kuramoto()
    k_status = 'safe' if (k_ok and k_safe) else ('unsafe' if k_ok else 'failed')
    k_detail = f"viol={k_det.get('violations','?')}  min={k_det.get('min_cm','?')}cm"
    trial['steps']['kur'] = {'status': k_status, 'time': t, 'detail': k_detail}

    if k_ok and k_safe:
        print(ok(f"Kuramoto resolved — {k_detail}  [{t:.2f}s]"))
        return do_gazebo(trial, t0, 'kuramoto', no_exec)

    print(warn(f"Kuramoto insufficient — {k_detail}  [{t:.2f}s]"))

    # ── 5. RRT ────────────────────────────────────────────────────────────────
    print(f"\n[5] RRT-Connect Planner")
    ok_p, _, t = run_node('rrt_connect_planner', timeout=T_RRT)
    r_ok, r_safe, r_det = read_rrt()
    r_status = 'safe' if (r_ok and r_safe) else ('unsafe' if r_ok else 'failed')
    r_detail = f"chk={r_det.get('check','?')}  min={r_det.get('min_cm','?')}cm"
    trial['steps']['rrt'] = {'status': r_status, 'time': t, 'detail': r_detail}

    if r_ok and r_safe:
        print(ok(f"RRT safe path — {r_detail}  [{t:.2f}s]"))
        return do_gazebo(trial, t0, 'rrt', no_exec)

    print(fail(f"RRT failed — {r_detail}  [{t:.2f}s]"))
    trial['error'] = 'no_safe_path'
    return close_trial(trial, t0, 'failed')

# ─────────────────────────────────────────────────────────────────────────────
# FINAL REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_report(results, elapsed, outdir):
    n        = len(results)
    direct   = sum(1 for r in results if r['method'] == 'direct')
    kur      = sum(1 for r in results if r['method'] == 'kuramoto')
    rrt      = sum(1 for r in results if r['method'] == 'rrt')
    failed   = sum(1 for r in results if r['method'] == 'failed')
    exec_ok  = sum(1 for r in results if r['executed'])
    col_tot  = n - direct
    resolved = kur + rrt

    def step_times(key):
        return [r['steps'][key]['time'] for r in results
                if key in r['steps']
                and r['steps'][key]['status'] not in ('skipped',)
                and r['steps'][key]['time'] > 0]

    def fmt_times(ts):
        if not ts: return 'N/A'
        a = np.array(ts)
        return f"avg={a.mean():.2f}s  min={a.min():.2f}s  max={a.max():.2f}s  (n={len(a)})"

    sc_cnt = {}
    for r in results:
        sc = r.get('scenario', 'unknown')
        if sc not in sc_cnt:
            sc_cnt[sc] = dict(n=0, direct=0, kur=0, rrt=0, failed=0, exec=0)
        sc_cnt[sc]['n'] += 1
        m = r['method']
        if m == 'direct':    sc_cnt[sc]['direct'] += 1
        elif m == 'kuramoto': sc_cnt[sc]['kur']   += 1
        elif m == 'rrt':     sc_cnt[sc]['rrt']    += 1
        elif m == 'failed':  sc_cnt[sc]['failed'] += 1
        if r['executed']:    sc_cnt[sc]['exec']   += 1

    lines = []
    def p(line=''):
        print(line); lines.append(line)

    p(f"\n{DIV}")
    p(col("  FINAL RESULTS REPORT", B))
    p(f"  Date     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    p(f"  Trials   : {n}")
    p(f"  Duration : {elapsed/60:.1f} min  ({elapsed:.0f}s)")
    p(DIV)

    p(f"\n{SEP}"); p("  RESOLUTION METHOD"); p(SEP)
    p(f"  {'Direct B-spline (no collision):':<44} {direct:4d}  ({direct/n*100:5.1f}%)")
    p(f"  {'Kuramoto resolved:':<44} {kur:4d}  ({kur/n*100:5.1f}%)")
    p(f"  {'RRT-Connect resolved:':<44} {rrt:4d}  ({rrt/n*100:5.1f}%)")
    p(f"  {'Failed (no safe path):':<44} {failed:4d}  ({failed/n*100:5.1f}%)")
    p(f"  {'─'*44}")
    p(f"  {'Executed in Gazebo:':<44} {exec_ok:4d}  ({exec_ok/n*100:5.1f}%)")

    p(f"\n{SEP}"); p("  COLLISION RESOLUTION"); p(SEP)
    p(f"  Trials with collision: {col_tot:4d}  ({col_tot/n*100:5.1f}%)")
    if col_tot > 0:
        p(f"  Resolved (Kuramoto+RRT): {resolved:4d}  ({resolved/col_tot*100:5.1f}%)")
        p(f"  Kuramoto success       : {kur:4d}  ({kur/col_tot*100:5.1f}%)")
        kf = col_tot - kur
        if kf > 0:
            p(f"  RRT success (of Kuramoto fail): {rrt:4d}  ({rrt/kf*100:5.1f}%)")

    p(f"\n{SEP}"); p("  AVERAGE STEP TIMING"); p(SEP)
    for key, label in [('ik', 'IK Solver          '), ('traj', 'Trajectory Gen     '),
                        ('coll', 'Collision Check    '), ('kur', 'Kuramoto Sync      '),
                        ('rrt', 'RRT-Connect        '), ('exec', 'Gazebo Executor    ')]:
        p(f"  {label}: {fmt_times(step_times(key))}")

    tt = [r['total_s'] for r in results]
    p(f"\n  Per-trial: avg={np.mean(tt):.2f}s  median={float(np.median(tt)):.2f}s  max={max(tt):.2f}s")

    p(f"\n{SEP}"); p("  BY SCENARIO"); p(SEP)
    p(f"  {'Scenario':<12} {'Total':>6} {'Direct':>8} {'Kuram':>7} {'RRT':>5} {'Failed':>7} {'Exec':>6}")
    p(f"  {'─'*12} {'─'*6} {'─'*8} {'─'*7} {'─'*5} {'─'*7} {'─'*6}")
    for sc, c in sorted(sc_cnt.items()):
        p(f"  {sc:<12} {c['n']:>6} {c['direct']:>8} {c['kur']:>7} {c['rrt']:>5} {c['failed']:>7} {c['exec']:>6}")

    fl = [r for r in results if r['method'] == 'failed']
    if fl:
        p(f"\n{SEP}"); p(f"  FAILED TRIALS  ({len(fl)})"); p(SEP)
        for r in fl:
            tg = r['targets']
            p(f"  #{r['num']:3d} | {r['scenario']:10s} | "
              f"DSR01({tg['dsr01']['x']:+.2f},{tg['dsr01']['y']:+.2f},{tg['dsr01']['z']:.2f})"
              f"  DSR02({tg['dsr02']['x']:+.2f},{tg['dsr02']['y']:+.2f},{tg['dsr02']['z']:.2f})"
              f"  err={r.get('error','?')}")

    ef = [r for r in results if r['method'] != 'failed' and not r['executed']]
    if ef:
        p(f"\n{SEP}"); p(f"  GAZEBO EXECUTION FAILED  ({len(ef)})"); p(SEP)
        for r in ef:
            p(f"  #{r['num']:3d} | {r['method']:8s} | err={r.get('error','?')}")

    p(f"\n{DIV}\n")

    json_path = outdir / 'results.json'
    txt_path  = outdir / 'report.txt'
    with open(json_path, 'w') as f:
        json.dump({'summary': {'total': n, 'total_time_s': round(elapsed, 2),
                               'direct': direct, 'kuramoto': kur, 'rrt': rrt,
                               'failed': failed, 'executed': exec_ok,
                               'collision_trials': col_tot, 'collision_resolved': resolved},
                   'trials': results}, f, indent=2, default=str)
    with open(txt_path, 'w') as f:
        f.write('\n'.join(re.sub(r'\033\[[0-9;]*m', '', ln) for ln in lines))

    print(ok(f"Results → {json_path}"))
    print(ok(f"Report  → {txt_path}"))

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Dual-Arm Pipeline Trial Runner')
    ap.add_argument('--trials',  type=int,  default=100)
    ap.add_argument('--seed',    type=int,  default=None)
    ap.add_argument('--start',   type=int,  default=1)
    ap.add_argument('--no-exec', action='store_true', help='Skip Gazebo (planning only)')
    args = ap.parse_args()

    ts     = datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir = Path(f"trial_results_{ts}")
    outdir.mkdir(exist_ok=True)
    rng    = np.random.default_rng(args.seed)

    print(f"\n{DIV}")
    print(col("  DUAL-ARM PIPELINE TRIAL RUNNER", B))
    print(DIV)
    print(f"  Trials    : {args.trials}")
    print(f"  Seed      : {args.seed if args.seed else 'random'}")
    print(f"  Output    : {outdir}")
    print(f"  Gazebo    : {'DISABLED (--no-exec)' if args.no_exec else 'ENABLED (daemon mode)'}")
    print(f"\n  Constraints: x∈[{X_MIN},{X_MAX}]  y∈[{Y_MIN},{Y_MAX}]"
          f"  z∈[{Z_MIN},{Z_MAX}]  sep≥{MIN_SEP}m")
    print(f"\n  Flow: IK → patch_start → Traj → Collision → Gazebo daemon → save_pos → next")

    if Path(STATE_FILE).exists():
        try:
            s = json.load(open(STATE_FILE))
            d1 = [round(math.degrees(v), 2) for v in s['dsr01']]
            d2 = [round(math.degrees(v), 2) for v in s['dsr02']]
            print(f"\n  Resuming from saved state:")
            print(f"    DSR01 (deg): {d1}")
            print(f"    DSR02 (deg): {d2}")
        except Exception:
            pass
    else:
        print(f"\n  No saved state — trial 1 starts from home")

    print(DIV + "\n")

    # ── Start daemon (unless planning-only) ───────────────────────────────────
    if not args.no_exec:
        if not start_daemon():
            print(fail("Could not start Gazebo executor daemon. Aborting."))
            print("  Make sure Gazebo is running and joint states are available.")
            return

    all_results = []
    interrupted = False

    def _sigint(sig, frame):
        nonlocal interrupted
        interrupted = True
        print(f"\n\n{warn('Interrupted — saving results...')}")

    signal.signal(signal.SIGINT, _sigint)
    total_start = time.time()

    try:
        for i in range(args.start, args.trials + 1):
            if interrupted:
                break
            # Check daemon still alive
            if not args.no_exec and not daemon_alive():
                print(fail(f"\nDaemon died unexpectedly at trial {i}. Stopping."))
                break
            cleanup()
            targets = generate_targets(rng)
            result  = run_trial(i, args.trials, targets, args.no_exec)
            all_results.append(result)
            with open(outdir / 'results_partial.json', 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
    finally:
        stop_daemon()

    total_elapsed = time.time() - total_start
    if all_results:
        print_report(all_results, total_elapsed, outdir)
    else:
        print("No trials completed.")


if __name__ == '__main__':
    main()