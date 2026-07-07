#!/usr/bin/env python3
"""
gazebo_executor.py  —  Dual-Arm Gazebo Trajectory Executor

Modes:
  Normal (interactive):
    ros2 run dual_arm_sync gazebo_executor
    Waits for ENTER, executes once, exits.

  Auto (single-shot, used by old scripts):
    ros2 run dual_arm_sync gazebo_executor --auto
    Executes immediately, exits when done.

  Daemon (persistent, used by trial runner):
    ros2 run dual_arm_sync gazebo_executor --daemon
    Stays alive forever. Watches for 'exec_trigger.json'.
    When trigger file appears → load trajectory → execute →
    write 'exec_done.json' → delete trigger.

Trigger file format  (exec_trigger.json):
    {"trial": 5}

Done file format  (exec_done.json):
    {"success": true, "trial": 5, "source": "kuramoto", "elapsed": 10.34}

Topics published:
  /dsr01/gz/dsr_position_controller/commands  [Float64MultiArray]
  /dsr02/gz/dsr_position_controller/commands  [Float64MultiArray]

JSON formats understood
───────────────────────
Our pipeline writes two trajectory files.  Both store trajectories under a
top-level "trajectories" key, with each arm's data as a list of
{"time": t, "joints": [j1..j6]} dicts under "trajectory_points".

  trajectories.json          (from trajectory_generation)
  ─────────────────
  {
    "arm_ids": ["dsr01","dsr02"],
    "duration": 10.0,
    "trajectories": {
      "dsr01": {
        "trajectory_points": [{"time": 0.0, "joints": [...]}, ...],
        ...
      },
      "dsr02": { ... }
    }
  }

  synchronized_trajectories.json   (from kuramoto_sync)
  ───────────────────────────────
  {
    "arm_ids": ["dsr01","dsr02"],
    "collision_free": true,
    "trajectories": {
      "dsr01": {
        "trajectory_points": [{"time":0.0,"time_eff":0.0,"joints":[...]}, ...],
        ...
      },
      "dsr02": { ... }
    }
  }
"""

import sys
import os
import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# FILE NAMES
# ─────────────────────────────────────────────────────────────────────────────

FILE_DIRECT   = 'trajectories.json'
FILE_KURAMOTO = 'synchronized_trajectories.json'
FILE_RRT      = 'rrt_trajectories.json'

TRIGGER_FILE  = 'exec_trigger.json'
DONE_FILE     = 'exec_done.json'


# ─────────────────────────────────────────────────────────────────────────────
# UNIVERSAL KEY EXTRACTORS
# ─────────────────────────────────────────────────────────────────────────────

def _get_positions(arm_dict: dict) -> np.ndarray:
    """
    Extract (T, 6) joint positions from one arm's dict.

    Handles both pipeline formats:
      1. trajectory_points list: [{"joints": [...], "time": t}, ...]  ← our format
      2. Legacy flat arrays:     {"positions": [[...], ...], ...}
    """
    # ── Primary: our pipeline format ──────────────────────────────────────────
    if 'trajectory_points' in arm_dict:
        pts = arm_dict['trajectory_points']
        joints = [p['joints'] for p in pts]
        return np.array(joints, dtype=float)

    # ── Legacy flat-array fallbacks ───────────────────────────────────────────
    for key in ('positions', 'joint_positions', 'waypoints', 'joints'):
        if key in arm_dict:
            return np.array(arm_dict[key], dtype=float)

    # ── Nested under 'trajectory' sub-dict ───────────────────────────────────
    if 'trajectory' in arm_dict:
        t = arm_dict['trajectory']
        if 'trajectory_points' in t:
            return np.array([p['joints'] for p in t['trajectory_points']], dtype=float)
        for key in ('positions', 'joint_positions', 'waypoints'):
            if key in t:
                return np.array(t[key], dtype=float)

    raise KeyError(
        f"Cannot extract joint positions from arm dict. "
        f"Top-level keys: {list(arm_dict.keys())}"
    )


def _get_time(arm_dict: dict) -> np.ndarray:
    """
    Extract (T,) time vector from one arm's dict.

    Handles both pipeline formats.
    Uses 'time' (wall-clock) from trajectory_points, not 'time_eff'
    (effective time after Kuramoto phase shift).
    """
    # ── Primary: our pipeline format ──────────────────────────────────────────
    if 'trajectory_points' in arm_dict:
        pts = arm_dict['trajectory_points']
        t_arr = np.array([p['time'] for p in pts], dtype=float)
        # If all times are equal (e.g. Kuramoto output where wall-time
        # is very short), fall back to time_eff for playback pacing
        if t_arr[-1] - t_arr[0] < 1e-6 and 'time_eff' in pts[0]:
            t_arr = np.array([p['time_eff'] for p in pts], dtype=float)
        # Ensure monotonically increasing from 0
        if t_arr[-1] <= t_arr[0]:
            n = len(t_arr)
            duration = arm_dict.get('duration', 10.0)
            t_arr = np.linspace(0.0, float(duration), n)
        return t_arr

    # ── Legacy flat-array fallbacks ───────────────────────────────────────────
    for key in ('time', 'timestamps', 'time_vector', 't', 'times'):
        if key in arm_dict:
            return np.array(arm_dict[key], dtype=float)
        if 'trajectory' in arm_dict and key in arm_dict['trajectory']:
            return np.array(arm_dict['trajectory'][key], dtype=float)

    raise KeyError(
        f"Cannot extract time vector from arm dict. "
        f"Top-level keys: {list(arm_dict.keys())}"
    )


def _arm_data(data: dict, arm_id: str) -> dict:
    """
    Locate per-arm dict in both flat and nested JSON layouts.

    Our pipeline:   data['trajectories']['dsr01']
    Legacy:         data['dsr01']
    """
    if 'trajectories' in data and arm_id in data['trajectories']:
        return data['trajectories'][arm_id]
    if arm_id in data:
        return data[arm_id]
    raise KeyError(
        f"Arm '{arm_id}' not found. "
        f"Top-level keys: {list(data.keys())}, "
        f"trajectories keys: {list(data.get('trajectories', {}).keys())}"
    )


def _arm_ids(data: dict) -> list:
    """Return arm IDs present in the file."""
    if 'arm_ids' in data:
        return list(data['arm_ids'])
    if 'trajectories' in data:
        return list(data['trajectories'].keys())
    # Legacy: any key that looks like an arm name
    return [k for k in data if k.startswith('dsr') or k.startswith('arm')]


# ─────────────────────────────────────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def load_kuramoto(filepath: str):
    """
    Load synchronized_trajectories.json produced by kuramoto_sync.run().

    Structure:
        data['collision_free']           ← is_safe
        data['trajectories']['dsr0x']    ← per-arm dict
          └ trajectory_points[i]['time']   ← wall-clock time
          └ trajectory_points[i]['joints'] ← 6 joint angles (rad)
    """
    with open(filepath) as f:
        data = json.load(f)

    arm_ids = _arm_ids(data)
    if len(arm_ids) < 2:
        raise ValueError(f"Expected ≥ 2 arms in {filepath}, got {arm_ids}")

    a1, a2 = arm_ids[0], arm_ids[1]
    d1 = _arm_data(data, a1)
    d2 = _arm_data(data, a2)

    pos1 = _get_positions(d1)
    pos2 = _get_positions(d2)
    t1   = _get_time(d1)

    # Safety flag — our pipeline writes 'collision_free' directly
    is_safe = bool(data.get('collision_free',
                   not data.get('post_sync_verification', {}).get('has_collision', True)))

    n_iter   = data.get('refinement_iterations', 0)
    spread   = data.get('final_spread_rad', 0.0)
    convg    = data.get('converged', True)

    print(f"[LOADER] Kuramoto: arms={arm_ids}  steps={len(pos1)}  "
          f"collision_free={is_safe}  converged={convg}  "
          f"spread={spread:.5f}rad  refinement_iters={n_iter}")

    return (pos1, pos2, t1,
            {'source': 'kuramoto',
             'is_safe': is_safe,
             'arm_ids': arm_ids,
             'converged': convg,
             'spread_rad': spread,
             'refinement_iterations': n_iter})


def load_direct(filepath: str):
    """
    Load trajectories.json produced by trajectory_generation.run().

    Structure:
        data['trajectories']['dsr0x']    ← per-arm dict
          └ trajectory_points[i]['time']
          └ trajectory_points[i]['joints']
    """
    with open(filepath) as f:
        data = json.load(f)

    arm_ids = _arm_ids(data)
    if len(arm_ids) < 2:
        raise ValueError(f"Expected ≥ 2 arms in {filepath}, got {arm_ids}")

    a1, a2 = arm_ids[0], arm_ids[1]
    d1 = _arm_data(data, a1)
    d2 = _arm_data(data, a2)

    pos1 = _get_positions(d1)
    pos2 = _get_positions(d2)
    t1   = _get_time(d1)

    print(f"[LOADER] Direct B-spline: arms={arm_ids}  steps={len(pos1)}")
    return (pos1, pos2, t1,
            {'source': 'direct', 'is_safe': True, 'arm_ids': arm_ids})


def load_rrt(filepath: str):
    """
    Load rrt_trajectories.json (fallback / external planner).
    """
    with open(filepath) as f:
        data = json.load(f)

    arm_ids = _arm_ids(data)
    a1, a2  = arm_ids[0], arm_ids[1]
    d1 = _arm_data(data, a1)
    d2 = _arm_data(data, a2)

    pos1 = _get_positions(d1)
    pos2 = _get_positions(d2)
    t1   = _get_time(d1)

    meta    = data.get('metadata', {})
    is_safe = (data.get('success', False) and
               meta.get('simultaneous_check') == 'PASSED')

    print(f"[LOADER] RRT: arms={arm_ids}  steps={len(pos1)}  is_safe={is_safe}")
    return (pos1, pos2, t1,
            {'source': 'rrt', 'is_safe': is_safe,
             'simultaneous_check': meta.get('simultaneous_check', 'UNKNOWN')})


def select_trajectory(force_rrt: bool = False):
    """
    Priority:  Kuramoto (collision_free) → RRT (verified) → direct B-spline.
    If force_rrt=True, skip straight to RRT.
    """
    if force_rrt:
        if not os.path.exists(FILE_RRT):
            raise FileNotFoundError(f"--rrt flag set but '{FILE_RRT}' not found.")
        print(f"[LOADER] --rrt: loading {FILE_RRT}")
        return load_rrt(FILE_RRT)

    # ── Try Kuramoto output ──────────────────────────────────────────────────
    if os.path.exists(FILE_KURAMOTO):
        try:
            res = load_kuramoto(FILE_KURAMOTO)
            if res[3]['is_safe']:
                print("[LOADER] ✓ Using Kuramoto synchronized trajectories")
                return res
            print(f"[LOADER] Kuramoto trajectories not collision-free "
                  f"→ trying RRT")
        except Exception as e:
            print(f"[LOADER] Kuramoto load error: {e}  → trying RRT")

    # ── Try RRT ─────────────────────────────────────────────────────────────
    if os.path.exists(FILE_RRT):
        try:
            res = load_rrt(FILE_RRT)
            if res[3]['is_safe']:
                print("[LOADER] ✓ Using RRT-Connect trajectories")
                return res
            print(f"[LOADER] RRT trajectories not verified safe → trying direct")
        except Exception as e:
            print(f"[LOADER] RRT load error: {e}  → trying direct")

    # ── Fall back to direct B-spline ─────────────────────────────────────────
    if os.path.exists(FILE_DIRECT):
        res = load_direct(FILE_DIRECT)
        print("[LOADER] ✓ Using direct B-spline trajectories (unverified)")
        return res

    raise FileNotFoundError(
        "No trajectory file found. "
        f"Expected one of: {FILE_KURAMOTO}, {FILE_RRT}, {FILE_DIRECT}. "
        "Run the planning pipeline first."
    )


# ─────────────────────────────────────────────────────────────────────────────
# ROS2 NODE
# ─────────────────────────────────────────────────────────────────────────────

class GazeboExecutor(Node):

    def __init__(self):
        super().__init__('gazebo_trajectory_executor')

        self.pub1 = self.create_publisher(
            Float64MultiArray,
            '/dsr01/gz/dsr_position_controller/commands', 10)
        self.pub2 = self.create_publisher(
            Float64MultiArray,
            '/dsr02/gz/dsr_position_controller/commands', 10)

        self.create_subscription(
            JointState, '/dsr01/gz/joint_states', self._cb1, 10)
        self.create_subscription(
            JointState, '/dsr02/gz/joint_states', self._cb2, 10)

        self.got1 = False
        self.got2 = False

        self.traj1   = None
        self.traj2   = None
        self.tvec    = None
        self.meta    = {}
        self.index   = 0
        self.running = False
        self.timer   = None

    def _cb1(self, _):
        if not self.got1:
            self.got1 = True
            self.get_logger().info('✓ DSR01 joint states received')

    def _cb2(self, _):
        if not self.got2:
            self.got2 = True
            self.get_logger().info('✓ DSR02 joint states received')

    def load(self, force_rrt: bool = False) -> bool:
        """Load trajectory files. Returns True on success."""
        try:
            self.traj1, self.traj2, self.tvec, self.meta = \
                select_trajectory(force_rrt)

            # Validate shapes
            assert self.traj1.ndim == 2 and self.traj1.shape[1] == 6, \
                f"traj1 shape {self.traj1.shape} — expected (T,6)"
            assert self.traj2.ndim == 2 and self.traj2.shape[1] == 6, \
                f"traj2 shape {self.traj2.shape} — expected (T,6)"
            assert len(self.tvec) == len(self.traj1), \
                f"time vector length {len(self.tvec)} ≠ traj length {len(self.traj1)}"

            return True

        except Exception as e:
            self.get_logger().error(f"❌ Load failed: {e}")
            import traceback; traceback.print_exc()
            return False

    def start(self) -> bool:
        if self.traj1 is None:
            return False
        # Use wall-clock step: real time between trajectory samples
        t_arr = self.tvec
        dt = float(t_arr[1] - t_arr[0]) if len(t_arr) > 1 else 0.05
        dt = max(dt, 0.005)   # floor at 5 ms to avoid hammering
        dur = float(t_arr[-1] - t_arr[0])
        self.get_logger().info(
            f"\n{'='*60}\n"
            f"  STARTING EXECUTION\n"
            f"  Source    : {self.meta.get('source', '?')}\n"
            f"  Waypoints : {len(self.traj1)}\n"
            f"  Duration  : {dur:.2f} s\n"
            f"  Rate      : {1/dt:.1f} Hz\n"
            f"  is_safe   : {self.meta.get('is_safe', '?')}\n"
            f"{'='*60}"
        )
        self.index   = 0
        self.running = True
        self.timer   = self.create_timer(dt, self._step)
        return True

    def _step(self):
        if not self.running:
            return

        if self.index >= len(self.traj1):
            self.running = False
            self.timer.cancel()
            self.timer = None
            self.get_logger().info(
                f"\n{'='*60}\n"
                f"✓ TRAJECTORY EXECUTION COMPLETE\n"
                f"{'='*60}\n"
            )
            return

        msg1 = Float64MultiArray()
        msg2 = Float64MultiArray()
        msg1.data = [float(v) for v in self.traj1[self.index]]
        msg2.data = [float(v) for v in self.traj2[self.index]]
        self.pub1.publish(msg1)
        self.pub2.publish(msg2)

        if self.index % 50 == 0:
            pct = self.index / len(self.traj1) * 100
            t   = float(self.tvec[self.index])
            self.get_logger().info(
                f"  {pct:5.1f}%  step={self.index}/{len(self.traj1)}"
                f"  t={t:.2f}s")

        self.index += 1

    def is_done(self) -> bool:
        return (not self.running
                and self.timer is None
                and self.traj1 is not None)

    def _reset(self):
        """Reset execution state between daemon trials."""
        self.traj1   = None
        self.traj2   = None
        self.tvec    = None
        self.meta    = {}
        self.index   = 0
        self.running = False
        self.timer   = None


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main(args=None):
    force_rrt   = '--rrt'    in sys.argv
    auto_mode   = '--auto'   in sys.argv
    force_run   = '--force'  in sys.argv
    daemon_mode = '--daemon' in sys.argv
    for flag in ('--rrt', '--auto', '--force', '--daemon'):
        try:
            sys.argv.remove(flag)
        except ValueError:
            pass

    rclpy.init(args=args)
    node = GazeboExecutor()

    # ── DAEMON MODE ─────────────────────────────────────────────────────────
    if daemon_mode:
        print("[DAEMON] Gazebo executor daemon started")
        print(f"[DAEMON] Watching for: {TRIGGER_FILE}")
        print(f"[DAEMON] Will write:   {DONE_FILE}")

        print("[DAEMON] Waiting for joint states from both robots...")
        t0 = time.time()
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.got1 and node.got2:
                print("[DAEMON] ✓ Both robots ready. Waiting for trigger...")
                break
            if time.time() - t0 > 30.0:
                print("[DAEMON] ❌ Timeout waiting for joint states")
                node.destroy_node()
                rclpy.shutdown()
                return

        for f in (TRIGGER_FILE, DONE_FILE):
            try:
                os.unlink(f)
            except FileNotFoundError:
                pass

        while rclpy.ok():
            if os.path.exists(TRIGGER_FILE):
                try:
                    trigger   = json.load(open(TRIGGER_FILE))
                    trial_num = trigger.get('trial', 0)
                    print(f"[DAEMON] Trigger received for trial {trial_num}")
                except Exception:
                    trial_num = 0

                try:
                    os.unlink(TRIGGER_FILE)
                except Exception:
                    pass

                t_exec_start = time.time()
                if not node.load():
                    result = {'success': False, 'trial': trial_num,
                              'error': 'trajectory_load_failed', 'elapsed': 0.0}
                    json.dump(result, open(DONE_FILE, 'w'))
                    print("[DAEMON] ✗ Load failed")
                    continue

                source = node.meta.get('source', '?')
                print(f"[DAEMON] Loaded '{source}' — starting execution...")
                node.index   = 0
                node.running = False
                node.timer   = None
                node.start()

                while rclpy.ok() and not node.is_done():
                    rclpy.spin_once(node, timeout_sec=0.01)

                elapsed = round(time.time() - t_exec_start, 3)
                result = {'success': True, 'trial': trial_num,
                          'source': source, 'elapsed': elapsed}
                json.dump(result, open(DONE_FILE, 'w'))
                print(f"[DAEMON] ✓ Trial {trial_num} complete ({elapsed:.2f}s)")
                node._reset()
            else:
                rclpy.spin_once(node, timeout_sec=0.05)

        node.destroy_node()
        rclpy.shutdown()
        return

    # ── SINGLE-SHOT MODES ────────────────────────────────────────────────────

    if not node.load(force_rrt):
        node.destroy_node()
        rclpy.shutdown()
        return

    # Safety gate
    if not node.meta.get('is_safe', True) and not force_run:
        src = node.meta.get('source', '?')
        print(f"\n{'='*60}")
        print(f"❌ SAFETY GATE: '{src}' trajectory is NOT verified safe.")
        print(f"   Options:")
        print(f"   1. Run: ros2 run dual_arm_sync kuramoto_sync")
        print(f"      (Kuramoto will refine until collision_free=True)")
        print(f"   2. Override (use at own risk): add --force flag")
        print(f"{'='*60}")
        node.destroy_node()
        rclpy.shutdown()
        return

    # Wait for joint states
    print("\nWaiting for joint states from both robots...")
    t0 = time.time()
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.got1 and node.got2:
            print("✓ Both robots ready\n")
            break
        if time.time() - t0 > 15.0:
            print("❌ Timeout waiting for joint states")
            node.destroy_node()
            rclpy.shutdown()
            return

    print(f"\n{'='*60}")
    print(f"  Source    : {node.meta.get('source', '?')}")
    print(f"  Arm IDs   : {node.meta.get('arm_ids', ['dsr01','dsr02'])}")
    print(f"  Waypoints : {len(node.traj1)}")
    print(f"  Duration  : {node.tvec[-1]:.2f} s")
    print(f"  is_safe   : {node.meta.get('is_safe', True)}")
    print(f"{'='*60}")

    if auto_mode:
        print("[AUTO] Starting execution immediately...")
    else:
        try:
            input("\nPress ENTER to start (Ctrl+C to cancel): ")
        except KeyboardInterrupt:
            print("\nCancelled.")
            node.destroy_node()
            rclpy.shutdown()
            return

    if node.start():
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            print("\nExecution interrupted.")

    try:
        node.destroy_node()
    except Exception:
        pass
    try:
        if rclpy.ok():
            rclpy.shutdown()
    except Exception:
        pass


if __name__ == '__main__':
    main()