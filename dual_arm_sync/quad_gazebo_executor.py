#!/usr/bin/env python3
"""
quad_gazebo_executor.py
Gazebo trajectory executor for 4-arm system.

Auto-selects best available trajectory file:
  1. synchronized_trajectories.json  (Kuramoto — preferred)
  2. trajectories.json               (direct B-spline)

Flags:
  --auto    Skip ENTER prompt (for trial runner)
  --force   Override safety gate

Topics published:
  /dsr01/gz/dsr_position_controller/commands  [Float64MultiArray]
  /dsr02/gz/dsr_position_controller/commands  [Float64MultiArray]
  /dsr03/gz/dsr_position_controller/commands  [Float64MultiArray]
  /dsr04/gz/dsr_position_controller/commands  [Float64MultiArray]

Usage:
    ros2 run dual_arm_sync quad_gazebo_executor
    ros2 run dual_arm_sync quad_gazebo_executor --auto
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

try:
    from dual_arm_sync.quad_arm_config import ARM_NAMES, N_ARMS
except ImportError:
    try:
        from quad_arm_config import ARM_NAMES, N_ARMS
    except ImportError:
        # Fallback hardcoded for 4 arms
        ARM_NAMES = ['dsr01', 'dsr02', 'dsr03', 'dsr04']
        N_ARMS    = 4

FILE_KURAMOTO = 'synchronized_trajectories.json'
FILE_DIRECT   = 'trajectories.json'


# ── Trajectory loader ─────────────────────────────────────────────────────────

def _get_positions(robot: dict) -> np.ndarray:
    if 'trajectory' in robot:
        for k in ('positions', 'joint_positions', 'waypoints'):
            if k in robot['trajectory']:
                return np.array(robot['trajectory'][k])
    for k in ('positions', 'joint_positions', 'waypoints'):
        if k in robot:
            return np.array(robot[k])
    raise KeyError(f"Positions not found. Keys: {list(robot.keys())}")


def _get_time(robot: dict) -> np.ndarray:
    for k in ('time', 'timestamps', 'time_vector', 't'):
        if k in robot:
            return np.array(robot[k])
        if 'trajectory' in robot and k in robot['trajectory']:
            return np.array(robot['trajectory'][k])
    raise KeyError(f"Time not found. Keys: {list(robot.keys())}")


def load_trajectories(force_direct: bool = False):
    """
    Load best available trajectory. Returns (trajs_dict, time_vec, meta).
    trajs_dict: {arm_name: np.ndarray of shape (N_steps, 6)}
    """
    candidates = []
    if not force_direct and os.path.exists(FILE_KURAMOTO):
        candidates.append((FILE_KURAMOTO, 'kuramoto'))
    if os.path.exists(FILE_DIRECT):
        candidates.append((FILE_DIRECT, 'direct'))

    for filepath, label in candidates:
        try:
            with open(filepath) as f:
                data = json.load(f)

            # Safety check for kuramoto
            if label == 'kuramoto':
                ver = data.get('post_sync_verification', {})
                if ver.get('has_collision', True):
                    print(f"[LOADER] {filepath}: unsafe (has_collision=True) → skipping")
                    continue

            trajs = {}
            for name in ARM_NAMES:
                if name not in data:
                    print(f"[LOADER] ⚠ {name} missing from {filepath}")
                    continue
                trajs[name] = _get_positions(data[name])

            if not trajs:
                continue

            # Use first arm's time vector
            first_name = list(trajs.keys())[0]
            time_vec   = _get_time(data[first_name])

            # Ensure all trajs same length
            min_len = min(len(v) for v in trajs.values())
            for n in trajs:
                trajs[n] = trajs[n][:min_len]
            time_vec = time_vec[:min_len]

            meta = {
                'source':   label,
                'filepath': filepath,
                'is_safe':  True,
                'n_arms':   len(trajs),
            }
            print(f"[LOADER] Using {filepath} ({label})"
                  f"  arms={len(trajs)}  steps={min_len}"
                  f"  duration={float(time_vec[-1]):.2f}s")
            return trajs, time_vec, meta

        except Exception as e:
            print(f"[LOADER] {filepath}: {e}")
            continue

    raise FileNotFoundError(
        "No trajectory file found. Run:\n"
        "  ros2 run dual_arm_sync quad_trajectory_generation\n"
        "  ros2 run dual_arm_sync quad_collision_checker\n"
        "  ros2 run dual_arm_sync quad_kuramoto_synchronization"
    )


# ── ROS2 Node ─────────────────────────────────────────────────────────────────

class QuadGazeboExecutor(Node):

    def __init__(self, force_direct: bool = False):
        super().__init__('quad_gazebo_executor')

        # Publishers — one per arm
        self.pubs = {}
        for name in ARM_NAMES:
            topic = f'/{name}/gz/dsr_position_controller/commands'
            self.pubs[name] = self.create_publisher(Float64MultiArray, topic, 10)

        # Joint-state subscriptions (just to detect readiness)
        self.received = {name: False for name in ARM_NAMES}
        for name in ARM_NAMES:
            topic_js = f'/{name}/gz/joint_states'
            self.create_subscription(
                JointState, topic_js,
                lambda msg, n=name: self._cb(n), 10,
            )

        # Load trajectory
        self.trajs   = None
        self.tvec    = None
        self.meta    = {}
        self.index   = 0
        self.running = False
        self.timer   = None

        try:
            self.trajs, self.tvec, self.meta = load_trajectories(force_direct)
        except Exception as e:
            self.get_logger().error(f"\n✗ {e}")

    def _cb(self, name: str):
        if not self.received[name]:
            self.received[name] = True
            self.get_logger().info(f"✓ {name} joint states received")

    def all_ready(self) -> bool:
        return all(self.received[n] for n in ARM_NAMES if n in self.trajs)

    def start(self) -> bool:
        if self.trajs is None:
            return False
        dt  = float(self.tvec[1] - self.tvec[0])
        dur = float(self.tvec[-1])
        self.get_logger().info(
            f"\n{'='*60}\n"
            f"  STARTING EXECUTION\n"
            f"  Source   : {self.meta['source']}\n"
            f"  Arms     : {list(self.trajs.keys())}\n"
            f"  Steps    : {len(self.tvec)}\n"
            f"  Duration : {dur:.2f}s\n"
            f"  Rate     : {1/dt:.1f} Hz\n"
            f"{'='*60}"
        )
        self.index  = 0
        self.running = True
        self.timer  = self.create_timer(dt, self._step)
        return True

    def _step(self):
        if not self.running:
            return

        max_steps = min(len(v) for v in self.trajs.values())

        if self.index >= max_steps:
            self.running = False
            self.timer.cancel()
            self.get_logger().info(
                f"\n{'='*60}\n"
                f"✓ TRAJECTORY EXECUTION COMPLETE\n"
                f"{'='*60}\n"
            )
            return

        for name, traj in self.trajs.items():
            msg = Float64MultiArray()
            msg.data = traj[self.index].tolist()
            self.pubs[name].publish(msg)

        if self.index % 200 == 0:
            pct = self.index / max_steps * 100
            t   = float(self.tvec[self.index])
            self.get_logger().info(
                f"  {pct:.0f}%  t={t:.1f}s / {self.tvec[-1]:.1f}s"
            )

        self.index += 1


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    auto_mode    = '--auto'   in sys.argv
    force_direct = '--direct' in sys.argv
    force_run    = '--force'  in sys.argv

    for flag in ('--auto', '--direct', '--force'):
        try:
            sys.argv.remove(flag)
        except ValueError:
            pass

    rclpy.init(args=args)
    node = QuadGazeboExecutor(force_direct=force_direct)

    if node.trajs is None:
        node.destroy_node()
        rclpy.shutdown()
        return

    if not node.meta.get('is_safe', True) and not force_run:
        print(f"\n✗ SAFETY GATE: trajectory not verified safe.")
        print(f"  Add --force to override.")
        node.destroy_node()
        rclpy.shutdown()
        return

    # Wait for joint states
    if not auto_mode:
        print(f"\nWaiting for joint states from {N_ARMS} arms...")
        t0 = time.time()
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.all_ready():
                print("✓ All arms ready\n")
                break
            if time.time() - t0 > 20.0:
                print("⚠ Timeout — some arms may not be ready, continuing anyway")
                break
    else:
        for _ in range(20):
            rclpy.spin_once(node, timeout_sec=0.05)

    # Summary
    print(f"\n{'='*60}")
    print(f"  Source   : {node.meta['source']}")
    print(f"  Arms     : {list(node.trajs.keys())}")
    print(f"  Steps    : {len(node.tvec)}")
    print(f"  Duration : {node.tvec[-1]:.2f}s")
    print(f"{'='*60}")

    if auto_mode:
        print("[AUTO] Starting execution...")
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

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()