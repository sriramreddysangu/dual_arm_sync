#!/usr/bin/env python3
"""
quad_arm_test_pub.py
Test publisher for 4-arm Gazebo system.

Publishes joint commands reliably (repeats 5x per command so the
controller always receives it — unlike `ros2 topic pub --once`).

Usage:
    # Interactive mode
    ros2 run dual_arm_sync quad_arm_test_pub

    # Direct command — all arms
    ros2 run dual_arm_sync quad_arm_test_pub --all 0 0 0.5 0 0 0

    # Single arm
    ros2 run dual_arm_sync quad_arm_test_pub --arm dsr03 0 0.5 0.5 0 0 0

    # Home all arms
    ros2 run dual_arm_sync quad_arm_test_pub --home

    # Equivalent bash (no need for this script):
    timeout 5 ros2 topic pub /dsr01/gz/dsr_position_controller/commands \
        std_msgs/msg/Float64MultiArray "{data: [0.0,0.0,0.5,0.0,0.0,0.0]}"
"""

import sys
import time
import argparse
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

try:
    from dual_arm_sync.quad_arm_config import ARM_NAMES, N_ARMS
except ImportError:
    try:
        from quad_arm_config import ARM_NAMES, N_ARMS
    except ImportError:
        ARM_NAMES = ['dsr01', 'dsr02', 'dsr03', 'dsr04']
        N_ARMS    = 4

REPEAT_COUNT  = 5      # publish N times per command
REPEAT_DELAY  = 0.10   # seconds between repeats
WAIT_TIMEOUT  = 30.0   # seconds to wait for subscribers


class QuadArmTestPub(Node):

    def __init__(self):
        super().__init__('quad_arm_test_pub')
        self.pubs = {}
        for name in ARM_NAMES:
            topic = f'/{name}/gz/dsr_position_controller/commands'
            self.pubs[name] = self.create_publisher(Float64MultiArray, topic, 10)
        self.get_logger().info(
            f"Publishers ready for {N_ARMS} arms: {ARM_NAMES}"
        )

    def subscriber_count(self, name: str) -> int:
        return self.pubs[name].get_subscription_count()

    def wait_for_subscribers(self, names=None, timeout=WAIT_TIMEOUT) -> bool:
        if names is None:
            names = ARM_NAMES
        t0 = time.time()
        print(f"\nWaiting for subscribers ({len(names)} arms)...")
        while True:
            rclpy.spin_once(self, timeout_sec=0.3)
            missing = [n for n in names if self.subscriber_count(n) == 0]
            if not missing:
                print("✓ All subscribers ready\n")
                return True
            elapsed = time.time() - t0
            if elapsed > timeout:
                print(f"\n⚠ Timeout after {timeout:.0f}s. Missing: {missing}")
                ans = input("Continue anyway? (y/n): ").strip().lower()
                return ans == 'y'
            ready = len(names) - len(missing)
            print(f"  {ready}/{len(names)} ready  missing={missing}  [{elapsed:.0f}s]   ",
                  end='\r')

    def send(self, name: str, joints) -> bool:
        """Send joint command to one arm, repeated for reliability."""
        msg      = Float64MultiArray()
        msg.data = [float(j) for j in joints]
        subs     = self.subscriber_count(name)
        if subs == 0:
            print(f"  ⚠ {name}: no subscriber")
        for i in range(REPEAT_COUNT):
            self.pubs[name].publish(msg)
            if i < REPEAT_COUNT - 1:
                time.sleep(REPEAT_DELAY)
        print(f"  ✓ {name}: {[round(j,3) for j in joints]}")
        return True

    def send_all(self, joints):
        print(f"\nSending to all {N_ARMS} arms: {joints}")
        for name in ARM_NAMES:
            self.send(name, joints)
        print("Done.\n")

    def home_all(self):
        self.send_all([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def status(self):
        print(f"\n  {'Arm':<8}  {'Subs':>5}  Status")
        print(f"  {'─'*8}  {'─'*5}  {'─'*10}")
        for name in ARM_NAMES:
            c = self.subscriber_count(name)
            flag = "✓ ready" if c > 0 else "✗ waiting"
            print(f"  {name:<8}  {c:>5}  {flag}")
        print()

    def interactive(self):
        print(f"\n{'='*60}")
        print(f"  QUAD-ARM TEST PUBLISHER  ({N_ARMS} arms)")
        print("=" * 60)
        print("Commands:")
        print("  all  j1 j2 j3 j4 j5 j6   — send to all arms (rad)")
        print("  home                       — home all [0,0,0,0,0,0]")
        print("  <dsr0N>  j1..j6            — send to one arm")
        print("  status                     — show subscriber counts")
        print("  list                       — list arm names")
        print("  q / quit                   — exit")
        print("=" * 60)

        # Brief spin to discover existing subscribers
        print("\nSpinning 2s to discover existing subscribers...")
        t0 = time.time()
        while time.time() - t0 < 2.0:
            rclpy.spin_once(self, timeout_sec=0.1)
        self.status()

        while True:
            try:
                raw = input("> ").strip()
                if not raw:
                    continue
                parts = raw.split()
                cmd   = parts[0].lower()

                if cmd in ('q', 'quit', 'exit'):
                    break

                elif cmd == 'list':
                    print(f"  Arms: {ARM_NAMES}")

                elif cmd == 'status':
                    self.status()

                elif cmd == 'home':
                    if not self.wait_for_subscribers():
                        continue
                    self.home_all()

                elif cmd == 'all':
                    if len(parts) != 7:
                        print("  Usage: all j1 j2 j3 j4 j5 j6")
                        continue
                    joints = [float(p) for p in parts[1:]]
                    if not self.wait_for_subscribers():
                        continue
                    self.send_all(joints)

                elif cmd in [n.lower() for n in ARM_NAMES]:
                    name = next(n for n in ARM_NAMES if n.lower() == cmd)
                    if len(parts) != 7:
                        print(f"  Usage: {name} j1 j2 j3 j4 j5 j6")
                        continue
                    joints = [float(p) for p in parts[1:]]
                    if self.subscriber_count(name) == 0:
                        print(f"  ⚠ {name} has no subscriber yet")
                        ans = input("  Continue? (y/n): ").strip().lower()
                        if ans != 'y':
                            continue
                    self.send(name, joints)

                else:
                    print(f"  Unknown: '{cmd}'  (try 'list' or 'help')")

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except ValueError as e:
                print(f"  Bad value: {e}")
            except Exception as e:
                print(f"  Error: {e}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    ap = argparse.ArgumentParser(description=f'Quad-arm test publisher ({N_ARMS} arms)')
    ap.add_argument('--all',  nargs=6,  type=float, metavar='J',
                    help='Send 6 joint values (rad) to all arms')
    ap.add_argument('--arm',  type=str, help='Target arm name (e.g. dsr03)')
    ap.add_argument('--home', action='store_true', help='Home all arms')
    ap.add_argument('--no-wait', action='store_true',
                    help='Skip waiting for subscribers')
    known, _ = ap.parse_known_args()

    rclpy.init()
    node = QuadArmTestPub()

    try:
        if known.home:
            if not known.no_wait:
                node.wait_for_subscribers()
            node.home_all()

        elif known.all:
            names = [known.arm] if known.arm and known.arm in ARM_NAMES else ARM_NAMES
            if not known.no_wait:
                node.wait_for_subscribers(names)
            if known.arm and known.arm in ARM_NAMES:
                node.send(known.arm, known.all)
            else:
                node.send_all(known.all)

        else:
            node.interactive()

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()