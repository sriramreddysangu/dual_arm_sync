#!/usr/bin/env python3
"""
quad_arm_ik_solver.py
IK Solver for 4-arm system (dsr01–dsr04).

Input:  interactive prompts (x, y, z per arm)
Output: ik_solutions.json  (same schema as dual_arm version)

Usage:
    ros2 run dual_arm_sync quad_arm_ik_solver
"""

import numpy as np
from typing import Dict, Optional
import json
import sys

try:
    from dual_arm_sync.ik_solver import (
        forward_kinematics, solve_ik_numerical,
        select_optimal_solution,
    )
    try:
        from dual_arm_sync.quad_arm_config import (
            ARM_NAMES, BASE_POSITIONS, N_ARMS,
            JOINT_POSITION_LIMITS,
        )
    except ImportError:
        from quad_arm_config import (
            ARM_NAMES, BASE_POSITIONS, N_ARMS,
            JOINT_POSITION_LIMITS,
        )
except ImportError as e:
    print(f"ERROR: Cannot import required modules - {e}")
    sys.exit(1)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _solve_one(robot_name: str,
               target_world: np.ndarray,
               current_joints: np.ndarray) -> Optional[Dict]:
    """Solve IK for one arm, return result dict or None."""
    base = BASE_POSITIONS[robot_name]
    target_local = target_world - base

    print(f"\n{'='*70}")
    print(f"  SOLVING IK: {robot_name.upper()}")
    print(f"{'='*70}")
    print(f"  Current joints (deg): {np.round(np.degrees(current_joints), 2)}")

    fk_local, _, _ = forward_kinematics(current_joints)
    print(f"  Current EE (world):   {np.round(fk_local + base, 4)}")
    print(f"  Target    (world):    {np.round(target_world, 4)}")
    print(f"  Target    (local):    {np.round(target_local, 4)}")

    solutions = solve_ik_numerical(target_local, None, current_joints)

    if not solutions:
        print(f"  ✗ No IK solutions found for {robot_name}")
        return None

    result = select_optimal_solution(solutions, current_joints, verbose=False)
    if result is None:
        return None

    optimal_joints, info = result
    fk_local_achieved, _, _ = forward_kinematics(optimal_joints)
    fk_world = fk_local_achieved + base
    pos_err = np.linalg.norm(fk_world - target_world)

    print(f"  Found {len(solutions)} solutions → cost={info['cost']:.4f}")
    print(f"  Optimal joints (deg): {np.round(np.degrees(optimal_joints), 2)}")
    print(f"  Achieved (world):     {np.round(fk_world, 4)}")
    print(f"  Position error:       {pos_err*1000:.3f} mm")

    return {
        'robot_name':          robot_name,
        'current_joints':      current_joints.tolist(),
        'current_joints_deg':  np.degrees(current_joints).tolist(),
        'optimal_joints':      optimal_joints.tolist(),
        'optimal_joints_deg':  np.degrees(optimal_joints).tolist(),
        'displacement':        (optimal_joints - current_joints).tolist(),
        'displacement_deg':    np.degrees(optimal_joints - current_joints).tolist(),
        'cost':                info['cost'],
        'num_solutions':       info['num_solutions'],
        'target_world':        target_world.tolist(),
        'achieved_world':      fk_world.tolist(),
        'pos_error_mm':        pos_err * 1000,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

class QuadArmIKSolver:
    def __init__(self):
        self.current_joints = {name: np.zeros(6) for name in ARM_NAMES}

    def _update_state(self):
        """Compute current EE positions for display."""
        print(f"\n{'='*70}")
        print("  CURRENT STATE")
        print(f"{'='*70}")
        for name in ARM_NAMES:
            j = self.current_joints[name]
            base = BASE_POSITIONS[name]
            fk_local, _, _ = forward_kinematics(j)
            world = fk_local + base
            print(f"  {name}: EE={np.round(world, 4)}  joints(deg)={np.round(np.degrees(j), 1)}")

    def run(self):
        print(f"\n{'='*70}")
        print(f"  QUAD-ARM IK SOLVER  ({N_ARMS} arms)")
        print(f"{'='*70}")
        print("  Enter target position (x, y, z) for each arm.")
        print("  Press Ctrl+C or type 'q' to quit.\n")

        self._update_state()

        while True:
            try:
                print(f"\n{'='*70}")
                print("  ENTER TARGET POSITIONS")
                print(f"{'='*70}")

                targets = {}
                for name in ARM_NAMES:
                    print(f"\n  {name}:")
                    raw = input("    x y z (space-separated, or 'skip'): ").strip()
                    if raw.lower() in ('q', 'quit', 'exit'):
                        return
                    if raw.lower() == 'skip':
                        # Keep current position
                        base = BASE_POSITIONS[name]
                        fk_local, _, _ = forward_kinematics(self.current_joints[name])
                        targets[name] = fk_local + base
                        print(f"    → Skipped, keeping current EE pos")
                        continue
                    parts = raw.split()
                    if len(parts) != 3:
                        print("    ✗ Need 3 values. Try again.")
                        return
                    targets[name] = np.array([float(p) for p in parts])

                # Solve all
                print(f"\n{'='*70}")
                print("  SOLVING IK FOR ALL ARMS")
                print(f"{'='*70}")

                solutions = {}
                failed = []

                for name in ARM_NAMES:
                    sol = _solve_one(name, targets[name], self.current_joints[name])
                    if sol is None:
                        failed.append(name)
                    else:
                        solutions[name] = sol

                if failed:
                    print(f"\n  ✗ IK FAILED for: {failed}")
                    cont = input("  Continue with remaining arms? (y/n): ").strip().lower()
                    if cont != 'y':
                        continue

                # Update internal state for solved arms
                for name, sol in solutions.items():
                    self.current_joints[name] = np.array(sol['optimal_joints'])

                # Save
                output = {name: solutions[name] for name in solutions}
                output['duration'] = 10.0
                output['n_arms'] = N_ARMS

                with open('ik_solutions.json', 'w') as f:
                    json.dump(output, f, indent=2)

                print(f"\n  ✓ Saved ik_solutions.json  ({len(solutions)}/{N_ARMS} arms solved)")

                if not failed:
                    print("\n  Next steps:")
                    print("    ros2 run dual_arm_sync quad_trajectory_generation")
                    print("    ros2 run dual_arm_sync quad_collision_checker")
                    print("    ros2 run dual_arm_sync quad_gazebo_executor")

                cont = input("\n  Solve another? (y/n): ").strip().lower()
                if cont != 'y':
                    break

            except KeyboardInterrupt:
                print("\n\n  Exiting.\n")
                break
            except ValueError as e:
                print(f"  Bad input: {e}")
            except Exception as e:
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()


def main():
    solver = QuadArmIKSolver()
    solver.run()


if __name__ == '__main__':
    main()