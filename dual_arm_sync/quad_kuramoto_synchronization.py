#!/usr/bin/env python3
"""
quad_kuramoto_synchronization.py
N-oscillator Kuramoto synchronization for 4-arm system.

Each pair of arms has its own coupling strength based on their proximity.
Per-pair leadership: the arm that is "ahead" in phase is the leader.

Reads:  collision_report.json  (+ trajectories.json as fallback)
Writes: synchronized_trajectories.json

Usage:
    ros2 run dual_arm_sync quad_kuramoto_synchronization
"""

import numpy as np
from scipy.integrate import odeint
from typing import Dict, Tuple, List
import json
import os
import sys

try:
    from dual_arm_sync.constants import DHParameters
    try:
        from dual_arm_sync.quad_arm_config import (
            ARM_NAMES, BASE_POSITIONS, COLLISION_PAIRS,
            N_ARMS, MIN_SAFE_DISTANCE, LINK_RADII,
        )
    except ImportError:
        from quad_arm_config import (
            ARM_NAMES, BASE_POSITIONS, COLLISION_PAIRS,
            N_ARMS, MIN_SAFE_DISTANCE, LINK_RADII,
        )
except ImportError as e:
    print(f"ERROR: Cannot import required modules - {e}")
    sys.exit(1)


# ── Configuration ─────────────────────────────────────────────────────────────

class KConfig:
    BASE_COUPLING       = 2.0
    MAX_COUPLING        = 12.0
    REPULSION_THRESHOLD = 0.30   # 30cm — start repulsion
    REPULSION_STRENGTH  = 80.0
    EMERGENCY_BRAKE     = 200.0
    LEADERSHIP_THRESHOLD= 0.05
    ADAPTATION_RATE     = 0.5
    RECOVERY_RATE       = 0.1
    DT                  = 0.01
    MAX_PHASE_VELOCITY  = 2.0


# ── FK / distance ─────────────────────────────────────────────────────────────

def _link_positions(joints: np.ndarray, base: np.ndarray) -> np.ndarray:
    dh = DHParameters.get_dh_params(joints)
    T  = np.eye(4)
    pts = []
    for i in range(6):
        alpha, a, theta, d = dh[i]
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        T = T @ np.array([
            [ct, -st, 0, a],
            [st*ca, ct*ca, -sa, -sa*d],
            [st*sa, ct*sa,  ca,  ca*d],
            [0, 0, 0, 1],
        ])
        pts.append(T[:3, 3] + base)
    return np.array(pts)


def _pair_distance(joints_a: np.ndarray, base_a: np.ndarray,
                    joints_b: np.ndarray, base_b: np.ndarray) -> float:
    la = _link_positions(joints_a, base_a)
    lb = _link_positions(joints_b, base_b)
    return float(np.min([np.linalg.norm(la[i] - lb[j])
                          for i in range(6) for j in range(6)]))


def _interpolate(traj: np.ndarray, phase: float) -> np.ndarray:
    """Linearly interpolate trajectory at fractional phase ∈ [0,1]."""
    n = len(traj)
    phi = np.clip(phase, 0.0, 1.0)
    lo  = int(phi * (n - 1))
    hi  = min(lo + 1, n - 1)
    alpha = phi * (n - 1) - lo
    return (1 - alpha) * traj[lo] + alpha * traj[hi]


# ── N-oscillator dynamics ─────────────────────────────────────────────────────

def _derivatives(state: np.ndarray, t: float,
                  trajs: List[np.ndarray]) -> np.ndarray:
    """
    State: [phi_0, phi_1, ..., phi_{N-1},
            omega_0, omega_1, ..., omega_{N-1}]
    2*N values total.
    """
    n = len(trajs)
    phis   = state[:n]
    omegas = state[n:]

    dphi   = omegas.copy()
    domega = np.zeros(n)

    # Evaluate current joint positions for all arms
    joints = [_interpolate(trajs[i], phis[i]) for i in range(n)]
    bases  = [BASE_POSITIONS[ARM_NAMES[i]] for i in range(n)]

    # For each pair, compute coupling + repulsion
    for idx, (name_a, name_b) in enumerate(COLLISION_PAIRS):
        i = ARM_NAMES.index(name_a)
        j = ARM_NAMES.index(name_b)

        dist = _pair_distance(joints[i], bases[i], joints[j], bases[j])

        # Distance factors
        df = np.clip(1.0 - dist / KConfig.REPULSION_THRESHOLD, 0.0, 1.0)
        danger = np.clip(1.0 - dist / MIN_SAFE_DISTANCE, 0.0, 1.0)

        # Phase difference → leadership
        delta = (phis[i] - phis[j] + 0.5) % 1.0 - 0.5
        if delta > KConfig.LEADERSHIP_THRESHOLD:
            leader = i   # arm i is ahead
        elif delta < -KConfig.LEADERSHIP_THRESHOLD:
            leader = j
        else:
            leader = -1  # synchronized

        # Adaptive coupling
        K = KConfig.BASE_COUPLING * (1.0 + 3.0 * df)
        if leader >= 0 and dist < KConfig.REPULSION_THRESHOLD:
            K = min(K * 2.0, KConfig.MAX_COUPLING)

        # Kuramoto coupling
        dphi[i] += K * np.sin(phis[j] - phis[i])
        dphi[j] += K * np.sin(phis[i] - phis[j])

        # Repulsion
        if dist < KConfig.REPULSION_THRESHOLD:
            rep = KConfig.REPULSION_STRENGTH * (df ** 2) * 30.0

            if leader == i:       # i is ahead → slow i, speed j
                dphi[i] -= rep
                dphi[j] += rep * 0.3
            elif leader == j:     # j is ahead
                dphi[j] -= rep
                dphi[i] += rep * 0.3
            else:                  # synchronized
                dphi[i] -= rep * 0.7
                dphi[j] -= rep * 0.7

            # Emergency brake
            if dist < MIN_SAFE_DISTANCE:
                emg = KConfig.EMERGENCY_BRAKE * (danger ** 3)
                if leader == i:
                    dphi[i] -= emg * 2.0
                    dphi[j] -= emg * 0.5
                elif leader == j:
                    dphi[j] -= emg * 2.0
                    dphi[i] -= emg * 0.5
                else:
                    dphi[i] -= emg
                    dphi[j] -= emg

            # Frequency adaptation
            if leader == i:
                domega[i] -= KConfig.ADAPTATION_RATE * df * omegas[i]
                domega[j] += KConfig.ADAPTATION_RATE * 0.3 * df * (omegas[i] - omegas[j])
            elif leader == j:
                domega[j] -= KConfig.ADAPTATION_RATE * df * omegas[j]
                domega[i] += KConfig.ADAPTATION_RATE * 0.3 * df * (omegas[j] - omegas[i])
            else:
                domega[i] += KConfig.RECOVERY_RATE * (omegas[j] - omegas[i])
                domega[j] += KConfig.RECOVERY_RATE * (omegas[i] - omegas[j])

    # Clamp phase velocities
    dphi = np.clip(dphi, -KConfig.MAX_PHASE_VELOCITY, KConfig.MAX_PHASE_VELOCITY)

    return np.concatenate([dphi, domega])


# ── Synchronization ───────────────────────────────────────────────────────────

def synchronize(traj_data: Dict) -> Tuple[Dict, Dict]:
    """
    Run Kuramoto integration and return (synchronized_output, report).
    """
    print(f"\n{'='*70}")
    print(f"  QUAD-ARM KURAMOTO SYNCHRONIZATION  ({N_ARMS} oscillators)")
    print(f"{'='*70}")

    # Parse all trajectory arrays
    trajs = []
    times = []
    for name in ARM_NAMES:
        t = traj_data[name]['trajectory']
        trajs.append(np.array(t['positions']))
        times.append(np.array(t['time']))

    durations = [float(tv[-1]) for tv in times]
    duration  = max(durations)

    print(f"\n  Arm durations: {[f'{d:.2f}s' for d in durations]}")
    print(f"  Integration duration: {duration:.2f}s")
    print(f"  Pairs: {len(COLLISION_PAIRS)}")
    print(f"  DT: {KConfig.DT}s")

    # Initial state: all phases=0, omegas=1/duration
    n = N_ARMS
    omega0 = np.array([1.0 / duration] * n)
    state0 = np.concatenate([np.zeros(n), omega0])

    t_int = np.linspace(0, duration, int(duration / KConfig.DT))

    print(f"\n  Integrating ({len(t_int)} steps)...")

    sol = odeint(_derivatives, state0, t_int, args=(trajs,))

    phases = np.clip(sol[:, :n], 0.0, 1.0)

    print("  ✓ Integration complete")

    # Build synchronized trajectories
    print("  Building synchronized trajectories...")
    sync = [[] for _ in range(n)]
    for k in range(len(t_int)):
        for i in range(n):
            sync[i].append(_interpolate(trajs[i], phases[k, i]))
    sync = [np.array(s) for s in sync]

    # Analyze all pairs
    print("  Analyzing distances...")
    pair_stats = {}
    overall_violations = 0
    overall_min = np.inf

    for name_a, name_b in COLLISION_PAIRS:
        i = ARM_NAMES.index(name_a)
        j = ARM_NAMES.index(name_b)
        dists = [_pair_distance(sync[i][k], BASE_POSITIONS[name_a],
                                 sync[j][k], BASE_POSITIONS[name_b])
                 for k in range(len(t_int))]
        min_d = min(dists)
        viols = sum(1 for d in dists if d < MIN_SAFE_DISTANCE)
        overall_min = min(overall_min, min_d)
        overall_violations += viols
        pair_stats[f"{name_a}_{name_b}"] = {
            'min_distance_cm': round(min_d * 100, 2),
            'violations':      viols,
        }
        status = "✗" if viols > 0 else "✓"
        print(f"    {status} {name_a}↔{name_b}: min={min_d*100:.1f}cm  viols={viols}")

    is_safe = overall_violations == 0

    print(f"\n  Overall: {'✓ SAFE' if is_safe else '✗ UNSAFE'}"
          f"  min={overall_min*100:.1f}cm  total_violations={overall_violations}")

    # Build output
    output = {}
    for i, name in enumerate(ARM_NAMES):
        output[name] = {
            'robot_name': name,
            'metadata':   traj_data[name].get('metadata', {}),
            'trajectory': {
                'time':      t_int.tolist(),
                'positions': sync[i].tolist(),
                'num_samples': len(t_int),
            },
        }

    report = {
        'success':             is_safe,
        'min_distance_m':      float(overall_min),
        'min_distance_cm':     round(float(overall_min) * 100, 2),
        'total_violations':    overall_violations,
        'pair_stats':          pair_stats,
    }

    return output, report


# ── Post-sync verification ────────────────────────────────────────────────────

def verify(sync_output: Dict) -> Dict:
    print(f"\n{'='*70}")
    print("  POST-SYNC VERIFICATION")
    print(f"{'='*70}")

    trajs = {name: np.array(sync_output[name]['trajectory']['positions'])
             for name in ARM_NAMES if name in sync_output}
    n_steps = min(len(v) for v in trajs.values())

    overall_viols = 0
    overall_min   = np.inf

    for name_a, name_b in COLLISION_PAIRS:
        if name_a not in trajs or name_b not in trajs:
            continue
        dists = [_pair_distance(trajs[name_a][k], BASE_POSITIONS[name_a],
                                 trajs[name_b][k], BASE_POSITIONS[name_b])
                 for k in range(n_steps)]
        min_d = min(dists)
        viols = sum(1 for d in dists if d < MIN_SAFE_DISTANCE)
        overall_min   = min(overall_min, min_d)
        overall_viols += viols
        status = "✗" if viols > 0 else "✓"
        print(f"  {status} {name_a}↔{name_b}: min={min_d*100:.1f}cm viols={viols}")

    has_col = overall_viols > 0
    print(f"\n  Result: {'✗ UNSAFE' if has_col else '✓ SAFE FOR EXECUTION'}")
    print(f"  Min clearance: {overall_min*100:.1f}cm")
    print(f"{'='*70}\n")

    return {
        'has_collision':       has_col,
        'min_distance':        float(overall_min),
        'num_collision_points': overall_viols,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*70}")
    print("  QUAD-ARM KURAMOTO SYNCHRONIZATION")
    print(f"{'='*70}")

    # Try loading original trajectories (not collision_report which nests them)
    traj_data = None
    if os.path.exists('trajectories.json'):
        try:
            with open('trajectories.json') as f:
                traj_data = json.load(f)
            print("  ✓ Using trajectories.json")
        except Exception as e:
            print(f"  ⚠ Could not load trajectories.json: {e}")

    if traj_data is None:
        print("  ✗ trajectories.json not found")
        print("    Run: ros2 run dual_arm_sync quad_trajectory_generation")
        return

    sync_output, report = synchronize(traj_data)
    verification        = verify(sync_output)

    # Save
    output_data = {
        **{name: sync_output[name] for name in ARM_NAMES if name in sync_output},
        'synchronization_report':  report,
        'post_sync_verification':  verification,
        'parameters': {
            'min_safe_distance':   MIN_SAFE_DISTANCE,
            'base_coupling':       KConfig.BASE_COUPLING,
            'repulsion_strength':  KConfig.REPULSION_STRENGTH,
            'n_arms':              N_ARMS,
        },
    }

    with open('synchronized_trajectories.json', 'w') as f:
        json.dump(output_data, f, indent=2)

    size_kb = os.path.getsize('synchronized_trajectories.json') / 1024
    print(f"  ✓ Saved: synchronized_trajectories.json ({size_kb:.1f} KB)")

    if verification['has_collision']:
        print("\n  ✗ Still unsafe — run RRT planner:")
        print("    ros2 run dual_arm_sync quad_rrt_planner")
    else:
        print("\n  ✓ Safe — proceed to executor:")
        print("    ros2 run dual_arm_sync quad_gazebo_executor")

    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()