#!/usr/bin/env python3
"""
quad_trajectory_generation.py
Trajectory generation for 4-arm system.

Reads:  ik_solutions.json
Writes: trajectories.json  (same schema as dual-arm version)

Usage:
    ros2 run dual_arm_sync quad_trajectory_generation
"""

import numpy as np
from scipy.interpolate import splrep, splev
from typing import Dict, List, Tuple
import json
import os
import sys

try:
    from dual_arm_sync.constants import JointLimits
    try:
        from dual_arm_sync.quad_arm_config import ARM_NAMES, N_ARMS
    except ImportError:
        from quad_arm_config import ARM_NAMES, N_ARMS
except ImportError as e:
    print(f"ERROR: Cannot import required modules - {e}")
    sys.exit(1)


# ── B-spline helpers (identical to dual-arm version) ─────────────────────────

def _adaptive_control_points(start: np.ndarray,
                              end: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    delta_deg = np.abs(np.degrees(end - start))
    cp = np.zeros(6, dtype=int)
    for i in range(6):
        d = delta_deg[i]
        if d < 5.0:
            cp[i] = 0
        elif d < 15.0:
            cp[i] = 1
        else:
            cp[i] = int(np.ceil((d - 5.0) / 10.0))
    return cp, delta_deg


def _num_segments(cp: np.ndarray) -> int:
    total = int(np.sum(cp))
    if total == 0:
        return 1
    return max(1, min(int(np.ceil(total / 5.0)), 10))


def _distribute_cp(cp: np.ndarray, n_seg: int) -> np.ndarray:
    dist = np.zeros((n_seg, 6), dtype=int)
    for j in range(6):
        base = cp[j] // n_seg
        rem  = cp[j] % n_seg
        for s in range(n_seg):
            dist[s, j] = base + (1 if s < rem else 0)
    return dist


def _segment_waypoints(start: np.ndarray, end: np.ndarray,
                        n_seg: int, cp_dist: np.ndarray) -> List[Dict]:
    segs = []
    t_breaks = np.linspace(0, 1, n_seg + 1)
    for i in range(n_seg):
        t0, t1 = t_breaks[i], t_breaks[i + 1]
        n_wp = max(3, int(np.sum(cp_dist[i])) + 2)
        ts   = np.linspace(t0, t1, n_wp)
        wps  = np.array([start + t * (end - start) for t in ts])
        segs.append({
            'segment_id':    i,
            'time_start':    float(t0),
            'time_end':      float(t1),
            'num_waypoints': n_wp,
            'waypoints':     wps,
            'control_points': cp_dist[i].tolist(),
        })
    return segs


def _bspline(waypoints: np.ndarray, duration: float,
             dt: float = 0.01) -> Dict:
    n_wp = len(waypoints)
    t_params = np.linspace(0, 1, n_wp)
    all_pos, all_vel, all_acc = [], [], []

    for j in range(6):
        k = min(3, n_wp - 1)
        tck = splrep(t_params, waypoints[:, j], k=k, s=0)
        n_s = int(duration / dt)
        t_n = np.linspace(0, 1, n_s)
        all_pos.append(splev(t_n, tck, der=0))
        all_vel.append(splev(t_n, tck, der=1) / duration)
        all_acc.append(splev(t_n, tck, der=2) / (duration ** 2))

    positions     = np.array(all_pos).T
    velocities    = np.array(all_vel).T
    accelerations = np.array(all_acc).T
    time_vec      = np.linspace(0, duration, len(all_pos[0]))
    return {
        'time': time_vec, 'positions': positions,
        'velocities': velocities, 'accelerations': accelerations,
        'num_samples': len(time_vec),
    }


def _check_scale(traj: Dict, duration: float,
                  waypoints: np.ndarray) -> Dict:
    vel, acc = traj['velocities'], traj['accelerations']
    scale = 1.0
    for j in range(6):
        vr = np.max(np.abs(vel[:, j])) / JointLimits.VELOCITY_LIMITS[j]
        ar = np.sqrt(max(0, np.max(np.abs(acc[:, j])) /
                         JointLimits.ACCELERATION_LIMITS[j]))
        scale = max(scale, vr, ar)
    if scale > 1.0:
        new_dur = duration * scale * 1.1
        print(f"    ⚠ Scaling duration {duration:.2f}s → {new_dur:.2f}s")
        return _bspline(waypoints, new_dur)
    return traj


def generate_trajectory(start: np.ndarray, end: np.ndarray,
                         duration: float, robot_name: str) -> Dict:
    cp, delta_deg = _adaptive_control_points(start, end)
    n_seg  = _num_segments(cp)
    cp_dist = _distribute_cp(cp, n_seg)
    segs   = _segment_waypoints(start, end, n_seg, cp_dist)

    # flatten waypoints
    wps = []
    for i, seg in enumerate(segs):
        wps.extend(seg['waypoints'][:-1] if i < n_seg - 1 else seg['waypoints'])
    wps = np.array(wps)

    traj = _bspline(wps, duration)
    traj = _check_scale(traj, duration, wps)

    final_dur = float(traj['time'][-1])
    for seg in segs:
        seg['time_start_abs'] = seg['time_start'] * final_dur
        seg['time_end_abs']   = seg['time_end']   * final_dur
        seg['waypoints']      = seg['waypoints'].tolist()

    return {
        'robot_name': robot_name,
        'segments':   segs,
        'trajectory': traj,
        'metadata': {
            'total_control_points':    int(np.sum(cp)),
            'control_points_per_joint': cp.tolist(),
            'delta_angles_deg':        delta_deg.tolist(),
            'num_segments':            n_seg,
            'duration':                final_dur,
            'num_samples':             traj['num_samples'],
            'start_joints':            start.tolist(),
            'end_joints':              end.tolist(),
            'start_joints_deg':        np.degrees(start).tolist(),
            'end_joints_deg':          np.degrees(end).tolist(),
        },
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*70}")
    print(f"  QUAD-ARM TRAJECTORY GENERATION  ({N_ARMS} arms)")
    print(f"{'='*70}")

    try:
        with open('ik_solutions.json') as f:
            ik = json.load(f)
    except FileNotFoundError:
        print("  ✗ ik_solutions.json not found")
        print("    Run: ros2 run dual_arm_sync quad_arm_ik_solver")
        return

    duration = float(ik.get('duration', 10.0))
    output = {}

    for name in ARM_NAMES:
        if name not in ik:
            print(f"  ⚠ {name} missing from ik_solutions.json — skipping")
            continue

        sol = ik[name]
        start = np.array(sol['current_joints'])
        end   = np.array(sol['optimal_joints'])

        print(f"\n  {name}:")
        print(f"    Start (deg): {np.round(np.degrees(start), 1)}")
        print(f"    End   (deg): {np.round(np.degrees(end), 1)}")

        traj = generate_trajectory(start, end, duration, name)

        print(f"    Segments   : {traj['metadata']['num_segments']}")
        print(f"    CPs/joint  : {traj['metadata']['control_points_per_joint']}")
        print(f"    Duration   : {traj['metadata']['duration']:.2f}s")
        print(f"    Samples    : {traj['metadata']['num_samples']}")

        output[name] = {
            'robot_name': name,
            'metadata':   traj['metadata'],
            'segments':   traj['segments'],
            'trajectory': {
                'time':          traj['trajectory']['time'].tolist(),
                'positions':     traj['trajectory']['positions'].tolist(),
                'velocities':    traj['trajectory']['velocities'].tolist(),
                'accelerations': traj['trajectory']['accelerations'].tolist(),
                'num_samples':   traj['trajectory']['num_samples'],
            },
        }

    with open('trajectories.json', 'w') as f:
        json.dump(output, f, indent=2)

    size_kb = os.path.getsize('trajectories.json') / 1024
    print(f"\n  ✓ Saved: trajectories.json ({size_kb:.1f} KB)  [{len(output)}/{N_ARMS} arms]")
    print("\n  Next: ros2 run dual_arm_sync quad_collision_checker")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()