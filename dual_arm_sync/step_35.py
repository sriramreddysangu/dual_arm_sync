#!/usr/bin/env python3
"""
step_35.py  --  ROS2 Executor for Temporal Pipeline   [DUAL ARM]
================================================================
INPUT  : s34_synchronized.json + s31_ik.json
OUTPUT : s35_execution.json
Dual-arm version (imports _robot2x). Executes the re-timed (path-velocity
coordination) schedule from step_34: each arm keeps its seed PATH, only the
relative TIMING differs.

JOINT ORDER (verified): Gazebo's state-broadcaster lists /joint_states as
[j1,j2,j4,j5,j3,j6] -- handled on READ by mapping by name in _cb. The command
interface, however, expects straight DH order array[i] -> joint_(i+1), so
REORDER_CMD = False. Reading-by-name + writing-DH-order is the correct combo.
"""

import json, os, sys, time
from typing import Dict
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _robot2x import (NDOF, POS_LIM, ROBOT_BASES, ARM_NAMES, DH_TO_CMD,
                       fk_world, pair_min_dist)

# Execution gate uses REAL link clearance, identical to step_34. The built-in
# pair_collides / deepest_link_pair margin flags arms ~11cm apart as colliding
# (see step_34_diag), falsely refusing valid phase-lag solutions. Gate on actual
# distance. Tune via env DUAL_ARM_CLEARANCE_M (default 0.05 = 5cm).
CLEAR_M = float(os.environ.get('DUAL_ARM_CLEARANCE_M', '0.05'))

RATE_HZ   = 100.0
HOLD_S    = 1.5
SETTLE_S  = 2.5
EE_TOL_MM = 25.0
JT_TOL_D  = 5.0

# Joint order (VERIFIED from your Gazebo echo): the COMMAND interface expects
# straight DH order  array[i] -> joint_(i+1).  The scrambled [j1,j2,j4,j5,j3,j6]
# you see is only how the state-broadcaster lists /joint_states, which is handled
# on the READ side (_cb maps by name).  Do NOT reorder commands -> keep False.
# (True swaps q3/q4/q5 and produces ~258 deg errors / joint4 flailing.)
REORDER_CMD = False

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float64MultiArray
    from sensor_msgs.msg import JointState
    _ROS_OK = True
except ImportError:
    _ROS_OK = False


def _cmd(q):
    """Reorder a DH-order joint vector into controller command order if enabled."""
    return q[DH_TO_CMD] if REORDER_CMD else q


if _ROS_OK:
    class ExecNode(Node):
        def __init__(self, arm_names):
            super().__init__('step_35_executor')
            self._names = arm_names
            self._pubs = {n: self.create_publisher(
                Float64MultiArray, f'/{n}/gz/dsr_position_controller/commands', 10)
                for n in arm_names}
            self._cur_q = {n: None for n in arm_names}
            for name in arm_names:
                for t in (f'/{name}/gz/joint_states', f'/{name}/joint_states'):
                    self.create_subscription(JointState, t,
                        lambda msg, n=name: self._cb(msg, n), 10)

        def _cb(self, msg, name):
            if len(msg.position) < NDOF: return
            jmap = {n: i for i, n in enumerate(msg.name)}
            keys = [f'joint_{k}' for k in range(1, NDOF + 1)]
            q = (np.array([msg.position[jmap[k]] for k in keys])
                 if all(k in jmap for k in keys)
                 else np.array(msg.position[:NDOF]))
            self._cur_q[name] = q.astype(float)

        def execute(self, arm_pos, duration):
            dt_ns = int(1e9 / RATE_HZ)
            n_steps = max(len(arm_pos[n]) for n in self._names)
            msg = Float64MultiArray()
            self.get_logger().info(f'step_35: {n_steps} steps @ {RATE_HZ:.0f} Hz '
                                   f'(REORDER_CMD={REORDER_CMD})')
            t0_wall = time.monotonic()
            for k in range(n_steps):
                t0 = time.monotonic_ns()
                for name in self._names:
                    idx = min(k, len(arm_pos[name]) - 1)
                    msg.data = [float(v) for v in _cmd(arm_pos[name][idx])]
                    self._pubs[name].publish(msg)
                rclpy.spin_once(self, timeout_sec=0.)
                remain = dt_ns - (time.monotonic_ns() - t0)
                if remain > 0: time.sleep(remain * 1e-9)
                if k % max(1, n_steps//10) == 0:
                    self.get_logger().info(f'  {int(100*k/n_steps):3d}%')
            hold_steps = int(HOLD_S * RATE_HZ)
            for _ in range(hold_steps):
                t0 = time.monotonic_ns()
                for name in self._names:
                    msg.data = [float(v) for v in _cmd(arm_pos[name][-1])]
                    self._pubs[name].publish(msg)
                rclpy.spin_once(self, timeout_sec=0.)
                remain = dt_ns - (time.monotonic_ns() - t0)
                if remain > 0: time.sleep(remain * 1e-9)
            wall_s = time.monotonic() - t0_wall
            VEL_SETTLED = 0.01
            t_settle = time.time()
            final_q = {n: None for n in self._names}; last_q = {n: None for n in self._names}
            settled_count = 0; settle_dt_ns = int(1e9 / RATE_HZ)
            while time.time() - t_settle < SETTLE_S * 3.0:
                t0_sub = time.monotonic_ns()
                for name in self._names:
                    msg.data = [float(v) for v in _cmd(arm_pos[name][-1])]
                    self._pubs[name].publish(msg)
                rclpy.spin_once(self, timeout_sec=0.)
                max_vel = 0.0
                for n in self._names:
                    if self._cur_q[n] is not None and last_q[n] is not None:
                        v = float(np.max(np.abs(self._cur_q[n] - last_q[n])) / 0.01)
                        if v > max_vel: max_vel = v
                    if self._cur_q[n] is not None: last_q[n] = self._cur_q[n].copy()
                if max_vel < VEL_SETTLED and all(self._cur_q[n] is not None for n in self._names):
                    settled_count += 1
                    if settled_count >= 30: break
                else:
                    settled_count = 0
                remain = settle_dt_ns - (time.monotonic_ns() - t0_sub)
                if remain > 0: time.sleep(remain * 1e-9)
            for n in self._names:
                if self._cur_q[n] is not None: final_q[n] = self._cur_q[n].copy()
            self.get_logger().info(f'  Settled after {time.time()-t_settle:.2f}s')
            self.get_logger().info(f'  100%  done  wall={wall_s:.2f}s')
            return {'success': True, 'mode': 'ros2', 'steps_sent': n_steps,
                    'wall_time_s': round(wall_s, 3),
                    'final_joints': {n: final_q[n].tolist() if final_q[n] is not None
                                     else None for n in self._names}, 'error': None}


def dry_run(arm_names, arm_pos, duration):
    n = max(len(arm_pos[nm]) for nm in arm_names)
    print(f'\n  DRY-RUN: {n} steps  ({duration:.2f}s @ {RATE_HZ:.0f} Hz)')
    for k in range(0, n+1, max(1, n//10)):
        k = min(k, n - 1)
        for nm in arm_names:
            q = arm_pos[nm][min(k, len(arm_pos[nm]) - 1)]
            print(f'  {int(100*k/n):3d}%  [{nm}] [{", ".join(f"{np.degrees(v):.1f}" for v in q)}] deg')
    return {'success': True, 'mode': 'dry_run', 'steps_sent': n, 'wall_time_s': None,
            'final_joints': {nm: arm_pos[nm][-1].tolist() for nm in arm_names}, 'error': None}


def preflight(arm_names, arm_pos, bases):
    issues = []; N = len(arm_names)
    for i in range(N):
        for j in range(i+1, N):
            ni, nj = arm_names[i], arm_names[j]; pi, pj = arm_pos[ni], arm_pos[nj]
            nc = sum(1 for k in range(min(len(pi), len(pj)))
                     if pair_min_dist(pi[k], bases[ni], pj[k], bases[nj]) < CLEAR_M)
            if nc > 0: issues.append(f'{ni}<->{nj}: {nc} steps closer than {CLEAR_M*100:.0f}cm')
    return issues


def verify_arm(name, ref_q, final_q):
    """Verify against the ACTUAL commanded final pose (trajectory endpoint), with
    winding-aware joint error and FK-based EE (both 2pi-periodic-correct), so a
    joint that settles in a different 2pi winding does not show a false error."""
    base = np.array(ROBOT_BASES.get(name, [0, 0, 0]))
    ref_q = np.array(ref_q, float)
    tgt_ee = fk_world(ref_q, base)
    res = {'commanded_final_deg': np.degrees(ref_q).tolist()}
    if final_q is not None:
        fq = np.array(final_q, float); ee = fk_world(fq, base)
        dq = np.angle(np.exp(1j * (fq - ref_q)))      # wrap each joint diff to [-pi, pi]
        res.update({'final_joints_deg': np.degrees(fq).tolist(),
            'joint_error_deg': round(float(np.max(np.abs(np.degrees(dq)))), 4),
            'ee_error_mm': round(float(np.linalg.norm(ee - tgt_ee) * 1000), 3)})
        res['ee_ok'] = res['ee_error_mm'] <= EE_TOL_MM
        res['joints_ok'] = res['joint_error_deg'] <= JT_TOL_D
    return res


def main(args=None):
    print('\n' + '='*66)
    print('  STEP 35  --  ROS2 Executor for Temporal Pipeline [DUAL ARM]')
    print('='*66)
    for fname, step in [('s34_synchronized.json', 'step_34'), ('s31_ik.json', 'step_31')]:
        if not os.path.exists(fname): print(f'  {fname} not found'); sys.exit(1)
    with open('s34_synchronized.json') as fh: sdata = json.load(fh)
    with open('s31_ik.json') as fh: ik = json.load(fh)
    arm_names = sdata.get('arm_names', ARM_NAMES); duration = float(sdata['duration'])
    bases = {n: np.array(ROBOT_BASES.get(n, [0, 0, 0])) for n in arm_names}
    arm_pos = {}
    for name in arm_names:
        pos = np.array(sdata[name]['trajectory']['positions'], dtype=float)
        nout = max(2, int(round(duration * RATE_HZ)))
        sin = np.linspace(0, 1, len(pos)); sout = np.linspace(0, 1, nout)
        r = np.zeros((nout, NDOF))
        for j in range(NDOF): r[:, j] = np.interp(sout, sin, pos[:, j])
        arm_pos[name] = np.clip(r, POS_LIM[:, 0], POS_LIM[:, 1])
    max_steps = max(len(arm_pos[n]) for n in arm_names)
    print(f'\n  Arms: {arm_names}  dur={duration:.2f}s  steps={max_steps}')
    print('\n  Pre-flight...')
    issues = preflight(arm_names, arm_pos, bases)
    if issues: print(f'  FAIL: {issues}'); sys.exit(1)
    print('  OK Passed')
    if _ROS_OK:
        rclpy.init(args=args); node = ExecNode(arm_names)
        try: exec_res = node.execute(arm_pos, duration)
        except KeyboardInterrupt:
            exec_res = {'success': False, 'mode': 'interrupted', 'steps_sent': 0,
                'wall_time_s': None, 'final_joints': {n: None for n in arm_names},
                'error': 'KeyboardInterrupt'}
        finally:
            try: node.destroy_node()
            except: pass
            rclpy.shutdown()
    else:
        exec_res = dry_run(arm_names, arm_pos, duration)
    verif = {n: verify_arm(n, arm_pos[n][-1], exec_res['final_joints'].get(n)) for n in arm_names}
    report = {'arm_names': arm_names, 'duration_s': duration, 'execution': exec_res,
              'verification': verif, 'pipeline': 'temporal coordination (step_31..35) [DUAL ARM]'}
    print(f'\n  {"="*66}')
    for name in arm_names:
        v = verif[name]
        print(f'  [{name}]  j_err={v.get("joint_error_deg","?")} deg  '
              f'ee_err={v.get("ee_error_mm","?")} mm  ee_ok={v.get("ee_ok","?")}')
    with open('s35_execution.json', 'w') as fh: json.dump(report, fh, indent=2)
    print(f'\n  Saved: s35_execution.json\n')


if __name__ == '__main__': main()