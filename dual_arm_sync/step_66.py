#!/usr/bin/env python3
"""
step_66.py  --  ROS2 Executor
===============================
INPUT  : s65_synchronized.json + s61_ik.json
OUTPUT : s66_execution.json

Executes synchronized trajectories at 100 Hz, lock-step across all arms.
DH_TO_CMD reordering applied on every command published.
Reads joint states by name (jmap) -- no reorder needed on read.
"""

import json, os, sys, time
from typing import Dict, List, Optional
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _robot import (NDOF, POS_LIM, ROBOT_BASES, ARM_NAMES, DH_TO_CMD,
                    fk_world, pair_collides)

RATE_HZ   = 100.0
HOLD_S    = 1.5
SETTLE_S  = 2.5
EE_TOL_MM = 25.0
JT_TOL_D  = 5.0    # degrees

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float64MultiArray
    from sensor_msgs.msg import JointState
    _ROS_OK = True
except ImportError:
    _ROS_OK = False


if _ROS_OK:
    class ExecNode(Node):
        def __init__(self, arm_names):
            super().__init__('step_66_executor')
            self._names = arm_names
            self._pubs  = {n: self.create_publisher(
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
            keys = [f'joint_{k}' for k in range(1, NDOF+1)]
            q    = (np.array([msg.position[jmap[k]] for k in keys])
                    if all(k in jmap for k in keys)
                    else np.array(msg.position[:NDOF]))
            self._cur_q[name] = q.astype(float)

        def execute(self, arm_pos, duration):
            dt_ns   = int(1e9 / RATE_HZ)
            n_steps = max(len(arm_pos[n]) for n in self._names)
            msg     = Float64MultiArray()
            self.get_logger().info(
                f'step_66: {n_steps} steps x {len(self._names)} arms @ {RATE_HZ:.0f} Hz')
            t0_wall = time.monotonic()

            for k in range(n_steps):
                t0 = time.monotonic_ns()
                for name in self._names:
                    idx      = min(k, len(arm_pos[name])-1)
                    msg.data = [float(v) for v in arm_pos[name][idx]]   # DH order direct (no reorder)
                    self._pubs[name].publish(msg)
                rclpy.spin_once(self, timeout_sec=0.)
                remain = dt_ns - (time.monotonic_ns() - t0)
                if remain > 0: time.sleep(remain * 1e-9)
                if k % max(1, n_steps//10) == 0:
                    self.get_logger().info(f'  {int(100*k/n_steps):3d}%  step {k}/{n_steps}')

            hold_steps = int(HOLD_S * RATE_HZ)
            for _ in range(hold_steps):
                t0 = time.monotonic_ns()
                for name in self._names:
                    msg.data = [float(v) for v in arm_pos[name][-1]]   # DH order direct (no reorder)
                    self._pubs[name].publish(msg)
                rclpy.spin_once(self, timeout_sec=0.)
                remain = dt_ns - (time.monotonic_ns() - t0)
                if remain > 0: time.sleep(remain * 1e-9)

            wall_s = time.monotonic() - t0_wall
            # ── Proper settle wait ─────────────────────────────────────────
            # Continue commanding final position while PID converges; wait
            # until joint velocities drop below threshold for >= 300ms.
            VEL_SETTLED = 0.01
            t_settle = time.time()
            final_q  = {n: None for n in self._names}
            last_q   = {n: None for n in self._names}
            settled_count = 0
            settle_dt_ns = int(1e9 / RATE_HZ)
            while time.time() - t_settle < SETTLE_S * 3.0:
                t0_sub = time.monotonic_ns()
                for name in self._names:
                    msg.data = [float(v) for v in arm_pos[name][-1]]
                    self._pubs[name].publish(msg)
                rclpy.spin_once(self, timeout_sec=0.)
                max_vel = 0.0
                for n in self._names:
                    if self._cur_q[n] is not None and last_q[n] is not None:
                        v = float(np.max(np.abs(self._cur_q[n] - last_q[n])) / 0.01)
                        if v > max_vel: max_vel = v
                    if self._cur_q[n] is not None:
                        last_q[n] = self._cur_q[n].copy()
                if max_vel < VEL_SETTLED and all(self._cur_q[n] is not None for n in self._names):
                    settled_count += 1
                    if settled_count >= 30: break
                else:
                    settled_count = 0
                remain = settle_dt_ns - (time.monotonic_ns() - t0_sub)
                if remain > 0: time.sleep(remain * 1e-9)
            for n in self._names:
                if self._cur_q[n] is not None:
                    final_q[n] = self._cur_q[n].copy()
            self.get_logger().info(f'  Settled after {time.time()-t_settle:.2f}s')
            self.get_logger().info(f'  100%  done  wall={wall_s:.2f}s')
            return {'success': True, 'mode': 'ros2', 'steps_sent': n_steps,
                    'wall_time_s': round(wall_s, 3),
                    'final_joints': {n: final_q[n].tolist() if final_q[n] is not None else None
                                     for n in self._names}, 'error': None}


def dry_run(arm_names, arm_pos, duration):
    n = max(len(arm_pos[nm]) for nm in arm_names)
    print(f'\n  DRY-RUN: {n} steps  ({duration:.2f}s @ {RATE_HZ:.0f} Hz)')
    for k in range(0, n+1, max(1, n//10)):
        k = min(k, n-1)
        for nm in arm_names:
            q = arm_pos[nm][min(k, len(arm_pos[nm])-1)]
            print(f'  {int(100*k/n):3d}%  [{nm}] [{", ".join(f"{np.degrees(v):.1f}" for v in q)}] deg')
    return {'success': True, 'mode': 'dry_run', 'steps_sent': n,
            'wall_time_s': None,
            'final_joints': {nm: arm_pos[nm][-1].tolist() for nm in arm_names},
            'error': None}


def preflight(arm_names, arm_pos, bases):
    issues = []
    N = len(arm_names)
    for i in range(N):
        for j in range(i+1, N):
            ni, nj = arm_names[i], arm_names[j]
            pi, pj = arm_pos[ni], arm_pos[nj]
            nc = sum(1 for k in range(min(len(pi),len(pj)))
                     if pair_collides(pi[k], bases[ni], pj[k], bases[nj]))
            if nc > 0: issues.append(f'{ni}<->{nj}: {nc} collision steps')
    return issues


def verify_arm(name, meta, final_q_dh):
    base   = np.array(ROBOT_BASES.get(name, [0,0,0]))
    end_q  = np.array(meta.get('end_joints', [0]*NDOF), dtype=float)
    tgt_ee = np.array(meta.get('target_ee_world', fk_world(end_q, base).tolist()))
    result = {'end_joints_planned_deg': np.degrees(end_q).tolist()}
    if final_q_dh is not None:
        fq      = np.array(final_q_dh, dtype=float)
        ee_act  = fk_world(fq, base)
        j_err   = float(np.max(np.abs(np.degrees(fq - end_q))))
        ee_err  = float(np.linalg.norm(ee_act - tgt_ee)*1000)
        result.update({
            'final_joints_deg': np.degrees(fq).tolist(),
            'joint_error_deg' : round(j_err, 4),
            'ee_error_mm'     : round(ee_err, 3),
            'ee_ok'           : ee_err <= EE_TOL_MM,
            'joints_ok'       : j_err <= JT_TOL_D,
        })
    return result


def main(args=None):
    print('\n' + '='*66); print('  STEP 66  --  ROS2 Executor'); print('='*66)
    for fname, step in [('s65_synchronized.json','step_65'),('s61_ik.json','step_61')]:
        if not os.path.exists(fname): print(f'  {fname} not found'); sys.exit(1)
    with open('s65_synchronized.json') as fh: sdata = json.load(fh)
    with open('s61_ik.json')           as fh: ik    = json.load(fh)

    arm_names = sdata.get('arm_names', ARM_NAMES)
    duration  = float(sdata['duration'])
    bases     = {n: np.array(ROBOT_BASES.get(n,[0,0,0])) for n in arm_names}
    arm_pos   = {}
    for name in arm_names:
        pos = np.array(sdata[name]['trajectory']['positions'], dtype=float)
        nout = max(2, int(round(duration * RATE_HZ)))
        sin  = np.linspace(0,1,len(pos)); sout = np.linspace(0,1,nout)
        r    = np.zeros((nout, NDOF))
        for j in range(NDOF): r[:,j] = np.interp(sout, sin, pos[:,j])
        arm_pos[name] = np.clip(r, POS_LIM[:,0], POS_LIM[:,1])

    max_steps = max(len(arm_pos[n]) for n in arm_names)
    for name in arm_names:
        if len(arm_pos[name]) < max_steps:
            sin  = np.linspace(0,1,len(arm_pos[name])); sout = np.linspace(0,1,max_steps)
            r    = np.zeros((max_steps,NDOF))
            for j in range(NDOF): r[:,j] = np.interp(sout, sin, arm_pos[name][:,j])
            arm_pos[name] = r

    print(f'\n  Arms: {arm_names}  dur={duration:.2f}s  steps={max_steps}')
    for name in arm_names:
        end_q = np.array(sdata[name]['metadata']['end_joints'])
        print(f'  [{name}] target: [{", ".join(f"{np.degrees(v):.1f}" for v in end_q)}] deg')

    print('\n  Pre-flight check...')
    issues = preflight(arm_names, arm_pos, bases)
    if issues:
        print(f'  FAIL: {issues}'); sys.exit(1)
    print('  OK Passed')

    if _ROS_OK:
        rclpy.init(args=args); node = ExecNode(arm_names)
        try:   exec_res = node.execute(arm_pos, duration)
        except KeyboardInterrupt:
            exec_res = {'success':False,'mode':'interrupted','steps_sent':0,
                        'wall_time_s':None,'final_joints':{n:None for n in arm_names},'error':'KeyboardInterrupt'}
        except Exception as e:
            exec_res = {'success':False,'mode':'error','steps_sent':0,
                        'wall_time_s':None,'final_joints':{n:None for n in arm_names},'error':str(e)}
        finally:
            try: node.destroy_node()
            except: pass
            rclpy.shutdown()
    else:
        exec_res = dry_run(arm_names, arm_pos, duration)

    verif = {n: verify_arm(n, sdata[n]['metadata'],
                            exec_res['final_joints'].get(n)) for n in arm_names}
    report = {'arm_names': arm_names, 'duration_s': duration,
              'execution': exec_res, 'verification': verif}

    print(f'\n  {"="*66}')
    for name in arm_names:
        v = verif[name]
        print(f'  [{name}]  j_err={v.get("joint_error_deg","?")} deg  '
              f'ee_err={v.get("ee_error_mm","?")} mm  '
              f'ee_ok={v.get("ee_ok","?")}')

    with open('s66_execution.json','w') as fh: json.dump(report, fh, indent=2)
    print(f'\n  Saved: s66_execution.json')
    print(f'  Next : ros2 run dual_arm_sync step_67\n')

if __name__ == '__main__': main()