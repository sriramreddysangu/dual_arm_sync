#!/usr/bin/env python3
"""
step_26.py  --  ROS2 Executor   [6 ARM]
=======================================
INPUT  : s25_synchronized.json + s21_ik.json
OUTPUT : s26_execution.json

Publishes Float64MultiArray position commands at 100 Hz to each arm's
controller. Joint order (verified): the command interface expects straight DH
order array[i] -> joint_(i+1).  /joint_states is published scrambled
[j1,j2,j4,j5,j3,j6] by the broadcaster and read back by NAME, so commands are
NOT reordered (REORDER_CMD = False).
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _robot6x import (NDOF, POS_LIM, ROBOT_BASES, ARM_NAMES, DH_TO_CMD,
                      fk_world, pair_collides)

RATE_HZ   = 100.0
HOLD_S    = 1.5
SETTLE_S  = 2.5
EE_TOL_MM = 25.0
JT_TOL_D  = 5.0
REORDER_CMD = False        # straight DH order -- see header

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float64MultiArray
    from sensor_msgs.msg import JointState
    _ROS_OK = True
except ImportError:
    _ROS_OK = False


def _cmd(q):
    return q[DH_TO_CMD] if REORDER_CMD else q


if _ROS_OK:
    class ExecNode(Node):
        def __init__(self, arm_names):
            super().__init__('step_26_executor')
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
                 if all(k in jmap for k in keys) else np.array(msg.position[:NDOF]))
            self._cur_q[name] = q.astype(float)

        def execute(self, arm_pos, duration):
            dt_ns = int(1e9 / RATE_HZ); n_steps = max(len(arm_pos[n]) for n in self._names)
            msg = Float64MultiArray()
            self.get_logger().info(f'step_26: {n_steps} steps @ {RATE_HZ:.0f} Hz '
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
            for _ in range(int(HOLD_S * RATE_HZ)):
                t0 = time.monotonic_ns()
                for name in self._names:
                    msg.data = [float(v) for v in _cmd(arm_pos[name][-1])]
                    self._pubs[name].publish(msg)
                rclpy.spin_once(self, timeout_sec=0.)
                remain = dt_ns - (time.monotonic_ns() - t0)
                if remain > 0: time.sleep(remain * 1e-9)
            wall_s = time.monotonic() - t0_wall
            # Settle: wait until each arm is BOTH slow AND actually near its
            # commanded pose. Velocity-only settling captured the pose while the
            # Gazebo position controller was still creeping in (low velocity,
            # still 1-2 deg short) -> false 40-60mm EE error. We now also require
            # winding-aware joint convergence to the commanded final, with a
            # longer timeout so big-retraction arms have time to arrive.
            VEL_SETTLED = 0.01; POS_SETTLED = np.radians(1.5)   # rad, ~converged
            cmd_final = {n: np.array(arm_pos[n][-1], float) for n in self._names}
            t_settle = time.time()
            final_q = {n: None for n in self._names}; last_q = {n: None for n in self._names}
            settled = 0
            while time.time() - t_settle < SETTLE_S * 6.0:
                t0s = time.monotonic_ns()
                for name in self._names:
                    msg.data = [float(v) for v in _cmd(arm_pos[name][-1])]
                    self._pubs[name].publish(msg)
                rclpy.spin_once(self, timeout_sec=0.)
                mv = 0.0; perr = 0.0; have_all = True
                for n in self._names:
                    if self._cur_q[n] is None:
                        have_all = False; continue
                    if last_q[n] is not None:
                        mv = max(mv, float(np.max(np.abs(self._cur_q[n] - last_q[n])) / 0.01))
                    dq = np.angle(np.exp(1j * (self._cur_q[n] - cmd_final[n])))  # wrap to [-pi,pi]
                    perr = max(perr, float(np.max(np.abs(dq))))
                    last_q[n] = self._cur_q[n].copy()
                if have_all and mv < VEL_SETTLED and perr < POS_SETTLED:
                    settled += 1
                    if settled >= 20: break
                else: settled = 0
                remain = dt_ns - (time.monotonic_ns() - t0s)
                if remain > 0: time.sleep(remain * 1e-9)
            for n in self._names:
                if self._cur_q[n] is not None: final_q[n] = self._cur_q[n].copy()
            self.get_logger().info(f'  done  wall={wall_s:.2f}s')
            return {'success': True, 'mode': 'ros2', 'steps_sent': n_steps,
                    'wall_time_s': round(wall_s, 3),
                    'final_joints': {n: final_q[n].tolist() if final_q[n] is not None
                                     else None for n in self._names}, 'error': None}


def dry_run(arm_names, arm_pos, duration):
    n = max(len(arm_pos[nm]) for nm in arm_names)
    print(f'\n  DRY-RUN: {n} steps ({duration:.2f}s @ {RATE_HZ:.0f} Hz)')
    for k in range(0, n + 1, max(1, n // 5)):
        k = min(k, n - 1)
        nm = arm_names[0]; q = arm_pos[nm][min(k, len(arm_pos[nm]) - 1)]
        print(f'  {int(100*k/n):3d}%  [{nm}] [{", ".join(f"{np.degrees(v):.1f}" for v in q)}]')
    return {'success': True, 'mode': 'dry_run', 'steps_sent': n, 'wall_time_s': None,
            'final_joints': {nm: arm_pos[nm][-1].tolist() for nm in arm_names}, 'error': None}


def preflight(arm_names, arm_pos, bases):
    issues = []; N = len(arm_names)
    for i in range(N):
        for j in range(i+1, N):
            ni, nj = arm_names[i], arm_names[j]; pi, pj = arm_pos[ni], arm_pos[nj]
            nc = sum(1 for k in range(min(len(pi), len(pj)))
                     if pair_collides(pi[k], bases[ni], pj[k], bases[nj]))
            if nc > 0: issues.append(f'{ni}<->{nj}: {nc} collision steps')
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
    print('  STEP 26  --  ROS2 Executor [6 ARM]')
    print('='*66)
    for f, s in [('s25_synchronized.json', 'step_25'), ('s21_ik.json', 'step_21')]:
        if not os.path.exists(f): print(f'  {f} not found'); sys.exit(1)
    with open('s25_synchronized.json') as fh: sdata = json.load(fh)
    with open('s21_ik.json') as fh: ik = json.load(fh)
    arm_names = sdata.get('arm_names', ARM_NAMES); duration = float(sdata['duration'])
    bases = {n: np.array(ROBOT_BASES.get(n, [0, 0, 0])) for n in arm_names}
    arm_pos = {}
    for name in arm_names:
        pos = np.array(sdata[name]['trajectory']['positions'], float)
        nout = max(2, int(round(duration * RATE_HZ)))
        sin = np.linspace(0, 1, len(pos)); sout = np.linspace(0, 1, nout)
        arm_pos[name] = np.clip(np.vstack([np.interp(sout, sin, pos[:, j])
                                           for j in range(NDOF)]).T, POS_LIM[:, 0], POS_LIM[:, 1])
    print(f'\n  Arms: {arm_names}  dur={duration:.2f}s  '
          f'steps={max(len(arm_pos[n]) for n in arm_names)}')
    print('\n  Pre-flight...')
    issues = preflight(arm_names, arm_pos, bases)
    if issues: print(f'  FAIL: {issues}'); sys.exit(1)
    print('  OK Passed')
    if _ROS_OK:
        rclpy.init(args=args); node = ExecNode(arm_names)
        try: exec_res = node.execute(arm_pos, duration)
        finally:
            try: node.destroy_node()
            except: pass
            rclpy.shutdown()
    else:
        exec_res = dry_run(arm_names, arm_pos, duration)
    verif = {n: verify_arm(n, arm_pos[n][-1], exec_res['final_joints'].get(n)) for n in arm_names}
    report = {'arm_names': arm_names, 'duration_s': duration, 'execution': exec_res,
              'verification': verif, 'pipeline': 'kuramoto+retraction (step_2X) [6 ARM]'}
    print(f'\n  {"="*66}')
    no_data = [n for n in arm_names if exec_res['final_joints'].get(n) is None]
    for name in arm_names:
        v = verif[name]
        print(f'  [{name}]  j_err={v.get("joint_error_deg","?")} deg  '
              f'ee_err={v.get("ee_error_mm","?")} mm  ee_ok={v.get("ee_ok","?")}')
    if no_data:
        print(f'\n  NOTE: no joint_states received for {no_data} -- "?" means the')
        print('  executor could not read the robot back (Gazebo not publishing).')
        print('  Start Gazebo + controllers and confirm with:')
        print('    ros2 topic echo /dsr01/gz/joint_states --once')
    with open('s26_execution.json', 'w') as fh: json.dump(report, fh, indent=2)
    print(f'\n  Saved: s26_execution.json\n')


if __name__ == '__main__': main()