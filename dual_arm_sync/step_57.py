#!/usr/bin/env python3
"""
step_57.py  --  Executor + Tier Summary
==========================================
INPUT  : s54_tier1.json, s55_tier2.json, s56_tier3.json (whichever exists)
OUTPUT : s57_execution.json

  Decides which tier output to execute (the highest-numbered one that
  resolved). Prints a tier summary table showing which tier was needed for
  this case. Runs the Gazebo executor if ROS2 is available, dry-runs
  otherwise. DH-order joint commands published directly (no reorder).
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _robot5x import (NDOF, POS_LIM, ROBOT_BASES, ARM_NAMES,
                       fk_world, pair_collides)

RATE_HZ   = 100.0
HOLD_S    = 1.5
SETTLE_S  = 2.5
EE_TOL_MM = 25.0
JT_TOL_D  = 5.0

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
            super().__init__('step_57_executor')
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
            self.get_logger().info(f'step_57: {n_steps} steps @ {RATE_HZ:.0f} Hz')
            t0_wall = time.monotonic()
            for k in range(n_steps):
                t0 = time.monotonic_ns()
                for name in self._names:
                    idx = min(k, len(arm_pos[name]) - 1)
                    # Publish DH order directly (no DH_TO_CMD reorder)
                    msg.data = [float(v) for v in arm_pos[name][idx]]
                    self._pubs[name].publish(msg)
                rclpy.spin_once(self, timeout_sec=0.)
                remain = dt_ns - (time.monotonic_ns() - t0)
                if remain > 0: time.sleep(remain * 1e-9)
                if k % max(1, n_steps//10) == 0:
                    self.get_logger().info(f'  {int(100*k/n_steps):3d}%')
            for _ in range(int(HOLD_S * RATE_HZ)):
                t0 = time.monotonic_ns()
                for name in self._names:
                    msg.data = [float(v) for v in arm_pos[name][-1]]
                    self._pubs[name].publish(msg)
                rclpy.spin_once(self, timeout_sec=0.)
                remain = dt_ns - (time.monotonic_ns() - t0)
                if remain > 0: time.sleep(remain * 1e-9)
            wall_s = time.monotonic() - t0_wall
            # ── Proper settle wait ─────────────────────────────────────────
            # PID controller is still converging on target right after trajectory
            # ends. We must wait until BOTH conditions hold:
            #   (a) at least SETTLE_S has elapsed (give PID time to converge)
            #   (b) joint velocities have dropped below a small threshold
            # Then capture final_q. This replaces the previous logic that
            # broke out on the first joint_state callback (which gave a snapshot
            # while arm was still moving, producing huge reported errors).
            VEL_SETTLED = 0.01    # rad/s -- below this we consider arm settled
            t_settle = time.time()
            final_q = {n: None for n in self._names}
            last_q  = {n: None for n in self._names}
            settled_count = 0    # count of consecutive checks where vel is low
            # Continue commanding the last position during settle (keep PID active)
            settle_dt_ns = int(1e9 / RATE_HZ)
            while time.time() - t_settle < SETTLE_S * 3.0:    # allow up to 3x for slow settles
                t0_sub = time.monotonic_ns()
                # Keep publishing the final target so PID drives toward it
                for name in self._names:
                    msg.data = [float(v) for v in arm_pos[name][-1]]
                    self._pubs[name].publish(msg)
                rclpy.spin_once(self, timeout_sec=0.)
                # Estimate joint velocity from consecutive readings
                max_vel = 0.0
                for n in self._names:
                    if self._cur_q[n] is not None and last_q[n] is not None:
                        v = float(np.max(np.abs(self._cur_q[n] - last_q[n])) / 0.01)
                        if v > max_vel: max_vel = v
                    if self._cur_q[n] is not None:
                        last_q[n] = self._cur_q[n].copy()
                if max_vel < VEL_SETTLED and all(self._cur_q[n] is not None for n in self._names):
                    settled_count += 1
                    if settled_count >= 30:    # 30 * 10ms = 300ms of low velocity
                        break
                else:
                    settled_count = 0
                # Sleep to maintain rate
                remain = settle_dt_ns - (time.monotonic_ns() - t0_sub)
                if remain > 0: time.sleep(remain * 1e-9)

            # Final snapshot after settling
            for n in self._names:
                if self._cur_q[n] is not None:
                    final_q[n] = self._cur_q[n].copy()
            self.get_logger().info(f'  Settled after {time.time()-t_settle:.2f}s')
            self.get_logger().info(f'  100%  done  wall={wall_s:.2f}s')
            return {'success': True, 'mode': 'ros2', 'steps_sent': n_steps,
                    'wall_time_s': round(wall_s, 3),
                    'final_joints': {n: (final_q[n].tolist() if final_q[n] is not None else None)
                                      for n in self._names}, 'error': None}


def dry_run(arm_names, arm_pos, duration):
    n = max(len(arm_pos[nm]) for nm in arm_names)
    print(f'\n  DRY-RUN: {n} steps ({duration:.2f}s @ {RATE_HZ:.0f} Hz)')
    for k in range(0, n+1, max(1, n//10)):
        k = min(k, n - 1)
        for nm in arm_names:
            q = arm_pos[nm][min(k, len(arm_pos[nm]) - 1)]
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
            nc = sum(1 for k in range(min(len(pi), len(pj)))
                      if pair_collides(pi[k], bases[ni], pj[k], bases[nj]))
            if nc > 0: issues.append(f'{ni}<->{nj}: {nc} collision steps')
    return issues


def verify_arm(name, meta, final_q_dh):
    base = np.array(ROBOT_BASES.get(name, [0, 0, 0]))
    end_q = np.array(meta.get('end_joints', [0]*NDOF), dtype=float)
    tgt_ee = np.array(meta.get('target_ee_world', fk_world(end_q, base).tolist()))
    result = {'end_joints_planned_deg': np.degrees(end_q).tolist()}
    if final_q_dh is not None:
        fq = np.array(final_q_dh, dtype=float)
        ee_act = fk_world(fq, base)
        j_err = float(np.max(np.abs(np.degrees(fq - end_q))))
        ee_err = float(np.linalg.norm(ee_act - tgt_ee) * 1000)
        result.update({
            'final_joints_deg': np.degrees(fq).tolist(),
            'joint_error_deg' : round(j_err, 4),
            'ee_error_mm'     : round(ee_err, 3),
            'ee_ok'           : ee_err <= EE_TOL_MM,
            'joints_ok'       : j_err <= JT_TOL_D,
        })
    return result


def main(args=None):
    print('\n' + '='*66)
    print('  STEP 57  --  Executor + Tier Summary')
    print('='*66)

    # Load all available tier outputs, then choose the best:
    # 1) If any tier reports tier_resolved=True, use the HIGHEST resolved
    # 2) Otherwise pick the tier with fewest residual collisions (best-effort)
    selected = None
    selected_label = None
    tier_status = {}
    tier_data = {}
    for label, fname in [('tier_1', 's54_tier1.json'),
                          ('tier_2', 's55_tier2.json'),
                          ('tier_3', 's56_tier3.json')]:
        if os.path.exists(fname):
            with open(fname) as fh: d = json.load(fh)
            tier_data[label] = d
            fv = d.get('final_verification', {})
            tier_status[label] = {
                'resolved': d.get('tier_resolved', False),
                'next_tier': d.get('next_tier', '?'),
                'method': d.get('method', '?'),
                'residual_coll': int(fv.get('collisions_after_resample', -1)),
            }

    # Pick the best resolved tier (highest index that resolved)
    for label in ['tier_3', 'tier_2', 'tier_1']:
        if label in tier_status and tier_status[label]['resolved']:
            selected = tier_data[label]
            selected_label = label
            break

    # If none resolved, pick the tier with fewest residual collisions
    if selected is None:
        candidates = [(label, info['residual_coll'])
                       for label, info in tier_status.items()
                       if info['residual_coll'] >= 0]
        if candidates:
            best_label, best_coll = min(candidates, key=lambda x: x[1])
            # Only use it if residual is "small enough" to not crash pre-flight
            # (we let pre-flight decide -- here we just nominate best)
            selected = tier_data[best_label]
            selected_label = f'{best_label} (best-effort, {best_coll} residual)'
            print(f'\n  No tier fully resolved -- nominating {best_label} '
                  f'with {best_coll} residual collisions as best-effort')

    # Print tier summary table
    print(f'\n  TIER SUMMARY')
    print(f'  {"-"*60}')
    for label in ['tier_1', 'tier_2', 'tier_3']:
        if label in tier_status:
            ts = tier_status[label]
            icon = '✓' if ts['resolved'] else '✗'
            print(f'  {label}  {icon}  resolved={ts["resolved"]}  '
                  f'method={ts["method"]}  next={ts["next_tier"]}')
        else:
            print(f'  {label}  -  not run')
    print(f'  {"-"*60}')
    print(f'  SELECTED FOR EXECUTION: {selected_label or "NONE"}')

    if selected is None:
        print(f'\n  ✗ NO TIER RESOLVED -- cannot execute.')
        print(f'  Each tier produced infeasible output. This case is')
        print(f'  fundamentally unresolvable by the hierarchical pipeline.')
        out = {'success': False, 'mode': 'all_tiers_failed',
                'tier_status': tier_status,
                'final_joints': {}, 'error': 'all_tiers_failed'}
        with open('s57_execution.json', 'w') as fh: json.dump(out, fh, indent=2)
        sys.exit(1)

    arm_names = selected.get('arm_names', ARM_NAMES)
    duration  = float(selected['duration'])
    bases     = {n: np.array(ROBOT_BASES.get(n, [0, 0, 0])) for n in arm_names}
    arm_pos = {}
    for name in arm_names:
        pos = np.array(selected[name]['trajectory']['positions'], dtype=float)
        nout = max(2, int(round(duration * RATE_HZ)))
        sin = np.linspace(0, 1, len(pos)); sout = np.linspace(0, 1, nout)
        r = np.zeros((nout, NDOF))
        for j in range(NDOF): r[:, j] = np.interp(sout, sin, pos[:, j])
        arm_pos[name] = np.clip(r, POS_LIM[:, 0], POS_LIM[:, 1])
    max_steps = max(len(arm_pos[n]) for n in arm_names)
    print(f'\n  Arms: {arm_names}  dur={duration:.2f}s  steps={max_steps}')

    print('\n  Pre-flight check...')
    issues = preflight(arm_names, arm_pos, bases)
    if issues:
        print(f'  FAIL: {issues}'); sys.exit(1)
    print('  OK Passed')

    if _ROS_OK:
        rclpy.init(args=args); node = ExecNode(arm_names)
        try: exec_res = node.execute(arm_pos, duration)
        except KeyboardInterrupt:
            exec_res = {'success': False, 'mode': 'interrupted',
                         'steps_sent': 0, 'wall_time_s': None,
                         'final_joints': {n: None for n in arm_names},
                         'error': 'KeyboardInterrupt'}
        finally:
            try: node.destroy_node()
            except: pass
            rclpy.shutdown()
    else:
        exec_res = dry_run(arm_names, arm_pos, duration)

    verif = {n: verify_arm(n, selected[n]['metadata'],
                            exec_res['final_joints'].get(n)) for n in arm_names}
    report = {
        'arm_names'       : arm_names,
        'duration_s'      : duration,
        'execution'       : exec_res,
        'verification'    : verif,
        'pipeline'        : 'hierarchical_5x',
        'tier_selected'   : selected_label,
        'tier_status'     : tier_status,
    }

    print(f'\n  {"="*66}')
    for name in arm_names:
        v = verif[name]
        print(f'  [{name}]  j_err={v.get("joint_error_deg","?")} deg  '
              f'ee_err={v.get("ee_error_mm","?")} mm  '
              f'ee_ok={v.get("ee_ok","?")}')

    with open('s57_execution.json', 'w') as fh: json.dump(report, fh, indent=2)
    print(f'\n  Saved: s57_execution.json\n')


if __name__ == '__main__': main()