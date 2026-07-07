#!/usr/bin/env python3
"""
step_83.py  —  Execute B-Spline Trajectory in Gazebo (Visualisation Only)
==========================================================================
Input  : step82_trajectories.json
Output : step83_report.json

Executes both arms in lock-step at 100 Hz via ROS2.
No collision checking. No Kuramoto. Pure trajectory execution.

DH -> Gazebo joint reordering applied on every command:
  DH order:     [j1, j2, j3, j4, j5, j6]
  Gazebo order: [j1, j2, j4, j5, j3, j6]
  DH_TO_GZ = [0, 1, 3, 4, 2, 5]

Run:
    ros2 run dual_arm_sync step_83
"""

import json, os, sys, time
from typing import Dict, List, Optional
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float64MultiArray
    from sensor_msgs.msg import JointState
    _ROS_OK = True
except ImportError:
    _ROS_OK = False
    print('[step_83] No ROS2 — dry-run mode')

# ─────────────────────────────────────────────────────────────────────────────
# ROBOT CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

_PI   = np.pi
_PI_2 = np.pi / 2.0
L1, L2, L3, L4 = 0.1525, 0.6200, 0.5590, 0.1210
A = 0.0345

DH = np.array([
    [0.0,    0.0,  0.0,    L1],
    [-_PI_2, 0.0, -_PI_2,  A ],
    [0.0,    L2,   _PI_2,  0.0],
    [_PI_2,  0.0,  0.0,    L3],
    [-_PI_2, 0.0,  0.0,    0.0],
    [_PI_2,  0.0,  0.0,    L4],
], dtype=float)

POS_LIM = np.array([
    [-2*_PI,  2*_PI ], [-1.6493, 1.6493], [-2.7925, 2.7925],
    [-2*_PI,  2*_PI ], [-2*_PI,  2*_PI ], [-2*_PI,  2*_PI ],
], dtype=float)

NDOF        = 6
RATE_HZ     = 100.0
HOLD_S      = 1.5     # hold final pose this many seconds after trajectory ends
SETTLE_S    = 2.0     # wait up to this long for arms to settle before reading final JS
CTRL_TOPIC  = '/{arm}/gz/dsr_position_controller/commands'
EE_TOL_MM   = 25.0    # mm — acceptable EE final error
JOINT_TOL   = 0.05    # rad — acceptable final joint error

ROBOT_BASES: Dict[str, np.ndarray] = {
    'dsr01': np.array([0.0,  0.5, 0.0]),
    'dsr02': np.array([0.0, -0.5, 0.0]),
}

# DH order -> Gazebo controller order
# DH:     j1 j2 j3 j4 j5 j6
# Gazebo: j1 j2 j4 j5 j3 j6
DH_TO_GZ = [0, 1, 3, 4, 2, 5]

# Gazebo order -> DH order (for reading joint states back)
GZ_TO_DH = [0, 1, 4, 2, 3, 5]


# ─────────────────────────────────────────────────────────────────────────────
# FK
# ─────────────────────────────────────────────────────────────────────────────

def fk_pos(q: np.ndarray, base: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    for i in range(NDOF):
        al, a, to, d = DH[i]
        th = q[i] + to
        ct, st = np.cos(th), np.sin(th)
        ca, sa = np.cos(al),  np.sin(al)
        T = T @ np.array([
            [ct,    -st,    0.,  a    ],
            [st*ca,  ct*ca, -sa, -sa*d],
            [st*sa,  ct*sa,  ca,  ca*d],
            [0.,     0.,    0.,  1.   ],
        ])
    return T[:3, 3] + base


# ─────────────────────────────────────────────────────────────────────────────
# ROS2 EXECUTOR NODE
# ─────────────────────────────────────────────────────────────────────────────

if _ROS_OK:
    class ExecutorNode(Node):
        def __init__(self, arm_names: List[str]):
            super().__init__('step_83_executor')
            self._names = arm_names
            self._pubs  = {
                n: self.create_publisher(
                    Float64MultiArray,
                    CTRL_TOPIC.format(arm=n), 10)
                for n in arm_names
            }
            self._cur_q: Dict[str, Optional[np.ndarray]] = {
                n: None for n in arm_names
            }
            for name in arm_names:
                for topic in (f'/{name}/gz/joint_states',
                              f'/{name}/joint_states'):
                    self.create_subscription(
                        JointState, topic,
                        lambda msg, n=name: self._cb(msg, n), 10)

        def _cb(self, msg: JointState, name: str):
            if len(msg.position) < NDOF:
                return
            jmap = {n: i for i, n in enumerate(msg.name)}
            keys = [f'joint_{k}' for k in range(1, NDOF + 1)]
            if all(k in jmap for k in keys):
                # Reading by joint NAME gives DH order directly - no reorder needed
                q_dh = np.array([msg.position[jmap[k]] for k in keys])
            else:
                # Fallback: positional read - must reorder Gazebo->DH
                q_dh = np.array(msg.position[:NDOF])[GZ_TO_DH]
            self._cur_q[name] = q_dh.astype(float)

        def read_joints(self, name: str) -> Optional[np.ndarray]:
            rclpy.spin_once(self, timeout_sec=0.05)
            return self._cur_q[name]

        def publish_hold(self, arm_pos: Dict[str, np.ndarray]):
            """Publish final pose for all arms (used during hold phase)."""
            msg = Float64MultiArray()
            for name in self._names:
                q_dh     = arm_pos[name][-1]
                msg.data = [float(v) for v in q_dh[DH_TO_GZ]]
                self._pubs[name].publish(msg)

        def execute(self,
                    arm_pos: Dict[str, np.ndarray],
                    duration: float) -> Dict:
            dt_ns   = int(1e9 / RATE_HZ)
            n_steps = max(len(arm_pos[n]) for n in self._names)
            msg     = Float64MultiArray()

            self.get_logger().info(
                f'step_83: executing {n_steps} steps x {len(self._names)} arms'
                f' at {RATE_HZ:.0f} Hz  ({duration:.2f}s)')

            t0_wall = time.monotonic()

            # ── Trajectory ──────────────────────────────────────────────────
            for k in range(n_steps):
                t0 = time.monotonic_ns()

                # Publish all arms in same 10ms window — lock-step
                for name in self._names:
                    idx      = min(k, len(arm_pos[name]) - 1)
                    q_dh     = arm_pos[name][idx]
                    msg.data = [float(v) for v in q_dh[DH_TO_GZ]]
                    self._pubs[name].publish(msg)

                rclpy.spin_once(self, timeout_sec=0.)

                # Busy-wait remainder of 10ms slot
                remain = dt_ns - (time.monotonic_ns() - t0)
                if remain > 0:
                    time.sleep(remain * 1e-9)

                # Progress print every 10%
                if k % max(1, n_steps // 10) == 0:
                    pct = int(100 * k / n_steps)
                    self.get_logger().info(f'  {pct:3d}%  step {k}/{n_steps}')

            # ── Hold final pose ─────────────────────────────────────────────
            hold_steps = int(HOLD_S * RATE_HZ)
            for _ in range(hold_steps):
                t0 = time.monotonic_ns()
                self.publish_hold(arm_pos)
                rclpy.spin_once(self, timeout_sec=0.)
                remain = dt_ns - (time.monotonic_ns() - t0)
                if remain > 0:
                    time.sleep(remain * 1e-9)

            wall_time = time.monotonic() - t0_wall

            # ── Read final joint states ─────────────────────────────────────
            # Spin briefly to flush any queued messages
            t_settle = time.time()
            final_q: Dict[str, Optional[np.ndarray]] = {n: None for n in self._names}
            while time.time() - t_settle < SETTLE_S:
                rclpy.spin_once(self, timeout_sec=0.05)
                if all(self._cur_q[n] is not None for n in self._names):
                    for n in self._names:
                        final_q[n] = self._cur_q[n].copy()
                    break

            self.get_logger().info(f'  100%  done  wall={wall_time:.2f}s')

            return {
                'success'    : True,
                'mode'       : 'ros2',
                'steps_sent' : int(n_steps),
                'wall_time_s': round(wall_time, 3),
                'final_joints': {
                    n: final_q[n].tolist() if final_q[n] is not None else None
                    for n in self._names
                },
                'error': None,
            }


# ─────────────────────────────────────────────────────────────────────────────
# DRY RUN
# ─────────────────────────────────────────────────────────────────────────────

def dry_run(arm_names: List[str],
            arm_pos: Dict[str, np.ndarray],
            duration: float) -> Dict:
    n_steps   = max(len(arm_pos[n]) for n in arm_names)
    milestone = max(1, n_steps // 10)
    print(f'\n  DRY-RUN: {n_steps} steps  ({duration:.2f}s  @{RATE_HZ:.0f}Hz)')
    for k in range(0, n_steps + 1, milestone):
        k = min(k, n_steps - 1)
        pct = int(100 * k / n_steps)
        for name in arm_names:
            q = arm_pos[name][min(k, len(arm_pos[name]) - 1)]
            print(f'  {pct:3d}%  [{name}]  '
                  f'[{", ".join(f"{np.degrees(v):.1f}" for v in q)}] deg')
    print('  100%  done')
    return {
        'success'    : True,
        'mode'       : 'dry_run',
        'steps_sent' : int(n_steps),
        'wall_time_s': None,
        'final_joints': {n: arm_pos[n][-1].tolist() for n in arm_names},
        'error'      : None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# RESAMPLE
# ─────────────────────────────────────────────────────────────────────────────

def resample(pos: np.ndarray, duration: float) -> np.ndarray:
    n_out = max(2, int(round(duration * RATE_HZ)))
    if len(pos) == n_out:
        return pos
    s_in  = np.linspace(0, 1, len(pos))
    s_out = np.linspace(0, 1, n_out)
    out   = np.zeros((n_out, NDOF))
    for j in range(NDOF):
        out[:, j] = np.interp(s_out, s_in, pos[:, j])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def verify_arm(name: str,
               traj_data: Dict,
               final_q_gz: Optional[List[float]]) -> Dict:
    """
    Compare final Gazebo joint state against planned target_joints.
    final_q_gz: joint positions read from Gazebo after execution (DH order).
    """
    meta     = traj_data.get('metadata', {})
    base     = np.array(ROBOT_BASES.get(name, [0, 0, 0]))
    end_q    = np.array(meta.get('end_joints', [0]*NDOF), dtype=float)
    tgt_ee   = np.array(meta.get('target_ee_world', fk_pos(end_q, base).tolist()))
    pos_arr  = np.array(traj_data['trajectory']['positions'], dtype=float)

    # EE from planned last trajectory point
    ee_planned = fk_pos(pos_arr[-1], base)
    ee_target  = tgt_ee

    # joint error: planned last step vs planned target
    j_err_planned = float(np.max(np.abs(np.degrees(pos_arr[-1] - end_q))))

    result = {
        'end_joints_planned_deg'   : np.degrees(end_q).tolist(),
        'last_traj_joints_deg'     : np.degrees(pos_arr[-1]).tolist(),
        'joint_error_planned_deg'  : round(j_err_planned, 4),
        'ee_planned_m'             : ee_planned.tolist(),
        'ee_target_m'              : ee_target.tolist(),
        'ee_error_planned_mm'      : round(float(np.linalg.norm(ee_planned - ee_target) * 1000), 3),
    }

    # If we have actual Gazebo reading
    if final_q_gz is not None:
        fq = np.array(final_q_gz, dtype=float)
        ee_actual  = fk_pos(fq, base)
        j_err_act  = float(np.max(np.abs(np.degrees(fq - end_q))))
        ee_err_act = float(np.linalg.norm(ee_actual - ee_target) * 1000)
        result.update({
            'final_joints_gazebo_deg'  : np.degrees(fq).tolist(),
            'joint_error_actual_deg'   : round(j_err_act, 4),
            'ee_actual_m'              : ee_actual.tolist(),
            'ee_error_actual_mm'       : round(ee_err_act, 3),
            'ee_ok'                    : bool(ee_err_act <= EE_TOL_MM),
            'joints_ok'                : bool(j_err_act <= np.degrees(JOINT_TOL)),
        })
    else:
        result.update({
            'final_joints_gazebo_deg'  : None,
            'joint_error_actual_deg'   : None,
            'ee_error_actual_mm'       : None,
            'ee_ok'                    : None,
            'joints_ok'                : None,
        })

    return result


# ─────────────────────────────────────────────────────────────────────────────
# REPORT PRINTER
# ─────────────────────────────────────────────────────────────────────────────

def print_report(report: Dict, arm_names: List[str]):
    bar  = '=' * 64
    thin = '-' * 64
    print(f'\n{bar}')
    print('  STEP 83  —  EXECUTION REPORT')
    print(f'{bar}')

    ex = report.get('execution', {})
    mode = ex.get('mode', '?')
    ok   = ex.get('success', False)
    print(f'\n  Execution  : {"OK" if ok else "FAIL"}  ({mode})')
    print(f'  Steps sent : {ex.get("steps_sent", 0)}')
    if ex.get('wall_time_s'):
        print(f'  Wall time  : {ex["wall_time_s"]:.2f}s')
    if ex.get('error'):
        print(f'  Error      : {ex["error"]}')

    print(f'\n  PER-ARM RESULTS\n  {thin}')
    for name in arm_names:
        v = report.get('verification', {}).get(name, {})
        print(f'\n  [{name.upper()}]')
        print(f'    Planned target  (deg) : '
              f'[{", ".join(f"{x:.2f}" for x in v.get("end_joints_planned_deg", []))}]')
        print(f'    Last traj point (deg) : '
              f'[{", ".join(f"{x:.2f}" for x in v.get("last_traj_joints_deg", []))}]')
        print(f'    Joint err (planned)   : {v.get("joint_error_planned_deg", "?"):.3f} deg')
        print(f'    EE err (planned)      : {v.get("ee_error_planned_mm", "?"):.2f} mm')

        fq_deg = v.get('final_joints_gazebo_deg')
        if fq_deg is not None:
            j_act = v.get('joint_error_actual_deg', '?')
            e_act = v.get('ee_error_actual_mm', '?')
            j_ok  = 'OK' if v.get('joints_ok') else 'WARN'
            e_ok  = 'OK' if v.get('ee_ok')     else 'WARN'
            print(f'    Gazebo final    (deg) : '
                  f'[{", ".join(f"{x:.2f}" for x in fq_deg)}]')
            print(f'    Joint err (actual)    : {j_act:.3f} deg  [{j_ok}]')
            print(f'    EE err (actual)       : {e_act:.2f} mm  [{e_ok}]')
        else:
            print('    Gazebo final joints   : (not available — dry-run)')

    print(f'\n{bar}')
    all_ok = all(
        report.get('verification', {}).get(n, {}).get('ee_ok', True)
        for n in arm_names
    )
    if mode == 'dry_run':
        print('  DRY-RUN — trajectory generated, not sent to hardware')
    elif all_ok:
        print('  ALL ARMS REACHED TARGET  — execution complete')
    else:
        print('  WARN — one or more arms outside EE tolerance')
    print(f'{bar}\n')


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(args=None):
    print('\n' + '=' * 64)
    print('  STEP 83  —  Gazebo Execution (no collision / no Kuramoto)')
    print('=' * 64)

    if not os.path.exists('step82_trajectories.json'):
        print('\n  step82_trajectories.json not found — run step_82 first')
        sys.exit(1)

    with open('step82_trajectories.json') as fh:
        data = json.load(fh)

    arm_names = data.get('arm_names', sorted(
        [k for k in data if k.startswith('dsr')]))
    if not arm_names:
        print('\n  No arm data found in step82_trajectories.json')
        sys.exit(1)

    duration = max(float(data[n]['metadata']['duration']) for n in arm_names)

    print(f'\n  Arms     : {arm_names}')
    print(f'  Duration : {duration:.2f}s  ({int(duration * RATE_HZ)} steps @ {RATE_HZ:.0f} Hz)')
    print(f'  Hold     : {HOLD_S}s after trajectory ends')

    # Build arm_pos dict — resample to exact 100Hz step count
    arm_pos: Dict[str, np.ndarray] = {}
    for name in arm_names:
        pos = np.array(data[name]['trajectory']['positions'], dtype=float)
        dur = float(data[name]['metadata'].get('duration', duration))
        arm_pos[name] = resample(pos, dur)

    # Align all arms to same step count (longest)
    max_steps = max(len(arm_pos[n]) for n in arm_names)
    for name in arm_names:
        if len(arm_pos[name]) < max_steps:
            s_in  = np.linspace(0, 1, len(arm_pos[name]))
            s_out = np.linspace(0, 1, max_steps)
            r     = np.zeros((max_steps, NDOF))
            for j in range(NDOF):
                r[:, j] = np.interp(s_out, s_in, arm_pos[name][:, j])
            arm_pos[name] = r

    # Show planned targets
    print()
    for name in arm_names:
        end_q = np.array(data[name]['metadata']['end_joints'])
        deg   = [f'{np.degrees(v):.1f}' for v in end_q]
        print(f'  [{name}] target: [{", ".join(deg)}] deg')

    print(f'\n  Executing ...')

    # Execute
    if _ROS_OK:
        rclpy.init(args=args)
        node = ExecutorNode(arm_names)
        try:
            exec_res = node.execute(arm_pos, duration)
        except KeyboardInterrupt:
            exec_res = {
                'success': False, 'mode': 'interrupted',
                'steps_sent': 0, 'wall_time_s': None,
                'final_joints': {n: None for n in arm_names},
                'error': 'KeyboardInterrupt',
            }
        except Exception as e:
            exec_res = {
                'success': False, 'mode': 'error',
                'steps_sent': 0, 'wall_time_s': None,
                'final_joints': {n: None for n in arm_names},
                'error': str(e),
            }
        finally:
            try:
                node.destroy_node()
            except Exception:
                pass
            rclpy.shutdown()
    else:
        exec_res = dry_run(arm_names, arm_pos, duration)

    # Verify each arm
    final_joints = exec_res.get('final_joints', {})
    verification: Dict = {}
    for name in arm_names:
        fq = final_joints.get(name)
        verification[name] = verify_arm(name, data[name], fq)

    report = {
        'arm_names'   : arm_names,
        'duration_s'  : duration,
        'execution'   : exec_res,
        'verification': verification,
    }

    print_report(report, arm_names)

    with open('step83_report.json', 'w') as fh:
        json.dump(report, fh, indent=2)

    kb = os.path.getsize('step83_report.json') / 1024.0
    print(f'  Saved: step83_report.json  ({kb:.1f} KB)\n')


if __name__ == '__main__':
    main()