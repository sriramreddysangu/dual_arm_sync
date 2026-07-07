#!/usr/bin/env python3
"""
step_10.py  --  Synchronized Execution + Final Report
===============================================================================
Input  : synchronized_trajectories.json
Output : execution_report.json

FIXES vs previous version
--------------------------
1. Duration read directly from synchronized_trajectories.json
2. arm_metrics handles missing metadata keys gracefully
3. Per-arm EE error computed against actual planned end_joints EE (not
   missing 'target_world_pos' key)
4. Preflight checks the synchronized (Kuramoto-adjusted) positions
===============================================================================
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
    print('[step_10] no ROS2 -- dry-run mode')

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
    [-2*_PI,  2*_PI ],
    [-1.6493,  1.6493],
    [-2.7925,  2.7925],
    [-2*_PI,  2*_PI ],
    [-2*_PI,  2*_PI ],
    [-2*_PI,  2*_PI ],
], dtype=float)

VEL_LIM = np.array([2.094, 2.094, 3.140, 3.927, 3.927, 3.927])
NDOF    = 6
RATE_HZ = 100.0
CTRL_TOPIC  = '/{arm}/gz/dsr_position_controller/commands'
HOLD_AFTER  = 1.5    # seconds to hold final pose

ROBOT_BASES: Dict[str, np.ndarray] = {
    'dsr01': np.array([0.0,  0.5, 0.0]),
    'dsr02': np.array([0.0, -0.5, 0.0]),
}

LINK_RADII    = np.array([0.10, 0.10, 0.08, 0.08, 0.06, 0.06])
SAFETY_MARGIN = 0.12
EE_VER_TOL_MM = 25.0   # mm -- end-effector position tolerance
JOINT_TOL_RAD = 0.05   # rad -- joint angle tolerance

# Gazebo joint_states order (verified via 'ros2 topic echo'):
#   [joint_1, joint_2, joint_4, joint_5, joint_3, joint_6]
# DH / trajectory arrays use: [j1, j2, j3, j4, j5, j6]
# The dsr_position_controller expects commands in the SAME order as joint_states.
# Reorder DH -> Gazebo before publishing:
#   gz[0]=j1=dh[0]  gz[1]=j2=dh[1]  gz[2]=j4=dh[3]
#   gz[3]=j5=dh[4]  gz[4]=j3=dh[2]  gz[5]=j6=dh[5]
DH_TO_GZ = np.array([0, 1, 3, 4, 2, 5])   # q_gz = q_dh[DH_TO_GZ]

# Gazebo publishes/expects joints in order: [j1, j2, j4, j5, j3, j6]
# DH / trajectory arrays use order:        [j1, j2, j3, j4, j5, j6]
# When SENDING commands reorder DH -> Gazebo:
#   gz[0]=dh[0] gz[1]=dh[1] gz[2]=dh[3] gz[3]=dh[4] gz[4]=dh[2] gz[5]=dh[5]


# ─────────────────────────────────────────────────────────────────────────────
# FK
# ─────────────────────────────────────────────────────────────────────────────

def fk_pos(q, base):
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


def link_origins(q, base):
    T = np.eye(4)
    o = np.zeros((NDOF, 3))
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
        o[i] = T[:3, 3] + base
    return o


def pair_min_dist(qi, bi, qj, bj):
    oi = link_origins(qi, bi)
    oj = link_origins(qj, bj)
    return float(np.min([np.linalg.norm(oi[a] - oj[b])
                          for a in range(NDOF) for b in range(NDOF)]))


def pair_collides(qi, bi, qj, bj):
    oi = link_origins(qi, bi)
    oj = link_origins(qj, bj)
    for a in range(NDOF):
        for b in range(NDOF):
            if np.linalg.norm(oi[a] - oj[b]) < LINK_RADII[a] + LINK_RADII[b] + SAFETY_MARGIN:
                return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# PRE-FLIGHT CHECK
# ─────────────────────────────────────────────────────────────────────────────

def preflight(arm_names, data):
    issues   = []
    warnings = []

    # Basic shape check
    for name in arm_names:
        if name not in data:
            issues.append('{}: missing from file'.format(name)); continue
        pos = np.array(data[name]['trajectory']['positions'], dtype=float)
        if pos.ndim != 2 or pos.shape[1] != NDOF:
            issues.append('{}: bad positions shape {}'.format(name, pos.shape))

    # Inter-arm collision check on the synchronized positions
    pair_results = {}
    valid_names  = [n for n in arm_names if n in data]
    if len(valid_names) >= 2:
        min_n = min(len(np.array(data[n]['trajectory']['positions']))
                    for n in valid_names)
        for i in range(len(valid_names)):
            for j in range(i + 1, len(valid_names)):
                ni, nj = valid_names[i], valid_names[j]
                bi = ROBOT_BASES.get(ni, np.zeros(3))
                bj = ROBOT_BASES.get(nj, np.zeros(3))
                pi = np.array(data[ni]['trajectory']['positions'], dtype=float)
                pj = np.array(data[nj]['trajectory']['positions'], dtype=float)
                K  = min(len(pi), len(pj), min_n)
                nc = 0; dmin = float('inf')
                for k in range(K):
                    d = pair_min_dist(pi[k], bi, pj[k], bj)
                    dmin = min(dmin, d)
                    if pair_collides(pi[k], bi, pj[k], bj):
                        nc += 1
                if nc > 0:
                    issues.append(
                        '{}<->{}: {} collision steps in synchronized trajectory'.format(
                            ni, nj, nc))
                pair_results['{}<->{}'.format(ni, nj)] = {
                    'n_collisions': int(nc),
                    'min_dist_m'  : round(float(dmin), 5),
                    'safe'        : nc == 0,
                }

    return len(issues) == 0, {
        'passed'    : len(issues) == 0,
        'issues'    : issues,
        'warnings'  : warnings,
        'pair_check': pair_results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ARM METRICS
# ─────────────────────────────────────────────────────────────────────────────

def arm_metrics(name, traj):
    pos  = np.array(traj['trajectory']['positions'], dtype=float)
    t    = np.array(traj['trajectory']['time'],      dtype=float)
    meta = traj.get('metadata', {})
    base = ROBOT_BASES.get(name, np.zeros(3))
    dur  = float(t[-1]) if len(t) > 0 else 0.

    # End-effector path
    ee_path  = np.array([fk_pos(pos[k], base) for k in range(len(pos))])
    path_len = float(np.sum(np.linalg.norm(np.diff(ee_path, axis=0), axis=1))) \
               if len(ee_path) > 1 else 0.

    # Velocity / acceleration (finite differences)
    if len(pos) > 1:
        dt    = dur / max(len(pos) - 1, 1)
        vel   = np.gradient(pos, dt, axis=0)
        acc   = np.gradient(vel, dt, axis=0)
        max_v = float(np.max(np.abs(vel)))
        max_a = float(np.max(np.abs(acc)))
    else:
        max_v = max_a = 0.

    # EE accuracy: planned target vs what trajectory actually ends at.
    # end_joints = target the user typed in step_7.
    # pos[-1]    = what synchronized trajectory ends at after Kuramoto/home-CP.
    end_q_list = meta.get('end_joints', None)
    end_q      = np.array(end_q_list, dtype=float) if end_q_list is not None \
                 else pos[-1].copy()
    ee_target  = fk_pos(end_q, base)
    ee_actual  = fk_pos(pos[-1], base)
    ee_err_mm  = float(np.linalg.norm(ee_actual - ee_target) * 1000.0)
    j_err_rad  = float(np.max(np.abs(pos[-1] - end_q)))

    return {
        'duration_s'           : round(dur, 4),
        'n_samples'            : int(len(pos)),
        'ee_path_length_cm'    : round(path_len * 100, 3),
        'max_joint_vel_rad_s'  : round(max_v, 5),
        'max_joint_acc_rad_s2' : round(max_a, 5),
        'ee_target_world_m'    : ee_target.tolist(),
        'ee_actual_world_m'    : ee_actual.tolist(),
        'ee_error_mm'          : round(ee_err_mm, 3),
        'ee_ok'                : bool(ee_err_mm <= EE_VER_TOL_MM),
        'max_joint_error_rad'  : round(j_err_rad, 5),
        'max_joint_error_deg'  : round(float(np.degrees(j_err_rad)), 3),
        'joints_ok'            : bool(j_err_rad <= JOINT_TOL_RAD),
        'refine_iterations'    : int(meta.get('refine_iterations', 0)),
    }


def final_safety(arm_names, data, t_vec):
    result = {}
    valid  = [n for n in arm_names if n in data]
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            ni, nj = valid[i], valid[j]
            bi = ROBOT_BASES.get(ni, np.zeros(3))
            bj = ROBOT_BASES.get(nj, np.zeros(3))
            pi = np.array(data[ni]['trajectory']['positions'], dtype=float)
            pj = np.array(data[nj]['trajectory']['positions'], dtype=float)
            K  = min(len(pi), len(pj), len(t_vec))
            dists = [pair_min_dist(pi[k], bi, pj[k], bj) for k in range(K)]
            nc    = sum(1 for k in range(K) if pair_collides(pi[k], bi, pj[k], bj))
            result['{} <-> {}'.format(ni, nj)] = {
                'min_dist_m'   : round(float(min(dists)), 5) if dists else None,
                'avg_dist_m'   : round(float(np.mean(dists)), 5) if dists else None,
                'n_collisions' : int(nc),
                'collision_free': bool(nc == 0),
                'status'       : 'SAFE' if nc == 0 else 'COLLISION',
            }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# RESAMPLE
# ─────────────────────────────────────────────────────────────────────────────

def resample(pos, duration):
    n_out = max(2, int(round(duration * RATE_HZ)))
    if len(pos) == n_out: return pos
    s_in  = np.linspace(0, 1, len(pos))
    s_out = np.linspace(0, 1, n_out)
    out   = np.zeros((n_out, NDOF))
    for j in range(NDOF):
        out[:, j] = np.interp(s_out, s_in, pos[:, j])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# ROS2 EXECUTOR
# ─────────────────────────────────────────────────────────────────────────────

if _ROS_OK:
    class ExecutorNode(Node):
        def __init__(self, arm_names):
            super().__init__('step_10_executor')
            self._names = arm_names
            self._pubs  = {
                n: self.create_publisher(
                    Float64MultiArray,
                    CTRL_TOPIC.format(arm=n), 10)
                for n in arm_names
            }
            self._cur_q: Dict[str, Optional[np.ndarray]] = {n: None for n in arm_names}
            for name in arm_names:
                for topic in ('/{}/gz/joint_states'.format(name),
                              '/{}/joint_states'.format(name)):
                    self.create_subscription(
                        JointState, topic,
                        lambda msg, n=name: self._cb(msg, n), 10)

        def _cb(self, msg, name):
            if len(msg.position) < NDOF: return
            jmap = {n: i for i, n in enumerate(msg.name)}
            keys = ['joint_{}'.format(k) for k in range(1, NDOF + 1)]
            q = (np.array([msg.position[jmap[k]] for k in keys])
                 if all(k in jmap for k in keys)
                 else np.array(msg.position[:NDOF]))
            self._cur_q[name] = q.astype(float)

        def execute(self, arm_pos, duration):
            dt_ns   = int(1e9 / RATE_HZ)
            n_steps = max(len(arm_pos[n]) for n in self._names)
            self.get_logger().info(
                'Executing {} steps x {} arms at {:.0f}Hz ({:.2f}s)'.format(
                    n_steps, len(self._names), RATE_HZ, duration))
            msg  = Float64MultiArray()
            t0w  = time.monotonic()
            for k in range(n_steps):
                t0 = time.monotonic_ns()
                for name in self._names:
                    idx      = min(k, len(arm_pos[name]) - 1)
                    q_gz     = arm_pos[name][idx][DH_TO_GZ]  # reorder DH -> Gazebo
                    msg.data = [float(v) for v in q_gz]
                    self._pubs[name].publish(msg)
                rclpy.spin_once(self, timeout_sec=0.)
                rem = dt_ns - (time.monotonic_ns() - t0)
                if rem > 0: time.sleep(rem * 1e-9)

            # Hold final pose
            hold_steps = int(HOLD_AFTER * RATE_HZ)
            for _ in range(hold_steps):
                t0 = time.monotonic_ns()
                for name in self._names:
                    q_gz_hold = arm_pos[name][-1][DH_TO_GZ]  # reorder DH -> Gazebo
                    msg.data  = [float(v) for v in q_gz_hold]
                    self._pubs[name].publish(msg)
                rclpy.spin_once(self, timeout_sec=0.)
                rem = dt_ns - (time.monotonic_ns() - t0)
                if rem > 0: time.sleep(rem * 1e-9)

            wall = time.monotonic() - t0w
            fq   = {n: self._cur_q[n].tolist()
                    if self._cur_q[n] is not None else None
                    for n in self._names}
            return {
                'success'      : True,
                'mode'         : 'ros2',
                'steps_sent'   : int(n_steps),
                'wall_time_s'  : round(wall, 3),
                'final_joints' : fq,
                'error'        : None,
            }


def dry_run(arm_names, arm_pos, duration):
    n_steps  = max(len(arm_pos[n]) for n in arm_names)
    milestone = max(1, n_steps // 10)
    print('\n  DRY-RUN: {} steps  ({:.2f}s  @{:.0f}Hz)'.format(
        n_steps, duration, RATE_HZ))
    for k in range(0, n_steps, milestone):
        pct = int(100 * k / n_steps)
        q0  = arm_pos[arm_names[0]][min(k, len(arm_pos[arm_names[0]]) - 1)]
        print('    {:3d}%  {}: [{}]'.format(
            pct, arm_names[0],
            ', '.join('{:.3f}'.format(v) for v in q0)))
    print('    100%  done')
    return {
        'success'    : True,
        'mode'       : 'dry_run',
        'steps_sent' : int(n_steps),
        'wall_time_s': None,
        'final_joints': {n: arm_pos[n][-1].tolist() for n in arm_names},
        'error'      : None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PRINT REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_report(report):
    bar  = '=' * 68
    thin = '-' * 68
    print('\n{}\n  STEP 10  --  FINAL EXECUTION REPORT\n{}'.format(bar, bar))

    pf = report['preflight']
    print('\n  PRE-FLIGHT  :  {}'.format('OK PASSED' if pf['passed'] else 'FAIL'))
    for iss in pf.get('issues', []):   print('    FAIL  {}'.format(iss))
    for w   in pf.get('warnings', []): print('    WARN  {}'.format(w))

    print('\n  PER-ARM METRICS\n  ' + thin)
    for name, m in report.get('per_arm', {}).items():
        oe = 'OK' if m['ee_ok']     else 'FAIL'
        oj = 'OK' if m['joints_ok'] else 'FAIL'
        print('\n  -- {} ----------------------------------------'.format(name.upper()))
        print('    Duration          : {:.3f}s'.format(m['duration_s']))
        print('    Samples           : {}'.format(m['n_samples']))
        print('    EE path length    : {:.2f}cm'.format(m['ee_path_length_cm']))
        print('    Max joint vel     : {:.4f} rad/s'.format(m['max_joint_vel_rad_s']))
        print('    EE sync error     : {:.2f}mm  [{}]'.format(m['ee_error_mm'], oe))
        print('    Max joint error   : {:.3f} deg  [{}]'.format(m['max_joint_error_deg'], oj))
        print('    EE target  (world): {}'.format(
            [round(v, 4) for v in m['ee_target_world_m']]))
        print('    EE actual  (world): {}'.format(
            [round(v, 4) for v in m['ee_actual_world_m']]))
        print('    Refine iterations : {}'.format(m['refine_iterations']))

    print('\n  INTER-ARM SAFETY  (synchronized)\n  ' + thin)
    safety = report.get('inter_arm_safety', {})
    if not safety:
        print('  (single-arm mode -- no inter-arm check)')
    for pair, s in safety.items():
        icon = 'OK' if s['collision_free'] else 'FAIL'
        md   = s['min_dist_m']; av = s['avg_dist_m']
        print('  [{}]  {:<22}  min={:.2f}cm  avg={:.2f}cm  coll={}  {}'.format(
            icon, pair,
            md * 100 if md else 0,
            av * 100 if av else 0,
            s['n_collisions'], s['status']))

    ex = report.get('execution', {})
    print('\n  EXECUTION  :  {}'.format('OK SUCCESS' if ex.get('success') else 'FAIL'))
    print('    Mode      : {}'.format(ex.get('mode')))
    print('    Steps     : {}'.format(ex.get('steps_sent', 0)))
    if ex.get('wall_time_s'):
        print('    Wall time : {:.2f}s'.format(ex['wall_time_s']))
    if ex.get('error'):
        print('    Error     : {}'.format(ex['error']))

    sr = report.get('synchronisation_report', {})
    if sr.get('pair_reports'):
        print('\n  KURAMOTO SUMMARY\n  ' + thin)
        for pk, pr in sr['pair_reports'].items():
            icon = 'OK' if pr.get('collision_free') else 'FAIL'
            md   = pr.get('min_dist_m', 0.0)
            crit = pr.get('critical', 0)
            print('  [{}]  {:<22}  min={:.2f}cm  crit={}'.format(
                icon, pk, md * 100, crit))

    any_c = any(not s.get('collision_free', True)
                for s in report.get('inter_arm_safety', {}).values())
    print('\n' + bar)
    if not pf['passed']:
        print('  FAIL: PRE-FLIGHT FAILED -- hardware not commanded')
    elif any_c:
        print('  WARN: RESIDUAL COLLISIONS -> increase MAX_REFINE in step_9')
    else:
        print('  OK: ALL SAFE -- execution complete')
    print(bar + '\n')


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(args=None):
    print('\n' + '=' * 68)
    print('  STEP 10  --  Synchronized Execution + Final Report')
    print('=' * 68)

    if not os.path.exists('synchronized_trajectories.json'):
        print('\n  FAIL  synchronized_trajectories.json not found -- run step_9 first')
        sys.exit(1)

    with open('synchronized_trajectories.json') as fh:
        data = json.load(fh)

    arm_names = sorted([k for k in data if k.startswith('dsr')])
    if not arm_names:
        print('\n  FAIL  No arm data in synchronized_trajectories.json')
        sys.exit(1)

    # Read duration from synchronized_trajectories.json (the correct source)
    duration = max(float(data[n]['metadata']['duration']) for n in arm_names)

    print('\n  Arms     : {}'.format(arm_names))
    print('  Duration : {:.2f}s  (from synchronized_trajectories.json)'.format(duration))

    # Pre-flight
    print('\n  Pre-flight check ...')
    pf_ok, pf_rep = preflight(arm_names, data)
    print('  {}'.format('OK Passed' if pf_ok else 'FAIL'))
    for iss in pf_rep['issues']: print('    {}'.format(iss))

    if not pf_ok:
        report = {
            'preflight'      : pf_rep,
            'per_arm'        : {},
            'inter_arm_safety': {},
            'execution'      : {
                'success': False, 'mode': 'aborted',
                'steps_sent': 0, 'error': 'pre-flight failed',
            },
        }
        print_report(report)
        with open('execution_report.json', 'w') as fh:
            json.dump(report, fh, indent=2)
        sys.exit(1)

    # Resample and align all arms to the same step count
    arm_pos: Dict[str, np.ndarray] = {}
    for name in arm_names:
        pos = np.array(data[name]['trajectory']['positions'], dtype=float)
        dur = float(data[name]['metadata'].get('duration', duration))
        arm_pos[name] = resample(pos, dur)

    max_steps = max(len(arm_pos[n]) for n in arm_names)
    for name in arm_names:
        if len(arm_pos[name]) < max_steps:
            s_in  = np.linspace(0, 1, len(arm_pos[name]))
            s_out = np.linspace(0, 1, max_steps)
            r     = np.zeros((max_steps, NDOF))
            for j in range(NDOF):
                r[:, j] = np.interp(s_out, s_in, arm_pos[name][:, j])
            arm_pos[name] = r

    t_vec = np.linspace(0., duration, max_steps)

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
                'final_joints': {}, 'error': 'KeyboardInterrupt',
            }
        except Exception as e:
            exec_res = {
                'success': False, 'mode': 'error',
                'steps_sent': 0, 'wall_time_s': None,
                'final_joints': {}, 'error': str(e),
            }
        finally:
            try: node.destroy_node()
            except: pass
            rclpy.shutdown()
    else:
        exec_res = dry_run(arm_names, arm_pos, duration)

    per_arm = {n: arm_metrics(n, data[n]) for n in arm_names}
    safety  = final_safety(arm_names, data, t_vec)

    report = {
        'preflight'              : pf_rep,
        'per_arm'                : per_arm,
        'inter_arm_safety'       : safety,
        'execution'              : exec_res,
        'synchronisation_report' : data.get('synchronisation_report', {}),
        'refinement_history'     : data.get('refinement_history', []),
        'parameters'             : data.get('parameters', {}),
    }

    print_report(report)

    with open('execution_report.json', 'w') as fh:
        json.dump(report, fh, indent=2)

    kb = os.path.getsize('execution_report.json') / 1024.0
    print('  OK  Saved: execution_report.json  ({:.1f} KB)\n'.format(kb))


if __name__ == '__main__':
    main()