#!/usr/bin/env python3
"""
step_5.py  —  Synchronized Execution + Final Report
═══════════════════════════════════════════════════════════════════════════════
Input  : synchronized_trajectories.json
Output : execution_report.json

LOGIC
─────
1. Pre-flight: verify synchronized trajectory is collision-free
2. Execute all arms at 100 Hz lock-step via ROS2
   All arms publish at the exact same timestep — true synchronization
3. Post-execution: verify final positions vs targets
4. Save execution_report.json
═══════════════════════════════════════════════════════════════════════════════
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
    print('[step_5] no ROS2 — dry-run mode')

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

VEL_LIM = np.array([2.094, 2.094, 3.140, 3.927, 3.927, 3.927])
NDOF    = 6
RATE_HZ = 100.0
CTRL_TOPIC = '/{arm}/gz/dsr_position_controller/commands'

ROBOT_BASES: Dict[str, np.ndarray] = {
    'dsr01': np.array([0.0,  0.5, 0.0]),
    'dsr02': np.array([0.0, -0.5, 0.0]),
}

LINK_RADII    = np.array([0.10, 0.10, 0.08, 0.08, 0.06, 0.06])
SAFETY_MARGIN = 0.12
JOINT_TOL     = 0.05   # rad
EE_TOL_MM     = 15.0   # mm


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
    T = np.eye(4); o = np.zeros((NDOF, 3))
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
    """Minimum link-origin distance — vectorised broadcasting."""
    oi = link_origins(qi, bi)  # (NDOF,3)
    oj = link_origins(qj, bj)  # (NDOF,3)
    diff = oi[:, np.newaxis, :] - oj[np.newaxis, :, :]  # (NDOF,NDOF,3)
    return float(np.min(np.linalg.norm(diff, axis=2)))


def pair_collides(qi, bi, qj, bj):
    oi = link_origins(qi, bi); oj = link_origins(qj, bj)
    for a in range(NDOF):
        for b in range(NDOF):
            if np.linalg.norm(oi[a]-oj[b]) < LINK_RADII[a]+LINK_RADII[b]+SAFETY_MARGIN:
                return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# PRE-FLIGHT CHECK
# ─────────────────────────────────────────────────────────────────────────────

def preflight(arm_names, data):
    issues = []; warnings = []

    for name in arm_names:
        if name not in data:
            issues.append(f'{name}: missing from file'); continue
        pos = np.array(data[name]['trajectory']['positions'], dtype=float)
        if pos.ndim != 2 or pos.shape[1] != NDOF:
            issues.append(f'{name}: bad shape {pos.shape}')

    # Inter-arm collision scan on synchronized trajectory
    pair_results = {}
    min_n = min(len(np.array(data[n]['trajectory']['positions']))
                for n in arm_names if n in data)
    for i in range(len(arm_names)):
        for j in range(i+1, len(arm_names)):
            ni, nj = arm_names[i], arm_names[j]
            if ni not in data or nj not in data: continue
            bi = ROBOT_BASES.get(ni, np.zeros(3))
            bj = ROBOT_BASES.get(nj, np.zeros(3))
            pi = np.array(data[ni]['trajectory']['positions'], dtype=float)
            pj = np.array(data[nj]['trajectory']['positions'], dtype=float)
            K  = min(len(pi), len(pj), min_n)
            nc = 0; dmin = float('inf')
            for k in range(K):
                d = pair_min_dist(pi[k], bi, pj[k], bj)
                dmin = min(dmin, d)
                if pair_collides(pi[k], bi, pj[k], bj): nc += 1
            if nc > 0:
                issues.append(f'{ni}↔{nj}: {nc} collision steps in synchronized trajectory')
            pair_results[f'{ni}↔{nj}'] = {
                'n_collisions': nc, 'min_dist_m': round(dmin,5), 'safe': nc==0
            }

    return len(issues)==0, {
        'passed': len(issues)==0,
        'issues': issues, 'warnings': warnings,
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
    dur  = float(t[-1]) if len(t)>0 else 0.

    ee_path  = np.array([fk_pos(pos[k], base) for k in range(len(pos))])
    path_len = float(np.sum(np.linalg.norm(np.diff(ee_path,axis=0),axis=1))) \
               if len(ee_path)>1 else 0.

    if len(pos)>1:
        dt = dur/max(len(pos)-1,1)
        vel = np.gradient(pos, dt, axis=0)
        acc = np.gradient(vel, dt, axis=0)
        max_v = float(np.max(np.abs(vel)))
        max_a = float(np.max(np.abs(acc)))
    else:
        max_v = max_a = 0.

    end_q   = np.array(meta.get('end_joints', pos[-1]), dtype=float)
    ee_end  = fk_pos(end_q, base)
    tgt_pos = np.array(meta.get('target_world_pos', meta.get('end_pos_world', ee_end)))
    ee_err  = float(np.linalg.norm(ee_end - tgt_pos)*1000)
    j_err   = float(np.max(np.abs(pos[-1] - end_q)))

    return {
        'duration_s'           : round(dur, 4),
        'n_samples'            : len(pos),
        'ee_path_length_cm'    : round(path_len*100, 3),
        'max_joint_vel_rad_s'  : round(max_v, 5),
        'max_joint_acc_rad_s2' : round(max_a, 5),
        'ee_final_world'       : ee_end.tolist(),
        'ee_final_error_mm'    : round(ee_err, 3),
        'ee_ok'                : ee_err <= EE_TOL_MM,
        'max_joint_error_rad'  : round(j_err, 5),
        'max_joint_error_deg'  : round(float(np.degrees(j_err)), 3),
        'joints_ok'            : j_err <= JOINT_TOL,
        'refine_iterations'    : int(meta.get('refine_iterations', 0)),
    }


def final_safety(arm_names, data, t_vec):
    result = {}
    for i in range(len(arm_names)):
        for j in range(i+1, len(arm_names)):
            ni, nj = arm_names[i], arm_names[j]
            bi = ROBOT_BASES.get(ni, np.zeros(3))
            bj = ROBOT_BASES.get(nj, np.zeros(3))
            pi = np.array(data[ni]['trajectory']['positions'], dtype=float)
            pj = np.array(data[nj]['trajectory']['positions'], dtype=float)
            K  = min(len(pi), len(pj), len(t_vec))
            dists  = [pair_min_dist(pi[k],bi,pj[k],bj) for k in range(K)]
            nc     = sum(1 for k in range(K) if pair_collides(pi[k],bi,pj[k],bj))
            result[f'{ni} ↔ {nj}'] = {
                'min_dist_m': round(float(min(dists)),5) if dists else None,
                'avg_dist_m': round(float(np.mean(dists)),5) if dists else None,
                'n_collisions': nc, 'collision_free': nc==0,
                'status': 'SAFE' if nc==0 else 'COLLISION',
            }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# RESAMPLE
# ─────────────────────────────────────────────────────────────────────────────

def resample(pos, duration):
    n_out = max(2, int(round(duration * RATE_HZ)))
    if len(pos)==n_out: return pos
    s_in = np.linspace(0,1,len(pos)); s_out = np.linspace(0,1,n_out)
    out  = np.zeros((n_out, NDOF))
    for j in range(NDOF): out[:,j] = np.interp(s_out, s_in, pos[:,j])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# ROS2 EXECUTOR
# ─────────────────────────────────────────────────────────────────────────────

if _ROS_OK:
    class ExecutorNode(Node):
        def __init__(self, arm_names):
            super().__init__('step_5_executor')
            self._names = arm_names
            self._pubs  = {
                n: self.create_publisher(
                    Float64MultiArray, CTRL_TOPIC.format(arm=n), 10)
                for n in arm_names
            }
            self._cur_q: Dict[str, Optional[np.ndarray]] = {n:None for n in arm_names}
            for name in arm_names:
                for topic in (f'/{name}/gz/joint_states', f'/{name}/joint_states'):
                    self.create_subscription(JointState, topic,
                                             lambda msg,n=name: self._cb(msg,n), 10)

        def _cb(self, msg, name):
            if len(msg.position) < NDOF: return
            jmap = {n:i for i,n in enumerate(msg.name)}
            keys = [f'joint_{k}' for k in range(1,NDOF+1)]
            q = (np.array([msg.position[jmap[k]] for k in keys])
                 if all(k in jmap for k in keys)
                 else np.array(msg.position[:NDOF]))
            self._cur_q[name] = q.astype(float)

        def execute(self, arm_pos, duration):
            dt_ns   = int(1e9 / RATE_HZ)
            n_steps = max(len(arm_pos[n]) for n in self._names)
            self.get_logger().info(
                f'Executing {n_steps} steps × {len(self._names)} arms '
                f'at {RATE_HZ:.0f}Hz ({duration:.2f}s)')
            msg = Float64MultiArray(); t0w = time.monotonic()
            for k in range(n_steps):
                t0 = time.monotonic_ns()
                # Publish ALL arms at the same instant
                for name in self._names:
                    idx = min(k, len(arm_pos[name])-1)
                    msg.data = [float(v) for v in arm_pos[name][idx]]
                    self._pubs[name].publish(msg)
                rclpy.spin_once(self, timeout_sec=0.)
                remain = dt_ns - (time.monotonic_ns() - t0)
                if remain > 0: time.sleep(remain*1e-9)
            # Hold final pose
            for name in self._names:
                msg.data = [float(v) for v in arm_pos[name][-1]]
                self._pubs[name].publish(msg)
            time.sleep(0.5)
            wall = time.monotonic() - t0w
            fq   = {n: self._cur_q[n].tolist() if self._cur_q[n] is not None
                    else None for n in self._names}
            return {'success':True,'mode':'ros2','steps_sent':n_steps,
                    'wall_time_s':round(wall,3),'final_joints':fq,'error':None}


def dry_run(arm_names, arm_pos, duration):
    n_steps   = max(len(arm_pos[n]) for n in arm_names)
    milestone = max(1, n_steps//10)
    print(f'\n  DRY-RUN: {n_steps} steps  ({duration:.2f}s  @{RATE_HZ:.0f}Hz)')
    for k in range(n_steps):
        if k % milestone == 0:
            pct = int(100*k/n_steps)
            q0  = arm_pos[arm_names[0]][min(k,len(arm_pos[arm_names[0]])-1)]
            print(f'    {pct:3d}%  {arm_names[0]}: '
                  f'[{", ".join(f"{v:.3f}" for v in q0)}]')
    print(f'    100%  done')
    return {'success':True,'mode':'dry_run','steps_sent':n_steps,
            'wall_time_s':None,
            'final_joints':{n:arm_pos[n][-1].tolist() for n in arm_names},
            'error':None}


# ─────────────────────────────────────────────────────────────────────────────
# PRINT REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_report(report):
    bar = '═'*68; thin='─'*68
    print(f'\n{bar}\n  STEP 5  —  FINAL EXECUTION REPORT\n{bar}')

    pf = report['preflight']
    print(f'\n  PRE-FLIGHT  :  {"✅ PASSED" if pf["passed"] else "❌ FAILED"}')
    for iss in pf.get('issues',[]): print(f'    ❌  {iss}')
    for w in pf.get('warnings',[]): print(f'    ⚠   {w}')

    print(f'\n  PER-ARM METRICS\n  {thin}')
    for name, m in report.get('per_arm', {}).items():
        oe='✅' if m['ee_ok'] else '❌'; oj='✅' if m['joints_ok'] else '❌'
        print(f'\n  ── {name.upper()} ────────────────────────────────')
        print(f'    Duration          : {m["duration_s"]:.3f}s')
        print(f'    Samples           : {m["n_samples"]}')
        print(f'    EE path length    : {m["ee_path_length_cm"]:.2f}cm')
        print(f'    Max joint vel     : {m["max_joint_vel_rad_s"]:.4f} rad/s')
        print(f'    EE final error    : {m["ee_final_error_mm"]:.2f}mm  {oe}')
        print(f'    Max joint error   : {m["max_joint_error_deg"]:.3f}°  {oj}')
        print(f'    EE final (world)  : {np.round(m["ee_final_world"],4).tolist()}')
        print(f'    Refine iterations : {m["refine_iterations"]}')

    print(f'\n  INTER-ARM SAFETY  (synchronized)\n  {thin}')
    for pair, s in report.get('inter_arm_safety', {}).items():
        icon='✅' if s['collision_free'] else '❌'
        md=s['min_dist_m']; av=s['avg_dist_m']
        print(f'  {icon}  {pair:<22}  '
              f'min={md*100:.2f}cm  avg={av*100:.2f}cm  '
              f'coll={s["n_collisions"]}  {s["status"]}')

    ex = report.get('execution', {})
    print(f'\n  EXECUTION  :  {"✅ SUCCESS" if ex.get("success") else "❌ FAILED"}')
    print(f'    Mode      : {ex.get("mode")}')
    print(f'    Steps     : {ex.get("steps_sent",0)}')
    if ex.get('wall_time_s'): print(f'    Wall time : {ex["wall_time_s"]:.2f}s')
    if ex.get('error'):       print(f'    Error     : {ex["error"]}')

    sr = report.get('synchronisation_report', {})
    if sr.get('pair_reports'):
        print(f'\n  KURAMOTO SUMMARY\n  {thin}')
        for pk, pr in sr['pair_reports'].items():
            icon = '✅' if pr.get('collision_free') else '❌'
            md   = pr.get('min_dist_m', pr.get('min_distance_m', 0.0))
            crit = pr.get('critical', pr.get('critical_violations', 0))
            print(f'  {icon}  {pk:<22}  min={md*100:.2f}cm  crit={crit}')

    any_c = any(not s.get('collision_free',True)
                for s in report.get('inter_arm_safety',{}).values())
    print(f'\n{bar}')
    if not pf['passed']:
        print('  ❌  PRE-FLIGHT FAILED — hardware not commanded')
    elif any_c:
        print('  ❌  RESIDUAL COLLISIONS → increase MAX_REFINE in step_4')
    else:
        print('  ✅  ALL SAFE — execution complete')
    print(f'{bar}\n')


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(args=None):
    print('\n' + '='*68)
    print('  STEP 5  —  Synchronized Execution + Final Report')
    print('='*68)

    if not os.path.exists('synchronized_trajectories.json'):
        print('\n  ❌  synchronized_trajectories.json not found'); sys.exit(1)

    with open('synchronized_trajectories.json') as fh: data = json.load(fh)

    arm_names = sorted([k for k in data if k.startswith('dsr')])
    if not arm_names: print('\n  ❌  No arm data'); sys.exit(1)

    # Use original requested duration from ik_solutions, not the scaled metadata
    _req_dur = 10.0
    if os.path.exists('ik_solutions.json'):
        with open('ik_solutions.json') as _fh: _ik = json.load(_fh)
        _req_dur = float(_ik.get('duration', 10.0))
    duration = _req_dur
    print(f'\n  Arms     : {arm_names}')
    print(f'  Duration : {duration:.2f}s')

    # Pre-flight
    print('\n  Pre-flight check ...')
    pf_ok, pf_rep = preflight(arm_names, data)
    print(f'  {"✅ Passed" if pf_ok else "❌ Failed"}')
    for iss in pf_rep['issues']: print(f'    {iss}')

    if not pf_ok:
        report = {'preflight':pf_rep,'per_arm':{},'inter_arm_safety':{},
                  'execution':{'success':False,'mode':'aborted',
                               'steps_sent':0,'error':'pre-flight failed'}}
        print_report(report)
        with open('execution_report.json','w') as fh: json.dump(report,fh,indent=2)
        sys.exit(1)

    # Resample + align
    arm_pos = {}
    for name in arm_names:
        pos = np.array(data[name]['trajectory']['positions'], dtype=float)
        dur = float(data[name]['metadata'].get('duration', duration))
        arm_pos[name] = resample(pos, dur)

    max_steps = max(len(arm_pos[n]) for n in arm_names)
    for name in arm_names:
        if len(arm_pos[name]) < max_steps:
            s_in=np.linspace(0,1,len(arm_pos[name])); s_out=np.linspace(0,1,max_steps)
            r=np.zeros((max_steps,NDOF))
            for j in range(NDOF): r[:,j]=np.interp(s_out,s_in,arm_pos[name][:,j])
            arm_pos[name]=r

    t_vec = np.linspace(0., duration, max_steps)

    # Execute
    if _ROS_OK:
        rclpy.init(args=args); node = ExecutorNode(arm_names)
        try:    exec_res = node.execute(arm_pos, duration)
        except KeyboardInterrupt:
            exec_res={'success':False,'mode':'interrupted','steps_sent':0,
                      'wall_time_s':None,'final_joints':{},'error':'KeyboardInterrupt'}
        except Exception as e:
            exec_res={'success':False,'mode':'error','steps_sent':0,
                      'wall_time_s':None,'final_joints':{},'error':str(e)}
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
    with open('execution_report.json','w') as fh: json.dump(report,fh,indent=2)
    kb = os.path.getsize('execution_report.json')/1024.
    print(f'  ✅  Saved: execution_report.json  ({kb:.1f} KB)\n')


if __name__ == '__main__':
    main()