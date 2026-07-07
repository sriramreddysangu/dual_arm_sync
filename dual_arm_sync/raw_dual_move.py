#!/usr/bin/env python3
"""
raw_dual_move.py  --  Raw Simultaneous Dual-Arm Move (NO resolution layer)
===============================================================================
Purpose
-------
You type a target joint config for each arm. Both arms drive there
SIMULTANEOUSLY along a straight joint-space line. No B-spline retraction,
no Kuramoto phase shifting, no home-CP detour. If the configs collide,
the arms WILL collide -- both numerically (predicted below) and physically
in Gazebo (links pass through each other on the position controller).

This is the ground-truth check: "does the pair I typed actually collide?"

UNITS
-----
Input defaults to DEGREES (matches step_7). Use --rad to type radians.
Every parsed target is echoed in BOTH units so a unit mistake is obvious.
Anything outside joint limits is clamped AND warned about -- a clamp is the
usual reason an arm "moves but not to the target you typed".

Run
---
  python3 raw_dual_move.py            # degrees, prompt + confirm
  python3 raw_dual_move.py --rad      # type radians instead
  python3 raw_dual_move.py --yes      # skip the "execute?" confirmation
  python3 raw_dual_move.py --no-exec  # predict collision only, do not move

Both arms always share one duration so they start and finish together.
===============================================================================
"""

import sys, time
from typing import Dict, Optional
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float64MultiArray
    from sensor_msgs.msg import JointState
    _ROS_OK = True
except ImportError:
    _ROS_OK = False
    print('[raw_dual_move] no ROS2 -- prediction-only mode')

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS  (identical to step_8 / step_10)
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

ROBOT_BASES: Dict[str, np.ndarray] = {
    'dsr01': np.array([0.0,  0.5, 0.0]),
    'dsr02': np.array([0.0, -0.5, 0.0]),
}
ARM_NAMES = ['dsr01', 'dsr02']

# Sphere-pair collision model -- same as step_8
LINK_RADII    = np.array([0.10, 0.10, 0.08, 0.08, 0.06, 0.06])
LINK_NAMES    = ['base', 'shoulder', 'upper_arm', 'forearm', 'wrist1', 'wrist2']
SAFETY_MARGIN = 0.12
RADII_THR     = (LINK_RADII[:, None] + LINK_RADII[None, :]) + SAFETY_MARGIN  # (6,6)

# DH <-> joint mapping.
# VERIFIED from `ros2 topic echo /dsrXX/gz/joint_states`:
#   joint_states ARRAY order is [joint_1,joint_2,joint_4,joint_5,joint_3,joint_6],
#   BUT we read it BY NAME (joint_1..joint_6) -> that already yields DH order.
#   The position-controller command array is in NUMERICAL order [joint_1..joint_6],
#   which is also DH order. A round-trip test (command -> joint_states) confirmed
#   command[i] drives joint_(i+1). Therefore NO reorder is needed either way.
#   (The earlier [0,1,3,4,2,5] reorder, inherited from step_10, silently sent the
#    arm to a j3/j4/j5-permuted config -- this identity mapping fixes that.)
DH_TO_GZ   = np.array([0, 1, 2, 3, 4, 5])
GZ_TO_DH   = np.array([0, 1, 2, 3, 4, 5])
CTRL_TOPIC = '/{arm}/gz/dsr_position_controller/commands'

# Motion timing -- conservative so the position controller never trips
VEL_USE_FRAC = 0.50
MIN_DURATION = 4.0
WARMUP_S     = 1.0     # hold start pose before ramping (lets controller connect)
HOLD_AFTER   = 1.5     # hold target pose after ramp
SETTLE_READ  = 1.0     # spin time before reading final joints

# Closed-loop convergence (time is not a constraint -> reach the exact target)
CONVERGE_TOL_DEG = 0.05    # accept when every joint is within this of target
CONVERGE_TIMEOUT = 25.0    # max seconds to keep commanding the exact target


# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY
# ─────────────────────────────────────────────────────────────────────────────

def link_origins(q: np.ndarray, base: np.ndarray) -> np.ndarray:
    T = np.eye(4); o = np.zeros((NDOF, 3))
    for i in range(NDOF):
        al, a, to, d = DH[i]; th = q[i] + to
        ct, st = np.cos(th), np.sin(th); ca, sa = np.cos(al), np.sin(al)
        T = T @ np.array([
            [ct,    -st,    0.,  a    ],
            [st*ca,  ct*ca, -sa, -sa*d],
            [st*sa,  ct*sa,  ca,  ca*d],
            [0.,     0.,    0.,  1.   ],
        ])
        o[i] = T[:3, 3] + base
    return o


def pair_dist_matrix(qi, bi, qj, bj) -> np.ndarray:
    oi = link_origins(qi, bi)
    oj = link_origins(qj, bj)
    diff = oi[:, None, :] - oj[None, :, :]
    return np.linalg.norm(diff, axis=2)     # (6,6)


# ─────────────────────────────────────────────────────────────────────────────
# TRAJECTORY  (straight line in joint space, smooth ease-in/ease-out timing)
# ─────────────────────────────────────────────────────────────────────────────

def shared_duration(starts: Dict[str, np.ndarray],
                    targets: Dict[str, np.ndarray]) -> float:
    t = MIN_DURATION
    for n in ARM_NAMES:
        disp = np.abs(targets[n] - starts[n])
        t = max(t, float(np.max(disp / (VEL_LIM * VEL_USE_FRAC))))
    return t


def smoothstep(u: np.ndarray) -> np.ndarray:
    """Cubic ease 3u^2-2u^3: s(0)=0,s(1)=1,s'(0)=s'(1)=0 (zero end-velocity)."""
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def build_linear(start: np.ndarray, target: np.ndarray, duration: float):
    """
    PATH is the straight line in joint space (start->target, no deviation).
    TIMING uses smoothstep so the arm eases out and arrives with ~zero velocity,
    which lets the position controller settle exactly on target instead of
    overshooting/lagging. Returns (positions, s_profile). Peak joint velocity is
    1.5x the average, still well under the limit at VEL_USE_FRAC=0.5.
    """
    n   = max(2, int(round(duration * RATE_HZ)))
    s   = smoothstep(np.linspace(0.0, 1.0, n))                 # (n,)
    pos = start[None, :] + s[:, None] * (target - start)[None, :]
    return np.clip(pos, POS_LIM[:, 0], POS_LIM[:, 1]), s


# ─────────────────────────────────────────────────────────────────────────────
# COLLISION PREDICTION  (step_8 model, on the LINEAR path)
# ─────────────────────────────────────────────────────────────────────────────

def predict_collision(pos_i, bi, pos_j, bj, duration, s_profile=None) -> Dict:
    N = min(len(pos_i), len(pos_j))
    if s_profile is None:
        s_profile = np.linspace(0.0, 1.0, N)
    n_coll = 0; global_min = float('inf'); first_k = -1
    worst_pen = 0.0; worst_k = worst_li = worst_lj = -1

    for k in range(N):
        D = pair_dist_matrix(pos_i[k], bi, pos_j[k], bj)
        global_min = min(global_min, float(D.min()))
        slack = RADII_THR - D            # >0 means interpenetration
        if (slack > 0).any():
            n_coll += 1
            if first_k < 0:
                first_k = k
            li, lj = np.unravel_index(int(np.argmax(slack)), D.shape)
            p = float(slack[li, lj])
            if p > worst_pen:
                worst_pen = p; worst_k = k
                worst_li = int(li); worst_lj = int(lj)

    def arc(k):  return round(float(s_profile[k]), 3)
    def tsec(k): return round(k / max(N - 1, 1) * duration, 3)  # sampled at fixed RATE_HZ

    return {
        'collision'        : n_coll > 0,
        'n_steps'          : N,
        'n_collision_steps': n_coll,
        'collision_pct'    : round(100 * n_coll / max(N, 1), 2),
        'global_min_cm'    : round(global_min * 100, 2) if global_min < 1e9 else None,
        'first_time_s'     : tsec(first_k) if first_k >= 0 else None,
        'first_frac'       : arc(first_k) if first_k >= 0 else None,
        'worst_pen_cm'     : round(worst_pen * 100, 2),
        'worst_time_s'     : tsec(worst_k) if worst_k >= 0 else None,
        'worst_link_i'     : LINK_NAMES[worst_li] if worst_li >= 0 else None,
        'worst_link_j'     : LINK_NAMES[worst_lj] if worst_lj >= 0 else None,
    }


def print_prediction(rep: Dict):
    bar = '=' * 68
    print('\n' + bar)
    print('  COLLISION PREDICTION  (linear path, step_8 model, NO resolution)')
    print(bar)
    if rep['collision']:
        print('  VERDICT          : COLLISION  <-- arms will intersect in Gazebo')
        print('  Colliding steps  : {} / {}  ({:.1f}%)'.format(
            rep['n_collision_steps'], rep['n_steps'], rep['collision_pct']))
        print('  First contact    : t={:.3f}s  (arc {:.2f})'.format(
            rep['first_time_s'], rep['first_frac']))
        print('  Worst penetration: {:.2f}cm  @ t={:.3f}s'.format(
            rep['worst_pen_cm'], rep['worst_time_s']))
        print('  Worst link pair  : dsr01.{}  <->  dsr02.{}'.format(
            rep['worst_link_i'], rep['worst_link_j']))
        print('  Closest approach : {:.2f}cm'.format(rep['global_min_cm']))
    else:
        print('  VERDICT          : SAFE  (no step breaches {:.0f}cm safety margin)'.format(
            SAFETY_MARGIN * 100))
        print('  Closest approach : {:.2f}cm'.format(rep['global_min_cm']))
    print(bar)


# ─────────────────────────────────────────────────────────────────────────────
# INPUT
# ─────────────────────────────────────────────────────────────────────────────

def fmt(q: np.ndarray, use_deg: bool) -> str:
    vals = np.degrees(q) if use_deg else q
    return '[' + ', '.join('{:.2f}'.format(v) for v in vals) + ']'


def parse_joints(raw: str, use_deg: bool):
    """Return (clamped_q, clamped_flag) or (None, False) on parse error."""
    try:
        vals = [float(v) for v in raw.replace(',', ' ').split()]
    except ValueError as e:
        print('    parse error: {}'.format(e)); return None, False
    if len(vals) != NDOF:
        print('    need exactly {} values, got {}'.format(NDOF, len(vals)))
        return None, False
    q_raw = np.radians(vals) if use_deg else np.array(vals, dtype=float)
    q     = np.clip(q_raw, POS_LIM[:, 0], POS_LIM[:, 1])
    clamped = not np.allclose(q, q_raw, atol=1e-9)
    return q, clamped


def prompt_target(name: str, start: Optional[np.ndarray], use_deg: bool) -> np.ndarray:
    unit = 'deg' if use_deg else 'rad'
    print('\n  -- {}  base={}  --------------------------------'.format(
        name.upper(), ROBOT_BASES[name].tolist()))
    if start is not None:
        print('  start (Gazebo now, {}): {}'.format(unit, fmt(start, use_deg)))
    if use_deg:
        print('  limits (deg): J1=+-360  J2=+-94.5  J3=+-159.9  J4/5/6=+-360')
    while True:
        raw = input('  TARGET J1..J6 ({}): '.format(unit)).strip()
        q, clamped = parse_joints(raw, use_deg)
        if q is None:
            continue
        if clamped:
            print('    WARNING: input was out of joint limits and was CLAMPED.')
            print('    you typed -> clamped target now: {} {}'.format(fmt(q, use_deg), unit))
        # echo in both units so a wrong-unit entry is unmistakable
        print('    target confirmed:  {} deg   |   {} rad'.format(
            fmt(q, True), fmt(q, False)))
        return q


# ─────────────────────────────────────────────────────────────────────────────
# ROS2  -- read current joints + publish commands simultaneously
# ─────────────────────────────────────────────────────────────────────────────

if _ROS_OK:
    class RawMoveNode(Node):
        def __init__(self):
            super().__init__('raw_dual_move')
            self._cur: Dict[str, Optional[np.ndarray]] = {n: None for n in ARM_NAMES}
            self._pubs = {
                n: self.create_publisher(Float64MultiArray, CTRL_TOPIC.format(arm=n), 10)
                for n in ARM_NAMES
            }
            for name in ARM_NAMES:
                for topic in ('/{}/gz/joint_states'.format(name),
                              '/{}/joint_states'.format(name)):
                    self.create_subscription(
                        JointState, topic, lambda m, n=name: self._cb(m, n), 10)

        def _cb(self, msg, name):
            if len(msg.position) < NDOF:
                return
            jmap = {nm: i for i, nm in enumerate(msg.name)}
            keys = ['joint_{}'.format(k) for k in range(1, NDOF + 1)]
            if all(k in jmap for k in keys):
                q_gz = np.array([msg.position[jmap[k]] for k in keys])  # Gazebo order
            else:
                q_gz = np.array(msg.position[:NDOF])                    # assume Gazebo order
            # joint_states is in Gazebo order -> reorder to DH order to match targets
            self._cur[name] = q_gz.astype(float)[GZ_TO_DH]

        def wait_states(self, t=15.0) -> bool:
            t0 = time.time()
            while rclpy.ok() and time.time() - t0 < t:
                rclpy.spin_once(self, timeout_sec=0.05)
                if all(self._cur[n] is not None for n in ARM_NAMES):
                    return True
            return False

        def current(self, name) -> Optional[np.ndarray]:
            return None if self._cur[name] is None else self._cur[name].copy()

        def _publish_all(self, q_dh_by_arm: Dict[str, np.ndarray]):
            msg = Float64MultiArray()
            for name in ARM_NAMES:
                msg.data = [float(v) for v in q_dh_by_arm[name][DH_TO_GZ]]
                self._pubs[name].publish(msg)

        def _hold(self, q_by_arm: Dict[str, np.ndarray], seconds: float):
            dt_ns = int(1e9 / RATE_HZ)
            for _ in range(max(1, int(seconds * RATE_HZ))):
                t0 = time.monotonic_ns()
                self._publish_all(q_by_arm)
                rclpy.spin_once(self, timeout_sec=0.)
                rem = dt_ns - (time.monotonic_ns() - t0)
                if rem > 0:
                    time.sleep(rem * 1e-9)

        def _approach_planned_start(self, traj: Dict[str, np.ndarray]):
            """
            Re-read the live pose and ramp from it to the planned start.
            This is what prevents a sudden snap (e.g. toward home): the move
            always begins from where the arm ACTUALLY is, not a stale/assumed
            start. If poses match, this is just a brief hold.
            """
            dt_ns = int(1e9 / RATE_HZ)
            for _ in range(int(0.4 * RATE_HZ)):          # fresh joint_states
                rclpy.spin_once(self, timeout_sec=0.02)
            cur = {n: self.current(n) for n in ARM_NAMES}

            if any(cur[n] is None for n in ARM_NAMES):
                print('  (no live pose -- holding planned start for {:.1f}s)'.format(WARMUP_S))
                self._hold({n: traj[n][0] for n in ARM_NAMES}, WARMUP_S)
                return

            # velocity-limited duration for the corrective approach
            app_dur = WARMUP_S
            for n in ARM_NAMES:
                disp   = np.abs(traj[n][0] - cur[n])
                app_dur = max(app_dur, float(np.max(disp / (VEL_LIM * VEL_USE_FRAC))))
            gap_deg = max(float(np.max(np.abs(np.degrees(traj[n][0] - cur[n]))))
                          for n in ARM_NAMES)
            print('  Approaching planned start (gap={:.1f} deg, {:.1f}s)...'.format(
                gap_deg, app_dur))

            n_app = max(2, int(app_dur * RATE_HZ))
            for k in range(n_app):
                t0 = time.monotonic_ns()
                a  = k / (n_app - 1)
                self._publish_all({n: cur[n] + a * (traj[n][0] - cur[n]) for n in ARM_NAMES})
                rclpy.spin_once(self, timeout_sec=0.)
                rem = dt_ns - (time.monotonic_ns() - t0)
                if rem > 0:
                    time.sleep(rem * 1e-9)

        def execute(self, traj: Dict[str, np.ndarray], duration: float,
                    targets: Dict[str, np.ndarray]):
            dt_ns   = int(1e9 / RATE_HZ)
            n_steps = max(len(traj[n]) for n in ARM_NAMES)
            bi, bj  = ROBOT_BASES['dsr01'], ROBOT_BASES['dsr02']

            # Approach the planned start from the LIVE measured pose (no snap)
            self._approach_planned_start(traj)

            print('  Executing {} steps @ {:.0f}Hz  ({:.2f}s)  -- watch Gazebo'.format(
                n_steps, RATE_HZ, duration))
            min_seen = float('inf'); collided = False
            milestone = max(1, n_steps // 10)

            for k in range(n_steps):
                t0 = time.monotonic_ns()
                self._publish_all({n: traj[n][min(k, len(traj[n]) - 1)] for n in ARM_NAMES})
                rclpy.spin_once(self, timeout_sec=0.)

                if k % milestone == 0 or k == n_steps - 1:
                    D = pair_dist_matrix(
                        traj['dsr01'][min(k, len(traj['dsr01']) - 1)], bi,
                        traj['dsr02'][min(k, len(traj['dsr02']) - 1)], bj)
                    d = float(D.min()); min_seen = min(min_seen, d)
                    hit = bool((D < RADII_THR).any())
                    collided = collided or hit
                    print('    t={:5.2f}s  min_dist={:6.2f}cm{}'.format(
                        k / RATE_HZ, d * 100, '   <-- COLLISION' if hit else ''))

                rem = dt_ns - (time.monotonic_ns() - t0)
                if rem > 0:
                    time.sleep(rem * 1e-9)

            # Closed-loop convergence: keep commanding the EXACT target and read
            # back until every joint is within tolerance (time is no constraint).
            self._converge(targets, collided)

            print('\n  Live closest approach = {:.2f}cm  ({})'.format(
                min_seen * 100, 'COLLIDED' if collided else 'clear'))
            print('\n  ARRIVAL CHECK (measured Gazebo joints vs commanded target):')
            for name in ARM_NAMES:
                meas = self.current(name)
                if meas is None:
                    print('    {}: no joint_states received'.format(name)); continue
                err_deg = float(np.max(np.abs(np.degrees(meas - targets[name]))))
                ok = 'OK' if err_deg <= np.degrees(np.radians(CONVERGE_TOL_DEG)) * 4 else 'OFF'
                print('    {} [{}]  max joint error = {:.3f} deg'.format(name, ok, err_deg))
                print('        commanded: {} deg'.format(fmt(targets[name], True)))
                print('        measured : {} deg'.format(fmt(meas, True)))

        def _converge(self, targets: Dict[str, np.ndarray], collided: bool):
            """Hold the exact target in closed loop until joints settle on it."""
            dt_ns = int(1e9 / RATE_HZ)
            tol   = np.radians(CONVERGE_TOL_DEG)
            print('\n  Converging to exact target (tol={:.3f} deg, up to {:.0f}s)...'.format(
                CONVERGE_TOL_DEG, CONVERGE_TIMEOUT))
            t0 = time.time(); last_print = -10.0; err = float('inf')
            while time.time() - t0 < CONVERGE_TIMEOUT:
                loop = time.monotonic_ns()
                self._publish_all(targets)
                rclpy.spin_once(self, timeout_sec=0.)
                cur = {n: self.current(n) for n in ARM_NAMES}
                if all(cur[n] is not None for n in ARM_NAMES):
                    err = max(float(np.max(np.abs(cur[n] - targets[n]))) for n in ARM_NAMES)
                    if err <= tol:
                        print('    converged: max joint error {:.4f} deg'.format(np.degrees(err)))
                        return
                    now = time.time() - t0
                    if now - last_print >= 2.0:
                        print('    settling... max joint error {:.3f} deg'.format(np.degrees(err)))
                        last_print = now
                rem = dt_ns - (time.monotonic_ns() - loop)
                if rem > 0:
                    time.sleep(rem * 1e-9)
            note = ' (arms collided -- target is physically blocked)' if collided \
                   else ' (controller may have a steady-state offset)'
            print('    timeout: best max joint error {:.3f} deg{}'.format(
                np.degrees(err), note))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    use_deg  = '--rad' not in sys.argv      # DEGREES by default now
    auto_yes = '--yes' in sys.argv
    no_exec  = '--no-exec' in sys.argv

    bar = '=' * 68
    print('\n' + bar)
    print('  RAW DUAL-ARM MOVE  --  simultaneous, linear, NO collision avoidance')
    print(bar)
    print('  dsr01 base {}   dsr02 base {}'.format(
        ROBOT_BASES['dsr01'].tolist(), ROBOT_BASES['dsr02'].tolist()))
    print('  Input units: {}   ({})'.format(
        'DEGREES' if use_deg else 'RADIANS',
        'pass --rad for radians' if use_deg else 'pass nothing for degrees'))

    # ── start configs ──────────────────────────────────────────────────────────
    node = None
    starts: Dict[str, np.ndarray] = {}
    read_ok = True
    if _ROS_OK:
        rclpy.init()
        node = RawMoveNode()
        print('\n  Reading Gazebo joint states (15s timeout)...')
        if node.wait_states(15.0):
            for n in ARM_NAMES:
                starts[n] = node.current(n)
            print('  Both arms read. Using current state as START.')
        else:
            read_ok = False
            print('  TIMEOUT -- no joint_states received from one or both arms.')
            print('  NOT using zeros as start (that would snap the arm to HOME).')
            print('  Check the controllers/topics, then re-run. Prediction only below.')
            for n in ARM_NAMES:
                starts[n] = np.zeros(NDOF)   # placeholder for prediction display only
    else:
        for n in ARM_NAMES:
            starts[n] = np.zeros(NDOF)

    # ── target configs ─────────────────────────────────────────────────────────
    targets: Dict[str, np.ndarray] = {}
    for n in ARM_NAMES:
        targets[n] = prompt_target(n, starts[n], use_deg)

    # ── build straight-line trajectories (shared duration, smooth timing) ───────
    duration = shared_duration(starts, targets)
    built    = {n: build_linear(starts[n], targets[n], duration) for n in ARM_NAMES}
    traj     = {n: built[n][0] for n in ARM_NAMES}
    s_profile = built['dsr01'][1]   # same ease profile for both arms
    print('\n  Shared duration = {:.2f}s   samples/arm = {}'.format(
        duration, len(traj['dsr01'])))

    for n in ARM_NAMES:
        sweep = np.degrees(np.abs(targets[n] - starts[n]))
        big   = np.where(sweep > 180.0)[0]
        if len(big):
            print('  NOTE {}: joints {} sweep >180 deg ({}) -- linear path goes direct'.format(
                n, (big + 1).tolist(), [round(float(sweep[i])) for i in big]))

    # ── predict collision on the straight-line path ─────────────────────────────
    rep = predict_collision(
        traj['dsr01'], ROBOT_BASES['dsr01'],
        traj['dsr02'], ROBOT_BASES['dsr02'], duration, s_profile)
    print_prediction(rep)

    # ── execute ─────────────────────────────────────────────────────────────────
    if no_exec:
        print('\n  --no-exec set -- prediction only, arms not commanded.')
    elif not _ROS_OK:
        print('\n  No ROS2 -- cannot move Gazebo. Prediction above is the result.')
    elif not read_ok:
        print('\n  Joint states were not read -- refusing to execute (would risk a HOME snap).')
    else:
        go = 'y' if auto_yes else input('\n  Execute this move in Gazebo? [y/N]: ').strip().lower()
        if go == 'y':
            node.execute(traj, duration, targets)
        else:
            print('  Skipped execution.')

    if node is not None:
        try: node.destroy_node()
        except Exception: pass
        try: rclpy.shutdown()
        except Exception: pass

    print()


if __name__ == '__main__':
    main()