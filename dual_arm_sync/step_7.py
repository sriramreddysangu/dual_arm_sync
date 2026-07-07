#!/usr/bin/env python3
"""
step_7.py  —  Interactive Multi-Waypoint B-Spline Trajectory
═══════════════════════════════════════════════════════════════════════════════
Input  : interactive  (you type joint configs per arm)
Output : trajectories.json  (feeds step_8 → step_9 → step_10)

MODE SELECTION
──────────────
  [1] Single-arm  — pick one arm, visualise, no collision check needed
  [2] Dual-arm    — both arms, step_8 checks inter-arm collisions

HOW IT WORKS
────────────
A  Chord-length parameterisation  — u[k] proportional to joint-space arc length
B  de Boor averaging knot vector  — knots where path changes fastest
C  Interpolating B-spline          — solve A·P=Q, passes through EVERY waypoint
D  Velocity scaling                — duration >= max_disp/(VEL_LIM*0.65)*1.4
E  Segment CP structure            — LS refit onto n_seg x N_CP_SEG grid

RUN
───
  ros2 run dual_arm_sync step_7
  ros2 run dual_arm_sync step_7 --show
═══════════════════════════════════════════════════════════════════════════════
"""

import json, math, os, sys, time
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    _ROS_OK = True
except ImportError:
    _ROS_OK = False

# Gazebo joint order -> DH order (Doosan M1013)
# Gazebo names joints semantically so jmap lookup gives correct DH order.
GZ_TO_DH = [0, 1, 4, 2, 3, 5]

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
    [-2*_PI,  2*_PI ],
    [-1.6493,  1.6493],
    [-2.7925,  2.7925],
    [-2*_PI,  2*_PI ],
    [-2*_PI,  2*_PI ],
    [-2*_PI,  2*_PI ],
], dtype=float)

VEL_LIM = np.array([2.094, 2.094, 3.140, 3.927, 3.927, 3.927])
ACC_LIM = np.array([8.0,   8.0,   8.0,  12.0,  12.0,  12.0])
NDOF    = 6
RATE_HZ = 100.0

ALL_ROBOT_BASES: Dict[str, np.ndarray] = {
    'dsr01': np.array([0.0,  0.5, 0.0]),
    'dsr02': np.array([0.0, -0.5, 0.0]),
}
ALL_ROBOT_NAMES = ['dsr01', 'dsr02']

N_SEG_MIN    = 5
N_CP_SEG     = 4
DEG          = 3
MIN_DURATION = 5.0
VEL_USE_FRAC = 0.65


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
# CHORD-LENGTH PARAMETERISATION
# ─────────────────────────────────────────────────────────────────────────────

def chord_param(waypoints: np.ndarray) -> np.ndarray:
    if len(waypoints) < 2:
        return np.array([0.0])
    diffs  = np.diff(waypoints, axis=0)
    dists  = np.linalg.norm(diffs, axis=1)
    total  = float(np.sum(dists))
    if total < 1e-12:
        return np.linspace(0.0, 1.0, len(waypoints))
    u      = np.zeros(len(waypoints))
    u[1:]  = np.cumsum(dists) / total
    u[-1]  = 1.0
    return u


# ─────────────────────────────────────────────────────────────────────────────
# INTERPOLATING B-SPLINE
# ─────────────────────────────────────────────────────────────────────────────

def fit_interp_bspline(waypoints: np.ndarray) -> Tuple[object, np.ndarray, int]:
    n    = len(waypoints)
    k    = min(DEG, n - 1)
    u    = chord_param(waypoints)
    u[0] = 0.0; u[-1] = 1.0
    spl  = make_interp_spline(u, waypoints, k=k, bc_type=None)
    return spl, u, k


# ─────────────────────────────────────────────────────────────────────────────
# SEGMENT CP STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

def n_seg_for_waypoints(n_wp: int) -> int:
    n_seg = math.ceil((max(n_wp, N_SEG_MIN * 3 + 1) - 1) / (N_CP_SEG - 1))
    return max(N_SEG_MIN, n_seg)


def make_seg_knots(ncp: int, deg: int = DEG) -> np.ndarray:
    n_inner = max(0, ncp - deg - 1)
    inner   = np.linspace(0, 1, n_inner + 2)[1:-1] if n_inner > 0 else np.array([])
    return np.concatenate([np.zeros(deg + 1), inner, np.ones(deg + 1)])


def ls_fit_segments(pos_ref: np.ndarray,
                    n_seg: int,
                    n_cp_seg: int = N_CP_SEG) -> Tuple[np.ndarray, float]:
    n_global = n_seg * (n_cp_seg - 1) + 1
    N        = len(pos_ref)
    s        = np.linspace(0.0, 1.0, N)
    knots    = make_seg_knots(n_global)

    A = np.zeros((N, n_global))
    for i in range(n_global):
        c       = np.zeros(n_global); c[i] = 1.0
        A[:, i] = BSpline(knots, c, DEG, extrapolate=True)(s)

    cp_global = np.zeros((n_global, NDOF))
    residuals = np.zeros(NDOF)
    for j in range(NDOF):
        sol, res, _, _ = np.linalg.lstsq(A, pos_ref[:, j], rcond=None)
        cp_global[:, j] = sol
        if len(res) > 0:
            residuals[j] = float(np.sqrt(res[0] / N))

    cp_global = np.clip(cp_global, POS_LIM[:, 0], POS_LIM[:, 1])

    cp_segs = np.zeros((n_seg, n_cp_seg, NDOF))
    for seg in range(n_seg):
        i0           = seg * (n_cp_seg - 1)
        cp_segs[seg] = cp_global[i0: i0 + n_cp_seg]

    rms_deg = float(np.max(np.degrees(residuals)))
    return cp_segs, rms_deg


# ─────────────────────────────────────────────────────────────────────────────
# DURATION + EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def adaptive_duration(waypoints: np.ndarray, requested: float) -> float:
    max_t = MIN_DURATION
    for k in range(len(waypoints) - 1):
        disp  = np.abs(waypoints[k + 1] - waypoints[k])
        t_k   = float(np.max(disp / (VEL_LIM * VEL_USE_FRAC)))
        max_t = max(max_t, t_k)
    return max(max_t * 1.4, MIN_DURATION, requested)


def eval_from_spl(spl, duration: float):
    n   = max(2, int(round(duration * RATE_HZ)))
    s   = np.linspace(0.0, 1.0, n)
    pos = np.clip(spl(s), POS_LIM[:, 0], POS_LIM[:, 1])
    # Only compute derivatives up to the actual spline degree
    if spl.k >= 1:
        vel = spl.derivative(1)(s) / duration
    else:
        vel = np.zeros_like(pos)
    if spl.k >= 2:
        acc = spl.derivative(2)(s) / duration**2
    else:
        acc = np.zeros_like(pos)
    t = np.linspace(0.0, duration, n)
    return pos, vel, acc, t


def scale_duration_spl(spl, duration: float):
    pos, vel, acc, t = eval_from_spl(spl, duration)
    sv = sa = 1.0
    for j in range(NDOF):
        vp = float(np.max(np.abs(vel[:, j])))
        ap = float(np.max(np.abs(acc[:, j])))
        if vp > VEL_LIM[j]: sv = max(sv, vp / VEL_LIM[j])
        if spl.k >= 2 and ap > ACC_LIM[j]:
            sa = max(sa, float(np.sqrt(ap / ACC_LIM[j])))
    scale = max(sv, sa)
    if scale > 1.0:
        duration = duration * scale * 1.05
        pos, vel, acc, t = eval_from_spl(spl, duration)
    return pos, vel, acc, t, duration


# ─────────────────────────────────────────────────────────────────────────────
# EDUCATIONAL PRINTOUT
# ─────────────────────────────────────────────────────────────────────────────

def print_bspline_edu(name: str, spl, u: np.ndarray, waypoints: np.ndarray,
                      degree: int, n_seg: int, n_cp_seg: int,
                      duration: float, rms_deg: float):
    n_wp   = len(waypoints)
    n_cp_i = len(spl.c)
    n_cp_s = n_seg * (n_cp_seg - 1) + 1
    knots  = spl.t
    interp = knots[degree + 1: -(degree + 1)]
    W      = 66

    # Build strings first to avoid nested f-string evaluation bug
    title_str  = 'B-SPLINE ANALYSIS  [' + name + ']'
    kv_hdr     = 'KNOT VECTOR  t[0..' + str(len(knots) - 1) + ']'
    deg_note   = '(cubic)' if degree == 3 else '(quadratic)' if degree == 2 else '(linear)'
    end_label  = 'WP' + str(n_wp - 1)

    print(f'\n  ╔{"═"*W}╗')
    print(f'  ║{title_str:^{W}}║')
    print(f'  ╠{"═"*W}╣')
    print(f'  ║  {"Waypoints":<23}: {n_wp:<6} {"(each = 6-DOF joint configuration)":<{W-34}}║')
    print(f'  ║  {"Interp control pts":<23}: {n_cp_i:<6} {"(= N waypoints — exact interpolation)":<{W-34}}║')
    print(f'  ║  {"Degree":<23}: {degree:<6} {deg_note:<{W-34}}║')
    print(f'  ║  {"Knot vector length":<23}: {len(knots):<6} {"(= N_cp + degree + 1)":<{W-34}}║')
    print(f'  ║  {"Segment CPs (total)":<23}: {n_cp_s:<6} {"(for step_8/9/10 compatibility)":<{W-34}}║')
    print(f'  ║  {"Duration":<23}: {duration:.3f}s{"":<{W-31}}║')
    print(f'  ║  {"LS fit RMS residual":<23}: {rms_deg:.5f}° {"(should be < 0.1°)":<{W-33}}║')

    print(f'  ╠{"═"*W}╣')
    print(f'  ║{"CHORD-LENGTH PARAMETERISATION":^{W}}║')
    print(f'  ║  {"WP":>4}  | {"Label":<6} | {"u value":>9} | {"Jt-dist (deg)":>13} | {"time (s)":>10} ║')
    print(f'  ║  {"─"*4}──+─{"─"*6}─+─{"─"*9}─+─{"─"*13}─+─{"─"*10}─║')
    for i, ui in enumerate(u):
        dist_d = 0.0 if i == 0 else float(np.degrees(
            np.linalg.norm(waypoints[i] - waypoints[i - 1])))
        label = 'START' if i == 0 else ('END  ' if i == n_wp - 1 else 'WP {:2d}'.format(i))
        print(f'  ║  {i:4d}  | {label} | {ui:9.5f} | {dist_d:11.3f} deg | {ui*duration:9.3f}s ║')

    print(f'  ╠{"═"*W}╣')
    print(f'  ║{kv_hdr:^{W}}║')
    print(f'  ║  First {degree+1} = 0.000  clamped start (spline begins exactly at WP0)     ║')
    print(f'  ║  Last  {degree+1} = 1.000  clamped end   (spline ends exactly at {end_label})       ║')
    if len(interp) > 0:
        interp_str = '  '.join('{:.4f}'.format(v) for v in interp[:10])
        suffix     = ' ... ({} more)'.format(len(interp) - 10) if len(interp) > 10 else ''
        print('  ║  Interior [{}]: {}{}'.format(len(interp), interp_str, suffix))
        print('  ║  (de Boor averaging — knots cluster where path changes fastest)')
    else:
        print('  ║  No interior knots — max flexibility with {} CPs only'.format(n_cp_i))

    print(f'  ╠{"═"*W}╣')
    print(f'  ║{"INTERPOLATION VERIFICATION  (max err per waypoint)":^{W}}║')
    for i, ui in enumerate(u):
        actual  = spl(float(ui))
        err_deg = float(np.max(np.abs(np.degrees(actual - waypoints[i]))))
        ok      = err_deg < 0.01
        sym     = '✓' if ok else '✗'
        label   = 'START' if i == 0 else ('END  ' if i == n_wp-1 else 'WP {:2d}'.format(i))
        note    = '(OK — exact fit)' if ok else '(WARN)'
        print('  ║  {}  {}  u={:.5f}  max_error = {:.6f} deg  {:>{}}║'.format(
            sym, label, ui, err_deg, note, W - 42))

    print(f'  ╠{"═"*W}╣')
    print(f'  ║{"SEGMENT STRUCTURE  (step_8/9/10 compatible)":^{W}}║')
    seg_line = 'n_seg={}  n_cp_seg={}  total_seg_CPs={}'.format(n_seg, n_cp_seg, n_cp_s)
    print('  ║  {:<{}}║'.format(seg_line, W - 2))
    print('  ║  +1 waypoint => +1 interp CP => segment structure auto-expands  ║')
    print(f'  ╚{"═"*W}╝')


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION  (3 figures)
# ─────────────────────────────────────────────────────────────────────────────

def visualize(name: str, pos: np.ndarray, vel: np.ndarray, t: np.ndarray,
              spl, u: np.ndarray, waypoints: np.ndarray,
              cp_segs: np.ndarray, n_seg: int, n_cp_seg: int,
              duration: float, base: np.ndarray, show: bool = False):
    try:
        import matplotlib
        matplotlib.use('TkAgg' if show else 'Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa
    except ImportError:
        print('  (matplotlib not installed — skipping plots)')
        return

    n_wp     = len(waypoints)
    t_wp     = u * duration
    n_global = n_seg * (n_cp_seg - 1) + 1
    cp_flat  = np.zeros((n_global, NDOF))
    for seg in range(n_seg):
        i0 = seg * (n_cp_seg - 1)
        cp_flat[i0: i0 + n_cp_seg] = cp_segs[seg]
    t_cp   = np.linspace(0.0, duration, n_global)
    int_kt = [kt for kt in spl.t * duration if 0.02 < kt < duration - 0.02]

    # ── Figure 1: Joint angles ───────────────────────────────────────────────
    fig1, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    fig1.suptitle(
        '[{}]  Joint Trajectories\n{} waypoints | degree={} | {:.2f}s | {}seg x {}CP'.format(
            name, n_wp, spl.k, duration, n_seg, n_cp_seg),
        fontsize=11, fontweight='bold')
    for j, ax in enumerate(axes.flat):
        ax.plot(t, np.degrees(pos[:, j]), color='steelblue', lw=2.0,
                label='B-spline', zorder=3)
        ax.scatter(t_wp, np.degrees(waypoints[:, j]),
                   color='red', s=90, zorder=6, label='Waypoints')
        ax.plot(t_cp, np.degrees(cp_flat[:, j]),
                color='limegreen', lw=1.0, ls='--', marker='s', ms=4,
                alpha=0.8, zorder=4, label='Ctrl polygon')
        for kt in int_kt:
            ax.axvline(kt, color='darkorange', lw=0.8, ls=':', alpha=0.7)
        for si in range(1, n_seg):
            ax.axvline(si / n_seg * duration, color='#aaa', lw=0.5, alpha=0.4)
        ax.set_ylabel('Angle (deg)', fontsize=8)
        ax.set_title('Joint {}'.format(j + 1), fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.25); ax.tick_params(labelsize=7)
        if j == 0: ax.legend(fontsize=7, loc='best')
    for ax in axes[1]: ax.set_xlabel('Time (s)', fontsize=8)
    fig1.text(0.5, 0.01,
              'Red=waypoints  Green=ctrl polygon  Orange=knots  Grey=segment boundaries',
              ha='center', fontsize=8, color='#444')
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    f1 = '{}_joint_trajectories.png'.format(name)
    fig1.savefig(f1, dpi=150, bbox_inches='tight')
    print('  Saved: {}'.format(f1))
    if show: plt.show()
    plt.close(fig1)

    # ── Figure 2: Joint velocities ───────────────────────────────────────────
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    fig2.suptitle('[{}]  Joint Velocities  (red dashes = hardware limit)'.format(name),
                  fontsize=11, fontweight='bold')
    if spl.k >= 1:
        vel_plot = spl.derivative(1)(np.linspace(0, 1, len(t))) / duration
    else:
        vel_plot = np.zeros((len(t), NDOF))
    for j, ax in enumerate(axes2.flat):
        ax.plot(t, np.degrees(vel_plot[:, j]), color='steelblue', lw=1.5)
        lim = float(np.degrees(VEL_LIM[j]))
        ax.axhline( lim, color='red', ls='--', lw=1.0,
                   label='+/-{:.1f} deg/s'.format(lim))
        ax.axhline(-lim, color='red', ls='--', lw=1.0)
        ax.axhline(0, color='grey', ls='-', lw=0.5, alpha=0.5)
        ax.set_ylabel('Vel (deg/s)', fontsize=8)
        ax.set_title('Joint {}'.format(j + 1), fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.25); ax.tick_params(labelsize=7)
        if j == 0: ax.legend(fontsize=7)
    for ax in axes2[1]: ax.set_xlabel('Time (s)', fontsize=8)
    plt.tight_layout()
    f2 = '{}_joint_velocities.png'.format(name)
    fig2.savefig(f2, dpi=150, bbox_inches='tight')
    print('  Saved: {}'.format(f2))
    if show: plt.show()
    plt.close(fig2)

    # ── Figure 3: End-effector path ──────────────────────────────────────────
    ee    = np.array([fk_pos(pos[k], base) for k in range(len(pos))])
    wp_ee = np.array([fk_pos(waypoints[i], base) for i in range(n_wp)])

    fig3  = plt.figure(figsize=(14, 6))
    fig3.suptitle('[{}]  End-Effector Path  (world frame)'.format(name),
                  fontsize=11, fontweight='bold')

    ax3d = fig3.add_subplot(1, 2, 1, projection='3d')
    ax3d.plot(ee[:, 0], ee[:, 1], ee[:, 2],
              color='steelblue', lw=2.0, label='EE path')
    ax3d.scatter(wp_ee[:, 0], wp_ee[:, 1], wp_ee[:, 2],
                 color='red', s=80, zorder=5, label='WP end-effectors')
    ax3d.scatter(base[0], base[1], base[2],
                 color='black', s=140, marker='^', zorder=6, label='Base')
    sc = ax3d.scatter(ee[::10, 0], ee[::10, 1], ee[::10, 2],
                      c=np.linspace(0, 1, len(ee[::10])),
                      cmap='plasma', s=20, zorder=4)
    fig3.colorbar(sc, ax=ax3d, fraction=0.03, label='norm. time')
    ax3d.set_xlabel('X (m)'); ax3d.set_ylabel('Y (m)'); ax3d.set_zlabel('Z (m)')
    ax3d.set_title('3D path', fontsize=9); ax3d.legend(fontsize=7)

    ax_xy = fig3.add_subplot(1, 2, 2)
    ax_xy.plot(ee[:, 0], ee[:, 1], color='steelblue', lw=2.0)
    ax_xy.scatter(wp_ee[:, 0], wp_ee[:, 1], color='red', s=80, zorder=5)
    ax_xy.scatter(base[0], base[1], color='black', s=140, marker='^', zorder=6)
    for i, (x, y) in enumerate(wp_ee[:, :2]):
        ax_xy.annotate('WP{}'.format(i), (x, y), xytext=(4, 4),
                       textcoords='offset points', fontsize=7, color='red')
    ax_xy.set_xlabel('X (m)'); ax_xy.set_ylabel('Y (m)')
    ax_xy.set_title('Top view (XY plane)', fontsize=9)
    ax_xy.grid(True, alpha=0.25); ax_xy.set_aspect('equal', 'datalim')
    plt.tight_layout()
    f3 = '{}_ee_path.png'.format(name)
    fig3.savefig(f3, dpi=150, bbox_inches='tight')
    print('  Saved: {}'.format(f3))
    if show: plt.show()
    plt.close(fig3)


# =============================================================================
# ROS2 NODE -- reads current joint states from Gazebo
# =============================================================================

if _ROS_OK:
    class Step7Node(Node):
        def __init__(self):
            super().__init__('step_7_reader')
            self._arm = {n: {'q': np.zeros(NDOF), 'ready': False}
                         for n in ALL_ROBOT_NAMES}
            for name in ALL_ROBOT_NAMES:
                for topic in (f'/{name}/gz/joint_states',
                              f'/{name}/joint_states'):
                    self.create_subscription(
                        JointState, topic,
                        lambda msg, n=name: self._cb(msg, n), 10)

        def _cb(self, msg, name):
            if len(msg.position) < NDOF: return
            jmap = {n: i for i, n in enumerate(msg.name)}
            keys = [f'joint_{k}' for k in range(1, NDOF + 1)]
            if all(k in jmap for k in keys):
                q = np.array([msg.position[jmap[k]] for k in keys])
            else:
                q = np.array(msg.position[:NDOF])
            self._arm[name]['q']     = q.astype(float)
            self._arm[name]['ready'] = True

        def joints(self, name): return self._arm[name]['q'].copy()
        def ready(self, name):  return self._arm[name]['ready']

        def wait(self, t=15.0):
            t0 = time.time()
            while rclpy.ok() and time.time() - t0 < t:
                rclpy.spin_once(self, timeout_sec=0.05)
                if all(self._arm[n]['ready'] for n in ALL_ROBOT_NAMES):
                    return True
            return False


# =============================================================================
# INPUT HELPERS
# =============================================================================

def _parse_joints(raw: str, use_deg: bool) -> Optional[np.ndarray]:
    try:
        vals = [float(v) for v in raw.replace(',', ' ').split()]
        if len(vals) != NDOF:
            print('    ✗  Need exactly {} values, got {}'.format(NDOF, len(vals)))
            return None
        q = np.array(vals, dtype=float)
        if use_deg:
            q = np.radians(q)
        return np.clip(q, POS_LIM[:, 0], POS_LIM[:, 1])
    except ValueError as e:
        print('    ✗  Parse error: {}'.format(e))
        return None


def prompt_arm_waypoints(name: str, current_q: Optional[np.ndarray] = None) -> np.ndarray:
    bar      = '-' * 64
    base_str = str(ALL_ROBOT_BASES[name].tolist())
    print('\n  +{}+'.format(bar))
    print('  |  ARM: {:<57}|'.format(name.upper()))
    print('  |  base = {:<55}|'.format(base_str))
    print('  +{}+'.format(bar))

    u_raw   = input('  Unit? [D]egrees / [R]adians  (Enter = D): ').strip().lower()
    use_deg = (u_raw != 'r')
    ustr    = 'degrees' if use_deg else 'radians'
    print('  -> Using {}'.format(ustr))

    if current_q is not None:
        deg_str = '[' + ', '.join(f'{np.degrees(v):.2f}' for v in current_q) + ']'
        print(f'  Current Gazebo joints: {deg_str} deg')
        print('  (Enter current joints as START waypoint by pressing Enter when prompted)')

    while True:
        n_raw = input('  How many waypoints for {}? (min 2, START + END): '.format(name)).strip()
        try:
            n_wp = int(n_raw)
            if n_wp >= 2: break
            print('  Need at least 2 waypoints.')
        except ValueError:
            print('  Enter an integer.')

    print('\n  Enter exactly {} joint values ({}), space- or comma-separated.'.format(NDOF, ustr))
    if use_deg:
        print('  Limits: J1=+-360  J2=+-94.5  J3=+-159.9  J4/5/6=+-360  (degrees)')
    print()

    waypoints: List[np.ndarray] = []
    for i in range(n_wp):
        label = 'START' if i == 0 else ('END  ' if i == n_wp - 1 else 'WP {:2d}'.format(i))
        # For the START waypoint, allow Enter to use current Gazebo joints
        enter_hint = '  (Enter=use Gazebo current)' if (i == 0 and current_q is not None) else ''
        while True:
            raw = input('  [{}]  J1..J6 ({}){}:  '.format(label, ustr, enter_hint)).strip()
            if raw == '' and i == 0 and current_q is not None:
                q = np.clip(current_q.copy(), POS_LIM[:, 0], POS_LIM[:, 1])
                print('  -> Using current Gazebo state as START')
            else:
                q = _parse_joints(raw, use_deg)
            if q is not None:
                waypoints.append(q)
                deg_str = '  '.join('{:7.2f} deg'.format(v) for v in np.degrees(q))
                print('           -> {}'.format(deg_str))
                break

    return np.array(waypoints)


# ─────────────────────────────────────────────────────────────────────────────
# BUILD ONE ARM
# ─────────────────────────────────────────────────────────────────────────────

def build_arm(name: str, waypoints: np.ndarray,
              duration_req: float, show_plot: bool = False) -> Dict:
    base = ALL_ROBOT_BASES[name]
    n_wp = len(waypoints)
    print('\n  [{}] Fitting B-spline through {} waypoints ...'.format(name, n_wp))

    spl, u, degree = fit_interp_bspline(waypoints)
    print('  [{}]   Degree={}  interp_CPs={}  knots={}'.format(
        name, degree, len(spl.c), len(spl.t)))

    duration = adaptive_duration(waypoints, duration_req)
    pos, vel, acc, t, duration = scale_duration_spl(spl, duration)
    n_steps = len(pos)
    print('  [{}]   Duration={:.3f}s  samples={}'.format(name, duration, n_steps))

    n_seg            = n_seg_for_waypoints(n_wp)
    cp_segs, rms_deg = ls_fit_segments(pos, n_seg)
    n_global         = n_seg * (N_CP_SEG - 1) + 1
    print('  [{}]   Segments={}x{}CPs  total={}  LS_residual={:.5f} deg'.format(
        name, n_seg, N_CP_SEG, n_global, rms_deg))

    arc_fracs = np.linspace(0.0, 1.0, n_steps)
    ee_path   = np.array([fk_pos(pos[k], base) for k in range(n_steps)])
    path_len  = float(np.sum(np.linalg.norm(np.diff(ee_path, axis=0), axis=1)))

    print_bspline_edu(name, spl, u, waypoints, degree, n_seg, N_CP_SEG,
                      duration, rms_deg)

    print('\n  [{}] Generating plots ...'.format(name))
    visualize(name, pos, vel, t, spl, u, waypoints, cp_segs,
              n_seg, N_CP_SEG, duration, base, show=show_plot)

    seg_info = []
    for seg in range(n_seg):
        seg_info.append({
            'segment'  : int(seg),
            'arc_start': round(seg / n_seg, 4),
            'arc_end'  : round((seg + 1) / n_seg, 4),
            'arc_mid'  : round((seg + 0.5) / n_seg, 4),
            'cp'       : cp_segs[seg].tolist(),
        })

    return {
        'robot_name': name,
        'metadata': {
            'start_joints'     : waypoints[0].tolist(),
            'end_joints'       : waypoints[-1].tolist(),
            'start_joints_deg' : np.degrees(waypoints[0]).tolist(),
            'end_joints_deg'   : np.degrees(waypoints[-1]).tolist(),
            'n_waypoints'      : int(n_wp),
            'all_waypoints_deg': np.degrees(waypoints).tolist(),
            'duration'         : float(duration),
            'n_samples'        : int(n_steps),
            'ee_path_length_m' : round(path_len, 5),
            'n_seg'            : int(n_seg),
            'n_cp_seg'         : int(N_CP_SEG),
            'degree'           : int(DEG),
            'interp_degree'    : int(degree),
            'interp_n_cp'      : int(len(spl.c)),
            'chord_params'     : u.tolist(),
            'knot_vector'      : spl.t.tolist(),
            'ls_rms_deg'       : round(rms_deg, 6),
        },
        'spline': {
            'n_seg'    : int(n_seg),
            'n_cp_seg' : int(N_CP_SEG),   # step_9 load_seg_cps reads this
            'degree'   : int(DEG),
            'segments' : seg_info,         # each has 'cp' shape (N_CP_SEG, NDOF)
        },
        'trajectory': {
            'time'         : t.tolist(),
            'positions'    : pos.tolist(),
            'velocities'   : vel.tolist(),
            'accelerations': acc.tolist(),
            'arc_fracs'    : arc_fracs.tolist(),  # step_8 needs this
            'n_samples'    : int(n_steps),
        },
        'ee_path': {
            'positions': ee_path.tolist(),
            'arc_fracs': arc_fracs.tolist(),
            'length_m' : round(path_len, 5),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# MODE SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def select_mode() -> List[str]:
    bar = '=' * 68
    print('\n' + bar)
    print('  MODE SELECTION')
    print(bar)
    print('  [1]  Single-arm  -- one arm, full viz, step_8 reports SAFE (no pairs)')
    print('  [2]  Dual-arm    -- both arms, step_8 checks inter-arm collisions')
    print()

    while True:
        raw = input('  Choose mode [1/2]  (Enter = 2): ').strip()
        if raw in ('', '2'):
            print('  -> Dual-arm mode  (dsr01 + dsr02)')
            return list(ALL_ROBOT_NAMES)
        if raw == '1':
            break
        print('  Enter 1 or 2.')

    print()
    for idx, n in enumerate(ALL_ROBOT_NAMES):
        base_str = str(ALL_ROBOT_BASES[n].tolist())
        print('  [{}]  {}   base = {}'.format(idx + 1, n, base_str))
    print()

    while True:
        raw2 = input('  Which arm? [1..{}]  (Enter = 1): '.format(
            len(ALL_ROBOT_NAMES))).strip()
        if raw2 == '':
            chosen = ALL_ROBOT_NAMES[0]; break
        try:
            idx = int(raw2) - 1
            if 0 <= idx < len(ALL_ROBOT_NAMES):
                chosen = ALL_ROBOT_NAMES[idx]; break
        except ValueError:
            pass
        print('  Enter 1 – {}.'.format(len(ALL_ROBOT_NAMES)))

    print('  -> Single-arm mode: {}'.format(chosen))
    return [chosen]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    show_plot = '--show' in sys.argv

    bar = '=' * 68
    print('\n' + bar)
    print('  STEP 7  --  Interactive Multi-Waypoint B-Spline Trajectory')
    print(bar)
    for n in ALL_ROBOT_NAMES:
        print('  {}  base: {}'.format(n, str(ALL_ROBOT_BASES[n].tolist())))
    print()
    print('  KEY CONCEPTS:')
    print('  * Exact interpolation  -- spline passes through ALL waypoints')
    print('  * Chord-length u       -- proportional time per inter-WP gap')
    print('  * de Boor knots        -- concentrate resolution at fast changes')
    print('  * Adaptive duration    -- never exceeds velocity limits')
    if show_plot:
        print('\n  [--show] Interactive matplotlib windows enabled.')
    else:
        print('\n  Run with --show to open interactive matplotlib windows.')

    active_arms = select_mode()
    single_arm  = len(active_arms) == 1

    if single_arm:
        print('\n  INFO: Single-arm mode ({})'.format(active_arms[0]))
        print('        step_8 -> SAFE (no pairs to check)')
        print('        step_9 -> trivial pass')
        print('        step_10 -> executes {} only'.format(active_arms[0]))
    else:
        print('\n  INFO: Dual-arm mode -- step_8 will check inter-arm collisions')

    dur_raw = input('\n  Minimum trajectory duration [s]  (Enter = 8.0s): ').strip()
    dur_req = float(dur_raw) if dur_raw else 8.0

    out = {
        'duration': dur_req,
        'source'  : 'step_7_waypoint',
        'mode'    : 'single' if single_arm else 'dual',
    }

    # Start ROS2 and read current joint states from Gazebo
    _node = None
    current_qs: Dict[str, Optional[np.ndarray]] = {n: None for n in ALL_ROBOT_NAMES}
    if _ROS_OK:
        try:
            rclpy.init()
            _node = Step7Node()
            print('\n  Waiting for Gazebo joint states (15s timeout)...')
            if _node.wait(15.0):
                for n in ALL_ROBOT_NAMES:
                    current_qs[n] = _node.joints(n)
                print('  Both arms ready -- current joints loaded.')
            else:
                print('  Timeout -- will use zeros as start if Enter pressed.')
        except Exception as _e:
            print(f'  ROS2 init failed: {_e} -- using zeros.')

    all_durations: List[float] = []
    for name in active_arms:
        waypoints = prompt_arm_waypoints(name, current_qs.get(name))
        arm_data  = build_arm(name, waypoints, dur_req, show_plot)
        out[name] = arm_data
        all_durations.append(arm_data['metadata']['duration'])

    global_dur      = max(all_durations)
    out['duration'] = global_dur

    if _node is not None:
        try: _node.destroy_node()
        except: pass
        try: rclpy.shutdown()
        except: pass

    with open('trajectories.json', 'w') as fh:
        json.dump(out, fh, indent=2)

    kb = os.path.getsize('trajectories.json') / 1024.0
    print('\n  ' + bar)
    print('  OK  Saved: trajectories.json  ({:.1f} KB)'.format(kb))
    print('  Mode          : {}'.format(
        'single-arm (' + active_arms[0] + ')' if single_arm else 'dual-arm'))
    print('  Global duration: {:.3f}s'.format(global_dur))
    print('  Next  ->  ros2 run dual_arm_sync step_8')
    print('  ' + bar + '\n')


if __name__ == '__main__':
    main()