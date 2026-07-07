#!/usr/bin/env python3
"""
step_100.py  —  Single-Arm Interactive B-Spline + Direct Gazebo Execution
═══════════════════════════════════════════════════════════════════════════════
Workspace : ~/dual_arm_ws
Package   : dual_arm_sync

WHAT IT DOES (all in one file, no other steps needed)
──────────────────────────────────────────────────────
1. Ask which arm  (dsr01 / dsr02 / dsr03 / dsr04 / dsr05 / dsr06)
2. Ask waypoints  (degrees or radians, any count ≥ 2)
3. Fit interpolating B-spline through ALL waypoints exactly
4. Adaptive duration  (respects joint velocity / acceleration limits)
5. Print educational B-spline analysis table
6. Save plots:
     {arm}_joint_trajectories.png   — 6 joint angle curves + waypoints
     {arm}_joint_velocities.png     — 6 velocity profiles vs hw limit
     {arm}_ee_path.png              — 3D end-effector path
7. Execute at 100 Hz via ROS2 → Gazebo  (or dry-run if no ROS2)
8. Hold final pose for 2 s, print final joint angles

RUN
────
  ros2 run dual_arm_sync step_100
  ros2 run dual_arm_sync step_100 --show      ← open matplotlib windows

NO EXTERNAL FILES NEEDED  (no trajectories.json, no collision_report.json)
═══════════════════════════════════════════════════════════════════════════════
"""

import json, math, os, sys, time
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline

# ── ROS2 optional import ──────────────────────────────────────────────────────
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float64MultiArray
    from sensor_msgs.msg import JointState
    _ROS_OK = True
except ImportError:
    _ROS_OK = False


# ═════════════════════════════════════════════════════════════════════════════
# ROBOT CONSTANTS  (Doosan M1013)
# ═════════════════════════════════════════════════════════════════════════════

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

VEL_LIM = np.array([2.094, 2.094, 3.140, 3.927, 3.927, 3.927])   # rad/s
ACC_LIM = np.array([8.0,   8.0,   8.0,  12.0,  12.0,  12.0])     # rad/s²
NDOF    = 6
RATE_HZ = 100.0
HOLD_S  = 2.0        # seconds to hold final pose after trajectory

ROBOT_BASES: Dict[str, np.ndarray] = {
    'dsr01': np.array([0.0,  0.5, 0.0]),
    'dsr02': np.array([0.0, -0.5, 0.0]),
    'dsr03': np.array([1.0,  0.5, 0.0]),
    'dsr04': np.array([1.0, -0.5, 0.0]),
    'dsr05': np.array([-1.0,  0.5, 0.0]),
    'dsr06': np.array([-1.0, -0.5, 0.0]),
}
ROBOT_NAMES = sorted(ROBOT_BASES.keys())

CTRL_TOPIC   = '/{arm}/gz/dsr_position_controller/commands'

# Gazebo publishes/expects joints in order: [j1, j2, j4, j5, j3, j6]
# DH / trajectory arrays use order:        [j1, j2, j3, j4, j5, j6]
# When SENDING commands reorder DH -> Gazebo:
DH_TO_GZ = [0, 1, 3, 4, 2, 5]   # q_gz = q_dh[DH_TO_GZ]
JOINT_TOPIC_A = '/{arm}/gz/joint_states'
JOINT_TOPIC_B = '/{arm}/joint_states'

# B-spline structure
N_CP_SEG     = 4
DEG          = 3
N_SEG_MIN    = 5
MIN_DURATION = 5.0
VEL_USE_FRAC = 0.65


# ═════════════════════════════════════════════════════════════════════════════
# FORWARD KINEMATICS
# ═════════════════════════════════════════════════════════════════════════════

def fk_all(q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (ee_local_xyz, full 4×4 transform) — base NOT included."""
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
    return T[:3, 3].copy(), T


def fk_pos(q: np.ndarray, base: np.ndarray) -> np.ndarray:
    p, _ = fk_all(q)
    return p + base


def link_origins(q: np.ndarray, base: np.ndarray) -> np.ndarray:
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


# ═════════════════════════════════════════════════════════════════════════════
# CHORD-LENGTH PARAMETERISATION
# ═════════════════════════════════════════════════════════════════════════════

def chord_param(waypoints: np.ndarray) -> np.ndarray:
    if len(waypoints) < 2:
        return np.array([0.0])
    diffs = np.diff(waypoints, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    total = float(np.sum(dists))
    if total < 1e-12:
        return np.linspace(0.0, 1.0, len(waypoints))
    u      = np.zeros(len(waypoints))
    u[1:]  = np.cumsum(dists) / total
    u[-1]  = 1.0
    return u


# ═════════════════════════════════════════════════════════════════════════════
# INTERPOLATING B-SPLINE
# ═════════════════════════════════════════════════════════════════════════════

def fit_interp_bspline(waypoints: np.ndarray):
    n  = len(waypoints)
    k  = min(DEG, n - 1)
    u  = chord_param(waypoints)
    u[0] = 0.0; u[-1] = 1.0
    spl = make_interp_spline(u, waypoints, k=k, bc_type=None)
    return spl, u, k


# ═════════════════════════════════════════════════════════════════════════════
# SEGMENT CP STRUCTURE  (kept for JSON compatibility with step_8/9/10)
# ═════════════════════════════════════════════════════════════════════════════

def make_seg_knots(ncp: int, deg: int = DEG) -> np.ndarray:
    n_inner = max(0, ncp - deg - 1)
    inner   = np.linspace(0, 1, n_inner + 2)[1:-1] if n_inner > 0 else np.array([])
    return np.concatenate([np.zeros(deg + 1), inner, np.ones(deg + 1)])


def n_seg_for_waypoints(n_wp: int) -> int:
    n_seg = math.ceil((max(n_wp, N_SEG_MIN * 3 + 1) - 1) / (N_CP_SEG - 1))
    return max(N_SEG_MIN, n_seg)


def ls_fit_segments(pos_ref: np.ndarray, n_seg: int) -> Tuple[np.ndarray, float]:
    n_global = n_seg * (N_CP_SEG - 1) + 1
    N        = len(pos_ref)
    s        = np.linspace(0.0, 1.0, N)
    knots    = make_seg_knots(n_global)

    A = np.zeros((N, n_global))
    for i in range(n_global):
        c      = np.zeros(n_global); c[i] = 1.0
        A[:, i] = BSpline(knots, c, DEG, extrapolate=True)(s)

    cp_global = np.zeros((n_global, NDOF))
    residuals = np.zeros(NDOF)
    for j in range(NDOF):
        sol, res, _, _ = np.linalg.lstsq(A, pos_ref[:, j], rcond=None)
        cp_global[:, j] = sol
        if len(res) > 0:
            residuals[j] = float(np.sqrt(res[0] / N))

    cp_global = np.clip(cp_global, POS_LIM[:, 0], POS_LIM[:, 1])

    cp_segs = np.zeros((n_seg, N_CP_SEG, NDOF))
    for seg in range(n_seg):
        i0 = seg * (N_CP_SEG - 1)
        cp_segs[seg] = cp_global[i0: i0 + N_CP_SEG]

    rms_deg = float(np.max(np.degrees(residuals)))
    return cp_segs, rms_deg


# ═════════════════════════════════════════════════════════════════════════════
# DURATION + SCALING
# ═════════════════════════════════════════════════════════════════════════════

def adaptive_duration(waypoints: np.ndarray, requested: float) -> float:
    max_t = MIN_DURATION
    for k in range(len(waypoints) - 1):
        disp  = np.abs(waypoints[k + 1] - waypoints[k])
        t_k   = float(np.max(disp / (VEL_LIM * VEL_USE_FRAC)))
        max_t = max(max_t, t_k)
    return max(max_t * 1.4, MIN_DURATION, requested)


def eval_spl(spl, duration: float):
    n   = max(2, int(round(duration * RATE_HZ)))
    s   = np.linspace(0.0, 1.0, n)
    pos = np.clip(spl(s), POS_LIM[:, 0], POS_LIM[:, 1])
    vel = spl.derivative(1)(s) / duration
    acc = spl.derivative(2)(s) / duration**2
    t   = np.linspace(0.0, duration, n)
    return pos, vel, acc, t


def scale_if_needed(spl, duration: float):
    pos, vel, acc, t = eval_spl(spl, duration)
    sv = sa = 1.0
    for j in range(NDOF):
        vp = float(np.max(np.abs(vel[:, j])))
        ap = float(np.max(np.abs(acc[:, j])))
        if vp > VEL_LIM[j]: sv = max(sv, vp / VEL_LIM[j])
        if ap > ACC_LIM[j]: sa = max(sa, float(np.sqrt(ap / ACC_LIM[j])))
    scale = max(sv, sa)
    if scale > 1.0:
        duration = duration * scale * 1.05
        pos, vel, acc, t = eval_spl(spl, duration)
    return pos, vel, acc, t, duration


# ═════════════════════════════════════════════════════════════════════════════
# EDUCATIONAL PRINTOUT
# ═════════════════════════════════════════════════════════════════════════════

def print_bspline_edu(name: str, spl, u: np.ndarray, waypoints: np.ndarray,
                      degree: int, n_seg: int, duration: float, rms_deg: float):
    n_wp   = len(waypoints)
    n_cp_i = len(spl.c)
    n_cp_s = n_seg * (N_CP_SEG - 1) + 1
    knots  = spl.t
    interp = knots[degree + 1: -(degree + 1)]
    W = 66

    def row(label, val, note=''):
        return f'  ║  {label:<23}: {str(val):<10} {note:<{W-38}}║'

    print(f'\n  ╔{"═"*W}╗')
    print(f'  ║{"B-SPLINE ANALYSIS  [" + name + "]":^{W}}║')
    print(f'  ╠{"═"*W}╣')
    print(row('Waypoints',          n_wp,      '(each = 6-DOF joint configuration)'))
    print(row('Interp control pts', n_cp_i,    '(= N waypoints — exact interpolation)'))
    print(row('Degree',             degree,    '(cubic)' if degree == 3 else '(reduced)'))
    print(row('Knot vector length', len(knots),'(= N_cp + degree + 1)'))
    print(row('Segment CPs total',  n_cp_s,    '(step_8/9/10 compatible)'))
    print(row('Duration',           f'{duration:.3f}s', ''))
    print(row('LS RMS residual',    f'{rms_deg:.5f}°',  '(< 0.1° = good fit)'))

    print(f'  ╠{"═"*W}╣')
    print(f'  ║{"CHORD-LENGTH PARAMETERISATION":^{W}}║')
    print(f'  ║  {"WP":>4}  │ {"Label":<6} │ {"u value":>9} │ {"Jt-dist (°)":>13} │ {"time (s)":>11} ║')
    print(f'  ║  {"─"*4}  ┼ {"─"*6} ┼ {"─"*9} ┼ {"─"*13} ┼ {"─"*11} ║')
    for i, ui in enumerate(u):
        dist_d = 0.0 if i == 0 else float(np.degrees(
            np.linalg.norm(waypoints[i] - waypoints[i - 1])))
        label  = 'START' if i == 0 else ('END  ' if i == n_wp - 1 else f'WP {i:2d}')
        print(f'  ║  {i:4d}  │ {label} │ {ui:9.5f} │ {dist_d:11.3f}°  │ {ui*duration:9.3f}s  ║')

    print(f'  ╠{"═"*W}╣')
    print(f'  ║{"KNOT VECTOR":^{W}}║')
    print(f'  ║  First {degree+1} = 0.000  (clamped start — spline starts exactly at WP0)  ║')
    print(f'  ║  Last  {degree+1} = 1.000  (clamped end   — spline ends exactly at WP{n_wp-1})   ║')
    if len(interp) > 0:
        s = '  '.join(f'{v:.4f}' for v in interp[:8])
        sfx = f' ... ({len(interp)-8} more)' if len(interp) > 8 else ''
        print(f'  ║  Interior [{len(interp)}]: {s}{sfx}')
        print(f'  ║  (placed by de Boor averaging — concentrates knots near fast changes)')
    else:
        print(f'  ║  No interior knots → max flexibility with {n_cp_i} CPs')

    print(f'  ╠{"═"*W}╣')
    print(f'  ║{"INTERPOLATION VERIFICATION":^{W}}║')
    for i, ui in enumerate(u):
        err = float(np.max(np.abs(np.degrees(spl(float(ui)) - waypoints[i]))))
        ok  = err < 0.01
        lbl = 'START' if i == 0 else ('END  ' if i == n_wp - 1 else f'WP {i:2d}')
        note = '(OK — exact)' if ok else '(WARN)'
        print(f'  ║  {"✓" if ok else "✗"}  {lbl}  u={ui:.5f}  err={err:.6f}°  '
              f'{note:>{W-42}}║')

    print(f'  ╚{"═"*W}╝')


# ═════════════════════════════════════════════════════════════════════════════
# VISUALIZATION  (3 figures)
# ═════════════════════════════════════════════════════════════════════════════

def make_plots(name: str, pos: np.ndarray, vel: np.ndarray, t: np.ndarray,
               spl, u: np.ndarray, waypoints: np.ndarray,
               cp_segs: np.ndarray, n_seg: int, duration: float,
               base: np.ndarray, show: bool):
    try:
        import matplotlib
        matplotlib.use('TkAgg' if show else 'Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa
    except ImportError:
        print('  (matplotlib not available — skipping plots)')
        return

    n_wp  = len(waypoints)
    t_wp  = u * duration

    n_global = n_seg * (N_CP_SEG - 1) + 1
    cp_flat  = np.zeros((n_global, NDOF))
    for seg in range(n_seg):
        i0 = seg * (N_CP_SEG - 1)
        cp_flat[i0: i0 + N_CP_SEG] = cp_segs[seg]
    t_cp = np.linspace(0.0, duration, n_global)

    knot_t = spl.t * duration
    int_kt = [kt for kt in knot_t if 0.02 < kt < duration - 0.02]

    # ── Fig 1 : joint angles ─────────────────────────────────────────────────
    fig1, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    fig1.suptitle(
        f'[{name}]  Joint Angles\n'
        f'{n_wp} waypoints  |  degree={spl.k}  |  {duration:.2f}s',
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
        ax.set_ylabel('Angle (°)', fontsize=8)
        ax.set_title(f'J{j+1}', fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.25); ax.tick_params(labelsize=7)
        if j == 0: ax.legend(fontsize=7)
    for ax in axes[1]: ax.set_xlabel('Time (s)', fontsize=8)
    fig1.text(0.5, 0.01,
              'Red●=waypoints  Green■--=ctrl polygon  Orange⋮=knots  Grey│=segments',
              ha='center', fontsize=8, color='#444')
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    f1 = f'{name}_joint_trajectories.png'
    fig1.savefig(f1, dpi=150, bbox_inches='tight')
    print(f'  Saved: {f1}')
    if show: plt.show()
    plt.close(fig1)

    # ── Fig 2 : joint velocities ─────────────────────────────────────────────
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    fig2.suptitle(f'[{name}]  Joint Velocities  (red dashes = hardware limit)',
                  fontsize=11, fontweight='bold')
    for j, ax in enumerate(axes2.flat):
        ax.plot(t, np.degrees(vel[:, j]), color='steelblue', lw=1.5)
        lim = float(np.degrees(VEL_LIM[j]))
        ax.axhline( lim, color='red', ls='--', lw=1.0, label=f'±{lim:.1f}°/s')
        ax.axhline(-lim, color='red', ls='--', lw=1.0)
        ax.axhline(0,    color='grey', ls='-', lw=0.5, alpha=0.5)
        ax.set_ylabel('Vel (°/s)', fontsize=8)
        ax.set_title(f'J{j+1}', fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.25); ax.tick_params(labelsize=7)
        if j == 0: ax.legend(fontsize=7)
    for ax in axes2[1]: ax.set_xlabel('Time (s)', fontsize=8)
    plt.tight_layout()
    f2 = f'{name}_joint_velocities.png'
    fig2.savefig(f2, dpi=150, bbox_inches='tight')
    print(f'  Saved: {f2}')
    if show: plt.show()
    plt.close(fig2)

    # ── Fig 3 : EE 3D path ───────────────────────────────────────────────────
    ee     = np.array([fk_pos(pos[k], base) for k in range(len(pos))])
    wp_ee  = np.array([fk_pos(waypoints[i], base) for i in range(n_wp)])

    fig3   = plt.figure(figsize=(13, 5))
    fig3.suptitle(f'[{name}]  End-Effector Path  (world frame)',
                  fontsize=11, fontweight='bold')

    ax3d = fig3.add_subplot(1, 2, 1, projection='3d')
    ax3d.plot(ee[:, 0], ee[:, 1], ee[:, 2],
              color='steelblue', lw=2.0, label='EE path')
    ax3d.scatter(wp_ee[:, 0], wp_ee[:, 1], wp_ee[:, 2],
                 color='red', s=80, zorder=5, label='Waypoint EE')
    ax3d.scatter(*base, color='black', s=140, marker='^', zorder=6, label='Base')
    # colour-code by time
    sc = ax3d.scatter(ee[::10, 0], ee[::10, 1], ee[::10, 2],
                      c=np.linspace(0, 1, len(ee[::10])),
                      cmap='plasma', s=20, zorder=4, label='_')
    fig3.colorbar(sc, ax=ax3d, fraction=0.03, label='normalised time')
    ax3d.set_xlabel('X (m)'); ax3d.set_ylabel('Y (m)'); ax3d.set_zlabel('Z (m)')
    ax3d.set_title('3D path', fontsize=9)
    ax3d.legend(fontsize=7)

    # XY projection
    ax_xy = fig3.add_subplot(1, 2, 2)
    ax_xy.plot(ee[:, 0], ee[:, 1], color='steelblue', lw=2.0)
    ax_xy.scatter(wp_ee[:, 0], wp_ee[:, 1], color='red', s=80, zorder=5)
    ax_xy.scatter(base[0], base[1], color='black', s=140, marker='^', zorder=6,
                  label='Base')
    for i, (x, y) in enumerate(wp_ee[:, :2]):
        ax_xy.annotate(f'WP{i}', (x, y), textcoords='offset points',
                       xytext=(4, 4), fontsize=7, color='red')
    ax_xy.set_xlabel('X (m)'); ax_xy.set_ylabel('Y (m)')
    ax_xy.set_title('XY projection (top view)', fontsize=9)
    ax_xy.grid(True, alpha=0.25); ax_xy.set_aspect('equal', 'datalim')
    ax_xy.legend(fontsize=7)

    plt.tight_layout()
    f3 = f'{name}_ee_path.png'
    fig3.savefig(f3, dpi=150, bbox_inches='tight')
    print(f'  Saved: {f3}')
    if show: plt.show()
    plt.close(fig3)


# ═════════════════════════════════════════════════════════════════════════════
# INPUT HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _parse_joints(raw: str, use_deg: bool) -> Optional[np.ndarray]:
    try:
        vals = [float(v) for v in raw.replace(',', ' ').split()]
        if len(vals) != NDOF:
            print(f'    ✗  Need exactly {NDOF} values, got {len(vals)}')
            return None
        q = np.radians(np.array(vals)) if use_deg else np.array(vals, dtype=float)
        return np.clip(q, POS_LIM[:, 0], POS_LIM[:, 1])
    except ValueError as e:
        print(f'    ✗  Parse error: {e}')
        return None


def prompt_arm_choice() -> str:
    bar = '═' * 68
    print(f'\n{bar}')
    print(f'  SELECT ARM')
    print(f'{bar}')
    for idx, n in enumerate(ROBOT_NAMES):
        base_str = str(ROBOT_BASES[n].tolist())
        print(f'  [{idx+1}]  {n}   base = {base_str}')
    print()
    while True:
        raw = input(f'  Which arm? [1..{len(ROBOT_NAMES)}]  (Enter = 1): ').strip()
        if raw == '':
            chosen = ROBOT_NAMES[0]; break
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(ROBOT_NAMES):
                chosen = ROBOT_NAMES[idx]; break
        except ValueError:
            pass
        print(f'  Enter 1 – {len(ROBOT_NAMES)}.')
    print(f'  → Selected: {chosen}')
    return chosen


def prompt_waypoints(name: str) -> np.ndarray:
    bar = '─' * 64
    base_str = str(ROBOT_BASES[name].tolist())
    print(f'\n  ┌{bar}┐')
    print(f'  │  ARM: {name.upper():<57}│')
    print(f'  │  base = {base_str:<55}│')
    print(f'  └{bar}┘')

    u_raw   = input('  Unit? [D]egrees / [R]adians  (Enter = D): ').strip().lower()
    use_deg = (u_raw != 'r')
    ustr    = 'degrees' if use_deg else 'radians'
    print(f'  → Using {ustr}')

    while True:
        n_raw = input(f'  How many waypoints? (min 2, includes START + END): ').strip()
        try:
            n_wp = int(n_raw)
            if n_wp >= 2: break
            print('  Need at least 2 waypoints.')
        except ValueError:
            print('  Enter an integer.')

    print(f'\n  Enter {NDOF} joint values per waypoint ({ustr}), space- or comma-separated.')
    if use_deg:
        print('  Limits: J1=±360°  J2=±94.5°  J3=±159.9°  J4/5/6=±360°')
    print()

    waypoints: List[np.ndarray] = []
    for i in range(n_wp):
        label = 'START' if i == 0 else ('END  ' if i == n_wp - 1 else f'WP {i:2d}')
        while True:
            raw = input(f'  [{label}]  J1..J6: ').strip()
            q   = _parse_joints(raw, use_deg)
            if q is not None:
                waypoints.append(q)
                deg_str = '  '.join(f'{v:7.2f}°' for v in np.degrees(q))
                print(f'           → deg: {deg_str}')
                break

    return np.array(waypoints)


# ═════════════════════════════════════════════════════════════════════════════
# B-SPLINE PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def build_trajectory(name: str, waypoints: np.ndarray,
                     dur_req: float, show: bool) -> Dict:
    """Fit B-spline, scale duration, build plots. Returns full trajectory dict."""
    n_wp = len(waypoints)
    base = ROBOT_BASES[name]

    print(f'\n  ── B-SPLINE FIT  [{name}]  ({n_wp} waypoints) ──────────────────')
    spl, u, degree = fit_interp_bspline(waypoints)
    print(f'  Degree={degree}  interp_CPs={len(spl.c)}  knots={len(spl.t)}')

    duration = adaptive_duration(waypoints, dur_req)
    pos, vel, acc, t, duration = scale_if_needed(spl, duration)
    n_steps = len(pos)
    print(f'  Duration={duration:.3f}s  samples={n_steps}')

    n_seg           = n_seg_for_waypoints(n_wp)
    cp_segs, rms    = ls_fit_segments(pos, n_seg)
    n_global        = n_seg * (N_CP_SEG - 1) + 1
    print(f'  Segments={n_seg}×{N_CP_SEG}CPs  total={n_global}  LS_residual={rms:.5f}°')

    arc_fracs = np.linspace(0.0, 1.0, n_steps)
    ee_path   = np.array([fk_pos(pos[k], base) for k in range(n_steps)])
    path_len  = float(np.sum(np.linalg.norm(np.diff(ee_path, axis=0), axis=1)))

    print_bspline_edu(name, spl, u, waypoints, degree, n_seg, duration, rms)

    print(f'\n  Generating plots ...')
    make_plots(name, pos, vel, t, spl, u, waypoints, cp_segs,
               n_seg, duration, base, show)

    seg_info = []
    for seg in range(n_seg):
        seg_info.append({
            'segment'  : seg,
            'arc_start': round(seg / n_seg, 4),
            'arc_end'  : round((seg + 1) / n_seg, 4),
            'cp'       : cp_segs[seg].tolist(),
        })

    return {
        'name'      : name,
        'base'      : base.tolist(),
        'duration'  : float(duration),
        'n_steps'   : int(n_steps),
        'positions' : pos.tolist(),
        'velocities': vel.tolist(),
        'times'     : t.tolist(),
        'arc_fracs' : arc_fracs.tolist(),
        'waypoints_deg': np.degrees(waypoints).tolist(),
        'ee_path'   : ee_path.tolist(),
        'path_len_m': round(path_len, 5),
        'ls_rms_deg': round(rms, 6),
        'chord_u'   : u.tolist(),
        'degree'    : degree,
        'n_seg'     : n_seg,
        'segments'  : seg_info,
    }


# ═════════════════════════════════════════════════════════════════════════════
# ROS2 EXECUTION NODE
# ═════════════════════════════════════════════════════════════════════════════

if _ROS_OK:
    class SingleArmExecutor(Node):
        def __init__(self, arm_name: str):
            super().__init__('step100_executor')
            self._name = arm_name
            topic = CTRL_TOPIC.format(arm=arm_name)
            self._pub = self.create_publisher(Float64MultiArray, topic, 10)
            self._cur_q: Optional[np.ndarray] = None

            for tpl in (JOINT_TOPIC_A, JOINT_TOPIC_B):
                self.create_subscription(
                    JointState, tpl.format(arm=arm_name),
                    self._js_cb, 10)

            self.get_logger().info(f'Publishing to: {topic}')

        def _js_cb(self, msg: 'JointState'):
            if len(msg.position) < NDOF:
                return
            jmap = {n: i for i, n in enumerate(msg.name)}
            keys = [f'joint_{k}' for k in range(1, NDOF + 1)]
            if all(k in jmap for k in keys):
                q = np.array([msg.position[jmap[k]] for k in keys])
            else:
                q = np.array(msg.position[:NDOF])
            self._cur_q = q.astype(float)

        def _send(self, q: np.ndarray):
            msg = Float64MultiArray()
            msg.data = [float(v) for v in q[DH_TO_GZ]]  # reorder DH->Gazebo
            self._pub.publish(msg)

        def execute(self, pos: np.ndarray, duration: float) -> Dict:
            dt_ns   = int(1e9 / RATE_HZ)
            n_steps = len(pos)
            self.get_logger().info(
                f'Executing {n_steps} steps at {RATE_HZ:.0f} Hz  ({duration:.2f}s) ...')

            t0w = time.monotonic()
            for k in range(n_steps):
                t0 = time.monotonic_ns()
                self._send(pos[k])
                rclpy.spin_once(self, timeout_sec=0.)
                rem = dt_ns - (time.monotonic_ns() - t0)
                if rem > 0:
                    time.sleep(rem * 1e-9)

            # Hold final pose
            self.get_logger().info(f'Holding final pose for {HOLD_S:.1f}s ...')
            hold_steps = int(HOLD_S * RATE_HZ)
            for _ in range(hold_steps):
                t0 = time.monotonic_ns()
                self._send(pos[-1])
                rclpy.spin_once(self, timeout_sec=0.)
                rem = dt_ns - (time.monotonic_ns() - t0)
                if rem > 0:
                    time.sleep(rem * 1e-9)

            wall = time.monotonic() - t0w
            fq   = self._cur_q.tolist() if self._cur_q is not None else None
            return {
                'success'        : True,
                'mode'           : 'ros2_gazebo',
                'steps_sent'     : int(n_steps),
                'wall_time_s'    : round(wall, 3),
                'final_joints_fb': fq,
                'error'          : None,
            }


# ═════════════════════════════════════════════════════════════════════════════
# DRY-RUN  (no ROS2)
# ═════════════════════════════════════════════════════════════════════════════

def dry_run(name: str, pos: np.ndarray, duration: float) -> Dict:
    n     = len(pos)
    mile  = max(1, n // 10)
    print(f'\n  DRY-RUN (no ROS2): {n} steps  ({duration:.2f}s  @{RATE_HZ:.0f}Hz)')
    for k in range(0, n, mile):
        pct = int(100 * k / n)
        deg = '  '.join(f'{np.degrees(v):6.2f}°' for v in pos[k])
        print(f'    {pct:3d}%  [{deg}]')
    deg = '  '.join(f'{np.degrees(v):6.2f}°' for v in pos[-1])
    print(f'    100%  [{deg}]')
    print(f'  (Hold {HOLD_S:.1f}s — skipped in dry-run)')
    return {
        'success'        : True,
        'mode'           : 'dry_run',
        'steps_sent'     : int(n),
        'wall_time_s'    : None,
        'final_joints_fb': pos[-1].tolist(),
        'error'          : None,
    }


# ═════════════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ═════════════════════════════════════════════════════════════════════════════

def print_final_report(traj: Dict, exec_res: Dict):
    bar  = '═' * 68
    thin = '─' * 68
    name = traj['name']
    base = np.array(traj['base'])
    pos  = np.array(traj['positions'])

    start_ee = fk_pos(pos[0],  base)
    end_ee   = fk_pos(pos[-1], base)

    planned_end_q = np.array(traj['waypoints_deg'][-1])
    actual_fb     = exec_res.get('final_joints_fb')

    print(f'\n{bar}')
    print(f'  STEP 100  —  FINAL REPORT  [{name}]')
    print(f'{bar}')
    print(f'\n  TRAJECTORY SUMMARY\n  {thin}')
    print(f'  Arm             : {name}')
    print(f'  Base (world)    : {base.tolist()}')
    print(f'  Waypoints       : {len(traj["waypoints_deg"])}')
    print(f'  Duration        : {traj["duration"]:.3f}s')
    print(f'  Samples         : {traj["n_steps"]}  (@{RATE_HZ:.0f}Hz)')
    print(f'  EE path length  : {traj["path_len_m"]*100:.2f}cm')
    print(f'  B-spline degree : {traj["degree"]}')
    print(f'  LS fit residual : {traj["ls_rms_deg"]:.5f}°')

    print(f'\n  END-EFFECTOR POSITIONS\n  {thin}')
    print(f'  Start  EE (world): [{", ".join(f"{v:.4f}" for v in start_ee)}] m')
    print(f'  Target EE (world): [{", ".join(f"{v:.4f}" for v in end_ee)}] m')

    print(f'\n  JOINT ANGLES  (target end)\n  {thin}')
    print(f'  Planned:  {" ".join(f"{v:7.2f}°" for v in planned_end_q)}')
    if actual_fb is not None:
        actual_deg = [float(np.degrees(v)) for v in actual_fb]
        errs       = [abs(actual_deg[j] - planned_end_q[j]) for j in range(NDOF)]
        print(f'  Feedback: {" ".join(f"{v:7.2f}°" for v in actual_deg)}')
        print(f'  Error:    {" ".join(f"{v:7.2f}°" for v in errs)}')
        max_err = max(errs)
        ok = '✅' if max_err < 3.0 else '⚠ '
        print(f'  Max joint error: {max_err:.2f}°  {ok}')
    else:
        print(f'  Feedback: (not available — dry-run or no joint state received)')

    print(f'\n  EXECUTION\n  {thin}')
    icon = '✅' if exec_res['success'] else '❌'
    print(f'  Result    : {icon}  {"SUCCESS" if exec_res["success"] else "FAILED"}')
    print(f'  Mode      : {exec_res["mode"]}')
    print(f'  Steps sent: {exec_res["steps_sent"]}')
    if exec_res.get('wall_time_s'):
        print(f'  Wall time : {exec_res["wall_time_s"]:.2f}s  '
              f'(trajectory={traj["duration"]:.2f}s + hold={HOLD_S:.1f}s)')
    if exec_res.get('error'):
        print(f'  Error     : {exec_res["error"]}')

    print(f'\n  PLOTS SAVED\n  {thin}')
    for suffix in ('joint_trajectories', 'joint_velocities', 'ee_path'):
        fname = f'{name}_{suffix}.png'
        exists = '✅' if os.path.exists(fname) else '  '
        print(f'  {exists}  {fname}')

    print(f'\n{bar}')
    print(f'  {"✅  DONE" if exec_res["success"] else "❌  EXECUTION FAILED"}')
    print(f'{bar}\n')


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main(args=None):
    show = '--show' in sys.argv

    bar = '=' * 68
    print('\n' + bar)
    print('  STEP 100  —  Single-Arm B-Spline + Gazebo Execution')
    print('  All-in-one: waypoints → spline → plots → execute')
    print(bar)
    if not _ROS_OK:
        print('\n  ⚠   ROS2 not found — will run in DRY-RUN mode (no Gazebo)')
    if show:
        print('  [--show]  Interactive matplotlib windows enabled.')
    else:
        print('  Run with --show to open interactive plot windows.')

    # 1. Select arm
    name = prompt_arm_choice()

    # 2. Duration
    dur_raw = input('\n  Minimum trajectory duration [s]  (Enter = 8.0s): ').strip()
    dur_req = float(dur_raw) if dur_raw else 8.0

    # 3. Waypoints
    waypoints = prompt_waypoints(name)

    # 4. Build trajectory (spline + plots)
    traj = build_trajectory(name, waypoints, dur_req, show)

    pos      = np.array(traj['positions'])
    duration = traj['duration']

    # 5. Confirm before sending to Gazebo
    print(f'\n  {bar}')
    print(f'  Ready to execute:')
    print(f'    Arm      : {name}')
    print(f'    Duration : {duration:.2f}s  ({len(pos)} steps @{RATE_HZ:.0f}Hz)')
    print(f'    Hold     : {HOLD_S:.1f}s after trajectory')
    print(f'    Mode     : {"ROS2 → Gazebo" if _ROS_OK else "DRY-RUN"}')
    print(f'  {bar}')
    ans = input('\n  Send to Gazebo? [Y/n]: ').strip().lower()
    if ans == 'n':
        print('  Aborted.  Trajectory saved to plots only.')
        return

    # 6. Execute
    if _ROS_OK:
        rclpy.init(args=args)
        node = SingleArmExecutor(name)
        try:
            exec_res = node.execute(pos, duration)
        except KeyboardInterrupt:
            exec_res = {
                'success': False, 'mode': 'interrupted',
                'steps_sent': 0, 'wall_time_s': None,
                'final_joints_fb': None, 'error': 'KeyboardInterrupt',
            }
        except Exception as e:
            exec_res = {
                'success': False, 'mode': 'error',
                'steps_sent': 0, 'wall_time_s': None,
                'final_joints_fb': None, 'error': str(e),
            }
        finally:
            try: node.destroy_node()
            except: pass
            rclpy.shutdown()
    else:
        exec_res = dry_run(name, pos, duration)

    # 7. Save report JSON
    report = {**traj, 'execution': exec_res}
    report_file = f'{name}_step100_report.json'
    with open(report_file, 'w') as fh:
        json.dump(report, fh, indent=2)

    print_final_report(traj, exec_res)
    kb = os.path.getsize(report_file) / 1024.0
    print(f'  Report saved: {report_file}  ({kb:.1f} KB)\n')


if __name__ == '__main__':
    main()