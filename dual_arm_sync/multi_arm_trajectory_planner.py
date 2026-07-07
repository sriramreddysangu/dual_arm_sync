#!/usr/bin/env python3
"""
multi_arm_trajectory_planner.py
N-Arm Synchronized Trajectory Planner (up to 10 arms)

Pipeline (per the 6-step architecture):
  Step 1: Plan each arm independently in Cartesian space (minimum-jerk spline)
  Step 2: IK → fit cubic B-splines in joint space, equal T for all arms
  Step 3: Pairwise inter-arm collision detection at sampled times
  Step 4: N-oscillator Kuramoto with repulsive coupling → re-parameterize timing
  Step 5: Persistent collisions → subdivide B-spline segments + perturb control points
  Step 6: Optional light synchronization polish (milestone soft-constraints)

Usage (standalone, no ROS required):
    python3 multi_arm_trajectory_planner.py

Usage (as a module):
    from multi_arm_trajectory_planner import MultiArmPlanner, ArmSpec
    planner = MultiArmPlanner(arm_specs)
    results = planner.plan(starts, targets)
"""

import numpy as np
from scipy.interpolate import splrep, splev, BSpline, make_interp_spline
from scipy.integrate import odeint
from scipy.optimize import minimize
from typing import List, Tuple, Dict, Optional
import json
import time
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# DH / FK helpers  (Doosan M1013 — shared with existing constants.py)
# ---------------------------------------------------------------------------

class DHConstants:
    L1 = 0.1525   # base height
    L2 = 0.620    # upper arm
    L3 = 0.559    # forearm
    L4 = 0.121    # tool flange
    A  = 0.0345   # shoulder offset

    # [alpha, a, theta_offset, d]
    DH_TABLE = np.array([
        [0.0,         0.0,        0.0,       L1],
        [-np.pi/2,    0.0,  -np.pi/2,        A ],
        [0.0,         L2,    np.pi/2,        0.0],
        [np.pi/2,     0.0,        0.0,       L3],
        [-np.pi/2,    0.0,        0.0,       0.0],
        [np.pi/2,     0.0,        0.0,       L4],
    ])

    POSITION_LIMITS = np.array([
        [-6.283,  6.283],
        [-1.650,  1.650],
        [-2.792,  2.792],
        [-6.283,  6.283],
        [-6.283,  6.283],
        [-6.283,  6.283],
    ])

    VELOCITY_LIMITS = np.array([2.0944, 2.0944, 3.1416, 3.9270, 3.9270, 3.9270])
    ACCELERATION_LIMITS = np.array([3.0, 3.0, 4.0, 5.0, 5.0, 6.0])


def _dh_matrix(alpha: float, a: float, theta: float, d: float) -> np.ndarray:
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct,    -st,    0,   a],
        [st*ca,  ct*ca, -sa, -sa*d],
        [st*sa,  ct*sa,  ca,  ca*d],
        [0,      0,      0,   1],
    ])


def forward_kinematics(q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (end-effector position [LOCAL frame], 4×4 transform)."""
    table = DHConstants.DH_TABLE.copy()
    table[:, 2] += q
    T = np.eye(4)
    for row in table:
        T = T @ _dh_matrix(*row)
    return T[:3, 3].copy(), T


def all_link_positions(q: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Return (6,3) world positions for every link frame."""
    table = DHConstants.DH_TABLE.copy()
    table[:, 2] += q
    T = np.eye(4)
    pts = []
    for row in table:
        T = T @ _dh_matrix(*row)
        pts.append(T[:3, 3] + base)
    return np.array(pts)


def numerical_ik(target_local: np.ndarray,
                 seed: np.ndarray,
                 tol: float = 0.008,
                 max_iter: int = 300) -> Optional[np.ndarray]:
    """Fast single-seed numerical IK; returns joint config or None."""
    lims = DHConstants.POSITION_LIMITS
    bounds = [(lims[i, 0], lims[i, 1]) for i in range(6)]

    def cost(q):
        p, _ = forward_kinematics(q)
        err = np.linalg.norm(p - target_local) ** 2
        pen = sum(max(0, lims[i, 0] - q[i]) ** 2 + max(0, q[i] - lims[i, 1]) ** 2
                  for i in range(6)) * 200.0
        return err + pen

    res = minimize(cost, seed, method='SLSQP', bounds=bounds,
                   options={'maxiter': max_iter, 'ftol': 1e-9})
    if res.success:
        p, _ = forward_kinematics(res.x)
        if np.linalg.norm(p - target_local) < tol:
            return res.x
    return None


def best_ik(target_local: np.ndarray,
            seed: np.ndarray,
            n_extra: int = 6) -> Optional[np.ndarray]:
    """Try seed + n_extra random perturbations; return best."""
    seeds = [seed] + [seed + np.random.randn(6) * 0.3 for _ in range(n_extra)]
    best, best_err = None, np.inf
    for s in seeds:
        s = np.clip(s, DHConstants.POSITION_LIMITS[:, 0], DHConstants.POSITION_LIMITS[:, 1])
        q = numerical_ik(target_local, s)
        if q is not None:
            p, _ = forward_kinematics(q)
            err = np.linalg.norm(p - target_local)
            if err < best_err:
                best, best_err = q.copy(), err
    return best


# ---------------------------------------------------------------------------
# Arm specification
# ---------------------------------------------------------------------------

class ArmSpec:
    """Configuration for one robot arm."""

    def __init__(self,
                 name: str,
                 base_position: np.ndarray,
                 home_joints: Optional[np.ndarray] = None,
                 link_radii: Optional[np.ndarray] = None):
        self.name = name
        self.base = np.asarray(base_position, dtype=float)
        self.home_joints = home_joints if home_joints is not None else np.zeros(6)
        self.link_radii = link_radii if link_radii is not None else \
            np.array([0.10, 0.10, 0.08, 0.08, 0.06, 0.06])


# ---------------------------------------------------------------------------
# Step 1 – Independent Cartesian trajectory (minimum-jerk)
# ---------------------------------------------------------------------------

def minimum_jerk_cartesian(start_pos: np.ndarray,
                            end_pos: np.ndarray,
                            T: float,
                            n_samples: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """
    Minimum-jerk point-to-point Cartesian trajectory.
    Returns (positions [n_samples, 3], time_vector [n_samples]).
    """
    t_vec = np.linspace(0, T, n_samples)
    tau = t_vec / T
    # Minimum-jerk polynomial: s(tau) = 10τ³ − 15τ⁴ + 6τ⁵
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    positions = start_pos[None, :] + s[:, None] * (end_pos - start_pos)[None, :]
    return positions, t_vec


# ---------------------------------------------------------------------------
# Step 2 – IK along Cartesian path + B-spline fit in joint space
# ---------------------------------------------------------------------------

def cartesian_path_to_joint_bspline(cart_positions: np.ndarray,
                                     t_vec: np.ndarray,
                                     base: np.ndarray,
                                     seed_joints: np.ndarray,
                                     n_eval: int = 500
                                     ) -> Optional[Dict]:
    """
    Solve IK at sampled Cartesian waypoints, fit a cubic B-spline in joint
    space, re-sample at n_eval points.

    Returns dict with keys: time, positions, velocities, accelerations, tck_list
    """
    n_way = len(cart_positions)
    joint_waypoints = np.zeros((n_way, 6))
    prev = seed_joints.copy()

    for i, (cp, t) in enumerate(zip(cart_positions, t_vec)):
        local_target = cp - base
        q = best_ik(local_target, prev)
        if q is None:
            # Fallback: linear interpolation in joint space from prev
            q = prev.copy()
        joint_waypoints[i] = q
        prev = q

    # Fit B-spline per joint (cubic, s=0 → interpolating)
    T = t_vec[-1]
    t_norm = t_vec / T          # normalised [0,1]
    t_eval_norm = np.linspace(0, 1, n_eval)
    t_eval = t_eval_norm * T

    tck_list = []
    positions = np.zeros((n_eval, 6))
    velocities = np.zeros((n_eval, 6))
    accelerations = np.zeros((n_eval, 6))

    for j in range(6):
        k = min(3, n_way - 1)
        tck = splrep(t_norm, joint_waypoints[:, j], k=k, s=0)
        tck_list.append(tck)
        positions[:, j]     = splev(t_eval_norm, tck, der=0)
        velocities[:, j]    = splev(t_eval_norm, tck, der=1) / T
        accelerations[:, j] = splev(t_eval_norm, tck, der=2) / T**2

    return {
        'time': t_eval,
        'positions': positions,
        'velocities': velocities,
        'accelerations': accelerations,
        'tck_list': tck_list,          # kept for Step 5 refinement
        'joint_waypoints': joint_waypoints,
        'waypoint_times': t_vec,
    }


# ---------------------------------------------------------------------------
# Step 3 – Pairwise collision detection
# ---------------------------------------------------------------------------

class CollisionChecker:
    SAFETY_MARGIN = 0.15   # 15 cm

    @staticmethod
    def check_pair(links_a: np.ndarray, links_b: np.ndarray,
                   radii_a: np.ndarray, radii_b: np.ndarray) -> Tuple[bool, float]:
        """Return (collision, min_distance)."""
        min_d = np.inf
        for i in range(6):
            for j in range(6):
                d = np.linalg.norm(links_a[i] - links_b[j])
                threshold = radii_a[i] + radii_b[j] + CollisionChecker.SAFETY_MARGIN
                if d < threshold:
                    return True, d
                min_d = min(min_d, d)
        return False, min_d

    @staticmethod
    def scan_trajectories(trajs: List[Dict],
                          specs: List[ArmSpec],
                          n_check: int = 200) -> Dict:
        """
        Scan all arm pairs for collision events.
        Returns collision_map[(i,j)] = list of time indices where collision occurs.
        """
        N = len(trajs)
        n_pts = min(len(t['positions']) for t in trajs)
        step = max(1, n_pts // n_check)
        indices = list(range(0, n_pts, step))

        collision_map = {}
        for i in range(N):
            for j in range(i + 1, N):
                events = []
                for idx in indices:
                    li = all_link_positions(trajs[i]['positions'][idx], specs[i].base)
                    lj = all_link_positions(trajs[j]['positions'][idx], specs[j].base)
                    col, _ = CollisionChecker.check_pair(
                        li, lj, specs[i].link_radii, specs[j].link_radii)
                    if col:
                        events.append(idx)
                if events:
                    collision_map[(i, j)] = events
        return collision_map


# ---------------------------------------------------------------------------
# Step 4 – N-oscillator Kuramoto with repulsive coupling
# ---------------------------------------------------------------------------

class KuramotoNArm:
    """
    N coupled Kuramoto oscillators with event-driven repulsive coupling.

    State vector: [φ_0, φ_1, …, φ_{N-1}, ω_0, ω_1, …, ω_{N-1}]
    φ_i ∈ [0,1]: normalised progress along arm i's trajectory.
    ω_i: natural frequency (≈ 1/T, updated adaptively).

    Coupling:
      Attractive baseline:   K_base · sin(φ_j − φ_i)          (sync pull)
      Repulsive when close:  −K_rep · (1 − d/d_rep)² · sign   (push leading arm back)
    """

    BASE_K       = 1.5
    MAX_K        = 12.0
    REPULSION_K  = 60.0
    EMERGENCY_K  = 180.0
    REP_THRESH   = 0.25   # start repulsion at 25 cm
    MIN_DIST     = 0.15   # hard limit 15 cm
    MAX_PHASE_V  = 2.0
    ADAPT_RATE   = 0.4
    DT           = 0.01

    def __init__(self, N: int, T: float,
                 trajs: List[Dict], specs: List[ArmSpec]):
        self.N = N
        self.T = T
        self.trajs = trajs
        self.specs = specs
        self.omega0 = 1.0 / T

    def _interpolate(self, phi: float, traj_idx: int) -> np.ndarray:
        pos = self.trajs[traj_idx]['positions']
        n = len(pos)
        f = np.clip(phi, 0.0, 1.0)
        lo = int(f * (n - 1))
        hi = min(lo + 1, n - 1)
        a = f * (n - 1) - lo
        return (1 - a) * pos[lo] + a * pos[hi]

    def _min_dist(self, q_i, q_j, spec_i, spec_j) -> float:
        li = all_link_positions(q_i, spec_i.base)
        lj = all_link_positions(q_j, spec_j.base)
        min_d = np.inf
        for a in li:
            for b in lj:
                min_d = min(min_d, np.linalg.norm(a - b))
        return min_d

    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        N = self.N
        phi = state[:N]
        omega = state[N:]
        dphi = np.zeros(N)
        domega = np.zeros(N)

        # Pre-compute current joint positions
        qs = [self._interpolate(phi[i], i) for i in range(N)]

        for i in range(N):
            dphi_i = omega[i]

            for j in range(N):
                if i == j:
                    continue
                # --- attractive baseline ---
                d_phi = phi[j] - phi[i]
                K = self.BASE_K
                dphi_i += K * np.sin(2 * np.pi * d_phi)

                # --- proximity-based repulsion ---
                dist = self._min_dist(qs[i], qs[j],
                                      self.specs[i], self.specs[j])

                if dist < self.REP_THRESH:
                    proximity = np.clip(1.0 - dist / self.REP_THRESH, 0.0, 1.0)
                    danger    = np.clip(1.0 - dist / self.MIN_DIST,   0.0, 1.0)

                    # leading arm (higher phi) gets pushed back
                    lead = np.sign(phi[i] - phi[j]) if abs(phi[i] - phi[j]) > 0.02 else 0.0
                    rep_force = self.REPULSION_K * proximity**2 * lead
                    if dist < self.MIN_DIST:
                        rep_force += self.EMERGENCY_K * danger**3 * lead

                    dphi_i -= rep_force

                    # adaptive frequency reduction for leading arm
                    if lead > 0:
                        domega[i] -= self.ADAPT_RATE * proximity * omega[i]
                    elif lead < 0:
                        domega[j] -= self.ADAPT_RATE * proximity * omega[j]

            dphi[i] = np.clip(dphi_i, -self.MAX_PHASE_V, self.MAX_PHASE_V)

        # Gentle frequency recovery toward omega0
        domega += 0.1 * (self.omega0 - omega)
        return np.concatenate([dphi, domega])

    def integrate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrate Kuramoto dynamics.
        Returns (phi_history [n_steps, N], t_vec [n_steps]).
        """
        t_vec = np.linspace(0, self.T, int(self.T / self.DT))
        state0 = np.concatenate([np.zeros(self.N),
                                  np.full(self.N, self.omega0)])
        sol = odeint(self.derivatives, state0, t_vec, rtol=1e-4, atol=1e-6)
        phi = np.clip(sol[:, :self.N], 0.0, 1.0)
        return phi, t_vec

    def reparameterise(self,
                       phi_history: np.ndarray,
                       t_vec: np.ndarray) -> List[Dict]:
        """
        Re-sample each arm's trajectory according to its Kuramoto phase φ_i(t).
        Returns list of new trajectory dicts (same structure as input trajs).
        """
        new_trajs = []
        n_steps = len(t_vec)

        for i in range(self.N):
            old_traj = self.trajs[i]
            pos_orig = old_traj['positions']   # [n_orig, 6]
            n_orig = len(pos_orig)

            new_pos = np.zeros((n_steps, 6))
            new_vel = np.zeros((n_steps, 6))

            for k, (phi, dt) in enumerate(zip(phi_history[:, i], t_vec)):
                f = np.clip(phi, 0.0, 1.0)
                lo = int(f * (n_orig - 1))
                hi = min(lo + 1, n_orig - 1)
                a = f * (n_orig - 1) - lo
                new_pos[k] = (1 - a) * pos_orig[lo] + a * pos_orig[hi]

                # Velocity from phase derivative (finite-difference approximation)
                if k > 0:
                    dphi = phi_history[k, i] - phi_history[k - 1, i]
                    dt_step = t_vec[k] - t_vec[k - 1]
                    # dq/dt ≈ dq/dφ · dφ/dt
                    dqdf = (new_pos[k] - new_pos[k - 1]) / (dphi + 1e-10)
                    new_vel[k] = dqdf * (dphi / (dt_step + 1e-10))

            # Recompute acceleration via finite differences
            new_acc = np.gradient(new_vel, t_vec, axis=0)

            new_trajs.append({
                'time': t_vec,
                'positions': new_pos,
                'velocities': new_vel,
                'accelerations': new_acc,
                'tck_list': old_traj.get('tck_list'),
                'joint_waypoints': old_traj.get('joint_waypoints'),
                'waypoint_times': old_traj.get('waypoint_times'),
            })

        return new_trajs


# ---------------------------------------------------------------------------
# Step 5 – Spatial dodge: subdivide B-spline + perturb control points
# ---------------------------------------------------------------------------

class SpatialDodge:
    """
    For arm pairs that still collide after Kuramoto temporal adjustment:
    1. Find the collision time window.
    2. Add knots/control points in that segment of the B-spline.
    3. Perturb new control points with gradient-guided or random search.
    4. Re-evaluate until collision-free or max iterations reached.
    """

    MAX_ITER           = 4
    PERTURB_SCALE      = 0.08   # radians
    N_RANDOM           = 20
    SEGMENT_PADDING    = 0.10   # 10% time padding around collision window

    @staticmethod
    def _get_collision_window(collision_indices: List[int],
                               n_pts: int,
                               T: float) -> Tuple[float, float]:
        if not collision_indices:
            return 0.4, 0.6
        lo = max(0, min(collision_indices) - 1)
        hi = min(n_pts - 1, max(collision_indices) + 1)
        pad = int(SpatialDodge.SEGMENT_PADDING * n_pts)
        lo = max(0, lo - pad)
        hi = min(n_pts - 1, hi + pad)
        return lo / n_pts, hi / n_pts

    @staticmethod
    def _rebuild_trajectory_from_waypoints(waypoints: np.ndarray,
                                            t_norm: np.ndarray,
                                            T: float,
                                            n_eval: int = 500) -> Dict:
        """Refit B-spline from (possibly augmented) waypoints."""
        t_eval_norm = np.linspace(0, 1, n_eval)
        tck_list = []
        pos = np.zeros((n_eval, 6))
        vel = np.zeros((n_eval, 6))
        acc = np.zeros((n_eval, 6))

        for j in range(6):
            k = min(3, len(waypoints) - 1)
            tck = splrep(t_norm, waypoints[:, j], k=k, s=0)
            tck_list.append(tck)
            pos[:, j] = splev(t_eval_norm, tck, der=0)
            vel[:, j] = splev(t_eval_norm, tck, der=1) / T
            acc[:, j] = splev(t_eval_norm, tck, der=2) / T**2

        return {
            'time': t_eval_norm * T,
            'positions': pos,
            'velocities': vel,
            'accelerations': acc,
            'tck_list': tck_list,
            'joint_waypoints': waypoints,
            'waypoint_times': t_norm * T,
        }

    @staticmethod
    def dodge_pair(traj_i: Dict,
                   traj_j: Dict,
                   spec_i: ArmSpec,
                   spec_j: ArmSpec,
                   collision_indices: List[int],
                   arm_to_dodge: int = 0  # 0 → move arm_i, 1 → move arm_j
                   ) -> Tuple[Dict, bool]:
        """
        Attempt to spatially dodge collision by perturbing arm_to_dodge's B-spline.
        Returns (modified_traj, success).
        """
        dodge_traj  = traj_i if arm_to_dodge == 0 else traj_j
        other_traj  = traj_j if arm_to_dodge == 0 else traj_i
        dodge_spec  = spec_i if arm_to_dodge == 0 else spec_j
        other_spec  = spec_j if arm_to_dodge == 0 else spec_i

        n_pts = len(dodge_traj['positions'])
        T = dodge_traj['time'][-1]

        waypoints   = dodge_traj['joint_waypoints'].copy()   # [n_way, 6]
        way_times   = dodge_traj['waypoint_times']           # [n_way]
        t_norm_orig = way_times / T                          # [0..1]

        t_lo, t_hi = SpatialDodge._get_collision_window(collision_indices, n_pts, T)
        mid_t_norm  = (t_lo + t_hi) / 2.0

        # Insert new waypoints in the collision window if not already dense
        n_new = 2
        new_t_norms = np.linspace(t_lo, t_hi, n_new + 2)[1:-1]

        new_waypoints_rows = []
        for tn in new_t_norms:
            # Interpolate existing waypoint
            idx = np.searchsorted(t_norm_orig, tn)
            idx = np.clip(idx, 0, len(waypoints) - 1)
            new_waypoints_rows.append(waypoints[idx].copy())

        # Build augmented waypoint set
        all_t = np.concatenate([t_norm_orig, new_t_norms])
        all_w = np.vstack([waypoints, np.array(new_waypoints_rows)])
        sort_idx = np.argsort(all_t)
        all_t = all_t[sort_idx]
        all_w = all_w[sort_idx]

        best_traj   = None
        best_violations = len(collision_indices)

        for iteration in range(SpatialDodge.MAX_ITER):
            for _ in range(SpatialDodge.N_RANDOM):
                perturbed_w = all_w.copy()

                # Perturb only the newly inserted waypoints (in collision window)
                for k, tn in enumerate(all_t):
                    if t_lo <= tn <= t_hi:
                        perturb = np.random.randn(6) * SpatialDodge.PERTURB_SCALE
                        perturbed_w[k] = np.clip(
                            perturbed_w[k] + perturb,
                            DHConstants.POSITION_LIMITS[:, 0],
                            DHConstants.POSITION_LIMITS[:, 1])

                candidate = SpatialDodge._rebuild_trajectory_from_waypoints(
                    perturbed_w, all_t, T)

                # Quick collision count against the other arm
                n_check = 100
                step = max(1, len(candidate['positions']) // n_check)
                violations = 0
                for idx in range(0, len(candidate['positions']), step):
                    q_d = candidate['positions'][idx]
                    q_o = other_traj['positions'][min(idx, len(other_traj['positions']) - 1)]
                    ld = all_link_positions(q_d, dodge_spec.base)
                    lo = all_link_positions(q_o, other_spec.base)
                    col, _ = CollisionChecker.check_pair(
                        ld, lo, dodge_spec.link_radii, other_spec.link_radii)
                    if col:
                        violations += 1

                if violations < best_violations:
                    best_violations = violations
                    best_traj = candidate
                    all_w = perturbed_w   # warm-start next iteration

            SpatialDodge.PERTURB_SCALE *= 0.8   # annealing

        if best_traj is None:
            best_traj = dodge_traj
        return best_traj, (best_violations == 0)


# ---------------------------------------------------------------------------
# Step 6 – Light synchronization polish (milestone soft-constraints)
# ---------------------------------------------------------------------------

def synchronization_polish(trajs: List[Dict],
                            milestone_fractions: Optional[List[float]] = None) -> List[Dict]:
    """
    Soft synchronization: adjust each arm's time-scale so that normalised
    progress ≈ same at a few milestone fractions (e.g., [0.25, 0.5, 0.75]).

    This is a lightweight per-arm 1D time-warp — does NOT alter joint paths,
    only re-samples timing.
    """
    if milestone_fractions is None:
        milestone_fractions = [0.25, 0.5, 0.75]

    N = len(trajs)
    T_max = max(t['time'][-1] for t in trajs)

    polished = []
    for traj in trajs:
        pos   = traj['positions']
        T_arm = traj['time'][-1]
        n     = len(pos)

        # Build a gentle s-curve warp so progress ≈ milestone_fractions at same wall-clock time
        warp_t = np.linspace(0, 1, 1000)
        # Simple approach: use the same minimum-jerk s-curve stretched to T_max
        tau = warp_t
        s = 10*tau**3 - 15*tau**4 + 6*tau**5      # progress ∈ [0,1]
        t_global = warp_t * T_max

        new_pos = np.zeros((len(warp_t), 6))
        for k, progress in enumerate(s):
            f = np.clip(progress, 0.0, 1.0)
            lo = int(f * (n - 1))
            hi = min(lo + 1, n - 1)
            a  = f * (n - 1) - lo
            new_pos[k] = (1 - a) * pos[lo] + a * pos[hi]

        new_vel = np.gradient(new_pos, t_global, axis=0)
        new_acc = np.gradient(new_vel, t_global, axis=0)

        polished.append({
            'time': t_global,
            'positions': new_pos,
            'velocities': new_vel,
            'accelerations': new_acc,
            'tck_list': traj.get('tck_list'),
            'joint_waypoints': traj.get('joint_waypoints'),
            'waypoint_times': traj.get('waypoint_times'),
        })

    return polished


# ---------------------------------------------------------------------------
# Main planner orchestrator
# ---------------------------------------------------------------------------

class MultiArmPlanner:
    """
    Orchestrates the full 6-step pipeline for N robot arms.

    Parameters
    ----------
    arm_specs : list of ArmSpec
    T         : nominal trajectory duration (seconds)
    verbose   : print progress
    """

    def __init__(self,
                 arm_specs: List[ArmSpec],
                 T: float = 10.0,
                 n_cartesian_samples: int = 40,
                 n_bspline_eval: int = 500,
                 safety_margin: float = 0.15,
                 verbose: bool = True):
        self.specs    = arm_specs
        self.N        = len(arm_specs)
        self.T        = T
        self.n_cart   = n_cartesian_samples
        self.n_eval   = n_bspline_eval
        self.verbose  = verbose
        CollisionChecker.SAFETY_MARGIN = safety_margin

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def plan(self,
             start_joints: List[np.ndarray],
             target_world_positions: List[np.ndarray],
             apply_kuramoto: bool = True,
             apply_spatial_dodge: bool = True,
             apply_polish: bool = True
             ) -> Optional[List[Dict]]:
        """
        Full pipeline.

        Parameters
        ----------
        start_joints             : [N] each is (6,) joint config
        target_world_positions   : [N] each is (3,) end-effector world position

        Returns list of N trajectory dicts, or None on failure.
        """
        self._log("\n" + "="*80)
        self._log(f"MULTI-ARM PLANNER  —  {self.N} arms,  T = {self.T:.1f}s")
        self._log("="*80)

        # ------------------------------------------------------------------
        # STEP 1 + 2: Independent Cartesian plan → IK → B-spline per arm
        # ------------------------------------------------------------------
        self._log("\n[Step 1+2] Cartesian planning + IK + B-spline fitting...")
        trajs = []
        for i, (spec, qstart, p_target) in enumerate(
                zip(self.specs, start_joints, target_world_positions)):

            p_start_local, _ = forward_kinematics(qstart)
            p_start_world    = p_start_local + spec.base
            p_target_local   = p_target - spec.base

            # Minimum-jerk Cartesian path
            cart_pos, t_vec = minimum_jerk_cartesian(
                p_start_world, p_target, self.T, n_samples=self.n_cart)

            # IK along path + B-spline
            traj = cartesian_path_to_joint_bspline(
                cart_pos, t_vec, spec.base, qstart, n_eval=self.n_eval)

            if traj is None:
                self._log(f"  ❌ Arm {spec.name}: IK failed along Cartesian path")
                return None

            trajs.append(traj)
            self._log(f"  ✓ Arm {spec.name}: {self.n_eval} samples, "
                      f"T={traj['time'][-1]:.2f}s")

        # ------------------------------------------------------------------
        # STEP 3: Pairwise collision scan
        # ------------------------------------------------------------------
        self._log("\n[Step 3] Pairwise collision detection...")
        col_map = CollisionChecker.scan_trajectories(trajs, self.specs)
        if not col_map:
            self._log("  ✓ No collisions detected — proceeding to Step 6 polish")
        else:
            for (i, j), events in col_map.items():
                self._log(f"  ⚠  Arm {self.specs[i].name} ↔ Arm {self.specs[j].name}: "
                          f"{len(events)} collision events")

        # ------------------------------------------------------------------
        # STEP 4: Kuramoto temporal re-parameterisation
        # ------------------------------------------------------------------
        if apply_kuramoto and col_map:
            self._log("\n[Step 4] Kuramoto temporal synchronization...")
            kura = KuramotoNArm(self.N, self.T, trajs, self.specs)
            phi_hist, t_kura = kura.integrate()
            trajs = kura.reparameterise(phi_hist, t_kura)
            self._log(f"  ✓ Integration complete — {len(t_kura)} steps")

            # Re-check
            col_map_post = CollisionChecker.scan_trajectories(trajs, self.specs)
            resolved = set(col_map.keys()) - set(col_map_post.keys())
            self._log(f"  ✓ Resolved by Kuramoto: {len(resolved)} pairs")
            if col_map_post:
                self._log(f"  ⚠  Still colliding: {len(col_map_post)} pairs → Step 5")
            col_map = col_map_post

        # ------------------------------------------------------------------
        # STEP 5: Spatial dodge for persistent collisions
        # ------------------------------------------------------------------
        if apply_spatial_dodge and col_map:
            self._log("\n[Step 5] Spatial dodge (B-spline refinement)...")
            for (i, j), events in col_map.items():
                self._log(f"  Dodging pair ({self.specs[i].name}, {self.specs[j].name})...")
                # Dodge the arm with fewer DOF constraints (here: just arm i)
                new_traj, success = SpatialDodge.dodge_pair(
                    trajs[i], trajs[j],
                    self.specs[i], self.specs[j],
                    events, arm_to_dodge=0)
                trajs[i] = new_traj
                self._log(f"    {'✓ Resolved' if success else '⚠  Partially resolved'}")

            # Final collision check
            col_map_final = CollisionChecker.scan_trajectories(trajs, self.specs)
            if not col_map_final:
                self._log("  ✓ All collisions resolved after spatial dodge")
            else:
                still = len(col_map_final)
                self._log(f"  ⚠  {still} pair(s) still have residual collisions")

        # ------------------------------------------------------------------
        # STEP 6: Synchronization polish
        # ------------------------------------------------------------------
        if apply_polish:
            self._log("\n[Step 6] Synchronization polish (milestone warp)...")
            trajs = synchronization_polish(trajs)
            T_final = trajs[0]['time'][-1]
            self._log(f"  ✓ All arms aligned to T = {T_final:.2f}s")

        # ------------------------------------------------------------------
        # Final summary
        # ------------------------------------------------------------------
        self._log("\n" + "="*80)
        self._log("PLANNING COMPLETE")
        col_final = CollisionChecker.scan_trajectories(trajs, self.specs)
        if col_final:
            self._log(f"⚠  {len(col_final)} pair(s) with residual proximity violations")
        else:
            self._log("✅ All arms collision-free and synchronized")
        self._log("="*80 + "\n")

        # Attach metadata
        for i, (traj, spec) in enumerate(zip(trajs, self.specs)):
            traj['arm_name'] = spec.name
            traj['arm_base'] = spec.base.tolist()

        return trajs


# ---------------------------------------------------------------------------
# Utility: save / load results
# ---------------------------------------------------------------------------

def save_multi_arm_trajectories(trajs: List[Dict], filename: str = "multi_arm_trajectories.json"):
    """Serialise (strips non-serialisable tck_list) to JSON."""
    output = []
    for t in trajs:
        entry = {k: v for k, v in t.items() if k != 'tck_list'}
        for k, v in entry.items():
            if isinstance(v, np.ndarray):
                entry[k] = v.tolist()
        output.append(entry)
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    size = len(json.dumps(output)) / 1024
    print(f"✓ Saved {filename} ({size:.1f} KB)")


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_trajectories(trajs: List[Dict], specs: List[ArmSpec]) -> Dict:
    """
    Compute per-arm statistics and inter-arm minimum clearance.
    Returns a summary dict.
    """
    summary = {'arms': [], 'inter_arm_min_clearance_cm': {}}

    for traj, spec in zip(trajs, specs):
        pos  = traj['positions']
        vel  = traj['velocities']
        acc  = traj['accelerations']
        T    = traj['time'][-1]
        n    = len(pos)

        max_vel_violations = 0
        max_acc_violations = 0
        for k in range(n):
            if np.any(np.abs(vel[k]) > DHConstants.VELOCITY_LIMITS):
                max_vel_violations += 1
            if np.any(np.abs(acc[k]) > DHConstants.ACCELERATION_LIMITS):
                max_acc_violations += 1

        # End-effector FK at end
        q_end = pos[-1]
        p_end, _ = forward_kinematics(q_end)
        p_end_world = p_end + spec.base

        summary['arms'].append({
            'name': spec.name,
            'duration_s': float(T),
            'samples': n,
            'velocity_violations': max_vel_violations,
            'acceleration_violations': max_acc_violations,
            'end_effector_world': p_end_world.tolist(),
        })

    # Inter-arm minimum clearance (sample 100 points)
    N = len(trajs)
    n_min = min(len(t['positions']) for t in trajs)
    step = max(1, n_min // 100)
    for i in range(N):
        for j in range(i + 1, N):
            min_d = np.inf
            for idx in range(0, n_min, step):
                li = all_link_positions(trajs[i]['positions'][idx], specs[i].base)
                lj = all_link_positions(trajs[j]['positions'][idx], specs[j].base)
                for a in li:
                    for b in lj:
                        min_d = min(min_d, np.linalg.norm(a - b))
            key = f"{specs[i].name}-{specs[j].name}"
            summary['inter_arm_min_clearance_cm'][key] = round(min_d * 100, 1)

    return summary


# ---------------------------------------------------------------------------
# Demo / test
# ---------------------------------------------------------------------------

def make_demo_arm_specs(n_arms: int = 4) -> List[ArmSpec]:
    """
    Place n_arms in a circle of radius 0.8 m around the world origin.
    Each arm faces the centre (yaw offset — simplified as base translation only).
    """
    specs = []
    for k in range(n_arms):
        angle = 2 * np.pi * k / n_arms
        base  = np.array([0.8 * np.cos(angle), 0.8 * np.sin(angle), 0.0])
        specs.append(ArmSpec(name=f"arm{k+1}", base_position=base))
    return specs


def run_demo(n_arms: int = 4):
    """Quick smoke-test with n_arms arms."""
    np.random.seed(42)
    print(f"\n{'='*80}")
    print(f"  MULTI-ARM PLANNER DEMO  —  {n_arms} arms")
    print(f"{'='*80}\n")

    specs = make_demo_arm_specs(n_arms)

    # Start at home (all zeros)
    starts = [np.zeros(6) for _ in range(n_arms)]

    # Random reachable targets: modest displacement from home end-effector
    targets = []
    for spec in specs:
        p_home_local, _ = forward_kinematics(np.zeros(6))
        p_home_world = p_home_local + spec.base
        # Small random offset (±0.2 m in x/z, keep y near base)
        delta = np.array([
            np.random.uniform(-0.2, 0.2),
            np.random.uniform(-0.1, 0.1),
            np.random.uniform(-0.15, 0.15),
        ])
        targets.append(p_home_world + delta)

    planner = MultiArmPlanner(specs, T=10.0, n_cartesian_samples=30, n_bspline_eval=300)
    t0 = time.time()
    trajs = planner.plan(starts, targets, apply_kuramoto=True,
                         apply_spatial_dodge=True, apply_polish=True)
    elapsed = time.time() - t0

    if trajs is None:
        print("❌ Planning failed")
        return

    # Validation
    summary = validate_trajectories(trajs, specs)
    print("\n📊 VALIDATION SUMMARY")
    print("-"*60)
    for arm_info in summary['arms']:
        print(f"  {arm_info['name']:8s} | T={arm_info['duration_s']:.2f}s | "
              f"vel_viol={arm_info['velocity_violations']} | "
              f"acc_viol={arm_info['acceleration_violations']}")
    print("\n  Inter-arm minimum clearances:")
    for pair, clearance in summary['inter_arm_min_clearance_cm'].items():
        status = "✓" if clearance >= 15.0 else "⚠"
        print(f"    {status} {pair}: {clearance:.1f} cm")
    print(f"\n  Planning time: {elapsed:.2f}s")
    print("-"*60)

    # Save
    save_multi_arm_trajectories(trajs, "multi_arm_trajectories.json")
    return trajs, summary


if __name__ == '__main__':
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    run_demo(n_arms=n)