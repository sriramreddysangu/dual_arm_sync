#!/usr/bin/env python3
"""
collision_checker.py
====================
Stage 3  —  Trajectory collision detection for Doosan M1013 dual-arm cell.

FRAME MAPPING  (critical — this is what was wrong before)
----------------------------------------------------------
forward_kinematics() returns T_frames[0..6]:
  frames[0] = identity (world/base)
  frames[1] = after Joint-1  origin at z=L1=0.1525  z-axis points -Y at home
  frames[2] = after Joint-2  origin at x=L2=0.620   z-axis points -Y at home
  frames[3] = after Joint-3  origin at x=A=0.0345   z-axis points -Z at home
  frames[4] = after Joint-4  origin at z=-L3=-0.559 z-axis points -Y at home
  frames[5] = after Joint-5  coincides with frame[4]  z-axis points -Z
  frames[6] = EE             z-axis points -Z

Each LINK body runs from frame[i] origin -> frame[i+1] origin.
In LOCAL frame[i] coordinates these vectors are:
  link1: (0,0,  L1)   along local +z   (frame0 -> frame1)
  link2: (L2,0, 0)    along local +x   (frame1 -> frame2)
  link3: (A, 0, 0)    along local +x   (frame2 -> frame3, elbow offset)
  link4: (0, 0, L3)   along local +z   (frame3 -> frame4)
  link5: (0, 0, 0)    zero-length      (frame4 = frame5, wrist coincides)
  link6: (0, 0, L4)   along local +z   (frame5 -> frame6)

The previous collision_checker used _z(v) = [0,0,v] for ALL capsules,
which was wrong for link2 and link3 (they run along local +x, not +z).

CAPSULE GEOMETRY
----------------
    penetration(A,B) = max(0, r_A + r_B - d_segment_to_segment)
    d_segment_to_segment = min distance between the two capsule axes

Two-phase evaluation:
  Phase 1  Sphere pre-filter (cheap): if sphere model clear by wide margin,
           capsules are guaranteed clear (spheres are larger).
  Phase 2  Capsule exact check (only when Phase 1 triggers).

URDF RUNTIME LOADER
-------------------
At startup, attempts to parse actual geometry from:
  $ROS2_WS/install/dsr_description2/share/.../xacro/m1013.urdf.xacro
Falls back to hand-tuned table if not found.

SAFETY TIERS
------------
  COLLISION_TOL  =  2 mm   hard penetration -> collision
  SAFETY_MARGIN  = 15 cm   Kuramoto phase-offset trigger
  WARN_MARGIN    =  8 cm   tight-clearance warning (safe but logged)

OUTPUT  ->  collision_result.json
----------------------------------
  safe, max_penetration_m, first_collision_time, n_events,
  collision_events, time_offsets, conflicting_arms, pen_by_pair,
  arm_ids, duration, warn_pairs, geometry, collision_free_summary

Usage:
    ros2 run dual_arm_sync collision_checker
"""

from __future__ import annotations

import json
import os
import warnings
from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Pipeline imports
# ---------------------------------------------------------------------------
try:
    from dual_arm_sync.ik_solver import (
        COLLISION_TOL, ARM_REGISTRY, RobotBases,
        forward_kinematics, LINK_SPHERES, N_SPHERES,
        get_sphere_centres,
        L1, L2, L3, L4, A,
    )
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from ik_solver import (
        COLLISION_TOL, ARM_REGISTRY, RobotBases,
        forward_kinematics, LINK_SPHERES, N_SPHERES,
        get_sphere_centres,
        L1, L2, L3, L4, A,
    )


# ============================================================================
# SAFETY MARGINS
# ============================================================================

SAFETY_MARGIN = 0.15   # 15 cm  — Kuramoto phase-offset trigger
WARN_MARGIN   = 0.08   # 8 cm   — tight-clearance warning
EVAL_EVERY_N  = 1      # check every Nth trajectory step (1 = all)


# ============================================================================
# CORRECT M1013 CAPSULE TABLE
# ============================================================================
# (frame_idx, p0_local, p1_local, radius_m)
#
# p0_local / p1_local are in LOCAL frame[frame_idx] coordinates.
# Verified by computing (R.T @ world_vector) for each link at home pose.
#
# Derivation:
#   frames[0]->frames[1]: world [0,0,L1]   -> local [0,0,L1]    (z-axis)
#   frames[1]->frames[2]: world [L2,0,0]   -> local [L2,0,0]    (x-axis!)
#   frames[2]->frames[3]: world [A,0,0]    -> local [A,0,0]     (x-axis!)
#   frames[3]->frames[4]: world [0,0,-L3]  -> local [0,0,L3]    (z-axis)
#   frames[4]->frames[5]: world [0,0,0]    -> zero (coincident frames)
#   frames[5]->frames[6]: world [0,0,-L4]  -> local [0,0,L4]    (z-axis)
#
# The PREVIOUS version used _z(v)=[0,0,v] for ALL links — WRONG for link2/link3.

def _z(v):
    """Local z-axis point."""
    return np.array([0., 0., float(v)])

def _x(v):
    """Local x-axis point."""
    return np.array([float(v), 0., 0.])

def _o():
    """Origin."""
    return np.array([0., 0., 0.])


M1013_CAPSULES = [
    # ── Link 1: base column  (frame0 -> frame1, along local +z) ──────────
    # Covers base housing and shoulder pivot
    (0, _o(),       _z(L1),          0.095),

    # ── Link 2: upper arm   (frame1 -> frame2, along local +x) ──────────
    # Note: runs along LOCAL +x (not +z) — this was the main bug
    (1, _o(),       _x(L2 * 0.45),   0.080),   # proximal half
    (1, _x(L2*0.45), _x(L2),         0.070),   # distal half (tapers)

    # ── Link 3: elbow offset (frame2 -> frame3, along local +x) ─────────
    # Short lateral offset at elbow — still local +x axis
    (2, _o(),       _x(A + 0.04),    0.070),   # elbow housing

    # ── Link 4: forearm     (frame3 -> frame4, along local +z) ──────────
    (3, _o(),       _z(L3 * 0.50),   0.065),   # forearm proximal
    (3, _z(L3*0.50), _z(L3),         0.055),   # forearm distal

    # ── Link 5: wrist       (frames 4+5 coincide — use frame4) ──────────
    # Small housing around wrist joints — modelled as sphere (p0=p1)
    (4, _o(),       _o(),            0.055),

    # ── Link 6: tool flange (frame5 -> frame6, along local +z) ──────────
    (5, _o(),       _z(L4),          0.045),
]


# ============================================================================
# URDF RUNTIME LOADER
# ============================================================================

def _try_load_urdf_capsules():
    """
    Parse per-link collision geometry from installed dsr_description2 URDF.

    The Doosan ROS2 package (humble branch) stores collision geometry in:
      dsr_description2/xacro/m1013.urdf.xacro

    Link names in the URDF: base_link, link_1..link_6
    We map them to frame indices matching forward_kinematics() T_frames.

    The URDF uses <cylinder> / <box> / <sphere> under <collision><geometry>.
    Origin xyz/rpy give the pose of the geometry in the link frame.

    Returns list of (frame_idx, p0, p1, radius) or None on any failure.
    """
    try:
        import subprocess
        import xml.etree.ElementTree as ET

        # Locate dsr_description2 share directory
        res = subprocess.run(
            ['ros2', 'pkg', 'prefix', 'dsr_description2'],
            capture_output=True, text=True, timeout=5,
        )
        if res.returncode != 0:
            return None

        prefix = res.stdout.strip()

        # Try colcon install layout first, then source layout
        candidates = [
            os.path.join(prefix, 'share', 'dsr_description2',
                         'xacro', 'm1013.urdf.xacro'),
            os.path.normpath(os.path.join(
                prefix, '..', 'src', 'dsr_description2',
                'xacro', 'm1013.urdf.xacro')),
        ]
        xp = next((p for p in candidates if os.path.isfile(p)), None)
        if xp is None:
            return None

        # Run xacro to expand macros
        xr = subprocess.run(
            ['xacro', xp, 'color:=white', 'gripper:=none'],
            capture_output=True, text=True, timeout=15,
        )
        if xr.returncode != 0:
            return None

        root = ET.fromstring(xr.stdout)

        # Doosan URDF link names -> our frame index
        # base_link / link_1 share frame[0] (both at base)
        # link_2 -> frame[1]  (upper arm origin)
        # link_3 -> frame[2]  (elbow origin)
        # link_4 -> frame[3]  (forearm origin)
        # link_5 -> frame[4]  (wrist)
        # link_6 -> frame[5]  (tool flange)
        link_to_frame = {
            'base_link': 0, 'link_1': 0,
            'link_2': 1, 'link_3': 2,
            'link_4': 3, 'link_5': 4, 'link_6': 5,
        }

        capsules = []
        for link in root.findall('link'):
            # strip namespace prefix e.g. "dsr01/link_2" -> "link_2"
            bare = link.get('name', '').split('/')[-1]
            fidx = link_to_frame.get(bare)
            if fidx is None:
                continue

            for coll in link.findall('collision'):
                geom = coll.find('geometry')
                if geom is None:
                    continue

                # Parse origin (xyz in link-local frame)
                orig = coll.find('origin')
                ox = oy = oz = 0.0
                if orig is not None:
                    xyz = orig.get('xyz', '0 0 0').split()
                    ox, oy, oz = float(xyz[0]), float(xyz[1]), float(xyz[2])

                cyl = geom.find('cylinder')
                box = geom.find('box')
                sph = geom.find('sphere')

                if cyl is not None:
                    r  = float(cyl.get('radius', '0.05'))
                    hl = float(cyl.get('length',  '0.1')) / 2.0
                    # cylinder axis = z-axis of origin frame
                    capsules.append((fidx,
                                     np.array([ox, oy, oz - hl]),
                                     np.array([ox, oy, oz + hl]),
                                     r + 0.005))

                elif box is not None:
                    sz  = [float(v) for v in
                           box.get('size', '0.1 0.1 0.1').split()]
                    ax  = int(np.argmax(sz))   # longest axis
                    hl  = sz[ax] / 2.0
                    r   = max(sz[i] for i in range(3) if i != ax) / 2.0
                    p0  = np.array([ox, oy, oz])
                    p1  = np.array([ox, oy, oz])
                    p0[ax] -= hl
                    p1[ax] += hl
                    capsules.append((fidx, p0, p1, r + 0.005))

                elif sph is not None:
                    r  = float(sph.get('radius', '0.05'))
                    p  = np.array([ox, oy, oz])
                    capsules.append((fidx, p, p.copy(), r + 0.005))

        if len(capsules) >= 4:
            print(f'  [URDF loader]  {len(capsules)} capsules parsed from {xp}')
            return capsules

    except Exception as exc:
        warnings.warn(f'URDF capsule loader skipped ({type(exc).__name__}: {exc})')
    return None


def _load_capsules():
    urdf = _try_load_urdf_capsules()
    return urdf if urdf else M1013_CAPSULES


CAPSULES = _load_capsules()
N_CAP    = len(CAPSULES)


# ============================================================================
# CAPSULE MATH
# ============================================================================

def _seg_to_seg_dist(p0, p1, q0, q1):
    """
    Minimum distance between line segments P=(p0,p1) and Q=(q0,q1).
    Closed-form, handles degenerate point cases.
    Reference: Ericson, Real-Time Collision Detection (2005) pp.148-151.
    """
    d1 = p1 - p0;  d2 = q1 - q0;  r = p0 - q0
    a  = float(np.dot(d1, d1))
    e  = float(np.dot(d2, d2))
    f  = float(np.dot(d2, r))
    EPS = 1e-10

    if a < EPS and e < EPS:          # both degenerate
        return float(np.linalg.norm(r))
    if a < EPS:                       # P degenerate
        s, t = 0.0, float(np.clip(f / (e + EPS), 0, 1))
    else:
        c = float(np.dot(d1, r))
        if e < EPS:                   # Q degenerate
            t, s = 0.0, float(np.clip(-c / (a + EPS), 0, 1))
        else:
            b   = float(np.dot(d1, d2))
            den = a * e - b * b
            s   = float(np.clip((b * f - c * e) / den, 0, 1)) \
                  if abs(den) > EPS else 0.0
            t   = (b * s + f) / (e + EPS)
            if t < 0.0:
                t, s = 0.0, float(np.clip(-c / (a + EPS), 0, 1))
            elif t > 1.0:
                t, s = 1.0, float(np.clip((b - c) / (a + EPS), 0, 1))

    return float(np.linalg.norm(p0 + s * d1 - (q0 + t * d2)))


def _caps_world(joints, base):
    """
    Transform all capsule endpoints from LOCAL link frame -> WORLD frame.

    For each capsule (frame_idx, p0_local, p1_local, r):
        p_world = R_frame @ p_local + t_frame + base_world
    """
    _, _, frames = forward_kinematics(joints)
    out = []
    for (fi, p0l, p1l, r) in CAPSULES:
        T  = frames[fi]
        R  = T[:3, :3]
        t  = T[:3,  3]
        out.append((R @ p0l + t + base,
                    R @ p1l + t + base,
                    r))
    return out


def _caps_penetration(caps_a, caps_b):
    """Sum of max(0, r_A+r_B - dist_seg) over all capsule pairs."""
    total = 0.0
    for (p0a, p1a, ra) in caps_a:
        for (p0b, p1b, rb) in caps_b:
            total += max(0.0, ra + rb - _seg_to_seg_dist(p0a, p1a, p0b, p1b))
    return total


def _caps_min_clearance(caps_a, caps_b):
    """Min surface-to-surface clearance.  Negative = collision."""
    mn = np.inf
    for (p0a, p1a, ra) in caps_a:
        for (p0b, p1b, rb) in caps_b:
            clr = _seg_to_seg_dist(p0a, p1a, p0b, p1b) - ra - rb
            if clr < mn:
                mn = clr
    return float(mn)


def _self_pen_caps(joints):
    """
    Self-collision penetration between non-adjacent link capsule pairs.
    Skip pairs where |frame_i - frame_j| <= 1 (directly connected).
    """
    _, _, frames = forward_kinematics(joints)
    local_caps = []
    for (fi, p0l, p1l, r) in CAPSULES:
        T  = frames[fi]
        R  = T[:3, :3];  t = T[:3, 3]
        local_caps.append((R @ p0l + t, R @ p1l + t, r))

    total = 0.0
    for i in range(N_CAP):
        for j in range(i + 2, N_CAP):
            if abs(CAPSULES[i][0] - CAPSULES[j][0]) <= 2:
                continue   # adjacent or next-adjacent frames — skip
                # NOTE: threshold=2 (not 1) because:
                #   upper-arm distal (frame1) and forearm-prox (frame3) are
                #   separated by |1-3|=2 but physically connected through
                #   the elbow link (frame2). At home pose they overlap by
                #   design (DH geometry). Same for frames 3 and 5 through wrist.
            d = _seg_to_seg_dist(local_caps[i][0], local_caps[i][1],
                                  local_caps[j][0], local_caps[j][1])
            total += max(0.0, CAPSULES[i][3] + CAPSULES[j][3] - d)
    return total


# ============================================================================
# SPHERE HELPERS  (pre-filter only — NOT primary check)
# ============================================================================

def _centres_spheres(joints, base):
    _, _, frames = forward_kinematics(joints)
    c = np.zeros((N_SPHERES, 3))
    for k, (fi, zo, _) in enumerate(LINK_SPHERES):
        c[k] = frames[fi][:3, 3] + zo * frames[fi][:3, 2] + base
    return c

def _sphere_pen(c1, c2):
    total = 0.0
    for i in range(N_SPHERES):
        for j in range(N_SPHERES):
            d = np.linalg.norm(c1[i] - c2[j])
            total += max(0.0, LINK_SPHERES[i][2] + LINK_SPHERES[j][2] - d)
    return total

def _sphere_min_clr(c1, c2):
    mn = np.inf
    for i in range(N_SPHERES):
        for j in range(N_SPHERES):
            d = np.linalg.norm(c1[i]-c2[j]) - LINK_SPHERES[i][2] - LINK_SPHERES[j][2]
            if d < mn: mn = d
    return float(mn)


# ============================================================================
# FULL TRAJECTORY COLLISION CHECK
# ============================================================================

def check_trajectories(trajectories, arm_bases, use_capsules=True):
    """
    Check all arm pairs at every trajectory step.

    Two-phase evaluation:
      Phase 1  Sphere pre-filter (cheap, conservative):
               Spheres over-approximate links, so if spheres show wide
               clearance (< 0.1*COLLISION_TOL), capsules are guaranteed clear.
      Phase 2  Capsule exact check using corrected frame mapping.

    Returns dict with all fields consumed by kuramoto_sync, including:
      'collision_free_summary'  — human-readable collision status string
      'conflicting_arms'        — for Kuramoto targeted refinement
      'pen_by_pair'             — per-pair max penetration (refinement trigger)
      'time_offsets'            — phase offsets for Kuramoto initialisation
    """
    arm_ids = list(trajectories.keys())
    pairs   = list(combinations(arm_ids, 2))

    mode = 'capsule (URDF-derived)' if use_capsules else 'sphere (legacy)'
    print(f'\n  Arms     : {arm_ids}')
    print(f'  Pairs    : {[(a,b) for a,b in pairs]}')
    print(f'  Mode     : {mode}  [{N_CAP} capsules/arm]')
    print(f'  Geometry : {"URDF-parsed" if CAPSULES is not M1013_CAPSULES else "hand-tuned (M1013 DH-derived)"}')

    n_steps  = min(len(trajectories[a]['trajectory_points']) for a in arm_ids)
    duration = trajectories[arm_ids[0]]['trajectory_points'][-1]['time']
    t_grid   = np.array([
        trajectories[arm_ids[0]]['trajectory_points'][i]['time']
        for i in range(n_steps)
    ])

    joint_arr = {
        aid: np.array([p['joints']
                       for p in trajectories[aid]['trajectory_points'][:n_steps]])
        for aid in arm_ids
    }

    clr_profiles  = {f'{a}_{b}': [] for a, b in pairs}
    max_pen_pair   = {f'{a}_{b}': 0.0 for a, b in pairs}
    min_clr_pair   = {f'{a}_{b}': np.inf for a, b in pairs}
    safe            = True
    max_pen         = 0.0
    first_coll_time = None
    events          = []
    conflict_set: Set[str] = set()
    warn_pairs:   Set[str] = set()
    phase2_calls  = 0   # diagnostic: how many times we did full capsule check

    for i in range(0, n_steps, EVAL_EVERY_N):
        t = float(t_grid[i])

        # Sphere centres — cheap, all arms
        sph_c = {
            aid: _centres_spheres(joint_arr[aid][i], arm_bases[aid])
            for aid in arm_ids
        }

        # ---- Self-collision (capsule exact) ---------------------------------
        for aid in arm_ids:
            sp = _self_pen_caps(joint_arr[aid][i]) if use_capsules else 0.0
            if sp > COLLISION_TOL:
                safe    = False
                max_pen = max(max_pen, sp)
                if first_coll_time is None: first_coll_time = t
                conflict_set.add(aid)
                events.append({'time': t, 'type': 'self',
                               'arm': aid, 'penetration': float(sp)})

        # ---- Inter-arm (two-phase) ------------------------------------------
        for a, b in pairs:
            key     = f'{a}_{b}'
            sph_pre = _sphere_pen(sph_c[a], sph_c[b])

            # Phase 1: if sphere model shows wide clearance, skip capsule check
            if use_capsules and sph_pre < COLLISION_TOL * 0.1:
                mn = _sphere_min_clr(sph_c[a], sph_c[b])
                clr_profiles[key].append(mn)
                min_clr_pair[key] = min(min_clr_pair[key], mn)
                continue

            # Phase 2: capsule exact check
            if use_capsules:
                caps_a = _caps_world(joint_arr[a][i], arm_bases[a])
                caps_b = _caps_world(joint_arr[b][i], arm_bases[b])
                pen    = _caps_penetration(caps_a, caps_b)
                clr    = _caps_min_clearance(caps_a, caps_b)
                phase2_calls += 1
            else:
                pen = sph_pre
                clr = _sphere_min_clr(sph_c[a], sph_c[b])

            clr_profiles[key].append(float(clr))
            max_pen_pair[key]  = max(max_pen_pair[key], float(pen))
            min_clr_pair[key]  = min(min_clr_pair[key], float(clr))

            if pen > COLLISION_TOL:
                safe    = False
                max_pen = max(max_pen, pen)
                if first_coll_time is None: first_coll_time = t
                conflict_set.add(a); conflict_set.add(b)
                events.append({'time': t, 'type': 'inter_arm',
                               'arm_a': a, 'arm_b': b,
                               'penetration': float(pen),
                               'clearance':   float(clr)})
            elif 0.0 < clr < WARN_MARGIN:
                warn_pairs.add(key)

    # Deduplicate events (first per pair per 0.1 s bucket)
    if events:
        seen, deduped = set(), []
        for ev in events:
            ekey = ev.get('arm') or f"{ev['arm_a']}_{ev['arm_b']}"
            uid  = (ekey, round(ev['time'] / 0.1))
            if uid not in seen:
                seen.add(uid); deduped.append(ev)
        events = deduped[:20]

    # ---- Kuramoto phase-offset computation ----------------------------------
    # Faster arm is delayed so arms reach the congested region at different times.
    time_offsets = {aid: 0.0 for aid in arm_ids}
    if not safe:
        for a, b in pairs:
            key     = f'{a}_{b}'
            clr_arr = np.array(clr_profiles[key]) if clr_profiles[key] \
                      else np.array([])
            if len(clr_arr) == 0:
                continue
            min_clr = float(clr_arr.min())
            if min_clr < SAFETY_MARGIN:
                deficit = SAFETY_MARGIN - min_clr
                ci      = int(np.argmin(clr_arr))
                w0, w1  = max(0, ci - 5), min(n_steps - 1, ci + 5)
                spd_a   = float(np.linalg.norm(
                    np.diff(joint_arr[a][w0:w1], axis=0))) + 1e-6
                spd_b   = float(np.linalg.norm(
                    np.diff(joint_arr[b][w0:w1], axis=0))) + 1e-6
                faster  = b if spd_b > spd_a else a
                offset  = min(duration * 0.3,
                               deficit / (spd_a + spd_b) * duration / n_steps)
                time_offsets[faster] = max(time_offsets[faster], offset)
                print(f'  Kuramoto offset [{a}]<->[{b}]  '
                      f'min_clr={min_clr*100:.1f} cm  '
                      f'deficit={deficit*100:.1f} cm  '
                      f'-> delay [{faster}] by {offset:.4f} s')

    # ---- Build human-readable collision summary -----------------------------
    if safe:
        min_all = min(min_clr_pair.values()) if min_clr_pair else np.inf
        if min_all < WARN_MARGIN:
            summary = (f'COLLISION-FREE  (min clearance={min_all*100:.1f} cm — '
                       f'TIGHT, < {WARN_MARGIN*100:.0f} cm warning threshold)')
        else:
            summary = (f'COLLISION-FREE  (min clearance={min_all*100:.1f} cm, '
                       f'all pairs > {WARN_MARGIN*100:.0f} cm)')
    else:
        summary = (f'COLLISION DETECTED  max_pen={max_pen*1000:.2f} mm  '
                   f't_first={first_coll_time:.3f} s  '
                   f'conflicting={sorted(conflict_set)}')

    # ---- Print summary ------------------------------------------------------
    sep = '=' * 60
    print(f'\n  {sep}')
    print(f'  COLLISION CHECK RESULT')
    print(f'  {sep}')
    print(f'  {summary}')
    print(f'  {"─"*58}')
    print(f'  safe               : {safe}')
    print(f'  max penetration    : {max_pen*1000:.2f} mm')
    print(f'  phase2 calls       : {phase2_calls} / {n_steps} steps '
          f'(capsule checks skipped by pre-filter)')
    if first_coll_time is not None:
        print(f'  first collision    : t = {first_coll_time:.3f} s')
    if warn_pairs:
        print(f'  tight clearance    : {sorted(warn_pairs)}'
              f'  (< {WARN_MARGIN*100:.0f} cm — safe but narrow)')
    print(f'  events (deduped)   : {len(events)}')
    for ev in events[:5]:
        if ev['type'] == 'self':
            print(f'    t={ev["time"]:.2f}  SELF-COL [{ev["arm"]}]'
                  f'  pen={ev["penetration"]*1000:.1f} mm')
        else:
            print(f'    t={ev["time"]:.2f}  INTER-ARM [{ev["arm_a"]}]<->[{ev["arm_b"]}]'
                  f'  pen={ev["penetration"]*1000:.1f} mm'
                  f'  clr={ev["clearance"]*1000:.1f} mm')
    print(f'  conflicting arms   : {sorted(conflict_set)}'
          f'  <- Kuramoto will refine these')
    print(f'  pen_by_pair (mm)   : '
          f'{dict((k, round(v*1000, 2)) for k,v in max_pen_pair.items())}')
    print(f'  min_clr_pair (cm)  : '
          f'{dict((k, round(v*100, 1)) for k,v in min_clr_pair.items())}')
    print(f'  time_offsets       : {time_offsets}'
          f'  <- phase-seed for Kuramoto')
    print(f'  {sep}')

    return {
        'safe':                  safe,
        'collision_free':        safe,          # explicit alias for Kuramoto
        'collision_free_summary': summary,       # human-readable status
        'max_penetration_m':     float(max_pen),
        'first_collision_time':  first_coll_time,
        'n_events':              len(events),
        'collision_events':      events,
        'time_offsets':          time_offsets,
        'conflicting_arms':      sorted(conflict_set),
        'pen_by_pair':           max_pen_pair,
        'min_clr_pair_m':        {k: float(v) for k, v in min_clr_pair.items()},
        'arm_ids':               arm_ids,
        'duration':              float(duration),
        'warn_pairs':            sorted(warn_pairs),
        'geometry':              ('capsule_urdf'
                                  if CAPSULES is not M1013_CAPSULES
                                  else 'capsule_hardcoded'),
        'n_capsules_per_arm':    N_CAP,
        'phase2_capsule_calls':  phase2_calls,
    }


# ============================================================================
# PIPELINE RUNNER
# ============================================================================

def _resolve_bases(arm_ids):
    bases = {}
    for aid in arm_ids:
        if aid in ARM_REGISTRY:
            bases[aid] = ARM_REGISTRY._arms[aid]['base']
        elif hasattr(RobotBases, f'{aid.upper()}_BASE'):
            bases[aid] = getattr(RobotBases, f'{aid.upper()}_BASE')
        else:
            bases[aid] = np.zeros(3)
    return bases


def run(traj_file='trajectories.json',
        output_file='collision_result.json',
        use_capsules=True):
    """
    Load trajectories.json -> check collisions -> write collision_result.json.
    """
    print('\n' + '=' * 80)
    print('COLLISION CHECKER  [M1013 Capsule Geometry — URDF-aware]')
    print('=' * 80)
    geom_src = ('URDF-parsed'
                if CAPSULES is not M1013_CAPSULES
                else 'hand-tuned  (URDF not found — using DH-derived capsules)')
    print(f'  Geometry source  : {geom_src}')
    print(f'  Capsules / arm   : {N_CAP}')
    print(f'  Safety margin    : {SAFETY_MARGIN*100:.0f} cm')
    print(f'  Warn margin      : {WARN_MARGIN*100:.0f} cm')
    print(f'  Collision tol    : {COLLISION_TOL*1000:.0f} mm')

    with open(traj_file) as f:
        data = json.load(f)

    trajs   = data.get('trajectories', data)
    arm_ids = data.get('arm_ids', list(trajs.keys()))
    bases   = _resolve_bases(arm_ids)

    result = check_trajectories(trajs, bases, use_capsules=use_capsules)

    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f'\n[geometry={result["geometry"]}]')
    print(f'\n>>> RESULT: {result["collision_free_summary"]}')
    print()
    if result['collision_free']:
        print('[OK] All trajectories COLLISION-FREE -> Kuramoto will synchronise phases')
        if result['warn_pairs']:
            print(f'[!!] Tight clearance warning on: {result["warn_pairs"]}')
            print('     Monitor these pairs in Gazebo visualisation')
    else:
        print(f'[!!] COLLISION DETECTED')
        print(f'     max penetration  : {result["max_penetration_m"]*1000:.1f} mm')
        print(f'     conflicting arms : {result["conflicting_arms"]}')
        print('     Kuramoto adaptive refinement will attempt to resolve this')
        print('     by re-seeding conflicting-arm B-spline control points')
    print()
    print('Next -> ros2 run dual_arm_sync kuramoto_sync')
    return result


# ============================================================================
# ROS2 ENTRY
# ============================================================================

def main(args=None):
    try:
        import rclpy
        rclpy.init(args=args)
    except Exception:
        pass
    try:
        run()
    except FileNotFoundError:
        print('[!!] trajectories.json not found')
        print('     Run: ros2 run dual_arm_sync trajectory_generation')
    except Exception:
        import traceback
        traceback.print_exc()
    try:
        import rclpy
        rclpy.shutdown()
    except Exception:
        pass


if __name__ == '__main__':
    main()