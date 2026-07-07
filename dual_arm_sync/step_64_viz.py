#!/usr/bin/env python3
"""
step_64_viz.py  --  Per-Iteration EE Path Visualization (Reference Style)
==========================================================================
INPUT  : s64_resolved.json (must contain 'iteration_snapshots')
OUTPUT : two PNGs per iteration -- one per arm
         plus a combined comparison PNG per iteration
         plus a montage PNG showing all iterations

Each per-arm figure has TWO panels matching your reference image:
  LEFT  : 3D EE path with viridis-style time-gradient color, WP markers
          as solid red dots, base as triangle, with colorbar
  RIGHT : Top view (XY plane) with same WP markers labeled WP0, WP1, ...

Output naming:
  s64_viz_iter_00_original_dsr01.png
  s64_viz_iter_00_original_dsr02.png
  s64_viz_iter_00_original_combined.png
  s64_viz_iter_01_single_seg_a040_dsr01.png
  ...
  s64_viz_montage.png
"""

import json, os, sys
from typing import Dict, List
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    from matplotlib import cm
    from matplotlib.colors import Normalize
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.size': 10,
        'axes.titlesize': 12, 'axes.labelsize': 10,
        'legend.fontsize': 9, 'figure.dpi': 130,
    })
    _PLOT = True
except ImportError:
    _PLOT = False

sys.path.insert(0, os.path.dirname(__file__))
from _robot import (NDOF, ROBOT_BASES, ARM_NAMES, fk_world, pair_min_dist,
                    LINK_RADII, SAFETY_MARGIN)

# ── Arm-specific colormaps and colors (matches reference image style) ─────────
ARM_CMAPS  = {'dsr01': 'plasma', 'dsr02': 'plasma'}
ARM_PATH_C = {'dsr01': '#1f77b4', 'dsr02': '#d62728'}
WP_COLOR   = '#E53935'    # solid red for waypoint markers
COLL_C     = '#C62828'    # collision highlight


def ee_path(positions: np.ndarray, base: np.ndarray) -> np.ndarray:
    """FK all samples to world-frame EE positions. Shape (N, 3)."""
    return np.array([fk_world(positions[k], base) for k in range(len(positions))])


def waypoint_positions(snap: Dict, name: str, base: np.ndarray, n_seg: int):
    """
    Compute WP (control point boundary) EE positions in world frame.
    For 5 segments, we have 6 waypoints WP0..WP5 (segment boundaries).
    Returns array of (n_seg+1, 3) world-frame positions.
    """
    pos    = np.array(snap['positions'][name])
    n_smp  = len(pos)
    ee_all = ee_path(pos, base)
    wp_ee  = []
    for wp in range(n_seg + 1):
        arc_pos = wp / n_seg
        idx     = int(np.clip(round(arc_pos * (n_smp - 1)), 0, n_smp - 1))
        wp_ee.append(ee_all[idx])
    return np.array(wp_ee)


# ── PANEL 1: 3D path with time-gradient color (reference image style) ───────
def draw_3d_panel(ax, positions, base, arm_name, n_seg, coll_segs_set,
                   show_colorbar=False, fig=None):
    ee     = ee_path(positions, base)
    n_pts  = len(ee)
    arc    = np.linspace(0., 1., n_pts)
    cmap   = cm.get_cmap(ARM_CMAPS.get(arm_name, 'plasma'))

    # Plot the path as a continuous gradient line
    for k in range(n_pts - 1):
        ax.plot(ee[k:k+2, 0], ee[k:k+2, 1], ee[k:k+2, 2],
                color=cmap(arc[k]), lw=2.5, solid_capstyle='round',
                alpha=0.95)

    # Overlay collision segments in red
    for seg in coll_segs_set:
        s0, s1 = seg / n_seg, (seg + 1) / n_seg
        mask   = (arc >= s0 - 1e-9) & (arc <= s1 + 1e-9)
        if mask.sum() >= 2:
            ax.plot(ee[mask, 0], ee[mask, 1], ee[mask, 2],
                    color=COLL_C, lw=4.0, alpha=0.7, zorder=10)

    # Waypoint markers (solid red dots like reference image)
    wp_ee = waypoint_positions({'positions': {arm_name: positions}}, arm_name, base, n_seg)
    ax.scatter(wp_ee[:, 0], wp_ee[:, 1], wp_ee[:, 2],
               s=110, c=WP_COLOR, edgecolors='darkred', linewidths=1.2,
               zorder=20, label='WP end-effectors')

    # Base triangle (on floor)
    ax.scatter(base[0], base[1], 0, s=200, c='black', marker='^',
               edgecolors='black', zorder=22, label='Base')

    # Axis limits, view, labels
    all_x = np.concatenate([ee[:, 0], [base[0]]])
    all_y = np.concatenate([ee[:, 1], [base[1]]])
    all_z = np.concatenate([ee[:, 2], [0.0]])
    pad = 0.15
    ax.set_xlim(all_x.min() - pad, all_x.max() + pad)
    ax.set_ylim(all_y.min() - pad, all_y.max() + pad)
    ax.set_zlim(0., all_z.max() + pad)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.set_title('3D path')
    ax.view_init(elev=22, azim=-62)
    ax.grid(True, alpha=0.25)
    ax.legend(loc='upper left', framealpha=0.85, fontsize=8)

    # Colorbar for time gradient (only on combined or single-arm panel)
    if show_colorbar and fig is not None:
        sm = cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar_ax = fig.add_axes([0.46, 0.20, 0.012, 0.60])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('norm. time')


# ── PANEL 2: Top view (XY) with labeled WP markers (reference image style) ──
def draw_topview_panel(ax, positions, base, arm_name, n_seg, coll_segs_set):
    ee  = ee_path(positions, base)
    n   = len(ee)
    arc = np.linspace(0., 1., n)

    # Path line
    ax.plot(ee[:, 0], ee[:, 1], color=ARM_PATH_C.get(arm_name, '#1f77b4'),
            lw=2.5, alpha=0.9)

    # Highlight collision segments
    for seg in coll_segs_set:
        s0, s1 = seg / n_seg, (seg + 1) / n_seg
        mask = (arc >= s0 - 1e-9) & (arc <= s1 + 1e-9)
        if mask.sum() >= 2:
            ax.plot(ee[mask, 0], ee[mask, 1], color=COLL_C, lw=4.0, alpha=0.6)

    # Waypoint markers with labels WP0, WP1, ..., WP5
    wp_ee = waypoint_positions({'positions': {arm_name: positions}}, arm_name, base, n_seg)
    for wp_idx, (x, y, _) in enumerate(wp_ee):
        ax.scatter(x, y, s=110, c=WP_COLOR, edgecolors='darkred',
                    linewidths=1.2, zorder=15)
        ax.annotate(f'WP{wp_idx}', xy=(x, y),
                    xytext=(8, 4), textcoords='offset points',
                    fontsize=10, color='darkred')

    # Base triangle
    ax.scatter(base[0], base[1], s=200, c='black', marker='^',
               edgecolors='black', zorder=20)

    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_title('Top view (XY plane)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)


# ── Single-arm figure (two panels: 3D + top view) ───────────────────────────
def render_per_arm(snap, arm_name, base, n_seg, banner, out_fname):
    fig = plt.figure(figsize=(15, 6.5))
    gs  = fig.add_gridspec(1, 2, width_ratios=[1, 1.05], wspace=0.18)
    ax3d = fig.add_subplot(gs[0], projection='3d')
    axxy = fig.add_subplot(gs[1])

    positions     = np.array(snap['positions'][arm_name])
    coll_segs_set = set()
    for pair_key, seg in snap.get('first_coll_segs', {}).items():
        if arm_name in pair_key: coll_segs_set.add(seg)

    draw_3d_panel(ax3d, positions, base, arm_name, n_seg,
                   coll_segs_set, show_colorbar=True, fig=fig)
    draw_topview_panel(axxy, positions, base, arm_name, n_seg, coll_segs_set)

    fig.suptitle(f'[{arm_name}]  End-Effector Path  (world frame)\n{banner}',
                 fontsize=12.5, fontweight='bold', y=0.97)
    plt.savefig(out_fname, bbox_inches='tight')
    plt.close()


# ── Combined figure (both arms in same plot for direct comparison) ──────────
def render_combined(snap, arm_names, bases, n_seg, banner, out_fname):
    fig = plt.figure(figsize=(15, 6.5))
    gs  = fig.add_gridspec(1, 2, width_ratios=[1, 1.05], wspace=0.18)
    ax3d = fig.add_subplot(gs[0], projection='3d')
    axxy = fig.add_subplot(gs[1])

    coll_segs_set = set(snap.get('first_coll_segs', {}).values())

    for arm_name in arm_names:
        positions = np.array(snap['positions'][arm_name])
        draw_3d_panel(ax3d, positions, bases[arm_name], arm_name,
                       n_seg, coll_segs_set, show_colorbar=False)
        draw_topview_panel(axxy, positions, bases[arm_name], arm_name,
                            n_seg, coll_segs_set)

    fig.suptitle(f'Both Arms Combined\n{banner}',
                 fontsize=12.5, fontweight='bold', y=0.97)
    plt.savefig(out_fname, bbox_inches='tight')
    plt.close()


# ── Min-distance figure (one per iteration) ─────────────────────────────────
def render_min_dist(snap, arm_names, bases, n_seg, banner, out_fname):
    if len(arm_names) < 2: return
    ni, nj = arm_names[0], arm_names[1]
    p_i = np.array(snap['positions'][ni])
    p_j = np.array(snap['positions'][nj])
    K   = min(len(p_i), len(p_j))
    arc = np.linspace(0., 1., K)
    d   = np.array([pair_min_dist(p_i[k], bases[ni], p_j[k], bases[nj])
                     for k in range(K)])
    d_cm = d * 100
    thresh_cm = (LINK_RADII.min() * 2 + SAFETY_MARGIN) * 100

    coll_segs_set = set(snap.get('first_coll_segs', {}).values())

    fig, ax = plt.subplots(figsize=(12, 4))
    for seg in range(n_seg):
        s0, s1 = seg / n_seg, (seg + 1) / n_seg
        col = '#FFCDD2' if seg in coll_segs_set else '#C8E6C9'
        ax.axvspan(s0, s1, alpha=0.4, color=col)
        ax.text((s0 + s1) / 2, thresh_cm * 1.4, f'S{seg}',
                ha='center', fontsize=8, color='black')

    ax.plot(arc, d_cm, color='#1565C0', lw=2.5, label='min inter-arm dist')
    ax.axhline(y=thresh_cm, color='red', ls='--', lw=1.5,
                label=f'Collision threshold ({thresh_cm:.0f}cm)')
    coll_mask = d_cm < thresh_cm
    if coll_mask.any():
        ax.fill_between(arc, 0, d_cm, where=coll_mask, color='red', alpha=0.5,
                         label=f'In collision ({int(coll_mask.sum())} steps)')

    ax.set_xlim(0, 1); ax.set_ylim(0, max(d_cm.max() * 1.1, 50))
    ax.set_xlabel('Arc fraction'); ax.set_ylabel('Min inter-arm distance (cm)')
    ax.set_title(f'Inter-arm distance: {ni} <-> {nj}\n{banner}', fontsize=11)
    ax.legend(loc='upper right'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_fname, bbox_inches='tight')
    plt.close()


# ── Montage of all iterations ───────────────────────────────────────────────
def render_montage(snapshots, arm_names, bases, n_seg, out_fname):
    n     = len(snapshots)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig   = plt.figure(figsize=(6 * ncols, 5 * nrows))

    for idx, snap in enumerate(snapshots):
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection='3d')
        coll_segs_set = set(snap.get('first_coll_segs', {}).values())
        for arm_name in arm_names:
            positions = np.array(snap['positions'][arm_name])
            draw_3d_panel(ax, positions, bases[arm_name], arm_name, n_seg,
                           coll_segs_set, show_colorbar=False)

        # Compact title
        it_n  = snap.get('iteration', -1)
        nc    = snap.get('collisions_found')
        phase = snap.get('phase', '-')
        alpha = snap.get('alpha')
        if snap['label'] == 'original':
            t = 'BEFORE: original'
        elif 'resolved' in snap['label']:
            t = f'iter {it_n}: RESOLVED'
        elif 'unresolved' in snap['label']:
            t = f'iter {it_n}: UNRESOLVED'
        else:
            t = (f'iter {it_n}: {phase} alpha={alpha:.2f}  coll={nc}'
                 if alpha is not None else f'iter {it_n}: coll={nc}')
        ax.set_title(t, fontsize=10)

    fig.suptitle(f'SKAR-N Iterations -- {arm_names[0]} <-> {arm_names[1]}',
                  fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout()
    plt.savefig(out_fname, bbox_inches='tight')
    plt.close()


def banner_text(snap):
    it    = snap.get('iteration', -1)
    nc    = snap.get('collisions_found')
    phase = snap.get('phase', '-')
    alpha = snap.get('alpha')
    label = snap.get('label', '')
    if label == 'original':
        return 'BEFORE -- Original B-spline (no modification)'
    if 'resolved' in label:
        return f'AFTER iter {it} -- COLLISION-FREE'
    if 'unresolved' in label:
        return f'AFTER iter {it} -- UNRESOLVED (max iter reached)'
    alpha_str = f', alpha={alpha:.2f}' if alpha is not None else ''
    return f'AFTER iter {it} -- phase={phase}{alpha_str}, collisions={nc}'


def main():
    print('\n' + '=' * 66)
    print('  STEP 64 VIZ  --  Per-Iteration EE Path Visualization')
    print('=' * 66)
    if not _PLOT:
        print('  matplotlib not installed'); sys.exit(1)
    if not os.path.exists('s64_resolved.json'):
        print('  s64_resolved.json not found -- run step_64 first'); sys.exit(1)

    with open('s64_resolved.json') as fh: data = json.load(fh)
    snapshots = data.get('iteration_snapshots', [])
    if not snapshots:
        print('  No iteration_snapshots present. Re-run step_64.')
        sys.exit(1)

    arm_names = data.get('arm_names', ARM_NAMES)
    bases     = {n: np.array(ROBOT_BASES.get(n, [0, 0, 0])) for n in arm_names}
    n_seg     = int(data[arm_names[0]]['spline']['n_seg'])

    print(f'\n  Arms     : {arm_names}')
    print(f'  Segments : {n_seg} (waypoints WP0..WP{n_seg})')
    print(f'  Snapshots: {len(snapshots)}\n')

    n_files = 0
    for idx, snap in enumerate(snapshots):
        banner    = banner_text(snap)
        label_san = snap['label'].replace('/', '_')

        # Per-arm figure
        for arm_name in arm_names:
            fname = f's64_viz_iter_{idx:02d}_{label_san}_{arm_name}.png'
            render_per_arm(snap, arm_name, bases[arm_name], n_seg, banner, fname)
            n_files += 1; print(f'  {fname}')

        # Combined (both arms)
        fname = f's64_viz_iter_{idx:02d}_{label_san}_combined.png'
        render_combined(snap, arm_names, bases, n_seg, banner, fname)
        n_files += 1; print(f'  {fname}')

        # Min-distance plot
        fname = f's64_viz_iter_{idx:02d}_{label_san}_mindist.png'
        render_min_dist(snap, arm_names, bases, n_seg, banner, fname)
        n_files += 1; print(f'  {fname}')

    # Montage
    montage = 's64_viz_montage.png'
    render_montage(snapshots, arm_names, bases, n_seg, montage)
    n_files += 1
    print(f'  {montage}')

    print(f'\n  {n_files} files saved.')
    print(f'  Per arm files: s64_viz_iter_XX_*_dsr01.png  /  *_dsr02.png')
    print(f'  Combined     : s64_viz_iter_XX_*_combined.png')
    print(f'  Min distance : s64_viz_iter_XX_*_mindist.png')
    print(f'  Overview     : s64_viz_montage.png\n')


if __name__ == '__main__': main()