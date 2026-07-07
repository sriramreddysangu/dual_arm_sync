#!/usr/bin/env python3
"""
step_34_diag.py  --  Phase-lag diagnostic  [DUAL ARM]
=====================================================
Loads s32_trajectories.json and, for EVERY constant phase lag (both arms leading),
re-times the two arms (each on its UNCHANGED path, s:0->1) and counts collisions
using deepest_link_pair -- the SAME test as step_33 and the step_35 pre-flight.
Prints a table so you can see exactly what each lag does, including the one you
expect to work. Run after step_32:
    python3 step_34_diag.py
    python3 step_34_diag.py --lag 2.0      # detail one specific lag (seconds)
"""
import json, os, sys, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from _robot2x import ROBOT_BASES, deepest_link_pair, pair_min_dist

KUR_DT = 0.01
CLEAR_M = float(os.environ.get('DUAL_ARM_CLEARANCE_M', '0.05'))


def interp(pos, f):
    f = float(np.clip(f, 0, 1)); n = len(pos) - 1
    if n <= 0: return pos[0]
    i = min(int(f * n), n - 1); a = f * n - i
    return pos[i] + a * (pos[i + 1] - pos[i])


def build(posA, posB, lead, lag_s, dur):
    delta = lag_s / dur; span = 1.0 + delta
    n = max(2, int(round(span * dur / KUR_DT))); t = np.linspace(0, span * dur, n)
    offA = 0.0 if lead == 'A' else delta; offB = 0.0 if lead == 'B' else delta
    phA = np.clip(t / dur - offA, 0, 1); phB = np.clip(t / dur - offB, 0, 1)
    return (np.array([interp(posA, f) for f in phA]),
            np.array([interp(posB, f) for f in phB]), t)


def scan(qA, qB, ba, bb):
    K = min(len(qA), len(qB)); coll = 0; collc = 0; mn = float('inf'); fa = la = None
    for k in range(K):
        d = pair_min_dist(qA[k], ba, qB[k], bb); mn = min(mn, d)
        if d < CLEAR_M: collc += 1                      # real-clearance gate (step_34/35)
        if deepest_link_pair(qA[k], ba, qB[k], bb) is not None:
            coll += 1; la = k                           # oversized built-in margin
            if fa is None: fa = k
    arc = (f'{fa/K:.2f}-{la/K:.2f}' if fa is not None else '   --   ')
    return coll, collc, mn, arc


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--lag', type=float, default=None)
    a, _ = ap.parse_known_args()
    if not os.path.exists('s32_trajectories.json'):
        print('  s32_trajectories.json not found -- run step_32 first'); sys.exit(1)
    d = json.load(open('s32_trajectories.json'))
    names = d.get('arm_names', ['dsr01', 'dsr02']); dur = float(d['duration'])
    A, B = names; ba = np.array(ROBOT_BASES[A]); bb = np.array(ROBOT_BASES[B])
    posA = np.array(d[A]['trajectory']['positions']); posB = np.array(d[B]['trajectory']['positions'])

    print('\n' + '=' * 70)
    print(f'  PHASE-LAG DIAGNOSTIC  {A}<->{B}  dur={dur:.1f}s')
    print(f'  coll(margin) = oversized built-in deepest_link_pair')
    print(f'  coll@{CLEAR_M*100:.0f}cm = REAL clearance gate used by step_34 / step_35')
    print('=' * 70)
    c0, cc0, m0, arc0 = scan(posA, posB, ba, bb)
    print(f'  lockstep (no lag):  coll(margin)={c0:4d}  coll@{CLEAR_M*100:.0f}cm={cc0:4d}  min={m0*100:5.1f}cm')

    lags = [a.lag] if a.lag is not None else [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
    for lead, who in (('B', f'{B} leads ({A} lags)'), ('A', f'{A} leads ({B} lags)')):
        print(f'\n  {who}:')
        print(f'   lag[s]  coll(margin)  coll@{CLEAR_M*100:.0f}cm   min[cm]   colliding-arc')
        for L in lags:
            qA, qB, _ = build(posA, posB, lead, L, dur)
            c, cc, mn, arc = scan(qA, qB, ba, bb)
            flag = f'  <-- CLEAR (>= {CLEAR_M*100:.0f}cm)' if cc == 0 else ''
            print(f'   {L:5.1f}     {c:6d}       {cc:6d}     {mn*100:6.1f}    {arc}{flag}')
    print(f'\n  A row with coll@{CLEAR_M*100:.0f}cm = 0 is collision-free under the REAL gate:')
    print('  step_34 will select it and step_35 will execute it. The coll(margin)')
    print('  column is the old oversized test (flags arms ~11cm apart). Tune the gate')
    print('  with:  DUAL_ARM_CLEARANCE_M=0.03 ros2 run dual_arm_sync step_34\n')


if __name__ == '__main__':
    main()