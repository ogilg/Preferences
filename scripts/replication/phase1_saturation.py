"""Analyze per-pair saturation at max coefficient for boost_a and diff_ab."""
import json
import numpy as np
from collections import defaultdict

with open('experiments/steering/replication/results/steering_phase1.json') as f:
    d = json.load(f)
results = d['results']

# Per-pair P(a) at control, +2641, +5282 for boost_a
pair_data = defaultdict(lambda: {0.0: [], 2641: [], 5282: []})
for t in results:
    if t['condition'] not in ('control', 'boost_a'):
        continue
    coef = t['coefficient']
    key = 0.0 if t['condition'] == 'control' else (2641 if coef < 4000 else 5282)
    for r in t['responses']:
        if r != 'parse_fail':
            pair_data[t['pair_id']][key].append(1.0 if r == 'a' else 0.0)

shift_2641 = []
shift_5282 = []
for pid, d in pair_data.items():
    ctrl = np.mean(d[0.0]) if d[0.0] else None
    p2641 = np.mean(d[2641]) if d[2641] else None
    p5282 = np.mean(d[5282]) if d[5282] else None
    if ctrl is not None and p2641 is not None:
        shift_2641.append(p2641 - ctrl)
    if ctrl is not None and p5282 is not None:
        shift_5282.append(p5282 - ctrl)

shift_2641 = np.array(shift_2641)
shift_5282 = np.array(shift_5282)

print('boost_a per-pair shift:')
print(f'  At +2641: mean={np.mean(shift_2641)*100:+.1f}pp, pos={100*(shift_2641>0).mean():.1f}%, n={len(shift_2641)}')
print(f'  At +5282: mean={np.mean(shift_5282)*100:+.1f}pp, pos={100*(shift_5282>0).mean():.1f}%, n={len(shift_5282)}')
print(f'  Pairs reversed at +5282 (neg shift): {(shift_5282<0).sum()}/{len(shift_5282)}')
print(f'  Pairs positive at +5282 (pos shift): {(shift_5282>0).sum()}/{len(shift_5282)}')
print(f'  Correlation between shift_2641 and shift_5282: r={np.corrcoef(shift_2641, shift_5282)[0,1]:.3f}')

# Do the same for diff_ab
pair_diff = defaultdict(lambda: {0.0: [], 2641: [], 5282: []})
for t in results:
    if t['condition'] not in ('control', 'diff_ab'):
        continue
    coef = t['coefficient']
    if coef < 0:
        continue
    key = 0.0 if t['condition'] == 'control' else (2641 if coef < 4000 else 5282)
    for r in t['responses']:
        if r != 'parse_fail':
            pair_diff[t['pair_id']][key].append(1.0 if r == 'a' else 0.0)

diff_2641 = []
diff_5282 = []
for pid, d in pair_diff.items():
    ctrl = np.mean(d[0.0]) if d[0.0] else None
    p2641 = np.mean(d[2641]) if d[2641] else None
    p5282 = np.mean(d[5282]) if d[5282] else None
    if ctrl is not None and p2641 is not None:
        diff_2641.append(p2641 - ctrl)
    if ctrl is not None and p5282 is not None:
        diff_5282.append(p5282 - ctrl)

diff_2641 = np.array(diff_2641)
diff_5282 = np.array(diff_5282)

print('\ndiff_ab per-pair shift:')
print(f'  At +2641: mean={np.mean(diff_2641)*100:+.1f}pp, pos={100*(diff_2641>0).mean():.1f}%, n={len(diff_2641)}')
print(f'  At +5282: mean={np.mean(diff_5282)*100:+.1f}pp, pos={100*(diff_5282>0).mean():.1f}%, n={len(diff_5282)}')
