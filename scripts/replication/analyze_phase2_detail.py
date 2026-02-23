"""Detailed Phase 2 analysis: P(a) at specific coefficients by tercile."""
import json
import numpy as np
from collections import defaultdict

with open('experiments/steering/replication/results/steering_phase2.json') as f:
    d = json.load(f)
results = d['results']
t1, t2 = d['tercile_thresholds']

def tercile(dm):
    if dm <= t1: return 'small'
    if dm <= t2: return 'medium'
    return 'large'

# P(a) by tercile and condition at each coefficient
data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
# data[tercile][condition][coef] -> list of P(a) values
for r in results:
    ter = tercile(r['delta_mu'])
    cond = r['condition']
    coef = r['coefficient']
    for resp in r['responses']:
        if resp != 'parse_fail':
            data[ter][cond][coef].append(1.0 if resp == 'a' else 0.0)

print('P(a) by tercile, condition, and coefficient:')
print()

for ter in ['small', 'medium', 'large']:
    ctrl_all = data[ter]['control'].get(0.0, [])
    ctrl_mean = np.mean(ctrl_all) if ctrl_all else float('nan')
    print(f'Tercile: {ter} (control P(a)={ctrl_mean:.3f}, n={len(ctrl_all)})')

    for coef in sorted(c for c in data[ter]['boost_a'] if c > 0):
        vals = data[ter]['boost_a'][coef]
        m = np.mean(vals)
        shift = m - ctrl_mean
        print(f'  boost_a at +{coef:.0f}: P(a)={m:.3f} ({shift*100:+.1f}pp vs ctrl), n={len(vals)}')
    print()

# Count decisive pair characteristics
print('\nDecisive pair baseline P(a) analysis:')
pair_ctrl = defaultdict(list)
for r in results:
    if r['condition'] == 'control':
        for resp in r['responses']:
            if resp != 'parse_fail':
                pair_ctrl[r['pair_id']].append(1.0 if resp == 'a' else 0.0)

# Distribution of control P(a) for decisive pairs
ctrl_pa_vals = [np.mean(v) for v in pair_ctrl.values() if v]
ctrl_pa_arr = np.array(ctrl_pa_vals)
print(f'n pairs with control data: {len(ctrl_pa_arr)}')
print(f'Mean P(a): {ctrl_pa_arr.mean():.3f}')
print(f'Pairs with P(a) > 0.7: {(ctrl_pa_arr > 0.7).sum()} ({100*(ctrl_pa_arr>0.7).mean():.1f}%)')
print(f'Pairs with P(a) < 0.3: {(ctrl_pa_arr < 0.3).sum()} ({100*(ctrl_pa_arr<0.3).mean():.1f}%)')
print(f'Pairs with P(a) 0.3-0.7: {((ctrl_pa_arr>=0.3)&(ctrl_pa_arr<=0.7)).sum()} ({100*((ctrl_pa_arr>=0.3)&(ctrl_pa_arr<=0.7)).mean():.1f}%)')
