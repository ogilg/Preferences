"""Per-ordering slope analysis for Phase 1 — mirrors original finding 4 (task-identity tracking)."""
import json
import numpy as np
from collections import defaultdict
from scipy import stats

with open('experiments/steering/replication/results/steering_phase1.json') as f:
    d = json.load(f)
results = d['results']

# Control P(a) per ordering
ctrl_pa = {}
for ordering in ['original', 'swapped']:
    vals = [1.0 if r == 'a' else 0.0
            for t in results
            if t['condition'] == 'control' and t['ordering'] == ordering
            for r in t['responses'] if r != 'parse_fail']
    ctrl_pa[ordering] = np.mean(vals)
    print(f'Control P(a) - {ordering}: {ctrl_pa[ordering]:.3f} (n={len(vals)})')

print()

# For each condition and ordering, compute slope of P(a) vs coef (positive only)
# to check whether effect tracks task identity or position
for cond in ['boost_a', 'boost_b', 'diff_ab']:
    print(f'\n--- {cond} ---')
    for ordering in ['original', 'swapped']:
        p_by_coef = defaultdict(list)
        for t in results:
            if t['condition'] != cond or t['ordering'] != ordering:
                continue
            coef = t['coefficient']
            for r in t['responses']:
                if r != 'parse_fail':
                    p_by_coef[coef].append(1.0 if r == 'a' else 0.0)

        ctrl = ctrl_pa[ordering]
        # Positive coefficients only (positive intent)
        pos_coefs = sorted(c for c in p_by_coef if c > 0)
        if not pos_coefs:
            continue

        # Per-pair-coef observations
        coef_means = {c: np.mean(p_by_coef[c]) for c in pos_coefs}

        # Effect at moderate coefficient (+2641)
        if 2641.14 in p_by_coef or any(abs(c - 2641) < 100 for c in pos_coefs):
            mod_coef = min(pos_coefs, key=lambda c: abs(c - 2641))
            mod_p = np.mean(p_by_coef[mod_coef])
            print(f'  {ordering}: ctrl={ctrl:.3f}, P(a) at +{mod_coef:.0f}: {mod_p:.3f} ({(mod_p-ctrl)*100:+.1f}pp)')
        else:
            print(f'  {ordering}: no moderate coef found')

print()
print('--- Summary: does effect track task or position? ---')
print('boost_a: steered task is A (same task regardless of ordering)')
print('  If effect is task-identity: slope should be similar for original and swapped orderings')
print('  If effect is position-dependent: slope should differ between orderings')
