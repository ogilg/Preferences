"""Check position bias and additivity for Phase 1 steering results."""
import json
import numpy as np
from collections import defaultdict

with open('experiments/steering/replication/results/steering_phase1.json') as f:
    d = json.load(f)
results = d['results']

# Position bias: control P(a) by ordering
ctrl_orig = [1.0 if r == 'a' else 0.0
             for t in results
             if t['condition'] == 'control' and t['ordering'] == 'original'
             for r in t['responses'] if r != 'parse_fail']
ctrl_swap = [1.0 if r == 'a' else 0.0
             for t in results
             if t['condition'] == 'control' and t['ordering'] == 'swapped'
             for r in t['responses'] if r != 'parse_fail']
print(f'Control P(a) - original: {np.mean(ctrl_orig):.3f} (n={len(ctrl_orig)})')
print(f'Control P(a) - swapped:  {np.mean(ctrl_swap):.3f} (n={len(ctrl_swap)})')
print()

ctrl_pa = np.mean(ctrl_orig + ctrl_swap)


def get_pa_by_coef(cond):
    """Return P(a) by coefficient for a given condition."""
    p_by_coef = defaultdict(list)
    for t in results:
        if t['condition'] != cond:
            continue
        for r in t['responses']:
            if r != 'parse_fail':
                p_by_coef[t['coefficient']].append(1.0 if r == 'a' else 0.0)
    return {c: np.mean(v) for c, v in p_by_coef.items()}


boost_a_pa = get_pa_by_coef('boost_a')     # P(a): should go UP
boost_b_pa = get_pa_by_coef('boost_b')     # P(a): should go DOWN
suppress_a_pa = get_pa_by_coef('suppress_a')  # P(a): should go DOWN (but paradoxically goes UP)
suppress_b_pa = get_pa_by_coef('suppress_b')  # P(a): should go UP
diff_ab_pa = get_pa_by_coef('diff_ab')     # P(a): should go UP
diff_ba_pa = get_pa_by_coef('diff_ba')     # P(a): should go DOWN

print('All conditions — P(a) by coefficient:')
print(f'ctrl: {ctrl_pa:.3f}')
for cond, pa in [('boost_a', boost_a_pa), ('boost_b', boost_b_pa),
                 ('suppress_a', suppress_a_pa), ('suppress_b', suppress_b_pa),
                 ('diff_ab', diff_ab_pa), ('diff_ba', diff_ba_pa)]:
    vals = [(c, pa[c]) for c in sorted(pa)]
    print(f'{cond}: {[(f"{c:+.0f}", f"{v:.3f}") for c, v in vals]}')

print()
print('Additivity check: diff_ab should ~ boost_a + suppress_b - ctrl')
print('(all in P(a) terms; diff_ab: +A-B, so boost_a boosts A, suppress_b suppresses B -> raises P(a))')
print(f'{"coef":>8} | {"diff_ab":>7} | {"boost_a":>7} | {"supp_b":>7} | {"additive":>9} | {"delta":>7}')
for coef in sorted(c for c in boost_a_pa if c > 0):
    ba = boost_a_pa[coef]
    sb = suppress_b_pa.get(coef, float('nan'))  # suppress_b in P(a) terms: should be > ctrl
    da = diff_ab_pa.get(coef, float('nan'))
    additive = ba + sb - ctrl_pa
    print(f'{coef:+8.0f} | {da:7.3f} | {ba:7.3f} | {sb:7.3f} | {additive:9.3f} | {da - additive:+7.3f}')
