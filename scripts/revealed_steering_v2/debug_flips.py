"""Debug flip computation — why are there almost no flips?"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

EXPERIMENT_DIR = Path("experiments/revealed_steering_v2")

# Load checkpoint
trials = []
with open(EXPERIMENT_DIR / "checkpoint.jsonl") as f:
    for line in f:
        trials.append(json.loads(line))

probe_trials = [t for t in trials if t.get("condition", "probe") == "probe"]

pair_mult = defaultdict(list)
for t in probe_trials:
    mult = t.get("multiplier") or t.get("mult")
    if mult is None:
        coef = t["coefficient"]
        mult = round(coef / 52822.84, 3)
    pair_mult[(t["pair_id"], mult)].append(t)

# Get baseline and steered P(A)
baseline_pa = {}
steered_pa = {}
for pid in {p for p, _ in pair_mult.keys()}:
    base = pair_mult.get((pid, 0.0), [])
    if base:
        valid = [t for t in base if t.get("choice_original") in ("a", "b")]
        if valid:
            baseline_pa[pid] = sum(1 for t in valid if t["choice_original"] == "a") / len(valid)

    best = pair_mult.get((pid, 0.02), [])
    if best:
        valid = [t for t in best if t.get("choice_original") in ("a", "b")]
        if valid:
            steered_pa[pid] = sum(1 for t in valid if t["choice_original"] == "a") / len(valid)

common = sorted(set(baseline_pa) & set(steered_pa))

# Show pairs near 0.5 baseline and their steered values
print("Pairs with baseline P(A) between 0.3 and 0.7:")
print(f"{'pair_id':>20} | base P(A) | steered P(A) | shift | flipped?")
near_borderline = []
for pid in common:
    b = baseline_pa[pid]
    s = steered_pa[pid]
    if 0.3 <= b <= 0.7:
        flipped = (b - 0.5) * (s - 0.5) < 0
        near_borderline.append((pid, b, s, s - b, flipped))

near_borderline.sort(key=lambda x: x[1])
for pid, b, s, shift, flipped in near_borderline:
    print(f"{pid:>20} | {b:.2f}      | {s:.2f}         | {shift:+.2f}  | {'YES' if flipped else 'no'}")

print(f"\nTotal borderline pairs (0.3-0.7): {len(near_borderline)}")
print(f"Flipped: {sum(1 for _, _, _, _, f in near_borderline if f)}")

# Also check: how many baseline trials per pair?
print("\n--- Trial counts at baseline ---")
trial_counts = []
for pid in common:
    base = pair_mult.get((pid, 0.0), [])
    valid = [t for t in base if t.get("choice_original") in ("a", "b")]
    trial_counts.append(len(valid))
print(f"Mean valid trials at coef=0: {np.mean(trial_counts):.1f}")
print(f"Min: {min(trial_counts)}, Max: {max(trial_counts)}")

# Check: are baseline P(A) values all 0.0 or 1.0 or 0.5?
print("\n--- Distribution of baseline P(A) for 'common' pairs ---")
vals = [baseline_pa[p] for p in common]
from collections import Counter
c = Counter(round(v, 1) for v in vals)
for v in sorted(c):
    print(f"  {v:.1f}: {c[v]}")

# Check a specific example
print("\n--- Example pair near 0.5 ---")
for pid, b, s, shift, flipped in near_borderline[:5]:
    base_trials = pair_mult.get((pid, 0.0), [])
    best_trials = pair_mult.get((pid, 0.02), [])
    base_choices = [t.get("choice_original") for t in base_trials]
    best_choices = [t.get("choice_original") for t in best_trials]
    print(f"\n{pid}: baseline={b:.2f}, steered={s:.2f}")
    print(f"  Base choices: {base_choices}")
    print(f"  Steered choices: {best_choices}")
