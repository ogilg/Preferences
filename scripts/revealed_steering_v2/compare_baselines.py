"""Compare baseline distributions: old (OpenRouter, t=0.7, 20 trials) vs checkpoint (HF, t=1.0, 10 trials)."""

import json
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np

EXPERIMENT_DIR = Path("experiments/revealed_steering_v2")

# --- Old baseline (OpenRouter, t=0.7, 20 trials) ---
with open(EXPERIMENT_DIR / "baseline_pairwise.json") as f:
    old = json.load(f)

old_pa = {}
for pair in old["pairs"]:
    total = pair["a"] + pair["b"] + pair.get("refusal", 0)
    valid = pair["a"] + pair["b"]
    if valid > 0:
        old_pa[pair["pair_id"]] = pair["a"] / valid
print(f"Old baseline: {len(old_pa)} pairs, t=0.7, 20 trials/pair")

# --- New baseline (checkpoint coef=0, HF, t=1.0, 10 trials) ---
trials = []
with open(EXPERIMENT_DIR / "checkpoint.jsonl") as f:
    for line in f:
        trials.append(json.loads(line))

probe_trials = [t for t in trials if t.get("condition", "probe") == "probe"]

pair_baseline = defaultdict(list)
for t in probe_trials:
    coef = t["coefficient"]
    if abs(coef) < 1.0:
        pair_baseline[t["pair_id"]].append(t)

new_pa = {}
for pid, ts in pair_baseline.items():
    valid = [t for t in ts if t.get("choice_original") in ("a", "b")]
    if valid:
        new_pa[pid] = sum(1 for t in valid if t["choice_original"] == "a") / len(valid)
print(f"New baseline: {len(new_pa)} pairs, t=1.0, 10 trials/pair")

# --- Compare distributions ---
def summarise(pa_dict, label):
    vals = list(pa_dict.values())
    arr = np.array(vals)
    rounded = [round(v, 1) for v in vals]
    counts = Counter(rounded)

    n_decided_0 = sum(1 for v in vals if v <= 0.05)
    n_decided_1 = sum(1 for v in vals if v >= 0.95)
    n_borderline = sum(1 for v in vals if 0.3 <= v <= 0.7)

    print(f"\n--- {label} ---")
    print(f"  N pairs: {len(vals)}")
    print(f"  Mean P(A): {arr.mean():.3f}")
    print(f"  Std P(A):  {arr.std():.3f}")
    print(f"  Decided (P(A)<=0.05 or >=0.95): {n_decided_0 + n_decided_1} ({100*(n_decided_0+n_decided_1)/len(vals):.1f}%)")
    print(f"    P(A)<=0.05: {n_decided_0}")
    print(f"    P(A)>=0.95: {n_decided_1}")
    print(f"  Borderline (0.3-0.7): {n_borderline} ({100*n_borderline/len(vals):.1f}%)")
    print(f"  Distribution of rounded P(A):")
    for v in sorted(counts.keys()):
        print(f"    {v:.1f}: {counts[v]:4d}")

summarise(old_pa, "Old baseline (OpenRouter, t=0.7, 20 trials)")
summarise(new_pa, "New baseline (HF local, t=1.0, 10 trials)")

# --- Per-pair comparison for common pairs ---
common = sorted(set(old_pa.keys()) & set(new_pa.keys()))
print(f"\n--- Per-pair comparison ({len(common)} common pairs) ---")
if common:
    old_arr = np.array([old_pa[p] for p in common])
    new_arr = np.array([new_pa[p] for p in common])
    r = np.corrcoef(old_arr, new_arr)[0, 1]
    mae = np.mean(np.abs(old_arr - new_arr))
    print(f"  Correlation: r={r:.3f}")
    print(f"  Mean |diff|: {mae:.3f}")

    # How many flip decidedness
    n_both_decided = sum(1 for o, n in zip(old_arr, new_arr) if (o <= 0.1 or o >= 0.9) and (n <= 0.1 or n >= 0.9))
    n_old_decided_new_not = sum(1 for o, n in zip(old_arr, new_arr) if (o <= 0.1 or o >= 0.9) and not (n <= 0.1 or n >= 0.9))
    n_old_not_new_decided = sum(1 for o, n in zip(old_arr, new_arr) if not (o <= 0.1 or o >= 0.9) and (n <= 0.1 or n >= 0.9))
    print(f"  Both decided (P(A)<=0.1 or >=0.9): {n_both_decided}")
    print(f"  Old decided, new not: {n_old_decided_new_not}")
    print(f"  Old not decided, new decided: {n_old_not_new_decided}")
