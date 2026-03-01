"""Steerability vs decidedness using OLD baseline P(A) (OpenRouter, t=0.7, 20 trials)."""

import json
from collections import defaultdict, Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EXPERIMENT_DIR = Path("experiments/revealed_steering_v2")
BEST_MULT = 0.02
EFFECT_THRESHOLD = 0.05

# --- Old baseline P(A) ---
with open(EXPERIMENT_DIR / "baseline_pairwise.json") as f:
    old = json.load(f)

old_pa = {}
for pair in old["pairs"]:
    valid = pair["a"] + pair["b"]
    if valid > 0:
        old_pa[pair["pair_id"]] = pair["a"] / valid

# --- Steered P(A) at best coefficient from checkpoint ---
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

best_pa = {}
for pid in old_pa:
    best_trials = pair_mult.get((pid, BEST_MULT), [])
    if not best_trials:
        for m in [0.02, 0.019, 0.021]:
            best_trials = pair_mult.get((pid, m), [])
            if best_trials:
                break
    if best_trials:
        valid = [t for t in best_trials if t.get("choice_original") in ("a", "b")]
        if valid:
            best_pa[pid] = sum(1 for t in valid if t["choice_original"] == "a") / len(valid)

# Also get new baseline (coef=0) P(A) for computing shift against new baseline
new_baseline_pa = {}
for pid in old_pa:
    base_trials = pair_mult.get((pid, 0.0), [])
    if not base_trials:
        base_trials = [t for t in probe_trials if t["pair_id"] == pid and abs(t["coefficient"]) < 1.0]
    if base_trials:
        valid = [t for t in base_trials if t.get("choice_original") in ("a", "b")]
        if valid:
            new_baseline_pa[pid] = sum(1 for t in valid if t["choice_original"] == "a") / len(valid)

common = sorted(set(old_pa.keys()) & set(best_pa.keys()) & set(new_baseline_pa.keys()))
print(f"Pairs with old baseline + new baseline + steered data: {len(common)}")

# Use old baseline for x-axis (decidedness), new baseline for shift computation
x_old = np.array([old_pa[pid] for pid in common])
y_new_base = np.array([new_baseline_pa[pid] for pid in common])
y_best = np.array([best_pa[pid] for pid in common])
steerability = np.abs(y_best - y_new_base)
decidedness = np.abs(x_old - 0.5)

# Round decidedness to nearest 0.05 for finer bins (old baseline has 20 trials -> steps of 0.05)
bin_width = 0.05
# Compute flips: new baseline and steered on opposite sides of 0.5
flipped = ((y_new_base - 0.5) * (y_best - 0.5)) < 0

dec_groups_steer = defaultdict(list)
dec_groups_exceed = defaultdict(list)
dec_groups_flip = defaultdict(list)
for d, s, f in zip(decidedness, steerability, flipped):
    key = round(d / bin_width) * bin_width
    dec_groups_steer[key].append(s)
    dec_groups_exceed[key].append(1.0 if s > EFFECT_THRESHOLD else 0.0)
    dec_groups_flip[key].append(float(f))

dec_vals = sorted(dec_groups_steer.keys())
means = [np.mean(dec_groups_steer[v]) for v in dec_vals]
sems = [np.std(dec_groups_steer[v]) / np.sqrt(len(dec_groups_steer[v])) for v in dec_vals]
counts = [len(dec_groups_steer[v]) for v in dec_vals]
prob_exceed = [np.mean(dec_groups_exceed[v]) for v in dec_vals]
prob_exceed_se = [np.sqrt(p * (1 - p) / n) if n > 0 else 0 for p, n in zip(prob_exceed, counts)]
prob_flip = [np.mean(dec_groups_flip[v]) for v in dec_vals]
prob_flip_se = [np.sqrt(p * (1 - p) / n) if n > 0 else 0 for p, n in zip(prob_flip, counts)]

print(f"\n|P(A)-0.5| (old) | n | mean |shift| | SEM | P(>5%) | P(flip)")
for v, n, m, s, p, pf in zip(dec_vals, counts, means, sems, prob_exceed, prob_flip):
    print(f"  {v:.2f}            | {n:3d} | {m:.3f}        | {s:.3f} | {p:.3f}  | {pf:.3f}")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

ax1.bar(dec_vals, means, width=bin_width * 0.85, color="steelblue", alpha=0.8, yerr=sems, capsize=3)
for v, m, n, s in zip(dec_vals, means, counts, sems):
    if n >= 5:
        ax1.text(v, m + s + 0.005, f"{n}", ha="center", va="bottom", fontsize=7, color="gray")
ax1.set_xlabel("|P(A) - 0.5| from old baseline (t=0.7, 20 trials)", fontsize=11)
ax1.set_ylabel(f"Mean |shift in P(A)| at mult={BEST_MULT}", fontsize=11)
ax1.set_title("Mean steerability vs decidedness (old baseline x-axis)", fontsize=12)
ax1.set_xlim(-0.025, 0.525)
ax1.set_ylim(0, None)
ax1.grid(True, alpha=0.3, axis="y")

ax2.bar(dec_vals, prob_exceed, width=bin_width * 0.85, color="coral", alpha=0.8, yerr=prob_exceed_se, capsize=3)
for v, p, n, s in zip(dec_vals, prob_exceed, counts, prob_exceed_se):
    if n >= 5:
        ax2.text(v, p + s + 0.01, f"{n}", ha="center", va="bottom", fontsize=7, color="gray")
ax2.set_xlabel("|P(A) - 0.5| from old baseline (t=0.7, 20 trials)", fontsize=11)
ax2.set_ylabel(f"P(|shift in P(A)| > {EFFECT_THRESHOLD})", fontsize=11)
ax2.set_title(f"Probability of >{int(EFFECT_THRESHOLD*100)}% steering effect (old baseline x-axis)", fontsize=12)
ax2.set_xlim(-0.025, 0.525)
ax2.set_ylim(0, 1)
ax2.grid(True, alpha=0.3, axis="y")

ax3.bar(dec_vals, prob_flip, width=bin_width * 0.85, color="mediumseagreen", alpha=0.8, yerr=prob_flip_se, capsize=3)
for v, p, n, s in zip(dec_vals, prob_flip, counts, prob_flip_se):
    if n >= 5:
        ax3.text(v, p + s + 0.01, f"{n}", ha="center", va="bottom", fontsize=7, color="gray")
ax3.set_xlabel("|P(A) - 0.5| from old baseline (t=0.7, 20 trials)", fontsize=11)
ax3.set_ylabel("P(preference flipped)", fontsize=11)
ax3.set_title("Probability of flipping preference (old baseline x-axis)", fontsize=12)
ax3.set_xlim(-0.025, 0.525)
ax3.set_ylim(0, 1)
ax3.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
out = EXPERIMENT_DIR / "assets" / "plot_022826_steerability_vs_decidedness_oldbaseline.png"
plt.savefig(out, dpi=150)
print(f"\nSaved to {out}")
plt.close()
