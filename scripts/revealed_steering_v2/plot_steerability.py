"""Plot steerability at best coefficient vs decidedness |P(A)-0.5|."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EXPERIMENT_DIR = Path("experiments/revealed_steering_v2")
CHECKPOINT = EXPERIMENT_DIR / "checkpoint.jsonl"
BEST_MULT = 0.02
EFFECT_THRESHOLD = 0.05

# Load trials
trials = []
with open(CHECKPOINT) as f:
    for line in f:
        trials.append(json.loads(line))

# Filter to probe condition only
probe_trials = [t for t in trials if t.get("condition", "probe") == "probe"]

# Group by pair_id and multiplier
pair_mult = defaultdict(list)
for t in probe_trials:
    mult = t.get("multiplier") or t.get("mult")
    if mult is None:
        coef = t["coefficient"]
        mult = round(coef / 52822.84, 3)
    pair_mult[(t["pair_id"], mult)].append(t)

# Compute P(A) at baseline and at best coefficient per pair
pair_ids = sorted({pid for pid, _ in pair_mult.keys()})

baseline_pa = {}
best_pa = {}

for pid in pair_ids:
    base_trials = pair_mult.get((pid, 0.0), [])
    if not base_trials:
        base_trials = [t for t in probe_trials if t["pair_id"] == pid and abs(t["coefficient"]) < 1.0]
    if base_trials:
        valid = [t for t in base_trials if t.get("choice_original") in ("a", "b")]
        if valid:
            baseline_pa[pid] = sum(1 for t in valid if t["choice_original"] == "a") / len(valid)

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

common_pairs = sorted(set(baseline_pa.keys()) & set(best_pa.keys()))
print(f"Pairs with both baseline and best coef data: {len(common_pairs)}")

x_pa = np.array([baseline_pa[pid] for pid in common_pairs])
y_best = np.array([best_pa[pid] for pid in common_pairs])
steerability = np.abs(y_best - x_pa)
decidedness = np.abs(x_pa - 0.5)  # 0 = borderline, 0.5 = fully decided

# Compute flips: baseline and steered on opposite sides of 0.5
flipped = ((x_pa - 0.5) * (y_best - 0.5)) < 0  # True if they cross 0.5

# Group by discrete decidedness
dec_groups_steer = defaultdict(list)
dec_groups_exceed = defaultdict(list)
dec_groups_flip = defaultdict(list)
for d, s, f in zip(decidedness, steerability, flipped):
    key = round(d, 1)
    dec_groups_steer[key].append(s)
    dec_groups_exceed[key].append(1.0 if s > EFFECT_THRESHOLD else 0.0)
    dec_groups_flip[key].append(float(f))

dec_vals = sorted(dec_groups_steer.keys())
means = [np.mean(dec_groups_steer[v]) for v in dec_vals]
sems = [np.std(dec_groups_steer[v]) / np.sqrt(len(dec_groups_steer[v])) for v in dec_vals]
counts = [len(dec_groups_steer[v]) for v in dec_vals]
prob_exceed = [np.mean(dec_groups_exceed[v]) for v in dec_vals]
prob_exceed_se = [np.sqrt(p * (1 - p) / n) for p, n in zip(prob_exceed, counts)]
prob_flip = [np.mean(dec_groups_flip[v]) for v in dec_vals]
prob_flip_se = [np.sqrt(p * (1 - p) / n) if n > 0 else 0 for p, n in zip(prob_flip, counts)]

print(f"\n|P(A)-0.5| | n | mean |shift| | SEM | P(>5%) | P(flip)")
for v, n, m, s, p, pf in zip(dec_vals, counts, means, sems, prob_exceed, prob_flip):
    print(f"  {v:.1f}       | {n:3d} | {m:.3f}        | {s:.3f} | {p:.3f}  | {pf:.3f}")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

ax1.bar(dec_vals, means, width=0.08, color="steelblue", alpha=0.8, yerr=sems, capsize=4)
for v, m, n, s in zip(dec_vals, means, counts, sems):
    ax1.text(v, m + s + 0.01, f"n={n}", ha="center", va="bottom", fontsize=8, color="gray")
ax1.set_xlabel("|P(A) - 0.5| at baseline (0 = borderline, 0.5 = decided)", fontsize=11)
ax1.set_ylabel(f"Mean |shift in P(A)| at mult={BEST_MULT}", fontsize=11)
ax1.set_title(f"Mean steerability vs decidedness", fontsize=13)
ax1.set_xlim(-0.05, 0.55)
ax1.set_ylim(0, None)
ax1.grid(True, alpha=0.3, axis="y")

# Plot 2: P(effect > threshold) vs decidedness
ax2.bar(dec_vals, prob_exceed, width=0.08, color="coral", alpha=0.8, yerr=prob_exceed_se, capsize=4)
for v, p, n, s in zip(dec_vals, prob_exceed, counts, prob_exceed_se):
    ax2.text(v, p + s + 0.01, f"n={n}", ha="center", va="bottom", fontsize=8, color="gray")
ax2.set_xlabel("|P(A) - 0.5| at baseline (0 = borderline, 0.5 = decided)", fontsize=11)
ax2.set_ylabel(f"P(|shift in P(A)| > {EFFECT_THRESHOLD})", fontsize=11)
ax2.set_title(f"Probability of >{int(EFFECT_THRESHOLD*100)}% steering effect", fontsize=13)
ax2.set_xlim(-0.05, 0.55)
ax2.set_ylim(0, 1)
ax2.grid(True, alpha=0.3, axis="y")

# Plot 3: P(flip) vs decidedness
ax3.bar(dec_vals, prob_flip, width=0.08, color="mediumseagreen", alpha=0.8, yerr=prob_flip_se, capsize=4)
for v, p, n, s in zip(dec_vals, prob_flip, counts, prob_flip_se):
    ax3.text(v, p + s + 0.01, f"n={n}", ha="center", va="bottom", fontsize=8, color="gray")
ax3.set_xlabel("|P(A) - 0.5| at baseline (0 = borderline, 0.5 = decided)", fontsize=11)
ax3.set_ylabel("P(preference flipped)", fontsize=11)
ax3.set_title("Probability of flipping preference", fontsize=13)
ax3.set_xlim(-0.05, 0.55)
ax3.set_ylim(0, 1)
ax3.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
out = EXPERIMENT_DIR / "assets" / "plot_022826_steerability_vs_decidedness.png"
plt.savefig(out, dpi=150)
print(f"\nSaved to {out}")
plt.close()
