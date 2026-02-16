import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = "experiments/probe_science/active_learning_calibration/analysis_2_results.json"
OUT_PATH = "experiments/probe_science/active_learning_calibration/assets/plot_021626_random_subsampling.png"

with open(DATA_PATH) as f:
    results = json.load(f)

# Group by fraction
grouped: dict[float, list[dict]] = defaultdict(list)
for entry in results:
    grouped[entry["fraction"]].append(entry)

fractions = sorted(grouped.keys())
comparisons_per_task = [grouped[f][0]["comparisons_per_task"] for f in fractions]

rank_corr_mean = []
rank_corr_std = []
pairwise_acc_mean = []
pairwise_acc_std = []

for f in fractions:
    entries = grouped[f]
    rc = np.array([e["rank_correlation_with_full"] for e in entries])
    pa = np.array([e["pairwise_accuracy_mean"] for e in entries])
    rank_corr_mean.append(rc.mean())
    rank_corr_std.append(rc.std())
    pairwise_acc_mean.append(pa.mean())
    pairwise_acc_std.append(pa.std())

x = np.array(comparisons_per_task)
rank_corr_mean = np.array(rank_corr_mean)
rank_corr_std = np.array(rank_corr_std)
pairwise_acc_mean = np.array(pairwise_acc_mean)
pairwise_acc_std = np.array(pairwise_acc_std)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

# Top panel: rank correlation
ax1.plot(x, rank_corr_mean, "o-", color="C0", markersize=5)
ax1.fill_between(
    x,
    rank_corr_mean - rank_corr_std,
    rank_corr_mean + rank_corr_std,
    alpha=0.2,
    color="C0",
)
ax1.set_ylabel("Rank correlation with full data")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# Bottom panel: pairwise accuracy
ax2.plot(x, pairwise_acc_mean, "o-", color="C0", markersize=5)
ax2.fill_between(
    x,
    pairwise_acc_mean - pairwise_acc_std,
    pairwise_acc_mean + pairwise_acc_std,
    alpha=0.2,
    color="C0",
)
ax2.axhline(0.5, color="gray", linestyle="--", linewidth=1)
ax2.set_ylabel("Probe pairwise accuracy")
ax2.set_xlabel("Comparisons per task")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig(OUT_PATH, dpi=150)
print(f"Saved to {OUT_PATH}")
