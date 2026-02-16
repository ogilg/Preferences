import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

BASE = Path("experiments/probe_science/active_learning_calibration")

with open(BASE / "analysis_1_results.json") as f:
    a1 = json.load(f)

with open(BASE / "analysis_2_results.json") as f:
    a2 = json.load(f)

# Analysis 1: single values per iteration
a1_x = [r["comparisons_per_task"] for r in a1]
a1_rank_corr = [r["rank_correlation_with_full"] for r in a1]
a1_pw_mean = [r["pairwise_accuracy_mean"] for r in a1]
a1_pw_std = [r["pairwise_accuracy_std"] for r in a1]

# Analysis 2: group by fraction, compute mean +/- std across seeds
a2_by_fraction: dict[float, list[dict]] = defaultdict(list)
for r in a2:
    a2_by_fraction[r["fraction"]].append(r)

fractions_sorted = sorted(a2_by_fraction.keys())
a2_x = []
a2_rank_corr_mean = []
a2_rank_corr_std = []
a2_pw_mean = []
a2_pw_std = []

for frac in fractions_sorted:
    runs = a2_by_fraction[frac]
    cpt_values = [r["comparisons_per_task"] for r in runs]
    a2_x.append(np.mean(cpt_values))

    rank_corrs = [r["rank_correlation_with_full"] for r in runs]
    a2_rank_corr_mean.append(np.mean(rank_corrs))
    a2_rank_corr_std.append(np.std(rank_corrs))

    pw_accs = [r["pairwise_accuracy_mean"] for r in runs]
    a2_pw_mean.append(np.mean(pw_accs))
    a2_pw_std.append(np.std(pw_accs))

a2_x = np.array(a2_x)
a2_rank_corr_mean = np.array(a2_rank_corr_mean)
a2_rank_corr_std = np.array(a2_rank_corr_std)
a2_pw_mean = np.array(a2_pw_mean)
a2_pw_std = np.array(a2_pw_std)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

# Top panel: Rank correlation
ax1.plot(a1_x, a1_rank_corr, color="tab:blue", marker="o", linestyle="-",
         label="Active learning (iteration truncation)")
ax1.plot(a2_x, a2_rank_corr_mean, color="tab:orange", marker="^", linestyle="-",
         label="Random pair subsampling")
ax1.fill_between(a2_x,
                 a2_rank_corr_mean - a2_rank_corr_std,
                 a2_rank_corr_mean + a2_rank_corr_std,
                 color="tab:orange", alpha=0.2)
ax1.set_ylabel("Rank correlation with\nfull-data utilities")
ax1.legend(frameon=False)

# Bottom panel: Pairwise accuracy
ax1_pw_mean = np.array(a1_pw_mean)
ax1_pw_std = np.array(a1_pw_std)

ax2.plot(a1_x, a1_pw_mean, color="tab:blue", marker="o", linestyle="-")
ax2.errorbar(a1_x, a1_pw_mean, yerr=a1_pw_std, color="tab:blue", fmt="none", capsize=3)
ax2.plot(a2_x, a2_pw_mean, color="tab:orange", marker="^", linestyle="-")
ax2.fill_between(a2_x,
                 a2_pw_mean - a2_pw_std,
                 a2_pw_mean + a2_pw_std,
                 color="tab:orange", alpha=0.2)
ax2.axhline(0.5, color="gray", linestyle="--", linewidth=1)
ax2.set_xlabel("Comparisons per task")
ax2.set_ylabel("Probe pairwise accuracy\n(held-out)")

plt.tight_layout()

out_path = BASE / "assets" / "plot_021626_comparison.png"
fig.savefig(out_path, dpi=150)
print(f"Saved to {out_path}")
