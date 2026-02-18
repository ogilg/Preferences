import json
import numpy as np
import matplotlib.pyplot as plt

with open("experiments/probe_science/bt_scaling/experiment5_results.json") as f:
    data = json.load(f)

exp5a = data["experiment_5a"]
fractions = exp5a["fractions"]
conditions = exp5a["conditions"]

# Extract means and stds for each condition and probe type
def extract_stats(prefix, metric_key):
    means = []
    stds = []
    for frac in fractions:
        key = f"{prefix}_{frac}"
        values = [run[metric_key] for run in conditions[key]]
        means.append(np.mean(values) * 100)
        stds.append(np.std(values) * 100)
    return np.array(means), np.array(stds)

al_bt_mean, al_bt_std = extract_stats("al_order", "bt_mean")
rand_bt_mean, rand_bt_std = extract_stats("random", "bt_mean")
al_ridge_mean, al_ridge_std = extract_stats("al_order", "ridge_mean")
rand_ridge_mean, rand_ridge_std = extract_stats("random", "ridge_mean")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

frac_pct = [f * 100 for f in fractions]

# Left panel: BT + scaled scores
ax1.plot(frac_pct, al_bt_mean, "o-", color="tab:blue", label="AL-order")
ax1.errorbar(frac_pct, rand_bt_mean, yerr=rand_bt_std, fmt="o--", color="tab:orange",
             label="Random", capsize=3)
ax1.set_xlabel("Training pairs (%)")
ax1.set_ylabel("Held-out pairwise accuracy (%)")
ax1.set_title("BT probe + scaled BT scores")
ax1.legend()
ax1.set_ylim(55, None)
ax1.grid(alpha=0.3)

# Right panel: Ridge + Thurstonian scores
ax2.plot(frac_pct, al_ridge_mean, "o-", color="tab:blue", label="AL-order")
ax2.errorbar(frac_pct, rand_ridge_mean, yerr=rand_ridge_std, fmt="o--", color="tab:orange",
             label="Random", capsize=3)
ax2.set_xlabel("Training pairs (%)")
ax2.set_ylabel("Held-out pairwise accuracy (%)")
ax2.set_title("Ridge probe + Thurstonian scores")
ax2.legend()
ax2.set_ylim(55, None)
ax2.grid(alpha=0.3)

fig.suptitle("5a: AL-ordered vs Random subsampling", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("experiments/probe_science/bt_scaling/assets/plot_021726_5a_scaling_curves.png")
print("Saved plot.")
