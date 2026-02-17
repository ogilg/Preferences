import json
import numpy as np
import matplotlib.pyplot as plt

results_path = "/Users/oscargilg/Dev/MATS/Preferences/experiments/probe_science/bt_scaling/experiment3_results.json"
out_path = "/Users/oscargilg/Dev/MATS/Preferences/experiments/probe_science/bt_scaling/assets/plot_021626_scaling_curves.png"

with open(results_path) as f:
    data = json.load(f)

fractions = data["fractions"]

def get_mean_std(entries, fractions):
    means = []
    stds = []
    for frac in fractions:
        accs = [e["mean_acc"] for e in entries if e["fraction"] == frac]
        means.append(np.mean(accs))
        stds.append(np.std(accs) / np.sqrt(len(accs)))
    return np.array(means), np.array(stds)

ridge_mean, ridge_se = get_mean_std(data["ridge"], fractions)
bt_mean, bt_se = get_mean_std(data["bt"], fractions)
bts_mean, bts_se = get_mean_std(data["bt_scaled"], fractions)

fig, ax = plt.subplots(figsize=(7, 4.5))

ax.plot(fractions, ridge_mean, "o-", color="tab:blue", label="Ridge", markersize=5)
ax.fill_between(fractions, ridge_mean - ridge_se, ridge_mean + ridge_se, color="tab:blue", alpha=0.18)

ax.plot(fractions, bt_mean, "s-", color="tab:orange", label="BT standard", markersize=5)
ax.fill_between(fractions, bt_mean - bt_se, bt_mean + bt_se, color="tab:orange", alpha=0.18)

ax.plot(fractions, bts_mean, "^-", color="tab:green", label="BT scaled", markersize=5)
ax.fill_between(fractions, bts_mean - bts_se, bts_mean + bts_se, color="tab:green", alpha=0.18)

ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)

ax.set_xlabel("Fraction of training pairs")
ax.set_ylabel("Held-out pairwise accuracy")
ax.set_title("Scaling Curves: Accuracy vs Training Data Fraction")
ax.set_ylim(0.58, 0.76)
ax.set_xticks(fractions)
ax.legend(loc="lower right")
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(out_path, dpi=200)
print(f"Saved to {out_path}")
