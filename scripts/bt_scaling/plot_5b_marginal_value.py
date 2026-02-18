import json
import numpy as np
import matplotlib.pyplot as plt

with open("experiments/probe_science/bt_scaling/experiment5_results.json") as f:
    results = json.load(f)

data_5b = results["experiment_5b"]
base_sizes = data_5b["base_sizes"]
conditions = data_5b["conditions"]

# For each base size, compute delta = (condition accuracy - base accuracy) for each seed
# AL-next: deterministic (same across seeds), so delta has no variance
# Random: different across seeds, so compute mean and std of deltas

panels = [
    ("BT + scaled (Bradley-Terry probe)", "bt_mean"),
    ("Ridge + Thurstonian (Ridge probe)", "ridge_mean"),
]

fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150, sharey=True)

bar_width = 0.35

for ax, (title, metric) in zip(axes, panels):
    al_deltas_mean = []
    random_deltas_mean = []
    random_deltas_std = []

    for base_size in base_sizes:
        base_entries = conditions[f"base_only_{base_size}"]
        base_acc = base_entries[0][metric]  # same across seeds

        al_entries = conditions[f"al_next_{base_size}"]
        al_acc = al_entries[0][metric]  # deterministic, same across seeds
        al_delta = (al_acc - base_acc) * 100
        al_deltas_mean.append(al_delta)

        rand_entries = conditions[f"random_{base_size}"]
        rand_deltas = [(e[metric] - base_acc) * 100 for e in rand_entries]
        random_deltas_mean.append(np.mean(rand_deltas))
        random_deltas_std.append(np.std(rand_deltas))

    x = np.arange(len(base_sizes))

    ax.bar(x - bar_width / 2, al_deltas_mean, bar_width, label="AL-next", color="tab:blue")
    ax.bar(x + bar_width / 2, random_deltas_mean, bar_width, label="Random", color="tab:orange",
           yerr=random_deltas_std, capsize=3)

    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Base size (pairs)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s // 1000}K" for s in base_sizes])
    ax.set_title(title)
    ax.legend()

axes[0].set_ylabel("Delta accuracy (pp)")

fig.suptitle("5b: Marginal value of 2K additional pairs", fontsize=14)
plt.tight_layout()
plt.savefig("experiments/probe_science/bt_scaling/assets/plot_021726_5b_marginal_value.png")
print("Saved plot.")
