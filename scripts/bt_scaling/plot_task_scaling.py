import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

results_path = "/Users/oscargilg/Dev/MATS/Preferences-bt-al-ablation/experiments/probe_science/bt_scaling/experiment6_results.json"
output_path = "/Users/oscargilg/Dev/MATS/Preferences-bt-al-ablation/experiments/probe_science/bt_scaling/assets/plot_021726_task_scaling.png"

with open(results_path) as f:
    results = json.load(f)

# Collect per-fraction values
data_by_fraction = defaultdict(lambda: {"ridge_acc": [], "bt_acc": [], "ridge_alpha": [], "bt_lambda": [], "n_train_tasks": None})

for run in results["runs"]:
    frac = run["fraction"]
    entry = data_by_fraction[frac]
    entry["n_train_tasks"] = run["n_train_tasks"]

    if frac == 1.0:
        # Use mean across fold_results for this seed
        folds = run["fold_results"]
        entry["ridge_acc"].append(np.mean([f["ridge_acc"] for f in folds]))
        entry["bt_acc"].append(np.mean([f["bt_acc"] for f in folds]))
        entry["ridge_alpha"].append(np.mean([f["best_ridge_alpha"] for f in folds]))
        entry["bt_lambda"].append(np.mean([f["best_bt_lambda"] for f in folds]))
    else:
        entry["ridge_acc"].append(run["ridge_acc"])
        entry["bt_acc"].append(run["bt_acc"])
        entry["ridge_alpha"].append(run["best_ridge_alpha"])
        entry["bt_lambda"].append(run["best_bt_lambda"])

# Sort by n_train_tasks
fractions_sorted = sorted(data_by_fraction.keys())
n_tasks = [data_by_fraction[f]["n_train_tasks"] for f in fractions_sorted]

ridge_acc_mean = [np.mean(data_by_fraction[f]["ridge_acc"]) * 100 for f in fractions_sorted]
ridge_acc_std = [np.std(data_by_fraction[f]["ridge_acc"]) * 100 for f in fractions_sorted]
bt_acc_mean = [np.mean(data_by_fraction[f]["bt_acc"]) * 100 for f in fractions_sorted]
bt_acc_std = [np.std(data_by_fraction[f]["bt_acc"]) * 100 for f in fractions_sorted]

ridge_alpha_mean = [np.mean(data_by_fraction[f]["ridge_alpha"]) for f in fractions_sorted]
ridge_alpha_std = [np.std(data_by_fraction[f]["ridge_alpha"]) for f in fractions_sorted]
bt_lambda_mean = [np.mean(data_by_fraction[f]["bt_lambda"]) for f in fractions_sorted]
bt_lambda_std = [np.std(data_by_fraction[f]["bt_lambda"]) for f in fractions_sorted]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

# Left panel: Task scaling curves
ax1.errorbar(n_tasks, ridge_acc_mean, yerr=ridge_acc_std, color="tab:blue", marker="o", capsize=4, label="Ridge")
ax1.errorbar(n_tasks, bt_acc_mean, yerr=bt_acc_std, color="tab:green", marker="s", capsize=4, label="BT+scaled")
ax1.set_xlabel("Number of training tasks")
ax1.set_ylabel("Held-out pairwise accuracy (%)")
ax1.set_title("Task scaling")
ax1.set_ylim(65, 76)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right panel: Best regularization vs N tasks
ax2.errorbar(n_tasks, ridge_alpha_mean, yerr=ridge_alpha_std, color="tab:blue", marker="o", capsize=4, label="Ridge alpha")
ax2.errorbar(n_tasks, bt_lambda_mean, yerr=bt_lambda_std, color="tab:green", marker="s", capsize=4, label="BT lambda")
ax2.set_xlabel("Number of training tasks")
ax2.set_ylabel("Best hyperparameter")
ax2.set_yscale("log")
ax2.set_title("Optimal regularization")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_path)
print(f"Saved to {output_path}")
