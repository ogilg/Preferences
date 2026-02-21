"""Diagnostic plot for Exp 3 (minimal pairs): delta distributions.

Shows:
1. Joint histogram of behavioral vs probe deltas (all 2000 points)
2. Per-task distributions: for each task, the spread of behavioral and probe deltas across 40 conditions
3. Probe deltas demeaned (subtract condition mean) to isolate task-specific signal
"""

import json
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy import stats

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(REPO_ROOT, "experiments/ood_system_prompts/analysis_results_full.json")
ASSETS_DIR = os.path.join(REPO_ROOT, "experiments/ood_system_prompts/assets")


def main():
    os.makedirs(ASSETS_DIR, exist_ok=True)

    with open(DATA_PATH) as f:
        data = json.load(f)

    exp3 = data["exp3"]["L31"]
    bd = np.array(exp3["behavioral_deltas"])
    pd = np.array(exp3["probe_deltas"])
    labels = np.array(exp3["condition_labels"])

    unique_conds = sorted(np.unique(labels))

    # Demean probe deltas: subtract per-condition mean to remove the constant offset
    pd_demeaned = pd.copy()
    for cid in unique_conds:
        mask = labels == cid
        pd_demeaned[mask] -= pd[mask].mean()

    # --- Figure: 2x2 diagnostic ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: Raw scatter (same as main plot but bigger)
    ax = axes[0, 0]
    ax.scatter(bd, pd, alpha=0.15, s=8, color="#377eb8", rasterized=True)
    slope, intercept, *_ = stats.linregress(bd, pd)
    x_range = np.linspace(-1, 1, 200)
    ax.plot(x_range, slope * x_range + intercept, color="#e41a1c", linewidth=1.5)
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)
    r_raw = stats.pearsonr(bd, pd)[0]
    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel("Behavioral delta")
    ax.set_ylabel("Probe delta (raw)")
    ax.set_title(f"Raw deltas (r={r_raw:.3f})")

    # Panel 2: Demeaned probe scatter — removes the ~2.2 constant offset per condition
    ax = axes[0, 1]
    ax.scatter(bd, pd_demeaned, alpha=0.15, s=8, color="#377eb8", rasterized=True)
    slope_d, intercept_d, *_ = stats.linregress(bd, pd_demeaned)
    ax.plot(x_range, slope_d * x_range + intercept_d, color="#e41a1c", linewidth=1.5)
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)
    r_dem = stats.pearsonr(bd, pd_demeaned)[0]
    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel("Behavioral delta")
    ax.set_ylabel("Probe delta (condition-demeaned)")
    ax.set_title(f"Demeaned probe deltas (r={r_dem:.3f})")

    # Panel 3: Marginal distributions of behavioral deltas
    ax = axes[1, 0]
    ax.hist(bd, bins=60, color="#377eb8", alpha=0.7, edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Behavioral delta (Δp_choose)")
    ax.set_ylabel("Count")
    ax.set_title(f"Behavioral delta distribution (std={bd.std():.3f})")
    ax.set_xlim(-1.05, 1.05)

    # Panel 4: Marginal distributions of probe deltas (raw and demeaned)
    ax = axes[1, 1]
    ax.hist(pd, bins=60, color="#e41a1c", alpha=0.5, edgecolor="white", linewidth=0.3, label=f"Raw (mean={pd.mean():.2f})")
    ax.hist(pd_demeaned, bins=60, color="#377eb8", alpha=0.5, edgecolor="white", linewidth=0.3, label=f"Demeaned (mean={pd_demeaned.mean():.2f})")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Probe delta")
    ax.set_ylabel("Count")
    ax.set_title("Probe delta distributions")
    ax.legend(fontsize=9)

    fig.suptitle("Exp 3: Minimal Pairs — Delta Distributions at L31 (n=2000)", fontsize=13, y=1.01)
    fig.tight_layout()
    save_path = os.path.join(ASSETS_DIR, "plot_022126_exp3_distributions.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
