import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = "/workspace/repo/experiments/steering/stated_steering/results/phase1_arm_a.json"
OUTPUT_PATH = "/workspace/repo/experiments/steering/stated_steering/assets/plot_022426_arm_a_slope_vs_mu.png"

POSITIONS = ["generation", "last_token", "task_tokens"]

with open(DATA_PATH) as f:
    data = json.load(f)

# Group records by (task_id, position)
groups = defaultdict(list)
for record in data:
    key = (record["task_id"], record["position"])
    groups[key].append(record)

# For each task+position, compute slope of mean_rating ~ coefficient
# Also track mu (constant per task_id)
task_mu = {}
slopes = {pos: {} for pos in POSITIONS}

for (task_id, position), records in groups.items():
    if position not in POSITIONS:
        continue

    # Filter records where mean_rating is not None
    valid = [r for r in records if r["mean_rating"] is not None]
    if len(valid) < 2:
        continue

    coefficients = np.array([r["coefficient"] for r in valid], dtype=float)
    ratings = np.array([r["mean_rating"] for r in valid], dtype=float)

    # Linear regression: slope of mean_rating ~ coefficient
    result = stats.linregress(coefficients, ratings)
    slope = result.slope

    slopes[position][task_id] = slope
    task_mu[task_id] = valid[0]["mu"]

# Build aligned arrays per position
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, position in zip(axes, POSITIONS):
    task_ids = sorted(slopes[position].keys())
    if not task_ids:
        ax.set_title(f"{position}\n(no data)")
        continue

    mus = np.array([task_mu[tid] for tid in task_ids])
    slope_vals = np.array([slopes[position][tid] for tid in task_ids])

    r, p = stats.pearsonr(mus, slope_vals)

    ax.scatter(mus, slope_vals, alpha=0.5, s=20, color="steelblue")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    # y-axis anchored to include 0 with padding
    y_max = max(abs(slope_vals.max()), abs(slope_vals.min()))
    padding = y_max * 0.15
    ax.set_ylim(-(y_max + padding), y_max + padding)

    # x-axis anchored around mu range with some padding
    x_pad = 1.0
    ax.set_xlim(mus.min() - x_pad, mus.max() + x_pad)

    p_str = f"p={p:.3f}" if p >= 0.001 else f"p={p:.2e}"
    ax.set_title(f"{position}\nr={r:.3f}, {p_str}", fontsize=11)
    ax.set_xlabel("Thurstonian mu", fontsize=10)
    ax.set_ylabel("Slope (rating/coef)", fontsize=10)

fig.suptitle("Per-task slope vs Thurstonian mu — Arm A (Phase 1)", fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved to {OUTPUT_PATH}")
