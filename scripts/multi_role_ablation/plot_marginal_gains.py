import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_PATH = Path("experiments/probe_generalization/multi_role_ablation/probe_results.json")
ASSETS_DIR = Path("experiments/probe_generalization/multi_role_ablation/assets")

PERSONA_NAMES = ["no_prompt", "villain", "midwest", "aesthete"]
PERSONA_LABELS = {"no_prompt": "No Prompt", "villain": "Villain", "midwest": "Midwest", "aesthete": "Aesthete"}

with open(RESULTS_PATH) as f:
    data = json.load(f)

conditions = {c["condition_name"]: c for c in data["conditions"]}


# --- Plot 1: Marginal gain bar chart ---
# For each n_personas, compute mean cross-persona r (r on eval personas NOT in training set)
# Baseline: best single-persona mean cross-persona r

def cross_persona_r(cond):
    """Mean r on eval personas not in the training set."""
    train_names = set(PERSONA_NAMES[p - 1] for p in cond["train_personas"])
    cross_rs = [cond["eval"][p]["pearson_r"] for p in PERSONA_NAMES if p not in train_names]
    if not cross_rs:
        return None
    return np.mean(cross_rs)


def all_eval_r(cond):
    return np.mean([cond["eval"][p]["pearson_r"] for p in PERSONA_NAMES])


# Group conditions by number of training personas
by_n = {1: [], 2: [], 3: [], 4: []}
for cond in data["conditions"]:
    n = len(cond["train_personas"])
    by_n[n].append(cond)

# Compute mean cross-persona r per group
mean_cross_by_n = {}
for n, conds in by_n.items():
    cross_rs = [cross_persona_r(c) for c in conds]
    cross_rs = [r for r in cross_rs if r is not None]
    if cross_rs:
        mean_cross_by_n[n] = np.mean(cross_rs)

# For n=4, use all-eval r since there's no held-out persona
mean_all_eval_by_n = {n: np.mean([all_eval_r(c) for c in conds]) for n, conds in by_n.items()}

# Marginal gains relative to n=1 baseline
baseline = mean_cross_by_n[1]
ns = [1, 2, 3]
gains = [mean_cross_by_n[n] - baseline for n in ns]
incremental = [0, gains[1] - gains[0], gains[2] - gains[1]]

fig, ax = plt.subplots(figsize=(7, 5))

bars = ax.bar(
    [1, 2, 3],
    [mean_cross_by_n[n] - mean_cross_by_n[max(1, n - 1)] if n > 1 else 0 for n in [1, 2, 3]],
    color=["#cccccc", "#4C72B0", "#4C72B0"],
    edgecolor="black",
    linewidth=0.8,
    width=0.6,
)

# Replace n=1 bar: show the absolute cross-persona r as text annotation instead
# Actually, let's show the incremental gain from each step: 1→2, 2→3
# And annotate with absolute values

increments = {
    "1→2": mean_cross_by_n[2] - mean_cross_by_n[1],
    "2→3": mean_cross_by_n[3] - mean_cross_by_n[2],
}

fig, ax = plt.subplots(figsize=(6, 4.5))

x_labels = ["1→2 personas", "2→3 personas"]
delta_values = [increments["1→2"], increments["2→3"]]
colors = ["#4C72B0", "#7BA0CC"]

bars = ax.bar(x_labels, delta_values, color=colors, edgecolor="black", linewidth=0.8, width=0.5)

for bar, val in zip(bars, delta_values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.0005,
        f"+{val:.3f}",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )

# Add absolute values as secondary annotation
abs_values = [mean_cross_by_n[1], mean_cross_by_n[2], mean_cross_by_n[3]]
ax.text(0.5, -0.15, f"Absolute mean cross-persona r:   1 persona = {abs_values[0]:.3f}   →   2 = {abs_values[1]:.3f}   →   3 = {abs_values[2]:.3f}",
        transform=ax.transAxes, ha="center", fontsize=9, color="gray")

ax.set_ylabel("Δ mean cross-persona r", fontsize=11)
ax.set_title("Incremental Gain from Adding Training Personas", fontsize=13)
ax.set_ylim(0, max(delta_values) * 1.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(ASSETS_DIR / "plot_022526_marginal_gains.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved marginal gains plot")
print(f"  1→2: +{increments['1→2']:.4f}")
print(f"  2→3: +{increments['2→3']:.4f}")


# --- Plot 2: Per-eval-persona panel ---
# For each eval persona, show how r changes across training conditions grouped by n_personas
# Show individual points + mean line

fig, axes = plt.subplots(1, 4, figsize=(14, 4.5), sharey=True)

for idx, eval_persona in enumerate(PERSONA_NAMES):
    ax = axes[idx]
    label = PERSONA_LABELS[eval_persona]

    for n_train in [1, 2, 3, 4]:
        rs = []
        includes_eval = []
        for cond in by_n[n_train]:
            r = cond["eval"][eval_persona]["pearson_r"]
            train_names = set(PERSONA_NAMES[p - 1] for p in cond["train_personas"])
            is_same = eval_persona in train_names
            rs.append(r)
            includes_eval.append(is_same)

        # Plot individual points, colored by whether eval persona is in training
        for r, is_same in zip(rs, includes_eval):
            marker = "o" if is_same else "x"
            color = "#2ca02c" if is_same else "#d62728"
            ax.plot(n_train, r, marker, color=color, markersize=6, alpha=0.7, zorder=3)

        # Mean line
        mean_r = np.mean(rs)
        ax.plot(n_train, mean_r, "s", color="black", markersize=8, zorder=4)

    # Connect mean points
    means = [np.mean([c["eval"][eval_persona]["pearson_r"] for c in by_n[n]]) for n in [1, 2, 3, 4]]
    ax.plot([1, 2, 3, 4], means, "-", color="black", linewidth=1.5, zorder=2)

    ax.set_title(label, fontsize=12, fontweight="bold")
    ax.set_xlabel("# training personas", fontsize=10)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_ylim(0.70, 0.92)
    ax.axhline(y=means[0], color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("Pearson r on eval set", fontsize=11)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c", markersize=8, label="Eval persona in training"),
    Line2D([0], [0], marker="x", color="#d62728", markersize=8, label="Eval persona held out", markeredgewidth=2),
    Line2D([0], [0], marker="s", color="black", markersize=8, label="Mean across conditions"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.02))

fig.suptitle("Per-Persona Probe Performance vs. Training Diversity", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(ASSETS_DIR / "plot_022526_per_persona_panel.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved per-persona panel plot")
