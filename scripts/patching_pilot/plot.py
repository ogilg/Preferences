import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ANALYSIS_PATH = "experiments/patching/pilot/analysis.json"
BASELINE_PATH = "experiments/patching/pilot/baseline_p_choose.json"
RESULTS_PATH = "experiments/patching/pilot/results.json"
ASSETS_DIR = "experiments/patching/pilot/assets"

with open(ANALYSIS_PATH) as f:
    analysis = json.load(f)

with open(BASELINE_PATH) as f:
    baseline_p = json.load(f)

with open(RESULTS_PATH) as f:
    results = json.load(f)

per_pair = analysis["per_pair"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------------------
# Plot 1: P(B) baseline vs |delta_mu| with patching overlays
# ---------------------------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(8, 5))

abs_delta = [abs(p["delta_mu"]) for p in per_pair]
p_b_base = [p["p_b_baseline"] for p in per_pair]
p_b_lts = [p["p_b_last_token_swap"] for p in per_pair]
p_b_ss = [p["p_b_span_swap"] for p in per_pair]

ax1.scatter(abs_delta, p_b_base, c="black", marker="o", s=40, label="baseline", zorder=3)
ax1.scatter(abs_delta, p_b_lts, c="#1f77b4", marker="^", s=40, label="last_token_swap", zorder=3)
ax1.scatter(abs_delta, p_b_ss, c="#d62728", marker="s", s=40, label="span_swap", zorder=3)
ax1.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, zorder=1)

ax1.set_xlabel("|$\\Delta\\mu$| (absolute utility gap)")
ax1.set_ylabel("P(choose B)")
ax1.set_ylim(0, 1.05)
ax1.set_xlim(left=0)
ax1.set_title("P(choose higher-utility task) vs utility gap, by condition")
ax1.legend(loc="lower right", frameon=False)
fig1.tight_layout()
fig1.savefig(f"{ASSETS_DIR}/plot_030626_p_b_vs_delta_mu.png", dpi=150)
plt.close(fig1)
print("Saved plot 1: plot_030626_p_b_vs_delta_mu.png")


# ---------------------------------------------------------------------------
# Plot 2: Shift from baseline (span_swap) vs |delta_mu|
# ---------------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(8, 5))

shifts = [p["p_b_span_swap"] - p["p_b_baseline"] for p in per_pair]

colors = []
for s in shifts:
    if s > 0:
        colors.append("green")
    elif s < 0:
        colors.append("red")
    else:
        colors.append("gray")

ax2.scatter(abs_delta, shifts, c=colors, s=40, zorder=3)
ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8, zorder=1)

for i, p in enumerate(per_pair):
    if abs(shifts[i]) > 0.3:
        # Build short label from task ids
        a_short = p["task_a"].split("_")[0][:4]
        b_short = p["task_b"].split("_")[0][:4]
        label = f"{a_short}-{b_short}"
        ax2.annotate(
            label,
            (abs_delta[i], shifts[i]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=7,
            color="black",
        )

ax2.set_xlabel("|$\\Delta\\mu$| (absolute utility gap)")
ax2.set_ylabel("P(B|span_swap) - P(B|baseline)")
ax2.set_ylim(-1, 1)
ax2.set_xlim(left=0)
ax2.set_title("Span-swap shift from baseline vs utility gap")
fig2.tight_layout()
fig2.savefig(f"{ASSETS_DIR}/plot_030626_span_shift_vs_gap.png", dpi=150)
plt.close(fig2)
print("Saved plot 2: plot_030626_span_shift_vs_gap.png")


# ---------------------------------------------------------------------------
# Plot 3: Position bias breakdown — P(choose A) by condition
# ---------------------------------------------------------------------------
condition_a_counts = {"baseline": 0, "last_token_swap": 0, "span_swap": 0}
condition_totals = {"baseline": 0, "last_token_swap": 0, "span_swap": 0}

for trial in results:
    for cond in ["baseline", "last_token_swap", "span_swap"]:
        choices = trial["conditions"][cond]["choices"]
        condition_a_counts[cond] += sum(1 for c in choices if c == "a")
        condition_totals[cond] += len(choices)

conditions = ["baseline", "last_token_swap", "span_swap"]
p_choose_a = [condition_a_counts[c] / condition_totals[c] for c in conditions]

fig3, ax3 = plt.subplots(figsize=(6, 5))
bar_colors = ["black", "#1f77b4", "#d62728"]
bars = ax3.bar(conditions, p_choose_a, color=bar_colors, width=0.5, zorder=3)
ax3.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, zorder=1)

for bar, val in zip(bars, p_choose_a):
    ax3.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.02,
        f"{val:.3f}",
        ha="center",
        va="bottom",
        fontsize=10,
    )

ax3.set_ylabel("P(choose position A)")
ax3.set_ylim(0, 1)
ax3.set_title("Position bias: P(choose A) across all trials")
ax3.set_xticks(range(len(conditions)))
ax3.set_xticklabels(conditions, rotation=15, ha="right")
fig3.tight_layout()
fig3.savefig(f"{ASSETS_DIR}/plot_030626_position_bias.png", dpi=150)
plt.close(fig3)
print("Saved plot 3: plot_030626_position_bias.png")

print("Done.")
