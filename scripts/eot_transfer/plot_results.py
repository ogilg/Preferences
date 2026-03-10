import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

ASSETS_DIR = "/workspace/repo/experiments/patching/eot_transfer/assets"


def binomial_ci(p, n, z=1.96):
    se = np.sqrt(p * (1 - p) / n)
    return z * se


# ---------------------------------------------------------------------------
# Plot 1: Flip rate by condition
# ---------------------------------------------------------------------------
def plot_flip_rate_by_condition():
    conditions = ["Control", "Swap headers", "Swap both", "Swap target\n(same topic)", "Swap target\n(cross topic)"]
    flip_rates = [83.6, 75.1, 30.6, 29.2, 12.1]
    ns = [195, 189, 395, 391, 356]
    colors = ["#4878CF", "#B0D0F0", "#E8892E", "#59A84B", "#D44D4D"]

    cis = [binomial_ci(r / 100, n) * 100 for r, n in zip(flip_rates, ns)]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(conditions))
    bars = ax.bar(x, flip_rates, color=colors, edgecolor="black", linewidth=0.5,
                  yerr=cis, capsize=5, error_kw={"linewidth": 1.2})

    ax.axhline(83.6, color="#4878CF", linestyle="--", linewidth=1.2, alpha=0.7,
               label="Control rate (83.6%)")

    ax.set_ylim(0, 100)
    ax.set_ylabel("Flip Rate (%)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=11)
    ax.set_title("EOT Patching Flip Rate by Transfer Condition", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)

    for i, (rate, n, ci) in enumerate(zip(flip_rates, ns, cis)):
        ax.text(i, rate + ci + 1.5, f"n={n}", ha="center", va="bottom", fontsize=9.5, color="gray")

    ax.set_xlim(-0.6, len(conditions) - 0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{ASSETS_DIR}/plot_030726_flip_rate_by_condition.png", dpi=200)
    plt.close(fig)
    print("Saved plot_030726_flip_rate_by_condition.png")


# ---------------------------------------------------------------------------
# Plot 2: Direction breakdown
# ---------------------------------------------------------------------------
def plot_direction_breakdown():
    groups = ["Swap both", "Swap target\n(same topic)", "Swap target\n(cross topic)"]
    dir_a_labels = ["cd", "orig", "orig"]
    dir_b_labels = ["dc", "swap", "swap"]
    dir_a_rates = [30.8, 41.8, 19.1]
    dir_b_rates = [30.5, 16.8, 5.1]
    dir_a_ns = [198, 194, 178]
    dir_b_ns = [197, 197, 178]

    dir_a_cis = [binomial_ci(r / 100, n) * 100 for r, n in zip(dir_a_rates, dir_a_ns)]
    dir_b_cis = [binomial_ci(r / 100, n) * 100 for r, n in zip(dir_b_rates, dir_b_ns)]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(groups))
    width = 0.35

    bars_a = ax.bar(x - width / 2, dir_a_rates, width, color="#5A9BD5", edgecolor="black",
                    linewidth=0.5, yerr=dir_a_cis, capsize=4, error_kw={"linewidth": 1.2})
    bars_b = ax.bar(x + width / 2, dir_b_rates, width, color="#ED7D31", edgecolor="black",
                    linewidth=0.5, yerr=dir_b_cis, capsize=4, error_kw={"linewidth": 1.2})

    # Add value labels on bars
    for bar, rate, n, label in zip(bars_a, dir_a_rates, dir_a_ns, dir_a_labels):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + dir_a_cis[0] + 1.5,
                f"{rate}%\n(n={n})", ha="center", va="bottom", fontsize=9)
    for bar, rate, n, label in zip(bars_b, dir_b_rates, dir_b_ns, dir_b_labels):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + dir_b_cis[0] + 1.5,
                f"{rate}%\n(n={n})", ha="center", va="bottom", fontsize=9)

    # Legend with direction labels
    ax.legend([bars_a, bars_b],
              [f"Direction A (cd / orig)", f"Direction B (dc / swap)"],
              loc="upper right", fontsize=10)

    ax.set_ylim(0, 100)
    ax.set_ylabel("Flip Rate (%)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=11)
    ax.set_title("Flip Rate by Recipient Ordering Direction", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{ASSETS_DIR}/plot_030726_direction_breakdown.png", dpi=200)
    plt.close(fig)
    print("Saved plot_030726_direction_breakdown.png")


# ---------------------------------------------------------------------------
# Plot 3: Harmful vs Benign interaction
# ---------------------------------------------------------------------------
def plot_harmful_interaction():
    conditions = ["Control", "Swap both", "Swap target\n(cross topic)"]
    harmful_rates = [81.3, 38.0, 19.7]
    benign_rates = [86.1, 23.1, 6.5]
    harmful_ns = [91, 187, 132]
    benign_ns = [101, 199, 185]

    harmful_cis = [binomial_ci(r / 100, n) * 100 for r, n in zip(harmful_rates, harmful_ns)]
    benign_cis = [binomial_ci(r / 100, n) * 100 for r, n in zip(benign_rates, benign_ns)]

    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(conditions))
    width = 0.35

    bars_h = ax.bar(x - width / 2, harmful_rates, width, color="#D44D4D", edgecolor="black",
                    linewidth=0.5, label="Harmful", yerr=harmful_cis, capsize=4,
                    error_kw={"linewidth": 1.2})
    bars_b = ax.bar(x + width / 2, benign_rates, width, color="#4878CF", edgecolor="black",
                    linewidth=0.5, label="Benign", yerr=benign_cis, capsize=4,
                    error_kw={"linewidth": 1.2})

    # Add value labels on bars
    for i, (bar, rate, n) in enumerate(zip(bars_h, harmful_rates, harmful_ns)):
        ci = harmful_cis[i]
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ci + 1.5,
                f"{rate}%\n(n={n})", ha="center", va="bottom", fontsize=9)
    for i, (bar, rate, n) in enumerate(zip(bars_b, benign_rates, benign_ns)):
        ci = benign_cis[i]
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ci + 1.5,
                f"{rate}%\n(n={n})", ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0, 100)
    ax.set_ylabel("Flip Rate (%)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=11)
    ax.set_title("Flip Rate: Harmful vs Benign Source Orderings", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{ASSETS_DIR}/plot_030726_harmful_interaction.png", dpi=200)
    plt.close(fig)
    print("Saved plot_030726_harmful_interaction.png")


if __name__ == "__main__":
    plot_flip_rate_by_condition()
    plot_direction_breakdown()
    plot_harmful_interaction()
    print("All plots saved.")
