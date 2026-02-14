"""Generate plots for the single-task steering experiment."""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ANALYSIS_PATH = "results/experiments/single_task_steering/analysis.json"
OUTPUT_DIR = "experiments/steering/revealed_preference/assets"

with open(ANALYSIS_PATH) as f:
    data = json.load(f)


def setup_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "legend.fontsize": 10,
    })


# ---------------------------------------------------------------------------
# Plot 1: Position-Controlled Dose-Response
# ---------------------------------------------------------------------------
def plot_dose_response():
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    conditions = [
        ("boost", "Boost", "#1f77b4", "o"),
        ("suppress", "Suppress", "#ff7f0e", "s"),
        ("differential", "Differential", "#2ca02c", "D"),
    ]

    for cond_key, label, color, marker in conditions:
        cond = data["position_controlled"][cond_key]
        by_coef = cond["by_coefficient"]
        reg = cond["regression"]

        coefs = sorted(by_coef.keys(), key=lambda x: int(x))
        x = np.array([int(c) for c in coefs])
        means = np.array([by_coef[c]["mean"] for c in coefs])
        ses = np.array([by_coef[c]["se"] for c in coefs])

        slope_per_1k = reg["slope"] * 1000
        legend_label = f"{label} (slope={slope_per_1k:.4f}/1k)"

        ax.errorbar(
            x, means, yerr=1.96 * ses,
            fmt=f"-{marker}", color=color, label=legend_label,
            capsize=4, markersize=7, linewidth=1.8,
        )

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlim(-3500, 3500)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Steering Coefficient")
    ax.set_ylabel("P(Pick Steered Task)")
    ax.set_title("Position-Controlled P(Pick Steered Task) vs Coefficient")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    fig.tight_layout()
    fig.savefig(
        f"{OUTPUT_DIR}/plot_021426_position_controlled_dose_response.png", dpi=150
    )
    plt.close(fig)
    print("Saved: plot_021426_position_controlled_dose_response.png")


# ---------------------------------------------------------------------------
# Plot 2: Per-Pair Slopes Histogram
# ---------------------------------------------------------------------------
def plot_per_pair_slopes():
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    slopes = np.array([entry["slope"] for entry in data["per_pair_slopes"]])
    n = len(slopes)
    mean_slope = np.mean(slopes)
    se_slope = np.std(slopes, ddof=1) / np.sqrt(n)
    t_stat = mean_slope / se_slope

    # Scale slopes to per-1000 for readability
    slopes_per_1k = slopes * 1000

    ax.hist(slopes_per_1k, bins=15, color="#1f77b4", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
    ax.axvline(
        mean_slope * 1000, color="red", linestyle="-", linewidth=1.5, alpha=0.8
    )

    annotation = (
        f"Mean slope = {mean_slope * 1000:.4f}/1k\n"
        f"t({n - 1}) = {t_stat:.3f}\n"
        f"n = {n} pairs"
    )
    ax.text(
        0.97, 0.95, annotation,
        transform=ax.transAxes, ha="right", va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9),
    )

    ax.set_xlabel("Per-Pair Slope (per 1k coefficient)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Per-Pair Steering Slopes (Boost)")

    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/plot_021426_per_pair_slopes.png", dpi=150)
    plt.close(fig)
    print("Saved: plot_021426_per_pair_slopes.png")


# ---------------------------------------------------------------------------
# Plot 3: Per-Ordering Comparison
# ---------------------------------------------------------------------------
def plot_per_ordering():
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    po = data["per_ordering"]
    conditions = [
        "boost_a_original",
        "boost_a_swapped",
        "boost_b_original",
        "boost_b_swapped",
    ]
    labels = [
        "Boost A\n(original)",
        "Boost A\n(swapped)",
        "Boost B\n(original)",
        "Boost B\n(swapped)",
    ]

    slopes_raw = [po[c]["slope"] for c in conditions]
    # Negate boost_b slopes so all bars show magnitude in "steered direction"
    slopes_display = [
        slopes_raw[0],     # boost_a_original: positive = correct
        slopes_raw[1],     # boost_a_swapped: positive = correct
        -slopes_raw[2],    # boost_b_original: negate (raw is dP(A)/dcoef, steered is B)
        -slopes_raw[3],    # boost_b_swapped: negate
    ]
    slopes_per_1k = [s * 1000 for s in slopes_display]

    colors = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78"]
    x_pos = np.arange(len(conditions))

    bars = ax.bar(x_pos, slopes_per_1k, color=colors, edgecolor="white", width=0.6)

    for bar, val in zip(bars, slopes_per_1k):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=10,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Slope magnitude (per 1k coef)")
    ax.set_title("Steering Slope by Condition and Ordering")
    ax.axhline(0, color="gray", linewidth=0.8)

    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/plot_021426_per_ordering_comparison.png", dpi=150)
    plt.close(fig)
    print("Saved: plot_021426_per_ordering_comparison.png")


# ---------------------------------------------------------------------------
# Plot 4: Screening P(A) Distribution
# ---------------------------------------------------------------------------
def plot_screening_pa():
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    pa_values = np.array([entry["p_a"] for entry in data["screening"]["p_a_distribution"]])

    ax.hist(pa_values, bins=20, color="#1f77b4", edgecolor="white", alpha=0.85, zorder=3)

    # Shade borderline region [0.2, 0.8]
    ax.axvspan(0.2, 0.8, color="orange", alpha=0.15, zorder=1, label="Borderline region [0.2, 0.8]")

    n_total = len(pa_values)
    n_borderline = np.sum((pa_values >= 0.2) & (pa_values <= 0.8))

    annotation = (
        f"Total entries: {n_total}\n"
        f"Borderline (0.2-0.8): {n_borderline} ({100 * n_borderline / n_total:.1f}%)"
    )
    ax.text(
        0.97, 0.95, annotation,
        transform=ax.transAxes, ha="right", va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9),
    )

    ax.set_xlabel("P(A)")
    ax.set_ylabel("Count")
    ax.set_title("Screening Phase: Distribution of P(A)")
    ax.legend(loc="upper left")
    ax.set_xlim(-0.05, 1.05)

    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/plot_021426_screening_pa_distribution.png", dpi=150)
    plt.close(fig)
    print("Saved: plot_021426_screening_pa_distribution.png")


if __name__ == "__main__":
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_dose_response()
    plot_per_pair_slopes()
    plot_per_ordering()
    plot_screening_pa()
    print("\nAll plots saved to", OUTPUT_DIR)
