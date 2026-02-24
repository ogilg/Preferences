"""
Phase 1 dose-response plots for fine-grained steering experiment.
L31 single-layer, n~575 pairs each.

Generates:
  plot_022426_phase1_dose_response.png
  plot_022426_phase1_diff_ab_detail.png

Output directory:
  /workspace/repo/experiments/steering/replication/fine_grained/assets/
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUTPUT_DIR = "/workspace/repo/experiments/steering/replication/fine_grained/assets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MEAN_NORM_L31 = 52_823.0  # activation-space mean norm at layer 31

def coef_to_pct(coef):
    """Convert activation-space coefficient to % of mean norm."""
    return coef / MEAN_NORM_L31 * 100.0

# ---------------------------------------------------------------------------
# Hardcoded data: (coef_raw, effect_pp, se_pp, t_stat, p_value)
# ---------------------------------------------------------------------------

diff_ab_raw = [
    (-5282, 2.5,  1.1,  2.27,   0.0236),
    (-3962, 1.3,  0.9,  1.43,   0.1541),
    (-2641, -1.6, 0.8, -2.11,   0.0350),
    (-2113, -5.2, 0.9, -6.16,   0.0000),
    (-1585, -7.2, 0.9, -8.33,   0.0000),
    (-1056, -5.1, 0.7, -7.44,   0.0000),
    (-528,  -2.9, 0.5, -6.12,   0.0000),
    ( 528,   4.5, 0.6,  7.59,   0.0000),
    ( 1056,  8.9, 0.9,  9.84,   0.0000),
    ( 1585, 10.9, 1.0, 10.55,   0.0000),
    ( 2113,  8.0, 0.9,  8.55,   0.0000),
    ( 2641,  4.6, 0.9,  5.00,   0.0000),
    ( 3962,  2.4, 1.1,  2.08,   0.0383),
    ( 5282,  3.2, 1.3,  2.56,   0.0106),
]

boost_a_raw = [
    (-5282,  5.0, 0.9,  5.78,   0.0000),
    (-3962,  3.2, 0.7,  4.64,   0.0000),
    (-2641,  0.4, 0.5,  0.82,   0.4120),
    (-2113, -2.7, 0.5, -4.97,   0.0000),
    (-1585, -4.0, 0.6, -7.17,   0.0000),
    (-1056, -3.5, 0.5, -6.58,   0.0000),
    (-528,  -1.6, 0.4, -4.46,   0.0000),
    ( 528,   2.9, 0.4,  6.78,   0.0000),
    ( 1056,  5.1, 0.6,  8.42,   0.0000),
    ( 1585,  5.6, 0.7,  8.16,   0.0000),
    ( 2113,  4.0, 0.7,  6.10,   0.0000),
    ( 2641,  2.4, 0.7,  3.44,   0.0006),
    ( 3962, -1.0, 0.8, -1.21,   0.2263),
    ( 5282, -3.6, 0.9, -4.09,   0.0000),
]

# boost_b effects are reported as P(b) effects
boost_b_raw = [
    (-5282, -9.0, 1.1, -8.21,   0.0000),
    (-3962, -5.2, 1.0, -5.37,   0.0000),
    (-2641, -3.3, 0.8, -3.90,   0.0001),
    (-2113, -4.2, 0.7, -5.71,   0.0000),
    (-1585, -5.3, 0.7, -7.46,   0.0000),
    (-1056, -3.9, 0.6, -6.71,   0.0000),
    (-528,  -1.0, 0.4, -2.43,   0.0154),
    ( 528,   0.6, 0.4,  1.51,   0.1313),
    ( 1056,  1.6, 0.5,  2.96,   0.0032),
    ( 1585,  3.4, 0.7,  4.90,   0.0000),
    ( 2113,  3.2, 0.7,  4.45,   0.0000),
    ( 2641,  2.5, 0.7,  3.41,   0.0007),
    ( 3962,  2.2, 0.7,  2.90,   0.0038),
    ( 5282,  3.0, 0.9,  3.37,   0.0008),
]

def parse_data(raw):
    """Return arrays: x_pct, effect, se, pval."""
    coefs   = np.array([r[0] for r in raw], dtype=float)
    effects = np.array([r[1] for r in raw], dtype=float)
    ses     = np.array([r[2] for r in raw], dtype=float)
    pvals   = np.array([r[4] for r in raw], dtype=float)
    x_pct   = coef_to_pct(coefs)
    return x_pct, effects, ses, pvals

# ---------------------------------------------------------------------------
# Plot 1: three-panel dose-response overview
# ---------------------------------------------------------------------------

def make_overview_plot():
    datasets = [
        ("diff_ab",  diff_ab_raw,  r"$\Delta P(a)$ = P(a|steered) $-$ P(a|control)",     575),
        ("boost_a",  boost_a_raw,  r"$\Delta P(a)$ = P(a|steered) $-$ P(a|control)",     575),
        ("boost_b",  boost_b_raw,  r"$\Delta P(b)$ = P(b|steered) $-$ P(b|control)",     575),
    ]
    condition_labels = ["diff_ab", "boost_a", "boost_b"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.patch.set_facecolor("white")
    fig.suptitle("Phase 1 (L31 single-layer): Fine-grained dose-response",
                 fontsize=14, fontweight="bold", y=1.02)

    for ax, (cond, raw, ylabel_detail, n_pairs), clabel in zip(axes, datasets, condition_labels):
        x, eff, se, pval = parse_data(raw)

        ax.set_facecolor("white")
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8, zorder=1)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8, zorder=1)

        # Minor gridlines at each data point x-position
        for xi in x:
            ax.axvline(xi, color="#cccccc", linewidth=0.5, zorder=0)

        # Error bars (±1 SE)
        sig_mask  = pval < 0.05
        nsig_mask = ~sig_mask

        # Significant points: filled circles
        if sig_mask.any():
            ax.errorbar(x[sig_mask], eff[sig_mask], yerr=se[sig_mask],
                        fmt="o", color="#1f77b4", markerfacecolor="#1f77b4",
                        markeredgecolor="#1f77b4", markersize=6,
                        linewidth=1.2, capsize=3, zorder=3,
                        label="p < 0.05 (filled)")
        # Non-significant points: open circles
        if nsig_mask.any():
            ax.errorbar(x[nsig_mask], eff[nsig_mask], yerr=se[nsig_mask],
                        fmt="o", color="#888888", markerfacecolor="white",
                        markeredgecolor="#888888", markersize=6,
                        linewidth=1.2, capsize=3, zorder=3,
                        label="p \u2265 0.05 (open)")

        ax.set_xlabel("Steering coefficient\n(% of mean activation norm)", fontsize=10)
        ax.set_ylabel("Effect (pp)", fontsize=10)

        # X ticks at the nominal % values
        tick_pcts = [-10, -7.5, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 7.5, 10]
        ax.set_xticks(tick_pcts)
        ax.set_xticklabels([f"{v:g}%" for v in tick_pcts], rotation=45, ha="right", fontsize=7)
        ax.set_xlim(-11, 11)

        # Extra note for boost_b
        note = ""
        if cond == "boost_b":
            note = "\n(+) = P(b) increased [expected dir. for coef > 0]"

        ax.set_title(f"{cond}  (n\u2248{n_pairs} pairs){note}", fontsize=11, pad=6)
        ax.legend(fontsize=7, loc="upper left", framealpha=0.8)
        ax.grid(axis="y", color="#eeeeee", linewidth=0.6, zorder=0)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "plot_022426_phase1_dose_response.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: diff_ab detailed single-panel
# ---------------------------------------------------------------------------

def make_detail_plot():
    x, eff, se, pval = parse_data(diff_ab_raw)
    ci = 1.96 * se  # 95% CI

    # Significance tiers
    mask_high  = pval < 0.001
    mask_mid   = (pval >= 0.001) & (pval < 0.05)
    mask_low   = pval >= 0.05

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.axhline(0, color="black", linestyle="--", linewidth=0.9, zorder=1)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.9, zorder=1)

    # Minor gridlines at each x
    for xi in x:
        ax.axvline(xi, color="#dddddd", linewidth=0.6, zorder=0)
    ax.grid(axis="y", color="#eeeeee", linewidth=0.6, zorder=0)

    # p < 0.001: dark blue filled
    if mask_high.any():
        ax.errorbar(x[mask_high], eff[mask_high], yerr=ci[mask_high],
                    fmt="o", color="#003f8a", markerfacecolor="#003f8a",
                    markeredgecolor="#003f8a", markersize=8,
                    linewidth=1.4, capsize=4, zorder=4,
                    label="p < 0.001")
    # 0.001 <= p < 0.05: medium blue filled
    if mask_mid.any():
        ax.errorbar(x[mask_mid], eff[mask_mid], yerr=ci[mask_mid],
                    fmt="o", color="#4a90d9", markerfacecolor="#4a90d9",
                    markeredgecolor="#4a90d9", markersize=8,
                    linewidth=1.4, capsize=4, zorder=4,
                    label="0.001 \u2264 p < 0.05")
    # p >= 0.05: gray open
    if mask_low.any():
        ax.errorbar(x[mask_low], eff[mask_low], yerr=ci[mask_low],
                    fmt="o", color="#888888", markerfacecolor="white",
                    markeredgecolor="#888888", markersize=8,
                    linewidth=1.4, capsize=4, zorder=4,
                    label="p \u2265 0.05")

    # Annotate peak
    peak_idx = np.argmax(eff)
    peak_x   = x[peak_idx]
    peak_y   = eff[peak_idx]
    ax.annotate(
        f"  Peak: +{peak_y:.1f} pp\n  at {peak_x:+.1f}% norm",
        xy=(peak_x, peak_y),
        xytext=(peak_x + 1.5, peak_y - 1.5),
        fontsize=9,
        color="#003f8a",
        arrowprops=dict(arrowstyle="->", color="#003f8a", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#003f8a", lw=0.8),
        zorder=5,
    )

    # -----------------------------------------------------------------------
    # Summary annotation box (peak, zero-crossing, reversal)
    # -----------------------------------------------------------------------
    # Zero crossing: first positive x where eff changes sign (negative -> positive side)
    # Reversal: on positive side, where effect turns back down significantly
    # On negative side: eff is negative from ~-1% to ~-5%, then reverses sign near -7.5% and -10%
    # Identify zero crossing on positive side (between coef=0 and first sig positive)
    # and on negative side (between -7.5% and -5%)

    summary_lines = [
        "Key features (diff_ab)",
        "",
        f"  Peak:        +{peak_y:.1f} pp  at  {peak_x:+.1f}%",
        f"  Zero cross:  ~+0.5 to +1%  (pos. side)",
        f"               ~-5 to -7.5%  (neg. side, reversal)",
        f"  Neg. peak:   -7.2 pp  at  -3%",
        f"  Reversal:    sign flips at -7.5% (neg. coef)",
        f"  95% CI shown (±1.96 SE)",
    ]
    summary_text = "\n".join(summary_lines)
    ax.text(
        0.02, 0.97, summary_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", fc="#f7f7f7", ec="#aaaaaa", lw=0.8, alpha=0.9),
        zorder=5,
    )

    # Primary x-axis: % of mean norm
    tick_pcts = [-10, -7.5, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 7.5, 10]
    ax.set_xticks(tick_pcts)
    ax.set_xticklabels([f"{v:g}%" for v in tick_pcts], fontsize=9)
    ax.set_xlim(-11, 11)

    # Secondary x-axis showing raw coefficient values
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    raw_ticks_pct  = [-10, -7.5, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 7.5, 10]
    raw_ticks_abs  = [round(p / 100.0 * MEAN_NORM_L31 / 528) * 528 for p in raw_ticks_pct]
    # Use actual nominal values
    nominal_abs = [-5282, -3962, -2641, -2113, -1585, -1056, -528, 0, 528, 1056, 1585, 2113, 2641, 3962, 5282]
    ax2.set_xticks(tick_pcts)
    ax2.set_xticklabels([str(v) for v in nominal_abs], rotation=45, ha="left", fontsize=7)
    ax2.set_xlabel("Steering coefficient (activation-space units)", fontsize=9, labelpad=6)

    ax.set_xlabel("Steering coefficient (% of mean activation norm at L31)", fontsize=10)
    ax.set_ylabel("Effect: P(a|steered) \u2212 P(a|control)  (pp)", fontsize=10)
    ax.set_title("diff_ab dose-response at L31 (n = 575 pairs)", fontsize=13, fontweight="bold", pad=30)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "plot_022426_phase1_diff_ab_detail.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    make_overview_plot()
    make_detail_plot()
    print("Done. All plots written to:", OUTPUT_DIR)
