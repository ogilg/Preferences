#!/usr/bin/env python3
"""Plot baseline P(a) quartile vs diff_ab steerability at +2641."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXP_DIR = REPO_ROOT / "experiments" / "steering" / "replication"
RESULTS_DIR = EXP_DIR / "results"
ASSETS_DIR = EXP_DIR / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

PHASE1_PATH = RESULTS_DIR / "steering_phase1.json"
OUTPUT_PATH = ASSETS_DIR / "plot_022226_phase1_ctrl_pa_vs_effect.png"

TARGET_COEF = 2641.1419921875004
COEF_TOL = 10
N_BOOT = 2000
N_QUARTILES = 4


def p_a(responses: list[str]) -> float:
    valid = [r for r in responses if r in ("a", "b")]
    if not valid:
        return 0.5
    return sum(1 for r in valid if r == "a") / len(valid)


def bootstrap_ci(values: list[float], n_boot: int = N_BOOT, ci: float = 0.95) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    arr = np.array(values)
    rng = np.random.default_rng(42)
    boot_means = [
        float(np.mean(arr[rng.integers(0, len(arr), len(arr))]))
        for _ in range(n_boot)
    ]
    alpha = (1 - ci) / 2
    return (float(np.percentile(boot_means, 100 * alpha)), float(np.percentile(boot_means, 100 * (1 - alpha))))


def main() -> None:
    with open(PHASE1_PATH) as f:
        data = json.load(f)

    results = data["results"]

    # Collect per-pair control P(a): pool both orderings
    # key: pair_id -> list of P(a) estimates (one per ordering/row)
    pair_ctrl_responses: dict[str, list[str]] = {}
    # key: pair_id -> list of diff_ab P(a) at +2641 (one per ordering/row)
    pair_diffab_responses: dict[str, list[str]] = {}

    for row in results:
        pair_id = row["pair_id"]
        condition = row["condition"]
        coef = row["coefficient"]
        responses = row["responses"]

        if condition == "control":
            pair_ctrl_responses.setdefault(pair_id, []).extend(responses)
        elif condition == "diff_ab" and abs(coef - TARGET_COEF) < COEF_TOL:
            pair_diffab_responses.setdefault(pair_id, []).extend(responses)

    # Compute per-pair control P(a) and effect
    pair_ids = sorted(set(pair_ctrl_responses) & set(pair_diffab_responses))
    print(f"Pairs with both control and diff_ab @ +2641: {len(pair_ids)}")

    ctrl_pa: dict[str, float] = {}
    effect_pa: dict[str, float] = {}

    for pid in pair_ids:
        ctrl_pa[pid] = p_a(pair_ctrl_responses[pid])
        diffab_pa = p_a(pair_diffab_responses[pid])
        effect_pa[pid] = (diffab_pa - ctrl_pa[pid]) * 100  # in percentage points

    # Sort pairs by control P(a) and divide into quartiles
    sorted_pairs = sorted(pair_ids, key=lambda pid: ctrl_pa[pid])
    n = len(sorted_pairs)
    quartile_size = n // N_QUARTILES

    quartile_pairs: list[list[str]] = []
    for q in range(N_QUARTILES):
        start = q * quartile_size
        # Last quartile gets any remainder
        end = (q + 1) * quartile_size if q < N_QUARTILES - 1 else n
        quartile_pairs.append(sorted_pairs[start:end])

    # Compute quartile stats
    quartile_ctrl_ranges: list[tuple[float, float]] = []
    quartile_effects: list[list[float]] = []

    for qpairs in quartile_pairs:
        ctrls = [ctrl_pa[pid] for pid in qpairs]
        effects = [effect_pa[pid] for pid in qpairs]
        quartile_ctrl_ranges.append((min(ctrls), max(ctrls)))
        quartile_effects.append(effects)

    means = [float(np.mean(e)) for e in quartile_effects]
    cis = [bootstrap_ci(e) for e in quartile_effects]
    ns = [len(e) for e in quartile_effects]

    print("\nQuartile summary:")
    for i, (qpairs, rng, mean, ci, n_q) in enumerate(
        zip(quartile_pairs, quartile_ctrl_ranges, means, cis, ns)
    ):
        print(f"  Q{i+1}: ctrl P(a)=[{rng[0]:.2f},{rng[1]:.2f}], n={n_q}, "
              f"mean effect={mean:+.1f}pp, 95% CI=[{ci[0]:+.1f}, {ci[1]:+.1f}]")

    # Build x-axis labels with quartile ranges
    x_labels = []
    for i, (rng_lo, rng_hi) in enumerate(quartile_ctrl_ranges):
        x_labels.append(f"Q{i+1} ({rng_lo:.2f}–{rng_hi:.2f})")

    # Colors: blue for Q1-Q2 (lower preference), red for Q3-Q4 (higher preference)
    colors = ["#4878CF", "#6baed6", "#ef6548", "#a50f15"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(N_QUARTILES)
    bar_width = 0.6

    bars = ax.bar(x, means, width=bar_width, color=colors, alpha=0.85, zorder=3)

    # Error bars
    yerr_low = [mean - ci[0] for mean, ci in zip(means, cis)]
    yerr_high = [ci[1] - mean for mean, ci in zip(means, cis)]
    ax.errorbar(
        x, means,
        yerr=[yerr_low, yerr_high],
        fmt="none",
        color="black",
        capsize=5,
        linewidth=1.5,
        zorder=4,
    )

    # Horizontal dashed line at y=0
    ax.axhline(0, color="black", linestyle="--", linewidth=1.0, zorder=2)

    # Text annotations above each bar
    for i, (mean, n_q) in enumerate(zip(means, ns)):
        offset = max(yerr_high[i] + 0.5, abs(mean) * 0.05 + 1.0)
        y_text = mean + (offset if mean >= 0 else -(offset + 2))
        va = "bottom" if mean >= 0 else "top"
        ax.text(
            x[i], mean + (offset if mean >= 0 else -offset),
            f"{mean:+.1f}pp\n(n={n_q})",
            ha="center", va=va, fontsize=9.5, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_xlabel("Control P(a) quartile", fontsize=11)
    ax.set_ylabel("diff_ab effect at +2641 (pp)", fontsize=11)
    ax.set_ylim(-10, 25)
    ax.set_title("Baseline Preference Predicts Steerability (diff_ab, coef=+2641)", fontsize=12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    print(f"\nSaved plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
