"""Analyze multi-turn pairwise validation results (v2 format with resampling).

Per-prefill: correlation with Thurstonian scores, ordering bias (P(choose A-position)),
and refusal rate.

Usage:
    python -m scripts.multi_turn_pairwise.analyze
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

RESULTS_DIR = Path("experiments/steering/multi_turn_pairwise/results")
ASSETS_DIR = Path("experiments/steering/multi_turn_pairwise/assets")


def load_data():
    with open(RESULTS_DIR / "validation_results_v2.json") as f:
        raw = json.load(f)
    with open(RESULTS_DIR / "thurstonian_scores.json") as f:
        scores = json.load(f)
    return raw, scores


def compute_win_rates(results: list[dict]) -> dict[str, float]:
    """Compute per-task win rate from pairwise results (using canonical task_i/task_j)."""
    wins: dict[str, int] = defaultdict(int)
    total: dict[str, int] = defaultdict(int)

    for r in results:
        if r["choice"] == "refusal":
            continue
        i_id, j_id = r["task_i_id"], r["task_j_id"]
        total[i_id] += 1
        total[j_id] += 1
        chose_a = r["choice"] == "a"
        chose_i = chose_a if r["i_is_a"] else not chose_a
        if chose_i:
            wins[i_id] += 1
        else:
            wins[j_id] += 1

    return {tid: wins[tid] / total[tid] for tid in total if total[tid] > 0}


def compute_position_bias(results: list[dict]) -> dict:
    """Compute P(choose A-position task) — measures position bias.

    0.5 = no bias. >0.5 = prefers whatever is in position A (first turn).
    """
    chose_a = 0
    total = 0
    for r in results:
        if r["choice"] == "refusal":
            continue
        total += 1
        if r["choice"] == "a":
            chose_a += 1

    p_a = chose_a / total if total > 0 else float("nan")
    # Binomial test against 0.5
    if total > 0:
        binom_p = stats.binomtest(chose_a, total, 0.5).pvalue
    else:
        binom_p = float("nan")

    return {"p_choose_a": p_a, "n": total, "binom_p": binom_p}


def compute_pairwise_agreement(results: list[dict]) -> dict:
    """For pairs with multiple resamples in different orderings, compute agreement.

    Groups by canonical pair (task_i, task_j), then for pairs that have both
    i_is_a=True and i_is_a=False resamples, compute how often they agree on
    which task is preferred.
    """
    by_pair: dict[tuple[str, str], dict[str, list]] = defaultdict(lambda: {"same": [], "flipped": []})
    for r in results:
        if r["choice"] == "refusal":
            continue
        key = (r["task_i_id"], r["task_j_id"])
        chose_a = r["choice"] == "a"
        chose_i = chose_a if r["i_is_a"] else not chose_a
        bucket = "same" if r["i_is_a"] else "flipped"
        by_pair[key][bucket].append(chose_i)

    agreements = 0
    comparisons = 0
    for key, buckets in by_pair.items():
        if not buckets["same"] or not buckets["flipped"]:
            continue
        # Compare each same-order choice with each flipped-order choice
        for s in buckets["same"]:
            for f in buckets["flipped"]:
                if s == f:
                    agreements += 1
                comparisons += 1

    return {
        "agreement": agreements / comparisons if comparisons > 0 else float("nan"),
        "n_comparisons": comparisons,
    }


def analyze():
    raw, scores = load_data()

    print(f"{'Prefill':<35} {'r':<8} {'P(A)':<8} {'binom p':<10} {'Agree':<8} {'Refusal%':<10}")
    print("-" * 82)

    summary = []
    for prefill_label in sorted(raw.keys()):
        data = raw[prefill_label]
        results = data["results"]

        # Win rate correlation with Thurstonian
        win_rates = compute_win_rates(results)
        common_ids = sorted(set(win_rates.keys()) & set(scores.keys()))
        if len(common_ids) >= 5:
            wr = np.array([win_rates[tid] for tid in common_ids])
            sc = np.array([scores[tid] for tid in common_ids])
            r_val, r_p = stats.pearsonr(wr, sc)
        else:
            r_val = float("nan")

        # Position bias
        pos = compute_position_bias(results)

        # Pairwise agreement across orderings
        agree = compute_pairwise_agreement(results)

        # Refusal rate
        n_refusals = sum(1 for r in results if r["choice"] == "refusal")
        refusal_pct = 100 * n_refusals / len(results) if results else 0

        print(
            f"{prefill_label:<35} {r_val:<8.3f} {pos['p_choose_a']:<8.3f} "
            f"{pos['binom_p']:<10.2e} {agree['agreement']:<8.3f} {refusal_pct:<10.1f}"
        )
        summary.append({
            "prefill": prefill_label,
            "r": r_val,
            "p_choose_a": pos["p_choose_a"],
            "position_bias_p": pos["binom_p"],
            "n_choices": pos["n"],
            "agreement": agree["agreement"],
            "n_agreement_comparisons": agree["n_comparisons"],
            "refusal_pct": refusal_pct,
        })

    with open(RESULTS_DIR / "analysis_summary_v2.json", "w") as f:
        json.dump(summary, f, indent=2)

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    plot_summary(summary)
    plot_scatter(raw, scores, summary)


def plot_summary(summary):
    prefills = [s["prefill"] for s in summary]
    x = np.arange(len(prefills))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Correlation
    axes[0].bar(x, [s["r"] for s in summary])
    axes[0].set_ylabel("Pearson r with Thurstonian")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Score correlation")

    # Position bias
    axes[1].bar(x, [s["p_choose_a"] for s in summary])
    axes[1].axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    axes[1].set_ylabel("P(choose A-position)")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Position bias")

    # Agreement
    axes[2].bar(x, [s["agreement"] for s in summary])
    axes[2].axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    axes[2].set_ylabel("Cross-order agreement")
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Ordering agreement")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(prefills, rotation=20, ha="right", fontsize=8)

    fig.suptitle("Multi-turn pairwise validation", fontsize=12)
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "plot_031326_summary.png", dpi=150)
    plt.close(fig)
    print("Saved summary plot")


def plot_scatter(raw, scores, summary):
    best = max(summary, key=lambda s: s["r"])
    best_label = best["prefill"]
    results = raw[best_label]["results"]

    win_rates = compute_win_rates(results)
    common_ids = sorted(set(win_rates.keys()) & set(scores.keys()))
    wr = np.array([win_rates[tid] for tid in common_ids])
    sc = np.array([scores[tid] for tid in common_ids])
    r_val, p_val = stats.pearsonr(wr, sc)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(sc, wr, alpha=0.5, s=15)
    ax.set_xlabel("Thurstonian score (single-turn)")
    ax.set_ylabel("Win rate (multi-turn)")
    ax.set_title(f"Prefill: '{best_label}'\nr = {r_val:.3f}, p = {p_val:.2e}")

    z = np.polyfit(sc, wr, 1)
    x_line = np.linspace(sc.min(), sc.max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), "r--", alpha=0.5)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "plot_031326_scatter_best_prefill.png", dpi=150)
    plt.close(fig)
    print("Saved scatter plot")


if __name__ == "__main__":
    analyze()
