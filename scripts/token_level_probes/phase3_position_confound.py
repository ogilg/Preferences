"""Phase 3: Position confound analysis for token-level probes.

Tests whether condition effects at critical spans are confounded by position.
Three analyses:
  1. Position of critical span by condition (do conditions differ in position?)
  2. Position-controlled regression (does condition survive after controlling position?)
  3. Divergence point analysis (do paired items diverge only at the critical span?)
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind
import statsmodels.api as sm

DATA_PATH = Path("experiments/token_level_probes/scoring_results.json")
NPZ_PATH = Path("experiments/token_level_probes/all_token_scores.npz")
ASSETS_DIR = Path("experiments/token_level_probes/assets")

PROBE = "task_mean_L39"

DOMAIN_CONTRASTS = {
    "truth": ("true", "false"),
    "harm": ("harmful", "benign"),
    "politics": ("left", "right"),
}


def load_data():
    with open(DATA_PATH) as f:
        results = json.load(f)
    items = results["items"]
    items_by_id = {item["id"]: item for item in items}
    return items, items_by_id


def extract_base_id(item_id: str) -> tuple[str, str]:
    """Extract (base_id, turn) from truth/harm item IDs.

    E.g. 'truth_0_true_user' -> ('truth_0', 'user')
    """
    parts = item_id.split("_")
    return f"{parts[0]}_{parts[1]}", parts[-1]


# ── Analysis 1: Position of critical span by condition ──


def analysis1_position_by_condition(items: list[dict]) -> None:
    print("\n" + "=" * 90)
    print("ANALYSIS 1: Position of critical span by condition")
    print("=" * 90)
    print("(relative position = critical_span_start / total_tokens)")

    for domain in ["truth", "harm", "politics"]:
        cond_a, cond_b = DOMAIN_CONTRASTS[domain]
        domain_items = [it for it in items if it["domain"] == domain]

        # Gather relative positions per condition
        cond_positions: dict[str, list[float]] = defaultdict(list)
        for it in domain_items:
            n_tokens = len(it["tokens"])
            crit_start = min(it["critical_token_indices"])
            rel_pos = crit_start / n_tokens
            cond_positions[it["condition"]].append(rel_pos)

        print(f"\n{'─' * 90}")
        print(f"  {domain.upper()}")
        print(f"{'─' * 90}")
        header = f"  {'Condition':<12} {'N':>5} {'Mean rel pos':>14} {'Std':>10} {'Min':>8} {'Max':>8}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        all_conditions = sorted(cond_positions.keys())
        for cond in all_conditions:
            pos = np.array(cond_positions[cond])
            print(f"  {cond:<12} {len(pos):>5} {np.mean(pos):>14.4f} {np.std(pos):>10.4f} {np.min(pos):>8.4f} {np.max(pos):>8.4f}")

        # T-test between main contrast conditions
        pos_a = np.array(cond_positions[cond_a])
        pos_b = np.array(cond_positions[cond_b])
        t_stat, p_val = ttest_ind(pos_a, pos_b)
        print(f"\n  t-test ({cond_a} vs {cond_b}): t={t_stat:.4f}, p={p_val:.4e}")
        mean_diff = np.mean(pos_a) - np.mean(pos_b)
        print(f"  Mean difference: {mean_diff:.4f}")
        if p_val < 0.05:
            print(f"  ** Significant position difference detected (p < 0.05) **")
        else:
            print(f"  No significant position difference (p >= 0.05)")


# ── Analysis 2: Position-controlled regression ──


def analysis2_position_controlled_regression(items: list[dict]) -> None:
    print("\n" + "=" * 90)
    print("ANALYSIS 2: Position-controlled regression")
    print("  score ~ condition_dummy + relative_position")
    print("=" * 90)

    for domain in ["truth", "harm", "politics"]:
        cond_a, cond_b = DOMAIN_CONTRASTS[domain]
        domain_items = [
            it for it in items
            if it["domain"] == domain and it["condition"] in (cond_a, cond_b)
        ]

        scores = []
        condition_dummies = []
        rel_positions = []

        for it in domain_items:
            score = it["critical_span_mean_scores"][PROBE]
            n_tokens = len(it["tokens"])
            crit_start = min(it["critical_token_indices"])
            rel_pos = crit_start / n_tokens

            scores.append(score)
            condition_dummies.append(1.0 if it["condition"] == cond_a else 0.0)
            rel_positions.append(rel_pos)

        y = np.array(scores)
        X = np.column_stack([
            np.array(condition_dummies),
            np.array(rel_positions),
        ])
        X = sm.add_constant(X)

        model = sm.OLS(y, X).fit()

        print(f"\n{'─' * 90}")
        print(f"  {domain.upper()} (condition dummy: 1={cond_a}, 0={cond_b})")
        print(f"{'─' * 90}")
        print(f"  N = {len(y)}")
        print(f"  R² = {model.rsquared:.4f}")
        print(f"  {'Variable':<20} {'Coef':>10} {'Std Err':>10} {'t':>10} {'p':>12}")
        print(f"  {'-'*62}")

        var_names = ["intercept", f"condition ({cond_a}=1)", "relative_position"]
        for name, coef, se, t, p in zip(
            var_names, model.params, model.bse, model.tvalues, model.pvalues
        ):
            sig = " ***" if p < 0.001 else " **" if p < 0.01 else " *" if p < 0.05 else ""
            print(f"  {name:<20} {coef:>10.4f} {se:>10.4f} {t:>10.4f} {p:>12.4e}{sig}")

        # Also run condition-only model for comparison
        X_cond_only = sm.add_constant(np.array(condition_dummies))
        model_cond = sm.OLS(y, X_cond_only).fit()
        print(f"\n  Condition-only model: R² = {model_cond.rsquared:.4f}")
        cond_coef = model_cond.params[1]
        cond_p = model_cond.pvalues[1]
        print(f"    condition coef = {cond_coef:.4f}, p = {cond_p:.4e}")
        print(f"  Full model (+ position): R² = {model.rsquared:.4f}")
        print(f"    condition coef = {model.params[1]:.4f}, p = {model.pvalues[1]:.4e}")
        coef_change = ((model.params[1] - cond_coef) / abs(cond_coef)) * 100 if cond_coef != 0 else float("nan")
        print(f"    Condition coef change with position control: {coef_change:+.1f}%")


# ── Analysis 3: Divergence point analysis ──


def analysis3_divergence_curve(items: list[dict], items_by_id: dict) -> None:
    print("\n" + "=" * 90)
    print("ANALYSIS 3: Divergence point analysis (truth domain, user turn)")
    print("  Comparing true vs false pairs at shared prefix, critical span, and suffix")
    print("=" * 90)

    scores_npz = np.load(NPZ_PATH)

    # Focus on truth domain, user turn, true/false pairs
    truth_user_items = [
        it for it in items
        if it["domain"] == "truth" and it["turn"] == "user"
        and it["condition"] in ("true", "false")
    ]

    # Group by (base_id, turn) to find pairs
    grouped = defaultdict(dict)
    for it in truth_user_items:
        base_id, turn = extract_base_id(it["id"])
        grouped[(base_id, turn)][it["condition"]] = it

    # Collect per-relative-position absolute differences
    # We define three regions: before, at, and after the critical span
    all_before_diffs = []
    all_critical_diffs = []
    all_after_diffs = []

    # For the divergence curve: collect |diff| at each token position relative to critical span
    # Normalize so critical_span_start = 0
    position_diffs: dict[int, list[float]] = defaultdict(list)

    n_pairs = 0
    n_skipped_prefix_mismatch = 0

    for key, cond_map in grouped.items():
        if "true" not in cond_map or "false" not in cond_map:
            continue

        true_item = cond_map["true"]
        false_item = cond_map["false"]

        true_tokens = true_item["tokens"]
        false_tokens = false_item["tokens"]

        true_crit = true_item["critical_token_indices"]
        false_crit = false_item["critical_token_indices"]

        # Critical spans should start at the same position
        true_crit_start = min(true_crit)
        false_crit_start = min(false_crit)

        if true_crit_start != false_crit_start:
            n_skipped_prefix_mismatch += 1
            continue

        crit_start = true_crit_start

        # Verify shared prefix
        prefix_match = all(
            true_tokens[i] == false_tokens[i]
            for i in range(min(crit_start, len(true_tokens), len(false_tokens)))
        )
        if not prefix_match:
            n_skipped_prefix_mismatch += 1
            continue

        # Load all-token scores
        true_key = f"{true_item['id']}__{PROBE}"
        false_key = f"{false_item['id']}__{PROBE}"

        true_scores = scores_npz[true_key]
        false_scores = scores_npz[false_key]

        # Truncate to token count
        true_scores = true_scores[:len(true_tokens)]
        false_scores = false_scores[:len(false_tokens)]

        # Min length for comparison
        min_len = min(len(true_scores), len(false_scores))

        # Compute |diff| at each position, relative to critical span start
        for pos in range(min_len):
            rel_pos = pos - crit_start
            diff = abs(float(true_scores[pos]) - float(false_scores[pos]))
            position_diffs[rel_pos].append(diff)

            true_crit_end = max(true_crit)
            if pos < crit_start:
                all_before_diffs.append(diff)
            elif pos <= true_crit_end:
                all_critical_diffs.append(diff)
            else:
                all_after_diffs.append(diff)

        n_pairs += 1

    print(f"\n  Pairs found: {n_pairs}")
    print(f"  Skipped (prefix mismatch or different crit start): {n_skipped_prefix_mismatch}")

    # Print region statistics
    print(f"\n  Mean |score difference| by region:")
    print(f"  {'Region':<25} {'N tokens':>10} {'Mean |diff|':>12} {'Std':>10} {'Median':>10}")
    print(f"  {'-'*67}")

    for label, diffs in [
        ("Before critical span", all_before_diffs),
        ("At critical span", all_critical_diffs),
        ("After critical span", all_after_diffs),
    ]:
        arr = np.array(diffs) if diffs else np.array([0.0])
        print(
            f"  {label:<25} {len(diffs):>10} {np.mean(arr):>12.4f} "
            f"{np.std(arr):>10.4f} {np.median(arr):>10.4f}"
        )

    # T-tests between regions
    if all_before_diffs and all_critical_diffs:
        t, p = ttest_ind(all_before_diffs, all_critical_diffs)
        print(f"\n  t-test (before vs critical): t={t:.4f}, p={p:.4e}")
    if all_before_diffs and all_after_diffs:
        t, p = ttest_ind(all_before_diffs, all_after_diffs)
        print(f"  t-test (before vs after):    t={t:.4f}, p={p:.4e}")

    # Build divergence curve
    rel_positions_sorted = sorted(position_diffs.keys())
    mean_diffs = [np.mean(position_diffs[rp]) for rp in rel_positions_sorted]
    sem_diffs = [
        np.std(position_diffs[rp]) / np.sqrt(len(position_diffs[rp]))
        for rp in rel_positions_sorted
    ]
    counts = [len(position_diffs[rp]) for rp in rel_positions_sorted]

    # Filter to positions with enough data (at least 10 pairs)
    min_count = 10
    mask = [c >= min_count for c in counts]
    plot_positions = [rp for rp, m in zip(rel_positions_sorted, mask) if m]
    plot_means = [md for md, m in zip(mean_diffs, mask) if m]
    plot_sems = [se for se, m in zip(sem_diffs, mask) if m]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.fill_between(
        plot_positions,
        [m - s for m, s in zip(plot_means, plot_sems)],
        [m + s for m, s in zip(plot_means, plot_sems)],
        alpha=0.3,
        color="#1f77b4",
    )
    ax.plot(plot_positions, plot_means, "-o", markersize=3, color="#1f77b4", linewidth=1.5)

    # Mark the critical span region
    # Find typical critical span length
    crit_lengths = []
    for key, cond_map in grouped.items():
        if "true" in cond_map:
            crit_indices = cond_map["true"]["critical_token_indices"]
            crit_lengths.append(max(crit_indices) - min(crit_indices) + 1)

    if crit_lengths:
        median_crit_len = int(np.median(crit_lengths))
    else:
        median_crit_len = 2

    ax.axvspan(-0.5, median_crit_len - 0.5, alpha=0.15, color="red", label="Critical span (median extent)")
    ax.axvline(0, color="red", linewidth=1, linestyle="--", alpha=0.7)

    ax.set_xlabel("Token position relative to critical span start", fontsize=11)
    ax.set_ylabel("Mean |score difference| (true vs false)", fontsize=11)
    ax.set_title(
        f"Divergence curve: true vs false pairs (truth domain, user turn)\n"
        f"Probe: {PROBE}, N={n_pairs} pairs",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=9)

    # Set y-axis to start at 0
    ax.set_ylim(0, None)

    plt.tight_layout()

    out_path = ASSETS_DIR / "plot_031426_divergence_curve.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {out_path}")


# ── Main ──


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    items, items_by_id = load_data()
    print(f"Loaded {len(items)} items")

    analysis1_position_by_condition(items)
    analysis2_position_controlled_regression(items)
    analysis3_divergence_curve(items, items_by_id)

    print("\n" + "=" * 90)
    print("Phase 3 complete.")
    print("=" * 90)


if __name__ == "__main__":
    main()
