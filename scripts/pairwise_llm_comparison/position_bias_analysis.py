"""Check if results survive position bias correction.

The judge shows significant position bias (favors A). Check whether
the steered-vs-unsteered effects remain after controlling for this.
"""

from __future__ import annotations

import json

import numpy as np
from scipy import stats


def load_results(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def main():
    results = load_results("scripts/pairwise_llm_comparison/judge_results.json")
    results = [r for r in results if "error" not in r]

    dimensions = ["emotional_engagement", "hedging", "elaboration", "confidence"]

    print("=" * 70)
    print("POSITION BIAS CORRECTION")
    print("=" * 70)
    print()
    print("Split results by whether steered was in position A or B.")
    print("If the effect is real, it should appear in BOTH splits.")
    print("If it's position bias, it should only appear when steered=A.")
    print()

    for dim in dimensions:
        steered_a = [r for r in results if r["steered_is_a"]]
        steered_b = [r for r in results if not r["steered_is_a"]]

        scores_when_a = [r[f"{dim}_score"] for r in steered_a]
        scores_when_b = [r[f"{dim}_score"] for r in steered_b]

        print(f"  {dim}:")
        print(f"    steered=A (n={len(scores_when_a)}): mean={np.mean(scores_when_a):+.3f}")
        print(f"    steered=B (n={len(scores_when_b)}): mean={np.mean(scores_when_b):+.3f}")
        print(f"    Overall: mean={np.mean(scores_when_a + scores_when_b):+.3f}")

        # If position bias exists, when steered=A the bias helps (+), when steered=B it hurts (-)
        # A real effect should push both in the same direction
        # Position bias should push steered=A positive and steered=B negative
        print()

    # Now do the direction asymmetry test separately for steered=A and steered=B
    print("=" * 70)
    print("DIRECTION ASYMMETRY × POSITION")
    print("=" * 70)
    print()
    print("The critical test: does -3000 vs +3000 direction asymmetry persist")
    print("when we control for position?")
    print()

    for dim in dimensions:
        print(f"  {dim}:")
        for position_label, is_a in [("steered=A", True), ("steered=B", False)]:
            subset = [r for r in results if r["steered_is_a"] == is_a]
            pos = [r[f"{dim}_score"] for r in subset if r["steered_coefficient"] == 3000]
            neg = [r[f"{dim}_score"] for r in subset if r["steered_coefficient"] == -3000]
            print(f"    {position_label}: +3000={np.mean(pos):+.3f} (n={len(pos)}), -3000={np.mean(neg):+.3f} (n={len(neg)})")

        # Also test: paired by prompt. For each prompt, compute the score difference
        # between -3000 and +3000 comparisons. This cancels out prompt-level effects
        # and position bias (since position is randomized independently for each comparison).
        print()

    # Most robust test: for each prompt, take the difference in scores
    # between the -3000 comparison and the +3000 comparison.
    # Position bias cancels out because position was randomized per comparison.
    print("=" * 70)
    print("PAIRED BY PROMPT: score(-3000) minus score(+3000)")
    print("=" * 70)
    print()
    print("For each prompt, compute score_toward_steered(-3000) - score_toward_steered(+3000).")
    print("If negative steering produces MORE of the quality than positive steering,")
    print("these differences should be positive.")
    print()

    prompt_ids = sorted(set(r["prompt_id"] for r in results))

    for dim in dimensions:
        diffs = []
        for pid in prompt_ids:
            neg_results = [r for r in results if r["prompt_id"] == pid and r["steered_coefficient"] == -3000]
            pos_results = [r for r in results if r["prompt_id"] == pid and r["steered_coefficient"] == 3000]
            if neg_results and pos_results:
                neg_score = neg_results[0][f"{dim}_score"]
                pos_score = pos_results[0][f"{dim}_score"]
                diffs.append(neg_score - pos_score)

        diffs_arr = np.array(diffs)
        mean_diff = np.mean(diffs_arr)
        n_pos = sum(1 for d in diffs if d > 0)
        n_neg = sum(1 for d in diffs if d < 0)
        n_eq = sum(1 for d in diffs if d == 0)

        # Wilcoxon on differences
        nonzero = [d for d in diffs if d != 0]
        if len(nonzero) > 0:
            stat, p = stats.wilcoxon(nonzero)
        else:
            p = 1.0
        # Sign test
        sign_p = stats.binomtest(n_pos, n_pos + n_neg, 0.5).pvalue if (n_pos + n_neg) > 0 else 1.0

        print(f"  {dim}:")
        print(f"    mean diff = {mean_diff:+.3f}")
        print(f"    +:{n_pos}  =:{n_eq}  -:{n_neg}")
        print(f"    Wilcoxon p = {p:.4f}, sign test p = {sign_p:.4f}")
        print()

    # D+F only
    print("=" * 70)
    print("PAIRED BY PROMPT (D + F only)")
    print("=" * 70)
    print()

    df_prompts = sorted(set(r["prompt_id"] for r in results if r["category"] in ("D_valence", "F_affect")))

    for dim in dimensions:
        diffs = []
        for pid in df_prompts:
            neg_results = [r for r in results if r["prompt_id"] == pid and r["steered_coefficient"] == -3000]
            pos_results = [r for r in results if r["prompt_id"] == pid and r["steered_coefficient"] == 3000]
            if neg_results and pos_results:
                neg_score = neg_results[0][f"{dim}_score"]
                pos_score = pos_results[0][f"{dim}_score"]
                diffs.append(neg_score - pos_score)

        diffs_arr = np.array(diffs)
        mean_diff = np.mean(diffs_arr)
        n_pos = sum(1 for d in diffs if d > 0)
        n_neg = sum(1 for d in diffs if d < 0)
        n_eq = sum(1 for d in diffs if d == 0)

        nonzero = [d for d in diffs if d != 0]
        if len(nonzero) > 0:
            stat, p = stats.wilcoxon(nonzero)
        else:
            p = 1.0
        sign_p = stats.binomtest(n_pos, n_pos + n_neg, 0.5).pvalue if (n_pos + n_neg) > 0 else 1.0

        print(f"  {dim}:")
        print(f"    mean diff = {mean_diff:+.3f}")
        print(f"    +:{n_pos}  =:{n_eq}  -:{n_neg}")
        print(f"    Wilcoxon p = {p:.4f}, sign test p = {sign_p:.4f}")
        print()


if __name__ == "__main__":
    main()
