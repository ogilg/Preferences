"""Analyze pairwise judge results."""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
from scipy import stats


def load_results(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def analyze_results(results: list[dict]) -> None:
    # Filter out errors
    results = [r for r in results if "error" not in r]
    print(f"Analyzing {len(results)} judgments\n")

    dimensions = ["emotional_engagement", "hedging", "elaboration", "confidence"]

    # ============================================================
    # 1. Overall effect: mean score across all prompts
    # ============================================================
    print("=" * 70)
    print("1. OVERALL EFFECT (all prompts pooled)")
    print("=" * 70)
    print("Score > 0 = steered has MORE of this quality; < 0 = unsteered has more")
    print()

    for dim in dimensions:
        scores = [r[f"{dim}_score"] for r in results]
        mean = np.mean(scores)
        # Wilcoxon signed-rank test (exclude zeros)
        nonzero = [s for s in scores if s != 0]
        if len(nonzero) > 0:
            stat, p = stats.wilcoxon(nonzero)
        else:
            stat, p = 0, 1.0
        # Sign test
        n_pos = sum(1 for s in scores if s > 0)
        n_neg = sum(1 for s in scores if s < 0)
        n_eq = sum(1 for s in scores if s == 0)
        sign_p = stats.binomtest(n_pos, n_pos + n_neg, 0.5).pvalue if (n_pos + n_neg) > 0 else 1.0
        print(f"  {dim}:")
        print(f"    mean = {mean:+.3f}, median = {np.median(scores):+.1f}")
        print(f"    +:{n_pos}  =:{n_eq}  -:{n_neg}")
        print(f"    Wilcoxon p = {p:.4f}, sign test p = {sign_p:.4f}")
        print()

    # ============================================================
    # 2. Direction asymmetry: positive vs negative steering
    # ============================================================
    print("=" * 70)
    print("2. DIRECTION ASYMMETRY (positive vs negative steering)")
    print("=" * 70)
    print("If probe encodes a directed dimension, +3000 and -3000 should have")
    print("OPPOSITE effects (opposite signs).")
    print()

    for dim in dimensions:
        pos_scores = [r[f"{dim}_score"] for r in results if r["steered_coefficient"] == 3000]
        neg_scores = [r[f"{dim}_score"] for r in results if r["steered_coefficient"] == -3000]
        pos_mean = np.mean(pos_scores)
        neg_mean = np.mean(neg_scores)
        # Mann-Whitney U test
        if len(pos_scores) > 0 and len(neg_scores) > 0:
            stat, p = stats.mannwhitneyu(pos_scores, neg_scores, alternative="two-sided")
        else:
            stat, p = 0, 1.0
        print(f"  {dim}:")
        print(f"    +3000 mean = {pos_mean:+.3f} (n={len(pos_scores)})")
        print(f"    -3000 mean = {neg_mean:+.3f} (n={len(neg_scores)})")
        print(f"    difference = {pos_mean - neg_mean:+.3f}")
        print(f"    Mann-Whitney p = {p:.4f}")
        print()

    # ============================================================
    # 3. Category breakdown
    # ============================================================
    print("=" * 70)
    print("3. CATEGORY BREAKDOWN")
    print("=" * 70)

    # Group by category
    categories = sorted(set(r["category"] for r in results))
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        print(f"\n  --- {cat} (n={len(cat_results)}) ---")
        for dim in dimensions:
            scores = [r[f"{dim}_score"] for r in cat_results]
            mean = np.mean(scores)
            n_pos = sum(1 for s in scores if s > 0)
            n_neg = sum(1 for s in scores if s < 0)
            n_eq = sum(1 for s in scores if s == 0)
            print(f"    {dim}: mean={mean:+.3f}  (+:{n_pos} =:{n_eq} -:{n_neg})")

    # ============================================================
    # 4. Category × direction breakdown
    # ============================================================
    print("\n" + "=" * 70)
    print("4. CATEGORY × DIRECTION BREAKDOWN")
    print("=" * 70)

    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        print(f"\n  --- {cat} ---")
        for coef in [3000, -3000]:
            coef_results = [r for r in cat_results if r["steered_coefficient"] == coef]
            if not coef_results:
                continue
            print(f"    coef={coef:+d} (n={len(coef_results)}):")
            for dim in dimensions:
                scores = [r[f"{dim}_score"] for r in coef_results]
                mean = np.mean(scores)
                print(f"      {dim}: mean={mean:+.3f}")

    # ============================================================
    # 5. Per-prompt detail for D and F categories
    # ============================================================
    print("\n" + "=" * 70)
    print("5. PER-PROMPT DETAIL (D + F categories)")
    print("=" * 70)

    df_results = [r for r in results if r["category"] in ("D_valence", "F_affect")]
    prompts_in_df = sorted(set(r["prompt_id"] for r in df_results))

    for pid in prompts_in_df:
        prompt_results = [r for r in df_results if r["prompt_id"] == pid]
        print(f"\n  {pid}:")
        for r in sorted(prompt_results, key=lambda x: x["steered_coefficient"]):
            coef = r["steered_coefficient"]
            scores_str = ", ".join(
                f"{dim[:6]}={r[f'{dim}_score']:+d}" for dim in dimensions
            )
            print(f"    coef={coef:+d}: {scores_str}")

    # ============================================================
    # 6. Emotional engagement: D+F only, direction test
    # ============================================================
    print("\n" + "=" * 70)
    print("6. EMOTIONAL ENGAGEMENT — D+F DIRECTION TEST")
    print("=" * 70)

    df_results = [r for r in results if r["category"] in ("D_valence", "F_affect")]
    dim = "emotional_engagement"

    pos_scores = [r[f"{dim}_score"] for r in df_results if r["steered_coefficient"] == 3000]
    neg_scores = [r[f"{dim}_score"] for r in df_results if r["steered_coefficient"] == -3000]

    print(f"\n  D+F prompts only (n_pos={len(pos_scores)}, n_neg={len(neg_scores)}):")
    print(f"    +3000 mean emotional_engagement = {np.mean(pos_scores):+.3f}")
    print(f"    -3000 mean emotional_engagement = {np.mean(neg_scores):+.3f}")
    if len(pos_scores) > 0 and len(neg_scores) > 0:
        stat, p = stats.mannwhitneyu(pos_scores, neg_scores, alternative="two-sided")
        print(f"    Mann-Whitney p = {p:.4f}")

    # Combined D+F sign test for emotional engagement
    all_scores = pos_scores + neg_scores
    n_pos_total = sum(1 for s in all_scores if s > 0)
    n_neg_total = sum(1 for s in all_scores if s < 0)
    print(f"\n  Combined D+F emotional_engagement: +:{n_pos_total} -:{n_neg_total}")
    if n_pos_total + n_neg_total > 0:
        sign_p = stats.binomtest(n_pos_total, n_pos_total + n_neg_total, 0.5).pvalue
        print(f"  Sign test p = {sign_p:.4f}")

    # ============================================================
    # 7. Position bias check
    # ============================================================
    print("\n" + "=" * 70)
    print("7. POSITION BIAS CHECK")
    print("=" * 70)

    for dim in dimensions:
        # Check if the judge systematically favors A or B regardless of which is steered
        raw_choices = []
        for r in results:
            choice = r["judgment"][dim]
            if choice.endswith("_A"):
                raw_choices.append(1)
            elif choice.endswith("_B"):
                raw_choices.append(-1)
            else:
                raw_choices.append(0)
        n_a = sum(1 for c in raw_choices if c > 0)
        n_b = sum(1 for c in raw_choices if c < 0)
        n_eq = sum(1 for c in raw_choices if c == 0)
        if n_a + n_b > 0:
            bias_p = stats.binomtest(n_a, n_a + n_b, 0.5).pvalue
        else:
            bias_p = 1.0
        print(f"  {dim}: A={n_a} eq={n_eq} B={n_b}  bias_p={bias_p:.4f}")


if __name__ == "__main__":
    results = load_results("scripts/pairwise_llm_comparison/judge_results.json")
    analyze_results(results)
