"""Extended analysis: dose-response, position-swapped replication, and combined."""

from __future__ import annotations

import json

import numpy as np
from scipy import stats


def load_results(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return [r for r in data if "error" not in r]


def paired_direction_test(results: list[dict], dim: str) -> dict:
    """Paired by prompt: score(-coef) - score(+coef). Position-bias-immune."""
    prompt_ids = sorted(set(r["prompt_id"] for r in results))
    coefs = sorted(set(abs(r["steered_coefficient"]) for r in results if r["steered_coefficient"] != 0))

    # Find the coefficient magnitude
    if len(coefs) == 1:
        mag = coefs[0]
    else:
        mag = max(coefs)

    diffs = []
    for pid in prompt_ids:
        neg = [r for r in results if r["prompt_id"] == pid and r["steered_coefficient"] == -mag]
        pos = [r for r in results if r["prompt_id"] == pid and r["steered_coefficient"] == mag]
        if neg and pos:
            diffs.append(neg[0][f"{dim}_score"] - pos[0][f"{dim}_score"])

    diffs_arr = np.array(diffs)
    n_pos = sum(1 for d in diffs if d > 0)
    n_neg = sum(1 for d in diffs if d < 0)
    n_eq = sum(1 for d in diffs if d == 0)

    nonzero = [d for d in diffs if d != 0]
    w_p = stats.wilcoxon(nonzero).pvalue if len(nonzero) > 0 else 1.0
    s_p = stats.binomtest(n_pos, n_pos + n_neg, 0.5).pvalue if (n_pos + n_neg) > 0 else 1.0

    return {
        "mean_diff": float(np.mean(diffs_arr)),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_eq": n_eq,
        "wilcoxon_p": w_p,
        "sign_p": s_p,
    }


def main():
    original = load_results("scripts/pairwise_llm_comparison/judge_results.json")
    swapped = load_results("scripts/pairwise_llm_comparison/judge_results_swapped.json")
    dose_1000 = load_results("scripts/pairwise_llm_comparison/judge_results_1000.json")

    dimensions = ["emotional_engagement", "hedging", "elaboration", "confidence"]

    # ============================================================
    # 1. Position-swapped replication
    # ============================================================
    print("=" * 70)
    print("1. POSITION-SWAPPED REPLICATION (±3000)")
    print("=" * 70)
    print()
    print("Original vs swapped. If effect is real, paired direction test")
    print("should give same direction and similar significance.")
    print()

    print(f"{'Dimension':<25} {'Original':>30} {'Swapped':>30}")
    print(f"{'':25} {'mean(sign_p)':>30} {'mean(sign_p)':>30}")
    print("-" * 85)

    for dim in dimensions:
        orig_res = paired_direction_test(original, dim)
        swap_res = paired_direction_test(swapped, dim)
        print(f"  {dim:<23} {orig_res['mean_diff']:+.3f} (p={orig_res['sign_p']:.4f})         {swap_res['mean_diff']:+.3f} (p={swap_res['sign_p']:.4f})")

    # ============================================================
    # 2. Combined original + swapped (double the data)
    # ============================================================
    print()
    print("=" * 70)
    print("2. COMBINED ORIGINAL + SWAPPED (doubled data, ±3000)")
    print("=" * 70)
    print()

    # For each prompt × direction, average the original and swapped scores
    prompt_ids = sorted(set(r["prompt_id"] for r in original))

    for dim in dimensions:
        diffs = []
        for pid in prompt_ids:
            scores_neg = []
            scores_pos = []
            for dataset in [original, swapped]:
                neg = [r for r in dataset if r["prompt_id"] == pid and r["steered_coefficient"] == -3000]
                pos = [r for r in dataset if r["prompt_id"] == pid and r["steered_coefficient"] == 3000]
                if neg:
                    scores_neg.append(neg[0][f"{dim}_score"])
                if pos:
                    scores_pos.append(pos[0][f"{dim}_score"])
            if scores_neg and scores_pos:
                avg_neg = np.mean(scores_neg)
                avg_pos = np.mean(scores_pos)
                diffs.append(avg_neg - avg_pos)

        diffs_arr = np.array(diffs)
        n_pos = sum(1 for d in diffs if d > 0)
        n_neg = sum(1 for d in diffs if d < 0)
        n_eq = sum(1 for d in diffs if d == 0)

        nonzero = [d for d in diffs if d != 0]
        w_p = stats.wilcoxon(nonzero).pvalue if len(nonzero) > 0 else 1.0
        s_p = stats.binomtest(n_pos, n_pos + n_neg, 0.5).pvalue if (n_pos + n_neg) > 0 else 1.0

        print(f"  {dim}:")
        print(f"    mean diff = {np.mean(diffs_arr):+.3f}")
        print(f"    +:{n_pos}  =:{n_eq}  -:{n_neg}")
        print(f"    Wilcoxon p = {w_p:.4f}, sign test p = {s_p:.4f}")
        print()

    # ============================================================
    # 3. Dose-response: ±1000 vs ±3000
    # ============================================================
    print("=" * 70)
    print("3. DOSE-RESPONSE: ±1000 vs ±3000")
    print("=" * 70)
    print()
    print("If effects are dose-dependent, ±1000 should show weaker but")
    print("same-direction effects as ±3000.")
    print()

    print(f"{'Dimension':<25} {'±3000 (original)':>30} {'±1000':>30}")
    print("-" * 85)

    for dim in dimensions:
        res_3000 = paired_direction_test(original, dim)
        res_1000 = paired_direction_test(dose_1000, dim)
        print(f"  {dim:<23} {res_3000['mean_diff']:+.3f} (p={res_3000['sign_p']:.4f})         {res_1000['mean_diff']:+.3f} (p={res_1000['sign_p']:.4f})")

    # ============================================================
    # 4. Position bias check for swapped and 1000
    # ============================================================
    print()
    print("=" * 70)
    print("4. POSITION BIAS CHECK (swapped and ±1000)")
    print("=" * 70)
    print()

    for label, dataset in [("Swapped", swapped), ("±1000", dose_1000)]:
        print(f"  {label}:")
        for dim in dimensions:
            raw_choices = []
            for r in dataset:
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
            bias_p = stats.binomtest(n_a, n_a + n_b, 0.5).pvalue if (n_a + n_b) > 0 else 1.0
            print(f"    {dim}: A={n_a} eq={n_eq} B={n_b}  bias_p={bias_p:.4f}")
        print()

    # ============================================================
    # 5. Category breakdown for confidence (the strongest finding)
    # ============================================================
    print("=" * 70)
    print("5. CONFIDENCE DIRECTION ASYMMETRY BY CATEGORY (original ±3000)")
    print("=" * 70)
    print()

    categories = sorted(set(r["category"] for r in original))
    for cat in categories:
        cat_results = [r for r in original if r["category"] == cat]
        prompt_ids_cat = sorted(set(r["prompt_id"] for r in cat_results))
        diffs = []
        for pid in prompt_ids_cat:
            neg = [r for r in cat_results if r["prompt_id"] == pid and r["steered_coefficient"] == -3000]
            pos = [r for r in cat_results if r["prompt_id"] == pid and r["steered_coefficient"] == 3000]
            if neg and pos:
                diffs.append(neg[0]["confidence_score"] - pos[0]["confidence_score"])
        n_pos = sum(1 for d in diffs if d > 0)
        n_neg = sum(1 for d in diffs if d < 0)
        n_eq = sum(1 for d in diffs if d == 0)
        mean_diff = np.mean(diffs) if diffs else 0
        print(f"  {cat}: mean_diff={mean_diff:+.3f}, +:{n_pos} =:{n_eq} -:{n_neg} (n={len(diffs)})")


if __name__ == "__main__":
    main()
