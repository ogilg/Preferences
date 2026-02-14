"""Analyze pairwise judge results for generalization experiment.

Computes direction asymmetry, dose-response, category breakdown, position bias.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXP_DIR = REPO_ROOT / "experiments/steering/program/open_ended_effects/generalization_new_prompts"

DIMENSIONS = ["emotional_engagement", "hedging", "elaboration", "confidence"]
DIM_SHORT = {"emotional_engagement": "engagement", "hedging": "hedging", "elaboration": "elaboration", "confidence": "confidence"}


def load_results(path: Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return [r for r in data if "error" not in r]


def direction_asymmetry(results: list[dict], dim: str, magnitude: int) -> dict:
    """Compute direction asymmetry: score(-mag) - score(+mag) per prompt."""
    prompt_ids = sorted(set(r["prompt_id"] for r in results))
    diffs = []
    details = []
    for pid in prompt_ids:
        neg = [r for r in results if r["prompt_id"] == pid and r["steered_coefficient"] == -magnitude]
        pos = [r for r in results if r["prompt_id"] == pid and r["steered_coefficient"] == magnitude]
        if neg and pos:
            d = neg[0][f"{dim}_score"] - pos[0][f"{dim}_score"]
            diffs.append(d)
            details.append({"prompt_id": pid, "category": neg[0]["category"], "diff": d})

    arr = np.array(diffs)
    n_pos = sum(1 for d in diffs if d > 0)
    n_neg = sum(1 for d in diffs if d < 0)
    n_eq = sum(1 for d in diffs if d == 0)

    nonzero = [d for d in diffs if d != 0]
    w_p = stats.wilcoxon(nonzero).pvalue if len(nonzero) >= 10 else float("nan")
    s_p = stats.binomtest(n_pos, n_pos + n_neg, 0.5).pvalue if (n_pos + n_neg) > 0 else 1.0

    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_eq": n_eq,
        "wilcoxon_p": w_p,
        "sign_p": s_p,
        "details": details,
    }


def combined_direction_asymmetry(original: list[dict], swapped: list[dict], dim: str, magnitude: int) -> dict:
    """Average of original and swapped direction asymmetry per prompt."""
    prompt_ids = sorted(set(r["prompt_id"] for r in original))
    diffs = []
    details = []

    for pid in prompt_ids:
        neg_scores = []
        pos_scores = []
        category = None
        for dataset in [original, swapped]:
            neg = [r for r in dataset if r["prompt_id"] == pid and r["steered_coefficient"] == -magnitude]
            pos = [r for r in dataset if r["prompt_id"] == pid and r["steered_coefficient"] == magnitude]
            if neg:
                neg_scores.append(neg[0][f"{dim}_score"])
                category = neg[0]["category"]
            if pos:
                pos_scores.append(pos[0][f"{dim}_score"])
                category = pos[0]["category"]
        if neg_scores and pos_scores:
            d = float(np.mean(neg_scores) - np.mean(pos_scores))
            diffs.append(d)
            details.append({"prompt_id": pid, "category": category, "diff": d})

    arr = np.array(diffs)
    n_pos = sum(1 for d in diffs if d > 0)
    n_neg = sum(1 for d in diffs if d < 0)
    n_eq = sum(1 for d in diffs if d == 0)

    nonzero = [d for d in diffs if d != 0]
    w_p = stats.wilcoxon(nonzero).pvalue if len(nonzero) >= 10 else float("nan")
    s_p = stats.binomtest(n_pos, n_pos + n_neg, 0.5).pvalue if (n_pos + n_neg) > 0 else 1.0

    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_eq": n_eq,
        "wilcoxon_p": w_p,
        "sign_p": s_p,
        "details": details,
    }


def position_bias(results: list[dict], dim: str) -> dict:
    n_a = sum(1 for r in results if r["judgment"][dim].endswith("_A"))
    n_b = sum(1 for r in results if r["judgment"][dim].endswith("_B"))
    n_eq = sum(1 for r in results if r["judgment"][dim] == "equal")
    bias_p = stats.binomtest(n_a, n_a + n_b, 0.5).pvalue if (n_a + n_b) > 0 else 1.0
    return {"n_a": n_a, "n_eq": n_eq, "n_b": n_b, "bias_p": bias_p}


def category_breakdown(details: list[dict]) -> dict[str, dict]:
    categories = sorted(set(d["category"] for d in details))
    breakdown = {}
    for cat in categories:
        cat_diffs = [d["diff"] for d in details if d["category"] == cat]
        n_pos = sum(1 for d in cat_diffs if d > 0)
        n_neg = sum(1 for d in cat_diffs if d < 0)
        n_eq = sum(1 for d in cat_diffs if d == 0)
        breakdown[cat] = {
            "n": len(cat_diffs),
            "mean": float(np.mean(cat_diffs)),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "n_eq": n_eq,
        }
    return breakdown


def main():
    original = load_results(EXP_DIR / "judge_results_original.json")
    swapped = load_results(EXP_DIR / "judge_results_swapped.json")
    print(f"Loaded {len(original)} original, {len(swapped)} swapped results")

    # ================================================================
    # 1. COMBINED DIRECTION ASYMMETRY AT ±3000
    # ================================================================
    print("\n" + "=" * 70)
    print("1. COMBINED DIRECTION ASYMMETRY AT ±3000 (original + swapped)")
    print("=" * 70)
    print(f"{'Dimension':<25} {'Mean':>8} {'Pos/Eq/Neg':>15} {'Sign p':>10} {'Wilcoxon p':>12}")
    print("-" * 70)

    combined_3000 = {}
    for dim in DIMENSIONS:
        res = combined_direction_asymmetry(original, swapped, dim, 3000)
        combined_3000[dim] = res
        print(f"  {DIM_SHORT[dim]:<23} {res['mean']:+.3f} {res['n_pos']:>4}/{res['n_eq']}/{res['n_neg']:<4} {res['sign_p']:>10.4f} {res['wilcoxon_p']:>12.4f}")

    # ================================================================
    # 2. DIRECTION ASYMMETRY AT ±2000 (dose-response check)
    # ================================================================
    print("\n" + "=" * 70)
    print("2. DOSE-RESPONSE: ±3000 vs ±2000 (combined original + swapped)")
    print("=" * 70)
    print(f"{'Dimension':<25} {'±3000 mean(sign_p)':>25} {'±2000 mean(sign_p)':>25}")
    print("-" * 75)

    for dim in DIMENSIONS:
        res_3000 = combined_3000[dim]
        res_2000 = combined_direction_asymmetry(original, swapped, dim, 2000)
        print(f"  {DIM_SHORT[dim]:<23} {res_3000['mean']:+.3f} (p={res_3000['sign_p']:.4f})     {res_2000['mean']:+.3f} (p={res_2000['sign_p']:.4f})")

    # ================================================================
    # 3. REPLICATION: ORIGINAL vs SWAPPED
    # ================================================================
    print("\n" + "=" * 70)
    print("3. REPLICATION: ORIGINAL vs POSITION-SWAPPED (±3000)")
    print("=" * 70)
    print(f"{'Dimension':<25} {'Original':>30} {'Swapped':>30}")
    print("-" * 85)

    for dim in DIMENSIONS:
        orig = direction_asymmetry(original, dim, 3000)
        swap = direction_asymmetry(swapped, dim, 3000)
        print(f"  {DIM_SHORT[dim]:<23} {orig['mean']:+.3f} (p={orig['sign_p']:.4f})         {swap['mean']:+.3f} (p={swap['sign_p']:.4f})")

    # ================================================================
    # 4. CATEGORY BREAKDOWN FOR COMBINED ±3000
    # ================================================================
    print("\n" + "=" * 70)
    print("4. CATEGORY BREAKDOWN (combined ±3000)")
    print("=" * 70)

    for dim in ["confidence", "emotional_engagement"]:
        res = combined_3000[dim]
        breakdown = category_breakdown(res["details"])
        print(f"\n  {DIM_SHORT[dim]}:")
        print(f"  {'Category':<20} {'N':>4} {'Mean':>8} {'Pos/Neg':>10}")
        print(f"  {'-'*46}")
        for cat, info in breakdown.items():
            print(f"  {cat:<20} {info['n']:>4} {info['mean']:+.3f} {info['n_pos']:>4}/{info['n_neg']}")

    # ================================================================
    # 5. POSITION BIAS CHECK
    # ================================================================
    print("\n" + "=" * 70)
    print("5. POSITION BIAS CHECK")
    print("=" * 70)

    for label, dataset in [("Original", original), ("Swapped", swapped)]:
        print(f"\n  {label}:")
        for dim in DIMENSIONS:
            pb = position_bias(dataset, dim)
            print(f"    {DIM_SHORT[dim]:<20}: A={pb['n_a']} eq={pb['n_eq']} B={pb['n_b']}  bias_p={pb['bias_p']:.4f}")

    # ================================================================
    # 6. PER-PROMPT DETAILS (for plotting)
    # ================================================================
    print("\n" + "=" * 70)
    print("6. PER-PROMPT DIRECTION ASYMMETRY (confidence, combined ±3000)")
    print("=" * 70)

    conf_details = combined_3000["confidence"]["details"]
    conf_details.sort(key=lambda d: d["diff"], reverse=True)
    for d in conf_details:
        bar = "+" * int(max(0, d["diff"] * 5)) + "-" * int(max(0, -d["diff"] * 5))
        print(f"  {d['prompt_id']:<8} ({d['category']:<16}) {d['diff']:+.2f} {bar}")

    print("\n" + "=" * 70)
    print("6b. PER-PROMPT DIRECTION ASYMMETRY (engagement, combined ±3000)")
    print("=" * 70)

    eng_details = combined_3000["emotional_engagement"]["details"]
    eng_details.sort(key=lambda d: d["diff"], reverse=True)
    for d in eng_details:
        bar = "+" * int(max(0, d["diff"] * 5)) + "-" * int(max(0, -d["diff"] * 5))
        print(f"  {d['prompt_id']:<8} ({d['category']:<16}) {d['diff']:+.2f} {bar}")

    # ================================================================
    # 7. SAVE ANALYSIS RESULTS FOR PLOTTING
    # ================================================================
    analysis = {
        "combined_3000": {
            dim: {k: v for k, v in combined_3000[dim].items()}
            for dim in DIMENSIONS
        },
    }
    out_path = EXP_DIR / "analysis_results.json"
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"\nSaved analysis to {out_path}")


if __name__ == "__main__":
    main()
