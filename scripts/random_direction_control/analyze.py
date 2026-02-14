"""Analyze judge results: probe vs random direction asymmetries.

Computes direction asymmetry for each direction, compares probe to random,
and runs statistical tests.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXP_DIR = REPO_ROOT / "experiments/steering/program/open_ended_effects/random_direction_control"
ORIGINAL_PATH = EXP_DIR / "judge_results_original.json"
SWAPPED_PATH = EXP_DIR / "judge_results_swapped.json"
OUTPUT_PATH = EXP_DIR / "analysis_results.json"

DIMENSIONS = ["self_referential_framing", "emotional_engagement"]
DIRECTIONS = ["probe", "random_200", "random_201", "random_202", "random_203", "random_204"]


def load_results():
    with open(ORIGINAL_PATH) as f:
        original = json.load(f)
    with open(SWAPPED_PATH) as f:
        swapped = json.load(f)
    return original, swapped


def compute_direction_asymmetry(results_original, results_swapped):
    """Compute direction asymmetry (neg_score - pos_score) averaged over position orders.

    For each (prompt, direction, dimension):
      neg_score = judge score when coefficient = -3000
      pos_score = judge score when coefficient = +3000
      direction_asymmetry = neg_score - pos_score (averaged over both position orders)

    Positive = negative steering produces more of that quality.
    """
    # Build lookup: (prompt_id, direction, coefficient) -> [scores from original, swapped]
    scores = defaultdict(list)

    for run_results in [results_original, results_swapped]:
        for r in run_results:
            if "error" in r:
                continue
            key = (r["prompt_id"], r["direction"], r["steered_coefficient"])
            for dim in DIMENSIONS:
                scores[(key, dim)].append(r[f"{dim}_score"])

    # Compute per-prompt, per-direction mean scores and direction asymmetries
    # direction_asymmetry[dim][direction] = list of per-prompt asymmetries
    asymmetries = {dim: defaultdict(list) for dim in DIMENSIONS}
    per_prompt = {dim: defaultdict(dict) for dim in DIMENSIONS}

    prompt_ids = sorted(set(r["prompt_id"] for r in results_original if "error" not in r))

    for dim in DIMENSIONS:
        for direction in DIRECTIONS:
            for pid in prompt_ids:
                neg_key = ((pid, direction, -3000), dim)
                pos_key = ((pid, direction, 3000), dim)

                neg_scores = scores[neg_key]
                pos_scores = scores[pos_key]

                if neg_scores and pos_scores:
                    neg_mean = np.mean(neg_scores)
                    pos_mean = np.mean(pos_scores)
                    asym = neg_mean - pos_mean
                    asymmetries[dim][direction].append(asym)
                    per_prompt[dim][direction][pid] = {
                        "neg_score": neg_mean,
                        "pos_score": pos_mean,
                        "asymmetry": asym,
                    }

    return asymmetries, per_prompt


def sign_test(values):
    positive = sum(1 for v in values if v > 0)
    negative = sum(1 for v in values if v < 0)
    ties = sum(1 for v in values if v == 0)
    n = positive + negative
    if n == 0:
        return 1.0, positive, negative, ties
    p = stats.binomtest(positive, n, 0.5).pvalue if n > 0 else 1.0
    return p, positive, negative, ties


def analyze(asymmetries, per_prompt):
    results = {}

    for dim in DIMENSIONS:
        dim_results = {}

        for direction in DIRECTIONS:
            vals = asymmetries[dim][direction]
            mean_asym = np.mean(vals) if vals else 0
            p_sign, pos, neg, ties = sign_test(vals)

            # Wilcoxon signed-rank test
            non_zero = [v for v in vals if v != 0]
            if len(non_zero) >= 5:
                stat, p_wilcoxon = stats.wilcoxon(non_zero)
            else:
                stat, p_wilcoxon = float("nan"), float("nan")

            dim_results[direction] = {
                "mean_asymmetry": round(mean_asym, 4),
                "prompts_favoring_neg": pos,
                "prompts_favoring_pos": neg,
                "ties": ties,
                "sign_test_p": round(p_sign, 6),
                "wilcoxon_p": round(p_wilcoxon, 6) if not np.isnan(p_wilcoxon) else None,
                "per_prompt_asymmetries": vals,
            }

        # Probe vs random comparison
        probe_asym = asymmetries[dim]["probe"]
        probe_mean = np.mean(probe_asym)
        random_means = [np.mean(asymmetries[dim][d]) for d in DIRECTIONS if d.startswith("random_")]
        random_all = []
        for d in DIRECTIONS:
            if d.startswith("random_"):
                random_all.extend(asymmetries[dim][d])

        # Permutation test: is probe mean asymmetry > random mean asymmetry?
        random_mean = np.mean(random_means)
        random_max = max(random_means)
        random_std = np.std(random_means)

        dim_results["comparison"] = {
            "probe_mean": round(probe_mean, 4),
            "random_direction_means": {d: round(np.mean(asymmetries[dim][d]), 4) for d in DIRECTIONS if d.startswith("random_")},
            "random_grand_mean": round(random_mean, 4),
            "random_max": round(random_max, 4),
            "random_std": round(random_std, 4),
            "probe_vs_random_mean_ratio": round(probe_mean / random_mean, 4) if random_mean != 0 else float("inf"),
            "probe_exceeds_all_random": bool(probe_mean > random_max),
        }

        # Sign consistency across random directions
        random_signs = []
        for d in DIRECTIONS:
            if d.startswith("random_"):
                m = np.mean(asymmetries[dim][d])
                random_signs.append(1 if m > 0 else (-1 if m < 0 else 0))
        dim_results["comparison"]["random_sign_consistency"] = {
            "positive": sum(1 for s in random_signs if s > 0),
            "negative": sum(1 for s in random_signs if s < 0),
            "zero": sum(1 for s in random_signs if s == 0),
        }

        results[dim] = dim_results

    return results


def print_summary(results):
    for dim in DIMENSIONS:
        print(f"\n{'='*60}")
        print(f"  {dim.upper()}")
        print(f"{'='*60}")

        for direction in DIRECTIONS:
            r = results[dim][direction]
            label = "PROBE" if direction == "probe" else direction
            print(f"\n  {label}:")
            print(f"    Mean asymmetry: {r['mean_asymmetry']:+.3f}")
            print(f"    Favoring neg/pos (ties): {r['prompts_favoring_neg']}/{r['prompts_favoring_pos']} ({r['ties']})")
            print(f"    Sign test p: {r['sign_test_p']:.4f}")
            if r['wilcoxon_p'] is not None:
                print(f"    Wilcoxon p: {r['wilcoxon_p']:.4f}")

        comp = results[dim]["comparison"]
        print(f"\n  --- Probe vs Random ---")
        print(f"    Probe mean: {comp['probe_mean']:+.3f}")
        print(f"    Random grand mean: {comp['random_grand_mean']:+.3f}")
        print(f"    Random max: {comp['random_max']:+.3f}")
        print(f"    Probe/random ratio: {comp['probe_vs_random_mean_ratio']:.2f}x")
        print(f"    Probe > all random: {comp['probe_exceeds_all_random']}")
        print(f"    Random sign consistency: {comp['random_sign_consistency']}")


if __name__ == "__main__":
    original, swapped = load_results()
    asymmetries, per_prompt = compute_direction_asymmetry(original, swapped)
    results = analyze(asymmetries, per_prompt)
    print_summary(results)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved analysis to {OUTPUT_PATH}")
