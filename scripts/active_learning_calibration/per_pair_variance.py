"""Per-pair variance analysis: can we get away with fewer samples per pair?

Analyzes:
1. Per-pair win rate distribution (how decisive are pairs?)
2. Per-pair agreement rate (how consistent are the 5 samples?)
3. Majority vote stability: how often does k<5 samples give the same majority as 5?
4. Thurstonian rank correlation with k=1,2,3 vs k=5 samples per pair
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from scripts.active_learning_calibration.fast_loading import (
    MEASUREMENTS_JSON,
    load_full_thurstonian_scores,
)
from src.fitting.thurstonian_fitting.thurstonian import PairwiseData, fit_thurstonian
from src.task_data import OriginDataset, Task


def load_per_pair_data():
    """Load measurements grouped by pair."""
    with open(MEASUREMENTS_JSON) as f:
        raw = json.load(f)

    # Group by pair
    pair_outcomes: dict[tuple[str, str], list[str]] = {}
    for m in raw:
        a, b, c = m["a"], m["b"], m["c"]
        key = (min(a, b), max(a, b))
        winner = a if c == "a" else b
        pair_outcomes.setdefault(key, []).append(winner)

    return pair_outcomes


def win_rate_analysis(pair_outcomes):
    """Analyze per-pair win rate distribution."""
    print("=" * 60)
    print("1. Per-pair win rate distribution")
    print("=" * 60)

    win_rates = []
    n_samples_list = []
    for (a, b), winners in pair_outcomes.items():
        n = len(winners)
        n_samples_list.append(n)
        a_wins = sum(1 for w in winners if w == a)
        win_rates.append(a_wins / n)

    win_rates = np.array(win_rates)
    n_samples_arr = np.array(n_samples_list)

    print(f"Total pairs: {len(win_rates)}")
    print(f"Samples per pair: min={n_samples_arr.min()}, max={n_samples_arr.max()}, "
          f"median={np.median(n_samples_arr):.0f}, mean={n_samples_arr.mean():.1f}")
    print(f"Pairs with exactly 5 samples: {(n_samples_arr == 5).sum()}")
    print()

    # Win rate buckets
    print("Win rate distribution (fraction of pairs):")
    buckets = [(0.0, 0.0, "0/5 (unanimous B)"),
               (0.2, 0.2, "1/5"),
               (0.4, 0.4, "2/5"),
               (0.6, 0.6, "3/5"),
               (0.8, 0.8, "4/5"),
               (1.0, 1.0, "5/5 (unanimous A)")]
    for lo, hi, label in buckets:
        count = np.sum(np.isclose(win_rates, lo))
        print(f"  {label}: {count} ({count/len(win_rates)*100:.1f}%)")

    # Decisive vs close
    decisive = np.sum((win_rates == 0) | (win_rates == 1))
    close = np.sum((win_rates == 0.4) | (win_rates == 0.6))
    very_close = np.sum(win_rates == 0.4) + np.sum(win_rates == 0.6)
    print(f"\nUnanimous (5-0): {decisive} ({decisive/len(win_rates)*100:.1f}%)")
    print(f"Close (3-2): {close} ({close/len(win_rates)*100:.1f}%)")
    print(f"Strong majority (4-1 or 5-0): {np.sum((win_rates <= 0.2) | (win_rates >= 0.8))} "
          f"({np.sum((win_rates <= 0.2) | (win_rates >= 0.8))/len(win_rates)*100:.1f}%)")

    return win_rates


def majority_vote_stability(pair_outcomes):
    """How often does k<5 samples give the same majority winner as k=5?"""
    print("\n" + "=" * 60)
    print("2. Majority vote stability at reduced sample sizes")
    print("=" * 60)

    # Only use pairs with exactly 5 samples
    pairs_5 = {k: v for k, v in pair_outcomes.items() if len(v) == 5}
    print(f"Pairs with 5 samples: {len(pairs_5)}")

    n_seeds = 100
    rng = np.random.default_rng(42)

    for k in [1, 2, 3]:
        agreements = []
        for seed_i in range(n_seeds):
            agree = 0
            for (a, b), winners in pairs_5.items():
                # Full majority
                a_wins_full = sum(1 for w in winners if w == a)
                full_majority = a if a_wins_full > 2 else b  # >2 out of 5

                # Subsample k
                subsample = rng.choice(winners, size=k, replace=False)
                a_wins_k = sum(1 for w in subsample if w == a)
                if a_wins_k > k / 2:
                    k_majority = a
                elif a_wins_k < k / 2:
                    k_majority = b
                else:
                    # Tie: random
                    k_majority = rng.choice([a, b])

                if k_majority == full_majority:
                    agree += 1
            agreements.append(agree / len(pairs_5))

        agreements = np.array(agreements)
        print(f"k={k}: majority agreement with k=5: {agreements.mean():.4f} ± {agreements.std():.4f}")


def thurstonian_stability(pair_outcomes):
    """Refit Thurstonian with k=1,2,3 samples per pair, compare to k=5."""
    print("\n" + "=" * 60)
    print("3. Thurstonian rank stability at reduced sample sizes")
    print("=" * 60)

    full_scores = load_full_thurstonian_scores()
    task_ids = sorted(full_scores.keys())
    task_objects = [Task(prompt="", origin=OriginDataset.WILDCHAT, id=tid, metadata={}) for tid in task_ids]
    full_utilities = np.array([full_scores[tid] for tid in task_ids])

    # Only pairs with exactly 5 samples
    pairs_5 = {k: v for k, v in pair_outcomes.items() if len(v) == 5}

    n_seeds = 10
    rng = np.random.default_rng(42)

    for k in [1, 2, 3, 5]:
        rank_corrs = []
        for seed_i in range(n_seeds):
            # Build wins matrix from subsampled measurements
            wins = np.zeros((len(task_ids), len(task_ids)), dtype=np.int32)
            id_to_idx = {tid: i for i, tid in enumerate(task_ids)}

            for (a, b), winners in pairs_5.items():
                a_idx = id_to_idx[a]
                b_idx = id_to_idx[b]

                if k < 5:
                    subsample = rng.choice(winners, size=k, replace=False)
                else:
                    subsample = winners

                for w in subsample:
                    if w == a:
                        wins[a_idx, b_idx] += 1
                    else:
                        wins[b_idx, a_idx] += 1

            # Fit Thurstonian
            pairwise_data = PairwiseData(tasks=task_objects, wins=wins)
            result = fit_thurstonian(pairwise_data)
            fitted_utilities = result.mu

            corr = spearmanr(full_utilities, fitted_utilities).statistic
            rank_corrs.append(corr)

        rank_corrs = np.array(rank_corrs)
        if k == 5:
            print(f"k={k}: rank correlation with full-data: {rank_corrs.mean():.4f} (deterministic)")
        else:
            print(f"k={k}: rank correlation with full-data: {rank_corrs.mean():.4f} ± {rank_corrs.std():.4f}")


if __name__ == "__main__":
    pair_outcomes = load_per_pair_data()
    win_rate_analysis(pair_outcomes)
    majority_vote_stability(pair_outcomes)
    thurstonian_stability(pair_outcomes)
