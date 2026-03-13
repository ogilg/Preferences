"""Estimate how much of the r gap is explained by sampling variance.

Given Thurstonian scores, simulate what r we'd expect from 200 pairs
with 1 sample each (the multi-turn experiment setup), vs many resamples.
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats

RESULTS_DIR = Path("experiments/steering/multi_turn_pairwise/results")


def main():
    with open(RESULTS_DIR / "thurstonian_scores.json") as f:
        scores = json.load(f)

    with open(RESULTS_DIR / "pairs.json") as f:
        pairs = json.load(f)

    task_ids = sorted(scores.keys())
    mu = {tid: scores[tid] for tid in task_ids}

    # Thurstonian model: P(choose A over B) = Phi((mu_A - mu_B) / sqrt(2))
    # (assuming unit variance per task)
    from scipy.stats import norm

    rng = np.random.RandomState(0)
    n_sims = 1000

    # For each simulation: sample 1 outcome per pair, compute win rates, correlate with scores
    r_1sample = []
    r_10sample = []

    for _ in range(n_sims):
        wins_1 = {tid: 0 for tid in task_ids}
        total_1 = {tid: 0 for tid in task_ids}
        wins_10 = {tid: 0 for tid in task_ids}
        total_10 = {tid: 0 for tid in task_ids}

        for p in pairs:
            a_id, b_id = p["task_a"], p["task_b"]
            if a_id not in mu or b_id not in mu:
                continue
            prob_a = norm.cdf((mu[a_id] - mu[b_id]) / np.sqrt(2))

            # 1 sample
            chose_a = rng.random() < prob_a
            total_1[a_id] += 1
            total_1[b_id] += 1
            if chose_a:
                wins_1[a_id] += 1
            else:
                wins_1[b_id] += 1

            # 10 samples
            for _ in range(10):
                chose_a = rng.random() < prob_a
                total_10[a_id] += 1
                total_10[b_id] += 1
                if chose_a:
                    wins_10[a_id] += 1
                else:
                    wins_10[b_id] += 1

        # Compute win rates and correlate
        active = [tid for tid in task_ids if total_1[tid] > 0]
        wr_1 = np.array([wins_1[tid] / total_1[tid] for tid in active])
        wr_10 = np.array([wins_10[tid] / total_10[tid] for tid in active])
        sc = np.array([mu[tid] for tid in active])

        r_1sample.append(stats.pearsonr(wr_1, sc)[0])
        r_10sample.append(stats.pearsonr(wr_10, sc)[0])

    r_1 = np.array(r_1sample)
    r_10 = np.array(r_10sample)

    print("Simulated r (Thurstonian model, same 200 pairs):")
    print(f"  1 sample/pair:  mean={r_1.mean():.3f}, std={r_1.std():.3f}, 95% CI=[{np.percentile(r_1, 2.5):.3f}, {np.percentile(r_1, 97.5):.3f}]")
    print(f"  10 samples/pair: mean={r_10.mean():.3f}, std={r_10.std():.3f}, 95% CI=[{np.percentile(r_10, 2.5):.3f}, {np.percentile(r_10, 97.5):.3f}]")
    print()

    # Also simulate ordering bias under Thurstonian (no position effect)
    # Agreement = P(both canonical and reversed pick same task)
    agreements = []
    for _ in range(n_sims):
        agree = 0
        total = 0
        for p in pairs:
            a_id, b_id = p["task_a"], p["task_b"]
            if a_id not in mu or b_id not in mu:
                continue
            prob_a = norm.cdf((mu[a_id] - mu[b_id]) / np.sqrt(2))
            # Canonical: P(choose A)
            can_a = rng.random() < prob_a
            # Reversed: same probability (no position effect)
            rev_a = rng.random() < prob_a
            if can_a == rev_a:
                agree += 1
            total += 1
        agreements.append(agree / total)

    ag = np.array(agreements)
    print(f"Simulated ordering agreement (no position bias):")
    print(f"  mean={ag.mean():.3f}, std={ag.std():.3f}, 95% CI=[{np.percentile(ag, 2.5):.3f}, {np.percentile(ag, 97.5):.3f}]")
    print()

    # Observed values
    print("Observed values (best two prefills):")
    print(f"  'Got it...'     r_can=0.467, r_rev=0.473, agreement=0.756")
    print(f"  'Understood...' r_can=0.435, r_rev=0.427, agreement=0.734")


if __name__ == "__main__":
    main()
