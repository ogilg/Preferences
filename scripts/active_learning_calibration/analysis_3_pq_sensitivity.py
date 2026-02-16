"""Analysis 3: p/q threshold sensitivity.

Replay the active learning pair selection with different (p,q) threshold combinations.
For each combo, track how the degree distribution evolves across iterations.
Report Gini coefficient of degree distribution — lower is more balanced.
"""

import json
from pathlib import Path

import numpy as np

from src.fitting.thurstonian_fitting.active_learning import (
    ActiveLearningState,
    generate_d_regular_pairs,
    select_next_pairs,
)
from src.fitting.thurstonian_fitting.thurstonian import PairwiseData, fit_thurstonian

from scripts.active_learning_calibration.fast_loading import load_measurements_fast, get_task_objects

OUTPUT_PATH = Path("experiments/probe_science/active_learning_calibration/analysis_3_results.json")

P_VALUES = [0.1, 0.2, 0.3, 0.5]
Q_VALUES = [0.1, 0.2, 0.3, 0.5]
BATCH_SIZE = 2000
INITIAL_DEGREE = 5
N_ITERATIONS = 8
ITER_0_MEASUREMENTS = 7500 * 5


def gini_coefficient(values):
    """Compute Gini coefficient. 0 = perfect equality, 1 = perfect inequality."""
    values = np.array(values, dtype=float)
    if len(values) == 0 or values.sum() == 0:
        return 0.0
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * sorted_vals) - (n + 1) * np.sum(sorted_vals)) / (n * np.sum(sorted_vals)))


def main():
    print("Loading measurements (fast JSON)...")
    all_measurements = load_measurements_fast()
    task_objects = get_task_objects(all_measurements)
    print(f"Loaded {len(all_measurements)} measurements, {len(task_objects)} tasks")

    # Index measurements by pair for quick lookup
    pair_to_measurements = {}
    for m in all_measurements:
        key = tuple(sorted([m.task_a.id, m.task_b.id]))
        if key not in pair_to_measurements:
            pair_to_measurements[key] = []
        pair_to_measurements[key].append(m)

    # Initial d-regular phase (shared across all p/q combos)
    initial_measurements = all_measurements[:ITER_0_MEASUREMENTS]

    print("Fitting Thurstonian on initial data...")
    init_data = PairwiseData.from_comparisons(initial_measurements, task_objects)
    init_fit = fit_thurstonian(init_data)
    print(f"  Converged: {init_fit.converged}")

    results = []

    for p in P_VALUES:
        for q in Q_VALUES:
            print(f"\n=== p={p}, q={q} ===")

            state = ActiveLearningState(tasks=task_objects)
            state.add_comparisons(initial_measurements)
            state.current_fit = init_fit
            state.iteration = 1

            iter_ginis = []
            iter_degree_stats = []

            for iteration in range(N_ITERATIONS):
                rng_iter = np.random.default_rng(42 + iteration)
                next_pairs = select_next_pairs(
                    state, batch_size=BATCH_SIZE,
                    p_threshold=p, q_threshold=q, rng=rng_iter,
                )

                if not next_pairs:
                    print(f"  Iter {iteration+2}: No pairs to select")
                    break

                # Use actual outcomes for selected pairs
                simulated = []
                for a, b in next_pairs:
                    key = tuple(sorted([a.id, b.id]))
                    if key in pair_to_measurements:
                        simulated.extend(pair_to_measurements[key])

                if not simulated:
                    break

                state.add_comparisons(simulated)
                state.iteration = iteration + 2

                degrees = list(state._degrees.values())
                gini = gini_coefficient(degrees)
                iter_ginis.append(gini)
                iter_degree_stats.append({
                    "mean": float(np.mean(degrees)),
                    "std": float(np.std(degrees)),
                    "min": int(np.min(degrees)),
                    "max": int(np.max(degrees)),
                    "p10": float(np.percentile(degrees, 10)),
                    "p90": float(np.percentile(degrees, 90)),
                })

                # Refit for next iteration
                data = PairwiseData.from_comparisons(state.comparisons, task_objects)
                state.previous_fit = state.current_fit
                state.current_fit = fit_thurstonian(data)

                print(f"  Iter {iteration+2}: {len(next_pairs)} pairs, "
                      f"gini={gini:.3f}, mean_deg={np.mean(degrees):.1f}, "
                      f"unique_pairs={len(state.sampled_pairs)}")

            entry = {
                "p_threshold": p,
                "q_threshold": q,
                "n_iterations_completed": len(iter_ginis),
                "final_gini": iter_ginis[-1] if iter_ginis else None,
                "final_unique_pairs": len(state.sampled_pairs),
                "final_total_comparisons": len(state.comparisons),
                "gini_by_iteration": iter_ginis,
                "degree_stats_by_iteration": iter_degree_stats,
            }
            results.append(entry)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
