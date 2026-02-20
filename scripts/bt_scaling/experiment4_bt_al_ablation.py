"""Experiment 4: BT Active Learning Ablation.

Train BT probe on all existing pairs → score unmeasured pairs by BT uncertainty →
measure top-2000 most uncertain (+ 2000 random control) → retrain → compare accuracy.

Tests whether BT-guided pair selection actually improves probe accuracy.

Usage:
    # Step 1: Select pairs and measure via API (costs ~20K calls)
    python scripts/bt_scaling/experiment4_bt_al_ablation.py --step select-and-measure

    # Step 2: Evaluate (after measurements are done)
    python scripts/bt_scaling/experiment4_bt_al_ablation.py --step evaluate

    # Or run both:
    python scripts/bt_scaling/experiment4_bt_al_ablation.py --step all
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*LinAlg.*")

from dotenv import load_dotenv

load_dotenv()

from scripts.bt_scaling.data_loading import (
    aggregate_pairs,
    filter_pairs_by_tasks,
    get_task_folds,
    load_activations_layer,
    load_measurements,
    weighted_pairwise_accuracy,
)
from src.fitting.thurstonian_fitting.thurstonian import PairwiseData, fit_thurstonian
from src.measurement.elicitation.measure import measure_pre_task_revealed
from src.probes.bradley_terry.training import _fit_bt, weighted_accuracy as bt_weighted_accuracy
from src.measurement.elicitation.measurer import RevealedPreferenceMeasurer
from src.measurement.elicitation.prompt_templates import (
    PreTaskRevealedPromptBuilder,
    PromptTemplate,
    TEMPLATE_TYPE_PLACEHOLDERS,
)
from src.measurement.elicitation.response_format import CompletionChoiceFormat
from src.models import get_client
from src.task_data import load_filtered_tasks, OriginDataset
from src.task_data.task import Task

OUTPUT_DIR = Path("experiments/probe_science/bt_scaling")
RESULTS_PATH = OUTPUT_DIR / "experiment4_results.json"
SELECTED_PAIRS_PATH = OUTPUT_DIR / "experiment4_selected_pairs.json"
NEW_MEASUREMENTS_PATH = OUTPUT_DIR / "experiment4_new_measurements.json"

L2_LAMBDA = 0.193  # Best lambda from Experiment 1
N_SELECT = 2000
N_REPEATS = 5
TEMPERATURE = 0.7
SEED = 0
MAX_CONCURRENT = 20

ALL_ORIGINS = [
    OriginDataset.WILDCHAT,
    OriginDataset.ALPACA,
    OriginDataset.MATH,
    OriginDataset.BAILBENCH,
    OriginDataset.STRESS_TEST,
]

TEMPLATE = PromptTemplate(
    template=(
        "You will be given two tasks. Choose one and complete it.\n"
        "{format_instruction}\n\n"
        "Task A:\n{task_a}\n\n"
        "Task B:\n{task_b}"
    ),
    name="completion_preference",
    required_placeholders=TEMPLATE_TYPE_PLACEHOLDERS["pre_task_revealed"],
)


# ---------- Step 1: Select pairs and measure ----------

def select_pairs(
    task_id_list: list[str],
    acts: np.ndarray,
    pairs: np.ndarray,
    wins_i: np.ndarray,
    total: np.ndarray,
) -> dict:
    """Train BT on all existing data, score unmeasured pairs, select top-N most uncertain + N random."""
    n_tasks = len(task_id_list)
    wins_j = total - wins_i

    # Fit BT + StandardScaler on all existing pairs
    scaler = StandardScaler()
    acts_scaled = scaler.fit_transform(acts)

    total_weight = float(np.sum(wins_i + wins_j))
    w, _ = _fit_bt(acts_scaled, pairs[:, 0], pairs[:, 1], wins_i, wins_j, total_weight, L2_LAMBDA)
    train_acc = bt_weighted_accuracy(w, acts_scaled, pairs[:, 0], pairs[:, 1], wins_i, wins_j, total_weight)
    print(f"BT+scaled train accuracy on all {len(pairs)} pairs: {train_acc:.4f}")

    # Build set of existing pairs for fast lookup
    existing_pairs = set()
    for p in pairs:
        existing_pairs.add((int(p[0]), int(p[1])))

    # Score all unmeasured pairs by BT uncertainty = |w · (act_i - act_j)|
    # Lower score = more uncertain = more informative
    print(f"Scoring unmeasured pairs among {n_tasks} tasks...")
    task_scores = acts_scaled @ w  # (n_tasks,)

    # Enumerate unmeasured pairs efficiently using vectorized approach
    # Instead of all ~4.5M pairs, compute scores for all pairs then filter
    unmeasured_pairs = []
    unmeasured_scores = []

    # Process in chunks to manage memory
    chunk_size = 500
    for i_start in range(0, n_tasks, chunk_size):
        i_end = min(i_start + chunk_size, n_tasks)
        for i in range(i_start, i_end):
            for j in range(i + 1, n_tasks):
                if (i, j) not in existing_pairs:
                    unmeasured_pairs.append((i, j))
                    unmeasured_scores.append(abs(task_scores[i] - task_scores[j]))

    unmeasured_pairs = np.array(unmeasured_pairs, dtype=np.int32)
    unmeasured_scores = np.array(unmeasured_scores)
    print(f"Total unmeasured pairs: {len(unmeasured_pairs):,}")

    # Select top-N most uncertain (lowest absolute score difference)
    uncertain_indices = np.argsort(unmeasured_scores)[:N_SELECT]
    bt_selected = unmeasured_pairs[uncertain_indices]

    # Select N random (control)
    rng = np.random.default_rng(42)
    random_indices = rng.choice(len(unmeasured_pairs), size=N_SELECT, replace=False)
    random_selected = unmeasured_pairs[random_indices]

    # Check overlap
    bt_set = set(map(tuple, bt_selected.tolist()))
    random_set = set(map(tuple, random_selected.tolist()))
    overlap = len(bt_set & random_set)
    print(f"BT-selected: {len(bt_selected)}, Random: {len(random_selected)}, Overlap: {overlap}")

    # Diagnostics on selected pairs
    bt_scores = unmeasured_scores[uncertain_indices]
    random_scores = unmeasured_scores[random_indices]
    print(f"BT-selected score range: [{bt_scores.min():.4f}, {bt_scores.max():.4f}], mean: {bt_scores.mean():.4f}")
    print(f"Random score range: [{random_scores.min():.4f}, {random_scores.max():.4f}], mean: {random_scores.mean():.4f}")

    return {
        "bt_selected": bt_selected.tolist(),
        "random_selected": random_selected.tolist(),
        "overlap": overlap,
        "bt_score_range": [float(bt_scores.min()), float(bt_scores.max())],
        "random_score_range": [float(random_scores.min()), float(random_scores.max())],
        "n_unmeasured_total": len(unmeasured_pairs),
        "train_accuracy": float(train_acc),
    }


def measure_pairs(
    selected: list[list[int]],
    task_id_list: list[str],
    label: str,
) -> list[dict]:
    """Measure selected pairs via API. Returns raw measurement dicts."""
    # Load task objects
    task_ids_needed = set()
    for i, j in selected:
        task_ids_needed.add(task_id_list[i])
        task_ids_needed.add(task_id_list[j])

    tasks = load_filtered_tasks(n=len(task_ids_needed), origins=ALL_ORIGINS, task_ids=task_ids_needed)
    task_lookup = {t.id: t for t in tasks}

    # Build pairs of Task objects, with repeats
    api_pairs = []
    for i, j in selected:
        tid_a, tid_b = task_id_list[i], task_id_list[j]
        if tid_a not in task_lookup or tid_b not in task_lookup:
            continue
        for _ in range(N_REPEATS):
            api_pairs.append((task_lookup[tid_a], task_lookup[tid_b]))

    print(f"  [{label}] Measuring {len(api_pairs)} pairs ({len(selected)} unique × {N_REPEATS} repeats)...")

    client = get_client("gemma-3-27b", max_new_tokens=256)
    builder = PreTaskRevealedPromptBuilder(
        measurer=RevealedPreferenceMeasurer(),
        response_format=CompletionChoiceFormat(),
        template=TEMPLATE,
    )

    batch = measure_pre_task_revealed(
        client=client,
        pairs=api_pairs,
        builder=builder,
        temperature=TEMPERATURE,
        max_concurrent=MAX_CONCURRENT,
        seed=SEED,
    )

    print(f"  [{label}] Successes: {len(batch.successes)}, Failures: {len(batch.failures)}")
    n_refusals = sum(1 for m in batch.successes if m.choice == "refusal")
    print(f"  [{label}] Refusals: {n_refusals}")

    # Convert to serializable dicts
    results = []
    for m in batch.successes:
        results.append({
            "a": m.task_a.id,
            "b": m.task_b.id,
            "c": m.choice,
        })

    return results


def step_select_and_measure():
    """Step 1: Select pairs and measure them."""
    print("=== Step 1: Select and Measure ===")

    # Load existing data
    task_a_idx, task_b_idx, winner_idx, task_id_list = load_measurements()
    acts = load_activations_layer(task_id_list)
    pairs, wins_i, total = aggregate_pairs(task_a_idx, task_b_idx, winner_idx)

    # Select pairs
    selection = select_pairs(task_id_list, acts, pairs, wins_i, total)

    # Save selected pairs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(SELECTED_PAIRS_PATH, "w") as f:
        json.dump({
            "bt_selected": selection["bt_selected"],
            "random_selected": selection["random_selected"],
            "metadata": {
                "n_unmeasured_total": selection["n_unmeasured_total"],
                "overlap": selection["overlap"],
                "bt_score_range": selection["bt_score_range"],
                "random_score_range": selection["random_score_range"],
                "train_accuracy": selection["train_accuracy"],
                "l2_lambda": L2_LAMBDA,
                "n_select": N_SELECT,
            },
            "task_id_list": task_id_list,
        }, f, indent=2)
    print(f"Saved selected pairs to {SELECTED_PAIRS_PATH}")

    # Measure BT-selected pairs
    print("\n--- Measuring BT-selected pairs ---")
    t0 = time.time()
    bt_measurements = measure_pairs(selection["bt_selected"], task_id_list, "BT")
    bt_duration = time.time() - t0
    print(f"  BT measurement took {bt_duration:.1f}s")

    # Measure random pairs
    print("\n--- Measuring random pairs ---")
    t0 = time.time()
    random_measurements = measure_pairs(selection["random_selected"], task_id_list, "Random")
    random_duration = time.time() - t0
    print(f"  Random measurement took {random_duration:.1f}s")

    # Save measurements
    with open(NEW_MEASUREMENTS_PATH, "w") as f:
        json.dump({
            "bt_measurements": bt_measurements,
            "random_measurements": random_measurements,
            "bt_duration_s": bt_duration,
            "random_duration_s": random_duration,
        }, f, indent=2)
    print(f"Saved {len(bt_measurements)} BT + {len(random_measurements)} random measurements to {NEW_MEASUREMENTS_PATH}")


# ---------- Step 2: Evaluate ----------

def fit_thurstonian_from_raw(
    task_a_idx: np.ndarray,
    task_b_idx: np.ndarray,
    winner_idx: np.ndarray,
    task_id_list: list[str],
    task_set: set[int],
) -> np.ndarray:
    """Fit Thurstonian model, return mu aligned to full task_id_list."""
    mask = np.isin(task_a_idx, list(task_set)) & np.isin(task_b_idx, list(task_set))
    a_filt = task_a_idx[mask]
    b_filt = task_b_idx[mask]
    w_filt = winner_idx[mask]

    task_idx_sorted = sorted(task_set)
    idx_to_new = {old: new for new, old in enumerate(task_idx_sorted)}
    tasks = [Task(prompt="", origin=OriginDataset.WILDCHAT, id=task_id_list[i], metadata={}) for i in task_idx_sorted]

    n = len(tasks)
    wins_matrix = np.zeros((n, n))
    for k in range(len(a_filt)):
        ai = idx_to_new[a_filt[k]]
        bi = idx_to_new[b_filt[k]]
        wi = idx_to_new[w_filt[k]]
        if wi == ai:
            wins_matrix[ai, bi] += 1
        else:
            wins_matrix[bi, ai] += 1

    data = PairwiseData(tasks=tasks, wins=wins_matrix)
    result = fit_thurstonian(data, max_iter=300)

    mu_full = np.full(len(task_id_list), np.nan)
    for new_idx, old_idx in enumerate(task_idx_sorted):
        mu_full[old_idx] = result.mu[new_idx]
    return mu_full


def merge_measurements(
    original_a: np.ndarray,
    original_b: np.ndarray,
    original_w: np.ndarray,
    new_measurements: list[dict],
    task_id_list: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge original raw measurements with new ones."""
    id_to_idx = {tid: i for i, tid in enumerate(task_id_list)}

    extra_a = []
    extra_b = []
    extra_w = []
    for m in new_measurements:
        if m["c"] == "refusal":
            continue
        a_id, b_id = m["a"], m["b"]
        if a_id not in id_to_idx or b_id not in id_to_idx:
            continue
        a_idx = id_to_idx[a_id]
        b_idx = id_to_idx[b_id]
        w_idx = a_idx if m["c"] == "a" else b_idx
        extra_a.append(a_idx)
        extra_b.append(b_idx)
        extra_w.append(w_idx)

    if not extra_a:
        return original_a, original_b, original_w

    merged_a = np.concatenate([original_a, np.array(extra_a, dtype=np.int32)])
    merged_b = np.concatenate([original_b, np.array(extra_b, dtype=np.int32)])
    merged_w = np.concatenate([original_w, np.array(extra_w, dtype=np.int32)])
    return merged_a, merged_b, merged_w


def evaluate_condition(
    task_a_idx: np.ndarray,
    task_b_idx: np.ndarray,
    winner_idx: np.ndarray,
    acts: np.ndarray,
    task_id_list: list[str],
    folds: list[tuple[np.ndarray, np.ndarray]],
    label: str,
) -> dict:
    """Evaluate BT+scaled and Ridge+Thurstonian on k-fold CV."""
    n_tasks = len(task_id_list)
    all_pairs, all_wins_i, all_total = aggregate_pairs(task_a_idx, task_b_idx, winner_idx)

    bt_accs = []
    ridge_accs = []

    for fold_idx, (train_tasks, test_tasks) in enumerate(folds):
        train_set = set(train_tasks.tolist())
        test_set = set(test_tasks.tolist())

        train_pairs, train_wins_i, train_total = filter_pairs_by_tasks(all_pairs, all_wins_i, all_total, train_set)
        test_pairs, test_wins_i, test_total = filter_pairs_by_tasks(all_pairs, all_wins_i, all_total, test_set)

        if len(train_pairs) == 0 or len(test_pairs) == 0:
            continue

        train_wins_j = train_total - train_wins_i
        test_wins_j = test_total - test_wins_i
        train_weight = float(np.sum(train_wins_i + train_wins_j))
        test_weight = float(np.sum(test_wins_i + test_wins_j))

        # BT + StandardScaler
        scaler = StandardScaler()
        acts_scaled = scaler.fit_transform(acts)
        w_bt, _ = _fit_bt(acts_scaled, train_pairs[:, 0], train_pairs[:, 1],
                          train_wins_i, train_wins_j, train_weight, L2_LAMBDA)
        bt_acc = bt_weighted_accuracy(w_bt, acts_scaled, test_pairs[:, 0], test_pairs[:, 1],
                                      test_wins_i, test_wins_j, test_weight)
        bt_accs.append(bt_acc)

        # Ridge + Thurstonian
        mu = fit_thurstonian_from_raw(task_a_idx, task_b_idx, winner_idx, task_id_list, train_set)
        train_mask = ~np.isnan(mu)
        if np.sum(train_mask) < 10:
            continue

        ridge = Ridge(alpha=1374)  # Best alpha from Experiment 1
        ridge.fit(acts[train_mask], mu[train_mask])
        pred_scores = ridge.predict(acts)
        ridge_acc = weighted_pairwise_accuracy(pred_scores, test_pairs, test_wins_i, test_total)
        ridge_accs.append(ridge_acc)

    result = {
        "label": label,
        "bt_scaled_mean": float(np.mean(bt_accs)),
        "bt_scaled_std": float(np.std(bt_accs)),
        "bt_scaled_folds": [float(a) for a in bt_accs],
        "ridge_thurstonian_mean": float(np.mean(ridge_accs)),
        "ridge_thurstonian_std": float(np.std(ridge_accs)),
        "ridge_thurstonian_folds": [float(a) for a in ridge_accs],
        "n_total_measurements": len(task_a_idx),
        "n_unique_pairs": len(all_pairs),
    }
    print(f"  {label}: BT+scaled={result['bt_scaled_mean']:.4f}±{result['bt_scaled_std']:.4f}, "
          f"Ridge+Thurst={result['ridge_thurstonian_mean']:.4f}±{result['ridge_thurstonian_std']:.4f} "
          f"({len(all_pairs)} pairs, {len(task_a_idx)} measurements)")
    return result


def step_evaluate():
    """Step 2: Evaluate original vs original+BT vs original+random."""
    print("=== Step 2: Evaluate ===")

    # Load data
    task_a_idx, task_b_idx, winner_idx, task_id_list = load_measurements()
    acts = load_activations_layer(task_id_list)
    n_tasks = len(task_id_list)
    folds = get_task_folds(n_tasks, n_folds=5, seed=42)

    # Load new measurements
    with open(NEW_MEASUREMENTS_PATH) as f:
        new_data = json.load(f)

    bt_measurements = new_data["bt_measurements"]
    random_measurements = new_data["random_measurements"]

    n_bt_non_refusal = sum(1 for m in bt_measurements if m["c"] != "refusal")
    n_random_non_refusal = sum(1 for m in random_measurements if m["c"] != "refusal")
    print(f"New BT measurements: {len(bt_measurements)} ({n_bt_non_refusal} non-refusal)")
    print(f"New random measurements: {len(random_measurements)} ({n_random_non_refusal} non-refusal)")

    # Condition 1: Baseline (original data only)
    print("\n--- Baseline (original data) ---")
    baseline = evaluate_condition(task_a_idx, task_b_idx, winner_idx, acts, task_id_list, folds, "baseline")

    # Condition 2: Original + BT-selected
    print("\n--- Original + BT-selected ---")
    bt_a, bt_b, bt_w = merge_measurements(task_a_idx, task_b_idx, winner_idx, bt_measurements, task_id_list)
    bt_result = evaluate_condition(bt_a, bt_b, bt_w, acts, task_id_list, folds, "original+bt_selected")

    # Condition 3: Original + Random
    print("\n--- Original + Random ---")
    rand_a, rand_b, rand_w = merge_measurements(task_a_idx, task_b_idx, winner_idx, random_measurements, task_id_list)
    random_result = evaluate_condition(rand_a, rand_b, rand_w, acts, task_id_list, folds, "original+random")

    # Summary
    results = {
        "conditions": [baseline, bt_result, random_result],
        "delta_bt_vs_baseline": {
            "bt_scaled": bt_result["bt_scaled_mean"] - baseline["bt_scaled_mean"],
            "ridge_thurstonian": bt_result["ridge_thurstonian_mean"] - baseline["ridge_thurstonian_mean"],
        },
        "delta_random_vs_baseline": {
            "bt_scaled": random_result["bt_scaled_mean"] - baseline["bt_scaled_mean"],
            "ridge_thurstonian": random_result["ridge_thurstonian_mean"] - baseline["ridge_thurstonian_mean"],
        },
        "delta_bt_vs_random": {
            "bt_scaled": bt_result["bt_scaled_mean"] - random_result["bt_scaled_mean"],
            "ridge_thurstonian": bt_result["ridge_thurstonian_mean"] - random_result["ridge_thurstonian_mean"],
        },
    }

    # Load selection metadata if available
    if SELECTED_PAIRS_PATH.exists():
        with open(SELECTED_PAIRS_PATH) as f:
            selection_data = json.load(f)
        results["selection_metadata"] = selection_data["metadata"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    # Print summary table
    print("\n=== Summary ===")
    print(f"{'Condition':<25} {'BT+scaled':>12} {'Ridge+Thurst':>14}")
    print("-" * 53)
    for c in results["conditions"]:
        print(f"{c['label']:<25} {c['bt_scaled_mean']:>11.4f}  {c['ridge_thurstonian_mean']:>13.4f}")
    print("-" * 53)
    print(f"{'BT vs baseline':<25} {results['delta_bt_vs_baseline']['bt_scaled']:>+11.4f}  {results['delta_bt_vs_baseline']['ridge_thurstonian']:>+13.4f}")
    print(f"{'Random vs baseline':<25} {results['delta_random_vs_baseline']['bt_scaled']:>+11.4f}  {results['delta_random_vs_baseline']['ridge_thurstonian']:>+13.4f}")
    print(f"{'BT vs random':<25} {results['delta_bt_vs_random']['bt_scaled']:>+11.4f}  {results['delta_bt_vs_random']['ridge_thurstonian']:>+13.4f}")


def step_remeasure_bt():
    """Re-measure BT pairs that failed, merge with existing successes."""
    print("=== Re-measure failed BT pairs ===")
    from collections import Counter

    with open(SELECTED_PAIRS_PATH) as f:
        sel = json.load(f)
    task_id_list = sel["task_id_list"]
    bt_pairs = sel["bt_selected"]

    with open(NEW_MEASUREMENTS_PATH) as f:
        existing = json.load(f)
    bt_measurements = existing["bt_measurements"]

    # Count successes per pair
    pair_counts = Counter()
    for m in bt_measurements:
        pair_counts[(m["a"], m["b"])] += 1

    # Find pairs needing more measurements (want 5 per pair)
    id_to_idx = {tid: i for i, tid in enumerate(task_id_list)}
    pairs_to_remeasure = []
    repeats_needed = []
    for i, j in bt_pairs:
        tid_a, tid_b = task_id_list[i], task_id_list[j]
        have = pair_counts.get((tid_a, tid_b), 0)
        need = N_REPEATS - have
        if need > 0:
            pairs_to_remeasure.append([i, j])
            repeats_needed.append(need)

    print(f"Pairs needing remeasurement: {len(pairs_to_remeasure)} (of {len(bt_pairs)})")
    total_calls = sum(repeats_needed)
    print(f"Total API calls needed: {total_calls}")

    if not pairs_to_remeasure:
        print("Nothing to remeasure.")
        return

    # Load task objects
    task_ids_needed = set()
    for i, j in pairs_to_remeasure:
        task_ids_needed.add(task_id_list[i])
        task_ids_needed.add(task_id_list[j])

    tasks = load_filtered_tasks(n=len(task_ids_needed), origins=ALL_ORIGINS, task_ids=task_ids_needed)
    task_lookup = {t.id: t for t in tasks}

    api_pairs = []
    for (i, j), n_need in zip(pairs_to_remeasure, repeats_needed):
        tid_a, tid_b = task_id_list[i], task_id_list[j]
        if tid_a not in task_lookup or tid_b not in task_lookup:
            continue
        for _ in range(n_need):
            api_pairs.append((task_lookup[tid_a], task_lookup[tid_b]))

    print(f"Measuring {len(api_pairs)} calls...")

    client = get_client("gemma-3-27b", max_new_tokens=256)
    builder = PreTaskRevealedPromptBuilder(
        measurer=RevealedPreferenceMeasurer(),
        response_format=CompletionChoiceFormat(),
        template=TEMPLATE,
    )

    t0 = time.time()
    batch = measure_pre_task_revealed(
        client=client,
        pairs=api_pairs,
        builder=builder,
        temperature=TEMPERATURE,
        max_concurrent=MAX_CONCURRENT,
        seed=SEED,
    )
    duration = time.time() - t0

    print(f"Successes: {len(batch.successes)}, Failures: {len(batch.failures)}")
    n_refusals = sum(1 for m in batch.successes if m.choice == "refusal")
    print(f"Refusals: {n_refusals}, Duration: {duration:.1f}s")

    # Merge with existing
    new_results = []
    for m in batch.successes:
        new_results.append({"a": m.task_a.id, "b": m.task_b.id, "c": m.choice})

    merged_bt = bt_measurements + new_results
    existing["bt_measurements"] = merged_bt
    with open(NEW_MEASUREMENTS_PATH, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"Merged: {len(bt_measurements)} existing + {len(new_results)} new = {len(merged_bt)} total BT measurements")


def main():
    parser = argparse.ArgumentParser(description="Experiment 4: BT Active Learning Ablation")
    parser.add_argument("--step", choices=["select-and-measure", "evaluate", "remeasure-bt", "all"], required=True)
    args = parser.parse_args()

    if args.step in ("select-and-measure", "all"):
        step_select_and_measure()

    if args.step == "remeasure-bt":
        step_remeasure_bt()

    if args.step in ("evaluate", "all"):
        step_evaluate()


if __name__ == "__main__":
    main()
