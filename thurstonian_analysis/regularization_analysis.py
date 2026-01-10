"""Analyze L2 variance regularization effect on Thurstonian model.

Run:
    python -m thurstonian_analysis.regularization_analysis          # synthetic dense data
    python -m thurstonian_analysis.regularization_analysis --real   # real data from results/measurements/
    python -m thurstonian_analysis.regularization_analysis --sparse # synthetic sparse (active learning) data
    python -m thurstonian_analysis.regularization_analysis --both   # both synthetic (dense + sparse)
    python -m thurstonian_analysis.regularization_analysis --all    # all three modes
    python -m thurstonian_analysis.regularization_analysis --al     # full AL loop with different regularizations
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.special import ndtr

from src.preferences.ranking.thurstonian import PairwiseData, fit_thurstonian, _preference_prob
from src.preferences.ranking.utils import simulate_pairwise_comparisons
from src.preferences.ranking.active_learning import generate_d_regular_pairs
from src.task_data import Task, OriginDataset

from thurstonian_analysis.config import N_TASKS, RESULTS_DIR
from thurstonian_analysis.utils import split_wins

if TYPE_CHECKING:
    from src.preferences.ranking.thurstonian import ThurstonianResult
    from src.types import BinaryPreferenceMeasurement

OUTPUT_DIR = Path(__file__).parent / "plots" / "regularization"

LAMBDAS = np.logspace(-2, np.log10(150), 5)


def make_task(id: str) -> Task:
    return Task(prompt=f"Task {id}", origin=OriginDataset.ALPACA, id=id, metadata={})


def load_all_datasets() -> list[tuple[str, PairwiseData]]:
    datasets = []
    if not RESULTS_DIR.exists():
        return datasets

    for result_dir in sorted(RESULTS_DIR.iterdir()):
        if not result_dir.is_dir():
            continue
        if result_dir.name.startswith("rating_"):
            continue
        measurements_path = result_dir / "measurements.yaml"
        if not measurements_path.exists():
            continue

        with open(measurements_path) as f:
            measurements = yaml.load(f, Loader=yaml.CSafeLoader)
        if not measurements:
            continue

        task_ids = set()
        for m in measurements:
            task_ids.add(m["task_a"])
            task_ids.add(m["task_b"])

        tasks = [make_task(tid) for tid in sorted(task_ids)]
        id_to_idx = {t.id: i for i, t in enumerate(tasks)}
        n = len(tasks)
        wins = np.zeros((n, n), dtype=np.int32)

        for m in measurements:
            i, j = id_to_idx[m["task_a"]], id_to_idx[m["task_b"]]
            if m["choice"] == "a":
                wins[i, j] += 1
            else:
                wins[j, i] += 1

        datasets.append((result_dir.name, PairwiseData(tasks=tasks, wins=wins)))

    return datasets


def eval_nll(mu: np.ndarray, sigma: np.ndarray, wins: np.ndarray) -> float:
    mu_diff = mu[:, np.newaxis] - mu[np.newaxis, :]
    scale = np.sqrt(sigma[:, np.newaxis] ** 2 + sigma[np.newaxis, :] ** 2)
    p = ndtr(mu_diff / scale)
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return -float(np.sum(wins * np.log(p)))


@dataclass
class RegularizationResults:
    lambdas: np.ndarray
    train_nlls: list[float]
    test_nlls: list[float]
    train_nlls_std: list[float]
    test_nlls_std: list[float]
    sigma_maxs: list[float]
    held_out_accuracies: list[float]
    held_out_accuracies_std: list[float]
    data_source: str
    n_datasets: int
    n_splits: int
    n_tasks: int


def compute_held_out_accuracy(mu: np.ndarray, wins: np.ndarray) -> float:
    """Compute prediction accuracy on held-out pairs."""
    n = wins.shape[0]
    correct = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            n_ij = wins[i, j] + wins[j, i]
            if n_ij == 0:
                continue
            pred_i_wins = mu[i] > mu[j]
            emp_i_wins = wins[i, j] > wins[j, i]
            if pred_i_wins == emp_i_wins:
                correct += 1
            total += 1
    return correct / total if total > 0 else 1.0


def run_regularization_on_wins(
    tasks: list[Task],
    wins: np.ndarray,
    test_frac: float,
    n_splits: int,
    lambdas: np.ndarray,
    verbose: bool = True,
) -> dict[str, list]:
    all_train_nlls = {lam: [] for lam in lambdas}
    all_test_nlls = {lam: [] for lam in lambdas}
    all_sigma_maxs = {lam: [] for lam in lambdas}
    all_held_out_accs = {lam: [] for lam in lambdas}

    for split_idx in range(n_splits):
        if verbose:
            print(f"  split {split_idx + 1}/{n_splits}", end="", flush=True)
        split_rng = np.random.default_rng(split_idx * 1000 + 123)
        train_wins, test_wins = split_wins(wins, test_frac, split_rng)
        n_train = int(train_wins.sum())
        n_test = int(test_wins.sum())

        if n_train == 0 or n_test == 0:
            if verbose:
                print(" (skipped)", flush=True)
            continue

        for lam_idx, lam in enumerate(lambdas):
            data = PairwiseData(tasks=tasks, wins=train_wins)
            result = fit_thurstonian(data, lambda_sigma=lam, log_sigma_bounds=(-4, 4))
            train_nll_per_comp = eval_nll(result.mu, result.sigma, train_wins) / n_train
            test_nll_per_comp = eval_nll(result.mu, result.sigma, test_wins) / n_test
            all_train_nlls[lam].append(train_nll_per_comp)
            all_test_nlls[lam].append(test_nll_per_comp)
            all_sigma_maxs[lam].append(float(result.sigma.max()))
            all_held_out_accs[lam].append(compute_held_out_accuracy(result.mu, test_wins))
            if verbose:
                print(".", end="", flush=True)
        if verbose:
            print(flush=True)

    return {
        "train_nlls": all_train_nlls,
        "test_nlls": all_test_nlls,
        "sigma_maxs": all_sigma_maxs,
        "held_out_accs": all_held_out_accs,
    }


def run_regularization_path_synthetic(
    n_tasks: int = N_TASKS,
    n_comparisons_per_pair: int = 10,
    test_frac: float = 0.2,
    n_splits: int = 5,
) -> RegularizationResults:
    print(f"Synthetic dense: {n_tasks} tasks, {n_splits} splits, {len(LAMBDAS)} lambdas")
    data_rng = np.random.default_rng(42)
    true_mu = np.linspace(-3, 3, n_tasks)
    true_sigma = np.ones(n_tasks)
    tasks = [make_task(f"t{i}") for i in range(n_tasks)]
    wins = simulate_pairwise_comparisons(true_mu, true_sigma, n_comparisons_per_pair, data_rng)

    results = run_regularization_on_wins(tasks, wins, test_frac, n_splits, LAMBDAS)

    return RegularizationResults(
        lambdas=LAMBDAS,
        train_nlls=[np.mean(results["train_nlls"][lam]) for lam in LAMBDAS],
        test_nlls=[np.mean(results["test_nlls"][lam]) for lam in LAMBDAS],
        train_nlls_std=[np.std(results["train_nlls"][lam]) for lam in LAMBDAS],
        test_nlls_std=[np.std(results["test_nlls"][lam]) for lam in LAMBDAS],
        sigma_maxs=[np.mean(results["sigma_maxs"][lam]) for lam in LAMBDAS],
        held_out_accuracies=[np.mean(results["held_out_accs"][lam]) for lam in LAMBDAS],
        held_out_accuracies_std=[np.std(results["held_out_accs"][lam]) for lam in LAMBDAS],
        data_source="synthetic",
        n_datasets=1,
        n_splits=n_splits,
        n_tasks=n_tasks,
    )


def simulate_sparse_pairwise_comparisons(
    tasks: list[Task],
    true_mu: np.ndarray,
    true_sigma: np.ndarray,
    d: int,
    n_comparisons_per_pair: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate pairwise comparisons only for pairs in a d-regular graph (sparse)."""
    n = len(tasks)
    wins = np.zeros((n, n), dtype=np.int32)
    id_to_idx = {t.id: i for i, t in enumerate(tasks)}

    pairs = generate_d_regular_pairs(tasks, d=d, rng=rng)

    for task_a, task_b in pairs:
        i, j = id_to_idx[task_a.id], id_to_idx[task_b.id]
        p_i_beats_j = _preference_prob(true_mu[i], true_mu[j], true_sigma[i], true_sigma[j])
        for _ in range(n_comparisons_per_pair):
            if rng.random() < p_i_beats_j:
                wins[i, j] += 1
            else:
                wins[j, i] += 1

    return wins


def run_regularization_path_sparse(
    n_tasks: int = N_TASKS,
    d: int = 5,
    n_comparisons_per_pair: int = 10,
    test_frac: float = 0.2,
    n_splits: int = 5,
) -> RegularizationResults:
    """Run regularization analysis on sparse (d-regular graph) synthetic data."""
    print(f"Synthetic sparse: {n_tasks} tasks, d={d}, {n_splits} splits, {len(LAMBDAS)} lambdas")
    data_rng = np.random.default_rng(42)
    true_mu = np.linspace(-3, 3, n_tasks)
    true_sigma = np.ones(n_tasks)
    tasks = [make_task(f"t{i}") for i in range(n_tasks)]
    wins = simulate_sparse_pairwise_comparisons(
        tasks, true_mu, true_sigma, d, n_comparisons_per_pair, data_rng
    )

    results = run_regularization_on_wins(tasks, wins, test_frac, n_splits, LAMBDAS)

    return RegularizationResults(
        lambdas=LAMBDAS,
        train_nlls=[np.mean(results["train_nlls"][lam]) for lam in LAMBDAS],
        test_nlls=[np.mean(results["test_nlls"][lam]) for lam in LAMBDAS],
        train_nlls_std=[np.std(results["train_nlls"][lam]) for lam in LAMBDAS],
        test_nlls_std=[np.std(results["test_nlls"][lam]) for lam in LAMBDAS],
        sigma_maxs=[np.mean(results["sigma_maxs"][lam]) for lam in LAMBDAS],
        held_out_accuracies=[np.mean(results["held_out_accs"][lam]) for lam in LAMBDAS],
        held_out_accuracies_std=[np.std(results["held_out_accs"][lam]) for lam in LAMBDAS],
        data_source="sparse",
        n_datasets=1,
        n_splits=n_splits,
        n_tasks=n_tasks,
    )


def subsample_pairwise_data(
    tasks: list[Task],
    wins: np.ndarray,
    n_tasks: int,
    rng: np.random.Generator,
) -> tuple[list[Task], np.ndarray]:
    """Subsample to n_tasks tasks, keeping only comparisons between them."""
    if len(tasks) <= n_tasks:
        return tasks, wins
    indices = rng.choice(len(tasks), size=n_tasks, replace=False)
    indices = np.sort(indices)
    subsampled_tasks = [tasks[i] for i in indices]
    subsampled_wins = wins[np.ix_(indices, indices)]
    return subsampled_tasks, subsampled_wins


def run_regularization_path_real(
    test_frac: float = 0.2,
    n_splits: int = 5,
) -> RegularizationResults:
    print(f"Real data: {n_splits} splits, {len(LAMBDAS)} lambdas, subsampling to {N_TASKS} tasks")
    datasets = load_all_datasets()
    if not datasets:
        raise ValueError(f"No datasets found in {RESULTS_DIR}")

    all_train_nlls = {lam: [] for lam in LAMBDAS}
    all_test_nlls = {lam: [] for lam in LAMBDAS}
    all_sigma_maxs = {lam: [] for lam in LAMBDAS}
    all_held_out_accs = {lam: [] for lam in LAMBDAS}

    valid_datasets = [(n, d) for n, d in datasets if d.n_tasks >= N_TASKS]
    subsample_rng = np.random.default_rng(42)
    for ds_idx, (name, data) in enumerate(valid_datasets):
        print(f"Dataset {ds_idx + 1}/{len(valid_datasets)}: {name}")
        tasks, wins = subsample_pairwise_data(data.tasks, data.wins, N_TASKS, subsample_rng)
        results = run_regularization_on_wins(tasks, wins, test_frac, n_splits, LAMBDAS)

        for lam in LAMBDAS:
            all_train_nlls[lam].extend(results["train_nlls"][lam])
            all_test_nlls[lam].extend(results["test_nlls"][lam])
            all_sigma_maxs[lam].extend(results["sigma_maxs"][lam])
            all_held_out_accs[lam].extend(results["held_out_accs"][lam])

    if not all_train_nlls[LAMBDAS[0]]:
        raise ValueError(f"No valid datasets found (need >= {N_TASKS} tasks)")

    n_valid_datasets = len(valid_datasets)

    return RegularizationResults(
        lambdas=LAMBDAS,
        train_nlls=[np.mean(all_train_nlls[lam]) for lam in LAMBDAS],
        test_nlls=[np.mean(all_test_nlls[lam]) for lam in LAMBDAS],
        train_nlls_std=[np.std(all_train_nlls[lam]) for lam in LAMBDAS],
        test_nlls_std=[np.std(all_test_nlls[lam]) for lam in LAMBDAS],
        sigma_maxs=[np.mean(all_sigma_maxs[lam]) for lam in LAMBDAS],
        held_out_accuracies=[np.mean(all_held_out_accs[lam]) for lam in LAMBDAS],
        held_out_accuracies_std=[np.std(all_held_out_accs[lam]) for lam in LAMBDAS],
        data_source="real",
        n_datasets=n_valid_datasets,
        n_splits=n_splits,
        n_tasks=N_TASKS,
    )


def plot_regularization_path(results: RegularizationResults, output_path: Path, nll_clip: float = 2.0):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Clip NLL values for display
    train_nlls = np.clip(results.train_nlls, None, nll_clip)
    test_nlls = np.clip(results.test_nlls, None, nll_clip)
    train_std = np.clip(results.train_nlls_std, None, nll_clip)
    test_std = np.clip(results.test_nlls_std, None, nll_clip)

    axes[0].errorbar(results.lambdas, train_nlls, yerr=train_std, fmt="o-", label="Train", markersize=5, capsize=3)
    axes[0].errorbar(results.lambdas, test_nlls, yerr=test_std, fmt="s-", label="Test", markersize=5, capsize=3)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("λ (regularization strength)")
    axes[0].set_ylabel("NLL per comparison")
    axes[0].set_ylim(0, nll_clip)
    axes[0].legend()
    axes[0].set_title("Regularization Path (mean ± std)")

    axes[1].semilogx(results.lambdas, results.sigma_maxs, "o-", color="coral", markersize=6)
    axes[1].axhline(y=1.0, color="k", linestyle="--", alpha=0.5, label="σ=1")
    axes[1].set_xlabel("λ")
    axes[1].set_ylabel("max(σ)")
    axes[1].legend()
    axes[1].set_title("Maximum Variance Parameter")

    axes[2].errorbar(results.lambdas, results.held_out_accuracies, yerr=results.held_out_accuracies_std,
                     fmt="s-", color="seagreen", markersize=6, capsize=3)
    axes[2].set_xscale("log")
    axes[2].set_xlabel("λ")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_ylim(0.5, 1.0)
    axes[2].set_title("Held-out Accuracy")

    if results.data_source == "synthetic":
        title = f"L2 Variance Regularization (SYNTHETIC DENSE, N_TASKS={results.n_tasks})"
    elif results.data_source == "sparse":
        title = f"L2 Variance Regularization (SYNTHETIC SPARSE/AL, N_TASKS={results.n_tasks})"
    else:
        title = f"L2 Variance Regularization (REAL, n={results.n_datasets} datasets, N_TASKS>={results.n_tasks})"

    plt.suptitle(title, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


@dataclass
class ALRegularizationResults:
    lambdas: list[float]
    # Each entry is a list of values per iteration
    trajectories: dict[float, dict[str, list[float]]]  # lambda -> {metric_name -> [values per iter]}
    n_tasks: int
    n_runs: int


def run_al_with_regularization(
    n_tasks: int = N_TASKS,
    lambdas: list[float] | None = None,
    initial_degree: int = 3,
    batch_size: int = 100,
    max_iterations: int = 20,
    n_comparisons_per_pair: int = 3,
    held_out_fraction: float = 0.2,
    n_runs: int = 3,
    seed: int = 42,
) -> ALRegularizationResults:
    """Run full active learning simulation with different regularization strengths."""
    from itertools import combinations
    from src.preferences.ranking.active_learning import (
        ActiveLearningState,
        generate_d_regular_pairs,
        select_next_pairs,
    )
    from scipy.stats import spearmanr

    if lambdas is None:
        lambdas = [0.0, 0.1, 1.0, 10.0, 50.0]

    print(f"AL regularization: {n_tasks} tasks, {n_runs} runs, {len(lambdas)} lambdas, {max_iterations} max iters")

    trajectories: dict[float, dict[str, list[list[float]]]] = {
        lam: {"spearman_vs_true": [], "held_out_accuracy": [], "cumulative_pairs": []}
        for lam in lambdas
    }

    for run_idx in range(n_runs):
        print(f"Run {run_idx + 1}/{n_runs}")
        run_seed = seed + run_idx * 1000
        rng = np.random.default_rng(run_seed)

        # Generate synthetic scenario
        tasks = [make_task(f"t{i}") for i in range(n_tasks)]
        true_mu = np.linspace(-3, 3, n_tasks)
        true_sigma = np.ones(n_tasks)
        id_to_idx = {t.id: i for i, t in enumerate(tasks)}

        # Split pairs into train/held-out
        all_pairs = list(combinations(tasks, 2))
        n_held_out = int(len(all_pairs) * held_out_fraction)
        held_out_indices = rng.choice(len(all_pairs), size=n_held_out, replace=False)
        held_out_pairs = [all_pairs[i] for i in held_out_indices]
        train_pairs_set = {
            tuple(sorted([a.id, b.id]))
            for i, (a, b) in enumerate(all_pairs)
            if i not in held_out_indices
        }

        # Generate held-out comparisons
        held_out_comparisons = []
        for a, b in held_out_pairs:
            i, j = id_to_idx[a.id], id_to_idx[b.id]
            p_i_beats_j = _preference_prob(true_mu[i], true_mu[j], true_sigma[i], true_sigma[j])
            for _ in range(n_comparisons_per_pair):
                winner = a if rng.random() < p_i_beats_j else b
                choice = "a" if winner == a else "b"
                held_out_comparisons.append(
                    _make_comparison(a, b, choice)
                )

        # Run AL for each lambda
        for lam_idx, lam in enumerate(lambdas):
            print(f"  λ={lam} ({lam_idx + 1}/{len(lambdas)})", end="", flush=True)
            al_rng = np.random.default_rng(run_seed + 500)
            state = ActiveLearningState(tasks=tasks)

            # Initial d-regular pairs (filtered to train set)
            initial_pairs = generate_d_regular_pairs(tasks, d=initial_degree, rng=al_rng)
            initial_pairs = [
                (a, b) for a, b in initial_pairs
                if tuple(sorted([a.id, b.id])) in train_pairs_set
            ]

            # Generate initial comparisons
            initial_comparisons = []
            for a, b in initial_pairs:
                i, j = id_to_idx[a.id], id_to_idx[b.id]
                p_i_beats_j = _preference_prob(true_mu[i], true_mu[j], true_sigma[i], true_sigma[j])
                for _ in range(n_comparisons_per_pair):
                    winner = a if al_rng.random() < p_i_beats_j else b
                    choice = "a" if winner == a else "b"
                    initial_comparisons.append(_make_comparison(a, b, choice))

            state.add_comparisons(initial_comparisons)
            state.fit(lambda_sigma=lam)

            run_spearman = [float(spearmanr(state.current_fit.mu, true_mu).correlation)]
            run_accuracy = [_compute_held_out_accuracy(state.current_fit, held_out_comparisons)]
            run_pairs = [len(state.sampled_pairs)]

            # Active learning loop
            for iter_idx in range(max_iterations):
                unsampled = [
                    (a, b) for a, b in state.get_unsampled_pairs()
                    if tuple(sorted([a.id, b.id])) in train_pairs_set
                ]
                if not unsampled:
                    break

                next_pairs = select_next_pairs(state, batch_size=batch_size, rng=al_rng)
                next_pairs = [
                    (a, b) for a, b in next_pairs
                    if tuple(sorted([a.id, b.id])) in train_pairs_set
                ]
                if not next_pairs:
                    break

                new_comparisons = []
                for a, b in next_pairs:
                    i, j = id_to_idx[a.id], id_to_idx[b.id]
                    p_i_beats_j = _preference_prob(true_mu[i], true_mu[j], true_sigma[i], true_sigma[j])
                    for _ in range(n_comparisons_per_pair):
                        winner = a if al_rng.random() < p_i_beats_j else b
                        choice = "a" if winner == a else "b"
                        new_comparisons.append(_make_comparison(a, b, choice))

                state.add_comparisons(new_comparisons)
                state.fit(lambda_sigma=lam)

                run_spearman.append(float(spearmanr(state.current_fit.mu, true_mu).correlation))
                run_accuracy.append(_compute_held_out_accuracy(state.current_fit, held_out_comparisons))
                run_pairs.append(len(state.sampled_pairs))
                print(".", end="", flush=True)

            print(f" ({len(run_pairs)} iters)", flush=True)
            trajectories[lam]["spearman_vs_true"].append(run_spearman)
            trajectories[lam]["held_out_accuracy"].append(run_accuracy)
            trajectories[lam]["cumulative_pairs"].append(run_pairs)

    # Average trajectories across runs (align by iteration, pad with last value if needed)
    avg_trajectories: dict[float, dict[str, list[float]]] = {}
    for lam in lambdas:
        avg_trajectories[lam] = {}
        for metric in ["spearman_vs_true", "held_out_accuracy", "cumulative_pairs"]:
            runs = trajectories[lam][metric]
            max_len = max(len(r) for r in runs)
            padded = [r + [r[-1]] * (max_len - len(r)) for r in runs]
            avg_trajectories[lam][metric] = [float(np.mean([p[i] for p in padded])) for i in range(max_len)]

    return ALRegularizationResults(
        lambdas=lambdas,
        trajectories=avg_trajectories,
        n_tasks=n_tasks,
        n_runs=n_runs,
    )


def _make_comparison(a: Task, b: Task, choice: str) -> "BinaryPreferenceMeasurement":
    from src.types import BinaryPreferenceMeasurement, PreferenceType
    return BinaryPreferenceMeasurement(
        task_a=a,
        task_b=b,
        choice=choice,
        preference_type=PreferenceType.PRE_TASK_STATED,
    )


def _compute_held_out_accuracy(
    result: "ThurstonianResult",
    held_out: list["BinaryPreferenceMeasurement"],
) -> float:
    if not held_out:
        return 1.0
    correct = 0
    for c in held_out:
        prob_a = result.preference_probability(c.task_a, c.task_b)
        predicted = "a" if prob_a >= 0.5 else "b"
        if predicted == c.choice:
            correct += 1
    return correct / len(held_out)


def plot_al_regularization(results: ALRegularizationResults, output_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(results.lambdas)))

    for lam, color in zip(results.lambdas, colors):
        traj = results.trajectories[lam]
        pairs = traj["cumulative_pairs"]
        label = f"λ={lam}"

        axes[0].plot(pairs, traj["spearman_vs_true"], "-o", color=color, label=label, markersize=3)
        axes[1].plot(pairs, traj["held_out_accuracy"], "-o", color=color, label=label, markersize=3)

    axes[0].set_xlabel("Cumulative pairs queried")
    axes[0].set_ylabel("Spearman ρ vs true utilities")
    axes[0].set_title("Ranking Recovery")
    axes[0].legend()
    axes[0].set_ylim(0, 1.05)

    axes[1].set_xlabel("Cumulative pairs queried")
    axes[1].set_ylabel("Held-out accuracy")
    axes[1].set_title("Prediction Accuracy")
    axes[1].legend()
    axes[1].set_ylim(0.5, 1.0)

    plt.suptitle(
        f"Active Learning with Regularization (N_TASKS={results.n_tasks}, n_runs={results.n_runs})",
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Regularization path analysis")
    parser.add_argument("--real", action="store_true", help="Use real data from results/measurements/")
    parser.add_argument("--sparse", action="store_true", help="Use sparse (active learning) synthetic data")
    parser.add_argument("--both", action="store_true", help="Run on both dense and sparse synthetic data")
    parser.add_argument("--all", action="store_true", help="Run on dense, sparse, and real data")
    parser.add_argument("--al", action="store_true", help="Run active learning with different regularizations")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.all:
        print("Running dense synthetic...")
        synthetic_results = run_regularization_path_synthetic()
        plot_regularization_path(synthetic_results, OUTPUT_DIR / "regularization_path_synthetic.png")

        print("Running sparse synthetic...")
        sparse_results = run_regularization_path_sparse()
        plot_regularization_path(sparse_results, OUTPUT_DIR / "regularization_path_sparse.png")

        print("Running real data...")
        real_results = run_regularization_path_real()
        plot_regularization_path(real_results, OUTPUT_DIR / "regularization_path_real.png")

    elif args.both:
        print("Running dense synthetic...")
        synthetic_results = run_regularization_path_synthetic()
        plot_regularization_path(synthetic_results, OUTPUT_DIR / "regularization_path_synthetic.png")

        print("Running sparse synthetic...")
        sparse_results = run_regularization_path_sparse()
        plot_regularization_path(sparse_results, OUTPUT_DIR / "regularization_path_sparse.png")

    elif args.real:
        real_results = run_regularization_path_real()
        plot_regularization_path(real_results, OUTPUT_DIR / "regularization_path_real.png")

    elif args.sparse:
        sparse_results = run_regularization_path_sparse()
        plot_regularization_path(sparse_results, OUTPUT_DIR / "regularization_path_sparse.png")

    elif args.al:
        print("Running active learning with different regularizations...")
        al_results = run_al_with_regularization()
        plot_al_regularization(al_results, OUTPUT_DIR / "al_regularization.png")
        print(f"Saved to {OUTPUT_DIR / 'al_regularization.png'}")

    else:
        synthetic_results = run_regularization_path_synthetic()
        plot_regularization_path(synthetic_results, OUTPUT_DIR / "regularization_path_synthetic.png")


if __name__ == "__main__":
    main()
