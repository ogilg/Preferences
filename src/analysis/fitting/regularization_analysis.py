"""Analyze L2 variance regularization effect on Thurstonian model.

Regularization is L2 on log(sigma), which pulls sigma toward 1 (since log(1)=0).

Run:
    python -m src.analysis.fitting.regularization_analysis          # synthetic dense data
    python -m src.analysis.fitting.regularization_analysis --sparse # synthetic sparse (AL-style) data
    python -m src.analysis.fitting.regularization_analysis --real   # real data from cache
    python -m src.analysis.fitting.regularization_analysis --both   # both synthetic modes
    python -m src.analysis.fitting.regularization_analysis --all    # all three modes
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr

from src.analysis.fitting.config import N_TASKS
from src.analysis.fitting.utils import split_wins
from src.fitting.thurstonian_fitting.active_learning import generate_d_regular_pairs
from src.fitting.thurstonian_fitting.thurstonian import (
    PairwiseData,
    _preference_prob,
    fit_thurstonian,
)
from src.fitting.thurstonian_fitting.utils import simulate_pairwise_comparisons
from src.measurement.storage.base import load_yaml
from src.task_data import OriginDataset, Task

OUTPUT_DIR = Path(__file__).parent / "plots" / "regularization"
REVEALED_CACHE_DIR = Path("results/cache/revealed")

LAMBDAS = np.logspace(-2, np.log10(150), 5)


def make_task(id: str) -> Task:
    return Task(prompt=f"Task {id}", origin=OriginDataset.SYNTHETIC, id=id, metadata={})


def load_pairwise_from_cache() -> list[tuple[str, list[Task], np.ndarray]]:
    """Load all pairwise comparison data from the revealed cache.

    Returns list of (model_name, tasks, wins_matrix) tuples.
    Aggregates across all templates/configs for each model.
    """
    datasets = []

    if not REVEALED_CACHE_DIR.exists():
        return datasets

    for cache_file in sorted(REVEALED_CACHE_DIR.glob("*.yaml")):
        model_name = cache_file.stem
        data = load_yaml(cache_file)
        if not data:
            continue

        # Collect all task IDs and build wins matrix
        task_ids_set: set[str] = set()
        comparisons: list[tuple[str, str, str]] = []  # (task_a, task_b, choice)

        for entry in data.values():
            task_a = entry["task_a_id"]
            task_b = entry["task_b_id"]
            task_ids_set.add(task_a)
            task_ids_set.add(task_b)
            for sample in entry["samples"]:
                comparisons.append((task_a, task_b, sample["choice"]))

        if len(task_ids_set) < 10:  # Skip tiny datasets
            continue

        task_ids = sorted(task_ids_set)
        id_to_idx = {tid: i for i, tid in enumerate(task_ids)}
        n = len(task_ids)
        wins = np.zeros((n, n), dtype=np.int32)

        for task_a, task_b, choice in comparisons:
            i, j = id_to_idx[task_a], id_to_idx[task_b]
            if choice == "a":
                wins[i, j] += 1
            else:
                wins[j, i] += 1

        tasks = [make_task(tid) for tid in task_ids]
        datasets.append((model_name, tasks, wins))

    return datasets


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
    sigma_means: list[float]
    held_out_accuracies: list[float]
    held_out_accuracies_std: list[float]
    data_source: str
    n_splits: int
    n_tasks: int


def compute_held_out_accuracy(mu: np.ndarray, wins: np.ndarray) -> float:
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
) -> dict[str, dict]:
    all_train_nlls = {lam: [] for lam in lambdas}
    all_test_nlls = {lam: [] for lam in lambdas}
    all_sigma_maxs = {lam: [] for lam in lambdas}
    all_sigma_means = {lam: [] for lam in lambdas}
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

        for lam in lambdas:
            data = PairwiseData(tasks=tasks, wins=train_wins)
            result = fit_thurstonian(data, lambda_sigma=lam, log_sigma_bounds=(-4, 4))
            train_nll_per_comp = eval_nll(result.mu, result.sigma, train_wins) / n_train
            test_nll_per_comp = eval_nll(result.mu, result.sigma, test_wins) / n_test
            all_train_nlls[lam].append(train_nll_per_comp)
            all_test_nlls[lam].append(test_nll_per_comp)
            all_sigma_maxs[lam].append(float(result.sigma.max()))
            all_sigma_means[lam].append(float(result.sigma.mean()))
            all_held_out_accs[lam].append(compute_held_out_accuracy(result.mu, test_wins))
            if verbose:
                print(".", end="", flush=True)
        if verbose:
            print(flush=True)

    return {
        "train_nlls": all_train_nlls,
        "test_nlls": all_test_nlls,
        "sigma_maxs": all_sigma_maxs,
        "sigma_means": all_sigma_means,
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
        sigma_means=[np.mean(results["sigma_means"][lam]) for lam in LAMBDAS],
        held_out_accuracies=[np.mean(results["held_out_accs"][lam]) for lam in LAMBDAS],
        held_out_accuracies_std=[np.std(results["held_out_accs"][lam]) for lam in LAMBDAS],
        data_source="dense",
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
        sigma_means=[np.mean(results["sigma_means"][lam]) for lam in LAMBDAS],
        held_out_accuracies=[np.mean(results["held_out_accs"][lam]) for lam in LAMBDAS],
        held_out_accuracies_std=[np.std(results["held_out_accs"][lam]) for lam in LAMBDAS],
        data_source="sparse",
        n_splits=n_splits,
        n_tasks=n_tasks,
    )


def run_regularization_path_real(
    test_frac: float = 0.2,
    n_splits: int = 5,
    min_tasks: int = 50,
) -> RegularizationResults:
    print(f"Real data: {n_splits} splits, {len(LAMBDAS)} lambdas, min {min_tasks} tasks")
    datasets = load_pairwise_from_cache()
    if not datasets:
        raise ValueError(f"No datasets found in {REVEALED_CACHE_DIR}")

    all_train_nlls = {lam: [] for lam in LAMBDAS}
    all_test_nlls = {lam: [] for lam in LAMBDAS}
    all_sigma_maxs = {lam: [] for lam in LAMBDAS}
    all_sigma_means = {lam: [] for lam in LAMBDAS}
    all_held_out_accs = {lam: [] for lam in LAMBDAS}

    valid_datasets = [(name, tasks, wins) for name, tasks, wins in datasets if len(tasks) >= min_tasks]
    print(f"Found {len(valid_datasets)} datasets with >= {min_tasks} tasks")

    subsample_rng = np.random.default_rng(42)
    for ds_idx, (name, tasks, wins) in enumerate(valid_datasets):
        # Subsample if needed
        if len(tasks) > N_TASKS:
            tasks, wins = subsample_pairwise_data(tasks, wins, N_TASKS, subsample_rng)
        n_comparisons = int(wins.sum())
        print(f"  Dataset {ds_idx + 1}/{len(valid_datasets)}: {name} ({len(tasks)} tasks, {n_comparisons} comparisons)")

        if n_comparisons < 100:  # Skip if too few comparisons
            print("    (skipped - too few comparisons)")
            continue

        results = run_regularization_on_wins(tasks, wins, test_frac, n_splits, LAMBDAS, verbose=False)

        for lam in LAMBDAS:
            if results["train_nlls"][lam]:  # Only add if we got results
                all_train_nlls[lam].extend(results["train_nlls"][lam])
                all_test_nlls[lam].extend(results["test_nlls"][lam])
                all_sigma_maxs[lam].extend(results["sigma_maxs"][lam])
                all_sigma_means[lam].extend(results["sigma_means"][lam])
                all_held_out_accs[lam].extend(results["held_out_accs"][lam])

    if not all_train_nlls[LAMBDAS[0]]:
        raise ValueError(f"No valid datasets found (need >= {min_tasks} tasks with sufficient comparisons)")

    return RegularizationResults(
        lambdas=LAMBDAS,
        train_nlls=[np.mean(all_train_nlls[lam]) for lam in LAMBDAS],
        test_nlls=[np.mean(all_test_nlls[lam]) for lam in LAMBDAS],
        train_nlls_std=[np.std(all_train_nlls[lam]) for lam in LAMBDAS],
        test_nlls_std=[np.std(all_test_nlls[lam]) for lam in LAMBDAS],
        sigma_maxs=[np.mean(all_sigma_maxs[lam]) for lam in LAMBDAS],
        sigma_means=[np.mean(all_sigma_means[lam]) for lam in LAMBDAS],
        held_out_accuracies=[np.mean(all_held_out_accs[lam]) for lam in LAMBDAS],
        held_out_accuracies_std=[np.std(all_held_out_accs[lam]) for lam in LAMBDAS],
        data_source="real",
        n_splits=n_splits,
        n_tasks=N_TASKS,
    )


def plot_regularization_path(results: RegularizationResults, output_path: Path, nll_clip: float = 2.0):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Clip NLL values for display
    train_nlls = np.clip(results.train_nlls, None, nll_clip)
    test_nlls = np.clip(results.test_nlls, None, nll_clip)
    train_std = np.clip(results.train_nlls_std, None, nll_clip)
    test_std = np.clip(results.test_nlls_std, None, nll_clip)

    axes[0].errorbar(results.lambdas, train_nlls, yerr=train_std, fmt="o-", label="Train", markersize=5, capsize=3)
    axes[0].errorbar(results.lambdas, test_nlls, yerr=test_std, fmt="s-", label="Test", markersize=5, capsize=3)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("lambda")
    axes[0].set_ylabel("NLL per comparison")
    axes[0].set_ylim(0, nll_clip)
    axes[0].legend()
    axes[0].set_title("Train/Test NLL")

    axes[1].semilogx(results.lambdas, results.sigma_maxs, "o-", color="coral", markersize=6, label="max")
    axes[1].semilogx(results.lambdas, results.sigma_means, "s-", color="steelblue", markersize=6, label="mean")
    axes[1].axhline(y=1.0, color="k", linestyle="--", alpha=0.5, label="sigma=1")
    axes[1].set_xlabel("lambda")
    axes[1].set_ylabel("sigma")
    axes[1].legend()
    axes[1].set_title("Sigma Statistics")

    axes[2].errorbar(results.lambdas, results.held_out_accuracies, yerr=results.held_out_accuracies_std,
                     fmt="s-", color="seagreen", markersize=6, capsize=3)
    axes[2].set_xscale("log")
    axes[2].set_xlabel("lambda")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_ylim(0.5, 1.0)
    axes[2].set_title("Held-out Accuracy")

    # Log-sigma deviation from 0 (i.e., how far sigma is from 1)
    log_sigma_dev = np.abs(np.log(results.sigma_means))
    axes[3].semilogx(results.lambdas, log_sigma_dev, "o-", color="purple", markersize=6)
    axes[3].set_xlabel("lambda")
    axes[3].set_ylabel("|log(mean sigma)|")
    axes[3].set_title("Deviation from sigma=1")

    if results.data_source == "dense":
        title = f"L2 on log(sigma) Regularization (DENSE, N={results.n_tasks})"
    elif results.data_source == "sparse":
        title = f"L2 on log(sigma) Regularization (SPARSE, N={results.n_tasks})"
    else:
        title = f"L2 on log(sigma) Regularization (REAL DATA, N={results.n_tasks})"

    plt.suptitle(title, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Regularization path analysis")
    parser.add_argument("--sparse", action="store_true", help="Use sparse (AL-style) synthetic data")
    parser.add_argument("--real", action="store_true", help="Use real data from cache")
    parser.add_argument("--both", action="store_true", help="Run on both dense and sparse synthetic data")
    parser.add_argument("--all", action="store_true", help="Run on dense, sparse, and real data")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.all:
        print("Running dense synthetic...")
        synthetic_results = run_regularization_path_synthetic()
        plot_regularization_path(synthetic_results, OUTPUT_DIR / "plot_020326_regularization_dense.png")

        print("\nRunning sparse synthetic...")
        sparse_results = run_regularization_path_sparse()
        plot_regularization_path(sparse_results, OUTPUT_DIR / "plot_020326_regularization_sparse.png")

        print("\nRunning real data...")
        real_results = run_regularization_path_real()
        plot_regularization_path(real_results, OUTPUT_DIR / "plot_020326_regularization_real.png")

    elif args.both:
        print("Running dense synthetic...")
        synthetic_results = run_regularization_path_synthetic()
        plot_regularization_path(synthetic_results, OUTPUT_DIR / "plot_020326_regularization_dense.png")

        print("\nRunning sparse synthetic...")
        sparse_results = run_regularization_path_sparse()
        plot_regularization_path(sparse_results, OUTPUT_DIR / "plot_020326_regularization_sparse.png")

    elif args.real:
        real_results = run_regularization_path_real()
        plot_regularization_path(real_results, OUTPUT_DIR / "plot_020326_regularization_real.png")

    elif args.sparse:
        sparse_results = run_regularization_path_sparse()
        plot_regularization_path(sparse_results, OUTPUT_DIR / "plot_020326_regularization_sparse.png")

    else:
        synthetic_results = run_regularization_path_synthetic()
        plot_regularization_path(synthetic_results, OUTPUT_DIR / "plot_020326_regularization_dense.png")


if __name__ == "__main__":
    main()
