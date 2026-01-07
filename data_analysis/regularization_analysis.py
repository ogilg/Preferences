"""Analyze L2 variance regularization effect on Thurstonian model.

Run:
    python -m data_analysis.regularization_analysis          # synthetic data
    python -m data_analysis.regularization_analysis --real   # real data from results/binary/
    python -m data_analysis.regularization_analysis --both   # both
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.stats import norm

from src.preferences.ranking.thurstonian import PairwiseData, fit_thurstonian
from src.preferences.ranking.utils import simulate_pairwise_comparisons
from src.task_data import Task, OriginDataset


OUTPUT_DIR = Path(__file__).parent / "plots" / "regularization"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "binary"

LAMBDAS = np.logspace(-2, np.log10(200), 12)


def make_task(id: str) -> Task:
    return Task(prompt=f"Task {id}", origin=OriginDataset.ALPACA, id=id, metadata={})


def load_all_datasets() -> list[tuple[str, PairwiseData]]:
    """Load all measurement data from results/binary/ directory."""
    datasets = []
    if not RESULTS_DIR.exists():
        return datasets

    for result_dir in sorted(RESULTS_DIR.iterdir()):
        if not result_dir.is_dir():
            continue
        measurements_path = result_dir / "measurements.yaml"
        if not measurements_path.exists():
            continue

        with open(measurements_path) as f:
            measurements = yaml.safe_load(f)
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


def split_wins(
    wins: np.ndarray,
    test_frac: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out entire pairs (edges) for test. Returns (train_wins, test_wins).

    Each pair (i,j) is randomly assigned to test with probability test_frac.
    All comparisons for that pair go to either train or test, not split.
    """
    n = wins.shape[0]
    train = wins.copy()
    test = np.zeros_like(wins)

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if wins[i, j] + wins[j, i] > 0]

    n_test_pairs = int(len(pairs) * test_frac)
    test_pair_indices = rng.choice(len(pairs), size=n_test_pairs, replace=False)

    for idx in test_pair_indices:
        i, j = pairs[idx]
        test[i, j] = wins[i, j]
        test[j, i] = wins[j, i]
        train[i, j] = 0
        train[j, i] = 0

    return train, test


def eval_nll(mu: np.ndarray, sigma: np.ndarray, wins: np.ndarray) -> float:
    """Compute NLL on wins matrix using given parameters."""
    mu_diff = mu[:, np.newaxis] - mu[np.newaxis, :]
    scale = np.sqrt(sigma[:, np.newaxis] ** 2 + sigma[np.newaxis, :] ** 2)
    p = norm.sf(0, loc=mu_diff, scale=scale)
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
    sigma_stds: list[float]
    data_source: str
    n_datasets: int
    n_splits: int


def run_regularization_on_wins(
    tasks: list[Task],
    wins: np.ndarray,
    test_frac: float,
    n_splits: int,
    lambdas: np.ndarray,
) -> dict[str, list]:
    """Run regularization path on a single wins matrix with multiple train/test splits."""
    all_train_nlls = {lam: [] for lam in lambdas}
    all_test_nlls = {lam: [] for lam in lambdas}
    all_sigma_maxs = {lam: [] for lam in lambdas}
    all_sigma_stds = {lam: [] for lam in lambdas}

    for split_idx in range(n_splits):
        split_rng = np.random.default_rng(split_idx * 1000 + 123)
        train_wins, test_wins = split_wins(wins, test_frac, split_rng)
        n_train = int(train_wins.sum())
        n_test = int(test_wins.sum())

        if n_train == 0 or n_test == 0:
            continue

        for lam in lambdas:
            data = PairwiseData(tasks=tasks, wins=train_wins)
            result = fit_thurstonian(data, lambda_sigma=lam, log_sigma_bounds=(-4, 4))
            train_nll_per_comp = eval_nll(result.mu, result.sigma, train_wins) / n_train
            test_nll_per_comp = eval_nll(result.mu, result.sigma, test_wins) / n_test
            all_train_nlls[lam].append(train_nll_per_comp)
            all_test_nlls[lam].append(test_nll_per_comp)
            all_sigma_maxs[lam].append(float(result.sigma.max()))
            all_sigma_stds[lam].append(float(result.sigma.std()))

    return {
        "train_nlls": all_train_nlls,
        "test_nlls": all_test_nlls,
        "sigma_maxs": all_sigma_maxs,
        "sigma_stds": all_sigma_stds,
    }


def run_regularization_path_synthetic(
    n_tasks: int = 30,
    n_comparisons_per_pair: int = 10,
    test_frac: float = 0.2,
    n_splits: int = 10,
) -> RegularizationResults:
    """Run regularization path analysis on synthetic data."""
    print("Running regularization path on SYNTHETIC data...")

    data_rng = np.random.default_rng(42)
    true_mu = np.linspace(-3, 3, n_tasks)
    true_sigma = np.ones(n_tasks)
    tasks = [make_task(f"t{i}") for i in range(n_tasks)]
    wins = simulate_pairwise_comparisons(true_mu, true_sigma, n_comparisons_per_pair, data_rng)

    print(f"  Tasks: {n_tasks}, Comparisons/pair: {n_comparisons_per_pair}, Splits: {n_splits}")

    results = run_regularization_on_wins(tasks, wins, test_frac, n_splits, LAMBDAS)

    train_nlls = [np.mean(results["train_nlls"][lam]) for lam in LAMBDAS]
    test_nlls = [np.mean(results["test_nlls"][lam]) for lam in LAMBDAS]
    train_nlls_std = [np.std(results["train_nlls"][lam]) for lam in LAMBDAS]
    test_nlls_std = [np.std(results["test_nlls"][lam]) for lam in LAMBDAS]
    sigma_maxs = [np.mean(results["sigma_maxs"][lam]) for lam in LAMBDAS]
    sigma_stds = [np.mean(results["sigma_stds"][lam]) for lam in LAMBDAS]

    print("\nResults:")
    for i, lam in enumerate(LAMBDAS):
        print(f"  λ={lam:.4f}: train={train_nlls[i]:.4f}±{train_nlls_std[i]:.4f}, test={test_nlls[i]:.4f}±{test_nlls_std[i]:.4f}, max(σ)={sigma_maxs[i]:.3f}")

    return RegularizationResults(
        lambdas=LAMBDAS,
        train_nlls=train_nlls,
        test_nlls=test_nlls,
        train_nlls_std=train_nlls_std,
        test_nlls_std=test_nlls_std,
        sigma_maxs=sigma_maxs,
        sigma_stds=sigma_stds,
        data_source="synthetic",
        n_datasets=1,
        n_splits=n_splits,
    )


def run_regularization_path_real(
    test_frac: float = 0.2,
    n_splits: int = 5,
) -> RegularizationResults:
    """Run regularization path analysis on real data from results/binary/."""
    print("Running regularization path on REAL data...")

    datasets = load_all_datasets()
    if not datasets:
        raise ValueError(f"No datasets found in {RESULTS_DIR}")

    print(f"  Found {len(datasets)} datasets, using {n_splits} splits per dataset")

    all_train_nlls = {lam: [] for lam in LAMBDAS}
    all_test_nlls = {lam: [] for lam in LAMBDAS}
    all_sigma_maxs = {lam: [] for lam in LAMBDAS}
    all_sigma_stds = {lam: [] for lam in LAMBDAS}

    for name, data in datasets:
        if data.n_tasks < 5:
            print(f"  Skipping {name} (only {data.n_tasks} tasks)")
            continue

        print(f"  Processing {name} ({data.n_tasks} tasks, {int(data.wins.sum())} comparisons)...")
        results = run_regularization_on_wins(data.tasks, data.wins, test_frac, n_splits, LAMBDAS)

        for lam in LAMBDAS:
            all_train_nlls[lam].extend(results["train_nlls"][lam])
            all_test_nlls[lam].extend(results["test_nlls"][lam])
            all_sigma_maxs[lam].extend(results["sigma_maxs"][lam])
            all_sigma_stds[lam].extend(results["sigma_stds"][lam])

    if not all_train_nlls[LAMBDAS[0]]:
        raise ValueError("No valid datasets found (need at least 5 tasks)")

    train_nlls = [np.mean(all_train_nlls[lam]) for lam in LAMBDAS]
    test_nlls = [np.mean(all_test_nlls[lam]) for lam in LAMBDAS]
    train_nlls_std = [np.std(all_train_nlls[lam]) for lam in LAMBDAS]
    test_nlls_std = [np.std(all_test_nlls[lam]) for lam in LAMBDAS]
    sigma_maxs = [np.mean(all_sigma_maxs[lam]) for lam in LAMBDAS]
    sigma_stds = [np.mean(all_sigma_stds[lam]) for lam in LAMBDAS]

    n_valid_datasets = len([d for _, d in datasets if d.n_tasks >= 5])

    print("\nAggregated results:")
    for i, lam in enumerate(LAMBDAS):
        print(f"  λ={lam:.4f}: train={train_nlls[i]:.4f}±{train_nlls_std[i]:.4f}, test={test_nlls[i]:.4f}±{test_nlls_std[i]:.4f}, max(σ)={sigma_maxs[i]:.3f}")

    return RegularizationResults(
        lambdas=LAMBDAS,
        train_nlls=train_nlls,
        test_nlls=test_nlls,
        train_nlls_std=train_nlls_std,
        test_nlls_std=test_nlls_std,
        sigma_maxs=sigma_maxs,
        sigma_stds=sigma_stds,
        data_source="real",
        n_datasets=n_valid_datasets,
        n_splits=n_splits,
    )


def plot_regularization_path(results: RegularizationResults, output_path: Path):
    """Plot train/test NLL and sigma statistics vs lambda."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].errorbar(results.lambdas, results.train_nlls, yerr=results.train_nlls_std, fmt="o-", label="Train", markersize=5, capsize=3)
    axes[0].errorbar(results.lambdas, results.test_nlls, yerr=results.test_nlls_std, fmt="s-", label="Test", markersize=5, capsize=3)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("λ (regularization strength)")
    axes[0].set_ylabel("NLL per comparison")
    axes[0].legend()
    axes[0].set_title("Regularization Path (mean ± std)")

    axes[1].semilogx(results.lambdas, results.sigma_maxs, "o-", color="coral", markersize=6)
    axes[1].axhline(y=1.0, color="k", linestyle="--", alpha=0.5, label="σ=1")
    axes[1].set_xlabel("λ")
    axes[1].set_ylabel("max(σ)")
    axes[1].legend()
    axes[1].set_title("Maximum Variance Parameter")

    axes[2].semilogx(results.lambdas, results.sigma_stds, "s-", color="seagreen", markersize=6)
    axes[2].set_xlabel("λ")
    axes[2].set_ylabel("std(σ)")
    axes[2].set_title("Variance Parameter Spread")

    if results.data_source == "synthetic":
        title = "L2 Variance Regularization Analysis (SYNTHETIC)"
    else:
        title = f"L2 Variance Regularization Analysis (REAL, n={results.n_datasets} datasets)"

    plt.suptitle(title, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def print_best_lambda(results: RegularizationResults):
    """Print the optimal lambda based on test NLL."""
    best_idx = int(np.argmin(results.test_nlls))
    print(f"\nBest λ = {results.lambdas[best_idx]:.4f} (test NLL = {results.test_nlls[best_idx]:.4f} ± {results.test_nlls_std[best_idx]:.4f})")


def main():
    parser = argparse.ArgumentParser(description="Regularization path analysis")
    parser.add_argument("--real", action="store_true", help="Use real data from results/binary/")
    parser.add_argument("--both", action="store_true", help="Run on both synthetic and real data")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.both:
        # Run both synthetic and real
        synthetic_results = run_regularization_path_synthetic()
        plot_regularization_path(synthetic_results, OUTPUT_DIR / "regularization_path_synthetic.png")
        print_best_lambda(synthetic_results)

        print("\n" + "=" * 60 + "\n")

        real_results = run_regularization_path_real()
        plot_regularization_path(real_results, OUTPUT_DIR / "regularization_path_real.png")
        print_best_lambda(real_results)

    elif args.real:
        real_results = run_regularization_path_real()
        plot_regularization_path(real_results, OUTPUT_DIR / "regularization_path_real.png")
        print_best_lambda(real_results)

    else:
        # Default: synthetic
        synthetic_results = run_regularization_path_synthetic()
        plot_regularization_path(synthetic_results, OUTPUT_DIR / "regularization_path_synthetic.png")
        print_best_lambda(synthetic_results)


if __name__ == "__main__":
    main()
