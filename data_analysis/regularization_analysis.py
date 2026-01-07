"""Analyze L2 variance regularization effect on Thurstonian model.

Run: python data_analysis/regularization_analysis.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from src.preferences.ranking.thurstonian import PairwiseData, fit_thurstonian
from src.preferences.ranking.utils import simulate_pairwise_comparisons
from src.task_data import Task, OriginDataset


OUTPUT_DIR = Path(__file__).parent / "plots" / "regularization"


def make_task(id: str) -> Task:
    return Task(prompt=f"Task {id}", origin=OriginDataset.ALPACA, id=id, metadata={})


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

    # Get all pairs with comparisons
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if wins[i, j] + wins[j, i] > 0]

    # Randomly select pairs for test
    n_test_pairs = int(len(pairs) * test_frac)
    test_pair_indices = rng.choice(len(pairs), size=n_test_pairs, replace=False)

    for idx in test_pair_indices:
        i, j = pairs[idx]
        # Move all comparisons for this pair to test
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


def run_regularization_path(
    n_tasks: int = 30,
    n_comparisons_per_pair: int = 10,
    test_frac: float = 0.2,
    n_splits: int = 10,
):
    """Run regularization path analysis on synthetic data, averaged over multiple splits."""
    # Generate synthetic data once
    data_rng = np.random.default_rng(42)
    true_mu = np.linspace(-3, 3, n_tasks)
    true_sigma = np.ones(n_tasks)
    tasks = [make_task(f"t{i}") for i in range(n_tasks)]
    wins = simulate_pairwise_comparisons(true_mu, true_sigma, n_comparisons_per_pair, data_rng)

    # Fit at different lambda values
    lambdas = np.logspace(-2, np.log10(200), 12)

    # Accumulate results across splits
    all_train_nlls = {lam: [] for lam in lambdas}
    all_test_nlls = {lam: [] for lam in lambdas}
    all_sigma_maxs = {lam: [] for lam in lambdas}
    all_sigma_stds = {lam: [] for lam in lambdas}

    for split_idx in range(n_splits):
        split_rng = np.random.default_rng(split_idx * 1000 + 123)  # Different seed per split
        train_wins, test_wins = split_wins(wins, test_frac, split_rng)
        n_train = int(train_wins.sum())
        n_test = int(test_wins.sum())

        print(f"Split {split_idx + 1}/{n_splits}: train={n_train}, test={n_test}")

        for lam in lambdas:
            data = PairwiseData(tasks=tasks, wins=train_wins)
            result = fit_thurstonian(data, lambda_sigma=lam, log_sigma_bounds=(-4, 4))
            train_nll_per_comp = eval_nll(result.mu, result.sigma, train_wins) / n_train
            test_nll_per_comp = eval_nll(result.mu, result.sigma, test_wins) / n_test
            all_train_nlls[lam].append(train_nll_per_comp)
            all_test_nlls[lam].append(test_nll_per_comp)
            all_sigma_maxs[lam].append(float(result.sigma.max()))
            all_sigma_stds[lam].append(float(result.sigma.std()))

    # Average across splits
    train_nlls = [np.mean(all_train_nlls[lam]) for lam in lambdas]
    test_nlls = [np.mean(all_test_nlls[lam]) for lam in lambdas]
    train_nlls_std = [np.std(all_train_nlls[lam]) for lam in lambdas]
    test_nlls_std = [np.std(all_test_nlls[lam]) for lam in lambdas]
    sigma_maxs = [np.mean(all_sigma_maxs[lam]) for lam in lambdas]
    sigma_stds = [np.mean(all_sigma_stds[lam]) for lam in lambdas]

    print("\nAveraged results:")
    for i, lam in enumerate(lambdas):
        print(f"λ={lam:.4f}: train={train_nlls[i]:.4f}±{train_nlls_std[i]:.4f}, test={test_nlls[i]:.4f}±{test_nlls_std[i]:.4f}, max(σ)={sigma_maxs[i]:.3f}")

    return lambdas, train_nlls, test_nlls, train_nlls_std, test_nlls_std, sigma_maxs, sigma_stds


def plot_regularization_path(
    lambdas: np.ndarray,
    train_nlls: list[float],
    test_nlls: list[float],
    train_nlls_std: list[float],
    test_nlls_std: list[float],
    sigma_maxs: list[float],
    sigma_stds: list[float],
    output_path: Path,
):
    """Plot train/test NLL and sigma statistics vs lambda."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: Train vs Test NLL with error bars
    axes[0].errorbar(lambdas, train_nlls, yerr=train_nlls_std, fmt="o-", label="Train", markersize=5, capsize=3)
    axes[0].errorbar(lambdas, test_nlls, yerr=test_nlls_std, fmt="s-", label="Test", markersize=5, capsize=3)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("λ (regularization strength)")
    axes[0].set_ylabel("NLL per comparison")
    axes[0].legend()
    axes[0].set_title("Regularization Path (mean ± std)")

    # Panel 2: Sigma max
    axes[1].semilogx(lambdas, sigma_maxs, "o-", color="coral", markersize=6)
    axes[1].axhline(y=1.0, color="k", linestyle="--", alpha=0.5, label="σ=1")
    axes[1].set_xlabel("λ")
    axes[1].set_ylabel("max(σ)")
    axes[1].legend()
    axes[1].set_title("Maximum Variance Parameter")

    # Panel 3: Sigma std
    axes[2].semilogx(lambdas, sigma_stds, "s-", color="seagreen", markersize=6)
    axes[2].set_xlabel("λ")
    axes[2].set_ylabel("std(σ)")
    axes[2].set_title("Variance Parameter Spread")

    plt.suptitle("L2 Variance Regularization Analysis", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Running regularization path analysis...")
    lambdas, train_nlls, test_nlls, train_nlls_std, test_nlls_std, sigma_maxs, sigma_stds = run_regularization_path()

    plot_regularization_path(
        lambdas, train_nlls, test_nlls, train_nlls_std, test_nlls_std, sigma_maxs, sigma_stds,
        OUTPUT_DIR / "regularization_path.png",
    )

    # Find optimal lambda (lowest test NLL)
    best_idx = np.argmin(test_nlls)
    print(f"\nBest λ = {lambdas[best_idx]:.4f} (test NLL = {test_nlls[best_idx]:.4f} ± {test_nlls_std[best_idx]:.4f})")


if __name__ == "__main__":
    main()
