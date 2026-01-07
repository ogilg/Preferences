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
    """Per-pair proportional split. Returns (train_wins, test_wins).

    Each individual comparison is randomly assigned to test with probability test_frac.
    """
    n = wins.shape[0]
    train = np.zeros_like(wins)
    test = np.zeros_like(wins)
    for i in range(n):
        for j in range(n):
            total = wins[i, j]
            if total > 0:
                # Each comparison independently goes to test with prob test_frac
                n_test = rng.binomial(total, test_frac)
                test[i, j] = n_test
                train[i, j] = total - n_test
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
    seed: int = 42,
):
    """Run regularization path analysis on synthetic data."""
    rng = np.random.default_rng(seed)

    # Generate synthetic data
    true_mu = np.linspace(-3, 3, n_tasks)
    true_sigma = np.ones(n_tasks)
    tasks = [make_task(f"t{i}") for i in range(n_tasks)]
    wins = simulate_pairwise_comparisons(true_mu, true_sigma, n_comparisons_per_pair, rng)

    # Split into train/test
    train_wins, test_wins = split_wins(wins, test_frac, rng)
    n_train = int(train_wins.sum())
    n_test = int(test_wins.sum())
    print(f"Train comparisons: {n_train}, Test comparisons: {n_test}")

    # Fit at different lambda values
    lambdas = np.logspace(-2, 2, 10)
    train_nlls, test_nlls, sigma_maxs, sigma_stds = [], [], [], []

    for lam in lambdas:
        data = PairwiseData(tasks=tasks, wins=train_wins)
        result = fit_thurstonian(data, lambda_sigma=lam)
        # Normalize by number of comparisons for fair comparison
        train_nll_per_comp = eval_nll(result.mu, result.sigma, train_wins) / n_train
        test_nll_per_comp = eval_nll(result.mu, result.sigma, test_wins) / n_test
        train_nlls.append(train_nll_per_comp)
        test_nlls.append(test_nll_per_comp)
        sigma_maxs.append(float(result.sigma.max()))
        sigma_stds.append(float(result.sigma.std()))
        print(f"λ={lam:.4f}: train_nll/comp={train_nll_per_comp:.4f}, test_nll/comp={test_nll_per_comp:.4f}, max(σ)={sigma_maxs[-1]:.3f}")

    return lambdas, train_nlls, test_nlls, sigma_maxs, sigma_stds


def plot_regularization_path(
    lambdas: np.ndarray,
    train_nlls: list[float],
    test_nlls: list[float],
    sigma_maxs: list[float],
    sigma_stds: list[float],
    output_path: Path,
):
    """Plot train/test NLL and sigma statistics vs lambda."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: Train vs Test NLL (normalized per comparison)
    axes[0].semilogx(lambdas, train_nlls, "o-", label="Train", markersize=6)
    axes[0].semilogx(lambdas, test_nlls, "s-", label="Test", markersize=6)
    axes[0].set_xlabel("λ (regularization strength)")
    axes[0].set_ylabel("NLL per comparison")
    axes[0].legend()
    axes[0].set_title("Regularization Path")

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
    lambdas, train_nlls, test_nlls, sigma_maxs, sigma_stds = run_regularization_path()

    plot_regularization_path(
        lambdas, train_nlls, test_nlls, sigma_maxs, sigma_stds,
        OUTPUT_DIR / "regularization_path.png",
    )

    # Find optimal lambda (lowest test NLL)
    best_idx = np.argmin(test_nlls)
    print(f"\nBest λ = {lambdas[best_idx]:.4f} (test NLL = {test_nlls[best_idx]:.2f})")


if __name__ == "__main__":
    main()
