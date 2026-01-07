"""Analysis of how gradient norm and loss scale with number of tasks.

Run: python -m data_analysis.tolerance_scaling
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.preferences.ranking.thurstonian import (
    PairwiseData,
    fit_thurstonian,
    _preference_prob,
)
from src.task_data import Task, OriginDataset


OUTPUT_DIR = Path(__file__).parent / "plots" / "tolerance_scaling"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "binary"


def make_task(id: str) -> Task:
    return Task(prompt=f"Task {id}", origin=OriginDataset.ALPACA, id=id, metadata={})


def simulate_comparisons(
    mu: np.ndarray,
    sigma: np.ndarray,
    n_per_pair: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(mu)
    wins = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(i + 1, n):
            p = _preference_prob(mu[i], mu[j], sigma[i], sigma[j])
            for _ in range(n_per_pair):
                if rng.random() < p:
                    wins[i, j] += 1
                else:
                    wins[j, i] += 1
    return wins


@dataclass
class ScalingResult:
    n_tasks: int
    total_comparisons: int
    final_loss: float
    final_gradient_norm: float
    label: str = ""


def fit_and_measure(data: PairwiseData) -> ScalingResult:
    """Fit model and return scaling metrics."""
    n = data.n_tasks
    n_params = (n - 1) + n
    max_iter = max(5000, n_params * 100)

    result = fit_thurstonian(
        data,
        max_iter=max_iter,
        gradient_tol=1e-12,
        loss_tol=1e-15,
    )

    total_comps = int(data.wins.sum())

    return ScalingResult(
        n_tasks=n,
        total_comparisons=total_comps,
        final_loss=result.neg_log_likelihood,
        final_gradient_norm=result.gradient_norm,
    )


def run_synthetic(n_tasks: int, n_comparisons_per_pair: int, rng: np.random.Generator) -> ScalingResult:
    """Generate synthetic data and fit."""
    true_mu = np.linspace(-3, 3, n_tasks)
    true_sigma = np.ones(n_tasks)
    tasks = [make_task(f"t{i}") for i in range(n_tasks)]
    wins = simulate_comparisons(true_mu, true_sigma, n_comparisons_per_pair, rng)
    data = PairwiseData(tasks=tasks, wins=wins)
    return fit_and_measure(data)


def load_real_results(results_dir: Path = RESULTS_DIR) -> list[ScalingResult]:
    """Load results from saved runs (no refitting, just read stored values)."""
    results = []

    # Handle both flat (old) and nested (new: n{k}/run_dir) structures
    for item in sorted(results_dir.iterdir()):
        if item.is_dir() and item.name.startswith("n"):
            # New nested structure: n10/, n20/, etc.
            run_dirs = list(item.iterdir())
        else:
            # Old flat structure
            run_dirs = [item]

        for run_dir in run_dirs:
            config_path = run_dir / "config.yaml"
            thurstonian_path = run_dir / "thurstonian.yaml"
            measurements_path = run_dir / "measurements.yaml"

            if not all(p.exists() for p in [config_path, thurstonian_path, measurements_path]):
                continue

            with open(config_path) as f:
                config = yaml.safe_load(f)

            with open(thurstonian_path) as f:
                thurstonian = yaml.safe_load(f)

            with open(measurements_path) as f:
                measurements = yaml.safe_load(f)

            results.append(ScalingResult(
                n_tasks=config["n_tasks"],
                total_comparisons=len(measurements),
                final_loss=thurstonian["neg_log_likelihood"],
                final_gradient_norm=thurstonian["gradient_norm"],
                label=run_dir.name,
            ))

    return results


def plot_scaling(
    synthetic_results: list[ScalingResult],
    real_results: list[ScalingResult] | None = None,
    output_path: Path | None = None,
):
    """Plot loss and gradient norm vs n_tasks."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Synthetic
    n_vals = [r.n_tasks for r in synthetic_results]
    axes[0].plot(n_vals, [r.final_loss for r in synthetic_results], 'o-', label='Synthetic')
    axes[1].semilogy(n_vals, [r.final_gradient_norm for r in synthetic_results], 'o-', label='Synthetic')

    # Real (if provided)
    if real_results:
        n_vals_real = [r.n_tasks for r in real_results]
        axes[0].plot(n_vals_real, [r.final_loss for r in real_results], 's', alpha=0.5, label='Real')
        axes[1].semilogy(n_vals_real, [r.final_gradient_norm for r in real_results], 's', alpha=0.5, label='Real')

    axes[0].set_xlabel("n_tasks")
    axes[0].set_ylabel("Final NLL")
    axes[0].set_title("Loss vs N")
    axes[0].legend()

    axes[1].set_xlabel("n_tasks")
    axes[1].set_ylabel("Final Gradient Norm")
    axes[1].set_title("Gradient Norm vs N")
    axes[1].legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def print_table(results: list[ScalingResult], label: str = ""):
    """Print summary table."""
    if label:
        print(f"\n{label}")
    print(f"{'N':>6} {'comps':>10} {'loss':>12} {'grad':>12} {'loss/comp':>12}")
    print("-" * 58)
    for r in results:
        print(f"{r.n_tasks:>6} {r.total_comparisons:>10} {r.final_loss:>12.2f} "
              f"{r.final_gradient_norm:>12.2e} {r.final_loss/r.total_comparisons:>12.4f}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Synthetic
    rng = np.random.default_rng(42)
    n_tasks_values = [5, 10, 20, 30, 50, 75, 100]
    n_comparisons_per_pair = 10

    print("Running synthetic experiments...")
    synthetic_results = []
    for n in n_tasks_values:
        print(f"  n_tasks={n}")
        r = run_synthetic(n, n_comparisons_per_pair, rng)
        synthetic_results.append(r)

    print_table(synthetic_results, "SYNTHETIC DATA")

    # Real
    real_results = []
    if RESULTS_DIR.exists():
        print("\nLoading real data from results/binary/...")
        real_results = load_real_results(RESULTS_DIR)
        if real_results:
            print_table(real_results, "REAL DATA")
        else:
            print("  No runs found")

    plot_scaling(synthetic_results, real_results or None, output_path=OUTPUT_DIR / "scaling.png")


if __name__ == "__main__":
    main()
