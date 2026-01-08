"""Thurstonian model analysis and tests.

Tests (run with pytest):
    pytest data_analysis/thurstonian_analysis.py

Real data analysis:
    python -m data_analysis.thurstonian_analysis          # analyze real data from results/binary/
    python -m data_analysis.thurstonian_analysis --synthetic  # run synthetic diagnostics only
"""

import argparse
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.preferences.ranking.thurstonian import (
    PairwiseData,
    ThurstonianResult,
    fit_thurstonian,
    _preference_prob,
    _neg_log_likelihood,
    OptimizationHistory,
)
from src.preferences.ranking.plotting import normalize_mu_for_comparison
from src.task_data import Task, OriginDataset
from src.types import BinaryPreferenceMeasurement, PreferenceType


OUTPUT_DIR = Path(__file__).parent / "plots" / "thurstonian"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "binary"

# Number of tasks. Synthetic uses exactly this; real data filters to >= this.
N_TASKS = 50


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


def plot_real_data_diagnostics(
    result: ThurstonianResult,
    data: PairwiseData,
    name: str,
    output_dir: Path,
) -> None:
    """Plot diagnostics for real data (no ground truth comparison)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Panel 1: Loss curve
    axes[0, 0].plot(result.history.loss)
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("NLL")
    axes[0, 0].set_title(f"Loss curve (final={result.history.loss[-1]:.2f})")

    # Panel 2: Sigma max trajectory
    axes[0, 1].plot(result.history.sigma_max)
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("max(σ)")
    axes[0, 1].set_title(f"σ_max trajectory (final={result.sigma.max():.2f})")

    # Panel 3: Fitted μ distribution
    axes[0, 2].hist(result.mu, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 2].axvline(0, color='k', linestyle='--', alpha=0.5)
    axes[0, 2].set_xlabel("μ (utility)")
    axes[0, 2].set_ylabel("Count")
    axes[0, 2].set_title(f"Fitted μ: range=[{result.mu.min():.2f}, {result.mu.max():.2f}]")

    # Panel 4: Fitted σ distribution
    axes[1, 0].hist(result.sigma, bins=20, edgecolor='black', alpha=0.7, color='coral')
    axes[1, 0].axvline(1.0, color='k', linestyle='--', alpha=0.5, label='σ=1')
    axes[1, 0].set_xlabel("σ (uncertainty)")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].legend()
    axes[1, 0].set_title(f"Fitted σ: mean={result.sigma.mean():.2f}, std={result.sigma.std():.2f}")

    # Panel 5: μ vs σ scatter
    axes[1, 1].scatter(result.mu, result.sigma, alpha=0.6)
    axes[1, 1].set_xlabel("μ (utility)")
    axes[1, 1].set_ylabel("σ (uncertainty)")
    corr = np.corrcoef(result.mu, result.sigma)[0, 1]
    axes[1, 1].set_title(f"μ-σ correlation: r={corr:.3f}")

    # Panel 6: Comparisons per task
    comparisons_per_task = data.wins.sum(axis=1) + data.wins.sum(axis=0)
    axes[1, 2].bar(range(len(comparisons_per_task)), sorted(comparisons_per_task, reverse=True), alpha=0.7)
    axes[1, 2].set_xlabel("Task (sorted)")
    axes[1, 2].set_ylabel("Total comparisons")
    axes[1, 2].set_title(f"Comparisons/task: mean={comparisons_per_task.mean():.1f}")

    n_comparisons = int(data.wins.sum())
    fig.suptitle(
        f"{name} (REAL DATA)\n"
        f"n_tasks={data.n_tasks}, n_comparisons={n_comparisons}, "
        f"converged={result.converged}, n_iter={result.n_iterations}, "
        f"grad_norm={result.gradient_norm:.2e}"
    )
    plt.tight_layout()
    plt.savefig(output_dir / f"real_{name}.png", dpi=100)
    plt.close()


def plot_real_data_summary(
    all_results: list[tuple[str, ThurstonianResult, PairwiseData]],
    output_dir: Path,
) -> None:
    """Plot summary across all real datasets."""
    import matplotlib.pyplot as plt

    if not all_results:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    names = [name for name, _, _ in all_results]

    # Panel 1: Convergence status
    converged = [r.converged for _, r, _ in all_results]
    grad_norms = [r.gradient_norm for _, r, _ in all_results]
    colors = ['green' if c else 'red' for c in converged]
    axes[0, 0].barh(range(len(names)), grad_norms, color=colors, alpha=0.7)
    axes[0, 0].set_yticks(range(len(names)))
    axes[0, 0].set_yticklabels([n[:30] for n in names], fontsize=8)
    axes[0, 0].set_xlabel("Gradient norm")
    axes[0, 0].axvline(1.0, color='k', linestyle='--', alpha=0.5)
    axes[0, 0].set_title(f"Convergence ({sum(converged)}/{len(converged)} converged)")

    # Panel 2: σ statistics across datasets
    sigma_means = [r.sigma.mean() for _, r, _ in all_results]
    sigma_stds = [r.sigma.std() for _, r, _ in all_results]
    x = range(len(names))
    axes[0, 1].errorbar(x, sigma_means, yerr=sigma_stds, fmt='o', capsize=3, alpha=0.7)
    axes[0, 1].axhline(1.0, color='k', linestyle='--', alpha=0.5, label='σ=1')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([n[:15] for n in names], rotation=45, ha='right', fontsize=7)
    axes[0, 1].set_ylabel("σ (mean ± std)")
    axes[0, 1].legend()
    axes[0, 1].set_title("Uncertainty parameters across datasets")

    # Panel 3: μ range across datasets
    mu_mins = [r.mu.min() for _, r, _ in all_results]
    mu_maxs = [r.mu.max() for _, r, _ in all_results]
    mu_ranges = [max - min for min, max in zip(mu_mins, mu_maxs)]
    axes[1, 0].bar(x, mu_ranges, alpha=0.7, color='steelblue')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([n[:15] for n in names], rotation=45, ha='right', fontsize=7)
    axes[1, 0].set_ylabel("μ range (max - min)")
    axes[1, 0].set_title("Utility spread across datasets")

    # Panel 4: NLL per comparison
    nll_per_comp = [r.neg_log_likelihood / d.wins.sum() for _, r, d in all_results]
    axes[1, 1].bar(x, nll_per_comp, alpha=0.7, color='seagreen')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([n[:15] for n in names], rotation=45, ha='right', fontsize=7)
    axes[1, 1].set_ylabel("NLL / comparison")
    axes[1, 1].set_title("Model fit quality")

    plt.suptitle(f"Thurstonian Analysis Summary (n={len(all_results)} datasets, REAL DATA, N_TASKS>={N_TASKS})", fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "real_summary.png", dpi=150)
    plt.close()


def run_real_data_analysis():
    """Run Thurstonian analysis on all real datasets."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = load_all_datasets()
    if not datasets:
        return

    all_results = []
    for name, data in datasets:
        if data.n_tasks < N_TASKS:
            continue

        result = fit_thurstonian(data, max_iter=3000)
        plot_real_data_diagnostics(result, data, name, OUTPUT_DIR)
        all_results.append((name, result, data))

    if all_results:
        plot_real_data_summary(all_results, OUTPUT_DIR)


def run_synthetic_diagnostics():
    """Run synthetic data diagnostics (same as pytest but standalone)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    configs = [
        (f"{N_TASKS}_tasks_dense", N_TASKS, 20, np.ones(N_TASKS), 0.0),
        (f"{N_TASKS}_tasks_sparse", N_TASKS, 5, np.ones(N_TASKS), 0.0),
        (f"{N_TASKS}_tasks_varying_sigma", N_TASKS, 20, None, 0.0),  # None = random sigma
        (f"{N_TASKS}_tasks_20pct_noise", N_TASKS, 20, np.ones(N_TASKS), 0.2),
    ]

    for name, n_tasks, n_comp, true_sigma, noise_rate in configs:
        rng = np.random.default_rng(hash(name) % (2**32))
        true_mu = np.linspace(-3, 3, n_tasks)
        if true_sigma is None:
            true_sigma = 0.5 + rng.random(n_tasks) * 1.5

        tasks = [make_task(f"t{i}") for i in range(n_tasks)]
        wins = _simulate_comparisons_with_noise(true_mu, true_sigma, n_comp, noise_rate, rng)
        data = PairwiseData(tasks=tasks, wins=wins)

        result = fit_thurstonian(data, max_iter=3000)
        _plot_synthetic_diagnostics(result, true_mu, true_sigma, name, OUTPUT_DIR)


def _simulate_comparisons_with_noise(
    mu: np.ndarray,
    sigma: np.ndarray,
    n_comparisons_per_pair: int,
    noise_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate pairwise comparisons with optional noise."""
    n = len(mu)
    wins = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(i + 1, n):
            p_i_beats_j = _preference_prob(mu[i], mu[j], sigma[i], sigma[j])
            for _ in range(n_comparisons_per_pair):
                if noise_rate > 0 and rng.random() < noise_rate:
                    i_wins = rng.random() < 0.5
                else:
                    i_wins = rng.random() < p_i_beats_j
                if i_wins:
                    wins[i, j] += 1
                else:
                    wins[j, i] += 1
    return wins


def _plot_synthetic_diagnostics(
    result: ThurstonianResult,
    true_mu: np.ndarray,
    true_sigma: np.ndarray,
    name: str,
    output_dir: Path,
) -> None:
    """Plot diagnostics for synthetic data with ground truth comparison."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Loss curve
    axes[0, 0].plot(result.history.loss)
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("NLL")
    axes[0, 0].set_title(f"Loss curve (final={result.history.loss[-1]:.2f})")

    # Sigma max over iterations
    axes[0, 1].plot(result.history.sigma_max)
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("max(σ)")
    axes[0, 1].set_title(f"σ_max trajectory (final={result.sigma.max():.2f})")

    # Ranking comparison
    true_rank = np.argsort(np.argsort(-true_mu))
    fitted_rank = np.argsort(np.argsort(-result.mu))
    axes[0, 2].scatter(true_rank, fitted_rank, alpha=0.6)
    axes[0, 2].plot([0, len(true_mu)], [0, len(true_mu)], "k--", alpha=0.5)
    axes[0, 2].set_xlabel("True rank")
    axes[0, 2].set_ylabel("Fitted rank")
    rank_corr = np.corrcoef(true_rank, fitted_rank)[0, 1]
    axes[0, 2].set_title(f"Ranking recovery (r={rank_corr:.3f})")

    # True vs fitted mu
    shifted_true, scaled_fitted_mu, scale = normalize_mu_for_comparison(true_mu, result.mu)
    axes[1, 0].scatter(shifted_true, scaled_fitted_mu, alpha=0.6)
    lims = [min(shifted_true.min(), scaled_fitted_mu.min()), max(shifted_true.max(), scaled_fitted_mu.max())]
    axes[1, 0].plot(lims, lims, "k--", alpha=0.5)
    axes[1, 0].set_xlabel("True μ (shifted)")
    axes[1, 0].set_ylabel("Fitted μ (scaled)")
    corr = np.corrcoef(true_mu, result.mu)[0, 1]
    mae = np.abs(shifted_true - scaled_fitted_mu).mean()
    axes[1, 0].set_title(f"μ recovery: r={corr:.3f}, MAE={mae:.3f}")

    # True vs fitted sigma
    scaled_fitted_sigma = result.sigma * scale
    axes[1, 1].scatter(true_sigma, scaled_fitted_sigma, alpha=0.6)
    sigma_lims = [0, max(true_sigma.max(), scaled_fitted_sigma.max()) * 1.1]
    axes[1, 1].plot(sigma_lims, sigma_lims, "k--", alpha=0.5)
    axes[1, 1].set_xlabel("True σ")
    axes[1, 1].set_ylabel("Fitted σ (scaled)")
    sigma_corr = np.corrcoef(true_sigma, result.sigma)[0, 1] if true_sigma.std() > 0 else float('nan')
    axes[1, 1].set_title(f"σ recovery: r={sigma_corr:.3f}")

    # Fitted sigma distribution
    axes[1, 2].hist(scaled_fitted_sigma, bins=20, edgecolor='black', alpha=0.7)
    axes[1, 2].axvline(true_sigma.mean(), color='r', linestyle='--', label=f'true={true_sigma.mean():.2f}')
    axes[1, 2].axvline(scaled_fitted_sigma.mean(), color='b', linestyle='--', label=f'fitted={scaled_fitted_sigma.mean():.2f}')
    axes[1, 2].set_xlabel("σ (scaled)")
    axes[1, 2].set_ylabel("Count")
    axes[1, 2].legend()
    axes[1, 2].set_title("σ distribution")

    fig.suptitle(
        f"{name} (SYNTHETIC)\n"
        f"converged={result.converged}, n_iter={result.n_iterations}, "
        f"grad_norm={result.gradient_norm:.2e}, scale={1/scale:.2f}x"
    )
    plt.tight_layout()
    plt.savefig(output_dir / f"synthetic_{name}.png", dpi=100)
    plt.close()


# =============================================================================
# PYTEST TESTS (unchanged)
# =============================================================================

class TestPairwiseDataFromComparisons:
    def test_counts_wins_correctly(self):
        tasks = [make_task("a"), make_task("b"), make_task("c")]
        comparisons = [
            BinaryPreferenceMeasurement(tasks[0], tasks[1], "a", PreferenceType.PRE_TASK_STATED),
            BinaryPreferenceMeasurement(tasks[0], tasks[1], "a", PreferenceType.PRE_TASK_STATED),
            BinaryPreferenceMeasurement(tasks[0], tasks[1], "b", PreferenceType.PRE_TASK_STATED),
            BinaryPreferenceMeasurement(tasks[1], tasks[2], "a", PreferenceType.PRE_TASK_STATED),
        ]

        data = PairwiseData.from_comparisons(comparisons, tasks)

        # a beat b twice, b beat a once
        assert data.wins[0, 1] == 2
        assert data.wins[1, 0] == 1
        # b beat c once
        assert data.wins[1, 2] == 1
        assert data.wins[2, 1] == 0
        # a vs c never compared
        assert data.wins[0, 2] == 0
        assert data.wins[2, 0] == 0

    def test_empty_comparisons(self):
        tasks = [make_task("a"), make_task("b")]
        data = PairwiseData.from_comparisons([], tasks)

        assert data.wins.sum() == 0
        assert data.n_tasks == 2

    def test_total_comparisons(self):
        tasks = [make_task("a"), make_task("b"), make_task("c")]
        comparisons = [
            BinaryPreferenceMeasurement(tasks[0], tasks[1], "a", PreferenceType.PRE_TASK_STATED),
            BinaryPreferenceMeasurement(tasks[0], tasks[2], "b", PreferenceType.PRE_TASK_STATED),
            BinaryPreferenceMeasurement(tasks[1], tasks[2], "a", PreferenceType.PRE_TASK_STATED),
        ]

        data = PairwiseData.from_comparisons(comparisons, tasks)

        # task a: 1 win vs b, 1 loss vs c = 2 comparisons
        assert data.total_comparisons(tasks[0]) == 2
        # task b: 1 loss vs a, 1 win vs c = 2 comparisons
        assert data.total_comparisons(tasks[1]) == 2
        # task c: 1 win vs a, 1 loss vs b = 2 comparisons
        assert data.total_comparisons(tasks[2]) == 2


class TestPreferenceProb:
    def test_symmetry(self):
        """P(i > j) + P(j > i) = 1"""
        p_ij = _preference_prob(1.0, 0.5, 1.0, 1.0)
        p_ji = _preference_prob(0.5, 1.0, 1.0, 1.0)
        # Ensure we're not testing with p=0.5 (which would be trivially symmetric)
        assert p_ij != pytest.approx(0.5, abs=0.01)
        assert abs(p_ij + p_ji - 1.0) < 1e-10

    def test_equal_means_gives_half(self):
        """When μ_i = μ_j, P(i > j) = 0.5"""
        p = _preference_prob(1.0, 1.0, 0.5, 0.5)
        assert p == pytest.approx(0.5, abs=1e-10)
        # Verify unequal means don't give 0.5
        p_unequal = _preference_prob(1.0, 0.0, 0.5, 0.5)
        assert p_unequal != pytest.approx(0.5, abs=0.01)

    def test_higher_mean_more_likely(self):
        """Higher μ means higher win probability"""
        p = _preference_prob(2.0, 0.0, 1.0, 1.0)
        # Should be well above 0.5, not just marginally
        assert p > 0.9

    def test_larger_sigma_shrinks_toward_half(self):
        """Larger uncertainty shrinks probability toward 0.5"""
        p_low_sigma = _preference_prob(1.0, 0.0, 0.1, 0.1)
        p_high_sigma = _preference_prob(1.0, 0.0, 10.0, 10.0)
        # Low sigma should be near 1.0
        assert p_low_sigma > 0.99
        # High sigma should be close to 0.5
        assert 0.5 < p_high_sigma < 0.55


class TestNegLogLikelihood:
    def test_perfect_data_lower_nll(self):
        """Parameters matching the data should have lower NLL than mismatched."""
        tasks = [make_task("a"), make_task("b")]
        # a always beats b
        wins = np.array([[0, 10], [0, 0]])
        data = PairwiseData(tasks=tasks, wins=wins)

        # Good params: μ_a > μ_b (μ_0=0 fixed, so μ_1 < 0 means b is worse)
        # Actually: μ_0=0 for task a, μ_1 for task b
        # If a beats b, we want μ_a > μ_b, so μ_1 should be negative
        good_params = np.array([-2.0, 0.0, 0.0])  # μ_1=-2, log_σ=[0,0]
        bad_params = np.array([2.0, 0.0, 0.0])  # μ_1=2, log_σ=[0,0]

        nll_good = _neg_log_likelihood(good_params, wins, 2)
        nll_bad = _neg_log_likelihood(bad_params, wins, 2)

        # Ensure meaningful difference, not just floating point noise
        assert nll_good < nll_bad
        assert nll_bad - nll_good > 1.0  # Should differ substantially

    def test_symmetric_data_prefers_equal_means(self):
        """When a and b are equally matched, μ_a ≈ μ_b should be optimal."""
        wins = np.array([[0, 5], [5, 0]])

        equal_params = np.array([0.0, 0.0, 0.0])  # μ_1=0, log_σ=[0,0]
        unequal_params = np.array([2.0, 0.0, 0.0])  # μ_1=2

        nll_equal = _neg_log_likelihood(equal_params, wins, 2)
        nll_unequal = _neg_log_likelihood(unequal_params, wins, 2)

        assert nll_equal < nll_unequal
        assert nll_unequal - nll_equal > 0.5  # Should differ substantially

    def test_nll_increases_with_wrong_direction(self):
        """NLL should be monotonic as params move away from optimum."""
        wins = np.array([[0, 10], [0, 0]])  # a always beats b

        # μ_1 increasingly positive (wrong direction - says b is better)
        nll_0 = _neg_log_likelihood(np.array([0.0, 0.0, 0.0]), wins, 2)
        nll_1 = _neg_log_likelihood(np.array([1.0, 0.0, 0.0]), wins, 2)
        nll_2 = _neg_log_likelihood(np.array([2.0, 0.0, 0.0]), wins, 2)

        assert nll_0 < nll_1 < nll_2


class TestFitThurstonian:
    def test_clear_winner_has_higher_utility(self):
        """Task that always wins should have highest utility."""
        tasks = [make_task("a"), make_task("b"), make_task("c")]
        # a beats everyone, b beats c, c loses to everyone
        wins = np.array([
            [0, 10, 10],
            [0, 0, 10],
            [0, 0, 0],
        ])
        data = PairwiseData(tasks=tasks, wins=wins)

        result = fit_thurstonian(data)

        u_a = result.utility(tasks[0])
        u_b = result.utility(tasks[1])
        u_c = result.utility(tasks[2])

        # Utilities should be well-separated, not just barely different
        assert u_a > u_b + 0.5
        assert u_b > u_c + 0.5

    def test_ranking_order(self):
        """ranking() should return tasks sorted by utility."""
        tasks = [make_task("a"), make_task("b"), make_task("c")]
        wins = np.array([
            [0, 10, 10],
            [0, 0, 10],
            [0, 0, 0],
        ])
        data = PairwiseData(tasks=tasks, wins=wins)

        result = fit_thurstonian(data)
        ranking = result.ranking()

        assert ranking[0].id == "a"
        assert ranking[1].id == "b"
        assert ranking[2].id == "c"
        # Verify utilities are actually ordered (not all equal)
        assert result.utility(ranking[0]) > result.utility(ranking[1])
        assert result.utility(ranking[1]) > result.utility(ranking[2])

    def test_preference_probability_consistent(self):
        """Fitted model's preference_probability should reflect utilities."""
        tasks = [make_task("a"), make_task("b")]
        wins = np.array([[0, 8], [2, 0]])  # a wins 80% of the time
        data = PairwiseData(tasks=tasks, wins=wins)

        result = fit_thurstonian(data)

        p_a_over_b = result.preference_probability(tasks[0], tasks[1])
        # Should be close to empirical 0.8, definitely above 0.5
        assert p_a_over_b > 0.7
        # Symmetry check
        p_b_over_a = result.preference_probability(tasks[1], tasks[0])
        assert p_a_over_b + p_b_over_a == pytest.approx(1.0, abs=1e-10)

    def test_symmetric_data_gives_equal_utilities(self):
        """When a and b are equally matched, utilities should be similar."""
        tasks = [make_task("a"), make_task("b")]
        wins = np.array([[0, 50], [50, 0]])
        data = PairwiseData(tasks=tasks, wins=wins)

        result = fit_thurstonian(data)

        # Utilities should be approximately equal
        assert result.utility(tasks[0]) == pytest.approx(result.utility(tasks[1]), abs=0.1)

    def test_converges_on_simple_data(self):
        """Should converge on reasonable data."""
        tasks = [make_task("a"), make_task("b")]
        wins = np.array([[0, 5], [5, 0]])
        data = PairwiseData(tasks=tasks, wins=wins)

        result = fit_thurstonian(data)

        assert result.converged
        # NLL should be finite
        assert np.isfinite(result.neg_log_likelihood)

    def test_raises_on_single_task(self):
        """Should raise ValueError with fewer than 2 tasks."""
        tasks = [make_task("a")]
        data = PairwiseData(tasks=tasks, wins=np.zeros((1, 1)))

        with pytest.raises(ValueError, match="at least 2 tasks"):
            fit_thurstonian(data)

    def test_normalized_utility_bounds(self):
        """Normalized utility should be in [0, 1]."""
        tasks = [make_task("a"), make_task("b"), make_task("c")]
        wins = np.array([
            [0, 10, 10],
            [0, 0, 10],
            [0, 0, 0],
        ])
        data = PairwiseData(tasks=tasks, wins=wins)

        result = fit_thurstonian(data)

        for task in tasks:
            nu = result.normalized_utility(task)
            assert 0.0 <= nu <= 1.0

    def test_normalized_utility_ordering(self):
        """Normalized utility should preserve ranking order."""
        tasks = [make_task("a"), make_task("b"), make_task("c")]
        wins = np.array([
            [0, 10, 10],
            [0, 0, 10],
            [0, 0, 0],
        ])
        data = PairwiseData(tasks=tasks, wins=wins)

        result = fit_thurstonian(data)

        nu_a = result.normalized_utility(tasks[0])
        nu_b = result.normalized_utility(tasks[1])
        nu_c = result.normalized_utility(tasks[2])

        # Should preserve order and be well-separated
        assert nu_a > nu_b + 0.1
        assert nu_b > nu_c + 0.1

    def test_normalized_utility_symmetric_data(self):
        """Symmetric data should give normalized utility ≈ 0.5."""
        tasks = [make_task("a"), make_task("b")]
        wins = np.array([[0, 50], [50, 0]])
        data = PairwiseData(tasks=tasks, wins=wins)

        result = fit_thurstonian(data)

        assert result.normalized_utility(tasks[0]) == pytest.approx(0.5, abs=0.05)
        assert result.normalized_utility(tasks[1]) == pytest.approx(0.5, abs=0.05)

    def test_history_is_populated(self):
        """History should track optimization progress."""
        tasks = [make_task("a"), make_task("b")]
        wins = np.array([[0, 10], [5, 0]])
        data = PairwiseData(tasks=tasks, wins=wins)

        result = fit_thurstonian(data)

        assert len(result.history.loss) > 0
        assert len(result.history.sigma_max) == len(result.history.loss)
        # Loss should decrease
        assert result.history.loss[-1] <= result.history.loss[0]


class TestSyntheticParameterRecovery:
    """Test that model recovers known parameters from synthetic data."""

    def _simulate_comparisons(
        self, mu: np.ndarray, sigma: np.ndarray, n_comparisons_per_pair: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Simulate pairwise comparisons from known parameters."""
        n = len(mu)
        wins = np.zeros((n, n), dtype=np.int32)

        for i in range(n):
            for j in range(i + 1, n):
                p_i_beats_j = _preference_prob(mu[i], mu[j], sigma[i], sigma[j])
                for _ in range(n_comparisons_per_pair):
                    if rng.random() < p_i_beats_j:
                        wins[i, j] += 1
                    else:
                        wins[j, i] += 1

        return wins

    def test_recovers_ranking_order(self):
        """Fitted mu should preserve ranking order of true mu."""
        rng = np.random.default_rng(42)
        n_tasks = 5
        true_mu = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        true_sigma = np.ones(n_tasks)

        tasks = [make_task(f"t{i}") for i in range(n_tasks)]
        wins = self._simulate_comparisons(true_mu, true_sigma, n_comparisons_per_pair=20, rng=rng)
        data = PairwiseData(tasks=tasks, wins=wins)

        result = fit_thurstonian(data)

        # Ranking should be preserved
        fitted_order = np.argsort(-result.mu)
        true_order = np.argsort(-true_mu)
        assert list(fitted_order) == list(true_order)

    def test_mu_correlation_with_ground_truth(self):
        """Fitted mu should correlate strongly with true mu."""
        rng = np.random.default_rng(123)
        n_tasks = 8
        true_mu = rng.standard_normal(n_tasks)
        true_mu[0] = 0.0  # Fix first for identifiability
        true_sigma = np.ones(n_tasks)

        tasks = [make_task(f"t{i}") for i in range(n_tasks)]
        wins = self._simulate_comparisons(true_mu, true_sigma, n_comparisons_per_pair=30, rng=rng)
        data = PairwiseData(tasks=tasks, wins=wins)

        result = fit_thurstonian(data)

        correlation = np.corrcoef(true_mu, result.mu)[0, 1]
        assert correlation > 0.8

    def test_converges_with_many_tasks(self):
        """Should converge even with 20+ tasks."""
        rng = np.random.default_rng(456)
        n_tasks = 20
        true_mu = np.linspace(-2, 2, n_tasks)
        true_sigma = np.ones(n_tasks)

        tasks = [make_task(f"t{i}") for i in range(n_tasks)]
        wins = self._simulate_comparisons(true_mu, true_sigma, n_comparisons_per_pair=10, rng=rng)
        data = PairwiseData(tasks=tasks, wins=wins)

        result = fit_thurstonian(data)

        assert result.converged
        assert result.gradient_norm < 1.0
        # Correlation should be good even with many tasks
        correlation = np.corrcoef(true_mu, result.mu)[0, 1]
        assert correlation > 0.8


class TestLargeScaleDiagnostics:
    """Large scale tests with diagnostic plots saved to plots/thurstonian/."""

    PLOTS_DIR = Path(__file__).parent / "plots" / "thurstonian"

    @pytest.fixture(autouse=True)
    def setup_plots_dir(self):
        self.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    def _simulate_comparisons(
        self, mu: np.ndarray, sigma: np.ndarray, n_comparisons_per_pair: int, rng: np.random.Generator
    ) -> np.ndarray:
        n = len(mu)
        wins = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            for j in range(i + 1, n):
                p_i_beats_j = _preference_prob(mu[i], mu[j], sigma[i], sigma[j])
                for _ in range(n_comparisons_per_pair):
                    if rng.random() < p_i_beats_j:
                        wins[i, j] += 1
                    else:
                        wins[j, i] += 1
        return wins

    def _plot_diagnostics(
        self,
        result: ThurstonianResult,
        true_mu: np.ndarray,
        true_sigma: np.ndarray,
        name: str,
    ) -> None:
        _plot_synthetic_diagnostics(result, true_mu, true_sigma, name, self.PLOTS_DIR)

    def test_50_tasks_dense_data(self):
        """50 tasks with 20 comparisons per pair - should work well."""
        rng = np.random.default_rng(1001)
        n_tasks = 50
        true_mu = np.linspace(-3, 3, n_tasks)
        true_sigma = np.ones(n_tasks)

        tasks = [make_task(f"t{i}") for i in range(n_tasks)]
        wins = self._simulate_comparisons(true_mu, true_sigma, n_comparisons_per_pair=20, rng=rng)
        data = PairwiseData(tasks=tasks, wins=wins)

        result = fit_thurstonian(data, max_iter=3000)
        self._plot_diagnostics(result, true_mu, true_sigma, "synthetic_50_tasks_dense")

        assert result.gradient_norm < 1.0, f"grad={result.gradient_norm}"
        assert np.corrcoef(true_mu, result.mu)[0, 1] > 0.95

    def test_50_tasks_sparse_data(self):
        """50 tasks with only 5 comparisons per pair - harder case."""
        rng = np.random.default_rng(1002)
        n_tasks = 50
        true_mu = np.linspace(-3, 3, n_tasks)
        true_sigma = np.ones(n_tasks)

        tasks = [make_task(f"t{i}") for i in range(n_tasks)]
        wins = self._simulate_comparisons(true_mu, true_sigma, n_comparisons_per_pair=5, rng=rng)
        data = PairwiseData(tasks=tasks, wins=wins)

        result = fit_thurstonian(data, max_iter=3000)
        self._plot_diagnostics(result, true_mu, true_sigma, "synthetic_50_tasks_sparse")

        assert result.gradient_norm < 1.0, f"grad={result.gradient_norm}"
        assert np.corrcoef(true_mu, result.mu)[0, 1] > 0.85

    def test_50_tasks_varying_sigma(self):
        """50 tasks where true sigma varies - tests heteroscedastic recovery."""
        rng = np.random.default_rng(1003)
        n_tasks = 50
        true_mu = np.linspace(-3, 3, n_tasks)
        true_sigma = 0.5 + rng.random(n_tasks) * 1.5  # sigma in [0.5, 2.0]

        tasks = [make_task(f"t{i}") for i in range(n_tasks)]
        wins = self._simulate_comparisons(true_mu, true_sigma, n_comparisons_per_pair=20, rng=rng)
        data = PairwiseData(tasks=tasks, wins=wins)

        result = fit_thurstonian(data, max_iter=3000)
        self._plot_diagnostics(result, true_mu, true_sigma, "synthetic_50_tasks_varying_sigma")

        assert result.gradient_norm < 1.0, f"grad={result.gradient_norm}"
        assert np.corrcoef(true_mu, result.mu)[0, 1] > 0.9

    def test_100_tasks_dense(self):
        """100 tasks - stress test."""
        rng = np.random.default_rng(1004)
        n_tasks = 100
        true_mu = np.linspace(-4, 4, n_tasks)
        true_sigma = np.ones(n_tasks)

        tasks = [make_task(f"t{i}") for i in range(n_tasks)]
        wins = self._simulate_comparisons(true_mu, true_sigma, n_comparisons_per_pair=15, rng=rng)
        data = PairwiseData(tasks=tasks, wins=wins)

        result = fit_thurstonian(data, max_iter=5000)
        self._plot_diagnostics(result, true_mu, true_sigma, "synthetic_100_tasks_dense")

        corr = np.corrcoef(true_mu, result.mu)[0, 1]
        assert corr > 0.9, f"corr={corr}, grad={result.gradient_norm}"

    def test_50_tasks_clustered_utilities(self):
        """50 tasks with clustered utilities - some very similar items."""
        rng = np.random.default_rng(1005)
        n_tasks = 50
        # 5 clusters of 10 items each
        true_mu = np.repeat([-2, -1, 0, 1, 2], 10) + rng.normal(0, 0.1, n_tasks)
        true_sigma = np.ones(n_tasks)

        tasks = [make_task(f"t{i}") for i in range(n_tasks)]
        wins = self._simulate_comparisons(true_mu, true_sigma, n_comparisons_per_pair=20, rng=rng)
        data = PairwiseData(tasks=tasks, wins=wins)

        result = fit_thurstonian(data, max_iter=3000)
        self._plot_diagnostics(result, true_mu, true_sigma, "synthetic_50_tasks_clustered")

        corr = np.corrcoef(true_mu, result.mu)[0, 1]
        assert corr > 0.9, f"corr={corr}, grad={result.gradient_norm}"

    def _simulate_comparisons_noisy(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        n_comparisons_per_pair: int,
        noise_rate: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Simulate with random response flips (noise_rate = prob of random flip)."""
        n = len(mu)
        wins = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            for j in range(i + 1, n):
                p_i_beats_j = _preference_prob(mu[i], mu[j], sigma[i], sigma[j])
                for _ in range(n_comparisons_per_pair):
                    # With noise_rate probability, flip a coin instead
                    if rng.random() < noise_rate:
                        i_wins = rng.random() < 0.5
                    else:
                        i_wins = rng.random() < p_i_beats_j

                    if i_wins:
                        wins[i, j] += 1
                    else:
                        wins[j, i] += 1
        return wins

    def test_50_tasks_10pct_noise(self):
        """50 tasks with 10% random response noise."""
        rng = np.random.default_rng(1006)
        n_tasks = 50
        true_mu = np.linspace(-3, 3, n_tasks)
        true_sigma = np.ones(n_tasks)

        tasks = [make_task(f"t{i}") for i in range(n_tasks)]
        wins = self._simulate_comparisons_noisy(
            true_mu, true_sigma, n_comparisons_per_pair=20, noise_rate=0.1, rng=rng
        )
        data = PairwiseData(tasks=tasks, wins=wins)

        result = fit_thurstonian(data, max_iter=3000)
        self._plot_diagnostics(result, true_mu, true_sigma, "synthetic_50_tasks_10pct_noise")

        assert result.gradient_norm < 1.0, f"grad={result.gradient_norm}"
        assert np.corrcoef(true_mu, result.mu)[0, 1] > 0.9

    def test_50_tasks_20pct_noise(self):
        """50 tasks with 20% random response noise - realistic for LLMs."""
        rng = np.random.default_rng(1007)
        n_tasks = 50
        true_mu = np.linspace(-3, 3, n_tasks)
        true_sigma = np.ones(n_tasks)

        tasks = [make_task(f"t{i}") for i in range(n_tasks)]
        wins = self._simulate_comparisons_noisy(
            true_mu, true_sigma, n_comparisons_per_pair=20, noise_rate=0.2, rng=rng
        )
        data = PairwiseData(tasks=tasks, wins=wins)

        result = fit_thurstonian(data, max_iter=3000)
        self._plot_diagnostics(result, true_mu, true_sigma, "synthetic_50_tasks_20pct_noise")

        assert result.gradient_norm < 1.0, f"grad={result.gradient_norm}"
        assert np.corrcoef(true_mu, result.mu)[0, 1] > 0.85

    def test_50_tasks_30pct_noise(self):
        """50 tasks with 30% random response noise - very noisy."""
        rng = np.random.default_rng(1008)
        n_tasks = 50
        true_mu = np.linspace(-3, 3, n_tasks)
        true_sigma = np.ones(n_tasks)

        tasks = [make_task(f"t{i}") for i in range(n_tasks)]
        wins = self._simulate_comparisons_noisy(
            true_mu, true_sigma, n_comparisons_per_pair=20, noise_rate=0.3, rng=rng
        )
        data = PairwiseData(tasks=tasks, wins=wins)

        result = fit_thurstonian(data, max_iter=3000)
        self._plot_diagnostics(result, true_mu, true_sigma, "synthetic_50_tasks_30pct_noise")

        assert result.gradient_norm < 1.0, f"grad={result.gradient_norm}"
        assert np.corrcoef(true_mu, result.mu)[0, 1] > 0.7


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Thurstonian model analysis")
    parser.add_argument("--synthetic", action="store_true", help="Run synthetic diagnostics only")
    parser.add_argument("--both", action="store_true", help="Run both real and synthetic")
    args = parser.parse_args()

    if args.both:
        run_real_data_analysis()
        run_synthetic_diagnostics()
    elif args.synthetic:
        run_synthetic_diagnostics()
    else:
        run_real_data_analysis()


if __name__ == "__main__":
    main()
