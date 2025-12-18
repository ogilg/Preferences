"""Tests for Thurstonian model fitting."""

import numpy as np
import pytest

from src.preferences.ranking.thurstonian import (
    PairwiseData,
    ThurstonianResult,
    fit_thurstonian,
    _preference_prob,
    _neg_log_likelihood,
)
from src.task_data import Task, OriginDataset
from src.types import BinaryPreferenceMeasurement, PreferenceType


def make_task(id: str) -> Task:
    """Create a minimal task for testing."""
    return Task(prompt=f"Task {id}", origin=OriginDataset.ALPACA, id=id, metadata={})


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
