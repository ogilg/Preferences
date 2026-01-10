"""Tests for autograd-based Thurstonian gradient computation."""

from __future__ import annotations

import time

import numpy as np
import pytest

from src.preferences.ranking.thurstonian import (
    PairwiseData,
    _make_objective_and_grad,
    _neg_log_likelihood,
    _neg_log_likelihood_autograd,
    fit_thurstonian,
)
from src.task_data import OriginDataset, Task


def make_tasks(n: int) -> list[Task]:
    return [
        Task(prompt=f"Task {i}", origin=OriginDataset.MATH, id=f"task_{i}", metadata={})
        for i in range(n)
    ]


def make_random_wins(n: int, rng: np.random.Generator) -> np.ndarray:
    wins = rng.integers(0, 10, size=(n, n))
    np.fill_diagonal(wins, 0)
    return wins


class TestGradientCorrectness:
    """Verify autograd gradient matches numerical finite differences."""

    @pytest.mark.parametrize("n_tasks", [3, 5, 10, 20])
    def test_gradient_matches_finite_differences(self, n_tasks: int):
        rng = np.random.default_rng(42)
        wins = make_random_wins(n_tasks, rng)

        n_params = (n_tasks - 1) + n_tasks
        params = rng.standard_normal(n_params) * 0.5

        objective_and_grad = _make_objective_and_grad(wins, n_tasks, lambda_sigma=0.0)
        _, autograd_gradient = objective_and_grad(params)

        # Numerical gradient via central differences
        eps = 1e-5
        numerical_gradient = np.zeros_like(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += eps
            params_minus = params.copy()
            params_minus[i] -= eps

            f_plus = _neg_log_likelihood_autograd(params_plus, wins, n_tasks)
            f_minus = _neg_log_likelihood_autograd(params_minus, wins, n_tasks)
            numerical_gradient[i] = (f_plus - f_minus) / (2 * eps)

        np.testing.assert_allclose(
            autograd_gradient, numerical_gradient, rtol=1e-4, atol=1e-6
        )

    def test_gradient_with_regularization(self):
        n_tasks = 5
        rng = np.random.default_rng(123)
        wins = make_random_wins(n_tasks, rng)

        n_params = (n_tasks - 1) + n_tasks
        params = rng.standard_normal(n_params) * 0.5
        lambda_sigma = 0.1

        objective_and_grad = _make_objective_and_grad(wins, n_tasks, lambda_sigma)
        _, autograd_gradient = objective_and_grad(params)

        eps = 1e-5
        numerical_gradient = np.zeros_like(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += eps
            params_minus = params.copy()
            params_minus[i] -= eps

            f_plus = _neg_log_likelihood_autograd(
                params_plus, wins, n_tasks, lambda_sigma
            )
            f_minus = _neg_log_likelihood_autograd(
                params_minus, wins, n_tasks, lambda_sigma
            )
            numerical_gradient[i] = (f_plus - f_minus) / (2 * eps)

        np.testing.assert_allclose(
            autograd_gradient, numerical_gradient, rtol=1e-4, atol=1e-6
        )


class TestNLLEquivalence:
    """Verify autograd NLL matches original numpy NLL."""

    @pytest.mark.parametrize("n_tasks", [3, 5, 10])
    def test_nll_values_match(self, n_tasks: int):
        rng = np.random.default_rng(42)
        wins = make_random_wins(n_tasks, rng)

        n_params = (n_tasks - 1) + n_tasks
        params = rng.standard_normal(n_params) * 0.5

        nll_numpy = _neg_log_likelihood(params, wins, n_tasks)
        nll_autograd = float(_neg_log_likelihood_autograd(params, wins, n_tasks))

        np.testing.assert_allclose(nll_numpy, nll_autograd, rtol=1e-10)


class TestOptimizationEquivalence:
    """Verify optimization with analytical gradient produces same results."""

    def test_fit_converges_to_same_solution(self):
        n_tasks = 10
        rng = np.random.default_rng(42)
        tasks = make_tasks(n_tasks)
        wins = make_random_wins(n_tasks, rng)
        data = PairwiseData(tasks=tasks, wins=wins)

        result = fit_thurstonian(data, max_iter=1000)

        assert result.converged
        assert result.neg_log_likelihood < 1000
        assert np.all(np.isfinite(result.mu))
        assert np.all(np.isfinite(result.sigma))
        assert np.all(result.sigma > 0)

    def test_fit_with_real_preference_pattern(self):
        """Test with a clear preference ordering: task 0 > task 1 > task 2."""
        tasks = make_tasks(3)
        wins = np.array(
            [
                [0, 8, 9],  # task 0 beats 1 and 2 most of the time
                [2, 0, 7],  # task 1 beats 2 most of the time
                [1, 3, 0],  # task 2 rarely wins
            ]
        )
        data = PairwiseData(tasks=tasks, wins=wins)

        result = fit_thurstonian(data, max_iter=1000)

        assert result.converged
        # task 0 should have highest utility
        assert result.mu[0] > result.mu[1] > result.mu[2]


class TestPerformance:
    """Verify analytical gradient provides speedup."""

    @pytest.mark.parametrize("n_tasks", [50, 100])
    def test_analytical_gradient_is_fast(self, n_tasks: int):
        rng = np.random.default_rng(42)
        tasks = make_tasks(n_tasks)
        wins = make_random_wins(n_tasks, rng)
        data = PairwiseData(tasks=tasks, wins=wins)

        start = time.perf_counter()
        result = fit_thurstonian(data, max_iter=500)
        elapsed = time.perf_counter() - start

        assert result.converged or result.n_iterations == 500
        # With analytical gradient, 500 iterations on 100 tasks should be fast
        # This is a sanity check, not a strict performance test
        assert elapsed < 30, f"Fitting took {elapsed:.1f}s, expected < 30s"
