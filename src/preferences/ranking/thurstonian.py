"""Thurstonian model for pairwise preference data.

Fits utility values μ and uncertainty σ for each item based on
pairwise comparison outcomes. Uses the model:

    U(i) ~ N(μ_i, σ_i²)
    P(i ≻ j) = Φ((μ_i - μ_j) / √(σ_i² + σ_j²))

where Φ is the standard normal CDF.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

if TYPE_CHECKING:
    from ...task_data import Task
    from ...types import BinaryPreferenceMeasurement


@dataclass
class PairwiseData:
    """Aggregated pairwise comparison counts.

    Attributes:
        tasks: Ordered list of tasks being compared.
        wins: Matrix where wins[i,j] = number of times tasks[i] beat tasks[j].
    """

    tasks: list["Task"]
    wins: np.ndarray
    _id_to_idx: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._id_to_idx = {t.id: i for i, t in enumerate(self.tasks)}

    def index_of(self, task: "Task") -> int:
        """Get the index of a task in the ordering."""
        return self._id_to_idx[task.id]

    def total_comparisons(self, task: "Task") -> int:
        """Total number of comparisons involving this task."""
        idx = self.index_of(task)
        return int(self.wins[idx, :].sum() + self.wins[:, idx].sum())

    @property
    def n_tasks(self) -> int:
        return len(self.tasks)

    @classmethod
    def from_comparisons(
        cls,
        comparisons: list["BinaryPreferenceMeasurement"],
        tasks: list["Task"],
    ) -> "PairwiseData":
        """Create from a list of binary preference measurements."""
        id_to_idx = {t.id: i for i, t in enumerate(tasks)}
        n = len(tasks)
        wins = np.zeros((n, n), dtype=np.int32)

        for c in comparisons:
            i, j = id_to_idx[c.task_a.id], id_to_idx[c.task_b.id]
            if c.choice == "a":
                wins[i, j] += 1
            else:
                wins[j, i] += 1

        return cls(tasks=tasks, wins=wins)


@dataclass
class ThurstonianResult:
    """Fitted Thurstonian utilities and uncertainties.

    Attributes:
        tasks: Ordered list of tasks (same ordering as input).
        mu: Utility means for each task.
        sigma: Utility standard deviations for each task.
        converged: Whether the optimization converged.
        neg_log_likelihood: Final negative log-likelihood value.
    """

    tasks: list["Task"]
    mu: np.ndarray
    sigma: np.ndarray
    converged: bool
    neg_log_likelihood: float
    _id_to_idx: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._id_to_idx = {t.id: i for i, t in enumerate(self.tasks)}

    def utility(self, task: "Task") -> float:
        """Get the fitted utility mean for a task."""
        return float(self.mu[self._id_to_idx[task.id]])

    def uncertainty(self, task: "Task") -> float:
        """Get the fitted utility standard deviation for a task."""
        return float(self.sigma[self._id_to_idx[task.id]])

    def preference_probability(self, task_a: "Task", task_b: "Task") -> float:
        """Compute P(task_a ≻ task_b) under the fitted model."""
        i = self._id_to_idx[task_a.id]
        j = self._id_to_idx[task_b.id]
        return _preference_prob(self.mu[i], self.mu[j], self.sigma[i], self.sigma[j])

    def ranking(self) -> list["Task"]:
        """Return tasks sorted by utility (highest first)."""
        order = np.argsort(-self.mu)
        return [self.tasks[i] for i in order]

    def normalized_utility(self, task: "Task") -> float:
        """Get normalized utility in [0, 1] for a task.

        Computed as the average probability of being preferred over all other tasks.
        A value of 0.5 means the task is equally preferred to the average task.
        """
        idx = self._id_to_idx[task.id]
        probs = [
            self.preference_probability(task, other)
            for other in self.tasks
            if other.id != task.id
        ]
        return sum(probs) / len(probs) if probs else 0.5


def _preference_prob(mu_i: float, mu_j: float, sigma_i: float, sigma_j: float) -> float:
    """Compute P(i ≻ j) using the difference distribution."""
    scale = np.sqrt(sigma_i**2 + sigma_j**2)
    return float(norm.sf(0, loc=mu_i - mu_j, scale=scale))


def _neg_log_likelihood(
    params: np.ndarray,
    wins: np.ndarray,
    n: int,
) -> float:
    """Negative log-likelihood for Thurstonian model.

    Args:
        params: Array of [μ_1, ..., μ_{n-1}, log(σ_0), ..., log(σ_{n-1})].
                Note: μ_0 is fixed to 0 for identifiability.
        wins: Matrix of win counts.
        n: Number of items.

    Returns:
        Negative log-likelihood (to minimize).
    """
    # Unpack parameters
    mu = np.zeros(n)
    mu[1:] = params[: n - 1]
    sigma = np.exp(params[n - 1:])

    # Compute pairwise preference probabilities (vectorized)
    mu_diff = mu[:, np.newaxis] - mu[np.newaxis, :]
    scale = np.sqrt(sigma[:, np.newaxis]**2 + sigma[np.newaxis, :]**2)
    p = norm.sf(0, loc=mu_diff, scale=scale)
    p = np.clip(p, 1e-10, 1 - 1e-10)

    return -np.sum(wins * np.log(p))


def fit_thurstonian(
    data: PairwiseData,
    sigma_init: float = 1.0,
    max_iter: int = 1000,
) -> ThurstonianResult:
    """Fit a Thurstonian model to pairwise comparison data.

    Args:
        data: Pairwise comparison counts.
        sigma_init: Initial value for all σ parameters.
        max_iter: Maximum optimization iterations.

    Returns:
        ThurstonianResult with fitted utilities and uncertainties.
    """
    n = data.n_tasks

    if n < 2:
        raise ValueError("Need at least 2 tasks to fit")

    # Initial parameters
    # μ_0 = 0 (fixed), initialize others to 0
    mu_init = np.zeros(n - 1)
    log_sigma_init = np.full(n, np.log(sigma_init))

    params_init = np.concatenate([mu_init, log_sigma_init])

    # Optimize
    result = minimize(
        _neg_log_likelihood,
        params_init,
        args=(data.wins, n),
        method="L-BFGS-B",
        options={"maxiter": max_iter},
    )

    # Unpack results
    mu = np.zeros(n)
    mu[1:] = result.x[: n - 1]
    sigma = np.exp(result.x[n - 1 :])

    return ThurstonianResult(
        tasks=data.tasks,
        mu=mu,
        sigma=sigma,
        converged=result.success,
        neg_log_likelihood=result.fun,
    )
