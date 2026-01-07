"""Thurstonian model for pairwise preference data.

Fits utility values μ and uncertainty σ for each item based on
pairwise comparison outcomes. Uses the model:

    U(i) ~ N(μ_i, σ_i²)
    P(i ≻ j) = Φ((μ_i - μ_j) / √(σ_i² + σ_j²))

where Φ is the standard normal CDF.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import minimize
from scipy.stats import norm
from collections import Counter, defaultdict

from ...task_data import Task
from ...types import BinaryPreferenceMeasurement

# Tuned defaults (see data_analysis/tuning.md)
DEFAULT_MU_BOUNDS = (-10.0, 10.0)
DEFAULT_LOG_SIGMA_BOUNDS = (-2.0, 2.0)
DEFAULT_SIGMA_INIT = 1.0


@dataclass
class PairwiseData:
    """wins[i,j] = number of times tasks[i] beat tasks[j]."""

    tasks: list["Task"]
    wins: np.ndarray
    _id_to_idx: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._id_to_idx = {t.id: i for i, t in enumerate(self.tasks)}

    def index_of(self, task: "Task") -> int:
        return self._id_to_idx[task.id]

    def total_comparisons(self, task: "Task") -> int:
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
class OptimizationHistory:
    """Minimal optimization progress tracking."""

    loss: list[float] = field(default_factory=list)
    sigma_max: list[float] = field(default_factory=list)


@dataclass
class ThurstonianResult:
    """mu: utility means, sigma: utility standard deviations."""

    tasks: list["Task"]
    mu: np.ndarray
    sigma: np.ndarray
    converged: bool
    neg_log_likelihood: float
    n_iterations: int
    n_function_evals: int
    termination_message: str
    gradient_norm: float
    history: OptimizationHistory
    _id_to_idx: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._id_to_idx = {t.id: i for i, t in enumerate(self.tasks)}

    def utility(self, task: "Task") -> float:
        return float(self.mu[self._id_to_idx[task.id]])

    def uncertainty(self, task: "Task") -> float:
        return float(self.sigma[self._id_to_idx[task.id]])

    def preference_probability(self, task_a: "Task", task_b: "Task") -> float:
        """P(task_a ≻ task_b) under the fitted model."""
        i = self._id_to_idx[task_a.id]
        j = self._id_to_idx[task_b.id]
        return _preference_prob(self.mu[i], self.mu[j], self.sigma[i], self.sigma[j])

    def ranking(self) -> list["Task"]:
        """Tasks sorted by utility, highest first."""
        order = np.argsort(-self.mu)
        return [self.tasks[i] for i in order]

    def normalized_utility(self, task: "Task") -> float:
        """Average probability of being preferred over all other tasks (0.5 = average)."""
        probs = [
            self.preference_probability(task, other)
            for other in self.tasks
            if other.id != task.id
        ]
        if not probs:
            raise ValueError("Cannot compute normalized utility with fewer than 2 tasks")
        return sum(probs) / len(probs)


def _preference_prob(mu_i: float, mu_j: float, sigma_i: float, sigma_j: float) -> float:
    """P(i ≻ j) using the difference distribution."""
    scale = np.sqrt(sigma_i**2 + sigma_j**2)
    return float(norm.sf(0, loc=mu_i - mu_j, scale=scale))


def _neg_log_likelihood(
    params: np.ndarray,
    wins: np.ndarray,
    n: int,
) -> float:
    """params: [μ_1..μ_{n-1}, log(σ_0)..log(σ_{n-1})]. μ_0 fixed to 0."""
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
    sigma_init: float = DEFAULT_SIGMA_INIT,
    max_iter: int = 1000,
    log_sigma_bounds: tuple[float, float] = DEFAULT_LOG_SIGMA_BOUNDS,
    mu_bounds: tuple[float, float] = DEFAULT_MU_BOUNDS,
    gradient_tol: float = 1.0,
    loss_tol: float = 1e-8,
    lambda_sigma: float = 0.0,
) -> ThurstonianResult:
    n = data.n_tasks

    if n < 2:
        raise ValueError("Need at least 2 tasks to fit")

    mu_init = np.zeros(n - 1)
    log_sigma_init = np.full(n, np.log(sigma_init))
    params_init = np.concatenate([mu_init, log_sigma_init])

    bounds = [mu_bounds] * (n - 1) + [log_sigma_bounds] * n

    history = OptimizationHistory()

    def objective(params: np.ndarray) -> float:
        nll = _neg_log_likelihood(params, data.wins, n)
        if lambda_sigma > 0:
            sigma = np.exp(params[n - 1:])
            nll += lambda_sigma * float(np.sum(sigma ** 2))
        return nll

    def callback(params: np.ndarray) -> None:
        loss = _neg_log_likelihood(params, data.wins, n)
        sigma_max = float(np.exp(params[n - 1:]).max())
        history.loss.append(loss)
        history.sigma_max.append(sigma_max)

    result = minimize(
        objective,
        params_init,
        method="L-BFGS-B",
        bounds=bounds,
        callback=callback,
        options={
            "maxiter": max_iter,
            "maxfun": max_iter * 20,
            "gtol": gradient_tol,
            "ftol": loss_tol,
        },
    )

    mu = np.zeros(n)
    mu[1:] = result.x[: n - 1]
    sigma = np.exp(result.x[n - 1:])
    gradient_norm = float(np.linalg.norm(result.jac)) if result.jac is not None else -1.0

    return ThurstonianResult(
        tasks=data.tasks,
        mu=mu,
        sigma=sigma,
        converged=result.success,
        neg_log_likelihood=result.fun,
        n_iterations=result.nit,
        n_function_evals=result.nfev,
        termination_message=result.message,
        gradient_norm=gradient_norm,
        history=history,
    )


def save_thurstonian(result: ThurstonianResult, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "task_ids": [t.id for t in result.tasks],
        "mu": result.mu.tolist(),
        "sigma": result.sigma.tolist(),
        "converged": result.converged,
        "neg_log_likelihood": float(result.neg_log_likelihood),
        "n_iterations": result.n_iterations,
        "n_function_evals": result.n_function_evals,
        "termination_message": result.termination_message,
        "gradient_norm": result.gradient_norm,
        "history": {
            "loss": [float(x) for x in result.history.loss],
            "sigma_max": [float(x) for x in result.history.sigma_max],
        },
    }

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load_thurstonian(path: Path | str, tasks: list["Task"]) -> ThurstonianResult:
    """tasks must contain all task_ids saved in the file."""
    with open(path) as f:
        data = yaml.safe_load(f)

    saved_ids = data["task_ids"]
    task_dict = {t.id: t for t in tasks}

    # Verify all saved task IDs are present
    missing = set(saved_ids) - set(task_dict.keys())
    if missing:
        raise ValueError(f"Tasks not found for IDs: {missing}")

    # Reconstruct task list in saved order
    ordered_tasks = [task_dict[tid] for tid in saved_ids]

    history_data = data.get("history", {"loss": [], "sigma_max": []})
    history = OptimizationHistory(
        loss=history_data.get("loss", []),
        sigma_max=history_data.get("sigma_max", []),
    )

    return ThurstonianResult(
        tasks=ordered_tasks,
        mu=np.array(data["mu"]),
        sigma=np.array(data["sigma"]),
        converged=data["converged"],
        neg_log_likelihood=data["neg_log_likelihood"],
        n_iterations=data.get("n_iterations", -1),
        n_function_evals=data.get("n_function_evals", -1),
        termination_message=data.get("termination_message", "unknown (loaded from old format)"),
        gradient_norm=data.get("gradient_norm", -1.0),
        history=history,
    )


def compute_pair_agreement(
    comparisons: list["BinaryPreferenceMeasurement"],
) -> float:
    """Compute mean agreement rate across pairs with multiple samples.

    For each pair (A, B), agreement = max(n_a_wins, n_b_wins) / total.
    Returns mean agreement across all pairs (1.0 = perfect consistency).
    """
    pair_outcomes: dict[tuple[str, str], list[str]] = defaultdict(list)
    for c in comparisons:
        key = tuple(sorted([c.task_a.id, c.task_b.id]))
        winner = c.task_a.id if c.choice == "a" else c.task_b.id
        pair_outcomes[key].append(winner)

    agreements = []
    for outcomes in pair_outcomes.values():
        if len(outcomes) < 2:
            continue
        counts = Counter(outcomes)
        majority = max(counts.values())
        agreements.append(majority / len(outcomes))

    return float(np.mean(agreements)) if agreements else 1.0
