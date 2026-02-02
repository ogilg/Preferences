"""Active learning for efficient Thurstonian utility estimation.

Implements the iterative pair selection algorithm from:
"Utility Engineering: Analyzing and Controlling Emergent Value Systems in AIs"
(Mazeika et al., 2025), Appendix B.

The algorithm prioritizes pairs where:
1. Current utility estimates are close (ambiguous pairs)
2. Items have been compared fewer times (under-sampled)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from scipy.stats import spearmanr

from src.task_data import Task
from src.types import BinaryPreferenceMeasurement
from .thurstonian import PairwiseData, ThurstonianResult, fit_thurstonian


def _sorted_pair_key(a: Task, b: Task) -> tuple[str, str]:
    """Canonical key for a pair of tasks (sorted by id)."""
    return tuple(sorted([a.id, b.id]))


@dataclass
class ActiveLearningState:
    """Tracks state across active learning iterations."""

    tasks: list[Task]
    comparisons: list[BinaryPreferenceMeasurement] = field(default_factory=list)
    sampled_pairs: set[tuple[str, str]] = field(default_factory=set)
    current_fit: ThurstonianResult | None = None
    previous_fit: ThurstonianResult | None = None
    iteration: int = 0

    _id_to_task: dict[str, Task] = field(init=False, repr=False)
    _degrees: dict[str, int] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._id_to_task = {t.id: t for t in self.tasks}
        self._degrees = {t.id: 0 for t in self.tasks}

    def degree(self, task: Task) -> int:
        """Number of unique pairs this task has been involved in."""
        return self._degrees[task.id]

    def pair_degree_sum(self, a: Task, b: Task) -> int:
        """Sum of degrees of two tasks."""
        return self._degrees[a.id] + self._degrees[b.id]

    def add_comparisons(self, comparisons: list[BinaryPreferenceMeasurement]) -> None:
        """Add new comparisons and update degree counts."""
        for c in comparisons:
            key = _sorted_pair_key(c.task_a, c.task_b)
            if key not in self.sampled_pairs:
                self.sampled_pairs.add(key)
                self._degrees[c.task_a.id] += 1
                self._degrees[c.task_b.id] += 1
        self.comparisons.extend(comparisons)

    def get_unsampled_pairs(self) -> list[tuple[Task, Task]]:
        """Get all pairs that haven't been sampled yet."""
        unsampled = []
        for a, b in combinations(self.tasks, 2):
            if _sorted_pair_key(a, b) not in self.sampled_pairs:
                unsampled.append((a, b))
        return unsampled

    def fit(self, **fit_kwargs) -> ThurstonianResult:
        """Fit Thurstonian model to current comparisons."""
        data = PairwiseData.from_comparisons(self.comparisons, self.tasks)
        self.previous_fit = self.current_fit
        self.current_fit = fit_thurstonian(data, **fit_kwargs)
        return self.current_fit


def generate_d_regular_pairs(
    tasks: list[Task],
    d: int,
    rng: np.random.Generator,
) -> list[tuple[Task, Task]]:
    """Generate a random d-regular graph over tasks.

    Each task will be connected to exactly d other tasks (if possible).
    For n tasks with d edges each, we need n*d/2 unique edges.

    Args:
        tasks: List of tasks to connect
        d: Degree for each node (number of connections per task)
        rng: Random number generator

    Returns:
        List of (task_a, task_b) pairs forming a d-regular graph
    """
    n = len(tasks)
    if d >= n:
        # If d >= n-1, just return all pairs
        return list(combinations(tasks, 2))

    # Use a simple randomized approach: repeatedly try to add edges
    # until we reach target degree for all nodes
    degrees = {t.id: 0 for t in tasks}
    edges: set[tuple[str, str]] = set()
    task_ids = [t.id for t in tasks]

    # Keep trying to add edges until all nodes have degree d
    max_attempts = n * d * 10
    attempts = 0

    while attempts < max_attempts:
        # Find nodes that still need edges
        need_edges = [tid for tid in task_ids if degrees[tid] < d]
        if not need_edges:
            break

        # Pick two random nodes that need edges and aren't already connected
        rng.shuffle(need_edges)
        added = False
        for i, tid_a in enumerate(need_edges):
            for tid_b in need_edges[i + 1:]:
                key = tuple(sorted([tid_a, tid_b]))
                if key not in edges and degrees[tid_a] < d and degrees[tid_b] < d:
                    edges.add(key)
                    degrees[tid_a] += 1
                    degrees[tid_b] += 1
                    added = True
                    break
            if added:
                break
        attempts += 1

    # Convert edge set back to task pairs
    id_to_task = {t.id: t for t in tasks}
    return [(id_to_task[a], id_to_task[b]) for a, b in edges]


def select_next_pairs(
    state: ActiveLearningState,
    batch_size: int,
    p_threshold: float = 0.3,
    q_threshold: float = 0.3,
    relaxation_factor: float = 1.5,
    rng: np.random.Generator | None = None,
) -> list[tuple[Task, Task]]:
    """Select the next batch of pairs to query.

    Prioritizes pairs where:
    1. |μ(a) - μ(b)| is small (ambiguous utility difference)
    2. degree(a) + degree(b) is small (under-sampled)

    Args:
        state: Current active learning state with fitted model
        batch_size: Number of pairs to select
        p_threshold: Percentile threshold for utility differences (0-1)
        q_threshold: Percentile threshold for degree sums (0-1)
        relaxation_factor: Factor to relax thresholds if too few pairs found
        rng: Random number generator for tie-breaking

    Returns:
        List of (task_a, task_b) pairs to query next
    """
    if rng is None:
        rng = np.random.default_rng()

    unsampled = state.get_unsampled_pairs()
    if not unsampled:
        return []

    if len(unsampled) <= batch_size:
        return unsampled

    # If no fit yet, return random pairs
    if state.current_fit is None:
        rng.shuffle(unsampled)
        return unsampled[:batch_size]

    # Score each unsampled pair
    mu = state.current_fit.mu
    id_to_idx = state.current_fit._id_to_idx

    pair_scores = []
    for a, b in unsampled:
        mu_diff = abs(mu[id_to_idx[a.id]] - mu[id_to_idx[b.id]])
        degree_sum = state.pair_degree_sum(a, b)
        pair_scores.append((a, b, mu_diff, degree_sum))

    # Extract arrays for percentile computation
    mu_diffs = np.array([s[2] for s in pair_scores])
    degree_sums = np.array([s[3] for s in pair_scores])

    # Iteratively relax thresholds until we have enough pairs
    current_p = p_threshold
    current_q = q_threshold

    while True:
        # Find pairs in bottom percentiles
        mu_cutoff = np.percentile(mu_diffs, current_p * 100)
        degree_cutoff = np.percentile(degree_sums, current_q * 100)

        candidates = [
            (a, b)
            for a, b, mu_d, deg_s in pair_scores
            if mu_d <= mu_cutoff and deg_s <= degree_cutoff
        ]

        if len(candidates) >= batch_size:
            rng.shuffle(candidates)
            return candidates[:batch_size]

        # Relax thresholds
        current_p = min(1.0, current_p * relaxation_factor)
        current_q = min(1.0, current_q * relaxation_factor)

        # If thresholds are maxed out, just return what we have plus random
        if current_p >= 1.0 and current_q >= 1.0:
            remaining = [
                (a, b) for a, b, _, _ in pair_scores if (a, b) not in candidates
            ]
            rng.shuffle(remaining)
            return candidates + remaining[: batch_size - len(candidates)]


def check_convergence(
    state: ActiveLearningState,
    threshold: float = 0.99,
) -> tuple[bool, float]:
    """Check if utility estimates have converged.

    Compares rank correlation between current and previous iteration.

    Args:
        state: Current state with current_fit and previous_fit
        threshold: Minimum rank correlation to consider converged

    Returns:
        (converged, correlation)
    """
    if state.current_fit is None or state.previous_fit is None:
        return False, 0.0

    # Ensure same task ordering
    if state.current_fit.tasks != state.previous_fit.tasks:
        return False, 0.0

    corr = spearmanr(state.current_fit.mu, state.previous_fit.mu).correlation
    return corr >= threshold, float(corr)


