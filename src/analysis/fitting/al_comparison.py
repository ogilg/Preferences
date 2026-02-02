from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from scipy.stats import spearmanr

from src.task_data import Task, OriginDataset
from src.types import BinaryPreferenceMeasurement, PreferenceType
from src.fitting.thurstonian_fitting.thurstonian import (
    PairwiseData,
    ThurstonianResult,
    fit_thurstonian,
    _preference_prob,
)
from src.fitting.thurstonian_fitting.active_learning import (
    ActiveLearningState,
    generate_d_regular_pairs,
    select_next_pairs,
    check_convergence,
)


@dataclass
class ComparisonMetrics:
    spearman_rho: float
    held_out_accuracy: float
    pairs_queried: int
    total_pairs: int

    @property
    def efficiency(self) -> float:
        return self.pairs_queried / self.total_pairs if self.total_pairs > 0 else 0.0


@dataclass
class ConvergenceTrajectory:
    cumulative_pairs: list[int] = field(default_factory=list)
    spearman_vs_full_mle: list[float] = field(default_factory=list)
    held_out_accuracy: list[float] = field(default_factory=list)
    spearman_vs_true: list[float] | None = None


@dataclass
class SyntheticScenario:
    tasks: list[Task]
    true_mu: np.ndarray
    true_sigma: np.ndarray
    seed: int


@dataclass
class ComparisonResult:
    scenario: SyntheticScenario | None
    full_mle_result: ThurstonianResult
    al_final_result: ThurstonianResult
    trajectory: ConvergenceTrajectory
    al_vs_full_mle: ComparisonMetrics
    full_mle_held_out_accuracy: float


def compute_held_out_accuracy(
    result: ThurstonianResult,
    held_out_comparisons: list[BinaryPreferenceMeasurement],
) -> float:
    if not held_out_comparisons:
        return 1.0

    correct = 0
    for c in held_out_comparisons:
        prob_a = result.preference_probability(c.task_a, c.task_b)
        predicted = "a" if prob_a >= 0.5 else "b"
        if predicted == c.choice:
            correct += 1

    return correct / len(held_out_comparisons)


def _generate_synthetic_scenario(
    n_tasks: int,
    mu_range: tuple[float, float] = (-3.0, 3.0),
    sigma_range: tuple[float, float] = (0.5, 1.5),
    seed: int = 42,
) -> SyntheticScenario:
    rng = np.random.default_rng(seed)

    tasks = [
        Task(
            prompt=f"Synthetic task {i}",
            origin=OriginDataset.WILDCHAT,
            id=f"synthetic_{i}",
            metadata={},
        )
        for i in range(n_tasks)
    ]

    true_mu = rng.uniform(mu_range[0], mu_range[1], n_tasks)
    true_sigma = rng.uniform(sigma_range[0], sigma_range[1], n_tasks)

    return SyntheticScenario(
        tasks=tasks,
        true_mu=true_mu,
        true_sigma=true_sigma,
        seed=seed,
    )


def _simulate_oracle(
    task_a: Task,
    task_b: Task,
    true_mu: np.ndarray,
    true_sigma: np.ndarray,
    id_to_idx: dict[str, int],
    rng: np.random.Generator,
) -> BinaryPreferenceMeasurement:
    mu_a = true_mu[id_to_idx[task_a.id]]
    mu_b = true_mu[id_to_idx[task_b.id]]
    sigma_a = true_sigma[id_to_idx[task_a.id]]
    sigma_b = true_sigma[id_to_idx[task_b.id]]

    prob_a_wins = _preference_prob(mu_a, mu_b, sigma_a, sigma_b)
    choice = "a" if rng.random() < prob_a_wins else "b"

    return BinaryPreferenceMeasurement(
        task_a=task_a,
        task_b=task_b,
        choice=choice,
        preference_type=PreferenceType.PRE_TASK_STATED,
    )


def _split_edges(
    all_pairs: list[tuple[Task, Task]],
    held_out_fraction: float,
    rng: np.random.Generator,
) -> tuple[list[tuple[Task, Task]], list[tuple[Task, Task]]]:
    pairs = list(all_pairs)
    rng.shuffle(pairs)
    n_held_out = int(len(pairs) * held_out_fraction)
    return pairs[n_held_out:], pairs[:n_held_out]


def run_synthetic_comparison(
    n_tasks: int = 20,
    held_out_fraction: float = 0.2,
    n_comparisons_per_pair: int = 5,
    initial_degree: int = 3,
    batch_size: int = 10,
    max_iterations: int = 30,
    convergence_threshold: float | None = 0.995,
    seed: int = 42,
) -> ComparisonResult:
    rng = np.random.default_rng(seed)
    scenario = _generate_synthetic_scenario(n_tasks, seed=seed)
    tasks = scenario.tasks
    id_to_idx = {t.id: i for i, t in enumerate(tasks)}

    # All possible pairs
    all_pairs = list(combinations(tasks, 2))
    total_pairs = len(all_pairs)

    # Split into train and held-out edges
    train_pairs, held_out_pairs = _split_edges(all_pairs, held_out_fraction, rng)

    # Generate comparisons for all train pairs (for full MLE)
    train_comparisons: list[BinaryPreferenceMeasurement] = []
    for a, b in train_pairs:
        for _ in range(n_comparisons_per_pair):
            train_comparisons.append(
                _simulate_oracle(a, b, scenario.true_mu, scenario.true_sigma, id_to_idx, rng)
            )

    # Generate held-out comparisons
    held_out_comparisons: list[BinaryPreferenceMeasurement] = []
    for a, b in held_out_pairs:
        for _ in range(n_comparisons_per_pair):
            held_out_comparisons.append(
                _simulate_oracle(a, b, scenario.true_mu, scenario.true_sigma, id_to_idx, rng)
            )

    # --- Full MLE baseline ---
    full_mle_data = PairwiseData.from_comparisons(train_comparisons, tasks)
    full_mle_result = fit_thurstonian(full_mle_data)
    full_mle_held_out_acc = compute_held_out_accuracy(full_mle_result, held_out_comparisons)

    # --- Active Learning ---
    # Use a fresh RNG for AL to not affect reproducibility
    al_rng = np.random.default_rng(seed + 1000)

    state = ActiveLearningState(tasks=tasks)
    trajectory = ConvergenceTrajectory(spearman_vs_true=[])

    # Get initial pairs (d-regular graph, but only from train pairs)
    train_pairs_set = {tuple(sorted([a.id, b.id])) for a, b in train_pairs}
    initial_pairs = generate_d_regular_pairs(tasks, d=initial_degree, rng=al_rng)
    # Filter to only train pairs
    initial_pairs = [
        (a, b) for a, b in initial_pairs
        if tuple(sorted([a.id, b.id])) in train_pairs_set
    ]

    # Generate initial comparisons
    initial_comparisons = []
    for a, b in initial_pairs:
        for _ in range(n_comparisons_per_pair):
            initial_comparisons.append(
                _simulate_oracle(a, b, scenario.true_mu, scenario.true_sigma, id_to_idx, al_rng)
            )
    state.add_comparisons(initial_comparisons)
    state.fit()
    state.iteration = 1

    # Record initial metrics
    trajectory.cumulative_pairs.append(len(state.sampled_pairs))
    trajectory.spearman_vs_full_mle.append(
        float(spearmanr(state.current_fit.mu, full_mle_result.mu).correlation)
    )
    trajectory.held_out_accuracy.append(
        compute_held_out_accuracy(state.current_fit, held_out_comparisons)
    )
    trajectory.spearman_vs_true.append(
        float(spearmanr(state.current_fit.mu, scenario.true_mu).correlation)
    )

    # Active learning loop
    for iteration in range(1, max_iterations + 1):
        # Check convergence (rank correlation between successive iterations)
        if convergence_threshold is not None:
            converged, corr = check_convergence(state, threshold=convergence_threshold)
            if converged:
                break

        # Get unsampled pairs that are in train set
        unsampled = state.get_unsampled_pairs()
        unsampled_train = [
            (a, b) for a, b in unsampled
            if tuple(sorted([a.id, b.id])) in train_pairs_set
        ]

        if not unsampled_train:
            break

        # Select next pairs (filter to train set after selection)
        next_pairs = select_next_pairs(state, batch_size=batch_size, rng=al_rng)
        next_pairs = [
            (a, b) for a, b in next_pairs
            if tuple(sorted([a.id, b.id])) in train_pairs_set
        ]

        if not next_pairs:
            break

        # Generate comparisons
        new_comparisons = []
        for a, b in next_pairs:
            for _ in range(n_comparisons_per_pair):
                new_comparisons.append(
                    _simulate_oracle(a, b, scenario.true_mu, scenario.true_sigma, id_to_idx, al_rng)
                )
        state.add_comparisons(new_comparisons)
        state.fit()
        state.iteration = iteration + 1

        # Record metrics
        trajectory.cumulative_pairs.append(len(state.sampled_pairs))
        trajectory.spearman_vs_full_mle.append(
            float(spearmanr(state.current_fit.mu, full_mle_result.mu).correlation)
        )
        trajectory.held_out_accuracy.append(
            compute_held_out_accuracy(state.current_fit, held_out_comparisons)
        )
        trajectory.spearman_vs_true.append(
            float(spearmanr(state.current_fit.mu, scenario.true_mu).correlation)
        )

    # Final metrics
    al_final_result = state.current_fit
    final_spearman = float(spearmanr(al_final_result.mu, full_mle_result.mu).correlation)
    final_held_out_acc = compute_held_out_accuracy(al_final_result, held_out_comparisons)

    return ComparisonResult(
        scenario=scenario,
        full_mle_result=full_mle_result,
        al_final_result=al_final_result,
        trajectory=trajectory,
        al_vs_full_mle=ComparisonMetrics(
            spearman_rho=final_spearman,
            held_out_accuracy=final_held_out_acc,
            pairs_queried=len(state.sampled_pairs),
            total_pairs=len(train_pairs),
        ),
        full_mle_held_out_accuracy=full_mle_held_out_acc,
    )


def run_real_data_comparison(
    comparisons: list[BinaryPreferenceMeasurement],
    tasks: list[Task],
    held_out_fraction: float = 0.2,
    initial_degree: int = 3,
    batch_size: int = 10,
    max_iterations: int = 30,
    n_comparisons_per_pair: int = 1,
    convergence_threshold: float | None = 0.995,
    seed: int = 42,
) -> ComparisonResult:
    rng = np.random.default_rng(seed)

    # Group comparisons by pair
    pair_to_comparisons: dict[tuple[str, str], list[BinaryPreferenceMeasurement]] = {}
    for c in comparisons:
        key = tuple(sorted([c.task_a.id, c.task_b.id]))
        if key not in pair_to_comparisons:
            pair_to_comparisons[key] = []
        pair_to_comparisons[key].append(c)

    # Get all observed pairs as (Task, Task) tuples
    id_to_task = {t.id: t for t in tasks}
    all_observed_pairs = [
        (id_to_task[a], id_to_task[b]) for a, b in pair_to_comparisons.keys()
    ]
    total_pairs = len(all_observed_pairs)

    # Split into train and held-out edges
    train_pairs, held_out_pairs = _split_edges(all_observed_pairs, held_out_fraction, rng)
    train_pairs_set = {tuple(sorted([a.id, b.id])) for a, b in train_pairs}
    held_out_pairs_set = {tuple(sorted([a.id, b.id])) for a, b in held_out_pairs}

    # Get comparisons for train and held-out
    train_comparisons = [
        c for c in comparisons
        if tuple(sorted([c.task_a.id, c.task_b.id])) in train_pairs_set
    ]
    held_out_comparisons = [
        c for c in comparisons
        if tuple(sorted([c.task_a.id, c.task_b.id])) in held_out_pairs_set
    ]

    # --- Full MLE baseline ---
    full_mle_data = PairwiseData.from_comparisons(train_comparisons, tasks)
    full_mle_result = fit_thurstonian(full_mle_data)
    full_mle_held_out_acc = compute_held_out_accuracy(full_mle_result, held_out_comparisons)

    # --- Active Learning ---
    al_rng = np.random.default_rng(seed + 1000)

    state = ActiveLearningState(tasks=tasks)
    trajectory = ConvergenceTrajectory()

    # Get initial pairs (d-regular graph, but only from train pairs)
    initial_pairs = generate_d_regular_pairs(tasks, d=initial_degree, rng=al_rng)
    initial_pairs = [
        (a, b) for a, b in initial_pairs
        if tuple(sorted([a.id, b.id])) in train_pairs_set
    ]

    # Use real comparisons for initial pairs
    initial_comparisons = []
    for a, b in initial_pairs:
        key = tuple(sorted([a.id, b.id]))
        if key in pair_to_comparisons:
            # Take up to n_comparisons_per_pair
            initial_comparisons.extend(pair_to_comparisons[key][:n_comparisons_per_pair])

    if not initial_comparisons:
        raise ValueError("No comparisons found for initial pairs")

    state.add_comparisons(initial_comparisons)
    state.fit()
    state.iteration = 1

    # Record initial metrics
    trajectory.cumulative_pairs.append(len(state.sampled_pairs))
    trajectory.spearman_vs_full_mle.append(
        float(spearmanr(state.current_fit.mu, full_mle_result.mu).correlation)
    )
    trajectory.held_out_accuracy.append(
        compute_held_out_accuracy(state.current_fit, held_out_comparisons)
    )

    # Active learning loop
    for iteration in range(1, max_iterations + 1):
        # Check convergence (rank correlation between successive iterations)
        if convergence_threshold is not None:
            converged, corr = check_convergence(state, threshold=convergence_threshold)
            if converged:
                break

        # Get unsampled pairs that are in train set
        unsampled = state.get_unsampled_pairs()
        unsampled_train = [
            (a, b) for a, b in unsampled
            if tuple(sorted([a.id, b.id])) in train_pairs_set
        ]

        if not unsampled_train:
            break

        # Select next pairs
        next_pairs = select_next_pairs(state, batch_size=batch_size, rng=al_rng)
        next_pairs = [
            (a, b) for a, b in next_pairs
            if tuple(sorted([a.id, b.id])) in train_pairs_set
        ]

        if not next_pairs:
            break

        # Get real comparisons for selected pairs
        new_comparisons = []
        for a, b in next_pairs:
            key = tuple(sorted([a.id, b.id]))
            if key in pair_to_comparisons:
                new_comparisons.extend(pair_to_comparisons[key][:n_comparisons_per_pair])

        if not new_comparisons:
            break

        state.add_comparisons(new_comparisons)
        state.fit()
        state.iteration = iteration + 1

        # Record metrics
        trajectory.cumulative_pairs.append(len(state.sampled_pairs))
        trajectory.spearman_vs_full_mle.append(
            float(spearmanr(state.current_fit.mu, full_mle_result.mu).correlation)
        )
        trajectory.held_out_accuracy.append(
            compute_held_out_accuracy(state.current_fit, held_out_comparisons)
        )

    # Final metrics
    al_final_result = state.current_fit
    final_spearman = float(spearmanr(al_final_result.mu, full_mle_result.mu).correlation)
    final_held_out_acc = compute_held_out_accuracy(al_final_result, held_out_comparisons)

    return ComparisonResult(
        scenario=None,
        full_mle_result=full_mle_result,
        al_final_result=al_final_result,
        trajectory=trajectory,
        al_vs_full_mle=ComparisonMetrics(
            spearman_rho=final_spearman,
            held_out_accuracy=final_held_out_acc,
            pairs_queried=len(state.sampled_pairs),
            total_pairs=len(train_pairs),
        ),
        full_mle_held_out_accuracy=full_mle_held_out_acc,
    )
