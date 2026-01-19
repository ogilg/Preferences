"""Tests for experiment config."""

import pytest
from pathlib import Path

import numpy as np

from src.running_measurements.config import ExperimentConfig
from src.task_data import OriginDataset, Task
from src.types import BinaryPreferenceMeasurement, PreferenceType
from src.thurstonian_fitting import PairwiseData, fit_thurstonian


class TestShuffleAndReverseValidation:

    def test_cannot_set_both_shuffle_and_reverse(self):
        """Setting both task_shuffle_seed and include_reverse_order raises error."""
        with pytest.raises(ValueError, match="Cannot set both"):
            ExperimentConfig(
                preference_mode="pre_task_revealed",
                n_tasks=5,
                task_origins=["wildchat"],
                templates=Path("dummy.yaml"),
                task_shuffle_seed=42,
                include_reverse_order=True,
            )

    def test_shuffle_without_reverse_is_valid(self):
        """task_shuffle_seed alone is valid."""
        config = ExperimentConfig(
            preference_mode="pre_task_revealed",
            n_tasks=5,
            task_origins=["wildchat"],
            templates=Path("dummy.yaml"),
            task_shuffle_seed=42,
            include_reverse_order=False,
        )
        assert config.task_shuffle_seed == 42

    def test_reverse_without_shuffle_is_valid(self):
        """include_reverse_order alone is valid."""
        config = ExperimentConfig(
            preference_mode="pre_task_revealed",
            n_tasks=5,
            task_origins=["wildchat"],
            templates=Path("dummy.yaml"),
            task_shuffle_seed=None,
            include_reverse_order=True,
        )
        assert config.include_reverse_order is True

    def test_neither_shuffle_nor_reverse_is_valid(self):
        """Neither option set is valid (defaults)."""
        config = ExperimentConfig(
            preference_mode="pre_task_revealed",
            n_tasks=5,
            task_origins=["wildchat"],
            templates=Path("dummy.yaml"),
        )
        assert config.task_shuffle_seed is None
        assert config.include_reverse_order is False


class TestShuffledTasksInThurstonian:
    """Verify Thurstonian fitting uses task IDs correctly regardless of order."""

    def _make_tasks(self, n: int) -> list[Task]:
        return [
            Task(id=f"task_{i}", prompt=f"Task {i}", origin=OriginDataset.SYNTHETIC, metadata={})
            for i in range(n)
        ]

    def _make_comparisons(
        self, tasks: list[Task], winner_indices: list[tuple[int, int]]
    ) -> list[BinaryPreferenceMeasurement]:
        """Create comparisons where first index always wins."""
        comparisons = []
        for i, j in winner_indices:
            comparisons.append(BinaryPreferenceMeasurement(
                task_a=tasks[i],
                task_b=tasks[j],
                choice="a",
                preference_type=PreferenceType.PRE_TASK_REVEALED,
            ))
        return comparisons

    def test_thurstonian_uses_task_ids_not_positions(self):
        """Fitting with shuffled task list gives same ranking by task ID.

        Note: Absolute utilities differ because the model fixes the first task's
        utility to 0 for identification. But relative utilities (and thus rankings)
        should be identical.
        """
        tasks = self._make_tasks(4)

        # Task 0 beats everyone, task 1 beats 2 and 3, task 2 beats 3
        # Expected ranking: 0 > 1 > 2 > 3
        winner_pairs = [
            (0, 1), (0, 2), (0, 3),
            (1, 2), (1, 3),
            (2, 3),
        ]
        # Repeat each comparison to get stable estimates
        winner_pairs = winner_pairs * 5
        comparisons = self._make_comparisons(tasks, winner_pairs)

        # Fit with original order
        data_original = PairwiseData.from_comparisons(comparisons, tasks)
        result_original = fit_thurstonian(data_original, max_iter=2000)

        # Shuffle tasks (reverse order)
        tasks_reversed = list(reversed(tasks))
        data_shuffled = PairwiseData.from_comparisons(comparisons, tasks_reversed)
        result_shuffled = fit_thurstonian(data_shuffled, max_iter=2000)

        # Ranking should be the same regardless of task order
        ranking_original = [t.id for t in result_original.ranking()]
        ranking_shuffled = [t.id for t in result_shuffled.ranking()]
        assert ranking_original == ranking_shuffled

        # Pairwise preference probabilities should match (within optimization tolerance)
        for i, task_i in enumerate(tasks):
            for task_j in tasks[i + 1:]:
                prob_original = result_original.preference_probability(task_i, task_j)
                prob_shuffled = result_shuffled.preference_probability(task_i, task_j)
                assert np.isclose(prob_original, prob_shuffled, atol=0.02), (
                    f"P({task_i.id} > {task_j.id}): original={prob_original:.3f}, shuffled={prob_shuffled:.3f}"
                )

    def test_pairwise_data_uses_task_ids(self):
        """PairwiseData.from_comparisons maps by task ID."""
        tasks = self._make_tasks(3)
        comparisons = self._make_comparisons(tasks, [(0, 1), (0, 2), (1, 2)])

        # Original order
        data_original = PairwiseData.from_comparisons(comparisons, tasks)

        # Reversed order
        tasks_reversed = list(reversed(tasks))
        data_reversed = PairwiseData.from_comparisons(comparisons, tasks_reversed)

        # Win counts should be consistent: task_0 has 2 wins, task_1 has 1 win, task_2 has 0 wins
        # In original order [0,1,2]: wins[0,1]=1, wins[0,2]=1, wins[1,2]=1
        # In reversed order [2,1,0]: wins[2,1]=1, wins[2,0]=1, wins[1,0]=1

        # Get win counts per task
        def get_wins_for_task(data: PairwiseData, task_id: str) -> int:
            idx = next(i for i, t in enumerate(data.tasks) if t.id == task_id)
            return int(data.wins[idx, :].sum())

        for task in tasks:
            wins_original = get_wins_for_task(data_original, task.id)
            wins_reversed = get_wins_for_task(data_reversed, task.id)
            assert wins_original == wins_reversed, (
                f"Task {task.id}: wins differ (original={wins_original}, reversed={wins_reversed})"
            )
