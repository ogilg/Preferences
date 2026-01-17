"""Storage for post-task measurements."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

from src.measurement_storage.base import load_yaml, save_yaml

if TYPE_CHECKING:
    from src.task_data import Task
    from src.types import BinaryPreferenceMeasurement, MeasurementBatch, TaskScore


POST_STATED_DIR = Path("results/post_task_stated")
POST_REVEALED_DIR = Path("results/post_task_revealed")


class PostStatedCache:
    """Cache for post-task stated measurements."""

    def __init__(
        self,
        model_name: str,
        template_name: str,
        response_format: str,
        completion_seed: int,
        rating_seed: int,
    ):
        self.cache_dir = POST_STATED_DIR / (
            f"{template_name}_{model_name}_{response_format}"
            f"_cseed{completion_seed}_rseed{rating_seed}"
        )
        self._measurements_path = self.cache_dir / "measurements.yaml"
        self._config_path = self.cache_dir / "config.yaml"

    def exists(self) -> bool:
        return self._measurements_path.exists()

    def save(self, scores: list[TaskScore], config: dict) -> Path:
        if not self._config_path.exists():
            save_yaml(config, self._config_path)
        save_yaml(
            [{"task_id": s.task.id, "score": s.score} for s in scores],
            self._measurements_path,
        )
        return self.cache_dir


class PostRevealedCache:
    """Cache for post-task revealed measurements."""

    def __init__(
        self,
        model_name: str,
        template_name: str,
        response_format: str,
        order: str,
        completion_seed: int,
        rating_seed: int,
    ):
        self.cache_dir = POST_REVEALED_DIR / (
            f"{template_name}_{model_name}_{response_format}_{order}"
            f"_cseed{completion_seed}_rseed{rating_seed}"
        )
        self._measurements_path = self.cache_dir / "measurements.yaml"
        self._config_path = self.cache_dir / "config.yaml"

    def exists(self) -> bool:
        return self._measurements_path.exists()

    def get_existing_pairs(self) -> set[tuple[str, str]]:
        if not self._measurements_path.exists():
            return set()
        data = load_yaml(self._measurements_path)
        return {(m["task_a"], m["task_b"]) for m in data}

    def append(self, measurements: list[BinaryPreferenceMeasurement], config: dict) -> None:
        if not measurements:
            return

        new_data = [
            {"task_a": m.task_a.id, "task_b": m.task_b.id, "choice": m.choice}
            for m in measurements
        ]

        if self._measurements_path.exists():
            existing = load_yaml(self._measurements_path)
            new_data = existing + new_data
        else:
            save_yaml(config, self._config_path)

        save_yaml(new_data, self._measurements_path)

    def get_measurements(
        self,
        task_ids: set[str] | None = None,
    ) -> list[dict[str, str]]:
        """Load measurements, optionally filtered to pairs where both tasks in task_ids."""
        if not self._measurements_path.exists():
            return []

        data = load_yaml(self._measurements_path)

        if task_ids is not None:
            data = [m for m in data if m["task_a"] in task_ids and m["task_b"] in task_ids]

        return data

    def get_or_measure_post_task(
        self,
        pairs: list[tuple[Task, Task]],
        completion_lookup: dict[str, str],
        measure_fn: Callable[[list[tuple[Task, Task, str, str]]], MeasurementBatch],
        task_lookup: dict[str, Task],
        config: dict,
    ) -> tuple[MeasurementBatch, int, int]:
        """Check cache for each pair, call measure_fn for misses, return combined.

        Returns (batch, cache_hits, api_queries).
        """
        from src.measurement_storage.cache import reconstruct_measurements
        from src.types import MeasurementBatch

        if not pairs:
            return MeasurementBatch(successes=[], failures=[]), 0, 0

        cached_raw = self.get_measurements()
        cached_by_pair: dict[tuple[str, str], list[dict[str, str]]] = {}
        for m in cached_raw:
            cached_by_pair.setdefault((m["task_a"], m["task_b"]), []).append(m)

        cached_hits_raw: list[dict[str, str]] = []
        to_query: list[tuple[Task, Task]] = []

        for a, b in pairs:
            key = (a.id, b.id)
            if key in cached_by_pair and cached_by_pair[key]:
                cached_hits_raw.append(cached_by_pair[key].pop())
            else:
                to_query.append((a, b))

        cached_hits = reconstruct_measurements(cached_hits_raw, task_lookup)

        if to_query:
            # Build data tuples with completions
            data = [
                (a, b, completion_lookup[a.id], completion_lookup[b.id])
                for a, b in to_query
            ]
            fresh_batch = measure_fn(data)
            self.append(fresh_batch.successes, config)
        else:
            fresh_batch = MeasurementBatch(successes=[], failures=[])

        combined_successes = cached_hits + fresh_batch.successes
        return (
            MeasurementBatch(successes=combined_successes, failures=fresh_batch.failures),
            len(cached_hits),
            len(to_query),
        )
