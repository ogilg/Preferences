"""Storage for post-task measurements using unified caches."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

from src.measurement_storage.unified_cache import StatedCache, RevealedCache, template_config_from_template

if TYPE_CHECKING:
    from src.prompt_templates.template import PromptTemplate
    from src.task_data import Task
    from src.types import BinaryPreferenceMeasurement, MeasurementBatch, TaskScore


POST_STATED_DIR = Path("results/post_task_stated")
POST_REVEALED_DIR = Path("results/post_task_revealed")


class PostStatedCache:
    """Cache for post-task stated measurements using unified StatedCache."""

    def __init__(
        self,
        model_name: str,
        template: PromptTemplate,
        response_format: str,
        completion_seed: int,
        rating_seed: int,
    ):
        self._cache = StatedCache(model_name)
        self._template_config = template_config_from_template(template)
        self._response_format = response_format
        self._completion_seed = completion_seed
        self._rating_seed = rating_seed

    def get_existing_task_ids(self) -> set[str]:
        """Get task IDs that have been measured."""
        return self._cache.get_task_ids(
            template_config=self._template_config,
            response_format=self._response_format,
            rating_seed=self._rating_seed,
            completion_seed=self._completion_seed,
        )

    def save(self, scores: list[TaskScore]) -> None:
        """Save scores to cache."""
        for s in scores:
            self._cache.add(
                template_config=self._template_config,
                response_format=self._response_format,
                rating_seed=self._rating_seed,
                task_id=s.task.id,
                sample={"score": s.score},
                completion_seed=self._completion_seed,
            )
        self._cache.save()


class PostRevealedCache:
    """Cache for post-task revealed measurements using unified RevealedCache."""

    def __init__(
        self,
        model_name: str,
        template: PromptTemplate,
        response_format: str,
        order: str,
        completion_seed: int,
        rating_seed: int,
    ):
        self._cache = RevealedCache(model_name)
        self._template_config = template_config_from_template(template)
        self._response_format = response_format
        self._order = order
        self._completion_seed = completion_seed
        self._rating_seed = rating_seed

    def get_existing_pairs(self) -> set[tuple[str, str]]:
        return self._cache.get_pairs(
            template_config=self._template_config,
            response_format=self._response_format,
            order=self._order,
            rating_seed=self._rating_seed,
            completion_seed=self._completion_seed,
        )

    def append(self, measurements: list[BinaryPreferenceMeasurement]) -> None:
        if not measurements:
            return

        for m in measurements:
            self._cache.add(
                template_config=self._template_config,
                response_format=self._response_format,
                order=self._order,
                rating_seed=self._rating_seed,
                task_a_id=m.task_a.id,
                task_b_id=m.task_b.id,
                sample={"choice": m.choice},
                completion_seed=self._completion_seed,
            )
        self._cache.save()

    def get_measurements(
        self,
        task_ids: set[str] | None = None,
    ) -> list[dict[str, str]]:
        """Load measurements, optionally filtered to pairs where both tasks in task_ids."""
        return self._cache.get_measurements(
            template_config=self._template_config,
            response_format=self._response_format,
            order=self._order,
            rating_seed=self._rating_seed,
            task_ids=task_ids,
            completion_seed=self._completion_seed,
        )

    def get_or_measure_post_task(
        self,
        pairs: list[tuple[Task, Task]],
        completion_lookup: dict[str, str],
        measure_fn: Callable[[list[tuple[Task, Task, str, str]]], MeasurementBatch],
        task_lookup: dict[str, Task],
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
            data = [
                (a, b, completion_lookup[a.id], completion_lookup[b.id])
                for a, b in to_query
            ]
            fresh_batch = measure_fn(data)
            self.append(fresh_batch.successes)
        else:
            fresh_batch = MeasurementBatch(successes=[], failures=[])

        combined_successes = cached_hits + fresh_batch.successes
        return (
            MeasurementBatch(successes=combined_successes, failures=fresh_batch.failures),
            len(cached_hits),
            len(to_query),
        )
