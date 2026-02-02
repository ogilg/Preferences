from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Literal

from src.models import OpenAICompatibleClient
from src.measurement.storage.base import model_short_name
from src.measurement.storage.unified_cache import RevealedCache, template_config_from_template
from src.measurement.elicitation.prompt_templates.template import PromptTemplate
from src.task_data import Task
from src.types import BinaryPreferenceMeasurement, MeasurementBatch, MeasurementFailure, PreferenceType
from src.measurement.storage.failures import FailureLog

ResponseFormatName = Literal["regex", "tool_use"]
OrderName = Literal["canonical", "reversed"]


@dataclass
class MeasurementStats:
    """Stats from a measurement operation."""
    cache_hits: int = 0
    api_successes: int = 0
    api_failures: int = 0
    failures: list[MeasurementFailure] | None = None

    def __post_init__(self):
        if self.failures is None:
            self.failures = []

    @property
    def total_successes(self) -> int:
        return self.cache_hits + self.api_successes

    def failure_counts(self) -> dict[str, int]:
        """Get failure counts by category."""
        counts: dict[str, int] = {}
        for f in self.failures:
            cat = f.category.value
            counts[cat] = counts.get(cat, 0) + 1
        return counts

    def __iadd__(self, other: MeasurementStats) -> MeasurementStats:
        self.cache_hits += other.cache_hits
        self.api_successes += other.api_successes
        self.api_failures += other.api_failures
        self.failures.extend(other.failures)
        return self


class MeasurementCache:
    """Cache for binary preference measurements using unified RevealedCache.

    Storage: results/cache/revealed/{model_short}.yaml
    """

    def __init__(
        self,
        template: PromptTemplate,
        client: OpenAICompatibleClient,
        response_format: ResponseFormatName = "regex",
        order: OrderName = "canonical",
        seed: int | None = None,
        completion_seed: int | None = None,
    ):
        self.template = template
        self.client = client
        self.response_format = response_format
        self.order = order
        self.seed = seed
        self.completion_seed = completion_seed
        self.model_short = model_short_name(client.canonical_model_name)

        self._cache = RevealedCache(client.canonical_model_name)
        self._template_config = template_config_from_template(template)
        self._rating_seed = seed if seed is not None else 0

    def get_existing_pairs(self) -> set[tuple[str, str]]:
        """Return ordered pairs we have measurements for."""
        return self._cache.get_pairs(
            template_config=self._template_config,
            response_format=self.response_format,
            order=self.order,
            rating_seed=self._rating_seed,
            completion_seed=self.completion_seed,
        )

    def get_measurements(
        self,
        task_ids: set[str] | None = None,
    ) -> list[dict[str, str]]:
        """Load measurements, optionally filtered to pairs where both tasks in task_ids."""
        return self._cache.get_measurements(
            template_config=self._template_config,
            response_format=self.response_format,
            order=self.order,
            rating_seed=self._rating_seed,
            task_ids=task_ids,
            completion_seed=self.completion_seed,
        )

    def append(self, measurements: list[BinaryPreferenceMeasurement]) -> None:
        """Append new measurements to cache."""
        if not measurements:
            return

        for m in measurements:
            self._cache.add(
                template_config=self._template_config,
                response_format=self.response_format,
                order=self.order,
                rating_seed=self._rating_seed,
                task_a_id=m.task_a.id,
                task_b_id=m.task_b.id,
                sample={"choice": m.choice},
                completion_seed=self.completion_seed,
            )

        self._cache.save()

    def _partition_pairs(
        self,
        pairs: list[tuple[Task, Task]],
        task_lookup: dict[str, Task],
    ) -> tuple[list[BinaryPreferenceMeasurement], list[tuple[Task, Task]]]:
        """Split pairs into cached hits and pairs needing API calls."""
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

        return reconstruct_measurements(cached_hits_raw, task_lookup), to_query

    async def get_or_measure_async(
        self,
        pairs: list[tuple[Task, Task]],
        measure_fn: Callable[[list[tuple[Task, Task]]], Awaitable[MeasurementBatch]],
        task_lookup: dict[str, Task],
        failure_log: FailureLog | None = None,
    ) -> tuple[list[BinaryPreferenceMeasurement], MeasurementStats]:
        """Check cache for each pair, call measure_fn for misses, return combined."""
        if not pairs:
            return [], MeasurementStats()

        cached_hits, to_query = self._partition_pairs(pairs, task_lookup)
        stats = MeasurementStats(cache_hits=len(cached_hits))

        if to_query:
            fresh_batch = await measure_fn(to_query)
            stats.api_successes = len(fresh_batch.successes)
            stats.api_failures = len(fresh_batch.failures)
            stats.failures = fresh_batch.failures

            # Persist failures if log provided
            if failure_log and fresh_batch.failures:
                failure_log.append(
                    fresh_batch.failures,
                    run_info={
                        "template": self.template.name,
                        "response_format": self.response_format,
                    },
                )

            self.append(fresh_batch.successes)
            return cached_hits + fresh_batch.successes, stats

        return cached_hits, stats


def save_measurements(measurements: list[BinaryPreferenceMeasurement], path: Path | str) -> None:
    """Serialize measurements to YAML."""
    from src.measurement.storage.base import save_yaml
    data = [{"task_a": m.task_a.id, "task_b": m.task_b.id, "choice": m.choice} for m in measurements]
    save_yaml(data, Path(path))


def reconstruct_measurements(
    raw: list[dict[str, str]],
    tasks: dict[str, Task],
    preference_type: PreferenceType = PreferenceType.PRE_TASK_STATED,
) -> list[BinaryPreferenceMeasurement]:
    """Reconstruct BinaryPreferenceMeasurement objects from raw dicts."""
    return [
        BinaryPreferenceMeasurement(
            task_a=tasks[m["task_a"]],
            task_b=tasks[m["task_b"]],
            choice=m["choice"],
            preference_type=preference_type,
        )
        for m in raw
    ]
