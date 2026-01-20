from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Literal

from src.models import OpenAICompatibleClient
from src.measurement_storage.base import (
    build_measurement_config,
    model_short_name,
    save_yaml,
)
from src.measurement_storage.unified_cache import RevealedCache, template_config_from_template
from src.prompt_templates.template import PromptTemplate
from src.task_data import Task
from src.types import BinaryPreferenceMeasurement, MeasurementBatch, PreferenceType


PRE_TASK_REVEALED_DIR = Path("results/pre_task_revealed")

ResponseFormatName = Literal["regex", "tool_use"]
OrderName = Literal["canonical", "reversed"]


def categorize_failure(error_msg: str) -> str:
    """Categorize a failure message into a bucket."""
    error_lower = error_msg.lower()
    if error_lower.startswith("refusal ("):
        return "refusal"
    if "timeout" in error_lower or "timed out" in error_lower:
        return "timeout"
    if "rate" in error_lower and "limit" in error_lower:
        return "rate_limit"
    if "expected tool call but got text" in error_lower:
        return "tool_use_failure"
    if "connection" in error_lower or "connect" in error_lower:
        return "connection"
    if "content" in error_lower and "filter" in error_lower:
        return "content_filter"
    if "unexpected result type" in error_lower:
        return "parse_type"
    if "request failed" in error_lower:
        if "timeout" in error_lower:
            return "timeout"
        if "rate" in error_lower:
            return "rate_limit"
        return "api_error"
    if any(x in error_lower for x in ["parse", "extract", "invalid", "expected", "match"]):
        return "parse_error"
    return "other"


MAX_EXAMPLES_PER_CATEGORY = 5


@dataclass
class MeasurementStats:
    """Stats from a measurement operation."""
    cache_hits: int = 0
    api_successes: int = 0
    api_failures: int = 0
    failure_categories: dict[str, int] | None = None
    failure_examples: dict[str, list[str]] | None = None

    def __post_init__(self):
        if self.failure_categories is None:
            self.failure_categories = {}
        if self.failure_examples is None:
            self.failure_examples = {}

    @property
    def total_successes(self) -> int:
        return self.cache_hits + self.api_successes

    def __iadd__(self, other: MeasurementStats) -> MeasurementStats:
        self.cache_hits += other.cache_hits
        self.api_successes += other.api_successes
        self.api_failures += other.api_failures
        for cat, count in other.failure_categories.items():
            self.failure_categories[cat] = self.failure_categories.get(cat, 0) + count
        for cat, examples in other.failure_examples.items():
            if cat not in self.failure_examples:
                self.failure_examples[cat] = []
            for ex in examples:
                if len(self.failure_examples[cat]) < MAX_EXAMPLES_PER_CATEGORY:
                    self.failure_examples[cat].append(ex)
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
        results_dir: Path = PRE_TASK_REVEALED_DIR,
    ):
        self.template = template
        self.client = client
        self.response_format = response_format
        self.order = order
        self.seed = seed
        self.model_short = model_short_name(client.canonical_model_name)
        self.results_dir = Path(results_dir)

        self._cache = RevealedCache(client.canonical_model_name)
        self._template_config = template_config_from_template(template)
        self._rating_seed = seed if seed is not None else 0

        # Keep cache_dir for active learning output files
        seed_suffix = f"_seed{seed}" if seed is not None else ""
        self.cache_dir = self.results_dir / f"{template.name}_{self.model_short}_{response_format}_{order}{seed_suffix}"

    def get_existing_pairs(self) -> set[tuple[str, str]]:
        """Return ordered pairs we have measurements for."""
        return self._cache.get_pairs(
            template_config=self._template_config,
            response_format=self.response_format,
            order=self.order,
            rating_seed=self._rating_seed,
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
            for _, error_msg in fresh_batch.failures:
                cat = categorize_failure(error_msg)
                stats.failure_categories[cat] = stats.failure_categories.get(cat, 0) + 1
                if cat not in stats.failure_examples:
                    stats.failure_examples[cat] = []
                if len(stats.failure_examples[cat]) < MAX_EXAMPLES_PER_CATEGORY:
                    stats.failure_examples[cat].append(error_msg)
            self.append(fresh_batch.successes)
            return cached_hits + fresh_batch.successes, stats

        return cached_hits, stats


def save_measurements(measurements: list[BinaryPreferenceMeasurement], path: Path | str) -> None:
    """Serialize measurements to YAML."""
    from src.measurement_storage.base import save_yaml
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
