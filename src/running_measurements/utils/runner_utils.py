"""Shared utilities for experiment runners."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.measurement_storage.base import find_project_root
from src.types import MeasurementFailure


def get_activation_completions_path() -> Path:
    """Get path to activation completions file."""
    return find_project_root() / "probe_data" / "activations" / "completions_with_activations.json"


def load_activation_task_ids() -> set[str]:
    """Load task IDs from activation extraction completions file."""
    import json
    path = get_activation_completions_path()
    if not path.exists():
        raise FileNotFoundError(f"Activation completions not found at {path}")
    with open(path) as f:
        data = json.load(f)
    return {c["task_id"] for c in data}


def _get_activation_completions_path(use_tasks_with_activations: bool) -> Path | None:
    """Get path to activation completions if using activation tasks."""
    if use_tasks_with_activations:
        path = get_activation_completions_path()
        if path.exists():
            return path
    return None


@dataclass
class RunnerStats:
    total_runs: int = 0
    completed: int = 0
    successes: int = 0
    failures: int = 0
    cache_hits: int = 0
    skipped: int = 0
    all_failures: list[MeasurementFailure] | None = None

    def __post_init__(self):
        if self.all_failures is None:
            self.all_failures = []

    def failure_counts(self) -> dict[str, int]:
        """Get failure counts by category."""
        counts: dict[str, int] = {}
        for f in self.all_failures:
            cat = f.category.value
            counts[cat] = counts.get(cat, 0) + 1
        return counts

    def to_dict(self) -> dict:
        result = {
            "total_runs": self.total_runs,
            "successes": self.successes,
            "failures": self.failures,
            "cache_hits": self.cache_hits,
            "skipped": self.skipped,
        }
        counts = self.failure_counts()
        if counts:
            result["failure_categories"] = counts
        # Build failure examples for debug output (up to 5 per category)
        if self.all_failures:
            examples: dict[str, list[str]] = {}
            for f in self.all_failures:
                cat = f.category.value
                if cat not in examples:
                    examples[cat] = []
                if len(examples[cat]) < 5:
                    examples[cat].append(f.error_message)
            result["failure_examples"] = examples
        return result

    def mark_skipped(self) -> None:
        self.completed += 1
        self.skipped += 1

    def add_from_batch_stats(self, batch_stats) -> None:
        """Add results from a MeasurementStats batch."""
        self.successes += batch_stats.api_successes
        self.failures += batch_stats.api_failures
        self.cache_hits += batch_stats.cache_hits
        self.all_failures.extend(batch_stats.failures)
        self.completed += 1

    def add_batch_with_failures(self, successes: int, failures: list[MeasurementFailure], cache_hits: int = 0) -> None:
        """Add batch results with structured failures."""
        self.successes += successes
        self.failures += len(failures)
        self.cache_hits += cache_hits
        self.all_failures.extend(failures)
        self.completed += 1
