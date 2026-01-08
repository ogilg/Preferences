from __future__ import annotations

from pathlib import Path
from typing import Literal

from src.models import OpenAICompatibleClient
from src.preferences.storage.base import (
    load_yaml,
    model_short_name,
    save_yaml,
)
from src.preferences.templates.template import PromptTemplate
from src.task_data import Task
from src.types import BinaryPreferenceMeasurement, PreferenceType


MEASUREMENTS_DIR = Path("results/measurements")


class MeasurementCache:
    """Cache for binary preference measurements keyed by (template, model).

    Storage format:
        measurements/{template_name}_{model_short}/
            config.yaml        # template + model metadata
            measurements.yaml  # [{task_a, task_b, choice}, ...]

    Order matters: (a, b) and (b, a) are distinct pairs.
    """

    def __init__(
        self,
        template: PromptTemplate,
        client: OpenAICompatibleClient,
        results_dir: Path = MEASUREMENTS_DIR,
    ):
        self.template = template
        self.client = client
        self.model_short = model_short_name(client.canonical_model_name)
        self.results_dir = Path(results_dir)
        self.cache_dir = self.results_dir / f"{template.name}_{self.model_short}"
        self._measurements_path = self.cache_dir / "measurements.yaml"
        self._config_path = self.cache_dir / "config.yaml"

    def get_existing_pairs(self) -> set[tuple[str, str]]:
        """Return ordered pairs we have measurements for.

        Order matters: (a, b) and (b, a) are distinct.
        """
        if not self._measurements_path.exists():
            return set()

        data = load_yaml(self._measurements_path)
        return {(m["task_a"], m["task_b"]) for m in data}

    def get_measurements(
        self,
        task_ids: set[str] | None = None,
    ) -> list[dict[str, str]]:
        """Load measurements, optionally filtered to pairs where both tasks in task_ids.

        Returns list of {task_a: str, task_b: str, choice: str} dicts.
        """
        if not self._measurements_path.exists():
            return []

        data = load_yaml(self._measurements_path)

        if task_ids is not None:
            data = [m for m in data if m["task_a"] in task_ids and m["task_b"] in task_ids]

        return data

    def append(self, measurements: list[BinaryPreferenceMeasurement]) -> None:
        """Append new measurements to cache."""
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
            self._ensure_config()

        save_yaml(new_data, self._measurements_path)

    def _ensure_config(self) -> None:
        """Create config file if it doesn't exist."""
        if self._config_path.exists():
            return

        config = {
            "template_name": self.template.name,
            "template_tags": self.template.tags_dict,
            "model": self.client.model_name,
            "model_short": self.model_short,
        }
        save_yaml(config, self._config_path)


def save_measurements(measurements: list[BinaryPreferenceMeasurement], path: Path | str) -> None:
    """Serialize measurements to YAML."""
    data = [{"task_a": m.task_a.id, "task_b": m.task_b.id, "choice": m.choice} for m in measurements]
    save_yaml(data, Path(path))


def reconstruct_measurements(
    raw: list[dict[str, str]],
    tasks: dict[str, Task],
    preference_type: PreferenceType = PreferenceType.PRE_TASK_STATED,
) -> list[BinaryPreferenceMeasurement]:
    """Reconstruct BinaryPreferenceMeasurement objects from raw dicts.

    Args:
        raw: List of {task_a, task_b, choice} dicts
        tasks: Mapping from task ID to Task object
        preference_type: Type to assign (not stored in cache)
    """
    return [
        BinaryPreferenceMeasurement(
            task_a=tasks[m["task_a"]],
            task_b=tasks[m["task_b"]],
            choice=m["choice"],
            preference_type=preference_type,
        )
        for m in raw
    ]
