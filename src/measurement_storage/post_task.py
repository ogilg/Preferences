"""Storage for post-task measurements."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from src.measurement_storage.base import load_yaml, save_yaml

if TYPE_CHECKING:
    from src.types import BinaryPreferenceMeasurement, TaskScore


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
            f"post_stated_{template_name}_{model_name}_{response_format}"
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
            f"post_revealed_{template_name}_{model_name}_{response_format}_{order}"
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
