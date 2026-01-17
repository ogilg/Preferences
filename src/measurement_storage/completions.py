from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn

from src.measurement_storage.base import find_project_root, model_short_name, save_yaml, load_yaml
from src.task_data import OriginDataset, Task

from src.models.openai_compatible import GenerateRequest

if TYPE_CHECKING:
    from src.models import OpenAICompatibleClient


COMPLETIONS_DIR = find_project_root() / "results" / "completions"


@dataclass
class TaskCompletion:
    task: Task
    completion: str


def _completions_dir(client: OpenAICompatibleClient, seed: int | None) -> Path:
    short = model_short_name(client.canonical_model_name)
    seed_suffix = f"_seed{seed}" if seed is not None else ""
    return COMPLETIONS_DIR / f"{short}{seed_suffix}"


def _load_json(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _save_json(data: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


class CompletionStore:
    """Storage for task completions with caching support."""

    def __init__(
        self,
        client: OpenAICompatibleClient,
        seed: int | None = None,
    ):
        self.client = client
        self.seed = seed
        self.model_short = model_short_name(client.canonical_model_name)
        self.store_dir = _completions_dir(client, seed)
        self._completions_path = self.store_dir / "completions.json"
        self._config_path = self.store_dir / "config.yaml"

    def exists(self) -> bool:
        return self._completions_path.exists()

    def get_existing_task_ids(self) -> set[str]:
        if not self._completions_path.exists():
            return set()
        data = _load_json(self._completions_path)
        return {c["task_id"] for c in data}

    def load(self, task_lookup: dict[str, Task] | None = None) -> list[TaskCompletion]:
        """Load completions. If task_lookup provided, use those Task objects."""
        data = _load_json(self._completions_path)
        if task_lookup:
            return [
                TaskCompletion(
                    task=task_lookup[c["task_id"]],
                    completion=c["completion"],
                )
                for c in data
                if c["task_id"] in task_lookup
            ]
        # Reconstruct Task from stored data
        return [
            TaskCompletion(
                task=Task(
                    prompt=c["task_prompt"],
                    origin=OriginDataset[c.get("origin", "SYNTHETIC")],
                    id=c["task_id"],
                    metadata={},
                ),
                completion=c["completion"],
            )
            for c in data
        ]

    def save(self, completions: list[TaskCompletion], config: dict) -> Path:
        """Save completions to disk. Appends to existing if present."""
        new_data = [
            {
                "task_id": tc.task.id,
                "task_prompt": tc.task.prompt,
                "completion": tc.completion,
                "origin": tc.task.origin.name,
            }
            for tc in completions
        ]

        if self._completions_path.exists():
            existing = _load_json(self._completions_path)
            existing_ids = {c["task_id"] for c in existing}
            new_data = existing + [c for c in new_data if c["task_id"] not in existing_ids]
        else:
            save_yaml(config, self._config_path)

        _save_json(new_data, self._completions_path)
        return self.store_dir


def generate_completions(
    client: OpenAICompatibleClient,
    tasks: list[Task],
    temperature: float = 1.0,
    max_concurrent: int = 10,
    seed: int | None = None,
) -> list[TaskCompletion]:
    """Generate single-turn completions for tasks."""
    requests = [
        GenerateRequest(
            messages=[{"role": "user", "content": task.prompt}],
            temperature=temperature,
            seed=seed,
        )
        for task in tasks
    ]

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        progress_task = progress.add_task("Generating completions", total=len(requests))
        responses = client.generate_batch(
            requests, max_concurrent, on_complete=lambda: progress.update(progress_task, advance=1)
        )

    completions = []
    failures = []
    for task, response in zip(tasks, responses):
        if response.ok:
            completions.append(TaskCompletion(
                task=task,
                completion=response.unwrap(),
            ))
        else:
            failures.append((task.id, response.error_details() or "Unknown error"))

    if failures:
        print(f"\n{len(failures)} completion failures:")
        for task_id, error in failures[:5]:
            error_preview = error[:150] if len(error) > 150 else error
            print(f"  {task_id}: {error_preview}")
        if len(failures) > 5:
            print(f"  ... and {len(failures) - 5} more")

    return completions


def completions_exist(
    client: OpenAICompatibleClient,
    seed: int | None = None,
) -> bool:
    store_dir = _completions_dir(client, seed)
    return (store_dir / "completions.json").exists()


def load_completions(
    client: OpenAICompatibleClient,
    seed: int | None = None,
    task_lookup: dict[str, Task] | None = None,
) -> list[TaskCompletion]:
    """Convenience function to load completions without creating a store."""
    store = CompletionStore(client, seed)
    return store.load(task_lookup)
