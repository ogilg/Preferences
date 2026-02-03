import csv
import json
from pathlib import Path
from typing import Callable

import numpy as np
from pydantic import BaseModel

from .task import OriginDataset, Task


DATA_DIR = Path(__file__).parent / "data"

FILE_MAPPING = {
    OriginDataset.WILDCHAT: [
        "wildchat_en_8k.jsonl",
    ],
    OriginDataset.ALPACA: ["alpaca_tasks_nemocurator.jsonl"],
    OriginDataset.MATH: ["math.jsonl"],
    OriginDataset.BAILBENCH: ["bailBench.csv"],
}


class ParserConfig(BaseModel):
    origin: OriginDataset
    prompt_key: str
    id_key: str | None
    metadata_keys: list[str]
    metadata_defaults: dict | None = None

    def parse(self, row: dict, index: int) -> Task:
        metadata = {}
        defaults = self.metadata_defaults or {}
        for key in self.metadata_keys:
            if key in defaults:
                metadata[key] = row.get(key, defaults[key])
            else:
                metadata[key] = row[key]
        task_id = row[self.id_key] if self.id_key else f"{self.origin.name.lower()}_{index}"
        return Task(
            prompt=row[self.prompt_key],
            origin=self.origin,
            id=task_id,
            metadata=metadata,
        )


PARSER_CONFIGS = {
    OriginDataset.WILDCHAT: ParserConfig(
        origin=OriginDataset.WILDCHAT,
        prompt_key="text",
        id_key="id",
        metadata_keys=["type", "topic"],
    ),
    OriginDataset.ALPACA: ParserConfig(
        origin=OriginDataset.ALPACA,
        prompt_key="task_text",
        id_key="task_id",
        metadata_keys=["nemo_analysis"],
        metadata_defaults={"nemo_analysis": {}},
    ),
    OriginDataset.MATH: ParserConfig(
        origin=OriginDataset.MATH,
        prompt_key="text",
        id_key="id",
        metadata_keys=["type", "topic", "q_metadata"],
        metadata_defaults={"q_metadata": {}},
    ),
    OriginDataset.BAILBENCH: ParserConfig(
        origin=OriginDataset.BAILBENCH,
        prompt_key="content",
        id_key=None,
        metadata_keys=["subcategory", "category"],
    ),
}


def _load_jsonl(filepath: Path) -> list[dict]:
    with open(filepath) as f:
        return [json.loads(line) for line in f]


def _load_csv(filepath: Path) -> list[dict]:
    with open(filepath, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_origin(origin: OriginDataset) -> list[Task]:
    tasks = []
    config = PARSER_CONFIGS[origin]
    for filename in FILE_MAPPING[origin]:
        filepath = DATA_DIR / filename
        if filepath.exists():
            if filepath.suffix == ".csv":
                rows = _load_csv(filepath)
            else:
                rows = _load_jsonl(filepath)
            tasks.extend(config.parse(row, i) for i, row in enumerate(rows))
    return tasks


def load_tasks(
    n: int,
    origins: list[OriginDataset],
    seed: int | None = None,
    filter_fn: Callable[[Task], bool] | None = None,
) -> list[Task]:
    tasks = []
    for origin in origins:
        tasks.extend(_load_origin(origin))

    if filter_fn is not None:
        tasks = [t for t in tasks if filter_fn(t)]

    if seed is not None:
        rng = np.random.default_rng(seed)
        rng.shuffle(tasks)

    return tasks[:n]


def load_filtered_tasks(
    n: int,
    origins: list[OriginDataset],
    seed: int | None = None,
    consistency_model: str | None = None,
    consistency_keep_ratio: float = 0.7,
    task_ids: set[str] | None = None,
    filter_fn: Callable[[Task], bool] | None = None,
) -> list[Task]:
    """Load tasks with optional consistency and task ID filtering.

    Args:
        n: Number of tasks to load
        origins: Dataset origins to load from
        seed: Random seed for shuffling
        consistency_model: Model key for consistency filter (e.g., "gemma2")
        consistency_keep_ratio: Keep top X% by consistency (default 0.7)
        task_ids: Only include tasks with these IDs (e.g., tasks with activations)
        filter_fn: Additional custom filter function
    """
    filters: list[Callable[[Task], bool]] = []

    if consistency_model is not None:
        from .consistency import make_consistency_filter
        filters.append(make_consistency_filter(consistency_model, keep_ratio=consistency_keep_ratio))

    if task_ids is not None:
        filters.append(lambda t, ids=task_ids: t.id in ids)

    if filter_fn is not None:
        filters.append(filter_fn)

    combined_filter = None
    if filters:
        combined_filter = lambda t: all(f(t) for f in filters)

    return load_tasks(n=n, origins=origins, seed=seed, filter_fn=combined_filter)


def load_completions(path: Path) -> list[tuple[Task, str]]:
    """Load task-completion pairs from JSON.

    Expected format: [{"task_id": str, "task_prompt": str, "completion": str, "origin": str}, ...]
    """
    with open(path) as f:
        data = json.load(f)
    return [
        (
            Task(
                prompt=item["task_prompt"],
                origin=OriginDataset[item.get("origin", "SYNTHETIC")],
                id=item["task_id"],
                metadata={},
            ),
            item["completion"],
        )
        for item in data
    ]
