import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .task import OriginDataset, Task


DATA_DIR = Path(__file__).parent / "data"

FILE_MAPPING = {
    OriginDataset.WILDCHAT: [
        "wildchat_en_8k.jsonl",
        "wildchat_unclassified_en_35k.jsonl",
    ],
    OriginDataset.ALPACA: ["alpaca_tasks_nemocurator.jsonl"],
    OriginDataset.MATH: ["math.jsonl"],
}


@dataclass
class ParserConfig:
    origin: OriginDataset
    prompt_key: str
    id_key: str
    metadata_keys: list[str]
    metadata_defaults: dict | None = None

    def parse(self, row: dict) -> Task:
        metadata = {}
        for key in self.metadata_keys:
            default = (self.metadata_defaults or {}).get(key)
            metadata[key] = row.get(key, default) if default is not None else row.get(key)
        return Task(
            prompt=row[self.prompt_key],
            origin=self.origin,
            id=row[self.id_key],
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
}


def _load_jsonl(filepath: Path) -> list[dict]:
    with open(filepath) as f:
        return [json.loads(line) for line in f]


def _load_origin(origin: OriginDataset) -> list[Task]:
    tasks = []
    config = PARSER_CONFIGS[origin]
    for filename in FILE_MAPPING[origin]:
        filepath = DATA_DIR / filename
        if filepath.exists():
            rows = _load_jsonl(filepath)
            tasks.extend(config.parse(row) for row in rows)
    return tasks


def load_tasks(
    n: int,
    origin: OriginDataset | None = None,
    filter_fn: Callable[[Task], bool] | None = None,
) -> list[Task]:
    if origin is not None:
        tasks = _load_origin(origin)
    else:
        tasks = []
        for orig in OriginDataset:
            tasks.extend(_load_origin(orig))

    if filter_fn is not None:
        tasks = [t for t in tasks if filter_fn(t)]

    return tasks[:n]
