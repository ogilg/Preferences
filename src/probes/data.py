from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass
class ProbeDataPoint:
    task_id: str
    activations: dict[int, np.ndarray]
    score: float
    completion: str
    raw_rating_response: str
    truncated: bool = False
    origin: str | None = None
    task_metadata: dict | None = None
    task_prompt: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


def save_probe_dataset(
    data_points: list[ProbeDataPoint],
    output_dir: Path,
    metadata: dict | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_ids = [dp.task_id for dp in data_points]
    scores = np.array([dp.score for dp in data_points])

    layers = list(data_points[0].activations.keys())
    activations_by_layer = {
        f"layer_{layer}": np.stack([dp.activations[layer] for dp in data_points])
        for layer in layers
    }

    np.savez(
        output_dir / "activations.npz",
        task_ids=np.array(task_ids),
        scores=scores,
        **activations_by_layer,
    )

    completions = [
        {
            "task_id": dp.task_id,
            "task_prompt": dp.task_prompt,
            "completion": dp.completion,
            "rating_response": dp.raw_rating_response,
            "score": dp.score,
            "truncated": dp.truncated,
            "origin": dp.origin,
            "task_metadata": dp.task_metadata,
            "prompt_tokens": dp.prompt_tokens,
            "completion_tokens": dp.completion_tokens,
        }
        for dp in data_points
    ]
    with open(output_dir / "completions.json", "w") as f:
        json.dump(completions, f, indent=2)

    if metadata is not None:
        with open(output_dir / "metadata.yaml", "w") as f:
            yaml.dump(metadata, f)


def load_probe_dataset(output_dir: Path) -> list[ProbeDataPoint]:
    output_dir = Path(output_dir)

    data = np.load(output_dir / "activations.npz", allow_pickle=True)
    task_ids = data["task_ids"].tolist()
    scores = data["scores"]

    layer_keys = [k for k in data.keys() if k.startswith("layer_")]
    layers = [int(k.split("_")[1]) for k in layer_keys]

    with open(output_dir / "completions.json") as f:
        completions_list = json.load(f)
    completions_data = {record["task_id"]: record for record in completions_list}

    data_points = []
    for i, task_id in enumerate(task_ids):
        activations = {
            layer: data[f"layer_{layer}"][i]
            for layer in layers
        }
        record = completions_data[task_id]
        data_points.append(ProbeDataPoint(
            task_id=task_id,
            activations=activations,
            score=float(scores[i]),
            completion=record["completion"],
            raw_rating_response=record["rating_response"],
            truncated=record.get("truncated", False),
            origin=record.get("origin"),
            task_metadata=record.get("task_metadata"),
            task_prompt=record.get("task_prompt"),
            prompt_tokens=record.get("prompt_tokens"),
            completion_tokens=record.get("completion_tokens"),
        ))

    return data_points
