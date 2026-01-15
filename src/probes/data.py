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
    completion: str
    truncated: bool = False
    origin: str | None = None
    task_metadata: dict | None = None
    task_prompt: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


def save_probe_batch(
    data_points: list[ProbeDataPoint],
    output_dir: Path,
    batch_index: int,
) -> None:
    """Save a batch of probe data points to numbered files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_ids = [dp.task_id for dp in data_points]
    layers = list(data_points[0].activations.keys())
    activations_by_layer = {
        f"layer_{layer}": np.stack([dp.activations[layer] for dp in data_points])
        for layer in layers
    }

    np.savez(
        output_dir / f"activations_{batch_index}.npz",
        task_ids=np.array(task_ids),
        **activations_by_layer,
    )

    completions = [
        {
            "task_id": dp.task_id,
            "task_prompt": dp.task_prompt,
            "completion": dp.completion,
            "truncated": dp.truncated,
            "origin": dp.origin,
            "task_metadata": dp.task_metadata,
            "prompt_tokens": dp.prompt_tokens,
            "completion_tokens": dp.completion_tokens,
        }
        for dp in data_points
    ]
    with open(output_dir / f"completions_{batch_index}.json", "w") as f:
        json.dump(completions, f, indent=2)


def save_probe_metadata(output_dir: Path, metadata: dict) -> None:
    """Save experiment metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metadata.yaml", "w") as f:
        yaml.dump(metadata, f)


def get_next_batch_index(output_dir: Path) -> int:
    """Get the next available batch index."""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return 0
    existing = list(output_dir.glob("activations_*.npz"))
    if not existing:
        return 0
    indices = [int(p.stem.split("_")[1]) for p in existing]
    return max(indices) + 1


def get_existing_task_ids(output_dir: Path) -> set[str]:
    """Get all task IDs from existing batch files."""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return set()
    task_ids = set()
    for completions_file in output_dir.glob("completions_*.json"):
        with open(completions_file) as f:
            for record in json.load(f):
                task_ids.add(record["task_id"])
    return task_ids


def load_probe_dataset(output_dir: Path) -> list[ProbeDataPoint]:
    """Load all probe data from batch files in output_dir."""
    output_dir = Path(output_dir)

    activation_files = sorted(output_dir.glob("activations_*.npz"))
    if not activation_files:
        return []

    data_points = []
    for act_file in activation_files:
        batch_idx = act_file.stem.split("_")[1]
        completions_file = output_dir / f"completions_{batch_idx}.json"

        data = np.load(act_file, allow_pickle=True)
        task_ids = data["task_ids"].tolist()

        layer_keys = [k for k in data.keys() if k.startswith("layer_")]
        layers = [int(k.split("_")[1]) for k in layer_keys]

        with open(completions_file) as f:
            completions_list = json.load(f)
        completions_data = {record["task_id"]: record for record in completions_list}

        for i, task_id in enumerate(task_ids):
            activations = {layer: data[f"layer_{layer}"][i] for layer in layers}
            record = completions_data[task_id]
            data_points.append(ProbeDataPoint(
                task_id=task_id,
                activations=activations,
                completion=record["completion"],
                truncated=record.get("truncated", False),
                origin=record.get("origin"),
                task_metadata=record.get("task_metadata"),
                task_prompt=record.get("task_prompt"),
                prompt_tokens=record.get("prompt_tokens"),
                completion_tokens=record.get("completion_tokens"),
            ))

    return data_points
