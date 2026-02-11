"""Lightweight activation extraction — no manifest/store dependencies."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.models.huggingface_model import HuggingFaceModel
from src.task_data import Task
from src.types import Message


def extract_activations(
    model: HuggingFaceModel,
    tasks: list[Task],
    layers: list[int],
    selectors: list[str] = ("prompt_last",),
    batch_size: int = 32,
    save_path: Path | None = None,
    system_prompt: str | None = None,
) -> dict[str, dict[int, np.ndarray]]:
    """Extract activations for single tasks (not pairs).

    Returns {selector_name: {layer: (n_tasks, d_model) array}}.
    Optionally saves to npz files at save_path.
    """
    messages_per_task: list[list[Message]] = []
    for task in tasks:
        msgs: list[Message] = []
        if system_prompt is not None:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": task.prompt})
        messages_per_task.append(msgs)

    # Accumulate per-selector, per-layer lists of arrays
    collected: dict[str, dict[int, list[np.ndarray]]] = {
        s: {layer: [] for layer in layers} for s in selectors
    }

    for start in tqdm(range(0, len(tasks), batch_size), desc="Extracting"):
        batch_msgs = messages_per_task[start : start + batch_size]
        result = model.get_activations_batch(batch_msgs, layers, list(selectors))
        for selector in selectors:
            for layer in layers:
                collected[selector][layer].append(result[selector][layer])

    # Stack into single arrays
    stacked: dict[str, dict[int, np.ndarray]] = {}
    for selector in selectors:
        stacked[selector] = {}
        for layer in layers:
            stacked[selector][layer] = np.concatenate(collected[selector][layer], axis=0)

    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)
        task_ids = np.array([t.id for t in tasks])
        for selector in selectors:
            np.savez(
                save_path / f"activations_{selector}.npz",
                task_ids=task_ids,
                **{f"layer_{layer}": stacked[selector][layer] for layer in layers},
            )

    return stacked
