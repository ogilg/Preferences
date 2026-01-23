"""Extract activations from model completions under different system prompt conditions."""

from __future__ import annotations

import gc
import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.models import NnsightModel
from src.task_data import Task


def gpu_mem_gb() -> tuple[float, float]:
    return (
        torch.cuda.memory_allocated() / 1e9,
        torch.cuda.memory_reserved() / 1e9,
    )


@dataclass
class ExtractionMetadata:
    model: str
    n_tasks: int
    task_origins: list[str]
    layers_config: list[float | int]
    layers_resolved: list[int]
    n_model_layers: int
    temperature: float
    max_new_tokens: int
    seed: int | None
    system_prompt: str | None
    condition_name: str
    n_existing: int
    n_new: int
    n_failures: int
    n_truncated: int
    n_ooms: int
    last_updated: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["last_updated"] = datetime.now().isoformat()
        return d


def load_existing_data(output_dir: Path) -> tuple[list[str], dict[int, list[np.ndarray]], list[dict]]:
    """Load existing activations.npz and completions if resuming."""
    task_ids: list[str] = []
    layer_activations: dict[int, list[np.ndarray]] = defaultdict(list)
    completions: list[dict] = []

    npz_path = output_dir / "activations.npz"
    if npz_path.exists():
        data = np.load(npz_path, allow_pickle=True)
        task_ids = list(data["task_ids"])
        for key in data.keys():
            if key.startswith("layer_"):
                layer = int(key.split("_")[1])
                layer_activations[layer] = list(data[key])

    completions_path = output_dir / "completions.json"
    if completions_path.exists():
        with open(completions_path) as f:
            completions = json.load(f)

    return task_ids, layer_activations, completions


def save_activations(
    output_dir: Path,
    task_ids: list[str],
    layer_activations: dict[int, list[np.ndarray]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "activations.npz",
        task_ids=np.array(task_ids),
        **{f"layer_{layer}": np.stack(acts) for layer, acts in layer_activations.items()},
    )


def save_completions(output_dir: Path, completions: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "completions.json", "w") as f:
        json.dump(completions, f, indent=2)


def save_extraction_metadata(output_dir: Path, metadata: ExtractionMetadata) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "extraction_metadata.json", "w") as f:
        json.dump(metadata.to_dict(), f, indent=2)


def extract_activations_with_system_prompt(
    model: NnsightModel,
    tasks: list[Task],
    layers: list[int],
    system_prompt: str | None,
    condition_name: str,
    output_dir: Path,
    temperature: float = 1.0,
    max_new_tokens: int = 1024,
    resume: bool = False,
    save_every: int = 100,
    task_origins: list[str] | None = None,
    layers_config: list[float | int] | None = None,
    seed: int | None = None,
) -> None:
    """Extract activations from task completions under a specific system prompt condition.

    Args:
        model: NnsightModel instance
        tasks: List of tasks to complete
        layers: Resolved layer indices to extract from
        system_prompt: System prompt to prepend (None for no system prompt)
        condition_name: Name identifier for this condition
        output_dir: Directory to save outputs (activations.npz, completions.json)
        temperature: Generation temperature
        max_new_tokens: Max tokens to generate
        resume: If True, skip tasks already in output_dir
        save_every: Checkpoint frequency
        task_origins: List of origin names for metadata
        layers_config: Original layer config (floats/ints) for metadata
        seed: Random seed used for task sampling
    """
    task_ids: list[str] = []
    layer_activations: dict[int, list[np.ndarray]] = defaultdict(list)
    completions: list[dict] = []
    n_existing = 0

    if resume:
        task_ids, layer_activations, completions = load_existing_data(output_dir)
        existing_ids = set(task_ids)
        n_existing = len(existing_ids)
        tasks = [t for t in tasks if t.id not in existing_ids]
        print(f"Resume: found {n_existing} existing, {len(tasks)} remaining tasks")

    failures: list[tuple[str, str]] = []
    n_new = 0
    n_truncated = 0
    n_ooms = 0

    alloc, res = gpu_mem_gb()
    print(f"Extracting activations for condition '{condition_name}' with {len(tasks)} tasks...")
    print(f"System prompt: {system_prompt[:100] if system_prompt else 'None'}...")
    print(f"GPU: {alloc:.1f}GB alloc, {res:.1f}GB reserved")

    for i, task in enumerate(tqdm(tasks, desc=f"Condition: {condition_name}")):
        for attempt in range(2):
            try:
                # Build messages with optional system prompt
                messages: list[dict[str, str]] = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": task.prompt})

                result = model.generate_with_activations(
                    messages, layers=layers, temperature=temperature, max_new_tokens=max_new_tokens
                )
                truncated = result.completion_tokens >= max_new_tokens

                if truncated:
                    n_truncated += 1

                task_ids.append(task.id)
                for layer, act in result.activations.items():
                    layer_activations[layer].append(act)

                completions.append({
                    "task_id": task.id,
                    "origin": task.origin.name,
                    "task_prompt": task.prompt,
                    "system_prompt": system_prompt,
                    "condition": condition_name,
                    "completion": result.completion,
                    "truncated": truncated,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                })
                n_new += 1
                break

            except torch.cuda.OutOfMemoryError as e:
                n_ooms += 1
                tqdm.write(f"OOM on task {task.id} (attempt {attempt + 1}/2): {e}")
                torch.cuda.empty_cache()
                if attempt == 1:
                    failures.append((task.id, f"OOM after retry: {e}"))
            except Exception as e:
                failures.append((task.id, str(e)))
                break

        gc.collect()
        torch.cuda.empty_cache()

        if (i + 1) % 100 == 0:
            alloc, res = gpu_mem_gb()
            tqdm.write(f"[{i+1}] GPU: {alloc:.1f}GB alloc, {res:.1f}GB res | OOMs: {n_ooms}")

        # Checkpoint periodically
        if n_new > 0 and n_new % save_every == 0:
            tqdm.write(f"Checkpoint: saving {len(task_ids)} total activations...")
            save_activations(output_dir, task_ids, layer_activations)
            save_completions(output_dir, completions)
            metadata = ExtractionMetadata(
                model=model.model_name,
                n_tasks=len(tasks) + n_existing,
                task_origins=task_origins or [],
                layers_config=layers_config or layers,
                layers_resolved=layers,
                n_model_layers=model.n_layers,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                seed=seed,
                system_prompt=system_prompt,
                condition_name=condition_name,
                n_existing=n_existing,
                n_new=n_new,
                n_failures=len(failures),
                n_truncated=n_truncated,
                n_ooms=n_ooms,
            )
            save_extraction_metadata(output_dir, metadata)

    # Final save
    if n_new > 0:
        print(f"Saving {len(task_ids)} total activations...")
        save_activations(output_dir, task_ids, layer_activations)
        save_completions(output_dir, completions)

    print(f"\nCondition '{condition_name}': {n_new} new, {len(failures)} failures, {n_truncated} truncated, {n_ooms} OOMs")
    if failures:
        print(f"First few failures: {failures[:3]}")

    final_metadata = ExtractionMetadata(
        model=model.model_name,
        n_tasks=len(task_ids),
        task_origins=task_origins or [],
        layers_config=layers_config or layers,
        layers_resolved=layers,
        n_model_layers=model.n_layers,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        seed=seed,
        system_prompt=system_prompt,
        condition_name=condition_name,
        n_existing=n_existing,
        n_new=n_new,
        n_failures=len(failures),
        n_truncated=n_truncated,
        n_ooms=n_ooms,
    )
    save_extraction_metadata(output_dir, final_metadata)
