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

from src.measurement_storage.completions import extract_completion_text
from src.models import TransformerLensModel
from src.models.registry import supports_system_role
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


def load_existing_data(
    output_dir: Path,
    selector_names: list[str],
) -> tuple[list[str], dict[str, dict[int, list[np.ndarray]]], list[dict]]:
    """Load existing activations and completions if resuming."""
    task_ids: list[str] = []
    activations_by_selector: dict[str, dict[int, list[np.ndarray]]] = {
        name: defaultdict(list) for name in selector_names
    }
    completions: list[dict] = []

    # Load from first selector file to get task_ids
    for name in selector_names:
        npz_path = output_dir / f"activations_{name}.npz"
        if npz_path.exists():
            data = np.load(npz_path, allow_pickle=True)
            if not task_ids:
                task_ids = list(data["task_ids"])
            for key in data.keys():
                if key.startswith("layer_"):
                    layer = int(key.split("_")[1])
                    activations_by_selector[name][layer] = list(data[key])

    completions_path = output_dir / "completions.json"
    if completions_path.exists():
        with open(completions_path) as f:
            completions = json.load(f)

    return task_ids, activations_by_selector, completions


def save_activations(
    output_dir: Path,
    task_ids: list[str],
    activations_by_selector: dict[str, dict[int, list[np.ndarray]]],
) -> None:
    """Save activations for each selector to separate files.

    Creates activations_{selector_name}.npz for each selector.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for selector_name, layer_activations in activations_by_selector.items():
        stacked_layers = {}
        for layer, acts in layer_activations.items():
            shapes = [a.shape for a in acts]
            unique_shapes = set(shapes)
            if len(unique_shapes) > 1:
                print(f"ERROR: {selector_name} layer {layer} has mismatched shapes: {unique_shapes}")
                for i, (tid, shape) in enumerate(zip(task_ids, shapes)):
                    if shape != shapes[0]:
                        print(f"  Task {tid} (idx {i}) has shape {shape} (expected {shapes[0]})")
                raise ValueError(f"Cannot stack activations for {selector_name} layer {layer}")
            stacked_layers[f"layer_{layer}"] = np.stack(acts)

        # Save to temp file first, then rename for atomic write
        tmp_path = output_dir / f"activations_{selector_name}.npz.tmp"
        final_path = output_dir / f"activations_{selector_name}.npz"
        np.savez(tmp_path, task_ids=np.array(task_ids), **stacked_layers)
        tmp_path.rename(final_path)
        print(f"Saved {selector_name}: {len(task_ids)} tasks, {len(layer_activations)} layers")


def save_completions(output_dir: Path, completions: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "completions.json", "w") as f:
        json.dump(completions, f, indent=2)


def save_extraction_metadata(output_dir: Path, metadata: ExtractionMetadata) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "extraction_metadata.json", "w") as f:
        json.dump(metadata.to_dict(), f, indent=2)


def save_failures(output_dir: Path, failures: list[tuple[str, str, str]]) -> None:
    """Save failure log with task_id, error, and task prompt."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "failures.json", "w") as f:
        json.dump(
            [{"task_id": tid, "error": err, "prompt": prompt} for tid, err, prompt in failures],
            f,
            indent=2,
        )


def extract_activations_with_system_prompt(
    model: TransformerLensModel,
    tasks: list[Task],
    layers: list[int],
    system_prompt: str | None,
    condition_name: str,
    output_dir: Path,
    selector_names: list[str],
    temperature: float = 1.0,
    max_new_tokens: int = 1024,
    resume: bool = False,
    save_every: int = 100,
    task_origins: list[str] | None = None,
    layers_config: list[float | int] | None = None,
    seed: int | None = None,
) -> None:
    """Extract activations from task completions under a specific system prompt condition."""
    task_ids: list[str] = []
    activations_by_selector: dict[str, dict[int, list[np.ndarray]]] = {
        name: defaultdict(list) for name in selector_names
    }
    completions: list[dict] = []
    n_existing = 0

    if resume:
        task_ids, activations_by_selector, completions = load_existing_data(output_dir, selector_names)
        existing_ids = set(task_ids)
        n_existing = len(existing_ids)
        tasks = [t for t in tasks if t.id not in existing_ids]
        print(f"Resume: found {n_existing} existing, {len(tasks)} remaining tasks")

    failures: list[tuple[str, str, str]] = []  # (task_id, error, prompt)
    n_new = 0
    n_truncated = 0
    n_ooms = 0

    alloc, res = gpu_mem_gb()
    print(f"Extracting activations for condition '{condition_name}' with {len(tasks)} tasks...")
    print(f"Selectors: {selector_names}")
    print(f"System prompt: {system_prompt[:100] if system_prompt else 'None'}...")
    print(f"GPU: {alloc:.1f}GB alloc, {res:.1f}GB reserved")

    for i, task in enumerate(tqdm(tasks, desc=f"Condition: {condition_name}")):
        for attempt in range(2):
            try:
                # Build messages with optional system prompt
                messages: list[dict[str, str]] = []
                if system_prompt:
                    if supports_system_role(model.canonical_model_name):
                        messages.append({"role": "system", "content": system_prompt})
                        messages.append({"role": "user", "content": task.prompt})
                    else:
                        # Prepend system prompt to user message for models without system role
                        messages.append({"role": "user", "content": f"{system_prompt}\n\n{task.prompt}"})
                else:
                    messages.append({"role": "user", "content": task.prompt})

                result = model.generate_with_activations(
                    messages, layers=layers, selector_names=selector_names,
                    temperature=temperature, max_new_tokens=max_new_tokens
                )
                truncated = result.completion_tokens >= max_new_tokens

                if truncated:
                    n_truncated += 1

                task_ids.append(task.id)
                # result.activations is now {selector_name: {layer: activation}}
                for selector_name, layer_acts in result.activations.items():
                    for layer, act in layer_acts.items():
                        activations_by_selector[selector_name][layer].append(act)

                completions.append({
                    "task_id": task.id,
                    "origin": task.origin.name,
                    "task_prompt": task.prompt,
                    "system_prompt": system_prompt,
                    "condition": condition_name,
                    "completion": extract_completion_text(result.completion),
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
                    failures.append((task.id, f"OOM after retry: {e}", task.prompt))
            except Exception as e:
                tqdm.write(f"Error on task {task.id}: {e}")
                failures.append((task.id, str(e), task.prompt))
                break

        gc.collect()
        torch.cuda.empty_cache()

        if (i + 1) % 100 == 0:
            alloc, res = gpu_mem_gb()
            tqdm.write(f"[{i+1}] GPU: {alloc:.1f}GB alloc, {res:.1f}GB res | OOMs: {n_ooms}")

        # Checkpoint periodically
        if n_new > 0 and n_new % save_every == 0:
            tqdm.write(f"Checkpoint: saving {len(task_ids)} total activations...")
            save_activations(output_dir, task_ids, activations_by_selector)
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
        save_activations(output_dir, task_ids, activations_by_selector)
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
