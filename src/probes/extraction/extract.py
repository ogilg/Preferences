"""Core extraction logic: batched and generation-based paths."""

from __future__ import annotations

import gc
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.measurement.storage.completions import extract_completion_text
from src.measurement.runners.utils.runner_utils import (
    load_activation_task_ids,
    get_activation_completions_path,
)
from src.models.huggingface_model import HuggingFaceModel
from src.models.transformer_lens import TransformerLensModel
from src.models.registry import has_transformer_lens_support, has_hf_support
from src.task_data import load_filtered_tasks, OriginDataset, Task
from src.types import Message

from .config import ExtractionConfig
from .metadata import ExtractionMetadata, ExtractionStats
from .persistence import (
    load_existing_data,
    save_activations,
    save_manifest,
    save_extraction_metadata,
)

ActivationModel = TransformerLensModel | HuggingFaceModel


def _gpu_mem_gb() -> tuple[float, float]:
    return (
        torch.cuda.memory_allocated() / 1e9,
        torch.cuda.memory_reserved() / 1e9,
    )


def _load_model(config: ExtractionConfig) -> ActivationModel:
    if config.backend == "transformer_lens":
        if not has_transformer_lens_support(config.model):
            raise ValueError(
                f"Model {config.model} does not have TransformerLens support. "
                f"Use backend: huggingface instead."
            )
        return TransformerLensModel(config.model, max_new_tokens=config.max_new_tokens)
    # backend == "huggingface" (validated by Pydantic Literal)
    if not has_hf_support(config.model):
        raise ValueError(f"Model {config.model} does not have HuggingFace support.")
    return HuggingFaceModel(config.model, max_new_tokens=config.max_new_tokens)


def _load_tasks(config: ExtractionConfig) -> list[Task]:
    if config.activations_model is not None:
        activation_task_ids = load_activation_task_ids(config.activations_model)
        with open(get_activation_completions_path(config.activations_model)) as f:
            completions_data = json.load(f)
        return [
            Task(
                id=c["task_id"],
                prompt=c["task_prompt"],
                origin=OriginDataset[c["origin"]],
                metadata={},
            )
            for c in completions_data
            if c["task_id"] in activation_task_ids
        ][:config.n_tasks]

    task_origins = [OriginDataset[o.upper()] for o in config.task_origins]
    return load_filtered_tasks(
        n=config.n_tasks,
        origins=task_origins,
        seed=config.seed,
        consistency_model=config.consistency_filter_model,
        consistency_keep_ratio=config.consistency_keep_ratio,
    )


def _build_metadata(
    config: ExtractionConfig,
    resolved_layers: list[int],
    n_model_layers: int,
    n_existing: int,
    stats: ExtractionStats,
    source_completions: str | None = None,
) -> ExtractionMetadata:
    task_origins = [OriginDataset[o.upper()] for o in config.task_origins]
    return ExtractionMetadata(
        model=config.model,
        n_tasks=config.n_tasks,
        task_origins=[o.name for o in task_origins],
        layers_config=config.layers_to_extract,
        layers_resolved=resolved_layers,
        n_model_layers=n_model_layers,
        selectors=config.selectors,
        batch_size=config.batch_size,
        temperature=config.temperature,
        max_new_tokens=config.max_new_tokens,
        seed=config.seed,
        n_existing=n_existing,
        n_new=stats.n_new,
        n_failures=stats.n_failures,
        n_truncated=stats.n_truncated,
        n_ooms=stats.n_ooms,
        source_completions=source_completions,
    )


def run_extraction(config: ExtractionConfig) -> None:
    output_dir = config.resolved_output_dir

    print(f"Loading model: {config.model} (backend: {config.backend})...")
    model = _load_model(config)

    resolved_layers = [model.resolve_layer(layer) for layer in config.layers_to_extract]
    print(f"Layers: {config.layers_to_extract} -> {resolved_layers} ({model.n_layers} total)")
    print(f"Selectors: {config.selectors}")

    # --- From-completions path (separate flow, no manifest) ---
    if config.from_completions is not None:
        _run_from_completions(config, model, resolved_layers, output_dir)
        return

    # --- Standard path ---
    tasks = _load_tasks(config)

    task_ids: list[str] = []
    activations: dict[str, dict[int, list[np.ndarray]]] = {s: defaultdict(list) for s in config.selectors}
    completions: list[dict] = []
    n_existing = 0

    if config.resume:
        task_ids, activations, completions = load_existing_data(output_dir, config.selectors)
        n_existing = len(task_ids)
        tasks = [t for t in tasks if t.id not in set(task_ids)]
        print(f"Resume: found {n_existing} existing, {len(tasks)} remaining")

    if not tasks:
        print("No tasks remaining.")
        return

    print(f"{len(tasks)} tasks to process")

    if config.needs_generation:
        stats = generation_extraction(
            model=model, tasks=tasks, layers=resolved_layers,
            selectors=config.selectors, temperature=config.temperature,
            max_new_tokens=config.max_new_tokens, task_ids=task_ids,
            activations=activations, completions=completions,
            output_dir=output_dir, save_every=config.save_every,
        )
    else:
        if not isinstance(model, HuggingFaceModel):
            raise ValueError("Batched extraction requires HuggingFace backend")
        task_lookup = {task.id: task for task in tasks}
        items: list[tuple[str, list[Message]]] = [
            (task.id, [{"role": "user", "content": task.prompt}])
            for task in tasks
        ]
        n_before = len(task_ids)
        stats = batched_extraction(
            model=model, items=items, layers=resolved_layers,
            selectors=config.selectors, batch_size=config.batch_size,
            task_ids=task_ids, activations=activations,
            output_dir=output_dir, save_every=config.save_every,
        )
        # Build manifest from actually-succeeded task_ids (not tasks[:n_new])
        for tid in task_ids[n_before:]:
            task = task_lookup[tid]
            completions.append({
                "task_id": task.id,
                "task_prompt": task.prompt,
                "origin": task.origin.name,
            })

    if stats.n_new > 0:
        print(f"Saving {len(task_ids)} total activations...")
        save_activations(output_dir, task_ids, activations)
        save_manifest(output_dir, completions)

    metadata = _build_metadata(config, resolved_layers, model.n_layers, n_existing, stats)
    save_extraction_metadata(output_dir, metadata)
    print(f"\nDone! {stats.n_new} new, {stats.n_failures} failures, {stats.n_ooms} OOMs")


def _run_from_completions(
    config: ExtractionConfig,
    model: ActivationModel,
    resolved_layers: list[int],
    output_dir: Path,
) -> None:
    if not isinstance(model, HuggingFaceModel):
        raise ValueError("Batched extraction requires HuggingFace backend")

    with open(config.from_completions) as f:
        completions_data: list[dict] = json.load(f)

    task_ids: list[str] = []
    activations: dict[str, dict[int, list[np.ndarray]]] = {s: defaultdict(list) for s in config.selectors}
    n_existing = 0

    if config.resume:
        task_ids, activations, _ = load_existing_data(output_dir, config.selectors)
        n_existing = len(task_ids)
        completions_data = [c for c in completions_data if c["task_id"] not in set(task_ids)]
        print(f"Resume: found {n_existing} existing, {len(completions_data)} remaining")

    if not completions_data:
        print("No completions remaining.")
        return

    items: list[tuple[str, list[Message]]] = [
        (c["task_id"], [
            {"role": "user", "content": c["task_prompt"]},
            {"role": "assistant", "content": c["completion"]},
        ])
        for c in completions_data
    ]

    stats = batched_extraction(
        model=model, items=items, layers=resolved_layers,
        selectors=config.selectors, batch_size=config.batch_size,
        task_ids=task_ids, activations=activations,
        output_dir=output_dir, save_every=config.save_every,
    )

    if stats.n_new > 0:
        save_activations(output_dir, task_ids, activations)

    metadata = _build_metadata(
        config, resolved_layers, model.n_layers, n_existing, stats,
        source_completions=str(config.from_completions),
    )
    save_extraction_metadata(output_dir, metadata)
    print(f"\nDone! Extracted {stats.n_new} new, {stats.n_failures} failures")


def batched_extraction(
    model: HuggingFaceModel,
    items: list[tuple[str, list[Message]]],
    layers: list[int],
    selectors: list[str],
    batch_size: int,
    task_ids: list[str],
    activations: dict[str, dict[int, list[np.ndarray]]],
    output_dir: Path,
    save_every: int,
) -> ExtractionStats:
    """Batched forward-pass extraction with OOM retry via recursive halving.

    Mutates task_ids and activations in place.
    """
    stats = ExtractionStats()

    alloc, res = _gpu_mem_gb()
    print(f"Batched extraction: {len(items)} items, batch_size={batch_size} (GPU: {alloc:.1f}GB alloc, {res:.1f}GB reserved)")

    for batch_start in tqdm(range(0, len(items), batch_size), desc="Batches"):
        batch = items[batch_start:batch_start + batch_size]
        batch_ids = [item[0] for item in batch]
        batch_messages = [item[1] for item in batch]

        succeeded_ids, per_sample_acts, batch_ooms, batch_fails = _extract_batch_with_oom_retry(
            model, batch_ids, batch_messages, layers, selectors,
        )

        stats.n_ooms += batch_ooms
        stats.n_failures += batch_fails

        for i, task_id in enumerate(succeeded_ids):
            task_ids.append(task_id)
            for selector in selectors:
                for layer, layer_acts in per_sample_acts[selector].items():
                    activations[selector][layer].append(layer_acts[i])
            stats.n_new += 1

        gc.collect()
        torch.cuda.empty_cache()

        batch_num = batch_start // batch_size + 1
        if batch_num % 10 == 0:
            alloc, res = _gpu_mem_gb()
            tqdm.write(f"[batch {batch_num}] GPU: {alloc:.1f}GB alloc, {res:.1f}GB res | OOMs: {stats.n_ooms}")

        if stats.n_new > 0 and stats.n_new % save_every == 0:
            tqdm.write(f"Checkpoint: saving {len(task_ids)} total activations...")
            save_activations(output_dir, task_ids, activations)

    return stats


def _extract_batch_with_oom_retry(
    model: HuggingFaceModel,
    ids: list[str],
    messages_batch: list[list[Message]],
    layers: list[int],
    selectors: list[str],
) -> tuple[list[str], dict[str, dict[int, np.ndarray]], int, int]:
    """Try full batch; on OOM, split in half and retry each half.

    Single-sample OOM is absorbed (empty result for that sample).
    Returns: (succeeded_ids, {selector: {layer: (n_succeeded, d_model)}}, n_ooms, n_failures)
    """
    try:
        result = model.get_activations_batch(messages_batch, layers=layers, selector_names=selectors)
        return ids, result, 0, 0
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        gc.collect()

        if len(ids) == 1:
            tqdm.write(f"OOM on single sample {ids[0]}, skipping")
            empty: dict[str, dict[int, np.ndarray]] = {s: {} for s in selectors}
            return [], empty, 1, 1

        mid = len(ids) // 2
        tqdm.write(f"OOM on batch of {len(ids)}, splitting into {mid} + {len(ids) - mid}")

        ids_a, acts_a, ooms_a, fails_a = _extract_batch_with_oom_retry(
            model, ids[:mid], messages_batch[:mid], layers, selectors,
        )
        ids_b, acts_b, ooms_b, fails_b = _extract_batch_with_oom_retry(
            model, ids[mid:], messages_batch[mid:], layers, selectors,
        )

        merged_ids = ids_a + ids_b
        merged_acts: dict[str, dict[int, np.ndarray]] = {s: {} for s in selectors}
        for selector in selectors:
            for layer in layers:
                parts = []
                if layer in acts_a[selector]:
                    parts.append(acts_a[selector][layer])
                if layer in acts_b[selector]:
                    parts.append(acts_b[selector][layer])
                if parts:
                    merged_acts[selector][layer] = np.concatenate(parts, axis=0)

        return merged_ids, merged_acts, ooms_a + ooms_b, fails_a + fails_b


def generation_extraction(
    model: ActivationModel,
    tasks: list[Task],
    layers: list[int],
    selectors: list[str],
    temperature: float,
    max_new_tokens: int,
    task_ids: list[str],
    activations: dict[str, dict[int, list[np.ndarray]]],
    completions: list[dict],
    output_dir: Path,
    save_every: int,
) -> ExtractionStats:
    """Sequential generate-then-extract. Mutates task_ids/activations/completions in place."""
    stats = ExtractionStats()
    failures: list[tuple[str, str]] = []

    alloc, res = _gpu_mem_gb()
    print(f"Generation extraction: {len(tasks)} tasks (GPU: {alloc:.1f}GB alloc, {res:.1f}GB reserved)")

    for i, task in enumerate(tqdm(tasks, desc="Tasks")):
        for attempt in range(2):
            try:
                messages: list[Message] = [{"role": "user", "content": task.prompt}]
                result = model.generate_with_activations(
                    messages, layers=layers, selector_names=selectors, temperature=temperature,
                )
                truncated = result.completion_tokens >= max_new_tokens

                if truncated:
                    stats.n_truncated += 1

                task_ids.append(task.id)
                for selector in selectors:
                    for layer, act in result.activations[selector].items():
                        activations[selector][layer].append(act)

                completions.append({
                    "task_id": task.id,
                    "task_prompt": task.prompt,
                    "origin": task.origin.name,
                    "completion": extract_completion_text(result.completion),
                    "truncated": truncated,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                })
                stats.n_new += 1
                break

            except torch.cuda.OutOfMemoryError as e:
                stats.n_ooms += 1
                tqdm.write(f"OOM on task {task.id} (attempt {attempt + 1}/2): {e}")
                torch.cuda.empty_cache()
                if attempt == 1:
                    failures.append((task.id, f"OOM after retry: {e}"))
                    stats.n_failures += 1
            except Exception as e:
                failures.append((task.id, str(e)))
                stats.n_failures += 1
                break

        gc.collect()
        torch.cuda.empty_cache()

        if (i + 1) % 100 == 0:
            alloc, res = _gpu_mem_gb()
            tqdm.write(f"[{i+1}] GPU: {alloc:.1f}GB alloc, {res:.1f}GB res | OOMs: {stats.n_ooms}")

        if stats.n_new > 0 and stats.n_new % save_every == 0:
            tqdm.write(f"Checkpoint: saving {len(task_ids)} total activations...")
            save_activations(output_dir, task_ids, activations)
            save_manifest(output_dir, completions)

    if failures:
        print(f"First few failures: {failures[:3]}")

    return stats
