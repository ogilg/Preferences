"""Extract activations from model completions on tasks."""

from __future__ import annotations

import argparse
import gc
from datetime import datetime
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from src.models import NnsightModel
from src.probes.data import (
    ProbeDataPoint,
    save_probe_batch,
    save_probe_metadata,
    get_next_batch_index,
    get_existing_task_ids,
)
from src.task_data import load_tasks, OriginDataset


def gpu_mem_gb() -> tuple[float, float]:
    return (
        torch.cuda.memory_allocated() / 1e9,
        torch.cuda.memory_reserved() / 1e9,
    )

DEFAULT_SAVE_EVERY = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract activations from task completions")
    parser.add_argument("config", type=Path, help="Path to probe experiment config YAML")
    parser.add_argument("--resume", action="store_true", help="Skip tasks already in output_dir")
    parser.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY, help="Save batch every N tasks")
    return parser.parse_args()


def build_metadata(
    model_name: str,
    n_tasks: int,
    task_origins: list[OriginDataset],
    layers: list[float | int],
    resolved_layers: list[int],
    n_model_layers: int,
    temperature: float,
    max_new_tokens: int,
    seed: int | None,
    n_existing: int,
    n_saved: int,
    n_failures: int,
    n_truncated: int,
    n_ooms: int,
) -> dict:
    return {
        "model": model_name,
        "n_tasks": n_tasks,
        "task_origins": [o.value for o in task_origins],
        "layers_config": layers,
        "layers_resolved": resolved_layers,
        "n_model_layers": n_model_layers,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "seed": seed,
        "last_updated": datetime.now().isoformat(),
        "n_existing": n_existing,
        "n_new": n_saved,
        "n_failures": n_failures,
        "n_truncated": n_truncated,
        "n_ooms": n_ooms,
    }


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_name = config["model"]
    n_tasks = config["n_tasks"]
    task_origins = [OriginDataset[o.upper()] for o in config["task_origins"]]
    layers = config["layers_to_extract"]
    temperature = config.get("temperature", 1.0)
    max_new_tokens = config.get("max_new_tokens", 2048)
    seed = config.get("seed")
    output_dir = Path(config["output_dir"])

    print(f"Loading {n_tasks} tasks from {[o.value for o in task_origins]}...")
    tasks = load_tasks(n=n_tasks, origins=task_origins, seed=seed)

    existing_ids: set[str] = set()
    if args.resume:
        existing_ids = get_existing_task_ids(output_dir)
        original_count = len(tasks)
        tasks = [t for t in tasks if t.id not in existing_ids]
        print(f"Resume: found {len(existing_ids)} existing, {len(tasks)} remaining tasks")

    batch_index = get_next_batch_index(output_dir)
    save_every = args.save_every

    print(f"Loading model: {model_name}...")
    model = NnsightModel(model_name, max_new_tokens=max_new_tokens)

    resolved_layers = [model.resolve_layer(layer) for layer in layers]
    print(f"Extracting from layers: {layers} -> resolved to {resolved_layers} (model has {model.n_layers} layers)")

    data_points: list[ProbeDataPoint] = []
    failures: list[tuple[str, str]] = []

    n_saved = 0
    n_truncated = 0
    n_ooms = 0
    alloc, res = gpu_mem_gb()
    print(f"Collecting data for {len(tasks)} tasks... (GPU: {alloc:.1f}GB alloc, {res:.1f}GB reserved)")
    for i, task in enumerate(tqdm(tasks, desc="Tasks")):
        for attempt in range(2):
            try:
                messages = [{"role": "user", "content": task.prompt}]
                result = model.generate_with_activations(
                    messages, layers=resolved_layers, temperature=temperature
                )
                truncated = result.completion_tokens >= max_new_tokens

                if truncated:
                    n_truncated += 1

                data_points.append(ProbeDataPoint(
                    task_id=task.id,
                    activations=result.activations,
                    completion=result.completion,
                    truncated=truncated,
                    origin=task.origin.name,
                    task_metadata=task.metadata,
                    task_prompt=task.prompt,
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                ))
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

        if len(data_points) >= save_every:
            tqdm.write(f"Saving batch {batch_index} ({len(data_points)} points)...")
            save_probe_batch(data_points, output_dir, batch_index)
            n_saved += len(data_points)
            batch_index += 1
            data_points = []
            save_probe_metadata(output_dir, build_metadata(
                model_name, n_tasks, task_origins, layers, resolved_layers,
                model.n_layers, temperature, max_new_tokens, seed,
                len(existing_ids), n_saved, len(failures), n_truncated, n_ooms,
            ))

    if data_points:
        print(f"Saving final batch {batch_index} ({len(data_points)} points)...")
        save_probe_batch(data_points, output_dir, batch_index)
        n_saved += len(data_points)

    print(f"\nSaved {n_saved} new data points, {len(failures)} failures, {n_truncated} truncated, {n_ooms} OOMs")
    if failures:
        print(f"First few failures: {failures[:3]}")

    save_probe_metadata(output_dir, build_metadata(
        model_name, n_tasks, task_origins, layers, resolved_layers,
        model.n_layers, temperature, max_new_tokens, seed,
        len(existing_ids), n_saved, len(failures), n_truncated, n_ooms,
    ))
    print("Done!")


if __name__ == "__main__":
    main()
