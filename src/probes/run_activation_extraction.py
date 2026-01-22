"""Extract activations from model completions on tasks."""

from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.models import NnsightModel
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
                layer_activations[layer] = [act for act in data[key]]

    completions_path = output_dir / "completions_with_activations.json"
    if completions_path.exists():
        with open(completions_path) as f:
            completions = json.load(f)

    return task_ids, layer_activations, completions


def save_activations(
    output_dir: Path,
    task_ids: list[str],
    layer_activations: dict[int, list[np.ndarray]],
) -> None:
    """Save activations to .npz format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "activations.npz",
        task_ids=np.array(task_ids),
        **{f"layer_{layer}": np.stack(acts) for layer, acts in layer_activations.items()},
    )


def save_completions_metadata(output_dir: Path, completions: list[dict]) -> None:
    """Save completions metadata to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "completions_with_activations.json", "w") as f:
        json.dump(completions, f, indent=2)


def save_extraction_metadata(output_dir: Path, metadata: dict) -> None:
    """Save extraction metadata to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "extraction_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


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

    # Resume: load existing data if available
    task_ids: list[str] = []
    layer_activations: dict[int, list[np.ndarray]] = defaultdict(list)
    completions: list[dict] = []
    n_existing = 0

    if args.resume:
        task_ids, layer_activations, completions = load_existing_data(output_dir)
        existing_ids = set(task_ids)
        n_existing = len(existing_ids)
        tasks = [t for t in tasks if t.id not in existing_ids]
        print(f"Resume: found {n_existing} existing, {len(tasks)} remaining tasks")

    save_every = args.save_every

    print(f"Loading model: {model_name}...")
    model = NnsightModel(model_name, max_new_tokens=max_new_tokens)

    resolved_layers = [model.resolve_layer(layer) for layer in layers]
    print(f"Extracting from layers: {layers} -> resolved to {resolved_layers} (model has {model.n_layers} layers)")

    failures: list[tuple[str, str]] = []
    n_new = 0
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

                # Store activation for each layer
                task_ids.append(task.id)
                for layer, act in result.activations.items():
                    layer_activations[layer].append(act)

                # Store completion metadata
                completions.append({
                    "task_id": task.id,
                    "origin": task.origin.name,
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
            save_completions_metadata(output_dir, completions)
            save_extraction_metadata(output_dir, build_metadata(
                model_name, n_tasks, task_origins, layers, resolved_layers,
                model.n_layers, temperature, max_new_tokens, seed,
                n_existing, n_new, len(failures), n_truncated, n_ooms,
            ))

    # Final save
    if n_new > 0:
        print(f"Saving {len(task_ids)} total activations...")
        save_activations(output_dir, task_ids, layer_activations)
        save_completions_metadata(output_dir, completions)

    print(f"\nSaved {n_new} new data points, {len(failures)} failures, {n_truncated} truncated, {n_ooms} OOMs")
    if failures:
        print(f"First few failures: {failures[:3]}")

    save_extraction_metadata(output_dir, build_metadata(
        model_name, n_tasks, task_origins, layers, resolved_layers,
        model.n_layers, temperature, max_new_tokens, seed,
        n_existing, n_new, len(failures), n_truncated, n_ooms,
    ))
    print("Done!")


if __name__ == "__main__":
    main()
