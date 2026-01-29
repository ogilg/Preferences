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

from src.measurement_storage.completions import extract_completion_text
from src.models.transformer_lens import TransformerLensModel
from src.running_measurements.utils.runner_utils import load_activation_task_ids
from src.task_data import load_tasks, OriginDataset, Task


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
    parser.add_argument(
        "--from-completions",
        type=Path,
        help="Extract from existing completions JSON instead of generating new ones",
    )
    parser.add_argument(
        "--selectors",
        nargs="+",
        default=["last"],
        choices=["first", "mean", "last"],
        help="Token selectors to extract (default: last)",
    )
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


def load_existing_data(
    output_dir: Path, selectors: list[str]
) -> tuple[list[str], dict[str, dict[int, list[np.ndarray]]], list[dict]]:
    """Load existing activations and completions if resuming."""
    task_ids: list[str] = []
    activations: dict[str, dict[int, list[np.ndarray]]] = {s: defaultdict(list) for s in selectors}
    completions: list[dict] = []

    # Try loading from selector-specific files first, fall back to old format
    first_selector = selectors[0]
    npz_path = output_dir / f"activations_{first_selector}.npz"
    if not npz_path.exists():
        npz_path = output_dir / "activations.npz"

    if npz_path.exists():
        data = np.load(npz_path, allow_pickle=True)
        task_ids = list(data["task_ids"])
        for key in data.keys():
            if key.startswith("layer_"):
                layer = int(key.split("_")[1])
                # Old format: single selector (last)
                activations["last"][layer] = [act for act in data[key]]

    # Load selector-specific files if they exist
    for selector in selectors:
        selector_path = output_dir / f"activations_{selector}.npz"
        if selector_path.exists():
            data = np.load(selector_path, allow_pickle=True)
            if not task_ids:
                task_ids = list(data["task_ids"])
            for key in data.keys():
                if key.startswith("layer_"):
                    layer = int(key.split("_")[1])
                    activations[selector][layer] = [act for act in data[key]]

    completions_path = output_dir / "completions_with_activations.json"
    if completions_path.exists():
        with open(completions_path) as f:
            completions = json.load(f)

    return task_ids, activations, completions


def save_activations(
    output_dir: Path,
    task_ids: list[str],
    activations: dict[str, dict[int, list[np.ndarray]]],
) -> None:
    """Save activations to .npz format, one file per selector."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for selector_name, layer_acts in activations.items():
        if not layer_acts:
            continue
        np.savez(
            output_dir / f"activations_{selector_name}.npz",
            task_ids=np.array(task_ids),
            **{f"layer_{layer}": np.stack(acts) for layer, acts in layer_acts.items()},
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
    selectors = config.get("selectors", args.selectors)  # config overrides CLI default

    print(f"Loading model: {model_name}...")
    model = TransformerLensModel(model_name, max_new_tokens=max_new_tokens)

    resolved_layers = [model.resolve_layer(layer) for layer in layers]
    print(f"Extracting from layers: {layers} -> resolved to {resolved_layers} (model has {model.n_layers} layers)")
    print(f"Selectors: {selectors}")

    # Mode: extract from existing completions
    if args.from_completions:
        _extract_from_completions(
            args, model, resolved_layers, selectors, output_dir, model_name, layers
        )
        return

    # Mode: generate and extract
    use_tasks_with_activations = config.get("use_tasks_with_activations", False)

    if use_tasks_with_activations:
        activation_task_ids = load_activation_task_ids()
        from src.running_measurements.utils.runner_utils import get_activation_completions_path
        with open(get_activation_completions_path()) as f:
            completions_data = json.load(f)
        tasks = [
            Task(
                id=c["task_id"],
                prompt=c["task_prompt"],
                origin=OriginDataset[c.get("origin", "SYNTHETIC")],
                metadata={},
            )
            for c in completions_data
            if c["task_id"] in activation_task_ids
        ][:n_tasks]
        print(f"Using {len(tasks)} tasks from existing activation extraction")
    else:
        print(f"Loading {n_tasks} tasks from {[o.value for o in task_origins]}...")
        tasks = load_tasks(n=n_tasks, origins=task_origins, seed=seed)

    task_ids: list[str] = []
    activations: dict[str, dict[int, list[np.ndarray]]] = {s: defaultdict(list) for s in selectors}
    completions: list[dict] = []
    n_existing = 0

    if args.resume:
        task_ids, activations, completions = load_existing_data(output_dir, selectors)
        existing_ids = set(task_ids)
        n_existing = len(existing_ids)
        tasks = [t for t in tasks if t.id not in existing_ids]
        print(f"Resume: found {n_existing} existing, {len(tasks)} remaining tasks")

    save_every = args.save_every
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
                    messages, layers=resolved_layers, selector_names=selectors, temperature=temperature
                )
                truncated = result.completion_tokens >= max_new_tokens

                if truncated:
                    n_truncated += 1

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

        if n_new > 0 and n_new % save_every == 0:
            tqdm.write(f"Checkpoint: saving {len(task_ids)} total activations...")
            save_activations(output_dir, task_ids, activations)
            save_completions_metadata(output_dir, completions)
            save_extraction_metadata(output_dir, build_metadata(
                model_name, n_tasks, task_origins, layers, resolved_layers,
                model.n_layers, temperature, max_new_tokens, seed,
                n_existing, n_new, len(failures), n_truncated, n_ooms,
            ))

    if n_new > 0:
        print(f"Saving {len(task_ids)} total activations...")
        save_activations(output_dir, task_ids, activations)
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


def _extract_from_completions(
    args: argparse.Namespace,
    model: TransformerLensModel,
    resolved_layers: list[int],
    selectors: list[str],
    output_dir: Path,
    model_name: str,
    layers_config: list[float | int],
) -> None:
    """Extract activations from existing completions without generating."""
    with open(args.from_completions) as f:
        completions_data = json.load(f)

    task_ids: list[str] = []
    activations: dict[str, dict[int, list[np.ndarray]]] = {s: defaultdict(list) for s in selectors}
    n_existing = 0

    if args.resume:
        task_ids, activations, _ = load_existing_data(output_dir, selectors)
        existing_ids = set(task_ids)
        n_existing = len(existing_ids)
        completions_data = [c for c in completions_data if c["task_id"] not in existing_ids]
        print(f"Resume: found {n_existing} existing, {len(completions_data)} remaining")

    failures: list[tuple[str, str]] = []
    n_new = 0

    alloc, res = gpu_mem_gb()
    print(f"Extracting from {len(completions_data)} completions... (GPU: {alloc:.1f}GB alloc, {res:.1f}GB reserved)")

    for i, comp in enumerate(tqdm(completions_data, desc="Extracting")):
        task_id = comp["task_id"]
        task_prompt = comp.get("task_prompt", comp.get("prompt", ""))
        completion_text = comp["completion"]

        messages = [
            {"role": "user", "content": task_prompt},
            {"role": "assistant", "content": completion_text},
        ]

        try:
            result = model.get_activations(messages, layers=resolved_layers, selector_names=selectors)

            task_ids.append(task_id)
            for selector in selectors:
                for layer, act in result[selector].items():
                    activations[selector][layer].append(act)
            n_new += 1

        except Exception as e:
            failures.append((task_id, str(e)))
            tqdm.write(f"Failed on {task_id}: {e}")

        gc.collect()
        torch.cuda.empty_cache()

        if (i + 1) % 100 == 0:
            alloc, res = gpu_mem_gb()
            tqdm.write(f"[{i+1}] GPU: {alloc:.1f}GB alloc, {res:.1f}GB res")

        if n_new > 0 and n_new % args.save_every == 0:
            tqdm.write(f"Checkpoint: saving {len(task_ids)} activations...")
            save_activations(output_dir, task_ids, activations)
            save_extraction_metadata(output_dir, {
                "model": model_name,
                "selectors": selectors,
                "layers_config": layers_config,
                "layers_resolved": resolved_layers,
                "n_existing": n_existing,
                "n_new": n_new,
                "n_failures": len(failures),
                "source_completions": str(args.from_completions),
                "last_updated": datetime.now().isoformat(),
            })

    if n_new > 0:
        print(f"\nSaving {len(task_ids)} total activations...")
        save_activations(output_dir, task_ids, activations)

    save_extraction_metadata(output_dir, {
        "model": model_name,
        "selectors": selectors,
        "layers_config": layers_config,
        "layers_resolved": resolved_layers,
        "n_existing": n_existing,
        "n_new": n_new,
        "n_failures": len(failures),
        "source_completions": str(args.from_completions),
        "last_updated": datetime.now().isoformat(),
    })

    print(f"\nDone! Extracted {n_new} new, {len(failures)} failures")
    if failures:
        print(f"First few failures: {failures[:3]}")


if __name__ == "__main__":
    main()
