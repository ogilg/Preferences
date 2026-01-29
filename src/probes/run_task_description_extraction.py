"""Extract activations from task descriptions only (no generation).

This extracts the last-token activation from the task prompt before any completion,
to serve as a baseline for benchmarking completion-based probes.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.models import TransformerLensModel
from src.task_data import load_completions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract activations from task descriptions (no generation)"
    )
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument(
        "--layers",
        type=float,
        nargs="+",
        required=True,
        help="Layer indices or fractions",
    )
    parser.add_argument(
        "--completions-json",
        type=Path,
        required=True,
        help="Path to completions.json to match task_ids",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for activations",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading tasks from {args.completions_json}...")
    task_completion_pairs = load_completions(args.completions_json)
    print(f"Loaded {len(task_completion_pairs)} tasks")

    print(f"Loading model: {args.model}...")
    model = TransformerLensModel(args.model)

    resolved_layers = [model.resolve_layer(layer) for layer in args.layers]
    print(
        f"Extracting from layers: {args.layers} -> resolved to {resolved_layers} "
        f"(model has {model.n_layers} layers)"
    )

    task_ids: list[str] = []
    layer_activations: dict[int, list[np.ndarray]] = {l: [] for l in resolved_layers}
    n_failures = 0

    for task, _ in tqdm(task_completion_pairs, desc="Extracting activations"):
        try:
            messages = [{"role": "user", "content": task.prompt}]
            activations = model.get_activations(messages, layers=resolved_layers, selector_names=["last"])

            task_ids.append(task.id)
            for layer in resolved_layers:
                layer_activations[layer].append(activations["last"][layer])

        except torch.cuda.OutOfMemoryError as e:
            tqdm.write(f"OOM on task {task.id}: {e}")
            n_failures += 1
            torch.cuda.empty_cache()
        except Exception as e:
            tqdm.write(f"Error on task {task.id}: {e}")
            n_failures += 1

    print(f"\nExtracted {len(task_ids)} activations, {n_failures} failures")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save activations.npz in same format as completion activations
    save_dict = {"task_ids": np.array(task_ids)}
    for layer in resolved_layers:
        save_dict[f"layer_{layer}"] = np.stack(layer_activations[layer])

    npz_path = args.output_dir / "activations.npz"
    np.savez_compressed(npz_path, **save_dict)
    print(f"Saved activations to {npz_path}")

    # Save metadata
    metadata = {
        "model": args.model,
        "layers_config": args.layers,
        "layers_resolved": resolved_layers,
        "n_model_layers": model.n_layers,
        "n_tasks": len(task_ids),
        "n_failures": n_failures,
        "source_completions": str(args.completions_json),
        "created_at": datetime.now().isoformat(),
    }
    metadata_path = args.output_dir / "metadata.yaml"
    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f)
    print(f"Saved metadata to {metadata_path}")

    print("Done!")


if __name__ == "__main__":
    main()
