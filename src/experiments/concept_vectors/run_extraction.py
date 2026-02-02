"""CLI entry point for concept vector extraction via system prompt conditioning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from src.experiments.concept_vectors.config import load_config
from src.experiments.concept_vectors.difference import compute_all_concept_vectors, save_concept_vectors
from src.experiments.concept_vectors.extraction import extract_activations_with_system_prompt
from src.models import TransformerLensModel
from src.task_data import OriginDataset, load_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract concept vectors via system prompt conditioning"
    )
    parser.add_argument("config", type=Path, help="Path to config YAML")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--save-every", type=int, default=100, help="Checkpoint frequency")
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip extraction, only compute difference-in-means from existing data",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tasks once (shared across conditions)
    task_origins = [OriginDataset[o.upper()] for o in config.task_origins]
    print(f"Loading {config.n_tasks} tasks from {[o.value for o in task_origins]}...")
    tasks = load_tasks(n=config.n_tasks, origins=task_origins, seed=config.task_sampling_seed)
    print(f"Loaded {len(tasks)} tasks")

    if not args.skip_extraction:
        print(f"Loading model: {config.model}...")
        model = TransformerLensModel(config.model, max_new_tokens=config.max_new_tokens)

        resolved_layers = [model.resolve_layer(layer) for layer in config.layers_to_extract]
        print(
            f"Extracting from layers: {config.layers_to_extract} -> "
            f"resolved to {resolved_layers} (model has {model.n_layers} layers)"
        )

        # Extract activations for each condition
        for condition_key, condition_dict in config.conditions.items():
            condition_dir = output_dir / condition_key
            print(f"\n{'='*60}")
            print(f"Processing condition: {condition_key}")
            print(f"System prompt: {condition_dict['system_prompt'][:100]}...")
            print(f"{'='*60}")

            extract_activations_with_system_prompt(
                model=model,
                tasks=tasks,
                layers=resolved_layers,
                system_prompt=condition_dict["system_prompt"],
                condition_name=condition_dict["name"],
                output_dir=condition_dir,
                selector_names=config.selectors,
                temperature=config.temperature,
                max_new_tokens=config.max_new_tokens,
                resume=args.resume,
                save_every=args.save_every,
                task_origins=[o.value for o in task_origins],
                layers_config=config.layers_to_extract,
                seed=config.task_sampling_seed,
            )

    # Compute difference-in-means
    condition_keys = list(config.conditions.keys())
    if len(condition_keys) != 2:
        raise ValueError(f"Expected exactly 2 conditions, got {len(condition_keys)}")

    # Assume first is positive, second is negative (or use explicit naming)
    positive_key = "positive" if "positive" in condition_keys else condition_keys[0]
    negative_key = "negative" if "negative" in condition_keys else condition_keys[1]

    positive_dir = output_dir / positive_key
    negative_dir = output_dir / negative_key

    print(f"\n{'='*60}")
    print(f"Computing difference-in-means: {positive_key} - {negative_key}")
    print(f"{'='*60}")

    # Determine resolved layers from extraction metadata
    with open(positive_dir / "extraction_metadata.json") as f:
        pos_metadata = json.load(f)
    resolved_layers = pos_metadata["layers_resolved"]

    vectors_by_selector, vector_norms, activation_norms = compute_all_concept_vectors(
        positive_dir=positive_dir,
        negative_dir=negative_dir,
        selector_names=config.selectors,
        layers=resolved_layers,
    )

    # Build metadata for manifest
    manifest_metadata = {
        "experiment_id": config.experiment_id,
        "model": config.model,
        "n_tasks": config.n_tasks,
        "task_origins": config.task_origins,
        "task_sampling_seed": config.task_sampling_seed,
        "positive_condition": {
            "key": positive_key,
            **config.conditions[positive_key],
        },
        "negative_condition": {
            "key": negative_key,
            **config.conditions[negative_key],
        },
        "layers_config": config.layers_to_extract,
        "layers_resolved": resolved_layers,
        "temperature": config.temperature,
        "max_new_tokens": config.max_new_tokens,
    }

    save_concept_vectors(
        vectors_by_selector,
        output_dir,
        manifest_metadata,
        vector_norms=vector_norms,
        activation_norms=activation_norms,
    )

    print(f"\nSaved concept vectors to {output_dir}/vectors/")
    print(f"Manifest: {output_dir}/manifest.json")
    for selector_name, vectors in vectors_by_selector.items():
        print(f"\n  Selector: {selector_name}")
        print(f"  Layers: {list(vectors.keys())}")
        for layer, vec in vectors.items():
            orig_norm = vector_norms[selector_name][layer]
            act_norm = activation_norms[selector_name][layer]
            print(f"    Layer {layer}: shape={vec.shape}, orig_norm={orig_norm:.1f}, act_norm={act_norm:.1f}")


if __name__ == "__main__":
    main()
