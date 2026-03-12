"""Phase 1c+1d: Extract activations from filtered completions and compute persona vectors.

Loads Gemma 3-27B-IT once, extracts mean (response-token-averaged) activations for all
10 (persona, condition) combos, then computes unit-norm mean-difference vectors.
"""

import gc
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.models.huggingface_model import HuggingFaceModel
from src.probes.extraction.extract import batched_extraction
from src.probes.extraction.persistence import save_activations
from src.types import Message

PERSONAS = ["sadist", "villain", "aesthete", "lazy", "stem_obsessive"]
CONDITIONS = ["pos", "neg"]
LAYERS = [23, 29, 35, 41]
SELECTOR = "mean"
BATCH_SIZE = 16
SAVE_EVERY = 200

ARTIFACTS_DIR = Path("experiments/new_persona_steering/artifacts")
CONTRASTIVE_DIR = Path("results/experiments/persona_steering_v2/contrastive")
RESULTS_DIR = Path("results/experiments/persona_steering_v2")


def build_messages(task_prompt: str, completion: str, system_prompt: str) -> list[Message]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_prompt},
        {"role": "assistant", "content": completion},
    ]


def extract_all(model: HuggingFaceModel) -> None:
    resolved_layers = [model.resolve_layer(layer) for layer in LAYERS]
    print(f"Resolved layers: {LAYERS} -> {resolved_layers} ({model.n_layers} total)")

    for persona in PERSONAS:
        with open(ARTIFACTS_DIR / f"{persona}.json") as f:
            persona_data = json.load(f)

        for condition in CONDITIONS:
            output_dir = RESULTS_DIR / persona / "activations" / f"{condition}"
            activations_file = output_dir / f"activations_{SELECTOR}.npz"
            if activations_file.exists():
                print(f"Skipping {persona}/{condition} — already extracted")
                continue

            filtered_path = CONTRASTIVE_DIR / f"{persona}_{condition}_filtered.json"
            with open(filtered_path) as f:
                completions_data = json.load(f)

            system_prompt = persona_data["positive"] if condition == "pos" else persona_data["negative"]

            items: list[tuple[str, list[Message]]] = [
                (c["task_id"], build_messages(c["task_prompt"], c["completion"], system_prompt))
                for c in completions_data
            ]

            print(f"\n--- {persona}/{condition}: {len(items)} completions ---")

            task_ids: list[str] = []
            activations: dict[str, dict[int, list[np.ndarray]]] = {SELECTOR: defaultdict(list)}

            stats = batched_extraction(
                model=model,
                items=items,
                layers=resolved_layers,
                selectors=[SELECTOR],
                batch_size=BATCH_SIZE,
                task_ids=task_ids,
                activations=activations,
                output_dir=output_dir,
                save_every=SAVE_EVERY,
            )

            if stats.n_new > 0:
                save_activations(output_dir, task_ids, activations)
                print(f"Saved {stats.n_new} activations to {output_dir}")

            if stats.n_failures > 0:
                print(f"WARNING: {stats.n_failures} failures for {persona}/{condition}")

            gc.collect()
            torch.cuda.empty_cache()


def compute_vectors() -> None:
    print("\n=== Computing persona vectors ===")

    for persona in PERSONAS:
        vectors_dir = RESULTS_DIR / persona / "vectors"
        vectors_dir.mkdir(parents=True, exist_ok=True)

        for layer in LAYERS:
            pos_path = RESULTS_DIR / persona / "activations" / "pos" / f"activations_{SELECTOR}.npz"
            neg_path = RESULTS_DIR / persona / "activations" / "neg" / f"activations_{SELECTOR}.npz"

            pos_data = np.load(pos_path, allow_pickle=True)
            neg_data = np.load(neg_path, allow_pickle=True)

            pos_acts = pos_data[f"layer_{layer}"]  # (n_pos, d_model)
            neg_acts = neg_data[f"layer_{layer}"]  # (n_neg, d_model)

            pos_mean = pos_acts.mean(axis=0)
            neg_mean = neg_acts.mean(axis=0)

            direction = pos_mean - neg_mean
            norm = np.linalg.norm(direction)
            direction_unit = direction / norm

            vector_path = vectors_dir / f"{persona}_{SELECTOR}_L{layer}_direction.npy"
            np.save(vector_path, direction_unit)
            print(f"{persona} L{layer}: norm={norm:.2f}, shape={direction_unit.shape}, n_pos={len(pos_acts)}, n_neg={len(neg_acts)}")


if __name__ == "__main__":
    print("Loading Gemma 3-27B-IT...")
    model = HuggingFaceModel("gemma-3-27b", max_new_tokens=256)
    print(f"Model loaded: {model.n_layers} layers, hidden_dim={model.hidden_dim}")

    extract_all(model)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    compute_vectors()
    print("\nDone!")
