"""Extract activations for target tasks with and without system prompts.

Loads HuggingFace model, extracts activations at layers 31, 43, 55 using
prompt_last selector for each system prompt condition + baseline.
"""

import json
import time
from pathlib import Path

import numpy as np

from src.models import HuggingFaceModel
from src.task_data import load_filtered_tasks, OriginDataset

EXP_DIR = Path("experiments/ood_generalization")
OUTPUT_DIR = EXP_DIR / "activations"
LAYERS = [31, 43, 55]
SELECTOR = "prompt_last"

ALL_ORIGINS = [
    OriginDataset.WILDCHAT,
    OriginDataset.ALPACA,
    OriginDataset.MATH,
    OriginDataset.BAILBENCH,
    OriginDataset.STRESS_TEST,
]


def load_experiment_data():
    with open(EXP_DIR / "target_tasks.json") as f:
        targets = json.load(f)
    with open(EXP_DIR / "system_prompts.json") as f:
        prompts_data = json.load(f)
    return targets, prompts_data["prompts"]


def extract_for_condition(
    model: HuggingFaceModel,
    tasks: list,
    system_prompt: str | None,
    layers: list[int],
) -> dict[int, np.ndarray]:
    messages_batch = []
    for task in tasks:
        msgs = []
        if system_prompt is not None:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": task.prompt})
        messages_batch.append(msgs)

    result = model.get_activations_batch(
        messages_batch=messages_batch,
        layers=layers,
        selector_names=[SELECTOR],
    )
    return {layer: result[SELECTOR][layer] for layer in layers}


def main():
    targets_info, system_prompts = load_experiment_data()
    target_ids = {t["task_id"] for t in targets_info}

    print(f"Loading {len(target_ids)} target tasks...")
    tasks = load_filtered_tasks(n=100000, origins=ALL_ORIGINS, task_ids=target_ids)
    task_map = {t.id: t for t in tasks}

    ordered_ids = [t["task_id"] for t in targets_info]
    ordered_tasks = [task_map[tid] for tid in ordered_ids]
    print(f"Loaded tasks: {[t.id for t in ordered_tasks]}")

    print("Loading HuggingFace model...")
    model = HuggingFaceModel(
        model_name="gemma-3-27b",
        dtype="bfloat16",
        device="cuda",
    )
    print(f"Model loaded. n_layers={model.n_layers}, hidden_dim={model.hidden_dim}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Baseline (no system prompt)
    print("\nExtracting baseline (no system prompt)...")
    start = time.time()
    baseline_acts = extract_for_condition(model, ordered_tasks, None, LAYERS)
    print(f"  Done in {time.time() - start:.1f}s")

    np.savez(
        OUTPUT_DIR / "baseline.npz",
        task_ids=np.array(ordered_ids),
        **{f"layer_{l}": baseline_acts[l] for l in LAYERS},
    )

    # Each system prompt condition
    all_results = {}
    for i, sp in enumerate(system_prompts):
        print(f"\n[{i+1}/{len(system_prompts)}] Extracting: {sp['id']}...")
        start = time.time()
        acts = extract_for_condition(model, ordered_tasks, sp["text"], LAYERS)
        elapsed = time.time() - start
        print(f"  Done in {elapsed:.1f}s. Shape: {acts[LAYERS[0]].shape}")

        np.savez(
            OUTPUT_DIR / f"{sp['id']}.npz",
            task_ids=np.array(ordered_ids),
            **{f"layer_{l}": acts[l] for l in LAYERS},
        )
        all_results[sp["id"]] = acts

    # Save metadata
    metadata = {
        "task_ids": ordered_ids,
        "layers": LAYERS,
        "selector": SELECTOR,
        "n_prompts": len(system_prompts),
        "prompt_ids": [sp["id"] for sp in system_prompts],
    }
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nExtraction complete. {len(system_prompts) + 1} conditions saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
