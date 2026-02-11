"""Extract activations for competing preferences experiment.

For each competing prompt condition, extracts activations at the target crossed task.
Also extracts baseline (no system prompt) activations.

Saves per-prompt .npz files compatible with the evaluation script.
"""

import json
import time
from pathlib import Path

import numpy as np

from src.models import HuggingFaceModel
from src.task_data import OriginDataset, Task

EXP_DIR = Path("experiments/competing_preferences")
CROSSED_DIR = Path("experiments/crossed_preferences")
OUTPUT_DIR = EXP_DIR / "activations"
LAYERS = [31, 43, 55]
SELECTOR = "prompt_last"


def load_crossed_tasks() -> list[Task]:
    with open(CROSSED_DIR / "crossed_tasks.json") as f:
        raw = json.load(f)
    return [
        Task(
            prompt=t["prompt"],
            origin=OriginDataset.SYNTHETIC,
            id=t["task_id"],
            metadata={"topic": t["topic"], "category_shell": t["category_shell"]},
        )
        for t in raw
    ]


def load_competing_prompts() -> list[dict]:
    with open(EXP_DIR / "competing_prompts.json") as f:
        return json.load(f)["prompts"]


def extract_for_condition(
    model: HuggingFaceModel,
    tasks: list[Task],
    system_prompt: str | None,
    layers: list[int],
) -> dict[int, np.ndarray]:
    messages_batch = []
    for task in tasks:
        msgs: list[dict[str, str]] = []
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
    tasks = load_crossed_tasks()
    prompts = load_competing_prompts()
    ordered_ids = [t.id for t in tasks]

    print(f"Tasks: {len(tasks)}")
    print(f"Competing prompts: {len(prompts)}")

    print("Loading HuggingFace model...")
    model = HuggingFaceModel(
        model_name="gemma-3-27b",
        dtype="bfloat16",
        device="cuda",
    )
    print(f"Model loaded. n_layers={model.n_layers}, hidden_dim={model.hidden_dim}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Baseline
    baseline_path = OUTPUT_DIR / "baseline.npz"
    if baseline_path.exists():
        print("Baseline already exists, skipping")
    else:
        print("\nExtracting baseline (no system prompt)...")
        start = time.time()
        baseline_acts = extract_for_condition(model, tasks, None, LAYERS)
        print(f"  Done in {time.time() - start:.1f}s, shape: {baseline_acts[LAYERS[0]].shape}")
        np.savez(
            baseline_path,
            task_ids=np.array(ordered_ids),
            **{f"layer_{l}": baseline_acts[l] for l in LAYERS},
        )

    # Each competing prompt
    for i, sp in enumerate(prompts):
        npz_path = OUTPUT_DIR / f"{sp['id']}.npz"
        if npz_path.exists():
            print(f"[{i+1}/{len(prompts)}] {sp['id']} already exists, skipping")
            continue

        print(f"\n[{i+1}/{len(prompts)}] Extracting: {sp['id']}...")
        start = time.time()
        acts = extract_for_condition(model, tasks, sp["text"], LAYERS)
        elapsed = time.time() - start
        print(f"  Done in {elapsed:.1f}s, shape: {acts[LAYERS[0]].shape}")

        np.savez(
            npz_path,
            task_ids=np.array(ordered_ids),
            **{f"layer_{l}": acts[l] for l in LAYERS},
        )

    # Save metadata
    metadata = {
        "task_ids": ordered_ids,
        "layers": LAYERS,
        "selector": SELECTOR,
        "n_tasks": len(tasks),
        "n_prompts": len(prompts),
        "prompt_ids": [sp["id"] for sp in prompts],
    }
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nExtraction complete. {len(prompts) + 1} conditions saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
