"""Extract activations for hidden preferences experiment.

Loads HuggingFace model, extracts activations at layers 31, 43, 55 using
prompt_last selector for each system prompt condition + baseline.

Handles both hidden-preference tasks (16 synthetic) and OOD control tasks (6).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.models import HuggingFaceModel
from src.task_data import load_filtered_tasks, OriginDataset, Task

EXP_DIR = Path("experiments/hidden_preferences")
OOD_DIR = Path("experiments/ood_generalization")
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


def load_hidden_target_tasks() -> list[Task]:
    """Load 16 synthetic target tasks."""
    with open(EXP_DIR / "target_tasks.json") as f:
        raw = json.load(f)
    return [
        Task(
            prompt=t["prompt"],
            origin=OriginDataset.SYNTHETIC,
            id=t["task_id"],
            metadata={"topic": t["topic"]},
        )
        for t in raw
    ]


def load_ood_target_tasks() -> list[Task]:
    """Load 6 OOD target tasks for positive controls."""
    with open(OOD_DIR / "target_tasks.json") as f:
        raw = json.load(f)
    target_ids = {t["task_id"] for t in raw}
    tasks = load_filtered_tasks(n=100000, origins=ALL_ORIGINS, task_ids=target_ids)
    task_map = {t.id: t for t in tasks}
    return [task_map[t["task_id"]] for t in raw]


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-file", default="system_prompts.json")
    parser.add_argument("--control-mode", action="store_true",
                        help="Extract for OOD target tasks (positive controls)")
    args = parser.parse_args()

    with open(EXP_DIR / args.prompt_file) as f:
        prompt_data = json.load(f)
    system_prompts = prompt_data["prompts"]

    if args.control_mode:
        tasks = load_ood_target_tasks()
        output_dir = OUTPUT_DIR / "controls"
        print(f"Control mode: extracting for {len(tasks)} OOD target tasks")
    else:
        tasks = load_hidden_target_tasks()
        output_dir = OUTPUT_DIR
        print(f"Extracting for {len(tasks)} hidden-preference target tasks")

    ordered_ids = [t.id for t in tasks]
    print(f"Tasks: {ordered_ids}")

    print("Loading HuggingFace model...")
    model = HuggingFaceModel(
        model_name="gemma-3-27b",
        dtype="bfloat16",
        device="cuda",
    )
    print(f"Model loaded. n_layers={model.n_layers}, hidden_dim={model.hidden_dim}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Baseline (no system prompt)
    print("\nExtracting baseline (no system prompt)...")
    start = time.time()
    baseline_acts = extract_for_condition(model, tasks, None, LAYERS)
    print(f"  Done in {time.time() - start:.1f}s")

    np.savez(
        output_dir / "baseline.npz",
        task_ids=np.array(ordered_ids),
        **{f"layer_{l}": baseline_acts[l] for l in LAYERS},
    )

    # Each system prompt condition
    for i, sp in enumerate(system_prompts):
        print(f"\n[{i+1}/{len(system_prompts)}] Extracting: {sp['id']}...")
        start = time.time()
        acts = extract_for_condition(model, tasks, sp["text"], LAYERS)
        elapsed = time.time() - start
        print(f"  Done in {elapsed:.1f}s. Shape: {acts[LAYERS[0]].shape}")

        np.savez(
            output_dir / f"{sp['id']}.npz",
            task_ids=np.array(ordered_ids),
            **{f"layer_{l}": acts[l] for l in LAYERS},
        )

    # Save metadata
    metadata = {
        "task_ids": ordered_ids,
        "layers": LAYERS,
        "selector": SELECTOR,
        "n_prompts": len(system_prompts),
        "prompt_ids": [sp["id"] for sp in system_prompts],
        "control_mode": args.control_mode,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nExtraction complete. {len(system_prompts) + 1} conditions saved to {output_dir}")


if __name__ == "__main__":
    main()
