"""Extract activations for crossed preferences experiment.

Loads HuggingFace model, extracts activations at layers 31, 43, 55 using
prompt_last selector for all target tasks under each system prompt condition.

Handles crossed tasks (40), pure reference tasks (16), and subtle tasks (8).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.models import HuggingFaceModel
from src.task_data import OriginDataset, Task

EXP_DIR = Path("experiments/crossed_preferences")
HIDDEN_DIR = Path("experiments/hidden_preferences")
OUTPUT_DIR = EXP_DIR / "activations"
LAYERS = [31, 43, 55]
SELECTOR = "prompt_last"


def load_all_tasks() -> list[Task]:
    tasks = []

    # Crossed tasks
    with open(EXP_DIR / "crossed_tasks.json") as f:
        for t in json.load(f):
            tasks.append(Task(
                prompt=t["prompt"],
                origin=OriginDataset.SYNTHETIC,
                id=t["task_id"],
                metadata={"topic": t["topic"], "category_shell": t["category_shell"]},
            ))

    # Pure reference tasks
    with open(HIDDEN_DIR / "target_tasks.json") as f:
        for t in json.load(f):
            tasks.append(Task(
                prompt=t["prompt"],
                origin=OriginDataset.SYNTHETIC,
                id=t["task_id"],
                metadata={"topic": t["topic"]},
            ))

    # Subtle tasks
    with open(EXP_DIR / "subtle_target_tasks.json") as f:
        for t in json.load(f):
            tasks.append(Task(
                prompt=t["prompt"],
                origin=OriginDataset.SYNTHETIC,
                id=t["task_id"],
                metadata={"topic": t["topic"]},
            ))

    return tasks


def load_system_prompts(prompt_source: str) -> list[dict]:
    prompts = []
    if prompt_source in ("iteration", "all"):
        with open(HIDDEN_DIR / "system_prompts.json") as f:
            prompts.extend(json.load(f)["prompts"])
    if prompt_source in ("holdout", "all"):
        with open(HIDDEN_DIR / "holdout_prompts.json") as f:
            prompts.extend(json.load(f)["prompts"])
    if prompt_source in ("subtle", "all"):
        with open(EXP_DIR / "subtle_prompts.json") as f:
            prompts.extend(json.load(f)["prompts"])
    return prompts


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
    parser.add_argument("--prompt-source", default="all",
                        choices=["iteration", "holdout", "subtle", "all"])
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip conditions where .npz already exists")
    args = parser.parse_args()

    tasks = load_all_tasks()
    system_prompts = load_system_prompts(args.prompt_source)
    ordered_ids = [t.id for t in tasks]

    print(f"Tasks: {len(tasks)}")
    print(f"System prompts: {len(system_prompts)}")

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
    if args.skip_existing and baseline_path.exists():
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

    # Each system prompt
    for i, sp in enumerate(system_prompts):
        npz_path = OUTPUT_DIR / f"{sp['id']}.npz"
        if args.skip_existing and npz_path.exists():
            print(f"[{i+1}/{len(system_prompts)}] {sp['id']} already exists, skipping")
            continue

        print(f"\n[{i+1}/{len(system_prompts)}] Extracting: {sp['id']}...")
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
        "n_prompts": len(system_prompts),
        "prompt_ids": [sp["id"] for sp in system_prompts],
    }
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nExtraction complete. {len(system_prompts) + 1} conditions saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
