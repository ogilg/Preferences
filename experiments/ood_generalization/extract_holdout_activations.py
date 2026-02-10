"""Extract activations for holdout system prompts."""

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


def main():
    with open(EXP_DIR / "target_tasks.json") as f:
        targets = json.load(f)
    with open(EXP_DIR / "holdout_prompts.json") as f:
        holdout_prompts = json.load(f)["prompts"]

    target_ids = {t["task_id"] for t in targets}
    tasks = load_filtered_tasks(n=100000, origins=ALL_ORIGINS, task_ids=target_ids)
    task_map = {t.id: t for t in tasks}
    ordered_ids = [t["task_id"] for t in targets]
    ordered_tasks = [task_map[tid] for tid in ordered_ids]

    print("Loading HuggingFace model...")
    model = HuggingFaceModel(model_name="gemma-3-27b", dtype="bfloat16", device="cuda")
    print(f"Model loaded. n_layers={model.n_layers}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for i, sp in enumerate(holdout_prompts):
        print(f"[{i+1}/{len(holdout_prompts)}] {sp['id']}...")
        messages_batch = []
        for task in ordered_tasks:
            msgs = [{"role": "system", "content": sp["text"]},
                    {"role": "user", "content": task.prompt}]
            messages_batch.append(msgs)

        result = model.get_activations_batch(
            messages_batch=messages_batch, layers=LAYERS, selector_names=[SELECTOR]
        )
        acts = {layer: result[SELECTOR][layer] for layer in LAYERS}

        np.savez(
            OUTPUT_DIR / f"{sp['id']}.npz",
            task_ids=np.array(ordered_ids),
            **{f"layer_{l}": acts[l] for l in LAYERS},
        )

    print(f"\nDone. {len(holdout_prompts)} holdout conditions extracted.")


if __name__ == "__main__":
    main()
