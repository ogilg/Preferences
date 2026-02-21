"""Extract main activations for probe training.

Extracts activations for training, eval, and OOD tasks at layers 31, 43, 55.
Saves to activations/gemma_3_27b/activations_prompt_last.npz.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

LAYERS = [31, 43, 55]
BATCH_SIZE = 16
SAVE_PATH = REPO_ROOT / "activations" / "gemma_3_27b" / "activations_prompt_last.npz"


def load_task_ids_from_thurstonian(csv_path: Path) -> list[str]:
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        return [row["task_id"] for row in reader]


def load_ood_task_ids() -> list[str]:
    beh_path = REPO_ROOT / "results/ood/minimal_pairs_v7/behavioral.json"
    with open(beh_path) as f:
        data = json.load(f)
    return list(data["conditions"]["baseline"]["task_rates"].keys())


def load_tasks_by_ids(task_ids: list[str]) -> list:
    from src.task_data import load_tasks, OriginDataset

    all_tasks = load_tasks(
        n=100000,
        origins=[
            OriginDataset.WILDCHAT,
            OriginDataset.ALPACA,
            OriginDataset.MATH,
            OriginDataset.BAILBENCH,
            OriginDataset.STRESS_TEST,
        ],
    )
    lookup = {t.id: t for t in all_tasks}
    missing = [tid for tid in task_ids if tid not in lookup]
    if missing:
        print(f"WARNING: {len(missing)} tasks not found: {missing[:5]}")
    return [lookup[tid] for tid in task_ids if tid in lookup]


def main() -> None:
    from src.models.huggingface_model import HuggingFaceModel
    from src.probes.extraction.simple import extract_activations

    # Collect task IDs from training and eval runs + OOD tasks
    train_csv = REPO_ROOT / "results/experiments/gemma3_10k_run1/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0/thurstonian_80fa9dc8.csv"
    eval_csv = REPO_ROOT / "results/experiments/gemma3_4k_pre_task/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0/thurstonian_a67822c5.csv"

    train_ids = load_task_ids_from_thurstonian(train_csv)
    eval_ids = load_task_ids_from_thurstonian(eval_csv)
    ood_ids = load_ood_task_ids()

    # Union (dedup) - maintain order: train first, then eval, then OOD extras
    all_ids = list(dict.fromkeys(train_ids + eval_ids + ood_ids))
    print(f"Train tasks: {len(train_ids)}, Eval tasks: {len(eval_ids)}, OOD tasks: {len(ood_ids)}, Union: {len(all_ids)}")

    tasks = load_tasks_by_ids(all_ids)
    print(f"Loaded {len(tasks)} tasks from datasets")

    if SAVE_PATH.exists():
        print(f"Main activations already exist at {SAVE_PATH}, skipping")
        return

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model = HuggingFaceModel("gemma-3-27b")
    print(f"Model loaded: {model.n_layers} layers, {model.hidden_dim}D")

    print(f"Extracting activations for {len(tasks)} tasks at layers {LAYERS}...")
    # Use simple extractor - saves directly to a dir, then rename to expected path
    tmp_dir = SAVE_PATH.parent / "tmp_extract"
    extract_activations(
        model=model,
        tasks=tasks,
        layers=LAYERS,
        selectors=["prompt_last"],
        batch_size=BATCH_SIZE,
        save_path=tmp_dir,
        system_prompt=None,
    )

    # Rename to expected path
    tmp_file = tmp_dir / "activations_prompt_last.npz"
    tmp_file.rename(SAVE_PATH)
    tmp_dir.rmdir()
    print(f"Saved to {SAVE_PATH}")


if __name__ == "__main__":
    main()
