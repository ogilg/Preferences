"""Extract content embeddings from Gemma-2 9B base.

For each task prompt, runs it through google/gemma-2-9b (base model, no chat template),
takes the last-token hidden state from the final layer as the content embedding (3584d).

The base model is loaded in bf16 to fit in GPU memory.
"""

import csv
import gc
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.task_data import OriginDataset, load_filtered_tasks

RUN_DIR = Path("results/experiments/gemma3_3k_run2/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0")
OUTPUT_DIR = Path("activations/content_embeddings")
OUTPUT_PATH = OUTPUT_DIR / "embeddings_gemma-2-9b-base.npz"
MODEL_NAME = "google/gemma-2-9b"
BATCH_SIZE = 16


def load_experimental_task_ids() -> set[str]:
    csv_path = RUN_DIR / "thurstonian_a1ebd06e.csv"
    task_ids = set()
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_ids.add(row["task_id"])
    return task_ids


def main() -> None:
    # Load task prompts
    experimental_ids = load_experimental_task_ids()
    origins = [OriginDataset.WILDCHAT, OriginDataset.ALPACA, OriginDataset.MATH, OriginDataset.BAILBENCH]
    tasks = load_filtered_tasks(n=30000, origins=origins, seed=42, task_ids=experimental_ids)
    print(f"Loaded {len(tasks)} tasks")

    task_ids = [t.id for t in tasks]
    prompts = [t.prompt for t in tasks]

    # Check for resume
    if OUTPUT_PATH.exists():
        existing = np.load(OUTPUT_PATH, allow_pickle=True)
        existing_set = set(existing["task_ids"])
        remaining_indices = [i for i, tid in enumerate(task_ids) if tid not in existing_set]
        if not remaining_indices:
            print("All embeddings already extracted.")
            return
        print(f"Resume: {len(existing_set)} existing, {len(remaining_indices)} remaining")
        remaining_ids = [task_ids[i] for i in remaining_indices]
        remaining_prompts = [prompts[i] for i in remaining_indices]
    else:
        remaining_ids = task_ids
        remaining_prompts = prompts

    # Load model
    print(f"Loading {MODEL_NAME} in bf16...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir="/workspace/hf_cache/hub",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
        cache_dir="/workspace/hf_cache/hub",
    )
    model.eval()
    print(f"Model loaded. Hidden dim: {model.config.hidden_size}, Layers: {model.config.num_hidden_layers}")

    # Storage
    all_task_ids: list[str] = []
    all_embeddings: list[np.ndarray] = []

    if OUTPUT_PATH.exists():
        existing = np.load(OUTPUT_PATH, allow_pickle=True)
        all_task_ids = list(existing["task_ids"])
        all_embeddings = list(existing["embeddings"])

    n_processed = 0
    for batch_start in tqdm(range(0, len(remaining_ids), BATCH_SIZE), desc="Embedding"):
        batch_ids = remaining_ids[batch_start:batch_start + BATCH_SIZE]
        batch_prompts = remaining_prompts[batch_start:batch_start + BATCH_SIZE]

        try:
            encoded = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to("cuda")

            with torch.inference_mode():
                outputs = model(**encoded, output_hidden_states=True)

            # Last layer hidden states, last token position
            last_hidden = outputs.hidden_states[-1]  # (batch, seq, 3584)
            # With left-padding, the last real token is always at position -1
            embeddings = last_hidden[:, -1, :].float().cpu().numpy()

            for i, tid in enumerate(batch_ids):
                all_task_ids.append(tid)
                all_embeddings.append(embeddings[i])
                n_processed += 1

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            tqdm.write(f"OOM on batch of {len(batch_ids)}, processing one by one")
            for tid, prompt in zip(batch_ids, batch_prompts):
                try:
                    encoded = tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=2048,
                    ).to("cuda")
                    with torch.inference_mode():
                        outputs = model(**encoded, output_hidden_states=True)
                    embedding = outputs.hidden_states[-1][0, -1, :].float().cpu().numpy()
                    all_task_ids.append(tid)
                    all_embeddings.append(embedding)
                    n_processed += 1
                    del outputs
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    gc.collect()
                    tqdm.write(f"OOM on {tid}, skipping")

        if n_processed > 0 and n_processed % 500 == 0:
            _save(all_task_ids, all_embeddings)
            tqdm.write(f"Checkpoint: {len(all_task_ids)} embeddings")

        del outputs
        gc.collect()
        torch.cuda.empty_cache()

    _save(all_task_ids, all_embeddings)
    print(f"Done! {n_processed} new, {len(all_task_ids)} total, dim={all_embeddings[0].shape[0]}")


def _save(task_ids: list[str], embeddings: list[np.ndarray]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTPUT_PATH,
        task_ids=np.array(task_ids),
        embeddings=np.stack(embeddings),
        model_name=np.array("google/gemma-2-9b"),
    )


if __name__ == "__main__":
    main()
