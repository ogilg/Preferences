"""Extract Gemma-3 27B activations for the experimental tasks.

Extracts prompt_last activations at layers 31, 43, 55.
Uses unsloth/gemma-3-27b-it-bnb-4bit to fit within workspace disk quota.
Hidden states are computed in bf16 despite quantized weights.
"""

import csv
import gc
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.probes.extraction.persistence import save_activations, save_manifest
from src.task_data import OriginDataset, load_filtered_tasks
from src.models.base import BATCHED_SELECTOR_REGISTRY

RUN_DIR = Path("results/experiments/gemma3_3k_run2/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0")
OUTPUT_DIR = Path("activations/gemma_3_27b")
MODEL_NAME = "unsloth/gemma-3-27b-it-bnb-4bit"
LAYERS = [31, 43, 55]
BATCH_SIZE = 4
SAVE_EVERY = 500


def load_experimental_task_ids() -> set[str]:
    csv_path = RUN_DIR / "thurstonian_a1ebd06e.csv"
    task_ids = set()
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_ids.add(row["task_id"])
    return task_ids


def main() -> None:
    experimental_ids = load_experimental_task_ids()
    print(f"Experimental task IDs: {len(experimental_ids)}")

    # Load tasks (stress_test data unavailable on this pod)
    origins = [OriginDataset.WILDCHAT, OriginDataset.ALPACA, OriginDataset.MATH, OriginDataset.BAILBENCH]
    tasks = load_filtered_tasks(n=30000, origins=origins, seed=42, task_ids=experimental_ids)
    print(f"Loaded {len(tasks)} tasks (2400 expected — stress_test data unavailable)")

    # Check for resume
    output_npz = OUTPUT_DIR / "activations_prompt_last.npz"
    existing_ids: set[str] = set()
    task_ids_list: list[str] = []
    activations: dict[str, dict[int, list[np.ndarray]]] = {"prompt_last": defaultdict(list)}
    completions: list[dict] = []

    if output_npz.exists():
        data = np.load(output_npz, allow_pickle=True)
        existing_ids = set(data["task_ids"])
        task_ids_list = list(data["task_ids"])
        for layer in LAYERS:
            key = f"layer_{layer}"
            if key in data:
                activations["prompt_last"][layer] = list(data[key])
        completions_path = OUTPUT_DIR / "completions_with_activations.json"
        if completions_path.exists():
            with open(completions_path) as f:
                completions = json.load(f)
        print(f"Resume: found {len(existing_ids)} existing activations")

    tasks_to_process = [t for t in tasks if t.id not in existing_ids]
    print(f"Tasks to process: {len(tasks_to_process)}")

    if not tasks_to_process:
        print("Nothing to do.")
        return

    # Load model with 4-bit quantization
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir="/workspace/hf_cache/hub",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="cuda",
        cache_dir="/workspace/hf_cache/hub",
        dtype=torch.bfloat16,  # Force bf16 for non-quantized layers to prevent fp16 overflow/NaN
    )
    model.eval()

    # Get layer accessor for Gemma-3 (language_model.model.layers)
    layers_module = model.model.language_model.layers
    n_layers = len(layers_module)
    hidden_dim = model.config.text_config.hidden_size
    print(f"Model loaded. Layers: {n_layers}, Hidden dim: {hidden_dim}")
    print(f"Extracting layers: {LAYERS}")

    # Process tasks one at a time (safer with quantized model)
    n_processed = 0
    for i, task in enumerate(tqdm(tasks_to_process, desc="Tasks")):
        try:
            messages = [{"role": "user", "content": task.prompt}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
            seq_len = encoded["input_ids"].shape[1]

            # Register hooks
            hook_activations: dict[int, torch.Tensor] = {}
            handles = []
            for layer_idx in LAYERS:
                def make_hook(l: int):
                    def hook_fn(module, input, output):
                        hidden = output[0] if isinstance(output, tuple) else output
                        hook_activations[l] = hidden.detach().cpu()
                    return hook_fn
                handles.append(layers_module[layer_idx].register_forward_hook(make_hook(layer_idx)))

            with torch.inference_mode():
                model(**encoded)

            for h in handles:
                h.remove()

            # Extract prompt_last: last token before generation prompt
            task_ids_list.append(task.id)
            for layer_idx in LAYERS:
                # prompt_last: last token of the input sequence
                act = hook_activations[layer_idx][0, -1, :].float().numpy()
                activations["prompt_last"][layer_idx].append(act)

            completions.append({
                "task_id": task.id,
                "task_prompt": task.prompt,
                "origin": task.origin.name,
            })
            n_processed += 1

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            tqdm.write(f"OOM on task {task.id}, skipping")
        except Exception as e:
            tqdm.write(f"Error on task {task.id}: {e}")

        if n_processed > 0 and n_processed % SAVE_EVERY == 0:
            tqdm.write(f"Checkpoint: saving {len(task_ids_list)} activations...")
            save_activations(OUTPUT_DIR, task_ids_list, activations)
            save_manifest(OUTPUT_DIR, completions)

        if (i + 1) % 100 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # Final save
    print(f"Saving {len(task_ids_list)} total activations...")
    save_activations(OUTPUT_DIR, task_ids_list, activations)
    save_manifest(OUTPUT_DIR, completions)
    print(f"Done! Processed {n_processed} new tasks, {len(task_ids_list)} total")


if __name__ == "__main__":
    main()
