"""Extract content embeddings from Gemma-2 27B base.

For each task prompt, runs it through google/gemma-2-27b (base model, no chat template),
takes the last-token hidden state as the content embedding.

This is a base model — no chat template, just raw text. We tokenize the
task prompt directly and take the final hidden state from the last layer.
"""

import gc
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

COMPLETIONS_JSON = Path("activations/gemma_3_27b/completions_with_activations.json")
OUTPUT_DIR = Path("activations/content_embeddings")
OUTPUT_PATH = OUTPUT_DIR / "embeddings_gemma-2-27b-base.npz"
MODEL_NAME = "google/gemma-2-27b"
BATCH_SIZE = 8


def main() -> None:
    print(f"Loading task prompts from {COMPLETIONS_JSON}")
    with open(COMPLETIONS_JSON) as f:
        completions = json.load(f)
    print(f"  {len(completions)} tasks")

    task_ids = [c["task_id"] for c in completions]
    prompts = [c["task_prompt"] for c in completions]

    # Check for resume
    if OUTPUT_PATH.exists():
        existing = np.load(OUTPUT_PATH, allow_pickle=True)
        existing_ids = set(existing["task_ids"])
        remaining = [(tid, p) for tid, p in zip(task_ids, prompts) if tid not in existing_ids]
        if not remaining:
            print("All embeddings already extracted.")
            return
        print(f"Resume: {len(existing_ids)} existing, {len(remaining)} remaining")
    else:
        remaining = list(zip(task_ids, prompts))

    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    )
    model.eval()
    print(f"Model loaded. Hidden dim: {model.config.hidden_size}")

    all_task_ids: list[str] = []
    all_embeddings: list[np.ndarray] = []

    # Load existing if resuming
    if OUTPUT_PATH.exists():
        existing = np.load(OUTPUT_PATH, allow_pickle=True)
        all_task_ids = list(existing["task_ids"])
        all_embeddings = list(existing["embeddings"])

    n_processed = 0
    for batch_start in tqdm(range(0, len(remaining), BATCH_SIZE), desc="Embedding batches"):
        batch = remaining[batch_start:batch_start + BATCH_SIZE]
        batch_ids = [b[0] for b in batch]
        batch_prompts = [b[1] for b in batch]

        try:
            # Tokenize with left-padding for batched last-token extraction
            tokenizer.padding_side = "left"
            encoded = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to("cuda")

            with torch.inference_mode():
                outputs = model(**encoded, output_hidden_states=True)

            # Last layer hidden states: (batch, seq_len, d_model)
            last_hidden = outputs.hidden_states[-1]

            # Get last non-padding token for each sample
            attention_mask = encoded["attention_mask"]
            # With left-padding, last real token is at the last position
            seq_lengths = attention_mask.sum(dim=1)  # number of non-pad tokens
            # last real token index = seq_len - 1 (since left-padded, all end at the end)
            batch_size = last_hidden.shape[0]
            last_token_idx = last_hidden.shape[1] - 1  # last position

            embeddings = last_hidden[:, last_token_idx, :].float().cpu().numpy()

            for i, tid in enumerate(batch_ids):
                all_task_ids.append(tid)
                all_embeddings.append(embeddings[i])
                n_processed += 1

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            tqdm.write(f"OOM on batch of {len(batch)}, processing one by one")
            for tid, prompt in batch:
                try:
                    encoded = tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=2048,
                    ).to("cuda")
                    with torch.inference_mode():
                        outputs = model(**encoded, output_hidden_states=True)
                    last_hidden = outputs.hidden_states[-1]
                    embedding = last_hidden[0, -1, :].float().cpu().numpy()
                    all_task_ids.append(tid)
                    all_embeddings.append(embedding)
                    n_processed += 1
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    gc.collect()
                    tqdm.write(f"OOM on single prompt {tid}, skipping")

        # Checkpoint every 500
        if n_processed > 0 and n_processed % 500 == 0:
            _save(all_task_ids, all_embeddings)
            tqdm.write(f"Checkpoint: {len(all_task_ids)} embeddings saved")

        # Free intermediate tensors
        del outputs
        gc.collect()
        torch.cuda.empty_cache()

    _save(all_task_ids, all_embeddings)
    print(f"Done! {n_processed} new embeddings, {len(all_task_ids)} total")


def _save(task_ids: list[str], embeddings: list[np.ndarray]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    emb_array = np.stack(embeddings)
    np.savez(
        OUTPUT_PATH,
        task_ids=np.array(task_ids),
        embeddings=emb_array,
        model_name=np.array("google/gemma-2-27b"),
    )


if __name__ == "__main__":
    main()
