"""Extract sentence-transformer embeddings for all tasks, saving in activation-compatible npz format."""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

COMPLETIONS_PATH = Path("activations/gemma_3_27b/completions_with_activations.json")
OUTPUT_PATH = Path("activations/sentence_transformer/embeddings.npz")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    with open(COMPLETIONS_PATH) as f:
        completions = json.load(f)

    task_ids = [c["task_id"] for c in completions]
    prompts = [c["task_prompt"] for c in completions]
    print(f"Loaded {len(task_ids)} tasks")

    model = SentenceTransformer(MODEL_NAME)
    print(f"Encoding with {MODEL_NAME}...")
    embeddings = model.encode(prompts, show_progress_bar=True, batch_size=256)
    print(f"Embeddings shape: {embeddings.shape}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTPUT_PATH,
        task_ids=np.array(task_ids),
        layer_0=embeddings,
    )
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
