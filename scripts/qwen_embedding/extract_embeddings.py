"""Extract Qwen3-Embedding-8B embeddings for all task prompts."""

import gc
import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.probes.content_embedding import save_content_embeddings

completions_json = Path("activations/gemma_3_27b/completions_with_activations.json")
output_path = Path("activations/qwen3_emb_8b/activations_prompt_last.npz")

with open(completions_json) as f:
    completions = json.load(f)

task_ids = np.array([c["task_id"] for c in completions])
prompts = [c["task_prompt"] for c in completions]

print(f"Loaded {len(prompts)} task prompts")
print("Loading Qwen3-Embedding-8B...")

model = SentenceTransformer("Qwen/Qwen3-Embedding-8B")
print(f"Model loaded. Encoding with batch_size=64...")

embeddings = model.encode(prompts, show_progress_bar=True, convert_to_numpy=True, batch_size=64)

print(f"Extracted {len(task_ids)} embeddings, shape: {embeddings.shape}")

save_content_embeddings(
    path=output_path,
    task_ids=task_ids,
    embeddings=embeddings,
)

print("Done. Freeing GPU memory...")
del model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("GPU memory freed.")
