"""Embed task prompts with sentence-transformers (all-MiniLM-L6-v2).

Covers all task IDs from the 10k train set + 4k eval set.
Saves activations/st_minilm/activations_prompt_last.npz with:
  task_ids: (N,) array of task ID strings
  layer_0:  (N, 384) float32 embedding matrix
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parents[2]))
from src.task_data.loader import load_tasks
from src.task_data.task import OriginDataset

load_dotenv()

TRAIN_CSV = Path(
    "results/experiments/gemma3_10k_run1/pre_task_active_learning"
    "/completion_preference_gemma-3-27b_completion_canonical_seed0"
    "/thurstonian_80fa9dc8.csv"
)
EVAL_CSV = Path(
    "results/experiments/gemma3_4k_pre_task/pre_task_active_learning"
    "/completion_preference_gemma-3-27b_completion_canonical_seed0"
    "/thurstonian_a67822c5.csv"
)
OUT_PATH = Path("activations/st_minilm/activations_prompt_last.npz")
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 256

# Collect all task IDs we need
train_ids = set(pd.read_csv(TRAIN_CSV)["task_id"].tolist())
eval_ids = set(pd.read_csv(EVAL_CSV)["task_id"].tolist())
all_ids = train_ids | eval_ids
print(f"Train IDs: {len(train_ids)}, eval IDs: {len(eval_ids)}, total: {len(all_ids)}")

# Load all tasks from source data, filter to needed IDs
all_origins = [
    OriginDataset.WILDCHAT,
    OriginDataset.ALPACA,
    OriginDataset.MATH,
    OriginDataset.BAILBENCH,
    OriginDataset.STRESS_TEST,
]
print("Loading tasks from source data...")
tasks = load_tasks(n=999999, origins=all_origins, filter_fn=lambda t: t.id in all_ids)
print(f"Loaded {len(tasks)} tasks")

found_ids = {t.id for t in tasks}
missing = all_ids - found_ids
if missing:
    print(f"WARNING: {len(missing)} task IDs not found in source data")
    print(f"  Examples: {list(missing)[:5]}")

# Embed
print(f"Embedding with {MODEL_NAME}...")
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(MODEL_NAME)

prompts = [t.prompt for t in tasks]
task_ids_arr = np.array([t.id for t in tasks])

embeddings = model.encode(
    prompts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=False,
    convert_to_numpy=True,
)
print(f"Embeddings shape: {embeddings.shape}")

# Save
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
np.savez(OUT_PATH, task_ids=task_ids_arr, layer_0=embeddings.astype(np.float32))
print(f"Saved {OUT_PATH}")
