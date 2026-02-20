"""Compute pairwise accuracy for each Gemma-2 HOO fold at L23, update hoo_summary.json."""

import json
import sys
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from src.probes.residualization import build_task_groups

load_dotenv()

HOO_SUMMARY = Path("results/probes/gemma2_10k_hoo_topic/hoo_summary.json")
PROBES_DIR = Path("results/probes/gemma2_10k_hoo_topic/probes")
ACTIVATIONS_PATH = Path("activations/gemma_2_27b_base/activations_prompt_last.npz")
TOPICS_JSON = Path("src/analysis/topic_classification/output/topics.json")
THURSTONIAN_CSV = Path(
    "results/experiments/gemma3_10k_run1/pre_task_active_learning/"
    "completion_preference_gemma-3-27b_completion_canonical_seed0/"
    "thurstonian_80fa9dc8.csv"
)
LAYER = 23
N_PAIRS = 100_000
SEED = 0

# Load activations once
print("Loading activations...")
data = np.load(ACTIVATIONS_PATH, allow_pickle=True)
task_ids_all = data["task_ids"]
acts_all = data[f"layer_{LAYER}"]
print(f"Activations shape: {acts_all.shape}, tasks: {len(task_ids_all)}")

# Build task_id -> index for fast lookup
tid_to_idx = {tid: i for i, tid in enumerate(task_ids_all)}

# Load Thurstonian scores (train set, 10k tasks)
print("Loading Thurstonian scores...")
df = pd.read_csv(THURSTONIAN_CSV)
df = df[["task_id", "mu"]].copy()
tid_to_mu = dict(zip(df["task_id"], df["mu"]))
print(f"Scores: {len(tid_to_mu)} tasks")

# Build topic groups for all scored tasks
print("Building topic groups...")
all_scored_ids = set(tid_to_mu.keys())
task_groups = build_task_groups(all_scored_ids, grouping="topic", topics_json=TOPICS_JSON)
print(f"Tasks with topic labels: {len(task_groups)}")

# Load HOO summary
with open(HOO_SUMMARY) as f:
    summary = json.load(f)

rng = np.random.default_rng(SEED)

for fold in summary["folds"]:
    fold_idx = fold["fold_idx"]
    held_out_groups = set(fold["held_out_groups"])

    # Get held-out task IDs: scored tasks whose topic is in held_out_groups
    hoo_task_ids = [
        tid for tid, grp in task_groups.items()
        if grp in held_out_groups and tid in tid_to_mu
    ]
    print(f"\nFold {fold_idx}: held_out={held_out_groups}, n_hoo={len(hoo_task_ids)}")

    if not hoo_task_ids:
        print("  No tasks — skipping")
        continue

    # Load probe weights
    probe_path = PROBES_DIR / f"probe_hoo_fold{fold_idx}_ridge_L{LAYER}.npy"
    if not probe_path.exists():
        print(f"  Probe not found: {probe_path} — skipping")
        continue
    w = np.load(probe_path)

    # Get activations for held-out tasks
    hoo_indices = [tid_to_idx[tid] for tid in hoo_task_ids if tid in tid_to_idx]
    hoo_task_ids_filtered = [tid for tid in hoo_task_ids if tid in tid_to_idx]
    X_hoo = acts_all[hoo_indices]

    # Compute probe predictions
    y_pred = X_hoo @ w[:-1] + w[-1]
    y_true = np.array([tid_to_mu[tid] for tid in hoo_task_ids_filtered])

    # Pairwise accuracy
    n = len(y_pred)
    i_idx = rng.integers(0, n, size=N_PAIRS)
    j_idx = rng.integers(0, n, size=N_PAIRS)
    valid = i_idx != j_idx
    i_idx, j_idx = i_idx[valid], j_idx[valid]

    pred_diff = y_pred[i_idx] - y_pred[j_idx]
    true_diff = y_true[i_idx] - y_true[j_idx]
    acc = float((np.sign(pred_diff) == np.sign(true_diff)).mean())
    print(f"  HOO pairwise acc (L{LAYER}): {acc:.4f}")

    # Update summary
    fold["layers"][f"ridge_L{LAYER}"]["hoo_acc"] = acc

# Write updated summary
with open(HOO_SUMMARY, "w") as f:
    json.dump(summary, f, indent=2)
print("\nUpdated hoo_summary.json with hoo_acc for all folds.")
