"""Compute real pairwise accuracy for Gemma-2 HOO folds at L23.

Uses actual pairwise choices from measurements.yaml (not Thurstonian scores).
Matches how Gemma-3 hoo_acc is computed in the HOO runner.
Updates hoo_summary.json with hoo_acc per fold.
"""

import json
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parents[2]))
from src.probes.bradley_terry.data import PairwiseActivationData
from src.probes.bradley_terry.training import pairwise_accuracy_from_scores
from src.probes.core.activations import load_activations
from src.probes.core.evaluate import score_with_probe
from src.probes.data_loading import load_pairwise_measurements
from src.probes.residualization import build_task_groups

load_dotenv()

HOO_SUMMARY = Path("results/probes/gemma2_10k_hoo_topic/hoo_summary.json")
PROBES_DIR = Path("results/probes/gemma2_10k_hoo_topic/probes")
ACTIVATIONS_PATH = Path("activations/gemma_2_27b_base/activations_prompt_last.npz")
TOPICS_JSON = Path("src/analysis/topic_classification/output/topics.json")
RUN_DIR = Path("/Users/oscargilg/Dev/MATS/Preferences/results/experiments/gemma3_10k_run1"
               "/pre_task_active_learning"
               "/completion_preference_gemma-3-27b_completion_canonical_seed0")
LAYER = 23

# Load activations
print("Loading activations...")
task_ids, activations_dict = load_activations(ACTIVATIONS_PATH, layers=[LAYER])
acts = activations_dict[LAYER]
print(f"Activations: {acts.shape}, tasks: {len(task_ids)}")

# Load real pairwise measurements
print("Loading measurements...")
measurements = load_pairwise_measurements(RUN_DIR)
print(f"Measurements: {len(measurements)} pairwise choices")

# Build full PairwiseActivationData (all tasks, for splitting by fold)
bt_data = PairwiseActivationData.from_measurements(measurements, task_ids, {LAYER: acts})
print(f"Unique pairs: {len(bt_data.pairs)}, total measurements: {bt_data.n_measurements}")

# Build topic groups for all tasks that have activations
all_task_ids = set(task_ids)
task_groups = build_task_groups(all_task_ids, grouping="topic", topics_json=TOPICS_JSON)
print(f"Tasks with topic labels: {len(task_groups)}")

# Load HOO summary
with open(HOO_SUMMARY) as f:
    summary = json.load(f)

for fold in summary["folds"]:
    fold_idx = fold["fold_idx"]
    held_out_set = set(fold["held_out_groups"])

    # Split pairwise data to held-out fold only
    _, eval_bt_data = bt_data.split_by_groups(task_ids, task_groups, held_out_set)

    if eval_bt_data.n_measurements == 0:
        print(f"Fold {fold_idx} ({held_out_set}): no pairwise data — skipping")
        continue

    # Load probe weights
    probe_path = PROBES_DIR / f"probe_hoo_fold{fold_idx}_ridge_L{LAYER}.npy"
    if not probe_path.exists():
        print(f"Fold {fold_idx}: probe not found at {probe_path} — skipping")
        continue
    w = np.load(probe_path)

    # Score all tasks, then compute pairwise accuracy on held-out pairs
    all_predicted = score_with_probe(w, acts)
    acc = pairwise_accuracy_from_scores(all_predicted, eval_bt_data)

    print(f"Fold {fold_idx} ({', '.join(sorted(held_out_set))}): "
          f"n_pairs={len(eval_bt_data.pairs)}, n_measurements={eval_bt_data.n_measurements}, "
          f"hoo_acc={acc:.4f}")

    fold["layers"][f"ridge_L{LAYER}"]["hoo_acc"] = acc

# Write updated summary
with open(HOO_SUMMARY, "w") as f:
    json.dump(summary, f, indent=2)
print("\nUpdated hoo_summary.json with real hoo_acc for all folds.")
