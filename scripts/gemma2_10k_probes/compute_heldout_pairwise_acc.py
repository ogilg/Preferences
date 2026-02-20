"""Compute real pairwise accuracy for Gemma-2 heldout eval at L23.

Uses actual pairwise choices from the 4k eval set measurements.yaml.
The heldout eval uses the second half of the eval set (same split as probe training).
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

load_dotenv()

PROBE_PATH = Path("results/probes/gemma2_10k_heldout_std_raw/probes/probe_ridge_L23.npy")
ACTIVATIONS_PATH = Path("activations/gemma_2_27b_base/activations_prompt_last.npz")
EVAL_RUN_DIR = Path(
    "/Users/oscargilg/Dev/MATS/Preferences/results/experiments/gemma3_4k_pre_task"
    "/pre_task_active_learning"
    "/completion_preference_gemma-3-27b_completion_canonical_seed0"
)
THURSTONIAN_CSV = Path(
    "/Users/oscargilg/Dev/MATS/Preferences/results/experiments/gemma3_4k_pre_task"
    "/pre_task_active_learning"
    "/completion_preference_gemma-3-27b_completion_canonical_seed0"
    "/thurstonian_a67822c5.csv"
)
LAYER = 23
EVAL_SPLIT_SEED = 42  # must match probe training config

# Load probe
w = np.load(PROBE_PATH)
print(f"Probe shape: {w.shape}")

# Load activations
print("Loading activations...")
task_ids, activations_dict = load_activations(ACTIVATIONS_PATH, layers=[LAYER])
acts = activations_dict[LAYER]
print(f"Activations: {acts.shape}")

# Load measurements
print("Loading measurements...")
measurements = load_pairwise_measurements(EVAL_RUN_DIR)
print(f"Measurements: {len(measurements)} pairwise choices")

# Reproduce the eval split: second half by task ID order (matches run_dir_probes.py)
import pandas as pd
df = pd.read_csv(THURSTONIAN_CSV)
eval_task_ids = sorted(df["task_id"].tolist())
rng = np.random.default_rng(EVAL_SPLIT_SEED)
perm = rng.permutation(len(eval_task_ids))
half = len(eval_task_ids) // 2
final_ids = {eval_task_ids[i] for i in perm[half:]}
print(f"Final eval split: {len(final_ids)} tasks")

# Filter measurements to final eval tasks only
final_measurements = [
    m for m in measurements
    if m.task_a.id in final_ids and m.task_b.id in final_ids
]
print(f"Measurements in final eval split: {len(final_measurements)}")

# Build PairwiseActivationData on the full activation array (pairs index into it)
bt_data = PairwiseActivationData.from_measurements(final_measurements, task_ids, {LAYER: acts})
print(f"Unique pairs: {len(bt_data.pairs)}, total measurements: {bt_data.n_measurements}")

# Score all tasks, compute pairwise accuracy
all_predicted = score_with_probe(w, acts)
acc = pairwise_accuracy_from_scores(all_predicted, bt_data)
print(f"\nHeldout pairwise accuracy (L{LAYER}, real choices): {acc:.4f}")
