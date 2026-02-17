"""Check overlap between Gemma-2 base activations and preference scores."""
import csv
import numpy as np
from pathlib import Path

# Load activation task IDs
acts = np.load("activations/gemma_2_27b_base/activations_prompt_last.npz", allow_pickle=True)
act_ids = set(acts["task_ids"])
print(f"Activation task IDs: {len(act_ids)}")

# Load preference scores from CSV
run_dir = Path("results/experiments/gemma3_3k_run2/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0")
csv_file = run_dir / "thurstonian_a1ebd06e.csv"
score_ids = set()
with open(csv_file) as f:
    reader = csv.DictReader(f)
    for row in reader:
        score_ids.add(row["task_id"])
print(f"Preference score task IDs: {len(score_ids)}")

overlap = act_ids & score_ids
print(f"Overlap: {len(overlap)}")
print(f"Missing from activations: {len(score_ids - act_ids)}")

# Show sample of missing IDs
missing = score_ids - act_ids
if missing:
    for tid in sorted(missing)[:10]:
        print(f"  Missing: {tid}")
