import numpy as np
import csv
from collections import Counter

g2_data = np.load("activations/gemma_2_27b_base/activations_prompt_last.npz", allow_pickle=True)
g2_ids = set(g2_data["task_ids"])

with open("results/experiments/gemma3_3k_run2/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0/thurstonian_a1ebd06e.csv") as f:
    reader = csv.DictReader(f)
    pref_ids = set(row["task_id"] for row in reader)

missing = pref_ids - g2_ids
origins = Counter()
for tid in missing:
    if "stress" in tid:
        origins["stress_test"] += 1
    elif "wildchat" in tid:
        origins["wildchat"] += 1
    elif "alpaca" in tid:
        origins["alpaca"] += 1
    elif "math" in tid or "competition" in tid:
        origins["math"] += 1
    elif "bail" in tid:
        origins["bailbench"] += 1
    else:
        origins["unknown"] += 1

print(f"Missing by origin: {dict(origins)}")
print(f"Total missing: {sum(origins.values())}")
