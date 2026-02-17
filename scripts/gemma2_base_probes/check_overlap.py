import numpy as np
import csv

# Load Gemma-2 base task IDs
g2_data = np.load("activations/gemma_2_27b_base/activations_prompt_last.npz", allow_pickle=True)
g2_ids = set(g2_data["task_ids"])

# Load preference task IDs
with open("results/experiments/gemma3_3k_run2/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0/thurstonian_a1ebd06e.csv") as f:
    reader = csv.DictReader(f)
    pref_ids = set(row["task_id"] for row in reader)

overlap = pref_ids & g2_ids
missing = pref_ids - g2_ids

print(f"Preference tasks: {len(pref_ids)}")
print(f"G2 base tasks: {len(g2_ids)}")
print(f"Overlap: {len(overlap)}")
print(f"Missing from G2: {len(missing)}")

# Check origins of missing tasks
from collections import Counter
origins = Counter()
for tid in sorted(missing)[:20]:
    print(f"  {tid}")
    # Extract origin from task ID pattern
    parts = tid.split("_")
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

# Count all missing origins
for tid in missing:
    if "stress" in tid:
        origins["stress_test_all"] = origins.get("stress_test_all", 0) + 1

print(f"\nMissing origins (sample + full stress count): {dict(origins)}")
