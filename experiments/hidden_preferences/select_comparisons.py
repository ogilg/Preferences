"""Select comparison tasks for hidden preferences experiment.

Since hidden-preference target tasks are synthetic (no Thurstonian estimates),
select from a broad utility band (mu 0-5) to avoid extreme tasks.
All 16 targets share the same comparison pool.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

THURSTONIAN_PATH = Path(
    "results/experiments/gemma3_3k_run2/pre_task_active_learning/"
    "completion_preference_gemma-3-27b_completion_canonical_seed0/"
    "thurstonian_a1ebd06e.csv"
)
TARGETS_PATH = Path("experiments/hidden_preferences/target_tasks.json")

MU_MIN = 0.0
MU_MAX = 5.0
MAX_PER_ORIGIN = 10
N_COMPARISONS = 40

df = pd.read_csv(THURSTONIAN_PATH)
with open(TARGETS_PATH) as f:
    targets = json.load(f)

target_ids = {t["task_id"] for t in targets}

# Filter to broad utility band
candidates = df[
    (df["mu"] >= MU_MIN) &
    (df["mu"] <= MU_MAX) &
    (~df["task_id"].isin(target_ids))
].copy()

print(f"Candidates in mu [{MU_MIN}, {MU_MAX}]: {len(candidates)}")

# Extract origin prefix for diversity
candidates["origin_prefix"] = candidates["task_id"].apply(
    lambda x: x.rsplit("_", 1)[0] if "_" in x else x
)

rng = np.random.default_rng(42)

# Group by origin type (wildchat, alpaca, math, etc.)
candidates["origin_type"] = candidates["task_id"].apply(
    lambda x: x.split("_")[0] if "_" in x else "unknown"
)

# Shuffle then select with origin cap for diversity across the mu band
candidates = candidates.sample(frac=1, random_state=42).reset_index(drop=True)

selected = []
origin_counts: dict[str, int] = {}

for _, row in candidates.iterrows():
    otype = row["origin_type"]
    if origin_counts.get(otype, 0) >= MAX_PER_ORIGIN:
        continue
    selected.append(row["task_id"])
    origin_counts[otype] = origin_counts.get(otype, 0) + 1
    if len(selected) >= N_COMPARISONS:
        break

print(f"Selected {len(selected)} comparison tasks")
print(f"Origin distribution: {origin_counts}")
print(f"Mu range: [{df[df['task_id'].isin(selected)]['mu'].min():.2f}, {df[df['task_id'].isin(selected)]['mu'].max():.2f}]")

# All targets share the same pool
comparisons = {t["task_id"]: selected for t in targets}

output = Path("experiments/hidden_preferences/comparison_tasks.json")
with open(output, "w") as f:
    json.dump(comparisons, f, indent=2)
print(f"\nSaved to {output}")
