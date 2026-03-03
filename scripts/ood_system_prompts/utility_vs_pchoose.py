"""Compare per-task Thurstonian utility deltas vs raw p_choose deltas.

Find tasks where utility says one thing but p_choose says the opposite.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.ood_system_prompts.analyze_utility_shifts import (
    load_experiment_utilities,
    compute_deltas,
    assign_ground_truth,
)

PAIRWISE_PATH = Path("results/ood/hidden_preference/pairwise.json")
PROMPTS_PATH = Path("configs/ood/prompts/targeted_preference.json")

# Load Thurstonian utilities for exp1b
df, _ = load_experiment_utilities("exp1b")
thu_deltas = compute_deltas(df)
gt_df = assign_ground_truth(thu_deltas, "exp1b")

# Load p_choose from pairwise
with open(PAIRWISE_PATH) as f:
    pairwise_data = json.load(f)

wins = defaultdict(lambda: defaultdict(int))
total = defaultdict(lambda: defaultdict(int))
for entry in pairwise_data["results"]:
    cid = entry["condition_id"]
    ta, tb = entry["task_a"], entry["task_b"]
    non_refusal = entry["n_a"] + entry["n_b"]
    wins[cid][ta] += entry["n_a"]
    wins[cid][tb] += entry["n_b"]
    total[cid][ta] += non_refusal
    total[cid][tb] += non_refusal

# Load condition metadata
with open(PROMPTS_PATH) as f:
    prompts_data = json.load(f)
condition_meta = {c["condition_id"]: c for c in prompts_data["conditions"]}

# Only persona conditions
persona_conds = [c for c in thu_deltas.columns if c.endswith("_persona")]

print("=" * 100)
print("Tasks where Thurstonian Δu and p_choose Δ disagree in sign (on-target tasks only)")
print("=" * 100)

disagreements = []

for cond in sorted(persona_conds):
    if cond not in condition_meta:
        continue
    meta = condition_meta[cond]
    direction = meta["direction"]
    category = meta["category"]

    for tid in thu_deltas.index:
        # Only on-target tasks
        if category not in tid:
            continue

        thu_du = thu_deltas.loc[tid, cond]

        # Get p_choose delta
        w_b = wins["baseline"].get(tid, 0)
        t_b = total["baseline"].get(tid, 0)
        w_c = wins[cond].get(tid, 0)
        t_c = total[cond].get(tid, 0)

        if t_b == 0 or t_c == 0:
            continue

        pc_baseline = w_b / t_b
        pc_cond = w_c / t_c
        pc_delta = pc_cond - pc_baseline

        # Expected direction
        if direction == "neg":
            thu_wrong = thu_du > 0  # utility went UP (wrong for neg)
            pc_wrong = pc_delta > 0.05
        else:
            thu_wrong = thu_du < 0  # utility went DOWN (wrong for pos)
            pc_wrong = pc_delta < -0.05

        if thu_wrong and not pc_wrong:
            disagreements.append({
                "condition": cond,
                "direction": direction,
                "task_id": tid,
                "thu_delta": thu_du,
                "pc_baseline": pc_baseline,
                "pc_cond": pc_cond,
                "pc_delta": pc_delta,
            })

print(f"\nFound {len(disagreements)} cases: Thurstonian wrong direction, p_choose correct")
print()

for d in sorted(disagreements, key=lambda x: abs(x["thu_delta"]), reverse=True):
    print(f"  {d['condition']} | {d['task_id']}")
    print(f"    Thurstonian Δu: {d['thu_delta']:+.2f} (WRONG for {d['direction']})")
    print(f"    p_choose: {d['pc_baseline']:.3f} -> {d['pc_cond']:.3f} (Δ={d['pc_delta']:+.3f}, correct)")
    print()

# Also show ALL on-target tasks for rainy_weather_neg to understand the pattern
print("=" * 100)
print("ALL on-target tasks for rainy_weather_neg_persona")
print("=" * 100)

cond = "rainy_weather_neg_persona"
for tid in sorted(thu_deltas.index):
    if "rainy_weather" not in tid:
        continue

    thu_du = thu_deltas.loc[tid, cond]

    w_b = wins["baseline"].get(tid, 0)
    t_b = total["baseline"].get(tid, 0)
    w_c = wins[cond].get(tid, 0)
    t_c = total[cond].get(tid, 0)

    if t_b > 0 and t_c > 0:
        pc_b = w_b / t_b
        pc_c = w_c / t_c
        pc_d = pc_c - pc_b
        print(f"  {tid}: thu_Δu={thu_du:+.2f}, p_choose: {pc_b:.3f}->{pc_c:.3f} (Δ={pc_d:+.3f})")
    else:
        print(f"  {tid}: thu_Δu={thu_du:+.2f}, p_choose: NO DATA")
