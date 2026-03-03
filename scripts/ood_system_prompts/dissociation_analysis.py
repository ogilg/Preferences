"""Find cases where behavioral delta goes opposite to expected but probe goes right.

For neg conditions: expected behavioral delta is negative (model avoids on-target tasks).
A "dissociation" = behavioral delta is positive (model picks the task MORE) while
probe delta is negative (probe correctly says "this is devalued").
"""

import json
from pathlib import Path

import numpy as np

PROBE_PATH = Path("results/probes/gemma3_10k_heldout_std_demean/probes/probe_ridge_L31.npy")
ACT_DIR = Path("activations/ood/exp1_prompts/exp1_prompts")
PAIRWISE_PATH = Path("results/ood/hidden_preference/pairwise.json")
PROMPTS_PATH = Path("configs/ood/prompts/targeted_preference.json")
LAYER = 31

# Load probe
probe = np.load(PROBE_PATH)
weights, bias = probe[:-1], float(probe[-1])

# Load condition metadata (direction info)
with open(PROMPTS_PATH) as f:
    prompts_data = json.load(f)

condition_meta = {}
for c in prompts_data["conditions"]:
    condition_meta[c["condition_id"]] = {
        "direction": c["direction"],
        "category": c["category"],
        "system_prompt": c["system_prompt"],
    }

# Load pairwise results
with open(PAIRWISE_PATH) as f:
    pairwise_data = json.load(f)

# Compute p_choose per condition per task
from collections import defaultdict

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

def p_choose(cid, tid):
    w = wins[cid].get(tid, 0)
    t = total[cid].get(tid, 0)
    return w / t if t > 0 else float("nan")

# Load task prompts
with open(ACT_DIR / "baseline" / "completions_with_activations.json") as f:
    tasks = {t["task_id"]: t["task_prompt"] for t in json.load(f)}

# Score activations
def score_condition(cond_name):
    npz = ACT_DIR / cond_name / "activations_prompt_last.npz"
    if not npz.exists():
        return {}
    data = np.load(npz, allow_pickle=True)
    task_ids = list(data["task_ids"])
    acts = data[f"layer_{LAYER}"]
    return {tid: float(acts[i] @ weights + bias) for i, tid in enumerate(task_ids)}

baseline_scores = score_condition("baseline")

# Only look at persona conditions for exp1b hidden topics
persona_conditions = [c for c in condition_meta if c.endswith("_persona")]

print("=" * 90)
print("DISSOCIATION ANALYSIS: behavioral delta vs probe delta on on-target tasks")
print("=" * 90)

dissociations = []

for cid in sorted(persona_conditions):
    meta = condition_meta[cid]
    direction = meta["direction"]
    category = meta["category"]

    cond_scores = score_condition(cid)
    if not cond_scores:
        continue

    # On-target tasks: those containing the category name
    on_target_tasks = [tid for tid in cond_scores if category in tid]

    for tid in sorted(on_target_tasks):
        beh_baseline = p_choose("baseline", tid)
        beh_cond = p_choose(cid, tid)
        beh_delta = beh_cond - beh_baseline

        probe_baseline = baseline_scores.get(tid, float("nan"))
        probe_cond = cond_scores.get(tid, float("nan"))
        probe_delta = probe_cond - probe_baseline

        # For neg conditions: expected beh_delta < 0, expected probe_delta < 0
        # For pos conditions: expected beh_delta > 0, expected probe_delta > 0
        if direction == "neg":
            beh_wrong = beh_delta > 0.05  # behavioral goes UP (wrong for neg)
            probe_right = probe_delta < 0  # probe goes DOWN (correct for neg)
        else:
            beh_wrong = beh_delta < -0.05  # behavioral goes DOWN (wrong for pos)
            probe_right = probe_delta > 0  # probe goes UP (correct for pos)

        is_dissociation = beh_wrong and probe_right

        if is_dissociation:
            dissociations.append({
                "condition": cid,
                "direction": direction,
                "task_id": tid,
                "task_prompt": tasks.get(tid, "?"),
                "beh_delta": beh_delta,
                "probe_delta": probe_delta,
            })

print(f"\nFound {len(dissociations)} dissociations (behavioral wrong direction, probe right)")
print(f"(threshold: |beh_delta| > 0.05 in wrong direction)\n")

for d in sorted(dissociations, key=lambda x: abs(x["beh_delta"]), reverse=True):
    print(f"  {d['condition']} | {d['task_id']}")
    print(f"    beh_delta: {d['beh_delta']:+.3f} (WRONG for {d['direction']})")
    print(f"    probe_delta: {d['probe_delta']:+.3f} (correct for {d['direction']})")
    print(f"    task: {d['task_prompt'][:100]}")
    print()

# Also count: for all on-target neg tasks, how often does behavior go wrong vs probe go wrong?
print("=" * 90)
print("SUMMARY: On-target tasks by condition type")
print("=" * 90)

for direction in ["neg", "pos"]:
    conds = [c for c in persona_conditions if condition_meta[c]["direction"] == direction]
    n_total = 0
    n_beh_wrong = 0
    n_probe_wrong = 0
    n_both_wrong = 0
    n_dissociation = 0  # beh wrong, probe right

    for cid in conds:
        category = condition_meta[cid]["category"]
        cond_scores = score_condition(cid)
        if not cond_scores:
            continue
        on_target = [tid for tid in cond_scores if category in tid]

        for tid in on_target:
            beh_delta = p_choose(cid, tid) - p_choose("baseline", tid)
            probe_delta = cond_scores.get(tid, 0) - baseline_scores.get(tid, 0)

            n_total += 1
            if direction == "neg":
                bw = beh_delta > 0.05
                pw = probe_delta > 0
            else:
                bw = beh_delta < -0.05
                pw = probe_delta < 0

            if bw:
                n_beh_wrong += 1
            if pw:
                n_probe_wrong += 1
            if bw and pw:
                n_both_wrong += 1
            if bw and not pw:
                n_dissociation += 1

    print(f"\n  {direction.upper()} conditions ({len(conds)} conditions, {n_total} on-target task-condition pairs):")
    print(f"    Behavioral wrong direction: {n_beh_wrong}/{n_total} ({100*n_beh_wrong/n_total:.1f}%)")
    print(f"    Probe wrong direction:      {n_probe_wrong}/{n_total} ({100*n_probe_wrong/n_total:.1f}%)")
    print(f"    Both wrong:                 {n_both_wrong}/{n_total} ({100*n_both_wrong/n_total:.1f}%)")
    print(f"    DISSOCIATION (beh wrong, probe right): {n_dissociation}/{n_total} ({100*n_dissociation/n_total:.1f}%)")
