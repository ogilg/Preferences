"""Detailed view of behavioral and probe deltas for all on-target tasks, all neg conditions."""

import json
from pathlib import Path
from collections import defaultdict

import numpy as np

PROBE_PATH = Path("results/probes/gemma3_10k_heldout_std_demean/probes/probe_ridge_L31.npy")
ACT_DIR = Path("activations/ood/exp1_prompts/exp1_prompts")
PAIRWISE_PATH = Path("results/ood/hidden_preference/pairwise.json")
PROMPTS_PATH = Path("configs/ood/prompts/targeted_preference.json")
LAYER = 31

probe = np.load(PROBE_PATH)
weights, bias = probe[:-1], float(probe[-1])

with open(PROMPTS_PATH) as f:
    prompts_data = json.load(f)

condition_meta = {}
for c in prompts_data["conditions"]:
    condition_meta[c["condition_id"]] = c

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

def p_choose(cid, tid):
    w = wins[cid].get(tid, 0)
    t = total[cid].get(tid, 0)
    return w / t if t > 0 else float("nan")

with open(ACT_DIR / "baseline" / "completions_with_activations.json") as f:
    tasks = {t["task_id"]: t["task_prompt"] for t in json.load(f)}

def score_condition(cond_name):
    npz = ACT_DIR / cond_name / "activations_prompt_last.npz"
    if not npz.exists():
        return {}
    data = np.load(npz, allow_pickle=True)
    task_ids = list(data["task_ids"])
    acts = data[f"layer_{LAYER}"]
    return {tid: float(acts[i] @ weights + bias) for i, tid in enumerate(task_ids)}

baseline_scores = score_condition("baseline")

# Print all on-target tasks for ALL neg persona conditions
neg_conditions = sorted([c for c in condition_meta if c.endswith("_persona") and condition_meta[c]["direction"] == "neg"])

for cid in neg_conditions:
    category = condition_meta[cid]["category"]
    cond_scores = score_condition(cid)
    if not cond_scores:
        continue

    on_target = sorted([tid for tid in cond_scores if category in tid])

    print(f"\n{'='*80}")
    print(f"{cid} (expect: behavioral DOWN, probe DOWN)")
    print(f"{'='*80}")

    for tid in on_target:
        beh_b = p_choose("baseline", tid)
        beh_c = p_choose(cid, tid)
        beh_d = beh_c - beh_b
        pr_b = baseline_scores.get(tid, float("nan"))
        pr_c = cond_scores.get(tid, float("nan"))
        pr_d = pr_c - pr_b

        beh_ok = "OK" if beh_d <= 0 else "WRONG"
        pr_ok = "OK" if pr_d <= 0 else "WRONG"

        prompt_short = tasks.get(tid, "?")[:80]
        print(f"  {tid}")
        print(f"    beh: {beh_b:.3f} -> {beh_c:.3f} (delta {beh_d:+.3f}) [{beh_ok}]")
        print(f"    probe: {pr_b:.2f} -> {pr_c:.2f} (delta {pr_d:+.2f}) [{pr_ok}]")
        print(f"    {prompt_short}")
        print()
