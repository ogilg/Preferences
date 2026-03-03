"""Inspect probe scores for a specific task across conditions."""

import numpy as np
from pathlib import Path

PROBE_PATH = Path("results/probes/gemma3_10k_heldout_std_demean/probes/probe_ridge_L31.npy")
ACT_DIR = Path("activations/ood/exp1_prompts/exp1_prompts")
TASK_ID = "hidden_rainy_weather_4"
LAYER = 31

CONDITIONS = ["baseline", "rainy_weather_neg_persona", "rainy_weather_pos_persona"]

probe = np.load(PROBE_PATH)
weights, bias = probe[:-1], float(probe[-1])

for cond in CONDITIONS:
    npz_path = ACT_DIR / cond / "activations_prompt_last.npz"
    data = np.load(npz_path, allow_pickle=True)
    task_ids = list(data["task_ids"])
    acts = data[f"layer_{LAYER}"]

    idx = task_ids.index(TASK_ID)
    score = float(acts[idx] @ weights + bias)

    print(f"\n=== {cond} ===")
    print(f"  Probe score (L{LAYER}): {score:.4f}")

    # Also show all rainy_weather tasks
    print(f"\n  All rainy_weather tasks:")
    for i, tid in enumerate(task_ids):
        if "rainy_weather" in tid:
            s = float(acts[i] @ weights + bias)
            print(f"    {tid}: {s:.4f}")

# Show deltas
print("\n=== Deltas from baseline ===")
baseline_data = np.load(ACT_DIR / "baseline" / "activations_prompt_last.npz", allow_pickle=True)
baseline_ids = list(baseline_data["task_ids"])
baseline_acts = baseline_data[f"layer_{LAYER}"]

for cond in ["rainy_weather_neg_persona", "rainy_weather_pos_persona"]:
    cond_data = np.load(ACT_DIR / cond / "activations_prompt_last.npz", allow_pickle=True)
    cond_ids = list(cond_data["task_ids"])
    cond_acts = cond_data[f"layer_{LAYER}"]

    print(f"\n  {cond}:")
    for tid in sorted(set(baseline_ids) & set(cond_ids)):
        if "rainy_weather" not in tid:
            continue
        bi = baseline_ids.index(tid)
        ci = cond_ids.index(tid)
        b_score = float(baseline_acts[bi] @ weights + bias)
        c_score = float(cond_acts[ci] @ weights + bias)
        delta = c_score - b_score
        print(f"    {tid}: baseline={b_score:.4f}  cond={c_score:.4f}  delta={delta:+.4f}")
