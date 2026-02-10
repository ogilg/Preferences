"""Control analysis: are probe shifts specific to targeted tasks?

For each system prompt, compare probe delta on the targeted task vs non-targeted tasks.
Break down by positive and negative prompts separately.
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats

EXP_DIR = Path("experiments/ood_generalization")
ACT_DIR = EXP_DIR / "activations"
PROBE_DIR = Path("results/probes/gemma3_3k_completion_preference/probes")
LAYERS = [31]


def load_probe(layer: int) -> tuple[np.ndarray, float]:
    w = np.load(PROBE_DIR / f"probe_ridge_L{layer}.npy")
    return w[:-1], w[-1]


def main():
    with open(EXP_DIR / "target_tasks.json") as f:
        targets = json.load(f)
    with open(EXP_DIR / "system_prompts.json") as f:
        iter_prompts = json.load(f)["prompts"]
    with open(EXP_DIR / "holdout_prompts.json") as f:
        holdout_prompts = json.load(f)["prompts"]

    all_prompts = iter_prompts + holdout_prompts
    target_id_to_idx = {t["task_id"]: i for i, t in enumerate(targets)}
    category_to_target = {t["topic"]: t["task_id"] for t in targets}

    coef, intercept = load_probe(31)
    baseline_data = np.load(ACT_DIR / "baseline.npz", allow_pickle=True)
    baseline_scores = baseline_data["layer_31"] @ coef + intercept

    positive_prompts = [p for p in all_prompts if p["direction"] == "positive"]
    negative_prompts = [p for p in all_prompts if p["direction"] == "negative"]

    for direction, prompts in [("POSITIVE", positive_prompts), ("NEGATIVE", negative_prompts)]:
        print(f"\n{'='*80}")
        print(f"{direction} PROMPTS (n={len(prompts)}): Ridge L31 probe deltas")
        print(f"{'='*80}")

        targeted_deltas = []
        non_targeted_deltas = []

        # Per-prompt breakdown
        print(f"\n{'Prompt ID':<40} {'Target cat':<15} {'Targeted Δ':>12} {'Non-targeted mean Δ':>20} {'Non-targeted std':>18}")
        print("-" * 110)

        for sp in prompts:
            npz_path = ACT_DIR / f"{sp['id']}.npz"
            if not npz_path.exists():
                continue
            manip_data = np.load(npz_path, allow_pickle=True)
            manip_scores = manip_data["layer_31"] @ coef + intercept

            target_task_id = category_to_target[sp["target_category"]]
            target_idx = target_id_to_idx[target_task_id]

            this_targeted = None
            this_non_targeted = []

            for task_info in targets:
                idx = target_id_to_idx[task_info["task_id"]]
                delta = float(manip_scores[idx] - baseline_scores[idx])
                if idx == target_idx:
                    this_targeted = delta
                    targeted_deltas.append(delta)
                else:
                    this_non_targeted.append(delta)
                    non_targeted_deltas.append(delta)

            nt_arr = np.array(this_non_targeted)
            print(f"{sp['id']:<40} {sp['target_category']:<15} {this_targeted:>+12.1f} {nt_arr.mean():>+20.1f} {nt_arr.std():>18.1f}")

        targeted_arr = np.array(targeted_deltas)
        non_targeted_arr = np.array(non_targeted_deltas)

        print(f"\n--- Summary ({direction}) ---")
        print(f"  Targeted tasks     (n={len(targeted_arr):>3}): mean={targeted_arr.mean():>+8.1f}, std={targeted_arr.std():>7.1f}")
        print(f"  Non-targeted tasks (n={len(non_targeted_arr):>3}): mean={non_targeted_arr.mean():>+8.1f}, std={non_targeted_arr.std():>7.1f}")

        t_signed, p_signed = stats.ttest_1samp(non_targeted_arr, 0)
        print(f"  Non-targeted signed mean ≠ 0: t={t_signed:.2f}, p={p_signed:.3f}")

        t_diff, p_diff = stats.ttest_ind(np.abs(targeted_arr), np.abs(non_targeted_arr))
        print(f"  abs(targeted) vs abs(non-targeted): t={t_diff:.2f}, p={p_diff:.1e}")


if __name__ == "__main__":
    main()
