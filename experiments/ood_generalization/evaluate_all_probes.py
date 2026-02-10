"""Evaluate OOD generalization with both ridge and BT probes.
Also computes signed off-target control statistics.
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats

EXP_DIR = Path("experiments/ood_generalization")
ACT_DIR = EXP_DIR / "activations"
PROBE_DIR = Path("results/probes/gemma3_3k_completion_preference/probes")
LAYERS = [31, 43, 55]
PROBE_TYPES = ["ridge", "bt"]


def load_probe(probe_type: str, layer: int) -> tuple[np.ndarray, float]:
    w = np.load(PROBE_DIR / f"probe_{probe_type}_L{layer}.npy")
    return w[:-1], w[-1]


def evaluate_probe(probe_type: str, layer: int, results_data: list[dict],
                   targets: list[dict], category_to_target: dict,
                   target_id_to_idx: dict) -> list[dict]:
    coef, intercept = load_probe(probe_type, layer)

    baseline_data = np.load(ACT_DIR / "baseline.npz", allow_pickle=True)
    baseline_scores = baseline_data[f"layer_{layer}"] @ coef + intercept

    out = []
    for r in results_data:
        target_task_id = category_to_target[r["target_category"]]
        target_idx = target_id_to_idx[target_task_id]
        manip_data = np.load(ACT_DIR / f"{r['prompt_id']}.npz", allow_pickle=True)
        manip_scores = manip_data[f"layer_{layer}"] @ coef + intercept
        probe_delta = float(manip_scores[target_idx] - baseline_scores[target_idx])
        out.append({
            **r,
            "probe_delta": probe_delta,
        })
    return out


def compute_off_target_control(probe_type: str, layer: int, all_prompts: list[dict],
                                targets: list[dict], category_to_target: dict,
                                target_id_to_idx: dict):
    coef, intercept = load_probe(probe_type, layer)
    baseline_data = np.load(ACT_DIR / "baseline.npz", allow_pickle=True)
    baseline_scores = baseline_data[f"layer_{layer}"] @ coef + intercept

    on_target = []
    off_target_by_direction = {"positive": [], "negative": []}

    for sp_info in all_prompts:
        npz_path = ACT_DIR / f"{sp_info['id']}.npz"
        if not npz_path.exists():
            continue
        manip_data = np.load(npz_path, allow_pickle=True)
        manip_scores = manip_data[f"layer_{layer}"] @ coef + intercept

        target_task_id = category_to_target[sp_info["target_category"]]
        target_idx = target_id_to_idx[target_task_id]

        for task_info in targets:
            idx = target_id_to_idx[task_info["task_id"]]
            delta = float(manip_scores[idx] - baseline_scores[idx])
            if idx == target_idx:
                on_target.append(delta)
            else:
                off_target_by_direction[sp_info["direction"]].append(delta)

    return np.array(on_target), off_target_by_direction


def main():
    with open(EXP_DIR / "target_tasks.json") as f:
        targets = json.load(f)
    target_id_to_idx = {t["task_id"]: i for i, t in enumerate(targets)}
    category_to_target = {t["topic"]: t["task_id"] for t in targets}

    # Load all behavioral results
    with open(EXP_DIR / "results" / "probe_behavioral_comparison.json") as f:
        iteration_beh = json.load(f)
    with open(EXP_DIR / "results" / "holdout_probe_behavioral.json") as f:
        holdout_beh = json.load(f)

    # Build unified behavioral list (prompt_id, target_category, direction, behavioral_delta)
    combined_beh = []
    for r in iteration_beh + holdout_beh:
        combined_beh.append({
            "prompt_id": r["prompt_id"],
            "target_category": r["target_category"],
            "direction": r["direction"],
            "behavioral_delta": r["behavioral_delta"],
        })

    # Load all prompt definitions
    with open(EXP_DIR / "system_prompts.json") as f:
        iter_prompts = json.load(f)["prompts"]
    with open(EXP_DIR / "holdout_prompts.json") as f:
        holdout_prompts_list = json.load(f)["prompts"]
    all_prompts = iter_prompts + holdout_prompts_list

    print("=" * 90)
    print("PROBE COMPARISON: Ridge vs BT across layers")
    print("=" * 90)

    beh_deltas = np.array([r["behavioral_delta"] for r in combined_beh])

    for probe_type in PROBE_TYPES:
        print(f"\n--- {probe_type.upper()} probes ---")
        print(f"{'Layer':<8} {'Pearson r':<14} {'p-value':<12} {'Spearman r':<14} {'p-value':<12} {'Sign %':<10}")
        print("-" * 70)

        for layer in LAYERS:
            scored = evaluate_probe(probe_type, layer, combined_beh, targets,
                                    category_to_target, target_id_to_idx)
            probe_d = np.array([r["probe_delta"] for r in scored])
            pr, pp = stats.pearsonr(beh_deltas, probe_d)
            sr, sp_val = stats.spearmanr(beh_deltas, probe_d)
            sign = np.mean(np.sign(beh_deltas) == np.sign(probe_d))
            print(f"L{layer:<7} r={pr:+.3f}       p={pp:.1e}     r={sr:+.3f}       p={sp_val:.1e}     {sign:.1%}")

    # Off-target control with signed statistics
    print("\n" + "=" * 90)
    print("OFF-TARGET CONTROL (signed deltas)")
    print("=" * 90)

    for probe_type in PROBE_TYPES:
        for layer in LAYERS:
            on_arr, off_by_dir = compute_off_target_control(
                probe_type, layer, all_prompts, targets, category_to_target, target_id_to_idx
            )
            off_pos = np.array(off_by_dir["positive"])
            off_neg = np.array(off_by_dir["negative"])
            off_all = np.concatenate([off_pos, off_neg])

            print(f"\n  {probe_type.upper()} L{layer}:")
            print(f"    On-target  (n={len(on_arr):>3}): signed mean={on_arr.mean():>+8.1f}, std={on_arr.std():>7.1f}, |mean|={np.abs(on_arr).mean():>7.1f}")
            print(f"    Off-target (n={len(off_all):>3}): signed mean={off_all.mean():>+8.1f}, std={off_all.std():>7.1f}, |mean|={np.abs(off_all).mean():>7.1f}")
            print(f"      Positive prompts off-target (n={len(off_pos):>3}): signed mean={off_pos.mean():>+8.1f}")
            print(f"      Negative prompts off-target (n={len(off_neg):>3}): signed mean={off_neg.mean():>+8.1f}")

            # Two-sample t-test on absolute values
            t_abs, p_abs = stats.ttest_ind(np.abs(on_arr), np.abs(off_all))
            # One-sample t-test: is off-target signed mean different from 0?
            t_signed, p_signed = stats.ttest_1samp(off_all, 0)
            print(f"    |on| vs |off| t-test: t={t_abs:.2f}, p={p_abs:.1e}")
            print(f"    Off-target signed mean ≠ 0: t={t_signed:.2f}, p={p_signed:.1e}")


if __name__ == "__main__":
    main()
