"""Evaluate holdout set: correlate behavioral deltas with probe deltas."""

import json
from pathlib import Path

import numpy as np
from scipy import stats

EXP_DIR = Path("experiments/ood_generalization")
ACT_DIR = EXP_DIR / "activations"
PROBE_DIR = Path("results/probes/gemma3_3k_completion_preference/probes")
LAYERS = [31, 43, 55]


def load_probe(layer: int) -> tuple[np.ndarray, float]:
    weights = np.load(PROBE_DIR / f"probe_ridge_L{layer}.npy")
    return weights[:-1], weights[-1]


def main():
    with open(EXP_DIR / "target_tasks.json") as f:
        targets = json.load(f)
    with open(EXP_DIR / "holdout_prompts.json") as f:
        holdout_prompts = json.load(f)["prompts"]
    with open(EXP_DIR / "results" / "holdout_behavioral.json") as f:
        behavioral = json.load(f)

    behavioral_by_id = {b["prompt_id"]: b for b in behavioral}
    target_id_to_idx = {t["task_id"]: i for i, t in enumerate(targets)}
    category_to_target = {t["topic"]: t["task_id"] for t in targets}

    probes = {}
    for layer in LAYERS:
        coef, intercept = load_probe(layer)
        probes[layer] = (coef, intercept)

    # Load baseline
    baseline_data = np.load(ACT_DIR / "baseline.npz", allow_pickle=True)
    baseline_scores = {}
    for layer in LAYERS:
        coef, intercept = probes[layer]
        baseline_scores[layer] = baseline_data[f"layer_{layer}"] @ coef + intercept

    results = []
    for sp in holdout_prompts:
        beh = behavioral_by_id[sp["id"]]
        target_task_id = category_to_target[sp["target_category"]]
        target_idx = target_id_to_idx[target_task_id]

        manip_data = np.load(ACT_DIR / f"{sp['id']}.npz", allow_pickle=True)

        probe_deltas = {}
        for layer in LAYERS:
            coef, intercept = probes[layer]
            manip_score = manip_data[f"layer_{layer}"] @ coef + intercept
            probe_deltas[layer] = float(manip_score[target_idx] - baseline_scores[layer][target_idx])

        results.append({
            "prompt_id": sp["id"],
            "target_category": sp["target_category"],
            "direction": sp["direction"],
            "prompt_type": sp["type"],
            "behavioral_delta": beh["delta"],
            "probe_delta_L31": probe_deltas[31],
            "probe_delta_L43": probe_deltas[43],
            "probe_delta_L55": probe_deltas[55],
        })

    print(f"{'Prompt ID':<40} {'Dir':<5} {'Beh Δ':>8} {'Probe L31':>10} {'Probe L43':>10} {'Probe L55':>10}")
    print("-" * 80)
    for r in results:
        print(f"{r['prompt_id']:<40} {r['direction']:<5} {r['behavioral_delta']:>+8.3f} "
              f"{r['probe_delta_L31']:>+10.3f} {r['probe_delta_L43']:>+10.3f} {r['probe_delta_L55']:>+10.3f}")

    beh_deltas = np.array([r["behavioral_delta"] for r in results])

    print("\n=== HOLDOUT Correlations ===")
    for layer in LAYERS:
        key = f"probe_delta_L{layer}"
        probe_d = np.array([r[key] for r in results])
        pearson_r, pearson_p = stats.pearsonr(beh_deltas, probe_d)
        spearman_r, spearman_p = stats.spearmanr(beh_deltas, probe_d)
        sign_agree = np.mean(np.sign(beh_deltas) == np.sign(probe_d))
        print(f"\n  Layer {layer}:")
        print(f"    Pearson r={pearson_r:.3f} (p={pearson_p:.1e})")
        print(f"    Spearman r={spearman_r:.3f} (p={spearman_p:.1e})")
        print(f"    Sign agreement: {sign_agree:.1%} ({int(sign_agree * len(results))}/{len(results)})")

    # Strong manipulations only
    strong_mask = np.abs(beh_deltas) > 0.05
    print(f"\n=== HOLDOUT Strong manipulations (|delta|>0.05, n={strong_mask.sum()}) ===")
    for layer in LAYERS:
        key = f"probe_delta_L{layer}"
        probe_d = np.array([r[key] for r in results])
        if strong_mask.sum() > 2:
            pearson_r, pearson_p = stats.pearsonr(beh_deltas[strong_mask], probe_d[strong_mask])
            spearman_r, spearman_p = stats.spearmanr(beh_deltas[strong_mask], probe_d[strong_mask])
            sign_agree = np.mean(np.sign(beh_deltas[strong_mask]) == np.sign(probe_d[strong_mask]))
            print(f"\n  Layer {layer}:")
            print(f"    Pearson r={pearson_r:.3f} (p={pearson_p:.1e})")
            print(f"    Spearman r={spearman_r:.3f} (p={spearman_p:.1e})")
            print(f"    Sign agreement: {sign_agree:.1%}")

    # Save
    output_path = EXP_DIR / "results" / "holdout_probe_behavioral.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
