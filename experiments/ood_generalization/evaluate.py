"""Evaluate OOD generalization: correlate behavioral deltas with probe deltas.

For each system prompt manipulation:
1. Load baseline and manipulated activations for the 6 target tasks
2. Score with ridge probes at layers 31, 43, 55
3. Compute probe delta for the targeted task
4. Correlate probe delta with behavioral delta across all 20 manipulations
"""

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


def score_activations(activations: np.ndarray, coef: np.ndarray, intercept: float) -> np.ndarray:
    return activations @ coef + intercept


def load_activations(filename: str) -> dict[int, np.ndarray]:
    data = np.load(ACT_DIR / filename, allow_pickle=True)
    return {layer: data[f"layer_{layer}"] for layer in LAYERS}


def main():
    with open(EXP_DIR / "target_tasks.json") as f:
        targets = json.load(f)
    with open(EXP_DIR / "system_prompts.json") as f:
        system_prompts = json.load(f)["prompts"]
    with open(EXP_DIR / "results" / "behavioral_all_20.json") as f:
        behavioral = json.load(f)

    behavioral_by_id = {b["prompt_id"]: b for b in behavioral}
    target_id_to_idx = {t["task_id"]: i for i, t in enumerate(targets)}
    category_to_target = {t["topic"]: t["task_id"] for t in targets}

    # Load probes
    probes = {}
    for layer in LAYERS:
        coef, intercept = load_probe(layer)
        probes[layer] = (coef, intercept)

    # Load baseline activations and score
    baseline_acts = load_activations("baseline.npz")
    baseline_scores = {}
    for layer in LAYERS:
        coef, intercept = probes[layer]
        baseline_scores[layer] = score_activations(baseline_acts[layer], coef, intercept)

    print("Baseline probe scores (per target task):")
    print(f"  {'Task ID':<30} {'L31':>8} {'L43':>8} {'L55':>8}")
    for t in targets:
        idx = target_id_to_idx[t["task_id"]]
        scores_str = "  ".join(
            f"{baseline_scores[l][idx]:8.3f}" for l in LAYERS
        )
        print(f"  {t['task_id']:<30} {scores_str}")

    # For each system prompt, compute probe delta for the targeted task
    results = []
    for sp in system_prompts:
        beh = behavioral_by_id[sp["id"]]
        target_task_id = category_to_target[sp["target_category"]]
        target_idx = target_id_to_idx[target_task_id]

        # Load manipulated activations
        manip_acts = load_activations(f"{sp['id']}.npz")

        probe_deltas = {}
        for layer in LAYERS:
            coef, intercept = probes[layer]
            manip_score = score_activations(manip_acts[layer], coef, intercept)
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

    # Print results table
    print(f"\n{'Prompt ID':<30} {'Dir':<5} {'Beh Δ':>8} {'Probe L31':>10} {'Probe L43':>10} {'Probe L55':>10}")
    print("-" * 80)
    for r in results:
        print(f"{r['prompt_id']:<30} {r['direction']:<5} {r['behavioral_delta']:>+8.3f} "
              f"{r['probe_delta_L31']:>+10.3f} {r['probe_delta_L43']:>+10.3f} {r['probe_delta_L55']:>+10.3f}")

    # Compute correlations
    beh_deltas = np.array([r["behavioral_delta"] for r in results])

    print("\n=== Correlations (behavioral delta vs probe delta) ===")
    for layer in LAYERS:
        key = f"probe_delta_L{layer}"
        probe_d = np.array([r[key] for r in results])

        pearson_r, pearson_p = stats.pearsonr(beh_deltas, probe_d)
        spearman_r, spearman_p = stats.spearmanr(beh_deltas, probe_d)

        # Sign agreement
        sign_agree = np.mean(np.sign(beh_deltas) == np.sign(probe_d))

        print(f"\n  Layer {layer}:")
        print(f"    Pearson r={pearson_r:.3f} (p={pearson_p:.1e})")
        print(f"    Spearman r={spearman_r:.3f} (p={spearman_p:.1e})")
        print(f"    Sign agreement: {sign_agree:.1%} ({int(sign_agree * len(results))}/{len(results)})")

    # Also compute correlation only for "on-target" manipulations
    # (exclude prompts where behavioral delta is near zero due to ceiling/floor)
    strong_mask = np.abs(beh_deltas) > 0.05
    print(f"\n=== Strong manipulations only (|behavioral delta| > 0.05, n={strong_mask.sum()}) ===")
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

    # Save results
    output_path = EXP_DIR / "results" / "probe_behavioral_comparison.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved detailed results to {output_path}")


if __name__ == "__main__":
    main()
