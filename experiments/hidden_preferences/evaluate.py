"""Evaluate hidden preferences: correlate behavioral deltas with probe deltas.

For each system prompt manipulation x target task pair:
1. Load baseline and manipulated activations
2. Score with ridge probes at layers 31, 43, 55
3. Compute probe delta for the targeted task(s)
4. Correlate probe delta with behavioral delta
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

EXP_DIR = Path("experiments/hidden_preferences")
ACT_DIR = EXP_DIR / "activations"
PROBE_DIR = Path("results/probes/gemma3_3k_completion_preference/probes")
LAYERS = [31, 43, 55]


def load_probe(layer: int) -> tuple[np.ndarray, float]:
    weights = np.load(PROBE_DIR / f"probe_ridge_L{layer}.npy")
    return weights[:-1], weights[-1]


def score_activations(activations: np.ndarray, coef: np.ndarray, intercept: float) -> np.ndarray:
    return activations @ coef + intercept


def load_activations(act_dir: Path, filename: str) -> dict[int, np.ndarray]:
    data = np.load(act_dir / filename, allow_pickle=True)
    return {layer: data[f"layer_{layer}"] for layer in LAYERS}


def evaluate_set(behavioral_file: str, act_dir: Path, prompt_file: str, output_file: str):
    with open(EXP_DIR / "target_tasks.json") as f:
        targets = json.load(f)
    with open(EXP_DIR / prompt_file) as f:
        system_prompts = json.load(f)["prompts"]
    with open(EXP_DIR / "results" / behavioral_file) as f:
        behavioral = json.load(f)

    # Build lookup: (prompt_id, target_task_id) -> behavioral result
    behavioral_lookup = {(b["prompt_id"], b["target_task_id"]): b for b in behavioral}

    target_id_to_idx = {t["task_id"]: i for i, t in enumerate(targets)}
    topic_to_targets = {}
    for t in targets:
        topic_to_targets.setdefault(t["topic"], []).append(t["task_id"])

    # Load probes
    probes = {}
    for layer in LAYERS:
        coef, intercept = load_probe(layer)
        probes[layer] = (coef, intercept)

    # Load baseline activations and score
    baseline_acts = load_activations(act_dir, "baseline.npz")
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

    # For each system prompt x target task pair
    results = []
    for sp in system_prompts:
        npz_path = act_dir / f"{sp['id']}.npz"
        if not npz_path.exists():
            print(f"  WARNING: missing {npz_path}")
            continue

        manip_acts = load_activations(act_dir, f"{sp['id']}.npz")
        topic = sp["target_topic"]
        target_task_ids = topic_to_targets[topic]

        for target_task_id in target_task_ids:
            key = (sp["id"], target_task_id)
            if key not in behavioral_lookup:
                continue

            beh = behavioral_lookup[key]
            target_idx = target_id_to_idx[target_task_id]

            probe_deltas = {}
            for layer in LAYERS:
                coef, intercept = probes[layer]
                manip_score = score_activations(manip_acts[layer], coef, intercept)
                probe_deltas[layer] = float(manip_score[target_idx] - baseline_scores[layer][target_idx])

            results.append({
                "prompt_id": sp["id"],
                "target_topic": topic,
                "target_task_id": target_task_id,
                "direction": sp["direction"],
                "prompt_type": sp["type"],
                "behavioral_delta": beh["delta"],
                "probe_delta_L31": probe_deltas[31],
                "probe_delta_L43": probe_deltas[43],
                "probe_delta_L55": probe_deltas[55],
            })

    # Print results table
    print(f"\n{'Prompt ID':<35} {'Task':<25} {'Dir':<5} {'Beh D':>8} {'PL31':>10} {'PL43':>10} {'PL55':>10}")
    print("-" * 105)
    for r in results:
        print(f"{r['prompt_id']:<35} {r['target_task_id']:<25} {r['direction']:<5} "
              f"{r['behavioral_delta']:>+8.3f} {r['probe_delta_L31']:>+10.3f} "
              f"{r['probe_delta_L43']:>+10.3f} {r['probe_delta_L55']:>+10.3f}")

    # Compute correlations
    beh_deltas = np.array([r["behavioral_delta"] for r in results])

    print(f"\n=== Correlations (n={len(results)}) ===")
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

    # Save results
    output_path = EXP_DIR / "results" / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved detailed results to {output_path}")


def evaluate_controls():
    """Evaluate positive controls using OOD target tasks."""
    from pathlib import Path

    ood_dir = Path("experiments/ood_generalization")
    ctrl_act_dir = ACT_DIR / "controls"

    with open(ood_dir / "target_tasks.json") as f:
        ood_targets = json.load(f)
    with open(EXP_DIR / "positive_controls.json") as f:
        control_prompts = json.load(f)["prompts"]
    with open(EXP_DIR / "results" / "behavioral_controls.json") as f:
        behavioral = json.load(f)

    behavioral_lookup = {b["prompt_id"]: b for b in behavioral}
    target_id_to_idx = {t["task_id"]: i for i, t in enumerate(ood_targets)}
    category_to_target = {t["topic"]: t["task_id"] for t in ood_targets}

    probes = {}
    for layer in LAYERS:
        coef, intercept = load_probe(layer)
        probes[layer] = (coef, intercept)

    baseline_acts = load_activations(ctrl_act_dir, "baseline.npz")
    baseline_scores = {}
    for layer in LAYERS:
        coef, intercept = probes[layer]
        baseline_scores[layer] = score_activations(baseline_acts[layer], coef, intercept)

    print("\n=== Positive Control Evaluation ===")
    results = []
    for sp in control_prompts:
        npz_path = ctrl_act_dir / f"{sp['id']}.npz"
        if not npz_path.exists():
            continue

        manip_acts = load_activations(ctrl_act_dir, f"{sp['id']}.npz")
        target_task_id = category_to_target[sp["target_category"]]
        target_idx = target_id_to_idx[target_task_id]

        beh = behavioral_lookup[sp["id"]]

        probe_deltas = {}
        for layer in LAYERS:
            coef, intercept = probes[layer]
            manip_score = score_activations(manip_acts[layer], coef, intercept)
            probe_deltas[layer] = float(manip_score[target_idx] - baseline_scores[layer][target_idx])

        results.append({
            "prompt_id": sp["id"],
            "target_category": sp["target_category"],
            "direction": sp["direction"],
            "behavioral_delta": beh["delta"],
            "probe_delta_L31": probe_deltas[31],
            "probe_delta_L43": probe_deltas[43],
            "probe_delta_L55": probe_deltas[55],
        })

    print(f"{'Prompt ID':<25} {'Cat':<15} {'Dir':<5} {'Beh D':>8} {'PL31':>10}")
    print("-" * 65)
    for r in results:
        print(f"{r['prompt_id']:<25} {r['target_category']:<15} {r['direction']:<5} "
              f"{r['behavioral_delta']:>+8.3f} {r['probe_delta_L31']:>+10.3f}")

    # Save
    output_path = EXP_DIR / "results" / "probe_behavioral_controls.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved control results to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", choices=["iteration", "holdout", "controls", "all"], default="all")
    args = parser.parse_args()

    if args.set in ("iteration", "all"):
        print("=" * 80)
        print("ITERATION SET")
        print("=" * 80)
        evaluate_set(
            behavioral_file="behavioral_iteration.json",
            act_dir=ACT_DIR,
            prompt_file="system_prompts.json",
            output_file="probe_behavioral_iteration.json",
        )

    if args.set in ("holdout", "all"):
        print("\n" + "=" * 80)
        print("HOLDOUT SET")
        print("=" * 80)
        evaluate_set(
            behavioral_file="behavioral_holdout.json",
            act_dir=ACT_DIR,
            prompt_file="holdout_prompts.json",
            output_file="probe_behavioral_holdout.json",
        )

    if args.set in ("controls", "all"):
        evaluate_controls()


if __name__ == "__main__":
    main()
