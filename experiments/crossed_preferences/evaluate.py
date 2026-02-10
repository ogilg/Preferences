"""Evaluate crossed preferences: probe-behavioral correlations and category shell analysis.

Key analyses:
1. Cross-category generalization: does "you hate cheese" shift probe for math-about-cheese?
2. Category shell effect: does probe shift differ by category shell?
3. Pure vs crossed comparison: is effect attenuated in crossed tasks?
4. Harmful interaction: does topic preference interact with harmful framing?
5. Subtle prompt analysis: do subtle expressions work as well as direct ones?
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

EXP_DIR = Path("experiments/crossed_preferences")
HIDDEN_DIR = Path("experiments/hidden_preferences")
ACT_DIR = EXP_DIR / "activations"
PROBE_DIR = Path("results/probes/gemma3_3k_completion_preference/probes")
RESULTS_DIR = EXP_DIR / "results"
LAYERS = [31, 43, 55]


def load_probe(layer: int) -> tuple[np.ndarray, float]:
    weights = np.load(PROBE_DIR / f"probe_ridge_L{layer}.npy")
    return weights[:-1], weights[-1]


def score_activations(activations: np.ndarray, coef: np.ndarray, intercept: float) -> np.ndarray:
    return activations @ coef + intercept


def load_activations(filename: str) -> dict[int, np.ndarray]:
    data = np.load(ACT_DIR / filename, allow_pickle=True)
    return {layer: data[f"layer_{layer}"] for layer in LAYERS}


def load_task_index() -> dict[str, int]:
    data = np.load(ACT_DIR / "baseline.npz", allow_pickle=True)
    task_ids = list(data["task_ids"])
    return {tid: i for i, tid in enumerate(task_ids)}


def evaluate_probe_behavioral(behavioral_file: str, output_file: str):
    with open(RESULTS_DIR / behavioral_file) as f:
        behavioral = json.load(f)

    task_idx = load_task_index()

    # Load probes and baseline
    probes = {layer: load_probe(layer) for layer in LAYERS}
    baseline_acts = load_activations("baseline.npz")
    baseline_scores = {}
    for layer in LAYERS:
        coef, intercept = probes[layer]
        baseline_scores[layer] = score_activations(baseline_acts[layer], coef, intercept)

    results = []
    seen_prompts = set()

    for beh in behavioral:
        prompt_id = beh["prompt_id"]
        target_task_id = beh["target_task_id"]

        if target_task_id not in task_idx:
            print(f"  WARNING: {target_task_id} not in activation index")
            continue

        npz_path = ACT_DIR / f"{prompt_id}.npz"
        if not npz_path.exists():
            if prompt_id not in seen_prompts:
                print(f"  WARNING: missing activations for {prompt_id}")
                seen_prompts.add(prompt_id)
            continue

        tidx = task_idx[target_task_id]
        manip_acts = load_activations(f"{prompt_id}.npz")

        probe_deltas = {}
        for layer in LAYERS:
            coef, intercept = probes[layer]
            manip_score = score_activations(manip_acts[layer], coef, intercept)
            probe_deltas[layer] = float(manip_score[tidx] - baseline_scores[layer][tidx])

        results.append({
            "prompt_id": prompt_id,
            "target_topic": beh["target_topic"],
            "target_task_id": target_task_id,
            "direction": beh["direction"],
            "prompt_type": beh["prompt_type"],
            "task_set": beh["task_set"],
            "category_shell": beh["category_shell"],
            "behavioral_delta": beh["delta"],
            "probe_delta_L31": probe_deltas[31],
            "probe_delta_L43": probe_deltas[43],
            "probe_delta_L55": probe_deltas[55],
        })

    # Overall correlation
    if len(results) < 3:
        print(f"Only {len(results)} results, skipping correlation")
        return results

    beh_deltas = np.array([r["behavioral_delta"] for r in results])

    print(f"\n=== Overall Correlations (n={len(results)}) ===")
    for layer in LAYERS:
        key = f"probe_delta_L{layer}"
        probe_d = np.array([r[key] for r in results])
        pearson_r, pearson_p = stats.pearsonr(beh_deltas, probe_d)
        spearman_r, spearman_p = stats.spearmanr(beh_deltas, probe_d)
        sign_agree = np.mean(np.sign(beh_deltas) == np.sign(probe_d))
        print(f"  L{layer}: Pearson r={pearson_r:.3f} (p={pearson_p:.1e}), "
              f"Spearman r={spearman_r:.3f}, Sign={sign_agree:.1%}")

    # Category shell breakdown (key analysis)
    print(f"\n=== Category Shell Analysis (L31) ===")
    shells = sorted(set(r["category_shell"] for r in results))
    print(f"{'Shell':<20} {'N':>4} {'Mean BehD':>10} {'Mean ProbD':>10} {'Pearson r':>10} {'Sign%':>8}")
    print("-" * 65)
    for shell in shells:
        shell_results = [r for r in results if r["category_shell"] == shell]
        if len(shell_results) < 3:
            continue
        beh_d = np.array([r["behavioral_delta"] for r in shell_results])
        probe_d = np.array([r["probe_delta_L31"] for r in shell_results])
        r_val, p_val = stats.pearsonr(beh_d, probe_d) if len(shell_results) >= 3 else (float("nan"), 1.0)
        sign_pct = np.mean(np.sign(beh_d) == np.sign(probe_d))
        print(f"{shell:<20} {len(shell_results):>4} {np.mean(beh_d):>+10.3f} "
              f"{np.mean(probe_d):>+10.3f} {r_val:>+10.3f} {sign_pct:>8.1%}")

    # Task set breakdown (crossed vs pure vs subtle)
    print(f"\n=== Task Set Comparison (L31) ===")
    task_sets = sorted(set(r["task_set"] for r in results))
    print(f"{'Set':<10} {'N':>4} {'Mean |BehD|':>12} {'Mean |ProbD|':>12} {'Pearson r':>10} {'Sign%':>8}")
    print("-" * 60)
    for ts in task_sets:
        ts_results = [r for r in results if r["task_set"] == ts]
        if len(ts_results) < 3:
            continue
        beh_d = np.array([r["behavioral_delta"] for r in ts_results])
        probe_d = np.array([r["probe_delta_L31"] for r in ts_results])
        r_val, _ = stats.pearsonr(beh_d, probe_d)
        sign_pct = np.mean(np.sign(beh_d) == np.sign(probe_d))
        print(f"{ts:<10} {len(ts_results):>4} {np.mean(np.abs(beh_d)):>12.3f} "
              f"{np.mean(np.abs(probe_d)):>12.3f} {r_val:>+10.3f} {sign_pct:>8.1%}")

    # Direction agreement for crossed tasks
    crossed_results = [r for r in results if r["task_set"] == "crossed"]
    if crossed_results:
        pos = [r for r in crossed_results if r["direction"] == "positive"]
        neg = [r for r in crossed_results if r["direction"] == "negative"]
        pos_correct = sum(1 for r in pos if r["behavioral_delta"] > 0)
        neg_correct = sum(1 for r in neg if r["behavioral_delta"] < 0)
        total_correct = pos_correct + neg_correct
        total = len(crossed_results)
        print(f"\n=== Crossed Task Direction Agreement ===")
        print(f"Positive prompts: {pos_correct}/{len(pos)} correct")
        print(f"Negative prompts: {neg_correct}/{len(neg)} correct")
        print(f"Overall: {total_correct}/{total} ({total_correct/total:.1%})")

    # Harmful interaction
    harmful_results = [r for r in results if r["category_shell"] == "harmful"]
    if harmful_results:
        print(f"\n=== Harmful Shell Analysis ===")
        for r in harmful_results:
            print(f"  {r['prompt_id']:<35} {r['target_task_id']:<30} "
                  f"dir={r['direction']:<4} beh_d={r['behavioral_delta']:+.3f} "
                  f"probe_d={r['probe_delta_L31']:+.1f}")

    # Prompt type breakdown
    print(f"\n=== Prompt Type Analysis (L31, crossed tasks only) ===")
    if crossed_results:
        ptypes = sorted(set(r["prompt_type"] for r in crossed_results))
        print(f"{'Type':<15} {'N':>4} {'Mean |BehD|':>12} {'Mean |ProbD|':>12} {'Sign%':>8}")
        print("-" * 55)
        for pt in ptypes:
            pt_results = [r for r in crossed_results if r["prompt_type"] == pt]
            beh_d = np.array([r["behavioral_delta"] for r in pt_results])
            probe_d = np.array([r["probe_delta_L31"] for r in pt_results])
            sign_pct = np.mean(np.sign(beh_d) == np.sign(probe_d))
            print(f"{pt:<15} {len(pt_results):>4} {np.mean(np.abs(beh_d)):>12.3f} "
                  f"{np.mean(np.abs(probe_d)):>12.3f} {sign_pct:>8.1%}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / output_file
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--behavioral", default="behavioral_results.json")
    parser.add_argument("--output", default="probe_behavioral_results.json")
    args = parser.parse_args()

    evaluate_probe_behavioral(args.behavioral, args.output)


if __name__ == "__main__":
    main()
