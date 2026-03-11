"""Cross-evaluate base model probes on character persona activations.

For each (selector, layer) probe trained on the base Llama 3.1 8B model:
1. Load the trained probe weights
2. For each character persona:
   a. Load persona activations
   b. Load persona Thurstonian utilities
   c. Predict utilities from activations using the probe
   d. Compute Pearson r between predicted and actual utilities
3. Also compute baseline: correlation between base utilities and persona utilities
4. Save results as JSON and generate plots

Usage:
    python scripts/character_probes/cross_eval_probes.py
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

from src.probes.core.activations import load_activations
from src.probes.data_loading import load_thurstonian_scores

ACTIVATIONS_DIR = Path("activations/character_probes")
RESULTS_DIR = Path("results/experiments/character_probes")
PROBES_DIR = Path("results/probes/character_probes")
OUTPUT_DIR = Path("results/probes/character_probes/cross_eval")

PERSONAS = [
    "goodness", "humor", "impulsiveness", "loving", "mathematical",
    "nonchalance", "poeticism", "remorse", "sarcasm", "sycophancy",
]

SELECTORS = [
    "task_mean",
    "turn_boundary:-1",
    "turn_boundary:-2",
    "turn_boundary:-3",
    "turn_boundary:-4",
    "turn_boundary:-5",
]

LAYERS = [8, 12, 16, 20, 24]


def selector_to_dir_name(selector: str) -> str:
    return selector.replace(":", "_").replace("-", "m")


def load_persona_scores(persona: str, split: str = "a") -> dict[str, float]:
    """Load Thurstonian scores for a persona from split A."""
    persona_dir = RESULTS_DIR / persona / "pre_task_active_learning"
    split_dirs = list(persona_dir.glob(f"*split_{split}_*"))
    assert len(split_dirs) == 1, f"Expected 1 split dir for {persona}/{split}, got {len(split_dirs)}"
    return load_thurstonian_scores(split_dirs[0])


def load_base_scores(split: str = "a") -> dict[str, float]:
    base_dir = RESULTS_DIR / "pre_task_active_learning"
    split_dirs = list(base_dir.glob(f"*split_{split}_*"))
    assert len(split_dirs) == 1, f"Expected 1 split dir for base/{split}, got {len(split_dirs)}"
    return load_thurstonian_scores(split_dirs[0])


def load_probe_weights(selector: str, layer: int) -> np.ndarray:
    safe_name = selector_to_dir_name(selector)
    probe_path = PROBES_DIR / f"llama8b_base_{safe_name}" / "probes" / f"probe_ridge_L{layer:02d}.npy"
    return np.load(probe_path)


def predict_with_probe(weights: np.ndarray, activations: np.ndarray) -> np.ndarray:
    coef = weights[:-1]
    intercept = weights[-1]
    return activations @ coef + intercept


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load base model scores (split A)
    print("Loading base model scores...")
    base_scores = load_base_scores(split="a")
    print(f"  {len(base_scores)} tasks")

    # Load all persona scores
    print("Loading persona scores...")
    persona_scores = {}
    for persona in PERSONAS:
        persona_scores[persona] = load_persona_scores(persona, split="a")
        print(f"  {persona}: {len(persona_scores[persona])} tasks")

    # Compute baseline correlations (base utilities vs persona utilities)
    print("\nBaseline correlations (base vs persona utilities)...")
    baseline_r = {}
    for persona in PERSONAS:
        common = sorted(set(base_scores) & set(persona_scores[persona]))
        base_vals = np.array([base_scores[tid] for tid in common])
        persona_vals = np.array([persona_scores[persona][tid] for tid in common])
        r, _ = pearsonr(base_vals, persona_vals)
        baseline_r[persona] = float(r)
        print(f"  {persona}: r={r:.4f} ({len(common)} tasks)")

    # Cross-evaluate probes
    results = {
        "baseline_r": baseline_r,
        "probe_transfer": {},
        "base_within": {},
    }

    for selector in SELECTORS:
        safe_name = selector_to_dir_name(selector)
        print(f"\n{'='*60}")
        print(f"Selector: {selector}")
        print(f"{'='*60}")

        results["probe_transfer"][selector] = {}
        results["base_within"][selector] = {}

        for layer in LAYERS:
            print(f"\n  Layer {layer}:")
            weights = load_probe_weights(selector, layer)

            # Base model within-persona eval (for reference)
            base_act_path = ACTIVATIONS_DIR / "llama_3_1_8b_base" / f"activations_{selector}.npz"
            base_task_ids, base_acts = load_activations(
                base_act_path, task_id_filter=set(base_scores), layers=[layer],
            )
            base_id_to_idx = {tid: i for i, tid in enumerate(base_task_ids)}
            common_base = [tid for tid in base_scores if tid in base_id_to_idx]
            base_X = base_acts[layer][[base_id_to_idx[tid] for tid in common_base]]
            base_y = np.array([base_scores[tid] for tid in common_base])
            base_pred = predict_with_probe(weights, base_X)
            base_r, _ = pearsonr(base_y, base_pred)
            results["base_within"][selector][str(layer)] = {
                "r": float(base_r),
                "n": len(common_base),
            }
            print(f"    Base (within): r={base_r:.4f} ({len(common_base)} tasks)")

            results["probe_transfer"][selector][str(layer)] = {}

            for persona in PERSONAS:
                # Load persona activations
                persona_act_path = (
                    ACTIVATIONS_DIR / f"llama_3_1_8b_{persona}" / f"activations_{selector}.npz"
                )
                p_task_ids, p_acts = load_activations(
                    persona_act_path,
                    task_id_filter=set(persona_scores[persona]),
                    layers=[layer],
                )
                p_id_to_idx = {tid: i for i, tid in enumerate(p_task_ids)}

                # Align: tasks with both activations and scores
                common = [tid for tid in persona_scores[persona] if tid in p_id_to_idx]
                X = p_acts[layer][[p_id_to_idx[tid] for tid in common]]
                y = np.array([persona_scores[persona][tid] for tid in common])

                pred = predict_with_probe(weights, X)
                r, _ = pearsonr(y, pred)

                results["probe_transfer"][selector][str(layer)][persona] = {
                    "r": float(r),
                    "n": len(common),
                }
                print(f"    {persona}: r={r:.4f} ({len(common)} tasks)")

    # Save results
    output_path = OUTPUT_DIR / "cross_eval_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_path}")

    # Print summary: best selector/layer per persona
    print("\n" + "=" * 60)
    print("SUMMARY: Best probe transfer per persona")
    print("=" * 60)
    print(f"{'Persona':<16} {'Baseline r':>10} {'Best probe r':>12} {'Selector':>20} {'Layer':>6}")
    print("-" * 66)
    for persona in PERSONAS:
        best_r = -1
        best_sel = ""
        best_layer = 0
        for selector in SELECTORS:
            for layer in LAYERS:
                r = results["probe_transfer"][selector][str(layer)][persona]["r"]
                if r > best_r:
                    best_r = r
                    best_sel = selector
                    best_layer = layer
        bl = baseline_r[persona]
        print(f"{persona:<16} {bl:>10.4f} {best_r:>12.4f} {best_sel:>20} {best_layer:>6}")


if __name__ == "__main__":
    main()
