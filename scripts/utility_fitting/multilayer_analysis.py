"""Run the OOD utility fitting analysis across multiple layers."""

import json
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv

load_dotenv()

from scripts.utility_fitting.analyze_ood import (
    analyze_experiment,
    analyze_mra,
)


def main():
    probe_dir = Path("results/probes/gemma3_10k_heldout_std_raw")
    manifest = json.loads((probe_dir / "manifest.json").read_text())

    # Available layers in OOD activations: 31, 43, 55
    layers = [31, 43, 55]
    run_prefix = "completion_preference_gemma-3-27b_completion_canonical_seed0"

    all_results = []
    for layer in layers:
        probe_id = f"ridge_L{layer:02d}"
        probe_path = probe_dir / "probes" / f"probe_{probe_id}.npy"
        if not probe_path.exists():
            print(f"No probe for layer {layer}, skipping")
            continue

        probe_weights = np.load(probe_path)
        print(f"\n{'#'*80}")
        print(f"LAYER {layer} (probe shape: {probe_weights.shape})")
        print(f"{'#'*80}")

        # Exp 1b
        results = analyze_experiment(
            exp_name=f"exp1b_L{layer}",
            config_dir=Path("configs/measurement/active_learning/ood_exp1b"),
            results_dir=Path("results/experiments/ood_exp1b/pre_task_active_learning"),
            act_dir_base=Path("activations/ood/exp1_prompts"),
            run_prefix=run_prefix,
            task_prefix="hidden_",
            probe_weights=probe_weights,
            layer=layer,
        )
        for r in results:
            r["layer"] = layer
        all_results.extend(results)

        # Exp 1c
        results = analyze_experiment(
            exp_name=f"exp1c_L{layer}",
            config_dir=Path("configs/measurement/active_learning/ood_exp1c"),
            results_dir=Path("results/experiments/ood_exp1c/pre_task_active_learning"),
            act_dir_base=Path("activations/ood/exp1_prompts"),
            run_prefix=run_prefix,
            task_prefix="crossed_",
            probe_weights=probe_weights,
            layer=layer,
        )
        for r in results:
            r["layer"] = layer
        all_results.extend(results)

        # Exp 1d
        results = analyze_experiment(
            exp_name=f"exp1d_L{layer}",
            config_dir=Path("configs/measurement/active_learning/ood_exp1d"),
            results_dir=Path("results/experiments/ood_exp1d/pre_task_active_learning"),
            act_dir_base=Path("activations/ood/exp1_prompts"),
            run_prefix=run_prefix,
            task_prefix="crossed_",
            probe_weights=probe_weights,
            layer=layer,
        )
        for r in results:
            r["layer"] = layer
        all_results.extend(results)

        # MRA
        results = analyze_mra(probe_weights, layer)
        for r in results:
            r["layer"] = layer
        all_results.extend(results)

    # Save all results
    output_path = Path("experiments/ood_system_prompts/utility_fitting/multilayer_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved {len(all_results)} results to {output_path}")

    # Print layer summary
    print(f"\n{'='*80}")
    print("LAYER SUMMARY (mean across conditions)")
    print(f"{'='*80}")

    for layer in layers:
        layer_results = [r for r in all_results if r.get("layer") == layer]
        for exp_prefix in ["exp1b", "exp1c", "exp1d", "mra"]:
            exp_results = [r for r in layer_results if r["experiment"].startswith(exp_prefix)]
            if not exp_results:
                continue
            mean_cond_r = np.mean([r["cond_probe_r"] for r in exp_results])
            mean_cond_acc = np.mean([r["cond_probe_acc"] for r in exp_results])
            bl_probe_rs = [r["bl_probe_r"] for r in exp_results if "bl_probe_r" in r and not np.isnan(r["bl_probe_r"])]
            mean_bl_r = np.mean(bl_probe_rs) if bl_probe_rs else float("nan")
            print(
                f"  L{layer} {exp_prefix}: cond_r={mean_cond_r:.3f} cond_acc={mean_cond_acc:.3f} "
                f"bl_probe_r={mean_bl_r:.3f} (n={len(exp_results)})"
            )


if __name__ == "__main__":
    main()
