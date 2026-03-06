"""Analyze OOD system prompt experiments with EOT probes.

Exact same analysis as scripts/ood_system_prompts/analyze_ood.py,
but using EOT activations and EOT-trained probe.

Usage: python scripts/ood_eot/analyze_eot.py [--exp EXP] [--layer LAYER]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.ood.analysis import (
    compute_p_choose_from_pairwise,
    compute_deltas,
    correlate_deltas,
    per_condition_correlations,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ACTS_DIR = REPO_ROOT / "activations" / "ood_eot"
PROBE_DIR = REPO_ROOT / "results" / "probes" / "heldout_eval_gemma3_eot" / "probes"
RESULTS_OOD = REPO_ROOT / "results" / "ood"
LAYERS = [31]
ACTS_FILENAME = "activations_eot.npz"


def probe_path(layer: int) -> Path:
    return PROBE_DIR / f"probe_ridge_L{layer}.npy"


def summarize(name: str, layer: int, behavioral: np.ndarray, probe: np.ndarray, labels: np.ndarray) -> dict:
    if len(behavioral) == 0:
        print(f"  {name} L{layer}: NO DATA")
        return {}
    stats = correlate_deltas(behavioral, probe)
    per_cond = per_condition_correlations(behavioral, probe, labels)
    print(f"  {name} L{layer}: n={stats['n']}, r={stats['pearson_r']:.3f} (p={stats['permutation_p']:.3f}), "
          f"sign={stats['sign_agreement']:.1%} (n={stats['sign_n']})")
    return {
        "layer": layer,
        "n": stats["n"],
        "pearson_r": stats["pearson_r"],
        "pearson_p": stats["pearson_p"],
        "spearman_r": stats["spearman_r"],
        "permutation_p": stats["permutation_p"],
        "sign_agreement": stats["sign_agreement"],
        "sign_n": stats["sign_n"],
        "per_condition": per_cond,
        "behavioral_deltas": behavioral.tolist(),
        "probe_deltas": probe.tolist(),
        "condition_labels": labels.tolist(),
    }


def exp1a_analysis(layer: int) -> dict:
    pairwise = json.load(open(RESULTS_OOD / "category_preference" / "pairwise.json"))
    rates = compute_p_choose_from_pairwise(pairwise["results"])
    acts_dir = ACTS_DIR / "exp1_category"
    b, p, labels = compute_deltas(rates, acts_dir, probe_path(layer), layer,
                                  activations_filename=ACTS_FILENAME)
    return summarize("exp1a", layer, b, p, labels)


def exp1b_analysis(layer: int) -> dict:
    pairwise = json.load(open(RESULTS_OOD / "hidden_preference" / "pairwise.json"))
    rates = compute_p_choose_from_pairwise(pairwise["results"])
    targeted_rates = {k: v for k, v in rates.items() if not k.startswith("compete_")}
    targeted_rates = {
        k: {tid: v for tid, v in rate_dict.items() if tid.startswith("hidden_")}
        for k, rate_dict in targeted_rates.items()
    }
    acts_dir = ACTS_DIR / "exp1_prompts"
    b, p, labels = compute_deltas(targeted_rates, acts_dir, probe_path(layer), layer,
                                  activations_filename=ACTS_FILENAME)
    return summarize("exp1b", layer, b, p, labels)


def exp1c_analysis(layer: int) -> dict:
    pairwise = json.load(open(RESULTS_OOD / "crossed_preference" / "pairwise.json"))
    rates = compute_p_choose_from_pairwise(pairwise["results"])
    targeted_rates = {k: v for k, v in rates.items() if not k.startswith("compete_")}
    targeted_rates = {
        k: {tid: v for tid, v in rate_dict.items() if tid.startswith("crossed_")}
        for k, rate_dict in targeted_rates.items()
    }
    acts_dir = ACTS_DIR / "exp1_prompts"
    b, p, labels = compute_deltas(targeted_rates, acts_dir, probe_path(layer), layer,
                                  activations_filename=ACTS_FILENAME)
    return summarize("exp1c", layer, b, p, labels)


def _parse_competing_condition(cid: str) -> tuple[str, str, str]:
    parts = cid.split("_")
    direction = parts[-1]
    shell = parts[-2]
    topic = "_".join(parts[1:-2])
    return topic, shell, direction


def exp1d_analysis(layer: int) -> dict:
    pairwise = json.load(open(RESULTS_OOD / "crossed_preference" / "pairwise.json"))
    rates = compute_p_choose_from_pairwise(pairwise["results"])

    compete_cids = [k for k in rates if k.startswith("compete_")]
    on_target_rates: dict[str, dict[str, float]] = {"baseline": {}}

    for cid in compete_cids:
        topic, shell, direction = _parse_competing_condition(cid)
        target_task = f"crossed_{topic}_{shell}"
        if target_task not in rates.get(cid, {}):
            continue
        on_target_rates[cid] = {target_task: rates[cid][target_task]}
        if target_task in rates.get("baseline", {}):
            on_target_rates["baseline"][target_task] = rates["baseline"][target_task]

    acts_dir = ACTS_DIR / "exp1_prompts"
    b, p, labels = compute_deltas(on_target_rates, acts_dir, probe_path(layer), layer,
                                  activations_filename=ACTS_FILENAME)
    result = summarize("exp1d_ontarget", layer, b, p, labels)

    competing_rates = {k: v for k, v in rates.items()
                      if k.startswith("compete_") or k == "baseline"}
    competing_rates = {
        k: {tid: v for tid, v in rate_dict.items() if tid.startswith("crossed_")}
        for k, rate_dict in competing_rates.items()
    }
    b_full, p_full, labels_full = compute_deltas(competing_rates, acts_dir, probe_path(layer), layer,
                                                 activations_filename=ACTS_FILENAME)
    result_full = summarize("exp1d_full", layer, b_full, p_full, labels_full)

    return {"on_target": result, "full": result_full}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="all",
                       choices=["all", "exp1a", "exp1b", "exp1c", "exp1d"])
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    layers = [args.layer] if args.layer else LAYERS
    results = {}

    exp_fns = {
        "exp1a": exp1a_analysis,
        "exp1b": exp1b_analysis,
        "exp1c": exp1c_analysis,
        "exp1d": exp1d_analysis,
    }

    exps_to_run = list(exp_fns.keys()) if args.exp == "all" else [args.exp]

    for exp_name in exps_to_run:
        print(f"\n--- {exp_name} ---")
        results[exp_name] = {}
        for layer in layers:
            try:
                res = exp_fns[exp_name](layer)
                results[exp_name][f"L{layer}"] = res
            except FileNotFoundError as e:
                print(f"  {exp_name} L{layer}: MISSING — {e}")
            except Exception as e:
                print(f"  {exp_name} L{layer}: ERROR — {e}")
                import traceback
                traceback.print_exc()

    output_path = args.output or REPO_ROOT / "experiments" / "ood_eot" / "analysis_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _large_keys = {"behavioral_deltas", "probe_deltas", "condition_labels"}

    results_summary = {}
    for exp_name, layer_results in results.items():
        results_summary[exp_name] = {}
        for layer_key, res in layer_results.items():
            if not res:
                continue
            if "on_target" in res and "full" in res:
                results_summary[exp_name][layer_key] = {
                    "on_target": {k: v for k, v in res["on_target"].items() if k not in _large_keys},
                    "full": {k: v for k, v in res["full"].items() if k not in _large_keys},
                }
            else:
                results_summary[exp_name][layer_key] = {k: v for k, v in res.items() if k not in _large_keys}

    with open(output_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSaved results to {output_path}")

    full_output = output_path.parent / "analysis_results_full.json"
    with open(full_output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved full results to {full_output}")


if __name__ == "__main__":
    main()
