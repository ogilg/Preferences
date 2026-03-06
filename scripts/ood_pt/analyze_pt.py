"""Analyze OOD system prompt experiments with PT model and cross-model probes.

Runs three conditions for each experiment (1a-1d):
  1. PT probe on PT activations (main question)
  2. PT probe on IT activations (cross-model transfer)
  3. IT probe on PT activations (cross-model transfer)

IT probe on IT activations already exists in the original OOD report.

Usage: python scripts/ood_pt/analyze_pt.py [--exp EXP]
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
RESULTS_OOD = REPO_ROOT / "results" / "ood"
LAYER = 31

PT_PROBE = REPO_ROOT / "results/probes/gemma3_pt_10k_heldout_std_raw/probes/probe_ridge_L31.npy"
IT_PROBE = REPO_ROOT / "results/probes/gemma3_10k_heldout_std_raw/probes/probe_ridge_L31.npy"
PT_ACTS = REPO_ROOT / "activations" / "ood_pt"
IT_ACTS = REPO_ROOT / "activations" / "ood"

CONDITIONS = [
    ("PT probe / PT acts", PT_PROBE, PT_ACTS),
    ("PT probe / IT acts", PT_PROBE, IT_ACTS),
    ("IT probe / PT acts", IT_PROBE, PT_ACTS),
]


def summarize(name: str, cond_name: str, behavioral: np.ndarray, probe: np.ndarray, labels: np.ndarray) -> dict:
    if len(behavioral) == 0:
        print(f"  {name} [{cond_name}]: NO DATA")
        return {}
    stats = correlate_deltas(behavioral, probe)
    per_cond = per_condition_correlations(behavioral, probe, labels)
    print(f"  {name} [{cond_name}]: n={stats['n']}, r={stats['pearson_r']:.3f} "
          f"(perm_p={stats['permutation_p']:.3f}), sign={stats['sign_agreement']:.1%} (n={stats['sign_n']})")
    return {
        "condition": cond_name,
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


def run_exp(exp_name: str, rates: dict[str, dict[str, float]], acts_subdir: str) -> dict:
    print(f"\n--- {exp_name} ---")
    results = {}
    for cond_name, probe_path, acts_base in CONDITIONS:
        acts_dir = acts_base / acts_subdir
        if not acts_dir.exists():
            print(f"  {exp_name} [{cond_name}]: MISSING acts dir {acts_dir}")
            continue
        try:
            b, p, labels = compute_deltas(rates, acts_dir, probe_path, LAYER)
            results[cond_name] = summarize(exp_name, cond_name, b, p, labels)
        except FileNotFoundError as e:
            print(f"  {exp_name} [{cond_name}]: MISSING — {e}")
        except Exception as e:
            print(f"  {exp_name} [{cond_name}]: ERROR — {e}")
            import traceback
            traceback.print_exc()
    return results


def exp1a_analysis() -> dict:
    pairwise = json.load(open(RESULTS_OOD / "category_preference" / "pairwise.json"))
    rates = compute_p_choose_from_pairwise(pairwise["results"])
    return run_exp("exp1a", rates, "exp1_category")


def exp1b_analysis() -> dict:
    pairwise = json.load(open(RESULTS_OOD / "hidden_preference" / "pairwise.json"))
    rates = compute_p_choose_from_pairwise(pairwise["results"])
    targeted_rates = {k: v for k, v in rates.items() if not k.startswith("compete_")}
    targeted_rates = {
        k: {tid: v for tid, v in rate_dict.items() if tid.startswith("hidden_")}
        for k, rate_dict in targeted_rates.items()
    }
    return run_exp("exp1b", targeted_rates, "exp1_prompts")


def exp1c_analysis() -> dict:
    pairwise = json.load(open(RESULTS_OOD / "crossed_preference" / "pairwise.json"))
    rates = compute_p_choose_from_pairwise(pairwise["results"])
    targeted_rates = {k: v for k, v in rates.items() if not k.startswith("compete_")}
    targeted_rates = {
        k: {tid: v for tid, v in rate_dict.items() if tid.startswith("crossed_")}
        for k, rate_dict in targeted_rates.items()
    }
    return run_exp("exp1c", targeted_rates, "exp1_prompts")


def _parse_competing_condition(cid: str) -> tuple[str, str, str]:
    parts = cid.split("_")
    direction = parts[-1]
    shell = parts[-2]
    topic = "_".join(parts[1:-2])
    return topic, shell, direction


def exp1d_analysis() -> dict:
    pairwise = json.load(open(RESULTS_OOD / "crossed_preference" / "pairwise.json"))
    rates = compute_p_choose_from_pairwise(pairwise["results"])

    # Full competing rates (all crossed tasks)
    competing_rates = {k: v for k, v in rates.items()
                      if k.startswith("compete_") or k == "baseline"}
    competing_rates = {
        k: {tid: v for tid, v in rate_dict.items() if tid.startswith("crossed_")}
        for k, rate_dict in competing_rates.items()
    }

    print(f"\n--- exp1d ---")
    results = {}
    for cond_name, probe_path, acts_base in CONDITIONS:
        acts_dir = acts_base / "exp1_prompts"
        if not acts_dir.exists():
            print(f"  exp1d [{cond_name}]: MISSING acts dir {acts_dir}")
            continue
        try:
            b, p, labels = compute_deltas(competing_rates, acts_dir, probe_path, LAYER)
            results[cond_name] = summarize("exp1d_full", cond_name, b, p, labels)
        except FileNotFoundError as e:
            print(f"  exp1d [{cond_name}]: MISSING — {e}")
        except Exception as e:
            print(f"  exp1d [{cond_name}]: ERROR — {e}")
            import traceback
            traceback.print_exc()
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="all",
                       choices=["all", "exp1a", "exp1b", "exp1c", "exp1d"])
    args = parser.parse_args()

    exp_fns = {
        "exp1a": exp1a_analysis,
        "exp1b": exp1b_analysis,
        "exp1c": exp1c_analysis,
        "exp1d": exp1d_analysis,
    }

    exps_to_run = list(exp_fns.keys()) if args.exp == "all" else [args.exp]
    results = {}

    for exp_name in exps_to_run:
        results[exp_name] = exp_fns[exp_name]()

    output_dir = REPO_ROOT / "experiments" / "ood_pt"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save summary (without large arrays)
    _large_keys = {"behavioral_deltas", "probe_deltas", "condition_labels"}
    results_summary = {}
    for exp_name, cond_results in results.items():
        results_summary[exp_name] = {}
        for cond_name, res in cond_results.items():
            if not res:
                continue
            results_summary[exp_name][cond_name] = {k: v for k, v in res.items() if k not in _large_keys}

    with open(output_dir / "analysis_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSaved summary to {output_dir / 'analysis_results.json'}")

    # Save full results
    with open(output_dir / "analysis_results_full.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved full results to {output_dir / 'analysis_results_full.json'}")


if __name__ == "__main__":
    main()
