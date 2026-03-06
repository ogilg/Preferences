"""Ground truth analysis for OOD experiments with EOT probes.

Same logic as scripts/ood_system_prompts/analyze_ground_truth.py but using
EOT activations and EOT-trained probe.

Usage: python scripts/ood_eot/analyze_ground_truth_eot.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ACTS_DIR = REPO_ROOT / "activations" / "ood_eot"
PROBE_DIR = REPO_ROOT / "results" / "probes" / "heldout_eval_gemma3_eot" / "probes"
RESULTS_OOD = REPO_ROOT / "results" / "ood"
CONFIGS = REPO_ROOT / "configs" / "ood"
ACTS_FILENAME = "activations_eot.npz"
LAYER = 31

from scripts.ood_system_prompts.analyze_ground_truth import (
    _ground_truth_exp1a,
    _ground_truth_exp1b,
    _ground_truth_exp1c,
    _ground_truth_exp1d,
    analyze_experiment,
)
from src.ood.analysis import _split_probe, _score_activations, compute_p_choose_from_pairwise


def probe_path(layer: int) -> Path:
    return PROBE_DIR / f"probe_ridge_L{layer}.npy"


def recompute_with_task_ids(
    rates: dict[str, dict[str, float]],
    acts_dir: Path,
    layer: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    weights, bias = _split_probe(probe_path(layer))
    baseline_rates = rates["baseline"]
    baseline_npz = acts_dir / "baseline" / ACTS_FILENAME
    baseline_scores = _score_activations(baseline_npz, layer, weights, bias)

    all_behavioral, all_probe, all_labels, all_task_ids = [], [], [], []

    for cid in rates:
        if cid == "baseline":
            continue
        cond_npz = acts_dir / cid / ACTS_FILENAME
        if not cond_npz.exists():
            continue
        cond_scores = _score_activations(cond_npz, layer, weights, bias)

        for tid, cond_rate in rates[cid].items():
            if tid not in baseline_rates or tid not in baseline_scores or tid not in cond_scores:
                continue
            all_behavioral.append(cond_rate - baseline_rates[tid])
            all_probe.append(cond_scores[tid] - baseline_scores[tid])
            all_labels.append(cid)
            all_task_ids.append(tid)

    return (
        np.array(all_behavioral),
        np.array(all_probe),
        np.array(all_labels),
        all_task_ids,
    )


def recompute_experiment(key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Recompute beh, probe, labels, per_point_gt for one experiment."""
    gt_fns = {
        "exp1a": _ground_truth_exp1a,
        "exp1b": _ground_truth_exp1b,
        "exp1c": _ground_truth_exp1c,
        "exp1d": _ground_truth_exp1d,
    }

    if key == "exp1a":
        pairwise = json.load(open(RESULTS_OOD / "category_preference" / "pairwise.json"))
        rates = compute_p_choose_from_pairwise(pairwise["results"])
        acts_dir = ACTS_DIR / "exp1_category"
    elif key == "exp1b":
        pairwise = json.load(open(RESULTS_OOD / "hidden_preference" / "pairwise.json"))
        rates = compute_p_choose_from_pairwise(pairwise["results"])
        rates = {k: v for k, v in rates.items() if not k.startswith("compete_")}
        rates = {k: {tid: v for tid, v in rd.items() if tid.startswith("hidden_")} for k, rd in rates.items()}
        acts_dir = ACTS_DIR / "exp1_prompts"
    elif key == "exp1c":
        pairwise = json.load(open(RESULTS_OOD / "crossed_preference" / "pairwise.json"))
        rates = compute_p_choose_from_pairwise(pairwise["results"])
        rates = {k: v for k, v in rates.items() if not k.startswith("compete_")}
        rates = {k: {tid: v for tid, v in rd.items() if tid.startswith("crossed_")} for k, rd in rates.items()}
        acts_dir = ACTS_DIR / "exp1_prompts"
    elif key == "exp1d":
        pairwise = json.load(open(RESULTS_OOD / "crossed_preference" / "pairwise.json"))
        rates = compute_p_choose_from_pairwise(pairwise["results"])
        rates = {k: v for k, v in rates.items() if k.startswith("compete_") or k == "baseline"}
        rates = {k: {tid: v for tid, v in rd.items() if tid.startswith("crossed_")} for k, rd in rates.items()}
        acts_dir = ACTS_DIR / "exp1_prompts"
    else:
        raise ValueError(f"Unknown key: {key}")

    beh, probe, labels, task_ids = recompute_with_task_ids(rates, acts_dir, LAYER)
    per_point_gt = gt_fns[key](labels, task_ids)
    return beh, probe, labels, per_point_gt


def main() -> None:
    results = {}

    for exp_name in ["exp1a", "exp1b", "exp1c", "exp1d"]:
        print(f"\n--- {exp_name} ---")
        try:
            beh, probe, labels, gt = recompute_experiment(exp_name)
            results[exp_name] = analyze_experiment(exp_name, beh, probe, gt, LAYER)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    output_path = REPO_ROOT / "experiments" / "ood_eot" / "ground_truth_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
