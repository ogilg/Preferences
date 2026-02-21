"""Analyze OOD system prompt experiments: score with probe, compute deltas, correlate.

Usage: python scripts/ood_system_prompts/analyze_ood.py [--exp EXP] [--layer LAYER]

EXP: all, exp1a, exp1b, exp1c, exp1d, exp2, exp3
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from src.ood.analysis import (
    compute_p_choose_from_pairwise,
    compute_deltas,
    correlate_deltas,
    per_condition_correlations,
    _build_rate_lookup,
)

REPO_ROOT = Path(__file__).parent.parent.parent
ACTS_DIR = REPO_ROOT / "activations" / "ood"
PROBE_DIR = REPO_ROOT / "results" / "probes" / "gemma3_10k_heldout_std_demean" / "probes"
RESULTS_OOD = REPO_ROOT / "results" / "ood"
LAYERS = [31, 43, 55]


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
    """Exp 1a: Category preference."""
    pairwise = json.load(open(RESULTS_OOD / "category_preference" / "pairwise.json"))
    rates = compute_p_choose_from_pairwise(pairwise["results"])

    acts_dir = ACTS_DIR / "exp1_category"
    b, p, labels = compute_deltas(rates, acts_dir, probe_path(layer), layer)
    return summarize("exp1a", layer, b, p, labels)


def exp1b_analysis(layer: int) -> dict:
    """Exp 1b: Hidden preference (targeted conditions on hidden tasks)."""
    pairwise = json.load(open(RESULTS_OOD / "hidden_preference" / "pairwise.json"))
    rates = compute_p_choose_from_pairwise(pairwise["results"])

    # Filter to targeted conditions only (exclude competing)
    targeted_rates = {k: v for k, v in rates.items() if not k.startswith("compete_")}

    # Focus on hidden tasks (hidden_*) — the novel OOD topics
    targeted_rates = {
        k: {tid: v for tid, v in rate_dict.items() if tid.startswith("hidden_")}
        for k, rate_dict in targeted_rates.items()
    }

    acts_dir = ACTS_DIR / "exp1_prompts"
    b, p, labels = compute_deltas(targeted_rates, acts_dir, probe_path(layer), layer)
    return summarize("exp1b", layer, b, p, labels)


def exp1c_analysis(layer: int) -> dict:
    """Exp 1c: Crossed preference (targeted conditions on crossed tasks)."""
    pairwise = json.load(open(RESULTS_OOD / "crossed_preference" / "pairwise.json"))
    rates = compute_p_choose_from_pairwise(pairwise["results"])

    # Filter to targeted conditions only
    targeted_rates = {k: v for k, v in rates.items() if not k.startswith("compete_")}

    # Focus on crossed tasks (crossed_*)
    crossed_task_ids = set(t for rate_dict in targeted_rates.values() for t in rate_dict.keys()
                          if t.startswith("crossed_"))
    if crossed_task_ids:
        targeted_rates = {
            k: {tid: v for tid, v in rate_dict.items() if tid.startswith("crossed_")}
            for k, rate_dict in targeted_rates.items()
        }

    acts_dir = ACTS_DIR / "exp1_prompts"
    b, p, labels = compute_deltas(targeted_rates, acts_dir, probe_path(layer), layer)
    return summarize("exp1c", layer, b, p, labels)


def _parse_competing_condition(cid: str) -> tuple[str, str, str]:
    """Parse compete_{topic}_{shell}_{direction} -> (topic, shell, direction)."""
    parts = cid.split("_")
    direction = parts[-1]  # topicpos or shellpos
    shell = parts[-2]
    topic = "_".join(parts[1:-2])
    return topic, shell, direction


def exp1d_analysis(layer: int) -> dict:
    """Exp 1d: Competing preference — on-target pairs only (20 pairs × 2 directions = 40 points)."""
    pairwise = json.load(open(RESULTS_OOD / "crossed_preference" / "pairwise.json"))
    rates = compute_p_choose_from_pairwise(pairwise["results"])

    # Build on-target only rates: for each competing condition, only the matched crossed task
    # compete_{topic}_{shell}_{direction} -> crossed_{topic}_{shell}
    compete_cids = [k for k in rates if k.startswith("compete_")]
    on_target_rates: dict[str, dict[str, float]] = {"baseline": {}}

    for cid in compete_cids:
        topic, shell, direction = _parse_competing_condition(cid)
        target_task = f"crossed_{topic}_{shell}"
        if target_task not in rates.get(cid, {}):
            continue
        on_target_rates[cid] = {target_task: rates[cid][target_task]}
        # Add baseline for this task
        if target_task in rates.get("baseline", {}):
            on_target_rates["baseline"][target_task] = rates["baseline"][target_task]

    acts_dir = ACTS_DIR / "exp1_prompts"
    b, p, labels = compute_deltas(on_target_rates, acts_dir, probe_path(layer), layer)
    result = summarize("exp1d_ontarget", layer, b, p, labels)

    # Also run full analysis (all 1600 pairs including off-target)
    competing_rates = {k: v for k, v in rates.items()
                      if k.startswith("compete_") or k == "baseline"}
    competing_rates = {
        k: {tid: v for tid, v in rate_dict.items() if tid.startswith("crossed_")}
        for k, rate_dict in competing_rates.items()
    }
    b_full, p_full, labels_full = compute_deltas(competing_rates, acts_dir, probe_path(layer), layer)
    result_full = summarize("exp1d_full", layer, b_full, p_full, labels_full)

    return {"on_target": result, "full": result_full}


def _load_behavioral_json_rates(path: Path) -> dict[str, dict[str, float]]:
    """Load behavioral.json (aggregated format) → {condition_id: {task_id: p_choose}}."""
    data = json.load(open(path))
    rates = {}
    for cid, cond_data in data["conditions"].items():
        task_rates = cond_data["task_rates"]
        rates[cid] = {tid: v["p_choose"] for tid, v in task_rates.items()}
    return rates


def exp2_analysis(layer: int) -> dict:
    """Exp 2: Role-induced preferences."""
    # Merge role_playing and narrow_preference behavioral data
    rp_rates = _load_behavioral_json_rates(RESULTS_OOD / "role_playing" / "behavioral.json")
    np_rates = _load_behavioral_json_rates(RESULTS_OOD / "narrow_preference" / "behavioral.json")

    # Merge (both have baseline — use role_playing's baseline)
    rates = {**rp_rates, **np_rates}
    rates["baseline"] = rp_rates["baseline"]  # keep one baseline

    acts_dir = ACTS_DIR / "exp2_roles"
    b, p, labels = compute_deltas(rates, acts_dir, probe_path(layer), layer)
    return summarize("exp2", layer, b, p, labels)


def exp3_analysis(layer: int) -> dict:
    """Exp 3: Minimal pairs."""
    rates = _load_behavioral_json_rates(RESULTS_OOD / "minimal_pairs_v7" / "behavioral.json")

    # Subsample: 2 base roles × 10 targets × 2 versions + baseline
    mp_cfg = json.load(open(REPO_ROOT / "configs/ood/prompts/minimal_pairs_v7.json"))
    selected_roles = {"midwest", "brooklyn"}
    selected_versions = {"A", "B"}
    selected_cids = {
        c["condition_id"]
        for c in mp_cfg["conditions"]
        if c["base_role"] in selected_roles and c["version"] in selected_versions
    }
    selected_cids.add("baseline")
    rates = {k: v for k, v in rates.items() if k in selected_cids}

    acts_dir = ACTS_DIR / "exp3_minimal_pairs"
    b, p, labels = compute_deltas(rates, acts_dir, probe_path(layer), layer)
    return summarize("exp3", layer, b, p, labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="all",
                       choices=["all", "exp1a", "exp1b", "exp1c", "exp1d", "exp2", "exp3"])
    parser.add_argument("--layer", type=int, default=None, help="Single layer (default: all)")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    layers = [args.layer] if args.layer else LAYERS
    results = {}

    exp_fns = {
        "exp1a": exp1a_analysis,
        "exp1b": exp1b_analysis,
        "exp1c": exp1c_analysis,
        "exp1d": exp1d_analysis,
        "exp2": exp2_analysis,
        "exp3": exp3_analysis,
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

    # Save results
    output_path = args.output or REPO_ROOT / "experiments" / "ood_system_prompts" / "analysis_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _large_keys = {"behavioral_deltas", "probe_deltas", "condition_labels"}

    def _summarize_res(res: dict) -> dict:
        return {k: v for k, v in res.items() if k not in _large_keys}

    # Don't serialize full delta arrays to JSON by default — too large
    results_summary = {}
    for exp_name, layer_results in results.items():
        results_summary[exp_name] = {}
        for layer_key, res in layer_results.items():
            if not res:
                continue
            # exp1d returns {"on_target": ..., "full": ...}
            if "on_target" in res and "full" in res:
                results_summary[exp_name][layer_key] = {
                    "on_target": _summarize_res(res["on_target"]),
                    "full": _summarize_res(res["full"]),
                }
            else:
                results_summary[exp_name][layer_key] = _summarize_res(res)

    with open(output_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSaved results to {output_path}")

    # Save full results (with delta arrays) for plotting
    full_output = output_path.parent / "analysis_results_full.json"
    with open(full_output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved full results to {full_output}")


if __name__ == "__main__":
    main()
