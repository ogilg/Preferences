"""Analyze probe performance on on-target vs off-target tasks per condition."""

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from scipy import stats
from dotenv import load_dotenv

load_dotenv()

from scripts.utility_fitting.analyze_ood import (
    load_thurstonian_latest,
    load_activations_for_condition,
    score_with_probe,
    pairwise_accuracy,
)


def cross_group_pairwise_accuracy(
    on_scores: np.ndarray,
    off_scores: np.ndarray,
    on_utils: np.ndarray,
    off_utils: np.ndarray,
) -> float:
    """Fraction of (on-target, off-target) pairs ranked correctly by probe."""
    correct = 0
    total = 0
    for i in range(len(on_scores)):
        for j in range(len(off_scores)):
            if on_utils[i] == off_utils[j]:
                continue
            total += 1
            if (on_scores[i] - off_scores[j]) * (on_utils[i] - off_utils[j]) > 0:
                correct += 1
    return correct / total if total > 0 else float("nan")


def extract_topic_from_condition(cond_name: str) -> str:
    """Extract the topic from a condition name like 'cheese_pos_persona'."""
    for suffix in ["_pos_persona", "_neg_persona"]:
        if cond_name.endswith(suffix):
            return cond_name[: -len(suffix)]
    return ""


def extract_polarity(cond_name: str) -> str:
    if "_pos_" in cond_name:
        return "pos"
    if "_neg_" in cond_name:
        return "neg"
    return ""


def analyze_on_target(
    exp_name: str,
    config_dir: Path,
    results_dir: Path,
    act_dir_base: Path,
    run_prefix: str,
    task_prefix: str,
    probe_weights: np.ndarray,
    layer: int,
) -> list[dict]:
    print(f"\n{'='*60}")
    print(f"On-target analysis: {exp_name}")
    print(f"{'='*60}")

    results = []
    for config_file in sorted(config_dir.glob("*.yaml")):
        cond_name = config_file.stem
        if cond_name == "baseline":
            continue

        topic = extract_topic_from_condition(cond_name)
        polarity = extract_polarity(cond_name)
        if not topic or not polarity:
            continue

        # Map to result dir
        cfg = yaml.safe_load(config_file.read_text())
        sp = cfg.get("measurement_system_prompt", "")
        if not sp:
            continue
        h = hashlib.sha256(sp.encode()).hexdigest()[:8]
        result_dir = results_dir / f"{run_prefix}_sys{h}"
        if not result_dir.exists():
            continue

        # Load utilities
        cond_utils = load_thurstonian_latest(result_dir)

        # Load activations
        act_dir = act_dir_base / cond_name
        if not act_dir.exists():
            continue

        task_filter = [t for t in cond_utils if t.startswith(task_prefix)]
        if not task_filter:
            task_filter = list(cond_utils.keys())

        acts, task_ids = load_activations_for_condition(act_dir, layer, task_filter)
        act_idx = {t: i for i, t in enumerate(task_ids)}

        # Split into on-target and off-target
        shared = [t for t in task_ids if t in cond_utils]
        on_target = [t for t in shared if topic in t]
        off_target = [t for t in shared if topic not in t]

        if len(on_target) < 3 or len(off_target) < 3:
            continue

        on_acts = np.array([acts[act_idx[t]] for t in on_target])
        off_acts = np.array([acts[act_idx[t]] for t in off_target])
        on_utils = np.array([cond_utils[t] for t in on_target])
        off_utils = np.array([cond_utils[t] for t in off_target])
        on_scores = score_with_probe(probe_weights, on_acts)
        off_scores = score_with_probe(probe_weights, off_acts)

        # On-target metrics
        if len(on_target) >= 4:
            on_r, _ = stats.pearsonr(on_scores, on_utils)
            on_acc = pairwise_accuracy(on_scores, on_utils)
        else:
            on_r = float("nan")
            on_acc = float("nan")

        # Off-target metrics
        off_r, _ = stats.pearsonr(off_scores, off_utils)
        off_acc = pairwise_accuracy(off_scores, off_utils)

        # Cross-group pairwise accuracy
        cross_acc = cross_group_pairwise_accuracy(
            on_scores, off_scores, on_utils, off_utils
        )

        # All tasks combined
        all_scores = np.concatenate([on_scores, off_scores])
        all_utils = np.concatenate([on_utils, off_utils])
        all_r, _ = stats.pearsonr(all_scores, all_utils)
        all_acc = pairwise_accuracy(all_scores, all_utils)

        result = {
            "experiment": exp_name,
            "condition": cond_name,
            "topic": topic,
            "polarity": polarity,
            "n_on": len(on_target),
            "n_off": len(off_target),
            "on_r": on_r,
            "on_acc": on_acc,
            "off_r": off_r,
            "off_acc": off_acc,
            "cross_acc": cross_acc,
            "all_r": all_r,
            "all_acc": all_acc,
            "on_mean_score": float(np.mean(on_scores)),
            "off_mean_score": float(np.mean(off_scores)),
            "on_mean_util": float(np.mean(on_utils)),
            "off_mean_util": float(np.mean(off_utils)),
        }
        results.append(result)

        print(
            f"  {cond_name:40s} on={len(on_target)} off={len(off_target)} | "
            f"on r={on_r:.3f} acc={on_acc:.3f} | "
            f"off r={off_r:.3f} acc={off_acc:.3f} | "
            f"cross_acc={cross_acc:.3f} | "
            f"all r={all_r:.3f} acc={all_acc:.3f}"
        )

    return results


def main():
    probe_dir = Path("results/probes/gemma3_10k_heldout_std_raw")
    layer = 31
    probe_weights = np.load(
        probe_dir / "probes" / f"probe_ridge_L{layer:02d}.npy"
    )
    run_prefix = "completion_preference_gemma-3-27b_completion_canonical_seed0"

    all_results = []

    # Exp 1b
    all_results.extend(
        analyze_on_target(
            "exp1b",
            Path("configs/measurement/active_learning/ood_exp1b"),
            Path("results/experiments/ood_exp1b/pre_task_active_learning"),
            Path("activations/ood/exp1_prompts"),
            run_prefix,
            "hidden_",
            probe_weights,
            layer,
        )
    )

    # Exp 1c
    all_results.extend(
        analyze_on_target(
            "exp1c",
            Path("configs/measurement/active_learning/ood_exp1c"),
            Path("results/experiments/ood_exp1c/pre_task_active_learning"),
            Path("activations/ood/exp1_prompts"),
            run_prefix,
            "crossed_",
            probe_weights,
            layer,
        )
    )

    output_path = Path(
        "experiments/ood_system_prompts/utility_fitting/on_target_results.json"
    )
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved {len(all_results)} results to {output_path}")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY (mean across conditions)")
    print(f"{'='*80}")
    for exp in ["exp1b", "exp1c"]:
        exp_results = [r for r in all_results if r["experiment"] == exp]
        if not exp_results:
            continue
        for pol in ["pos", "neg"]:
            pol_results = [r for r in exp_results if r["polarity"] == pol]
            if not pol_results:
                continue
            print(
                f"  {exp} {pol}: "
                f"on r={np.mean([r['on_r'] for r in pol_results]):.3f} "
                f"on acc={np.mean([r['on_acc'] for r in pol_results]):.3f} | "
                f"off r={np.mean([r['off_r'] for r in pol_results]):.3f} "
                f"off acc={np.mean([r['off_acc'] for r in pol_results]):.3f} | "
                f"cross acc={np.mean([r['cross_acc'] for r in pol_results]):.3f} | "
                f"all r={np.mean([r['all_r'] for r in pol_results]):.3f} "
                f"all acc={np.mean([r['all_acc'] for r in pol_results]):.3f}"
            )


if __name__ == "__main__":
    main()
