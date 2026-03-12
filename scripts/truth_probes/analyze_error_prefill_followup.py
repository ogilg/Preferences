"""Follow-up analysis: assistant-turn selectors + lying system prompts.

Scores new activations (assistant selectors from Run A, all selectors from Run B)
with existing preference probes. Compares to baseline results from the first run.
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_ind
from sklearn.metrics import roc_auc_score

from src.probes.core.activations import load_activations
from src.probes.core.evaluate import score_with_probe

ROOT = Path(__file__).resolve().parents[2]

# Activation directories
ORIG_ACT_DIR = ROOT / "activations" / "gemma_3_27b_error_prefill"
LYING_ACT_DIR = ROOT / "activations" / "gemma_3_27b_lying_prefill"

PROBES_DIR = ROOT / "results" / "probes"
OUTPUT_DIR = ROOT / "experiments" / "truth_probes" / "error_prefill"
ASSETS_DIR = OUTPUT_DIR / "assets"

PROBES = {
    "tb-2": PROBES_DIR / "heldout_eval_gemma3_tb-2" / "probes",
    "tb-5": PROBES_DIR / "heldout_eval_gemma3_tb-5" / "probes",
    "task_mean": PROBES_DIR / "heldout_eval_gemma3_task_mean" / "probes",
}

LAYERS = [25, 32, 39, 46, 53]

# Original follow-up types (no "none" — excluded from first run too)
ORIG_FOLLOWUP_TYPES = ["neutral", "presupposes", "challenge", "same_domain", "control"]

# Lying experiment follow-up types
LYING_FOLLOWUP_TYPES = ["neutral", "presupposes"]
SYSTEM_PROMPT_TYPES = ["lie_direct", "lie_roleplay"]

# All selectors we have activations for
ASSISTANT_SELECTORS = ["assistant_mean", "assistant_tb:-1", "assistant_tb:-2", "assistant_tb:-3", "assistant_tb:-4", "assistant_tb:-5"]
TB_SELECTORS = ["turn_boundary:-2", "turn_boundary:-5"]


def parse_orig_task_id(task_id: str) -> tuple[str, str, str]:
    """Parse 'train_1234_correct_neutral' -> ('train_1234', 'correct', 'neutral')."""
    parts = task_id.split("_")
    ex_id = f"{parts[0]}_{parts[1]}"
    answer_condition = parts[2]
    followup_type = "_".join(parts[3:])
    return ex_id, answer_condition, followup_type


def parse_lying_task_id(task_id: str) -> tuple[str, str, str, str]:
    """Parse 'train_1234_correct_lie_direct_neutral'
    -> ('train_1234', 'correct', 'lie_direct', 'neutral')."""
    parts = task_id.split("_")
    ex_id = f"{parts[0]}_{parts[1]}"
    answer_condition = parts[2]
    # system_prompt_type is 'lie_direct' or 'lie_roleplay' (2 tokens)
    sys_prompt = f"{parts[3]}_{parts[4]}"
    followup_type = "_".join(parts[5:])
    return ex_id, answer_condition, sys_prompt, followup_type


def compute_metrics(correct_scores: np.ndarray, incorrect_scores: np.ndarray) -> dict:
    mean_diff = correct_scores.mean() - incorrect_scores.mean()
    pooled_std = np.sqrt(
        (correct_scores.var(ddof=1) * (len(correct_scores) - 1)
         + incorrect_scores.var(ddof=1) * (len(incorrect_scores) - 1))
        / (len(correct_scores) + len(incorrect_scores) - 2)
    )
    cohens_d = mean_diff / pooled_std
    _, p_value = ttest_ind(correct_scores, incorrect_scores, equal_var=False)

    labels = np.array([1] * len(correct_scores) + [0] * len(incorrect_scores))
    all_scores = np.concatenate([correct_scores, incorrect_scores])
    auc = roc_auc_score(labels, all_scores)

    return {
        "mean_correct": float(correct_scores.mean()),
        "mean_incorrect": float(incorrect_scores.mean()),
        "mean_diff": float(mean_diff),
        "cohens_d": float(cohens_d),
        "p_value": float(p_value),
        "auc": float(auc),
        "n_correct": len(correct_scores),
        "n_incorrect": len(incorrect_scores),
    }


def analyze_assistant_selectors() -> dict:
    """Run A: assistant selectors on original (no lying) conversations."""
    print("=" * 80)
    print("Part 1: Assistant-turn selectors (original conversations, no system prompt)")
    print("=" * 80)

    results = {}

    for selector in ASSISTANT_SELECTORS:
        act_path = ORIG_ACT_DIR / f"activations_{selector}.npz"
        if not act_path.exists():
            print(f"  SKIP {selector}: {act_path} not found")
            continue

        act_task_ids, layer_acts = load_activations(act_path, layers=LAYERS)
        parsed = [parse_orig_task_id(tid) for tid in act_task_ids]

        results[selector] = {}
        for probe_name, probe_dir in PROBES.items():
            results[selector][probe_name] = {}

            for followup_type in ORIG_FOLLOWUP_TYPES:
                correct_mask = np.array([
                    ac == "correct" and ft == followup_type
                    for _, ac, ft in parsed
                ])
                incorrect_mask = np.array([
                    ac == "incorrect" and ft == followup_type
                    for _, ac, ft in parsed
                ])

                if correct_mask.sum() == 0 or incorrect_mask.sum() == 0:
                    continue

                results[selector][probe_name][followup_type] = {}
                for layer in LAYERS:
                    probe_path = probe_dir / f"probe_ridge_L{layer}.npy"
                    probe_weights = np.load(probe_path)
                    scores = score_with_probe(probe_weights, layer_acts[layer])

                    metrics = compute_metrics(scores[correct_mask], scores[incorrect_mask])
                    results[selector][probe_name][followup_type][str(layer)] = metrics

                    print(
                        f"  {selector:20s} | {probe_name} | {followup_type:15s} | L{layer:02d} | "
                        f"d={metrics['cohens_d']:+.4f} | AUC={metrics['auc']:.3f}"
                    )
            print()

    return results


def analyze_lying_conversations() -> dict:
    """Run B: all selectors on lying conversations."""
    print("=" * 80)
    print("Part 2: Lying system prompts")
    print("=" * 80)

    all_selectors = TB_SELECTORS + ASSISTANT_SELECTORS
    results = {}

    for selector in all_selectors:
        act_path = LYING_ACT_DIR / f"activations_{selector}.npz"
        if not act_path.exists():
            print(f"  SKIP {selector}: {act_path} not found")
            continue

        act_task_ids, layer_acts = load_activations(act_path, layers=LAYERS)
        parsed = [parse_lying_task_id(tid) for tid in act_task_ids]

        results[selector] = {}
        for probe_name, probe_dir in PROBES.items():
            results[selector][probe_name] = {}

            for sys_prompt in SYSTEM_PROMPT_TYPES:
                results[selector][probe_name][sys_prompt] = {}

                for followup_type in LYING_FOLLOWUP_TYPES:
                    correct_mask = np.array([
                        ac == "correct" and sp == sys_prompt and ft == followup_type
                        for _, ac, sp, ft in parsed
                    ])
                    incorrect_mask = np.array([
                        ac == "incorrect" and sp == sys_prompt and ft == followup_type
                        for _, ac, sp, ft in parsed
                    ])

                    results[selector][probe_name][sys_prompt][followup_type] = {}
                    for layer in LAYERS:
                        probe_path = probe_dir / f"probe_ridge_L{layer}.npy"
                        probe_weights = np.load(probe_path)
                        scores = score_with_probe(probe_weights, layer_acts[layer])

                        metrics = compute_metrics(scores[correct_mask], scores[incorrect_mask])
                        results[selector][probe_name][sys_prompt][followup_type][str(layer)] = metrics

                        print(
                            f"  {selector:20s} | {probe_name} | {sys_prompt:15s} | {followup_type:15s} | "
                            f"L{layer:02d} | d={metrics['cohens_d']:+.4f} | AUC={metrics['auc']:.3f}"
                        )
                print()

    return results


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    assistant_results = analyze_assistant_selectors()
    lying_results = analyze_lying_conversations()

    combined = {
        "assistant_selectors_no_lying": assistant_results,
        "lying_conversations": lying_results,
    }

    out_path = OUTPUT_DIR / "error_prefill_followup_results.json"
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
