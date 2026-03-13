"""Analyze 10-prompt lying experiment.

Scores activations from both extraction sets (assistant-turn and user turn-boundary)
with tb-2, tb-5, and task_mean probes. Outputs a results JSON with Cohen's d and AUC
for each prompt × selector × probe × layer × followup combination.
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_ind
from sklearn.metrics import roc_auc_score

from src.probes.core.activations import load_activations
from src.probes.core.evaluate import score_with_probe

ROOT = Path(__file__).resolve().parents[2]

ASSISTANT_ACT_DIR = ROOT / "activations" / "gemma_3_27b_lying_10prompt_assistant"
USER_TB_ACT_DIR = ROOT / "activations" / "gemma_3_27b_lying_10prompt_user_tb"

PROBES_DIR = ROOT / "results" / "probes"
OUTPUT_DIR = ROOT / "experiments" / "truth_probes" / "error_prefill" / "lying_prompts"

PROBES = {
    "tb-2": PROBES_DIR / "heldout_eval_gemma3_tb-2" / "probes",
    "tb-5": PROBES_DIR / "heldout_eval_gemma3_tb-5" / "probes",
    "task_mean": PROBES_DIR / "heldout_eval_gemma3_task_mean" / "probes",
}

LAYERS = [25, 32, 39, 46, 53]

ASSISTANT_SELECTORS = [
    "assistant_mean", "assistant_tb:-1", "assistant_tb:-2",
    "assistant_tb:-3", "assistant_tb:-4", "assistant_tb:-5",
]
USER_TB_SELECTORS = ["turn_boundary:-2", "turn_boundary:-5"]

PROMPTS_PATH = ROOT / "data" / "creak" / "lying_system_prompts.json"


def parse_task_id(task_id: str) -> dict:
    """Parse task ID into components.

    Format: {ex_id}_{answer_condition}_{prompt_name}_{followup_type}
    Ex: train_1234_correct_lie_direct_minimal
    """
    parts = task_id.split("_")
    ex_id = f"{parts[0]}_{parts[1]}"
    answer_condition = parts[2]

    # prompt_name can be multi-word (e.g., lie_direct, direct_please_lie, roleplay_villain)
    # followup_type is always the last token: minimal, neutral, presupposes, challenge
    followup_type = parts[-1]
    prompt_name = "_".join(parts[3:-1])

    return {
        "ex_id": ex_id,
        "answer_condition": answer_condition,
        "prompt_name": prompt_name,
        "followup_type": followup_type,
    }


def compute_metrics(correct_scores: np.ndarray, incorrect_scores: np.ndarray) -> dict:
    if len(correct_scores) < 2 or len(incorrect_scores) < 2:
        return {"cohens_d": float("nan"), "auc": float("nan"), "n_correct": len(correct_scores), "n_incorrect": len(incorrect_scores)}

    mean_diff = correct_scores.mean() - incorrect_scores.mean()
    pooled_std = np.sqrt(
        (correct_scores.var(ddof=1) * (len(correct_scores) - 1)
         + incorrect_scores.var(ddof=1) * (len(incorrect_scores) - 1))
        / (len(correct_scores) + len(incorrect_scores) - 2)
    )
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0

    labels = np.array([1] * len(correct_scores) + [0] * len(incorrect_scores))
    all_scores = np.concatenate([correct_scores, incorrect_scores])
    auc = roc_auc_score(labels, all_scores)

    return {
        "cohens_d": float(cohens_d),
        "auc": float(auc),
        "n_correct": len(correct_scores),
        "n_incorrect": len(incorrect_scores),
    }


def analyze_set(act_dir: Path, selectors: list[str]) -> dict:
    results = {}

    for selector in selectors:
        act_path = act_dir / f"activations_{selector}.npz"
        if not act_path.exists():
            print(f"  SKIP {selector}: not found")
            continue

        task_ids, layer_acts = load_activations(act_path, layers=LAYERS)
        parsed = [parse_task_id(tid) for tid in task_ids]

        # Get unique prompts and followup types
        prompts = sorted(set(p["prompt_name"] for p in parsed))
        followups = sorted(set(p["followup_type"] for p in parsed))

        results[selector] = {}
        for probe_name, probe_dir in PROBES.items():
            results[selector][probe_name] = {}

            for prompt in prompts:
                results[selector][probe_name][prompt] = {}

                for followup in followups:
                    correct_mask = np.array([
                        p["answer_condition"] == "correct"
                        and p["prompt_name"] == prompt
                        and p["followup_type"] == followup
                        for p in parsed
                    ])
                    incorrect_mask = np.array([
                        p["answer_condition"] == "incorrect"
                        and p["prompt_name"] == prompt
                        and p["followup_type"] == followup
                        for p in parsed
                    ])

                    if correct_mask.sum() == 0:
                        continue

                    results[selector][probe_name][prompt][followup] = {}
                    for layer in LAYERS:
                        probe_path = probe_dir / f"probe_ridge_L{layer}.npy"
                        probe_weights = np.load(probe_path)
                        scores = score_with_probe(probe_weights, layer_acts[layer])

                        metrics = compute_metrics(scores[correct_mask], scores[incorrect_mask])
                        results[selector][probe_name][prompt][followup][str(layer)] = metrics

                        d = metrics["cohens_d"]
                        print(
                            f"  {selector:20s} | {probe_name:9s} | {prompt:25s} | {followup:12s} | L{layer} | d={d:+.2f}"
                        )

    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load prompt type mapping
    with open(PROMPTS_PATH) as f:
        prompt_groups = json.load(f)
    prompt_type_map = {}
    for group_type, group in prompt_groups.items():
        for name in group:
            prompt_type_map[name] = group_type

    print("=" * 80)
    print("Assistant-turn selectors (20k conversations, minimal follow-up)")
    print("=" * 80)
    assistant_results = analyze_set(ASSISTANT_ACT_DIR, ASSISTANT_SELECTORS)

    print("\n" + "=" * 80)
    print("User turn-boundary selectors (60k conversations, 3 follow-up types)")
    print("=" * 80)
    user_tb_results = analyze_set(USER_TB_ACT_DIR, USER_TB_SELECTORS)

    combined = {
        "assistant_selectors": assistant_results,
        "user_tb_selectors": user_tb_results,
        "prompt_type_map": prompt_type_map,
    }

    out_path = OUTPUT_DIR / "lying_10prompt_results.json"
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
