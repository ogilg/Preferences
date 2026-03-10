"""Compare probes trained on EOT, prompt_last, and concatenated activations.

Uses the same 10k train / 4k eval setup as the EOT heldout eval config.
"""

from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

from src.probes.core.activations import load_activations
from src.probes.data_loading import load_thurstonian_scores, load_eval_data
from src.probes.experiments.hoo_ridge import build_ridge_xy
from src.probes.experiments.run_dir_probes import train_ridge_heldout

TRAIN_RUN_DIR = Path("results/experiments/main_probes/gemma3_10k_run1/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0")
EVAL_RUN_DIR = Path("results/experiments/main_probes/gemma3_4k_pre_task/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0")
PROMPT_LAST_PATH = Path("activations/gemma_3_27b/activations_prompt_last.npz")
EOT_PATH = Path("activations/gemma_3_27b_eot/activations_eot.npz")
LAYER = 31


def main():
    # Load scores
    scores = load_thurstonian_scores(TRAIN_RUN_DIR)
    eval_scores, eval_measurements = load_eval_data(EVAL_RUN_DIR, set(scores.keys()))
    print(f"Train: {len(scores)} tasks, Eval: {len(eval_scores)} tasks")

    all_needed = set(scores.keys()) | set(eval_scores.keys())

    # Load activations
    pl_ids, pl_acts = load_activations(PROMPT_LAST_PATH, task_id_filter=all_needed, layers=[LAYER])
    eot_ids, eot_acts = load_activations(EOT_PATH, task_id_filter=all_needed, layers=[LAYER])

    # Find common task IDs preserving order
    common = set(pl_ids) & set(eot_ids)
    print(f"Common tasks: {len(common)}")

    # Build aligned arrays
    pl_id_to_idx = {tid: i for i, tid in enumerate(pl_ids)}
    eot_id_to_idx = {tid: i for i, tid in enumerate(eot_ids)}

    common_ids = sorted(common)
    pl_indices = [pl_id_to_idx[tid] for tid in common_ids]
    eot_indices = [eot_id_to_idx[tid] for tid in common_ids]

    pl_aligned = pl_acts[LAYER][pl_indices]
    eot_aligned = eot_acts[LAYER][eot_indices]
    concat_aligned = np.concatenate([pl_aligned, eot_aligned], axis=1)

    task_ids_arr = np.array(common_ids)
    print(f"Activation dims: prompt_last={pl_aligned.shape[1]}, eot={eot_aligned.shape[1]}, concat={concat_aligned.shape[1]}")

    # Filter scores to common tasks
    scores_common = {tid: scores[tid] for tid in common_ids if tid in scores}
    eval_scores_common = {tid: eval_scores[tid] for tid in common_ids if tid in eval_scores}

    conditions = [
        ("prompt_last", pl_aligned),
        ("eot", eot_aligned),
        ("concat", concat_aligned),
    ]

    results = {}
    for name, acts in conditions:
        print(f"\n{'='*50}")
        print(f"Training: {name} (dim={acts.shape[1]})")
        print(f"{'='*50}")

        train_indices, y_train = build_ridge_xy(task_ids_arr, scores_common)
        X_train = acts[train_indices]

        metrics = train_ridge_heldout(
            X_train, y_train, acts, task_ids_arr,
            eval_scores_common, eval_measurements, LAYER,
            standardize=True, alpha_sweep_size=20, eval_split_seed=42,
        )

        results[name] = metrics
        print(f"  Best alpha: {metrics['best_alpha']:.4g}")
        print(f"  Sweep r: {metrics['sweep_r']:.4f}")
        print(f"  Final r: {metrics['final_r']:.4f}")
        print(f"  Final acc: {metrics['final_acc']:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print(f"{'Condition':<15} {'Final r':>10} {'Final acc':>10} {'Best alpha':>12}")
    print(f"{'-'*15} {'-'*10} {'-'*10} {'-'*12}")
    for name in ["prompt_last", "eot", "concat"]:
        m = results[name]
        print(f"{name:<15} {m['final_r']:>10.4f} {m['final_acc']:>10.4f} {m['best_alpha']:>12.4g}")


if __name__ == "__main__":
    main()
