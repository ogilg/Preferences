"""Analyze error prefill experiment: does the preference direction separate correct from incorrect model answers?

Scores activations from follow-up conditions with existing preference probes.
Compares correct vs incorrect answer conditions per follow-up type.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind
from sklearn.metrics import roc_auc_score

from src.probes.core.activations import load_activations
from src.probes.core.evaluate import score_with_probe

ROOT = Path(__file__).resolve().parents[2]
ACTIVATIONS_DIR = ROOT / "activations" / "gemma_3_27b_error_prefill"
PROBES_DIR = ROOT / "results" / "probes"
OUTPUT_DIR = ROOT / "experiments" / "truth_probes" / "error_prefill"
ASSETS_DIR = OUTPUT_DIR / "assets"

PROBES = {
    "tb-2": {
        "selector": "turn_boundary:-2",
        "probe_dir": PROBES_DIR / "heldout_eval_gemma3_tb-2" / "probes",
    },
    "tb-5": {
        "selector": "turn_boundary:-5",
        "probe_dir": PROBES_DIR / "heldout_eval_gemma3_tb-5" / "probes",
    },
}

LAYERS = [25, 32, 39, 46, 53]

# Skip "none" — tb selectors read from wrong position without a follow-up turn
FOLLOWUP_TYPES = ["neutral", "presupposes", "challenge", "same_domain", "control"]


def parse_task_id(task_id: str) -> tuple[str, str, str]:
    """Parse 'train_1234_correct_neutral' -> ('train_1234', 'correct', 'neutral')."""
    parts = task_id.split("_")
    # ex_id is first two parts (e.g. train_1234), then answer_condition, then followup_type
    ex_id = f"{parts[0]}_{parts[1]}"
    answer_condition = parts[2]
    followup_type = "_".join(parts[3:])
    return ex_id, answer_condition, followup_type


def compute_metrics(correct_scores: np.ndarray, incorrect_scores: np.ndarray) -> dict:
    mean_diff = correct_scores.mean() - incorrect_scores.mean()
    pooled_std = np.sqrt(
        (correct_scores.var(ddof=1) * (len(correct_scores) - 1)
         + incorrect_scores.var(ddof=1) * (len(incorrect_scores) - 1))
        / (len(correct_scores) + len(incorrect_scores) - 2)
    )
    cohens_d = mean_diff / pooled_std
    _, p_value = ttest_ind(correct_scores, incorrect_scores, equal_var=False)

    # AUC: correct = 1, incorrect = 0
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


def permutation_test(correct_scores: np.ndarray, incorrect_scores: np.ndarray, n_perms: int = 1000) -> dict:
    observed_diff = abs(correct_scores.mean() - incorrect_scores.mean())
    combined = np.concatenate([correct_scores, incorrect_scores])
    n_correct = len(correct_scores)
    rng = np.random.default_rng(42)

    perm_diffs = np.empty(n_perms)
    for i in range(n_perms):
        perm = rng.permutation(combined)
        perm_diffs[i] = abs(perm[:n_correct].mean() - perm[n_correct:].mean())

    p_value = (np.sum(perm_diffs >= observed_diff) + 1) / (n_perms + 1)
    return {
        "perm_p": float(p_value),
        "observed_over_max_perm": float(observed_diff / perm_diffs.max()) if perm_diffs.max() > 0 else float("inf"),
    }


def run_analysis() -> dict:
    all_results = {}

    for probe_name, probe_info in PROBES.items():
        selector = probe_info["selector"]
        act_path = ACTIVATIONS_DIR / f"activations_{selector}.npz"
        act_task_ids, layer_acts = load_activations(act_path, layers=LAYERS)

        # Parse all task IDs
        parsed = [parse_task_id(tid) for tid in act_task_ids]

        all_results[probe_name] = {}

        for followup_type in FOLLOWUP_TYPES:
            all_results[probe_name][followup_type] = {}

            # Build masks for this follow-up type
            correct_mask = np.array([
                ac == "correct" and ft == followup_type
                for _, ac, ft in parsed
            ])
            incorrect_mask = np.array([
                ac == "incorrect" and ft == followup_type
                for _, ac, ft in parsed
            ])

            for layer in LAYERS:
                probe_path = probe_info["probe_dir"] / f"probe_ridge_L{layer}.npy"
                probe_weights = np.load(probe_path)
                scores = score_with_probe(probe_weights, layer_acts[layer])

                correct_scores = scores[correct_mask]
                incorrect_scores = scores[incorrect_mask]
                metrics = compute_metrics(correct_scores, incorrect_scores)

                # Permutation test for best layers
                if layer in [32, 39]:
                    perm = permutation_test(correct_scores, incorrect_scores)
                    metrics.update(perm)

                all_results[probe_name][followup_type][str(layer)] = metrics

                print(
                    f"{probe_name} | {followup_type:15s} | L{layer:02d} | "
                    f"d={metrics['cohens_d']:+.4f} | "
                    f"AUC={metrics['auc']:.3f} | "
                    f"p={metrics['p_value']:.2e} | "
                    f"n={metrics['n_correct']}+{metrics['n_incorrect']}"
                )

        print()

    return all_results


def plot_effect_sizes_by_followup(results: dict):
    """Cohen's d by layer, one panel per probe, lines for each follow-up type."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, probe_name in zip(axes, results):
        for followup_type in FOLLOWUP_TYPES:
            ds = [results[probe_name][followup_type][str(l)]["cohens_d"] for l in LAYERS]
            ax.plot(LAYERS, ds, marker="o", label=followup_type)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Cohen's d" if ax == axes[0] else "")
        ax.set_title(f"{probe_name} probe")
        ax.legend(fontsize=8)
        ax.set_xticks(LAYERS)
        ax.set_ylim(bottom=-0.5)

    fig.suptitle("Error prefill: correct vs incorrect answer separation by follow-up type", fontsize=12)
    plt.tight_layout()
    out = ASSETS_DIR / "plot_031126_error_prefill_effect_sizes.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_distributions_best(results: dict):
    """Violin plots for best condition across follow-up types."""
    fig, axes = plt.subplots(1, len(FOLLOWUP_TYPES), figsize=(3.5 * len(FOLLOWUP_TYPES), 5), sharey=True)

    probe_name = "tb-2"
    layer = 32
    selector = PROBES[probe_name]["selector"]
    act_path = ACTIVATIONS_DIR / f"activations_{selector}.npz"
    act_task_ids, layer_acts = load_activations(act_path, layers=[layer])
    probe_path = PROBES[probe_name]["probe_dir"] / f"probe_ridge_L{layer}.npy"
    probe_weights = np.load(probe_path)
    scores = score_with_probe(probe_weights, layer_acts[layer])

    parsed = [parse_task_id(tid) for tid in act_task_ids]

    for ax, followup_type in zip(axes, FOLLOWUP_TYPES):
        correct_mask = np.array([ac == "correct" and ft == followup_type for _, ac, ft in parsed])
        incorrect_mask = np.array([ac == "incorrect" and ft == followup_type for _, ac, ft in parsed])

        correct_scores = scores[correct_mask]
        incorrect_scores = scores[incorrect_mask]

        parts = ax.violinplot([incorrect_scores, correct_scores], positions=[0, 1], showmeans=True)
        for pc in parts["bodies"]:
            pc.set_alpha(0.7)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Incorrect", "Correct"], fontsize=9)
        d = results[probe_name][followup_type][str(layer)]["cohens_d"]
        ax.set_title(f"{followup_type}\n(d={d:+.2f})", fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel("Preference probe score")

    fig.suptitle(f"Probe scores: correct vs incorrect model answer ({probe_name}, L{layer})", fontsize=12)
    plt.tight_layout()
    out = ASSETS_DIR / "plot_031126_error_prefill_distributions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_auc_heatmap(results: dict):
    """Heatmap of AUC values: follow-up type × layer, one panel per probe."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, probe_name in zip(axes, results):
        auc_matrix = np.array([
            [results[probe_name][ft][str(l)]["auc"] for l in LAYERS]
            for ft in FOLLOWUP_TYPES
        ])
        im = ax.imshow(auc_matrix, cmap="RdYlGn", vmin=0.4, vmax=0.8, aspect="auto")
        ax.set_xticks(range(len(LAYERS)))
        ax.set_xticklabels([f"L{l}" for l in LAYERS])
        ax.set_yticks(range(len(FOLLOWUP_TYPES)))
        ax.set_yticklabels(FOLLOWUP_TYPES)
        ax.set_title(f"{probe_name} probe")

        for i in range(len(FOLLOWUP_TYPES)):
            for j in range(len(LAYERS)):
                ax.text(j, i, f"{auc_matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)

    fig.suptitle("AUC: correct vs incorrect classification", fontsize=12)
    plt.tight_layout()
    out = ASSETS_DIR / "plot_031126_error_prefill_auc_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Error Prefill Analysis")
    print("=" * 80)

    results = run_analysis()

    print("\n" + "=" * 80)
    print("Generating plots...")
    plot_effect_sizes_by_followup(results)
    plot_distributions_best(results)
    plot_auc_heatmap(results)

    out_path = OUTPUT_DIR / "error_prefill_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
