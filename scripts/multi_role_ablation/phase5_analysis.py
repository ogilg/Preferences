"""Phase 5: Analysis and plotting for multi-role ablation experiment.

Generates plots for the report:
1. Cross-persona generalization matrix (heatmap)
2. Scaling plot: mean cross-persona r vs number of training personas
3. Probe direction cosine similarity matrix (single-persona probes)
4. Utility correlation matrix
5. Per-persona eval breakdown for all conditions

All plots saved to experiments/probe_generalization/multi_role_ablation/assets/
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).parent.parent.parent
EXPERIMENT_DIR = REPO / "experiments/probe_generalization/multi_role_ablation"
ASSETS_DIR = EXPERIMENT_DIR / "assets"
ASSETS_DIR.mkdir(exist_ok=True)

PERSONA_NAMES = ["no_prompt", "villain", "midwest", "aesthete"]
PERSONA_DISPLAY = ["No Prompt", "Villain", "Midwest", "Aesthete"]


def load_results() -> dict:
    with open(EXPERIMENT_DIR / "probe_results.json") as f:
        return json.load(f)


def get_single_persona_conditions(results: dict) -> list[dict]:
    return [r for r in results["conditions"] if len(r["train_personas"]) == 1]


def plot_generalization_matrix(results: dict) -> None:
    """Heatmap of Pearson r for 15 conditions × 4 eval personas."""
    conditions = results["conditions"]

    # Build matrix
    n_conditions = len(conditions)
    n_eval = len(PERSONA_NAMES)
    matrix = np.zeros((n_conditions, n_eval))

    for i, cond in enumerate(conditions):
        for j, p_name in enumerate(PERSONA_NAMES):
            matrix[i, j] = cond["eval"][p_name]["pearson_r"]

    condition_labels = [r["condition_name"] for r in conditions]

    fig, ax = plt.subplots(figsize=(9, 12))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Pearson r")

    ax.set_xticks(range(n_eval))
    ax.set_xticklabels(PERSONA_DISPLAY, fontsize=10)
    ax.set_yticks(range(n_conditions))
    ax.set_yticklabels(condition_labels, fontsize=8)

    # Annotate cells
    for i in range(n_conditions):
        for j in range(n_eval):
            val = matrix[i, j]
            color = "white" if val < 0.3 or val > 0.8 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

    ax.set_xlabel("Eval Persona", fontsize=11)
    ax.set_ylabel("Training Condition", fontsize=11)
    ax.set_title("Probe Generalization: Pearson r\n(rows=training condition, cols=eval persona)", fontsize=11)

    plt.tight_layout()
    out = ASSETS_DIR / "plot_022526_generalization_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_scaling_curve(results: dict) -> None:
    """Mean cross-persona r vs number of training personas."""
    # For each number of personas trained on, compute:
    # 1. Same-persona r (diagonal of eval matrix for single-persona probes)
    # 2. Cross-persona r (off-diagonal: eval on personas NOT trained on)
    conditions = results["conditions"]

    n_training_list = []
    same_persona_r_list = []
    cross_persona_r_list = []
    all_eval_r_list = []

    for n_personas in range(1, 5):
        size_conditions = [r for r in conditions if len(r["train_personas"]) == n_personas]
        if not size_conditions:
            continue

        same_rs = []
        cross_rs = []
        all_rs = []

        for cond in size_conditions:
            train_names = set(["no_prompt", "villain", "midwest", "aesthete"][p-1] for p in cond["train_personas"])
            for p_name in PERSONA_NAMES:
                r = cond["eval"][p_name]["pearson_r"]
                all_rs.append(r)
                if p_name in train_names:
                    same_rs.append(r)
                else:
                    cross_rs.append(r)

        n_training_list.append(n_personas)
        same_persona_r_list.append(np.mean(same_rs) if same_rs else np.nan)
        cross_persona_r_list.append(np.mean(cross_rs) if cross_rs else np.nan)
        all_eval_r_list.append(np.mean(all_rs) if all_rs else np.nan)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(n_training_list, same_persona_r_list, "o-", color="C0", label="Same-persona", linewidth=2, markersize=7)
    ax.plot(n_training_list, cross_persona_r_list, "s-", color="C1", label="Cross-persona", linewidth=2, markersize=7)
    ax.plot(n_training_list, all_eval_r_list, "^--", color="C2", label="All eval (mean)", linewidth=1.5, markersize=6)
    ax.set_xlabel("Number of training personas", fontsize=11)
    ax.set_ylabel("Mean Pearson r", fontsize=11)
    ax.set_title("Generalization vs. Training Diversity", fontsize=12)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = ASSETS_DIR / "plot_022526_scaling_curve.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_cosine_similarity(results: dict) -> None:
    """Cosine similarity matrix for single-persona probe directions."""
    # Filter to single-persona conditions only
    conditions = results["conditions"]
    single_conditions = [r for r in conditions if len(r["train_personas"]) == 1]
    single_names = [r["condition_name"] for r in single_conditions]

    all_names = results["probe_cosine_similarity"]["condition_names"]
    all_matrix = np.array(results["probe_cosine_similarity"]["matrix"])

    # Find indices of single-persona conditions
    single_indices = [all_names.index(n) for n in single_names if n in all_names]
    sim_matrix = all_matrix[np.ix_(single_indices, single_indices)]

    display_names = [all_names[i] for i in single_indices]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(sim_matrix, cmap="RdYlGn", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="Cosine similarity")
    ax.set_xticks(range(len(display_names)))
    ax.set_xticklabels(display_names, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(display_names)))
    ax.set_yticklabels(display_names, fontsize=10)
    for i in range(len(display_names)):
        for j in range(len(display_names)):
            ax.text(j, i, f"{sim_matrix[i,j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Probe Direction Cosine Similarity\n(single-persona probes, layer 31)", fontsize=11)
    plt.tight_layout()
    out = ASSETS_DIR / "plot_022526_cosine_similarity.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_utility_correlation(results: dict) -> None:
    """Utility correlation matrix between Thurstonian μ values."""
    util_corr = results["utility_correlation"]
    matrix = np.array(util_corr["matrix"])
    names = util_corr["persona_names"]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=10)
    ax.set_title(f"Utility Correlation (Thurstonian μ)\nn={util_corr['n_common_tasks']} shared tasks", fontsize=11)
    plt.tight_layout()
    out = ASSETS_DIR / "plot_022526_utility_correlation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def summarize_results(results: dict) -> None:
    """Print key metrics from results."""
    conditions = results["conditions"]

    # Single-persona: same vs cross-persona gap
    print("\n=== SINGLE-PERSONA PROBES ===")
    print(f"{'Persona':<15} | {'same r':>7} | {'cross r':>8} | {'gap':>6}")
    print("-" * 45)
    for cond in [r for r in conditions if len(r["train_personas"]) == 1]:
        p_id = cond["train_personas"][0]
        p_name = ["no_prompt", "villain", "midwest", "aesthete"][p_id - 1]
        same_r = cond["eval"][p_name]["pearson_r"]
        cross_rs = [cond["eval"][n]["pearson_r"] for n in PERSONA_NAMES if n != p_name]
        cross_r = np.mean(cross_rs)
        print(f"{p_name:<15} | {same_r:>7.3f} | {cross_r:>8.3f} | {same_r-cross_r:>+6.3f}")

    # All-persona probe
    all_persona_cond = next(r for r in conditions if len(r["train_personas"]) == 4)
    print(f"\n=== ALL-PERSONA PROBE ===")
    print(f"{'Eval persona':<15} | {'Pearson r':>9}")
    print("-" * 28)
    for p_name in PERSONA_NAMES:
        r = all_persona_cond["eval"][p_name]["pearson_r"]
        print(f"{p_name:<15} | {r:>9.3f}")
    mean_all = np.mean([all_persona_cond["eval"][p]["pearson_r"] for p in PERSONA_NAMES])
    print(f"{'Mean':<15} | {mean_all:>9.3f}")


def main() -> None:
    print("Loading results...")
    results = load_results()

    print("Generating plots...")
    plot_generalization_matrix(results)
    plot_scaling_curve(results)
    plot_cosine_similarity(results)
    plot_utility_correlation(results)
    summarize_results(results)

    print("\nAll plots saved to", ASSETS_DIR)


if __name__ == "__main__":
    main()
