"""Violin plots of truth probe scores grouped by lying instruction cluster."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]

SCORES_PATH = ROOT / "experiments/truth_probes/error_prefill/lying_prompts/lying_10prompt_scores.json"
OUT_PATH = ROOT / "experiments/truth_probes/error_prefill/lying_prompts/assets/plot_031226_cluster_violins.png"

CLUSTERS = {
    "No explicit wrongness": {
        "prompts": ["roleplay_villain", "roleplay_sadist"],
        "color": "#2ca02c",
    },
    "Implicit deception": {
        "prompts": ["lie_roleplay", "direct_opposite_day", "direct_mislead"],
        "color": "#ff7f0e",
    },
    "Explicit wrongness": {
        "prompts": ["lie_direct", "direct_please_lie", "direct_wrong", "roleplay_trickster", "roleplay_exam"],
        "color": "#d62728",
    },
}

COLOR_CORRECT = "#5B8FC9"
COLOR_INCORRECT = "#E07B7B"
LAYERS = ["L32", "L53"]
LAYER_LABELS = {"L32": "Layer 32", "L53": "Layer 53"}

# Baseline values at L53
BASELINE_D = 3.29
BASELINE_AUC = 0.99


def parse_task_id(tid: str) -> dict:
    parts = tid.split("_")
    return {
        "ex_id": parts[0] + "_" + parts[1],
        "answer_condition": parts[2],
        "prompt_name": "_".join(parts[3:-1]),
        "followup": parts[-1],
    }


def cohens_d(correct: np.ndarray, incorrect: np.ndarray) -> float:
    n1, n2 = len(correct), len(incorrect)
    pooled_std = np.sqrt(((n1 - 1) * np.var(correct, ddof=1) + (n2 - 1) * np.var(incorrect, ddof=1)) / (n1 + n2 - 2))
    return (np.mean(correct) - np.mean(incorrect)) / pooled_std


def main():
    with open(SCORES_PATH) as f:
        data = json.load(f)

    task_ids = data["task_ids"]
    parsed = [parse_task_id(tid) for tid in task_ids]

    # Build prompt -> cluster mapping
    prompt_to_cluster = {}
    for cluster_name, info in CLUSTERS.items():
        for p in info["prompts"]:
            prompt_to_cluster[p] = cluster_name

    # Collect scores per cluster, layer, condition
    cluster_scores: dict[str, dict[str, dict[str, list[float]]]] = {
        cluster: {layer: {"correct": [], "incorrect": []} for layer in LAYERS}
        for cluster in CLUSTERS
    }

    for i, p in enumerate(parsed):
        cluster = prompt_to_cluster[p["prompt_name"]]
        for layer in LAYERS:
            cluster_scores[cluster][layer][p["answer_condition"]].append(data[layer][i])

    # Compute stats
    cluster_stats: dict[str, dict[str, dict]] = {}
    for cluster in CLUSTERS:
        cluster_stats[cluster] = {}
        for layer in LAYERS:
            correct = np.array(cluster_scores[cluster][layer]["correct"])
            incorrect = np.array(cluster_scores[cluster][layer]["incorrect"])
            d = cohens_d(correct, incorrect)
            labels = np.concatenate([np.ones(len(correct)), np.zeros(len(incorrect))])
            scores = np.concatenate([correct, incorrect])
            auc = roc_auc_score(labels, scores)
            cluster_stats[cluster][layer] = {"d": d, "auc": auc}

    # Plot: 2 rows (layers) x 3 columns (clusters)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=150, sharey="row")

    cluster_names = list(CLUSTERS.keys())

    for row, layer in enumerate(LAYERS):
        for col, cluster in enumerate(cluster_names):
            ax = axes[row, col]
            correct = cluster_scores[cluster][layer]["correct"]
            incorrect = cluster_scores[cluster][layer]["incorrect"]

            parts = ax.violinplot(
                [incorrect, correct],
                positions=[0, 1],
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )

            for i, body in enumerate(parts["bodies"]):
                body.set_facecolor(COLOR_INCORRECT if i == 0 else COLOR_CORRECT)
                body.set_alpha(0.7)

            # Median lines
            for i, (vals, color) in enumerate(
                [(incorrect, COLOR_INCORRECT), (correct, COLOR_CORRECT)]
            ):
                median = np.median(vals)
                ax.hlines(median, i - 0.25, i + 0.25, colors=color, linewidths=2)

            stats = cluster_stats[cluster][layer]
            ax.set_title(
                f"{cluster}\n(d={stats['d']:.2f}, AUC={stats['auc']:.2f})",
                fontsize=9,
            )
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Incorrect", "Correct"])

            if col == 0:
                ax.set_ylabel("Preference probe score (tb-2)")

        # Row label
        axes[row, 0].annotate(
            LAYER_LABELS[layer],
            xy=(-0.35, 0.5),
            xycoords="axes fraction",
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="center",
            rotation=90,
        )

    fig.suptitle(
        "Truth probe score distributions by lying instruction type (assistant tb:-1)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0.04, 0, 1, 0.94])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH)
    print(f"Saved to {OUT_PATH}")

    # Print summary
    for layer in LAYERS:
        print(f"\n{LAYER_LABELS[layer]}:")
        print(f"  Baseline: d={BASELINE_D:.2f}, AUC~{BASELINE_AUC:.2f}")
        for cluster in cluster_names:
            s = cluster_stats[cluster][layer]
            n_correct = len(cluster_scores[cluster][layer]["correct"])
            n_incorrect = len(cluster_scores[cluster][layer]["incorrect"])
            print(f"  {cluster}: d={s['d']:.2f}, AUC={s['auc']:.2f} (n={n_correct}+{n_incorrect})")


if __name__ == "__main__":
    main()
