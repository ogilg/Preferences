"""Violin plots of truth probe scores at user turn-boundary, grouped by follow-up type and lying cluster."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]

SCORES_PATH = ROOT / "experiments/truth_probes/error_prefill/lying_prompts/lying_10prompt_scores_user_tb.json"
OUT_PATH = ROOT / "experiments/truth_probes/error_prefill/lying_prompts/assets/plot_031226_cluster_violins_user_tb.png"

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
FOLLOWUPS = ["neutral", "presupposes", "challenge"]
FOLLOWUP_LABELS = {"neutral": "Neutral", "presupposes": "Presupposes", "challenge": "Challenge"}
CLUSTER_SHORT = {
    "No explicit wrongness": "Identity\nonly",
    "Implicit deception": "Implicit\ndeception",
    "Explicit wrongness": "Explicit\nwrongness",
}


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

    prompt_to_cluster = {}
    for cluster_name, info in CLUSTERS.items():
        for p in info["prompts"]:
            prompt_to_cluster[p] = cluster_name

    # Collect scores per followup, cluster, layer, condition
    # scores[followup][cluster][layer][condition] = list of floats
    scores: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {
        fu: {
            cluster: {layer: {"correct": [], "incorrect": []} for layer in LAYERS}
            for cluster in CLUSTERS
        }
        for fu in FOLLOWUPS
    }

    for i, p in enumerate(parsed):
        cluster = prompt_to_cluster[p["prompt_name"]]
        fu = p["followup"]
        for layer in LAYERS:
            scores[fu][cluster][layer][p["answer_condition"]].append(data[layer][i])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), dpi=150, sharey="row")

    cluster_names = list(CLUSTERS.keys())

    for row, layer in enumerate(LAYERS):
        for col, fu in enumerate(FOLLOWUPS):
            ax = axes[row, col]

            positions_correct = []
            positions_incorrect = []
            data_correct = []
            data_incorrect = []

            for ci, cluster in enumerate(cluster_names):
                center = ci * 2.5
                pos_inc = center - 0.4
                pos_cor = center + 0.4
                positions_incorrect.append(pos_inc)
                positions_correct.append(pos_cor)
                data_incorrect.append(scores[fu][cluster][layer]["incorrect"])
                data_correct.append(scores[fu][cluster][layer]["correct"])

            # Draw incorrect violins
            vp_inc = ax.violinplot(
                data_incorrect,
                positions=positions_incorrect,
                showmeans=False,
                showmedians=False,
                showextrema=False,
                widths=0.7,
            )
            for body in vp_inc["bodies"]:
                body.set_facecolor(COLOR_INCORRECT)
                body.set_alpha(0.7)

            # Draw correct violins
            vp_cor = ax.violinplot(
                data_correct,
                positions=positions_correct,
                showmeans=False,
                showmedians=False,
                showextrema=False,
                widths=0.7,
            )
            for body in vp_cor["bodies"]:
                body.set_facecolor(COLOR_CORRECT)
                body.set_alpha(0.7)

            # Median lines and annotations
            for ci, cluster in enumerate(cluster_names):
                center = ci * 2.5
                inc = np.array(data_incorrect[ci])
                cor = np.array(data_correct[ci])

                # Medians
                ax.hlines(np.median(inc), center - 0.4 - 0.2, center - 0.4 + 0.2, colors=COLOR_INCORRECT, linewidths=1.5)
                ax.hlines(np.median(cor), center + 0.4 - 0.2, center + 0.4 + 0.2, colors=COLOR_CORRECT, linewidths=1.5)

                # Stats
                d = cohens_d(cor, inc)
                labels = np.concatenate([np.ones(len(cor)), np.zeros(len(inc))])
                all_scores = np.concatenate([cor, inc])
                auc = roc_auc_score(labels, all_scores)

                y_top = ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else max(np.max(cor), np.max(inc)) * 1.05
                ax.text(
                    center, 0.97, f"d={d:.2f}  AUC={auc:.2f}",
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=6.5,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.8, edgecolor="none"),
                )

            tick_positions = [ci * 2.5 for ci in range(len(cluster_names))]
            tick_labels = [CLUSTER_SHORT[c] for c in cluster_names]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, fontsize=8)

            if row == 0:
                ax.set_title(FOLLOWUP_LABELS[fu], fontsize=11, fontweight="bold")

            if col == 0:
                ax.set_ylabel("Preference probe score (tb-2)", fontsize=9)

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

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_CORRECT, alpha=0.7, label="Correct answer"),
        Patch(facecolor=COLOR_INCORRECT, alpha=0.7, label="Incorrect answer"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=9, frameon=True)

    fig.suptitle(
        "Truth probe scores at user turn-boundary by follow-up type and lying cluster",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0.04, 0, 0.98, 0.91])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH)
    print(f"Saved to {OUT_PATH}")

    # Print summary
    for layer in LAYERS:
        print(f"\n{LAYER_LABELS[layer]}:")
        for fu in FOLLOWUPS:
            print(f"  {FOLLOWUP_LABELS[fu]}:")
            for cluster in cluster_names:
                cor = np.array(scores[fu][cluster][layer]["correct"])
                inc = np.array(scores[fu][cluster][layer]["incorrect"])
                d = cohens_d(cor, inc)
                labels_arr = np.concatenate([np.ones(len(cor)), np.zeros(len(inc))])
                all_s = np.concatenate([cor, inc])
                auc = roc_auc_score(labels_arr, all_s)
                print(f"    {cluster}: d={d:.2f}, AUC={auc:.2f} (n={len(cor)}+{len(inc)})")


if __name__ == "__main__":
    main()
