"""Key comparison plot: presupposes vs control follow-up, correct vs incorrect answers."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.probes.core.activations import load_activations
from src.probes.core.evaluate import score_with_probe

ROOT = Path(__file__).resolve().parents[2]
ACTIVATIONS_DIR = ROOT / "activations" / "gemma_3_27b_error_prefill"
PROBES_DIR = ROOT / "results" / "probes"
RESULTS_PATH = ROOT / "experiments" / "truth_probes" / "error_prefill" / "error_prefill_results.json"
ASSETS_DIR = ROOT / "experiments" / "truth_probes" / "error_prefill" / "assets"

PROBE_NAME = "tb-2"
SELECTOR = "turn_boundary:-2"
PROBE_DIR = PROBES_DIR / "heldout_eval_gemma3_tb-2" / "probes"
LAYER = 32


def parse_task_id(task_id: str) -> tuple[str, str, str]:
    parts = task_id.split("_")
    ex_id = f"{parts[0]}_{parts[1]}"
    answer_condition = parts[2]
    followup_type = "_".join(parts[3:])
    return ex_id, answer_condition, followup_type


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_PATH) as f:
        results = json.load(f)

    act_path = ACTIVATIONS_DIR / f"activations_{SELECTOR}.npz"
    act_task_ids, layer_acts = load_activations(act_path, layers=[LAYER])
    probe_weights = np.load(PROBE_DIR / f"probe_ridge_L{LAYER}.npy")
    scores = score_with_probe(probe_weights, layer_acts[LAYER])

    parsed = [parse_task_id(tid) for tid in act_task_ids]

    conditions = [
        ("presupposes", "Follow-up presupposes\nthe model's answer"),
        ("neutral", "Follow-up is\n\"Thank you\""),
        ("control", "Follow-up is an\nunrelated task"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.5), sharey=True)

    for ax, (followup_type, label) in zip(axes, conditions):
        correct_mask = np.array([ac == "correct" and ft == followup_type for _, ac, ft in parsed])
        incorrect_mask = np.array([ac == "incorrect" and ft == followup_type for _, ac, ft in parsed])

        correct_scores = scores[correct_mask]
        incorrect_scores = scores[incorrect_mask]

        parts = ax.violinplot(
            [incorrect_scores, correct_scores],
            positions=[0, 1],
            showmeans=True,
            showextrema=False,
        )

        colors = ["#d94040", "#3aa63a"]
        for pc, color in zip(parts["bodies"], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        parts["cmeans"].set_color("black")
        parts["cmeans"].set_linewidth(1.5)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Incorrect\nanswer", "Correct\nanswer"], fontsize=11)

        d = results[PROBE_NAME][followup_type][str(LAYER)]["cohens_d"]
        auc = results[PROBE_NAME][followup_type][str(LAYER)]["auc"]
        ax.set_title(label, fontsize=12, pad=10)
        ax.text(
            0.5, 0.97, f"d = {d:+.2f}   AUC = {auc:.2f}",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=10, fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8),
        )

        if ax == axes[0]:
            ax.set_ylabel("Preference probe score", fontsize=11)

    fig.suptitle(
        "Preference probe separates correct from incorrect model answers\n"
        f"(tb-2 probe, layer {LAYER}, n = 1000 per group)",
        fontsize=13,
    )
    plt.tight_layout()
    out = ASSETS_DIR / "plot_031126_error_prefill_presupposes_vs_control.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
