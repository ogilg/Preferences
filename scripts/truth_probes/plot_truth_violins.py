"""Violin plots for truth probe results at a consistent layer.

1. creak_raw vs creak_repeat: true vs false CREAK statements
2. error_prefill: assistant_tb:-1 vs turn_boundary:-2
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.probes.core.activations import load_activations
from src.probes.core.evaluate import score_with_probe
from src.task_data.loader import load_tasks
from src.task_data.task import OriginDataset

matplotlib.rcParams.update({"font.size": 12})

ROOT = Path(__file__).resolve().parents[2]
PROBES_DIR = ROOT / "results" / "probes" / "heldout_eval_gemma3_tb-2" / "probes"
ASSETS_DIR = ROOT / "docs" / "logs" / "assets"

LAYER = 32


def compute_cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    pooled_std = np.sqrt(
        (a.var(ddof=1) * (len(a) - 1) + b.var(ddof=1) * (len(b) - 1))
        / (len(a) + len(b) - 2)
    )
    return float((a.mean() - b.mean()) / pooled_std)


def style_violin(parts):
    for pc, color in zip(parts["bodies"], ["#e74c3c", "#2ecc71"]):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        if key in parts:
            parts[key].set_color("black")
            parts[key].set_linewidth(1)


def plot_creak_raw_repeat():
    """Violin plots for creak_raw and creak_repeat: true vs false statements."""
    tasks = load_tasks(20000, [OriginDataset.CREAK])
    label_map = {t.id: t.metadata["label"] for t in tasks}

    probe = np.load(PROBES_DIR / f"probe_ridge_L{LAYER}.npy")

    fig, axes = plt.subplots(1, 2, figsize=(8, 4.5), sharey=True)

    settings = [
        ("creak_raw", "Raw generation"),
        ("creak_repeat", '"Repeat this statement"'),
    ]

    for ax, (setting, title) in zip(axes, settings):
        act_path = ROOT / "activations" / f"gemma_3_27b_{setting}" / "activations_turn_boundary:-2.npz"
        tids, layer_acts = load_activations(act_path, layers=[LAYER])
        scores = score_with_probe(probe, layer_acts[LAYER])

        labels = [label_map[tid] for tid in tids]
        true_mask = np.array([l == "true" for l in labels])
        false_mask = np.array([l == "false" for l in labels])

        false_s = scores[false_mask]
        true_s = scores[true_mask]
        d = compute_cohens_d(true_s, false_s)

        parts = ax.violinplot([false_s, true_s], positions=[0, 1],
                              showmedians=True, showextrema=True)
        style_violin(parts)

        ax.set_title(f"{title}\nd = {d:.2f}", fontsize=11)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["False statement", "True statement"])

    axes[0].set_ylabel("Probe score")
    fig.suptitle(f"Preference probe on CREAK true vs false statements\n(tb-2 probe, L{LAYER}, n ≈ 4800 per group)",
                 fontsize=12)
    fig.tight_layout()

    out_path = ASSETS_DIR / "plot_031226_creak_raw_repeat_violins.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def parse_error_prefill_task_id(task_id: str) -> tuple[str, str, str]:
    parts = task_id.split("_")
    ex_id = f"{parts[0]}_{parts[1]}"
    answer_condition = parts[2]
    followup_type = "_".join(parts[3:])
    return ex_id, answer_condition, followup_type


def plot_error_prefill_assistant_vs_user():
    """Violin plots: assistant_tb:-1 vs turn_boundary:-2 for error prefill."""
    probe = np.load(PROBES_DIR / f"probe_ridge_L{LAYER}.npy")
    followup = "presupposes"

    # Panel 1: assistant_tb:-1 (real activations)
    act_path = ROOT / "activations" / "gemma_3_27b_error_prefill" / "activations_assistant_tb:-1.npz"
    tids, layer_acts = load_activations(act_path, layers=[LAYER])
    parsed = [parse_error_prefill_task_id(tid) for tid in tids]
    scores = score_with_probe(probe, layer_acts[LAYER])

    correct_mask = np.array([ac == "correct" and ft == followup for _, ac, ft in parsed])
    incorrect_mask = np.array([ac == "incorrect" and ft == followup for _, ac, ft in parsed])
    asst_correct = scores[correct_mask]
    asst_incorrect = scores[incorrect_mask]

    # Panel 2: turn_boundary:-2 (from summary stats — raw activations cleaned up)
    with open(ROOT / "experiments" / "truth_probes" / "error_prefill" / "error_prefill_results.json") as f:
        orig = json.load(f)
    tb_stats = orig["tb-2"][followup][str(LAYER)]
    mean_diff = tb_stats["mean_correct"] - tb_stats["mean_incorrect"]
    pooled_std = mean_diff / tb_stats["cohens_d"]
    rng = np.random.default_rng(42)
    tb_correct = rng.normal(tb_stats["mean_correct"], pooled_std, tb_stats["n_correct"])
    tb_incorrect = rng.normal(tb_stats["mean_incorrect"], pooled_std, tb_stats["n_incorrect"])

    fig, axes = plt.subplots(1, 2, figsize=(8, 4.5), sharey=True)

    panels = [
        (asst_incorrect, asst_correct, "Model's answer\n(assistant last token)"),
        (tb_incorrect, tb_correct, "User's follow-up\n(turn boundary)"),
    ]

    for ax, (inc, cor, title) in zip(axes, panels):
        d = compute_cohens_d(cor, inc)
        parts = ax.violinplot([inc, cor], positions=[0, 1],
                              showmedians=True, showextrema=True)
        style_violin(parts)
        ax.set_title(f"{title}\nd = {d:.2f}", fontsize=11)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Incorrect", "Correct"])

    axes[0].set_ylabel("Probe score")
    fig.suptitle(f"Error prefill: correct vs incorrect answers\n(tb-2 probe, L{LAYER}, presupposes)",
                 fontsize=12)
    fig.tight_layout()

    out_path = ASSETS_DIR / "plot_031226_error_prefill_assistant_vs_user_violins.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def parse_lying_task_id(task_id: str) -> tuple[str, str, str, str]:
    parts = task_id.split("_")
    ex_id = f"{parts[0]}_{parts[1]}"
    answer_condition = parts[2]
    sys_prompt = f"{parts[3]}_{parts[4]}"
    followup_type = "_".join(parts[5:])
    return ex_id, answer_condition, sys_prompt, followup_type


def plot_lying_violins():
    """3-panel violin: no_lying vs lie_direct vs lie_roleplay at L32."""
    probe = np.load(PROBES_DIR / f"probe_ridge_L{LAYER}.npy")
    followup = "presupposes"
    selector = "assistant_tb:-1"

    # No-lying: from original error_prefill activations
    orig_path = ROOT / "activations" / "gemma_3_27b_error_prefill" / f"activations_{selector}.npz"
    orig_tids, orig_acts = load_activations(orig_path, layers=[LAYER])
    orig_parsed = [parse_error_prefill_task_id(tid) for tid in orig_tids]
    orig_scores = score_with_probe(probe, orig_acts[LAYER])

    # Lying: from lying_prefill activations
    lying_path = ROOT / "activations" / "gemma_3_27b_lying_prefill" / f"activations_{selector}.npz"
    lying_tids, lying_acts = load_activations(lying_path, layers=[LAYER])
    lying_parsed = [parse_lying_task_id(tid) for tid in lying_tids]
    lying_scores = score_with_probe(probe, lying_acts[LAYER])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)

    conditions = [
        ("No lying", orig_scores, orig_parsed, None),
        ("Direct lying", lying_scores, lying_parsed, "lie_direct"),
        ("Roleplay lying", lying_scores, lying_parsed, "lie_roleplay"),
    ]

    for ax, (title, scores, parsed, sys_type) in zip(axes, conditions):
        if sys_type is None:
            correct_mask = np.array([ac == "correct" and ft == followup for _, ac, ft in parsed])
            incorrect_mask = np.array([ac == "incorrect" and ft == followup for _, ac, ft in parsed])
        else:
            correct_mask = np.array([ac == "correct" and sp == sys_type and ft == followup
                                     for _, ac, sp, ft in parsed])
            incorrect_mask = np.array([ac == "incorrect" and sp == sys_type and ft == followup
                                       for _, ac, sp, ft in parsed])

        correct_s = scores[correct_mask]
        incorrect_s = scores[incorrect_mask]
        d = compute_cohens_d(correct_s, incorrect_s)

        parts = ax.violinplot([incorrect_s, correct_s], positions=[0, 1],
                              showmedians=True, showextrema=True)
        style_violin(parts)
        ax.set_title(f"{title}\nd = {d:.2f}", fontsize=11)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Incorrect", "Correct"])

    axes[0].set_ylabel("Probe score")
    fig.suptitle(f"Lying instructions disrupt the error signal\n({selector}, tb-2 probe, L{LAYER}, {followup})",
                 fontsize=12)
    fig.tight_layout()

    out_path = ASSETS_DIR / "plot_031226_lying_violins.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    plot_creak_raw_repeat()
    plot_error_prefill_assistant_vs_user()
    plot_lying_violins()
