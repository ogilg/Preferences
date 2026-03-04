"""Section 5.1: Baseline probe transfer to persona conditions.

Two panels: Pearson r and pairwise accuracy.
Two bars per persona: noprompt probe, within-persona probe.
Within-persona probes retrained inline (split_a train, split_b alpha sweep).
Style matches plot_exp1b_overview.py.

Usage:
    python docs/lw_post/plots/plot_section5_mra_transfer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

plt.style.use("seaborn-v0_8-whitegrid")

ASSETS_DIR = Path(__file__).parent.parent / "assets"
LAYER = 31

from scripts.ood_system_prompts.analyze_mra_utilities import load_persona_utilities
from scripts.multi_role_ablation.run_mra_probes import (
    load_persona_split_data,
    train_probe_with_alpha_selection,
    ACTIVATION_PATHS,
)
from src.probes.core.activations import load_probe_data
from src.probes.data_loading import load_thurstonian_scores

PERSONAS = ["aesthete", "midwest", "villain", "sadist"]
PERSONA_LABELS = ["Aesthete", "Midwest", "Villain", "Sadist"]

PROBE_DIR = REPO_ROOT / "results" / "probes" / "gemma3_10k_heldout_std_raw"
ACT_DIRS = {
    "noprompt": REPO_ROOT / "activations" / "gemma_3_27b",
    "aesthete": REPO_ROOT / "activations" / "gemma_3_27b_aesthete",
    "midwest": REPO_ROOT / "activations" / "gemma_3_27b_midwest",
    "villain": REPO_ROOT / "activations" / "gemma_3_27b_villain",
    "sadist": REPO_ROOT / "activations" / "gemma_3_27b_sadist",
}

ERROR_KW = {"linewidth": 0.8, "alpha": 0.5}


def pairwise_accuracy(predicted: np.ndarray, actual: np.ndarray) -> float:
    n = len(predicted)
    correct = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if actual[i] == actual[j]:
                continue
            total += 1
            if (predicted[i] - predicted[j]) * (actual[i] - actual[j]) > 0:
                correct += 1
    return correct / total if total > 0 else float("nan")


def load_baseline_probe() -> np.ndarray:
    return np.load(PROBE_DIR / "probes" / f"probe_ridge_L{LAYER:02d}.npy")


def score_with_baseline(act_dir: Path, probe_weights: np.ndarray):
    act_path = act_dir / "activations_prompt_last.npz"
    d = np.load(act_path)
    act_task_ids = list(d["task_ids"])
    acts = d[f"layer_{LAYER}"]
    scores = {}
    for i, tid in enumerate(act_task_ids):
        scores[tid] = float(acts[i] @ probe_weights[:-1] + probe_weights[-1])
    return scores


def evaluate(scores: dict[str, float], utilities: dict[str, float]):
    shared = sorted(set(scores) & set(utilities))
    s = np.array([scores[t] for t in shared])
    u = np.array([utilities[t] for t in shared])
    r = scipy_stats.pearsonr(s, u)[0]
    acc = pairwise_accuracy(s, u)
    return r, acc, len(shared)


def train_and_eval_within_persona(persona: str):
    """Train within-persona probe on split_a, sweep alpha on split_b, eval on all tasks."""
    X_train, y_train, _ = load_persona_split_data(persona, "a", LAYER)
    X_val, y_val, _ = load_persona_split_data(persona, "b", LAYER)
    probe, scaler, best_alpha, _ = train_probe_with_alpha_selection(
        X_train, y_train, X_val, y_val
    )
    print(f"  {persona} within-persona probe: alpha={best_alpha:.1f}")

    # Eval on all tasks
    all_utils = load_persona_utilities(persona)
    all_task_ids = sorted(all_utils.keys())
    X_all, y_all, matched_ids = load_probe_data(
        ACTIVATION_PATHS[persona], all_utils, all_task_ids, LAYER
    )
    X_all_s = scaler.transform(X_all)
    y_pred = probe.predict(X_all_s)

    r = scipy_stats.pearsonr(y_all, y_pred)[0]
    acc = pairwise_accuracy(y_pred, y_all)
    return r, acc, len(matched_ids)


def main():
    baseline_probe = load_baseline_probe()
    noprompt_utils = load_persona_utilities("noprompt")

    results = {}
    for persona in PERSONAS:
        persona_utils = load_persona_utilities(persona)

        # Baseline probe
        bl_scores = score_with_baseline(ACT_DIRS[persona], baseline_probe)
        bl_r, bl_acc, bl_n = evaluate(bl_scores, persona_utils)
        print(f"{persona} baseline probe: r={bl_r:.3f} acc={bl_acc:.3f} (n={bl_n})")

        # Within-persona probe
        wp_r, wp_acc, wp_n = train_and_eval_within_persona(persona)
        print(f"{persona} within-persona: r={wp_r:.3f} acc={wp_acc:.3f} (n={wp_n})")

        results[persona] = {
            "bl_r": bl_r, "bl_acc": bl_acc,
            "wp_r": wp_r, "wp_acc": wp_acc,
        }

    # Baseline utilities correlation
    for persona in PERSONAS:
        persona_utils = load_persona_utilities(persona)
        shared = sorted(set(noprompt_utils) & set(persona_utils))
        np_vals = np.array([noprompt_utils[t] for t in shared])
        p_vals = np.array([persona_utils[t] for t in shared])
        bu_r = scipy_stats.pearsonr(np_vals, p_vals)[0]
        bu_acc = pairwise_accuracy(np_vals, p_vals)
        results[persona]["bu_r"] = bu_r
        results[persona]["bu_acc"] = bu_acc
        print(f"{persona} baseline utils: r={bu_r:.3f} acc={bu_acc:.3f}")

    # Plot
    x = np.arange(len(PERSONAS))
    width = 0.3
    color_bu = "#B0B0B0"
    color_bl = "#6675B0"
    color_wp = "#E07050"

    fig, (ax_r, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))

    # Pearson r
    bu_rs = [results[p]["bu_r"] for p in PERSONAS]
    bl_rs = [results[p]["bl_r"] for p in PERSONAS]
    wp_rs = [results[p]["wp_r"] for p in PERSONAS]

    bars_bu_r = ax_r.bar(x - width / 2, bu_rs, width, label="Baseline utilities",
                         color=color_bu, capsize=3, error_kw=ERROR_KW)
    bars_bl_r = ax_r.bar(x + width / 2, bl_rs, width, label="Probe",
                         color=color_bl, capsize=3, error_kw=ERROR_KW)

    for bar in list(bars_bu_r) + list(bars_bl_r):
        val = bar.get_height()
        y_off = val + 0.03 if val >= 0 else val - 0.03
        va = "bottom" if val >= 0 else "top"
        ax_r.text(bar.get_x() + bar.get_width() / 2, y_off,
                  f"{val:.2f}", ha="center", va=va, fontsize=9, fontweight="bold")

    ax_r.set_title("Pearson r", fontsize=13)
    ax_r.set_xticks(x)
    ax_r.set_xticklabels(PERSONA_LABELS, fontsize=11)
    all_r_vals = bu_rs + bl_rs
    r_min = min(all_r_vals)
    ax_r.set_ylim(min(r_min - 0.1, -0.2) if r_min < 0 else 0, 1.05)
    ax_r.set_ylabel("Pearson r", fontsize=11)

    # Pairwise accuracy
    bu_accs = [results[p]["bu_acc"] for p in PERSONAS]
    bl_accs = [results[p]["bl_acc"] for p in PERSONAS]
    wp_accs = [results[p]["wp_acc"] for p in PERSONAS]

    bars_bu_acc = ax_acc.bar(x - width / 2, bu_accs, width, label="Baseline utilities",
                             color=color_bu, capsize=3, error_kw=ERROR_KW)
    bars_bl_acc = ax_acc.bar(x + width / 2, bl_accs, width, label="Probe",
                             color=color_bl, capsize=3, error_kw=ERROR_KW)
    ax_acc.axhline(y=0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

    for bar in list(bars_bu_acc) + list(bars_bl_acc):
        val = bar.get_height()
        ax_acc.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax_acc.set_title("Pairwise accuracy", fontsize=13)
    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels(PERSONA_LABELS, fontsize=11)
    all_acc_vals = bu_accs + bl_accs
    acc_min = min(all_acc_vals)
    ax_acc.set_ylim(min(acc_min - 0.05, 0.35) if acc_min < 0.5 else 0.5, 1.0)
    ax_acc.set_ylabel("Pairwise accuracy", fontsize=11)

    # Shared legend below the figure
    handles, labels = ax_r.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Baseline probe transfer to persona conditions", fontsize=14)
    fig.tight_layout()

    out = ASSETS_DIR / "plot_030426_s5_mra_probe_transfer.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
