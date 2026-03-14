"""Follow-up analysis: assistant-turn selectors + lying system prompts.

Scores new activations (assistant selectors from error_prefill, all selectors
from lying_prefill) with existing preference probes. Computes Cohen's d and AUC,
generates comparison plots and saves results JSON.
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind
from sklearn.metrics import roc_auc_score

from src.probes.core.activations import load_activations
from src.probes.core.evaluate import score_with_probe

matplotlib.rcParams.update({"font.size": 11})

ROOT = Path(__file__).resolve().parents[2]

ORIG_ACT_DIR = ROOT / "activations" / "gemma_3_27b_error_prefill"
LYING_ACT_DIR = ROOT / "activations" / "gemma_3_27b_lying_prefill"

PROBES_DIR = ROOT / "results" / "probes"
OUTPUT_DIR = ROOT / "experiments" / "truth_probes" / "error_prefill"
ASSETS_DIR = OUTPUT_DIR / "assets"

PROBES = {
    "tb-2": PROBES_DIR / "heldout_eval_gemma3_tb-2" / "probes",
    "tb-5": PROBES_DIR / "heldout_eval_gemma3_tb-5" / "probes",
}

LAYERS = [25, 32, 39, 46, 53]
LAYERS_STR = [str(l) for l in LAYERS]

ORIG_FOLLOWUP_TYPES = ["neutral", "presupposes", "challenge", "same_domain", "control"]
LYING_FOLLOWUP_TYPES = ["neutral", "presupposes"]
SYSTEM_PROMPT_TYPES = ["lie_direct", "lie_roleplay"]

# Selectors available in each directory
ASSISTANT_SELECTORS = ["assistant_mean", "assistant_tb:-1", "assistant_tb:-2",
                       "assistant_tb:-3", "assistant_tb:-4", "assistant_tb:-5"]
TB_SELECTORS = ["turn_boundary:-2", "turn_boundary:-5"]

FOLLOWUP_COLORS = {
    "neutral": "#1f77b4",
    "presupposes": "#d62728",
    "challenge": "#ff7f0e",
    "same_domain": "#2ca02c",
    "control": "#7f7f7f",
}


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


# ---- Plotting functions ----

def plot_assistant_effect_sizes(assistant_results: dict):
    """Plot 1: Cohen's d across layers for each assistant selector, per follow-up type."""
    # Use the 3 main selectors
    sel_order = ["assistant_mean", "assistant_tb:-1", "assistant_tb:-5"]
    available = [s for s in sel_order if s in assistant_results]

    for probe_name in ["tb-2", "tb-5"]:
        fig, axes = plt.subplots(1, len(available), figsize=(6 * len(available), 5), sharey=True)
        if len(available) == 1:
            axes = [axes]
        fig.suptitle(f"Assistant-turn selectors: correct vs incorrect ({probe_name} probe)", fontsize=14)

        for ax, selector in zip(axes, available):
            sel_data = assistant_results[selector][probe_name]
            for followup in ORIG_FOLLOWUP_TYPES:
                if followup not in sel_data:
                    continue
                ds = [sel_data[followup][str(l)]["cohens_d"] for l in LAYERS]
                ax.plot(LAYERS, ds, marker="o", label=followup, color=FOLLOWUP_COLORS[followup])
            ax.set_title(selector)
            ax.set_xlabel("Layer")
            ax.set_xticks(LAYERS)
            ax.axhline(0, color="black", linewidth=0.5, linestyle="--")

        axes[0].set_ylabel("Cohen's d")
        axes[0].set_ylim(-1.5, 3.0)
        axes[-1].legend(loc="upper left", fontsize=9)
        fig.tight_layout()
        fig.savefig(ASSETS_DIR / f"plot_031226_assistant_selectors_effect_sizes_{probe_name}.png", dpi=150)
        plt.close(fig)
        print(f"Saved: assistant_selectors_effect_sizes_{probe_name}")


def plot_assistant_vs_followup_comparison(assistant_results: dict):
    """Plot 2: Best d across layers for each selector position (presupposes condition)."""
    # Baseline values from original report (follow-up user turn selectors)
    baseline_tb2 = {
        "turn_boundary:-2": {"presupposes": 2.58, "neutral": 1.80},
        "turn_boundary:-5": {"presupposes": 2.33, "neutral": 1.99},
    }

    for followup in ["presupposes", "neutral"]:
        for probe_name in ["tb-2", "tb-5"]:
            selector_order = []
            bar_values = []
            bar_colors = []

            # TB selectors (baselines from original report)
            for tb_sel in TB_SELECTORS:
                selector_order.append(f"{tb_sel}\n(follow-up turn)")
                bar_values.append(baseline_tb2[tb_sel][followup])
                bar_colors.append("#4c72b0")

            # Assistant selectors from new results
            for asst_sel in ["assistant_mean", "assistant_tb:-1", "assistant_tb:-5"]:
                if asst_sel in assistant_results and probe_name in assistant_results[asst_sel]:
                    sel_data = assistant_results[asst_sel][probe_name]
                    if followup in sel_data:
                        best_d = max(sel_data[followup][str(l)]["cohens_d"] for l in LAYERS)
                        selector_order.append(f"{asst_sel}\n(answer turn)")
                        bar_values.append(best_d)
                        bar_colors.append("#dd8452")

            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.bar(range(len(selector_order)), bar_values, color=bar_colors,
                          edgecolor="black", linewidth=0.5)
            ax.set_xticks(range(len(selector_order)))
            ax.set_xticklabels(selector_order, fontsize=9)
            ax.set_ylabel("Best Cohen's d across layers")
            ax.set_title(f"Signal by selector position ({followup}, {probe_name} probe)")
            ax.set_ylim(0, max(bar_values) * 1.15)
            for bar, val in zip(bars, bar_values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=10)
            fig.tight_layout()
            fig.savefig(ASSETS_DIR / f"plot_031226_assistant_vs_followup_{followup}_{probe_name}.png", dpi=150)
            plt.close(fig)
            print(f"Saved: assistant_vs_followup_{followup}_{probe_name}")


def plot_lying_effect_comparison(lying_results: dict, assistant_results: dict):
    """Plot 3: Lying system prompt effect — grouped bar chart comparing no_lying vs lie conditions."""
    # Baseline (no-lying) best d values from original report
    no_lying_baseline = {
        "turn_boundary:-2": {"tb-2": {"presupposes": 2.58, "neutral": 1.80},
                              "tb-5": {"presupposes": 2.33, "neutral": 1.99}},
    }

    for probe_name in ["tb-2", "tb-5"]:
        for followup in LYING_FOLLOWUP_TYPES:
            conditions = ["no_lying", "lie_direct", "lie_roleplay"]
            selectors_to_plot = ["turn_boundary:-2", "assistant_tb:-1"]

            fig, axes = plt.subplots(1, len(selectors_to_plot), figsize=(12, 5), sharey=True)
            fig.suptitle(f"Lying system prompt effect ({followup}, {probe_name} probe)", fontsize=14)

            for ax, selector in zip(axes, selectors_to_plot):
                x = np.arange(len(LAYERS))
                width = 0.25

                for i, condition in enumerate(conditions):
                    if condition == "no_lying":
                        if selector.startswith("turn_boundary"):
                            # Use baseline from original report (per-layer from first run)
                            # We only have best-d, so just note this
                            if selector in no_lying_baseline and probe_name in no_lying_baseline[selector]:
                                # We don't have per-layer baselines for tb selectors stored,
                                # just plot the assistant selectors
                                pass
                            # For assistant selectors, use our new results
                            if selector in assistant_results and probe_name in assistant_results[selector]:
                                ds = [assistant_results[selector][probe_name][followup][str(l)]["cohens_d"]
                                      for l in LAYERS]
                            else:
                                continue
                        else:
                            if selector in assistant_results and probe_name in assistant_results[selector]:
                                if followup in assistant_results[selector][probe_name]:
                                    ds = [assistant_results[selector][probe_name][followup][str(l)]["cohens_d"]
                                          for l in LAYERS]
                                else:
                                    continue
                            else:
                                continue
                    else:
                        if selector in lying_results and probe_name in lying_results[selector]:
                            ly_data = lying_results[selector][probe_name]
                            if condition in ly_data and followup in ly_data[condition]:
                                ds = [ly_data[condition][followup][str(l)]["cohens_d"]
                                      for l in LAYERS]
                            else:
                                continue
                        else:
                            continue

                    offset = (i - 1) * width
                    ax.bar(x + offset, ds, width, label=condition,
                           edgecolor="black", linewidth=0.5)

                ax.set_title(selector)
                ax.set_xticks(x)
                ax.set_xticklabels([f"L{l}" for l in LAYERS])
                ax.axhline(0, color="black", linewidth=0.5, linestyle="--")

            axes[0].set_ylabel("Cohen's d")
            axes[-1].legend(fontsize=9)
            fig.tight_layout()
            fig.savefig(ASSETS_DIR / f"plot_031226_lying_effect_{followup}_{probe_name}.png", dpi=150)
            plt.close(fig)
            print(f"Saved: lying_effect_{followup}_{probe_name}")


def plot_lying_heatmaps(lying_results: dict, assistant_results: dict):
    """Plot 4: Heatmaps — Cohen's d by layer for no_lying / lie_direct / lie_roleplay."""
    conditions = ["no_lying", "lie_direct", "lie_roleplay"]

    for probe_name in ["tb-2", "tb-5"]:
        for followup in LYING_FOLLOWUP_TYPES:
            # One heatmap per selector type: tb and assistant
            for sel_group_name, sel_list in [("follow-up turn", ["turn_boundary:-2", "turn_boundary:-5"]),
                                              ("answer turn", ["assistant_mean", "assistant_tb:-1", "assistant_tb:-5"])]:

                available_sels = []
                for s in sel_list:
                    has_lying = (s in lying_results and probe_name in lying_results[s])
                    has_baseline = (s in assistant_results and probe_name in assistant_results[s]
                                    and followup in assistant_results[s][probe_name])
                    if has_lying or has_baseline:
                        available_sels.append(s)

                if not available_sels:
                    continue

                n_rows = len(conditions)
                n_sels = len(available_sels)

                fig, axes = plt.subplots(n_sels, 1, figsize=(9, 3.5 * n_sels), squeeze=False)
                fig.suptitle(f"Cohen's d heatmap: {sel_group_name} selectors\n({followup}, {probe_name} probe)",
                             fontsize=13, y=1.02)

                for ax_idx, selector in enumerate(available_sels):
                    ax = axes[ax_idx, 0]
                    matrix = np.full((n_rows, len(LAYERS)), np.nan)

                    for row_idx, condition in enumerate(conditions):
                        if condition == "no_lying":
                            if selector in assistant_results and probe_name in assistant_results[selector]:
                                if followup in assistant_results[selector][probe_name]:
                                    for col_idx, layer in enumerate(LAYERS):
                                        matrix[row_idx, col_idx] = assistant_results[selector][probe_name][followup][str(layer)]["cohens_d"]
                        else:
                            if selector in lying_results and probe_name in lying_results[selector]:
                                if condition in lying_results[selector][probe_name]:
                                    if followup in lying_results[selector][probe_name][condition]:
                                        for col_idx, layer in enumerate(LAYERS):
                                            matrix[row_idx, col_idx] = lying_results[selector][probe_name][condition][followup][str(layer)]["cohens_d"]

                    vmin = min(-0.5, np.nanmin(matrix) - 0.2) if not np.all(np.isnan(matrix)) else -0.5
                    vmax = max(3.0, np.nanmax(matrix) + 0.2) if not np.all(np.isnan(matrix)) else 3.0
                    im = ax.imshow(matrix, cmap="RdYlGn", vmin=vmin, vmax=vmax, aspect="auto")

                    ax.set_xticks(range(len(LAYERS)))
                    ax.set_xticklabels([f"L{l}" for l in LAYERS])
                    ax.set_yticks(range(n_rows))
                    ax.set_yticklabels(conditions)
                    ax.set_title(selector, fontsize=11)

                    for i in range(matrix.shape[0]):
                        for j in range(matrix.shape[1]):
                            val = matrix[i, j]
                            if not np.isnan(val):
                                text_color = "white" if val < 0.5 else "black"
                                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                                        fontsize=10, color=text_color)

                    fig.colorbar(im, ax=ax, label="Cohen's d", shrink=0.8)

                fig.tight_layout()
                safe_name = sel_group_name.replace(" ", "_")
                fig.savefig(ASSETS_DIR / f"plot_031226_lying_heatmap_{safe_name}_{followup}_{probe_name}.png",
                            dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved: lying_heatmap_{safe_name}_{followup}_{probe_name}")


def plot_lying_score_distributions(lying_results: dict, assistant_results: dict):
    """Plot 5: Violin plots of probe score distributions for lying vs no-lying."""
    # Pick the most informative combo: assistant_tb:-1, tb-2 probe, presupposes
    selector = "assistant_tb:-1"
    probe_name = "tb-2"
    layer = 46  # typically strong layer

    # Load activations for both datasets
    orig_path = ORIG_ACT_DIR / f"activations_{selector}.npz"
    lying_path = LYING_ACT_DIR / f"activations_{selector}.npz"

    if not orig_path.exists() or not lying_path.exists():
        print(f"  SKIP violin plots: missing activation files")
        return

    probe_path = PROBES["tb-2"] / f"probe_ridge_L{layer}.npy"
    probe_weights = np.load(probe_path)

    orig_tids, orig_acts = load_activations(orig_path, layers=[layer])
    orig_parsed = [parse_orig_task_id(tid) for tid in orig_tids]
    orig_scores = score_with_probe(probe_weights, orig_acts[layer])

    lying_tids, lying_acts = load_activations(lying_path, layers=[layer])
    lying_parsed = [parse_lying_task_id(tid) for tid in lying_tids]
    lying_scores = score_with_probe(probe_weights, lying_acts[layer])

    for followup in LYING_FOLLOWUP_TYPES:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
        fig.suptitle(f"Probe score distributions ({selector}, {probe_name} probe, L{layer}, {followup})",
                     fontsize=13)

        conditions_plot = [
            ("no_lying", orig_scores, orig_parsed, None),
            ("lie_direct", lying_scores, lying_parsed, "lie_direct"),
            ("lie_roleplay", lying_scores, lying_parsed, "lie_roleplay"),
        ]

        for ax, (cond_name, scores, parsed, sys_type) in zip(axes, conditions_plot):
            if sys_type is None:
                # Original data
                correct_mask = np.array([ac == "correct" and ft == followup for _, ac, ft in parsed])
                incorrect_mask = np.array([ac == "incorrect" and ft == followup for _, ac, ft in parsed])
            else:
                correct_mask = np.array([ac == "correct" and sp == sys_type and ft == followup
                                         for _, ac, sp, ft in parsed])
                incorrect_mask = np.array([ac == "incorrect" and sp == sys_type and ft == followup
                                           for _, ac, sp, ft in parsed])

            correct_s = scores[correct_mask]
            incorrect_s = scores[incorrect_mask]

            parts = ax.violinplot([incorrect_s, correct_s], positions=[0, 1], showmedians=True)
            for pc, color in zip(parts["bodies"], ["#e74c3c", "#2ecc71"]):
                pc.set_facecolor(color)
                pc.set_alpha(0.6)

            d = compute_metrics(correct_s, incorrect_s)["cohens_d"]
            ax.set_title(f"{cond_name}\nd = {d:.2f}")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["incorrect", "correct"])

        axes[0].set_ylabel("Probe score")
        fig.tight_layout()
        fig.savefig(ASSETS_DIR / f"plot_031226_lying_violins_{followup}.png", dpi=150)
        plt.close(fig)
        print(f"Saved: lying_violins_{followup}")


def plot_comprehensive_auc_heatmap(assistant_results: dict, lying_results: dict):
    """Plot 6: Comprehensive AUC heatmap — all selectors × layers for best probe."""
    probe_name = "tb-2"

    for followup in LYING_FOLLOWUP_TYPES:
        # Rows: selector × condition combos
        row_labels = []
        matrix_rows = []

        # No-lying assistant selectors
        for sel in ["assistant_mean", "assistant_tb:-1", "assistant_tb:-5"]:
            if sel in assistant_results and probe_name in assistant_results[sel]:
                if followup in assistant_results[sel][probe_name]:
                    row = [assistant_results[sel][probe_name][followup][str(l)]["auc"] for l in LAYERS]
                    row_labels.append(f"{sel} (no lying)")
                    matrix_rows.append(row)

        # Lying selectors: both tb and assistant
        for sel in ["turn_boundary:-2", "turn_boundary:-5", "assistant_mean", "assistant_tb:-1", "assistant_tb:-5"]:
            for sys_type in SYSTEM_PROMPT_TYPES:
                if sel in lying_results and probe_name in lying_results[sel]:
                    if sys_type in lying_results[sel][probe_name]:
                        if followup in lying_results[sel][probe_name][sys_type]:
                            row = [lying_results[sel][probe_name][sys_type][followup][str(l)]["auc"] for l in LAYERS]
                            row_labels.append(f"{sel} ({sys_type})")
                            matrix_rows.append(row)

        if not matrix_rows:
            continue

        matrix = np.array(matrix_rows)

        fig, ax = plt.subplots(figsize=(10, max(4, len(row_labels) * 0.6 + 1)))
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")

        ax.set_xticks(range(len(LAYERS)))
        ax.set_xticklabels([f"L{l}" for l in LAYERS])
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=9)

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                text_color = "white" if val < 0.4 or val > 0.85 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=text_color)

        ax.set_title(f"AUC heatmap: all selectors ({followup}, {probe_name} probe)", fontsize=13)
        fig.colorbar(im, ax=ax, label="AUC", shrink=0.8)
        fig.tight_layout()
        fig.savefig(ASSETS_DIR / f"plot_031226_comprehensive_auc_{followup}_{probe_name}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: comprehensive_auc_{followup}_{probe_name}")


def plot_mean_score_shift(lying_results: dict, assistant_results: dict):
    """Plot 7: Mean probe score shift — shows whether lying shifts the overall distribution."""
    probe_name = "tb-2"
    followup = "presupposes"
    layer = 46

    selectors = ["assistant_tb:-1", "turn_boundary:-2"]
    available = [s for s in selectors if s in lying_results]

    if not available:
        return

    fig, axes = plt.subplots(1, len(available), figsize=(7 * len(available), 5), sharey=True)
    if len(available) == 1:
        axes = [axes]
    fig.suptitle(f"Mean probe score by condition ({followup}, {probe_name} probe, L{layer})", fontsize=13)

    for ax, selector in zip(axes, available):
        conditions = ["no_lying", "lie_direct", "lie_roleplay"]
        x = np.arange(len(conditions))

        correct_means = []
        incorrect_means = []

        for cond in conditions:
            if cond == "no_lying":
                if selector in assistant_results and probe_name in assistant_results[selector]:
                    if followup in assistant_results[selector][probe_name]:
                        data = assistant_results[selector][probe_name][followup][str(layer)]
                        correct_means.append(data["mean_correct"])
                        incorrect_means.append(data["mean_incorrect"])
                    else:
                        correct_means.append(np.nan)
                        incorrect_means.append(np.nan)
                else:
                    correct_means.append(np.nan)
                    incorrect_means.append(np.nan)
            else:
                if selector in lying_results and probe_name in lying_results[selector]:
                    data = lying_results[selector][probe_name][cond][followup][str(layer)]
                    correct_means.append(data["mean_correct"])
                    incorrect_means.append(data["mean_incorrect"])
                else:
                    correct_means.append(np.nan)
                    incorrect_means.append(np.nan)

        width = 0.35
        ax.bar(x - width / 2, correct_means, width, label="correct", color="#2ecc71",
               edgecolor="black", linewidth=0.5)
        ax.bar(x + width / 2, incorrect_means, width, label="incorrect", color="#e74c3c",
               edgecolor="black", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(conditions, fontsize=9)
        ax.set_title(selector)
        ax.legend(fontsize=9)

    axes[0].set_ylabel("Mean probe score")
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / f"plot_031226_mean_score_shift_{followup}_{probe_name}.png", dpi=150)
    plt.close(fig)
    print(f"Saved: mean_score_shift_{followup}_{probe_name}")


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    # Run analyses
    assistant_results = analyze_assistant_selectors()
    lying_results = analyze_lying_conversations()

    # Save results JSON
    combined = {
        "assistant_selectors_no_lying": assistant_results,
        "lying_conversations": lying_results,
    }
    out_path = OUTPUT_DIR / "error_prefill_followup_results.json"
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nSaved results: {out_path}")

    # Generate plots
    print("\n" + "=" * 80)
    print("Generating plots")
    print("=" * 80)

    plot_assistant_effect_sizes(assistant_results)
    plot_assistant_vs_followup_comparison(assistant_results)
    plot_lying_effect_comparison(lying_results, assistant_results)
    plot_lying_heatmaps(lying_results, assistant_results)
    plot_lying_score_distributions(lying_results, assistant_results)
    plot_comprehensive_auc_heatmap(assistant_results, lying_results)
    plot_mean_score_shift(lying_results, assistant_results)

    print("\nDone!")


if __name__ == "__main__":
    main()
