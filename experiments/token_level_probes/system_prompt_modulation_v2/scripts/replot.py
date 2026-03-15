"""Replot all system prompt modulation v2 results as violin plots."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

RESULTS_PATH = Path("experiments/token_level_probes/system_prompt_modulation_v2/scoring_results.json")
ASSETS_DIR = Path("experiments/token_level_probes/system_prompt_modulation_v2/assets")

BEST_PROBE = {"truth": "task_mean_L32", "harm": "task_mean_L39"}

TRUTH_PROMPT_ORDER = [
    "truthful", "neutral", "unreliable_narrator", "contrarian",
    "opposite_day", "lie_directive", "pathological_liar", "con_artist", "gaslighter",
]
HARM_PROMPT_ORDER = ["safe", "neutral", "unrestricted", "sinister_ai", "sadist"]

TRUTH_PROMPT_LABELS = {
    "truthful": "Truthful", "neutral": "Neutral",
    "unreliable_narrator": "Unreliable\nnarrator", "contrarian": "Contrarian",
    "opposite_day": "Opposite\nday", "lie_directive": "Lie\ndirective",
    "pathological_liar": "Pathological\nliar", "con_artist": "Con\nartist",
    "gaslighter": "Gaslighter",
}
HARM_PROMPT_LABELS = {
    "safe": "Safe", "neutral": "Neutral", "unrestricted": "Unrestricted",
    "sinister_ai": "Sinister AI", "sadist": "Sadist",
}

CONDITION_COLORS = {
    "true": "#2196F3", "false": "#F44336",
    "benign": "#4CAF50", "harmful": "#F44336",
}


def cohen_d(a, b):
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na - 1) * np.std(a, ddof=1)**2 + (nb - 1) * np.std(b, ddof=1)**2) / (na + nb - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled_std


def plot_violins(items, domain, score_key, prompt_order, prompt_labels, title, filename):
    probe = BEST_PROBE[domain]
    conditions = ["true", "false"] if domain == "truth" else ["benign", "harmful"]
    n_prompts = len(prompt_order)

    fig, ax = plt.subplots(figsize=(max(12, n_prompts * 1.4), 6))

    for pi, sp in enumerate(prompt_order):
        for ci, condition in enumerate(conditions):
            vals = [item[score_key][probe] for item in items
                    if item["system_prompt"] == sp and item["condition"] == condition]
            if not vals:
                continue

            offset = (ci - 0.5) * 0.35
            pos = pi + offset
            color = CONDITION_COLORS[condition]

            parts = ax.violinplot([vals], positions=[pos], showmeans=True,
                                  showextrema=False, widths=0.3)
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.5)
                pc.set_edgecolor("none")
            parts["cmeans"].set_color("black")
            parts["cmeans"].set_linewidth(1.5)

            ax.scatter(np.random.normal(pos, 0.03, len(vals)), vals,
                      alpha=0.25, s=6, color=color, zorder=3)

        # Add d annotation below
        c0_vals = [item[score_key][probe] for item in items
                   if item["system_prompt"] == sp and item["condition"] == conditions[0]]
        c1_vals = [item[score_key][probe] for item in items
                   if item["system_prompt"] == sp and item["condition"] == conditions[1]]
        if c0_vals and c1_vals:
            d = cohen_d(c0_vals, c1_vals)
            y_min = ax.get_ylim()[0] if pi > 0 else min(min(c0_vals), min(c1_vals))
            ax.annotate(f"d={d:.2f}", xy=(pi, 0), xytext=(pi, 0),
                       fontsize=7, ha="center", color="gray", alpha=0.7,
                       xycoords=("data", "axes fraction"), textcoords=("data", "axes fraction"),
                       va="bottom")

    # Legend
    for condition in conditions:
        ax.scatter([], [], color=CONDITION_COLORS[condition], alpha=0.6, s=30, label=condition)
    ax.legend(loc="upper right", fontsize=10)

    ax.set_xticks(range(n_prompts))
    ax.set_xticklabels([prompt_labels[sp] for sp in prompt_order], fontsize=9)
    ax.set_ylabel(f"Probe score ({probe})")
    ax.set_title(title, fontsize=13)
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {filename}")


def plot_paired_violins(items, domain, score_key, prompt_order, prompt_labels, title, filename):
    """Violin plots of per-item score differences vs neutral."""
    probe = BEST_PROBE[domain]
    conditions = ["true", "false"] if domain == "truth" else ["benign", "harmful"]
    prompts = [sp for sp in prompt_order if sp != "neutral"]

    # Index items
    item_scores = defaultdict(dict)
    for item in items:
        sp = item["system_prompt"]
        base_id = item["id"].rsplit(f"_{sp}", 1)[0]
        key = (base_id, item["condition"])
        item_scores[key][sp] = item[score_key][probe]

    fig, ax = plt.subplots(figsize=(max(12, len(prompts) * 1.4), 6))

    for pi, sp in enumerate(prompts):
        for ci, condition in enumerate(conditions):
            diffs = []
            for (base_id, cond), scores_by_sp in item_scores.items():
                if cond == condition and sp in scores_by_sp and "neutral" in scores_by_sp:
                    diffs.append(scores_by_sp[sp] - scores_by_sp["neutral"])
            if not diffs:
                continue

            offset = (ci - 0.5) * 0.35
            pos = pi + offset
            color = CONDITION_COLORS[condition]

            parts = ax.violinplot([diffs], positions=[pos], showmeans=True,
                                  showextrema=False, widths=0.3)
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.5)
                pc.set_edgecolor("none")
            parts["cmeans"].set_color("black")
            parts["cmeans"].set_linewidth(1.5)

            ax.scatter(np.random.normal(pos, 0.03, len(diffs)), diffs,
                      alpha=0.25, s=6, color=color, zorder=3)

    for condition in conditions:
        ax.scatter([], [], color=CONDITION_COLORS[condition], alpha=0.6, s=30, label=condition)
    ax.legend(loc="upper right" if "EOT" not in title else "lower left", fontsize=10)

    ax.set_xticks(range(len(prompts)))
    ax.set_xticklabels([prompt_labels[sp] for sp in prompts], fontsize=9)
    ax.set_ylabel(f"Score shift vs neutral ({probe})")
    ax.set_title(title, fontsize=13)
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {filename}")


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    items = json.load(open(RESULTS_PATH))["items"]
    truth_items = [i for i in items if i["domain"] == "truth"]
    harm_items = [i for i in items if i["domain"] == "harm"]

    # Main violin plots
    plot_violins(truth_items, "truth", "critical_span_mean_scores",
                 TRUTH_PROMPT_ORDER, TRUTH_PROMPT_LABELS,
                 "Does the model still distinguish true from false under lying prompts? (critical span)",
                 "plot_031426_truth_critical_span_by_sysprompt.png")

    plot_violins(truth_items, "truth", "eot_scores",
                 TRUTH_PROMPT_ORDER, TRUTH_PROMPT_LABELS,
                 "Does the model still distinguish true from false under lying prompts? (end-of-turn)",
                 "plot_031426_truth_eot_by_sysprompt.png")

    plot_violins(harm_items, "harm", "critical_span_mean_scores",
                 HARM_PROMPT_ORDER, HARM_PROMPT_LABELS,
                 "Do evil personas eliminate the benign/harmful distinction? (critical span)",
                 "plot_031426_harm_critical_span_by_sysprompt.png")

    plot_violins(harm_items, "harm", "eot_scores",
                 HARM_PROMPT_ORDER, HARM_PROMPT_LABELS,
                 "Do evil personas eliminate the benign/harmful distinction? (end-of-turn)",
                 "plot_031426_harm_eot_by_sysprompt.png")

    # Paired diff violin plots
    plot_paired_violins(truth_items, "truth", "critical_span_mean_scores",
                        TRUTH_PROMPT_ORDER, TRUTH_PROMPT_LABELS,
                        "How much does each lying prompt shift scores? (critical span, paired vs neutral)",
                        "plot_031426_truth_paired_diff_critical_span.png")

    plot_paired_violins(truth_items, "truth", "eot_scores",
                        TRUTH_PROMPT_ORDER, TRUTH_PROMPT_LABELS,
                        "How much does each lying prompt shift scores? (end-of-turn, paired vs neutral)",
                        "plot_031426_truth_paired_diff_eot.png")

    plot_paired_violins(harm_items, "harm", "critical_span_mean_scores",
                        HARM_PROMPT_ORDER, HARM_PROMPT_LABELS,
                        "How much does each evil persona shift scores? (critical span, paired vs neutral)",
                        "plot_031426_harm_paired_diff_critical_span.png")

    plot_paired_violins(harm_items, "harm", "eot_scores",
                        HARM_PROMPT_ORDER, HARM_PROMPT_LABELS,
                        "How much does each evil persona shift scores? (end-of-turn, paired vs neutral)",
                        "plot_031426_harm_paired_diff_eot.png")


if __name__ == "__main__":
    main()
