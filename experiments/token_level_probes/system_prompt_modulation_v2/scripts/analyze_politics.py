"""Analyze politics system prompt modulation with diverse prompts + logprob confound test."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

RESULTS_PATH = Path("experiments/token_level_probes/system_prompt_modulation_v2/politics_scoring_results.json")
ASSETS_DIR = Path("experiments/token_level_probes/system_prompt_modulation_v2/assets")

PROBE = "task_mean_L39"

PROMPT_ORDER = [
    "socialist", "democrat", "centrist", "neutral", "apolitical",
    "libertarian", "republican", "nationalist", "contrarian",
]
PROMPT_LABELS = {
    "socialist": "Socialist", "democrat": "Democrat", "centrist": "Centrist",
    "neutral": "Neutral", "apolitical": "Apolitical",
    "libertarian": "Libertarian", "republican": "Republican",
    "nationalist": "Nationalist", "contrarian": "Contrarian",
}
CONDITION_COLORS = {"left": "#2196F3", "right": "#F44336", "nonsense": "#9E9E9E"}


def cohen_d(a, b):
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na - 1) * np.std(a, ddof=1)**2 + (nb - 1) * np.std(b, ddof=1)**2) / (na + nb - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled_std


def plot_violins(items, score_key, title, filename):
    conditions = ["left", "right"]
    n_prompts = len(PROMPT_ORDER)

    fig, ax = plt.subplots(figsize=(max(12, n_prompts * 1.4), 6))

    for pi, sp in enumerate(PROMPT_ORDER):
        for ci, condition in enumerate(conditions):
            vals = [item[score_key][PROBE] for item in items
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

        left_vals = [item[score_key][PROBE] for item in items
                     if item["system_prompt"] == sp and item["condition"] == "left"]
        right_vals = [item[score_key][PROBE] for item in items
                      if item["system_prompt"] == sp and item["condition"] == "right"]
        if left_vals and right_vals:
            d = cohen_d(left_vals, right_vals)
            ax.annotate(f"d={d:.2f}", xy=(pi, 0), xytext=(pi, 0),
                       fontsize=7, ha="center", color="gray", alpha=0.7,
                       xycoords=("data", "axes fraction"), textcoords=("data", "axes fraction"),
                       va="bottom")

    for condition in conditions:
        ax.scatter([], [], color=CONDITION_COLORS[condition], alpha=0.6, s=30, label=condition)
    ax.legend(loc="upper right", fontsize=10)

    ax.set_xticks(range(n_prompts))
    ax.set_xticklabels([PROMPT_LABELS[sp] for sp in PROMPT_ORDER], fontsize=9)
    ax.set_ylabel(f"Probe score ({PROBE})")
    ax.set_title(title, fontsize=13)
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {filename}")


def logprob_analysis(items):
    """Test the logprob confound: does probe score correlate with logprob after controlling for condition?"""
    print(f"\n{'='*80}")
    print("LOGPROB CONFOUND ANALYSIS")
    print(f"{'='*80}")

    lr_items = [i for i in items if i["condition"] in ("left", "right")]

    # Overall correlation: probe score vs logprob
    probe_scores = [i["critical_span_mean_scores"][PROBE] for i in lr_items]
    logprobs = [i["critical_span_mean_logprob"] for i in lr_items]
    r, p = stats.pearsonr(probe_scores, logprobs)
    print(f"\nOverall correlation (probe score vs logprob): r={r:.3f}, p={p:.2e}")

    eot_scores = [i["eot_scores"][PROBE] for i in lr_items]
    eot_logprobs = [i["eot_logprob"] for i in lr_items]
    r_eot, p_eot = stats.pearsonr(eot_scores, eot_logprobs)
    print(f"Overall correlation (EOT score vs EOT logprob): r={r_eot:.3f}, p={p_eot:.2e}")

    # Within-condition correlation
    for condition in ["left", "right"]:
        cond_items = [i for i in lr_items if i["condition"] == condition]
        ps = [i["critical_span_mean_scores"][PROBE] for i in cond_items]
        lp = [i["critical_span_mean_logprob"] for i in cond_items]
        r, p = stats.pearsonr(ps, lp)
        print(f"  Within {condition} (critical span): r={r:.3f}, p={p:.2e}")

    # Per system prompt: does logprob explain the probe score difference between left and right?
    print(f"\n  Per system prompt — logprob vs probe score means:")
    print(f"  {'Prompt':<15} {'probe d':>8} {'logprob d':>10} {'probe corr with lp':>20}")
    print(f"  {'-'*55}")
    for sp in PROMPT_ORDER:
        sp_items = [i for i in lr_items if i["system_prompt"] == sp]
        left_probe = [i["critical_span_mean_scores"][PROBE] for i in sp_items if i["condition"] == "left"]
        right_probe = [i["critical_span_mean_scores"][PROBE] for i in sp_items if i["condition"] == "right"]
        left_lp = [i["critical_span_mean_logprob"] for i in sp_items if i["condition"] == "left"]
        right_lp = [i["critical_span_mean_logprob"] for i in sp_items if i["condition"] == "right"]

        if left_probe and right_probe and left_lp and right_lp:
            d_probe = cohen_d(left_probe, right_probe)
            d_lp = cohen_d(left_lp, right_lp)
            all_ps = left_probe + right_probe
            all_lp = left_lp + right_lp
            r, _ = stats.pearsonr(all_ps, all_lp)
            print(f"  {sp:<15} {d_probe:>+8.2f} {d_lp:>+10.2f} {r:>+20.3f}")

    # Scatter plot: probe score vs logprob, colored by condition
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, score_key, lp_key, label in [
        (axes[0], "critical_span_mean_scores", "critical_span_mean_logprob", "Critical span"),
        (axes[1], "eot_scores", "eot_logprob", "End-of-turn"),
    ]:
        for condition in ["left", "right"]:
            cond_items = [i for i in lr_items if i["condition"] == condition]
            ps = [i[score_key][PROBE] if isinstance(i[score_key], dict) else i[score_key] for i in cond_items]
            lp = [i[lp_key] for i in cond_items]
            ax.scatter(lp, ps, alpha=0.3, s=10, color=CONDITION_COLORS[condition], label=condition)

        ax.set_xlabel("Log probability")
        ax.set_ylabel(f"Probe score ({PROBE})")
        ax.set_title(f"{label}: probe score vs logprob")
        ax.legend()
        ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "plot_031426_politics_logprob_vs_probe.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved plot_031426_politics_logprob_vs_probe.png")


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    items = json.load(open(RESULTS_PATH))["items"]
    lr_items = [i for i in items if i["condition"] in ("left", "right")]
    print(f"Loaded {len(items)} items ({len(lr_items)} left/right)")

    # Stats table
    print(f"\n{'='*80}")
    print(f"POLITICS — critical span ({PROBE})")
    print(f"{'='*80}")
    print(f"{'System prompt':<15} {'left':>12} {'right':>12} {'d':>8} {'p':>12}")
    print("-" * 65)
    for sp in PROMPT_ORDER:
        left = [i["critical_span_mean_scores"][PROBE] for i in lr_items if i["system_prompt"] == sp and i["condition"] == "left"]
        right = [i["critical_span_mean_scores"][PROBE] for i in lr_items if i["system_prompt"] == sp and i["condition"] == "right"]
        if left and right:
            d = cohen_d(left, right)
            _, p = stats.mannwhitneyu(left, right, alternative="two-sided")
            p_str = f"{p:.2e}" if p < 0.001 else f"{p:.4f}"
            print(f"{sp:<15} {np.mean(left):>+6.2f}±{np.std(left,ddof=1):.2f} {np.mean(right):>+6.2f}±{np.std(right,ddof=1):.2f} {d:>+8.2f} {p_str:>12}")

    print(f"\n{'='*80}")
    print(f"POLITICS — EOT ({PROBE})")
    print(f"{'='*80}")
    print(f"{'System prompt':<15} {'left':>12} {'right':>12} {'d':>8} {'p':>12}")
    print("-" * 65)
    for sp in PROMPT_ORDER:
        left = [i["eot_scores"][PROBE] for i in lr_items if i["system_prompt"] == sp and i["condition"] == "left"]
        right = [i["eot_scores"][PROBE] for i in lr_items if i["system_prompt"] == sp and i["condition"] == "right"]
        if left and right:
            d = cohen_d(left, right)
            _, p = stats.mannwhitneyu(left, right, alternative="two-sided")
            p_str = f"{p:.2e}" if p < 0.001 else f"{p:.4f}"
            print(f"{sp:<15} {np.mean(left):>+6.2f}±{np.std(left,ddof=1):.2f} {np.mean(right):>+6.2f}±{np.std(right,ddof=1):.2f} {d:>+8.2f} {p_str:>12}")

    # Violin plots
    plot_violins(lr_items, "critical_span_mean_scores",
                 "How do political identity prompts shift left/right probe scores? (critical span)",
                 "plot_031426_politics_critical_span_by_sysprompt.png")

    plot_violins(lr_items, "eot_scores",
                 "How do political identity prompts shift left/right probe scores? (end-of-turn)",
                 "plot_031426_politics_eot_by_sysprompt.png")

    # Logprob analysis
    logprob_analysis(items)


if __name__ == "__main__":
    main()
