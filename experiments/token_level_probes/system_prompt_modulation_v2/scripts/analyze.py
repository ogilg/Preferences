"""Analyze system prompt modulation v2: lying personas + evil personas.

Produces:
1. Critical span scores by system prompt (truth and harm domains)
2. EOT scores by system prompt (truth and harm domains)
3. Paired score differences vs neutral
4. Summary statistics table

Usage:
    python experiments/token_level_probes/system_prompt_modulation_v2/scripts/analyze.py
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

RESULTS_PATH = Path("experiments/token_level_probes/system_prompt_modulation_v2/scoring_results.json")
ASSETS_DIR = Path("experiments/token_level_probes/system_prompt_modulation_v2/assets")

# Best probe per domain (from parent experiment)
BEST_PROBE = {
    "truth": "task_mean_L32",
    "harm": "task_mean_L39",
}

# System prompt ordering for plots
TRUTH_PROMPT_ORDER = [
    "truthful", "neutral", "unreliable_narrator", "contrarian",
    "opposite_day", "lie_directive", "pathological_liar", "con_artist", "gaslighter",
]
HARM_PROMPT_ORDER = ["safe", "neutral", "unrestricted", "sinister_ai", "sadist"]

TRUTH_PROMPT_LABELS = {
    "truthful": "Truthful",
    "neutral": "Neutral",
    "unreliable_narrator": "Unreliable\nnarrator",
    "contrarian": "Contrarian",
    "opposite_day": "Opposite\nday",
    "lie_directive": "Lie\ndirective",
    "pathological_liar": "Pathological\nliar",
    "con_artist": "Con\nartist",
    "gaslighter": "Gaslighter",
}
HARM_PROMPT_LABELS = {
    "safe": "Safe",
    "neutral": "Neutral",
    "unrestricted": "Unrestricted",
    "sinister_ai": "Sinister AI",
    "sadist": "Sadist",
}

CONDITION_COLORS = {
    "true": "#2196F3",
    "false": "#F44336",
    "nonsense": "#9E9E9E",
    "benign": "#4CAF50",
    "harmful": "#F44336",
}


def load_results():
    data = json.load(open(RESULTS_PATH))
    return data["items"]


def group_by(items, *keys):
    groups = defaultdict(list)
    for item in items:
        key = tuple(item[k] for k in keys)
        groups[key] = groups.get(key, [])
        groups[key].append(item)
    return groups


def cohen_d(a, b):
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na - 1) * np.std(a, ddof=1)**2 + (nb - 1) * np.std(b, ddof=1)**2) / (na + nb - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled_std


def paired_stats(a, b):
    diff = np.array(a) - np.array(b)
    t_stat, p_val = stats.ttest_rel(a, b)
    d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0.0
    return {"mean_diff": float(np.mean(diff)), "d": float(d), "t": float(t_stat), "p": float(p_val), "n": len(diff)}


def plot_by_system_prompt(items, domain, score_key, prompt_order, prompt_labels, condition_colors, title, filename):
    probe = BEST_PROBE[domain]
    fig, ax = plt.subplots(figsize=(max(10, len(prompt_order) * 1.2), 6))

    conditions = ["true", "false"] if domain == "truth" else ["benign", "harmful"]
    x_positions = np.arange(len(prompt_order))
    width = 0.35

    for ci, condition in enumerate(conditions):
        means, stds, ns = [], [], []
        for sp in prompt_order:
            vals = [item[score_key][probe] for item in items
                    if item["system_prompt"] == sp and item["condition"] == condition]
            means.append(np.mean(vals))
            stds.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))
            ns.append(len(vals))

        offset = (ci - 0.5) * width
        bars = ax.bar(x_positions + offset, means, width * 0.9, yerr=stds,
                      color=condition_colors[condition], alpha=0.7, label=f"{condition} (n={ns[0]})",
                      capsize=3, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([prompt_labels[sp] for sp in prompt_order], fontsize=9)
    ax.set_ylabel(f"Mean probe score ({probe})")
    ax.set_title(title)
    ax.legend()
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {filename}")


def compute_statistics_table(items, domain, score_key, prompt_order):
    probe = BEST_PROBE[domain]
    conditions = ["true", "false"] if domain == "truth" else ["benign", "harmful"]

    rows = []
    for sp in prompt_order:
        cond_vals = {}
        for condition in conditions:
            vals = [item[score_key][probe] for item in items
                    if item["system_prompt"] == sp and item["condition"] == condition]
            cond_vals[condition] = vals

        a, b = conditions[0], conditions[1]
        d = cohen_d(cond_vals[a], cond_vals[b])
        _, p = stats.mannwhitneyu(cond_vals[a], cond_vals[b], alternative="two-sided")

        row = {"system_prompt": sp}
        for c in conditions:
            row[f"{c}_mean"] = np.mean(cond_vals[c])
            row[f"{c}_std"] = np.std(cond_vals[c], ddof=1)
        row["d"] = d
        row["p"] = p
        rows.append(row)

    return rows


def compute_paired_vs_neutral(items, domain, score_key, prompt_order):
    """For each base item × condition, compute score difference between each system prompt and neutral."""
    probe = BEST_PROBE[domain]
    conditions = ["true", "false"] if domain == "truth" else ["benign", "harmful"]

    # Index items by (base_id_without_sysprompt, condition) -> {system_prompt: score}
    # The id format is like truth_0_true_assistant_{system_prompt}
    item_scores = defaultdict(dict)
    for item in items:
        sp = item["system_prompt"]
        # Strip the system_prompt suffix to get base id
        base_id = item["id"].rsplit(f"_{sp}", 1)[0]
        key = (base_id, item["condition"])
        item_scores[key][sp] = item[score_key][probe]

    results = {}
    for sp in prompt_order:
        if sp == "neutral":
            continue
        diffs = {c: [] for c in conditions}
        for (base_id, condition), scores_by_sp in item_scores.items():
            if condition not in conditions:
                continue
            if sp in scores_by_sp and "neutral" in scores_by_sp:
                diffs[condition].append(scores_by_sp[sp] - scores_by_sp["neutral"])

        results[sp] = {}
        for c in conditions:
            if diffs[c]:
                arr = np.array(diffs[c])
                t_stat, p_val = stats.ttest_1samp(arr, 0)
                results[sp][c] = {
                    "mean_diff": float(np.mean(arr)),
                    "std_diff": float(np.std(arr, ddof=1)),
                    "t": float(t_stat),
                    "p": float(p_val),
                    "n": len(arr),
                    "d": float(np.mean(arr) / np.std(arr, ddof=1)) if np.std(arr, ddof=1) > 0 else 0.0,
                }

    return results


def plot_paired_differences(paired_results, domain, prompt_order, prompt_labels, condition_colors, title, filename):
    conditions = ["true", "false"] if domain == "truth" else ["benign", "harmful"]
    prompts = [sp for sp in prompt_order if sp != "neutral"]

    fig, ax = plt.subplots(figsize=(max(10, len(prompts) * 1.2), 6))
    x_positions = np.arange(len(prompts))
    width = 0.35

    for ci, condition in enumerate(conditions):
        means = [paired_results[sp][condition]["mean_diff"] for sp in prompts]
        sems = [paired_results[sp][condition]["std_diff"] / np.sqrt(paired_results[sp][condition]["n"]) for sp in prompts]

        offset = (ci - 0.5) * width
        ax.bar(x_positions + offset, means, width * 0.9, yerr=sems,
               color=condition_colors[condition], alpha=0.7, label=condition,
               capsize=3, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([prompt_labels[sp] for sp in prompts], fontsize=9)
    ax.set_ylabel(f"Score difference vs neutral ({BEST_PROBE[domain]})")
    ax.set_title(title)
    ax.legend()
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {filename}")


def print_stats_table(rows, domain, score_type):
    conditions = ["true", "false"] if domain == "truth" else ["benign", "harmful"]
    c1, c2 = conditions

    print(f"\n{'='*80}")
    print(f"{domain.upper()} — {score_type} ({BEST_PROBE[domain]})")
    print(f"{'='*80}")
    print(f"{'System prompt':<20} {c1:>12} {c2:>12} {'d':>8} {'p':>12}")
    print("-" * 70)
    for row in rows:
        c1_str = f"{row[f'{c1}_mean']:.2f}±{row[f'{c1}_std']:.2f}"
        c2_str = f"{row[f'{c2}_mean']:.2f}±{row[f'{c2}_std']:.2f}"
        p_str = f"{row['p']:.2e}" if row['p'] < 0.001 else f"{row['p']:.4f}"
        print(f"{row['system_prompt']:<20} {c1_str:>12} {c2_str:>12} {row['d']:>8.2f} {p_str:>12}")


def print_paired_table(paired_results, domain, prompt_order, score_type):
    conditions = ["true", "false"] if domain == "truth" else ["benign", "harmful"]
    prompts = [sp for sp in prompt_order if sp != "neutral"]

    print(f"\n{'='*80}")
    print(f"{domain.upper()} — Paired difference vs neutral — {score_type}")
    print(f"{'='*80}")
    for c in conditions:
        print(f"\n  Condition: {c}")
        print(f"  {'System prompt':<20} {'mean diff':>10} {'d':>8} {'p':>12}")
        print(f"  {'-'*55}")
        for sp in prompts:
            r = paired_results[sp][c]
            p_str = f"{r['p']:.2e}" if r['p'] < 0.001 else f"{r['p']:.4f}"
            print(f"  {sp:<20} {r['mean_diff']:>+10.2f} {r['d']:>+8.2f} {p_str:>12}")


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    items = load_results()
    print(f"Loaded {len(items)} items")

    truth_items = [item for item in items if item["domain"] == "truth"]
    harm_items = [item for item in items if item["domain"] == "harm"]
    print(f"  Truth: {len(truth_items)}, Harm: {len(harm_items)}")

    # === Critical span analysis ===
    print("\n--- CRITICAL SPAN ANALYSIS ---")

    truth_cs_stats = compute_statistics_table(truth_items, "truth", "critical_span_mean_scores", TRUTH_PROMPT_ORDER)
    harm_cs_stats = compute_statistics_table(harm_items, "harm", "critical_span_mean_scores", HARM_PROMPT_ORDER)
    print_stats_table(truth_cs_stats, "truth", "critical span")
    print_stats_table(harm_cs_stats, "harm", "critical span")

    plot_by_system_prompt(truth_items, "truth", "critical_span_mean_scores",
                         TRUTH_PROMPT_ORDER, TRUTH_PROMPT_LABELS, CONDITION_COLORS,
                         "Truth: Critical span scores by system prompt",
                         "plot_031426_truth_critical_span_by_sysprompt.png")

    plot_by_system_prompt(harm_items, "harm", "critical_span_mean_scores",
                         HARM_PROMPT_ORDER, HARM_PROMPT_LABELS, CONDITION_COLORS,
                         "Harm: Critical span scores by system prompt",
                         "plot_031426_harm_critical_span_by_sysprompt.png")

    # === EOT analysis ===
    print("\n--- EOT ANALYSIS ---")

    truth_eot_stats = compute_statistics_table(truth_items, "truth", "eot_scores", TRUTH_PROMPT_ORDER)
    harm_eot_stats = compute_statistics_table(harm_items, "harm", "eot_scores", HARM_PROMPT_ORDER)
    print_stats_table(truth_eot_stats, "truth", "EOT")
    print_stats_table(harm_eot_stats, "harm", "EOT")

    plot_by_system_prompt(truth_items, "truth", "eot_scores",
                         TRUTH_PROMPT_ORDER, TRUTH_PROMPT_LABELS, CONDITION_COLORS,
                         "Truth: EOT scores by system prompt",
                         "plot_031426_truth_eot_by_sysprompt.png")

    plot_by_system_prompt(harm_items, "harm", "eot_scores",
                         HARM_PROMPT_ORDER, HARM_PROMPT_LABELS, CONDITION_COLORS,
                         "Harm: EOT scores by system prompt",
                         "plot_031426_harm_eot_by_sysprompt.png")

    # === Paired differences vs neutral ===
    print("\n--- PAIRED DIFFERENCES VS NEUTRAL ---")

    truth_cs_paired = compute_paired_vs_neutral(truth_items, "truth", "critical_span_mean_scores", TRUTH_PROMPT_ORDER)
    harm_cs_paired = compute_paired_vs_neutral(harm_items, "harm", "critical_span_mean_scores", HARM_PROMPT_ORDER)
    truth_eot_paired = compute_paired_vs_neutral(truth_items, "truth", "eot_scores", TRUTH_PROMPT_ORDER)
    harm_eot_paired = compute_paired_vs_neutral(harm_items, "harm", "eot_scores", HARM_PROMPT_ORDER)

    print_paired_table(truth_cs_paired, "truth", TRUTH_PROMPT_ORDER, "critical span")
    print_paired_table(harm_cs_paired, "harm", HARM_PROMPT_ORDER, "critical span")
    print_paired_table(truth_eot_paired, "truth", TRUTH_PROMPT_ORDER, "EOT")
    print_paired_table(harm_eot_paired, "harm", HARM_PROMPT_ORDER, "EOT")

    plot_paired_differences(truth_cs_paired, "truth", TRUTH_PROMPT_ORDER, TRUTH_PROMPT_LABELS, CONDITION_COLORS,
                           "Truth: Paired score difference vs neutral (critical span)",
                           "plot_031426_truth_paired_diff_critical_span.png")

    plot_paired_differences(truth_eot_paired, "truth", TRUTH_PROMPT_ORDER, TRUTH_PROMPT_LABELS, CONDITION_COLORS,
                           "Truth: Paired score difference vs neutral (EOT)",
                           "plot_031426_truth_paired_diff_eot.png")

    plot_paired_differences(harm_cs_paired, "harm", HARM_PROMPT_ORDER, HARM_PROMPT_LABELS, CONDITION_COLORS,
                           "Harm: Paired score difference vs neutral (critical span)",
                           "plot_031426_harm_paired_diff_critical_span.png")

    plot_paired_differences(harm_eot_paired, "harm", HARM_PROMPT_ORDER, HARM_PROMPT_LABELS, CONDITION_COLORS,
                           "Harm: Paired score difference vs neutral (EOT)",
                           "plot_031426_harm_paired_diff_eot.png")

    # === Save summary JSON ===
    summary = {
        "truth_critical_span": truth_cs_stats,
        "truth_eot": truth_eot_stats,
        "harm_critical_span": harm_cs_stats,
        "harm_eot": harm_eot_stats,
        "truth_paired_critical_span": {k: v for k, v in truth_cs_paired.items()},
        "truth_paired_eot": {k: v for k, v in truth_eot_paired.items()},
        "harm_paired_critical_span": {k: v for k, v in harm_cs_paired.items()},
        "harm_paired_eot": {k: v for k, v in harm_eot_paired.items()},
    }
    with open(ASSETS_DIR / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved analysis_summary.json")


if __name__ == "__main__":
    main()
