"""System prompt modulation analysis for truth and harm domains.

Analyzes whether system prompts shift probe scores for identical content,
extending the politics finding from the parent experiment.
"""

import json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel, wilcoxon


DATA_PATH = Path("experiments/token_level_probes/system_prompt_modulation/scoring_results.json")
ASSETS_DIR = Path("experiments/token_level_probes/system_prompt_modulation/assets")

ALL_PROBES = [
    f"{ps}_L{l}"
    for ps in ["tb-2", "tb-5", "task_mean"]
    for l in [32, 39, 53]
]

# Best probes per domain (from parent experiment)
BEST_PROBE = {
    "truth": "task_mean_L32",
    "harm": "task_mean_L39",
}

# System prompt contrasts per domain
DOMAIN_SP_CONTRAST = {
    "truth": ("truthful", "conspiracy"),
    "harm": ("safe", "unrestricted"),
}

DOMAIN_CONDITIONS = {
    "truth": ["true", "false", "nonsense"],
    "harm": ["harmful", "benign", "nonsense"],
}

# Colors: blue=positive, red=negative, gray=nonsense
CONDITION_COLORS = {
    "true": "#1f77b4",
    "false": "#d62728",
    "nonsense": "#999999",
    "benign": "#1f77b4",
    "harmful": "#d62728",
}

SP_ORDER = {
    "truth": ["truthful", "neutral", "conspiracy"],
    "harm": ["safe", "neutral", "unrestricted"],
}


def load_data() -> list[dict]:
    with open(DATA_PATH) as f:
        return json.load(f)["items"]


def cohens_d_paired(diffs: np.ndarray) -> float:
    return float(np.mean(diffs) / np.std(diffs, ddof=1))


def cohens_d_unpaired(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def extract_base_id(item: dict) -> str:
    """Extract base stimulus ID for pairing across system prompts.

    For items with source_id, use that. Otherwise parse from id field.
    Format: {domain}_{num}_{condition}_{turn}_{system_prompt}
    Base is {domain}_{num}.
    """
    if "source_id" in item:
        return item["source_id"]
    parts = item["id"].split("_")
    # domain_num_condition_turn_systemprompt
    return f"{parts[0]}_{parts[1]}"


def group_by_source(items: list[dict], domain: str) -> dict[tuple[str, str], dict[str, dict]]:
    """Group items by (base_id, condition) -> {system_prompt: item}."""
    groups: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for item in items:
        if item["domain"] != domain:
            continue
        base_id = extract_base_id(item)
        key = (base_id, item["condition"])
        groups[key][item["system_prompt"]] = item
    return dict(groups)


def plot_violin_by_system_prompt(items: list[dict], domain: str) -> None:
    """Violin plot: x=system prompt, color=condition, for best probe."""
    probe = BEST_PROBE[domain]
    conditions = DOMAIN_CONDITIONS[domain]
    sp_order = SP_ORDER[domain]

    fig, ax = plt.subplots(figsize=(10, 6))

    n_conditions = len(conditions)
    width = 0.25
    offsets = np.linspace(-width, width, n_conditions)

    for ci, cond in enumerate(conditions):
        positions = []
        data_lists = []
        for si, sp in enumerate(sp_order):
            scores = [
                item["critical_span_mean_scores"][probe]
                for item in items
                if item["domain"] == domain
                and item["condition"] == cond
                and item["system_prompt"] == sp
            ]
            positions.append(si + offsets[ci])
            data_lists.append(scores)

        parts = ax.violinplot(
            data_lists,
            positions=positions,
            widths=width * 0.9,
            showmeans=True,
            showextrema=False,
        )
        color = CONDITION_COLORS[cond]
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.4)
        parts["cmeans"].set_color(color)

        # Overlay individual points
        for pos, scores in zip(positions, data_lists):
            jitter = np.random.default_rng(42).normal(0, 0.02, len(scores))
            ax.scatter(
                pos + jitter, scores,
                c=color, alpha=0.3, s=10, zorder=3,
            )

    ax.set_xticks(range(len(sp_order)))
    ax.set_xticklabels(sp_order)
    ax.set_xlabel("System prompt")
    ax.set_ylabel(f"Critical span mean score ({probe})")
    ax.set_title(f"{domain.title()} domain: system prompt modulation ({probe})")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=CONDITION_COLORS[c], alpha=0.6, label=c)
        for c in conditions
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.3)

    plt.tight_layout()
    out = ASSETS_DIR / f"plot_031426_{domain}_system_prompt_modulation.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_paired_differences(items: list[dict]) -> None:
    """Violin plot of paired score differences (positive - negative SP) by condition."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, domain in zip(axes, ["truth", "harm"]):
        probe = BEST_PROBE[domain]
        sp_pos, sp_neg = DOMAIN_SP_CONTRAST[domain]
        conditions = DOMAIN_CONDITIONS[domain]
        groups = group_by_source(items, domain)

        condition_diffs: dict[str, list[float]] = {c: [] for c in conditions}
        for (source_id, cond), sp_dict in groups.items():
            if sp_pos not in sp_dict or sp_neg not in sp_dict:
                continue
            diff = (
                sp_dict[sp_pos]["critical_span_mean_scores"][probe]
                - sp_dict[sp_neg]["critical_span_mean_scores"][probe]
            )
            condition_diffs[cond].append(diff)

        data_lists = [condition_diffs[c] for c in conditions]
        positions = list(range(len(conditions)))

        parts = ax.violinplot(
            data_lists,
            positions=positions,
            showmeans=True,
            showextrema=False,
        )
        for ci, (pc, cond) in enumerate(zip(parts["bodies"], conditions)):
            color = CONDITION_COLORS[cond]
            pc.set_facecolor(color)
            pc.set_alpha(0.4)

        # Overlay points
        for pos, scores, cond in zip(positions, data_lists, conditions):
            jitter = np.random.default_rng(42).normal(0, 0.03, len(scores))
            ax.scatter(
                pos + jitter, scores,
                c=CONDITION_COLORS[cond], alpha=0.3, s=10, zorder=3,
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(conditions)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.set_xlabel("Condition")
        sp_label = f"{sp_pos} - {sp_neg}"
        ax.set_ylabel(f"Paired score difference ({sp_label})")
        ax.set_title(f"{domain.title()} ({probe})")

        # Print stats
        for cond in conditions:
            diffs = np.array(condition_diffs[cond])
            if len(diffs) < 3:
                continue
            mean = np.mean(diffs)
            d = cohens_d_paired(diffs)
            t_stat, t_p = ttest_rel(
                [sp_dict[sp_pos]["critical_span_mean_scores"][probe]
                 for (_, c), sp_dict in groups.items()
                 if c == cond and sp_pos in sp_dict and sp_neg in sp_dict],
                [sp_dict[sp_neg]["critical_span_mean_scores"][probe]
                 for (_, c), sp_dict in groups.items()
                 if c == cond and sp_pos in sp_dict and sp_neg in sp_dict],
            )
            print(f"  {domain}/{cond}: mean diff={mean:.3f}, d={d:.3f}, "
                  f"t={t_stat:.3f}, p={t_p:.2e}, n={len(diffs)}")

    plt.tight_layout()
    out = ASSETS_DIR / "plot_031426_paired_system_prompt_differences.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def compute_eot_vs_critical(items: list[dict]) -> dict:
    """Compare system prompt modulation at EOT vs critical span."""
    results = {}
    for domain in ["truth", "harm"]:
        probe = BEST_PROBE[domain]
        sp_pos, sp_neg = DOMAIN_SP_CONTRAST[domain]
        groups = group_by_source(items, domain)

        critical_diffs = []
        eot_diffs = []
        for (source_id, cond), sp_dict in groups.items():
            if sp_pos not in sp_dict or sp_neg not in sp_dict:
                continue
            critical_diffs.append(
                sp_dict[sp_pos]["critical_span_mean_scores"][probe]
                - sp_dict[sp_neg]["critical_span_mean_scores"][probe]
            )
            eot_diffs.append(
                sp_dict[sp_pos]["eot_scores"][probe]
                - sp_dict[sp_neg]["eot_scores"][probe]
            )

        critical_diffs = np.array(critical_diffs)
        eot_diffs = np.array(eot_diffs)

        results[domain] = {
            "critical_d": cohens_d_paired(critical_diffs),
            "eot_d": cohens_d_paired(eot_diffs),
            "critical_mean": float(np.mean(critical_diffs)),
            "eot_mean": float(np.mean(eot_diffs)),
            "n": len(critical_diffs),
        }
    return results


def plot_eot_vs_critical(eot_results: dict) -> None:
    """Bar plot comparing EOT d vs critical span d for each domain."""
    fig, ax = plt.subplots(figsize=(8, 5))

    domains = ["truth", "harm"]
    x = np.arange(len(domains))
    width = 0.35

    critical_ds = [eot_results[d]["critical_d"] for d in domains]
    eot_ds = [eot_results[d]["eot_d"] for d in domains]

    bars1 = ax.bar(x - width / 2, critical_ds, width, label="Critical span", color="#2ca02c", alpha=0.7)
    bars2 = ax.bar(x + width / 2, eot_ds, width, label="EOT token", color="#1f77b4", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels([d.title() for d in domains])
    ax.set_ylabel("Cohen's d (paired, system prompt effect)")
    ax.set_title("System prompt modulation: EOT vs critical span")
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            y_pos = height if height >= 0 else height
            va = "bottom" if height >= 0 else "top"
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, y_pos),
                ha="center", va=va, fontsize=9,
            )

    plt.tight_layout()
    out = ASSETS_DIR / "plot_031426_eot_vs_critical_span_modulation.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_cross_domain_comparison(eot_results: dict) -> None:
    """Compare effect sizes across domains (truth, harm, politics from parent)."""
    # Politics from parent: paired diff +8.07, t(76)=20.13
    # d = t / sqrt(n) for paired t-test
    politics_d = 20.13 / np.sqrt(77)

    fig, ax = plt.subplots(figsize=(8, 5))

    domains = ["Truth", "Harm", "Politics*"]
    critical_ds = [
        eot_results["truth"]["critical_d"],
        eot_results["harm"]["critical_d"],
        politics_d,
    ]

    colors = ["#2ca02c", "#d62728", "#1f77b4"]
    bars = ax.bar(range(len(domains)), critical_ds, color=colors, alpha=0.7, width=0.5)

    for bar, d_val in zip(bars, critical_ds):
        height = d_val
        va = "bottom" if height >= 0 else "top"
        ax.annotate(
            f"d={height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            ha="center", va=va, fontsize=10,
        )

    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels(domains)
    ax.set_ylabel("Cohen's d (paired, system prompt effect)")
    ax.set_title("System prompt modulation across domains\n(critical span scores)")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.3)

    # Set y-axis to start at 0
    y_min = min(0, min(critical_ds) - 0.3)
    y_max = max(critical_ds) + 0.5
    ax.set_ylim(y_min, y_max)

    ax.annotate(
        "*Politics d from parent experiment (t(76)=20.13, n=77)",
        xy=(0.5, 0.02), xycoords="axes fraction",
        fontsize=8, ha="center", color="gray",
    )

    plt.tight_layout()
    out = ASSETS_DIR / "plot_031426_cross_domain_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def print_full_statistics(items: list[dict]) -> None:
    """Print comprehensive statistics for all probes and domains."""
    print("\n" + "=" * 80)
    print("FULL STATISTICS: System Prompt Modulation")
    print("=" * 80)

    for domain in ["truth", "harm"]:
        sp_pos, sp_neg = DOMAIN_SP_CONTRAST[domain]
        groups = group_by_source(items, domain)
        print(f"\n{'─' * 60}")
        print(f"Domain: {domain.upper()} ({sp_pos} vs {sp_neg})")
        print(f"{'─' * 60}")

        # Overall (all conditions pooled)
        print(f"\n  All conditions pooled:")
        print(f"  {'Probe':<16} {'Position':<12} {'d':>7} {'t':>8} {'p':>12} {'Wilcoxon p':>12} {'n':>5} {'mean diff':>10}")
        for probe in ALL_PROBES:
            for position, score_key in [("critical", "critical_span_mean_scores"), ("eot", "eot_scores")]:
                pos_scores = []
                neg_scores = []
                for (source_id, cond), sp_dict in groups.items():
                    if sp_pos not in sp_dict or sp_neg not in sp_dict:
                        continue
                    pos_scores.append(sp_dict[sp_pos][score_key][probe])
                    neg_scores.append(sp_dict[sp_neg][score_key][probe])

                pos_arr = np.array(pos_scores)
                neg_arr = np.array(neg_scores)
                diffs = pos_arr - neg_arr

                if len(diffs) < 3:
                    continue

                d = cohens_d_paired(diffs)
                t_stat, t_p = ttest_rel(pos_arr, neg_arr)
                try:
                    _, w_p = wilcoxon(diffs)
                except ValueError:
                    w_p = 1.0
                mean_diff = float(np.mean(diffs))

                print(f"  {probe:<16} {position:<12} {d:>7.3f} {t_stat:>8.3f} {t_p:>12.2e} {w_p:>12.2e} {len(diffs):>5} {mean_diff:>10.3f}")

        # Per condition
        for cond in DOMAIN_CONDITIONS[domain]:
            print(f"\n  Condition: {cond}")
            print(f"  {'Probe':<16} {'Position':<12} {'d':>7} {'t':>8} {'p':>12} {'Wilcoxon p':>12} {'n':>5} {'mean diff':>10}")
            for probe in ALL_PROBES:
                for position, score_key in [("critical", "critical_span_mean_scores"), ("eot", "eot_scores")]:
                    pos_scores = []
                    neg_scores = []
                    for (source_id, cond_), sp_dict in groups.items():
                        if cond_ != cond:
                            continue
                        if sp_pos not in sp_dict or sp_neg not in sp_dict:
                            continue
                        pos_scores.append(sp_dict[sp_pos][score_key][probe])
                        neg_scores.append(sp_dict[sp_neg][score_key][probe])

                    pos_arr = np.array(pos_scores)
                    neg_arr = np.array(neg_scores)
                    diffs = pos_arr - neg_arr

                    if len(diffs) < 3:
                        continue

                    d = cohens_d_paired(diffs)
                    t_stat, t_p = ttest_rel(pos_arr, neg_arr)
                    try:
                        _, w_p = wilcoxon(diffs)
                    except ValueError:
                        w_p = 1.0
                    mean_diff = float(np.mean(diffs))

                    print(f"  {probe:<16} {position:<12} {d:>7.3f} {t_stat:>8.3f} {t_p:>12.2e} {w_p:>12.2e} {len(diffs):>5} {mean_diff:>10.3f}")


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    items = load_data()
    print(f"Loaded {len(items)} items")

    # Count by domain
    domain_counts = defaultdict(int)
    for item in items:
        domain_counts[(item["domain"], item["condition"], item["system_prompt"])] += 1
    print("\nItem counts by (domain, condition, system_prompt):")
    for key in sorted(domain_counts.keys()):
        print(f"  {key}: {domain_counts[key]}")

    # 1. System prompt modulation violin plots
    print("\n--- Plot 1: System prompt modulation violins ---")
    for domain in ["truth", "harm"]:
        plot_violin_by_system_prompt(items, domain)

    # 2. Paired score differences
    print("\n--- Plot 2: Paired score differences ---")
    plot_paired_differences(items)

    # 3. EOT vs critical span modulation
    print("\n--- Plot 3: EOT vs critical span ---")
    eot_results = compute_eot_vs_critical(items)
    for domain, res in eot_results.items():
        print(f"  {domain}: critical d={res['critical_d']:.3f} (mean={res['critical_mean']:.3f}), "
              f"eot d={res['eot_d']:.3f} (mean={res['eot_mean']:.3f}), n={res['n']}")
    plot_eot_vs_critical(eot_results)

    # 4. Cross-domain comparison
    print("\n--- Plot 4: Cross-domain comparison ---")
    plot_cross_domain_comparison(eot_results)

    # 5. Full statistics
    print_full_statistics(items)


if __name__ == "__main__":
    main()
