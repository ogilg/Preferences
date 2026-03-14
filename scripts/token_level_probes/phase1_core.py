"""Phase 1 core analysis for token-level probes experiment.

Generates distribution plots, turn comparisons, and probe comparison charts
from scoring_results.json.
"""

import json
from pathlib import Path
from itertools import product

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon


DATA_PATH = Path("experiments/token_level_probes/scoring_results.json")
ASSETS_DIR = Path("experiments/token_level_probes/assets")

PROBE_SETS = {
    "tb-2": ["tb-2_L32", "tb-2_L39", "tb-2_L53"],
    "tb-5": ["tb-5_L32", "tb-5_L39", "tb-5_L53"],
    "task_mean": ["task_mean_L32", "task_mean_L39", "task_mean_L53"],
}
PROBE_SET_ORDER = ["tb-2", "tb-5", "task_mean"]
LAYERS = [32, 39, 53]
ALL_PROBES = [f"{ps}_L{l}" for ps in PROBE_SET_ORDER for l in LAYERS]

DOMAIN_CONTRASTS = {
    "truth": ("true", "false"),
    "harm": ("harmful", "benign"),
    "politics": ("left", "right"),
}
DOMAIN_CONDITIONS = {
    "truth": ["true", "false", "nonsense"],
    "harm": ["harmful", "benign", "nonsense"],
    "politics": ["left", "right", "nonsense"],
}
CONDITION_COLORS = {
    "true": "#2ca02c",
    "false": "#d62728",
    "nonsense": "#999999",
    "harmful": "#d62728",
    "benign": "#2ca02c",
    "left": "#1f77b4",
    "right": "#d62728",
}


def load_data() -> list[dict]:
    with open(DATA_PATH) as f:
        return json.load(f)["items"]


def extract_base_id_truth_harm(item_id: str) -> tuple[str, str]:
    """Extract (base_id, turn) from truth/harm item IDs.

    E.g. 'truth_0_true_user' -> ('truth_0', 'user')
    """
    parts = item_id.split("_")
    # Format: domain_num_condition_turn
    return f"{parts[0]}_{parts[1]}", parts[-1]


def cohens_d_paired(x: np.ndarray, y: np.ndarray) -> float:
    diff = x - y
    return float(np.mean(diff) / np.std(diff, ddof=1))


def compute_paired_stats(
    items: list[dict],
    domain: str,
    probe: str,
) -> dict:
    """Compute paired Cohen's d and Wilcoxon p for the main contrast."""
    cond_a, cond_b = DOMAIN_CONTRASTS[domain]

    if domain in ("truth", "harm"):
        # Group by (base_id, turn)
        grouped: dict[tuple[str, str], dict[str, float]] = {}
        for item in items:
            if item["domain"] != domain or item["condition"] not in (cond_a, cond_b):
                continue
            base, turn = extract_base_id_truth_harm(item["id"])
            key = (base, turn)
            if key not in grouped:
                grouped[key] = {}
            grouped[key][item["condition"]] = item["critical_span_mean_scores"][probe]

        pairs = [(v[cond_a], v[cond_b]) for v in grouped.values() if cond_a in v and cond_b in v]
    else:
        # Politics: group by (issue, system_prompt)
        grouped_pol: dict[tuple[str, str], dict[str, float]] = {}
        for item in items:
            if item["domain"] != domain or item["condition"] not in (cond_a, cond_b):
                continue
            key = (item["issue"], item["system_prompt"])
            if key not in grouped_pol:
                grouped_pol[key] = {}
            grouped_pol[key][item["condition"]] = item["critical_span_mean_scores"][probe]

        pairs = [(v[cond_a], v[cond_b]) for v in grouped_pol.values() if cond_a in v and cond_b in v]

    if len(pairs) < 3:
        return {"d": float("nan"), "p": float("nan"), "n": len(pairs)}

    arr_a = np.array([p[0] for p in pairs])
    arr_b = np.array([p[1] for p in pairs])
    d = cohens_d_paired(arr_a, arr_b)
    _, p = wilcoxon(arr_a, arr_b)
    return {"d": d, "p": p, "n": len(pairs)}


def get_scores_by_condition(
    items: list[dict],
    domain: str,
    probe: str,
    condition: str,
) -> list[float]:
    return [
        item["critical_span_mean_scores"][probe]
        for item in items
        if item["domain"] == domain and item["condition"] == condition
    ]


def get_scores_by_condition_turn(
    items: list[dict],
    domain: str,
    probe: str,
    condition: str,
    turn: str,
) -> list[float]:
    return [
        item["critical_span_mean_scores"][probe]
        for item in items
        if item["domain"] == domain and item["condition"] == condition and item["turn"] == turn
    ]


# ── Analysis 1: Critical span score distributions by condition ──


def plot_domain_distributions(items: list[dict], domain: str) -> None:
    conditions = DOMAIN_CONDITIONS[domain]
    cond_a, cond_b = DOMAIN_CONTRASTS[domain]
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(
        f"{domain.title()} domain: Critical span scores by condition",
        fontsize=16,
        fontweight="bold",
    )

    for row_idx, ps_name in enumerate(PROBE_SET_ORDER):
        for col_idx, layer in enumerate(LAYERS):
            ax = axes[row_idx, col_idx]
            probe = f"{ps_name}_L{layer}"

            data_by_cond = []
            for cond in conditions:
                scores = get_scores_by_condition(items, domain, probe, cond)
                data_by_cond.append(scores)

            # Violin plot
            parts = ax.violinplot(
                data_by_cond,
                positions=range(len(conditions)),
                showmeans=True,
                showmedians=False,
            )
            for pc, cond in zip(parts["bodies"], conditions):
                pc.set_facecolor(CONDITION_COLORS[cond])
                pc.set_alpha(0.6)

            # Overlay jittered points
            rng = np.random.default_rng(42)
            for i, (cond, scores) in enumerate(zip(conditions, data_by_cond)):
                jitter = rng.uniform(-0.12, 0.12, size=len(scores))
                ax.scatter(
                    i + jitter,
                    scores,
                    alpha=0.3,
                    s=8,
                    color=CONDITION_COLORS[cond],
                    zorder=3,
                )

            ax.set_xticks(range(len(conditions)))
            ax.set_xticklabels(conditions, fontsize=9)

            # y-axis anchored at 0
            all_vals = [v for s in data_by_cond for v in s]
            ymin = min(min(all_vals), 0)
            ymax = max(max(all_vals), 0)
            margin = (ymax - ymin) * 0.1
            ax.set_ylim(ymin - margin, ymax + margin)
            ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)

            stats = compute_paired_stats(items, domain, probe)
            ax.set_title(
                f"{probe}\nd={stats['d']:.3f}, p={stats['p']:.1e}",
                fontsize=9,
            )
            if col_idx == 0:
                ax.set_ylabel("Score")

    plt.tight_layout()
    fname = f"plot_031426_{domain}_critical_span_by_condition.png"
    fig.savefig(ASSETS_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")


# ── Analysis 2: User vs assistant turn comparison ──


def plot_turn_comparison(items: list[dict], domain: str) -> None:
    conditions = DOMAIN_CONDITIONS[domain]
    cond_a, cond_b = DOMAIN_CONTRASTS[domain]
    turns = ["user", "assistant"]

    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    fig.suptitle(
        f"{domain.title()} domain: User vs assistant turn comparison (Layer 39)",
        fontsize=14,
        fontweight="bold",
    )

    for row_idx, ps_name in enumerate(PROBE_SET_ORDER):
        probe = f"{ps_name}_L39"
        for col_idx, turn in enumerate(turns):
            ax = axes[row_idx, col_idx]

            data_by_cond = []
            for cond in conditions:
                scores = get_scores_by_condition_turn(items, domain, probe, cond, turn)
                data_by_cond.append(scores)

            parts = ax.violinplot(
                data_by_cond,
                positions=range(len(conditions)),
                showmeans=True,
                showmedians=False,
            )
            for pc, cond in zip(parts["bodies"], conditions):
                pc.set_facecolor(CONDITION_COLORS[cond])
                pc.set_alpha(0.6)

            rng = np.random.default_rng(42)
            for i, (cond, scores) in enumerate(zip(conditions, data_by_cond)):
                jitter = rng.uniform(-0.12, 0.12, size=len(scores))
                ax.scatter(
                    i + jitter,
                    scores,
                    alpha=0.3,
                    s=8,
                    color=CONDITION_COLORS[cond],
                    zorder=3,
                )

            ax.set_xticks(range(len(conditions)))
            ax.set_xticklabels(conditions, fontsize=9)

            all_vals = [v for s in data_by_cond for v in s]
            if all_vals:
                ymin = min(min(all_vals), 0)
                ymax = max(max(all_vals), 0)
                margin = (ymax - ymin) * 0.1
                ax.set_ylim(ymin - margin, ymax + margin)
            ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)

            # Compute turn-specific paired stats
            # Filter items to this turn for pairing
            turn_items = [it for it in items if it.get("turn") == turn]
            stats = compute_paired_stats(turn_items, domain, probe)
            ax.set_title(f"{probe} | {turn}\nd={stats['d']:.3f}", fontsize=9)
            if col_idx == 0:
                ax.set_ylabel("Score")

    plt.tight_layout()
    fname = f"plot_031426_{domain}_turn_comparison.png"
    fig.savefig(ASSETS_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")


# ── Analysis 3: Probe variant comparison (summary bar chart) ──


def plot_probe_comparison(items: list[dict]) -> None:
    domains = ["truth", "harm", "politics"]
    n_probes = len(ALL_PROBES)
    n_domains = len(domains)

    fig, ax = plt.subplots(figsize=(16, 6))
    fig.suptitle(
        "|Cohen's d| by probe and domain (main contrast)",
        fontsize=14,
        fontweight="bold",
    )

    bar_width = 0.25
    domain_colors = {"truth": "#2ca02c", "harm": "#d62728", "politics": "#1f77b4"}
    x = np.arange(n_probes)

    for di, domain in enumerate(domains):
        d_values = []
        for probe in ALL_PROBES:
            stats = compute_paired_stats(items, domain, probe)
            d_values.append(abs(stats["d"]))
        ax.bar(
            x + di * bar_width,
            d_values,
            bar_width,
            label=domain.title(),
            color=domain_colors[domain],
            alpha=0.8,
        )

    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(ALL_PROBES, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("|Cohen's d|")
    ax.set_ylim(0, None)
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    fname = "plot_031426_probe_comparison_cohens_d.png"
    fig.savefig(ASSETS_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")


# ── Print statistics table ──


def print_statistics(items: list[dict]) -> None:
    domains = ["truth", "harm", "politics"]
    print("\n" + "=" * 100)
    print("STATISTICS TABLE: Cohen's d, Wilcoxon p, mean scores per condition")
    print("=" * 100)

    for domain in domains:
        cond_a, cond_b = DOMAIN_CONTRASTS[domain]
        conditions = DOMAIN_CONDITIONS[domain]
        print(f"\n{'─' * 100}")
        print(f"  {domain.upper()} (contrast: {cond_a} vs {cond_b})")
        print(f"{'─' * 100}")
        header = f"{'Probe':<16} {'d':>8} {'p':>12} {'n':>5}"
        for cond in conditions:
            header += f" {'mean(' + cond + ')':>16}"
        print(header)
        print("-" * len(header))

        for probe in ALL_PROBES:
            stats = compute_paired_stats(items, domain, probe)
            row = f"{probe:<16} {stats['d']:>8.3f} {stats['p']:>12.2e} {stats['n']:>5}"
            for cond in conditions:
                scores = get_scores_by_condition(items, domain, probe, cond)
                row += f" {np.mean(scores):>16.3f}"
            print(row)

    print("\n" + "=" * 100)


# ── Main ──


def main() -> None:
    items = load_data()
    print(f"Loaded {len(items)} items")

    # Analysis 1: Distribution plots per domain
    for domain in ["truth", "harm", "politics"]:
        plot_domain_distributions(items, domain)

    # Analysis 2: Turn comparisons (truth and harm only)
    for domain in ["truth", "harm"]:
        plot_turn_comparison(items, domain)

    # Analysis 3: Probe comparison bar chart
    plot_probe_comparison(items)

    # Print statistics
    print_statistics(items)


if __name__ == "__main__":
    main()
