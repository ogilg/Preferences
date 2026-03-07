"""Plot EOT vs prompt_last steering comparison.

Produces:
1. Per-ordering P(A) comparison (bar chart with CIs)
2. Aggregate steering effect comparison
3. Per-pair scatter: EOT vs prompt_last effects
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EOT_CHECKPOINT = Path("experiments/steering/eot_direction/checkpoint.jsonl")
V2_CHECKPOINT = Path("experiments/revealed_steering_v2/followup/checkpoint.jsonl")
ASSETS_DIR = Path("experiments/steering/eot_direction/assets")
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_pa_by_ordering(records: list[dict]) -> dict[str, dict[int, float]]:
    grouped: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r["choice_original"] in ("a", "b"):
            grouped[r["pair_id"]][r["ordering"]].append(r["choice_original"] == "a")
    result = {}
    for pair_id, orderings in grouped.items():
        result[pair_id] = {}
        for ordering, choices in orderings.items():
            result[pair_id][ordering] = np.mean(choices)
    return result


def compute_aggregate_effect(steered_pa, baseline_pa):
    effects = {}
    for pair_id in steered_pa:
        if pair_id not in baseline_pa:
            continue
        s = steered_pa[pair_id]
        b = baseline_pa[pair_id]
        if 0 in s and 1 in s and 0 in b and 1 in b:
            ab_shift = s[0] - b[0]
            ba_shift = b[1] - s[1]
            effects[pair_id] = (ab_shift + ba_shift) / 2
    return effects


def bootstrap_mean_ci(values, n_boot=10000, ci=0.95):
    arr = np.array(values)
    mean = np.mean(arr)
    boot_means = np.array([np.mean(np.random.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    return float(mean), float(np.percentile(boot_means, alpha * 100)), float(np.percentile(boot_means, (1 - alpha) * 100))


def plot_steering_comparison():
    np.random.seed(42)

    eot_records = load_jsonl(EOT_CHECKPOINT)
    v2_records = load_jsonl(V2_CHECKPOINT)

    baseline_records = [r for r in v2_records if r["condition"] == "baseline"]
    pl_pos = [r for r in v2_records if r["condition"] == "probe" and abs(r["multiplier"] - 0.03) < 1e-6]
    pl_neg = [r for r in v2_records if r["condition"] == "probe" and abs(r["multiplier"] - (-0.03)) < 1e-6]
    eot_pos = [r for r in eot_records if r["multiplier"] > 0]
    eot_neg = [r for r in eot_records if r["multiplier"] < 0]

    baseline_pa = compute_pa_by_ordering(baseline_records)
    pl_pos_pa = compute_pa_by_ordering(pl_pos)
    pl_neg_pa = compute_pa_by_ordering(pl_neg)
    eot_pos_pa = compute_pa_by_ordering(eot_pos)
    eot_neg_pa = compute_pa_by_ordering(eot_neg)

    # --- Plot 1: Aggregate steering effect comparison ---
    pl_pos_eff = compute_aggregate_effect(pl_pos_pa, baseline_pa)
    pl_neg_eff = compute_aggregate_effect(pl_neg_pa, baseline_pa)
    eot_pos_eff = compute_aggregate_effect(eot_pos_pa, baseline_pa)
    eot_neg_eff = compute_aggregate_effect(eot_neg_pa, baseline_pa)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    conditions = [
        ("Prompt-last\n-0.03", pl_neg_eff, "#4477AA"),
        ("Prompt-last\n+0.03", pl_pos_eff, "#4477AA"),
        ("EOT\n-0.03", eot_neg_eff, "#CC6677"),
        ("EOT\n+0.03", eot_pos_eff, "#CC6677"),
    ]

    x_pos = [0, 1, 2.5, 3.5]
    for i, (label, effects, color) in enumerate(conditions):
        vals = list(effects.values())
        mean, ci_lo, ci_hi = bootstrap_mean_ci(vals)
        ax.bar(x_pos[i], mean, width=0.7, color=color, alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.errorbar(x_pos[i], mean, yerr=[[mean - ci_lo], [ci_hi - mean]], fmt="none", color="black", capsize=4)
        ax.text(x_pos[i], mean + 0.008 * (1 if mean >= 0 else -1), f"{mean:+.3f}", ha="center", va="bottom" if mean >= 0 else "top", fontsize=9)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([c[0] for c in conditions])
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Aggregate steering effect")
    ax.set_title("EOT vs Prompt-last: Steering effect at |mult|=0.03")
    ax.set_ylim(-0.3, 0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "plot_030626_steering_effect_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {ASSETS_DIR / 'plot_030626_steering_effect_comparison.png'}")

    # --- Plot 2: Per-ordering P(A) comparison ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    cond_data = [
        ("Baseline", baseline_pa, "#999999"),
        ("PL -0.03", pl_neg_pa, "#88CCEE"),
        ("PL +0.03", pl_pos_pa, "#4477AA"),
        ("EOT -0.03", eot_neg_pa, "#DDCC77"),
        ("EOT +0.03", eot_pos_pa, "#CC6677"),
    ]

    x = np.arange(len(cond_data))
    width = 0.35

    for i, (label, pa_data, color) in enumerate(cond_data):
        ab_vals = [pa_data[p][0] for p in pa_data if 0 in pa_data[p]]
        ba_vals = [pa_data[p][1] for p in pa_data if 1 in pa_data[p]]
        ab_mean, ab_lo, ab_hi = bootstrap_mean_ci(ab_vals)
        ba_mean, ba_lo, ba_hi = bootstrap_mean_ci(ba_vals)

        ax.bar(i - width/2, ab_mean, width, color=color, alpha=0.7, edgecolor="black", linewidth=0.5, label="P(A|AB)" if i == 0 else "")
        ax.bar(i + width/2, ba_mean, width, color=color, alpha=0.4, edgecolor="black", linewidth=0.5, label="P(A|BA)" if i == 0 else "")
        ax.errorbar(i - width/2, ab_mean, yerr=[[ab_mean-ab_lo],[ab_hi-ab_mean]], fmt="none", color="black", capsize=3)
        ax.errorbar(i + width/2, ba_mean, yerr=[[ba_mean-ba_lo],[ba_hi-ba_mean]], fmt="none", color="black", capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in cond_data])
    ax.set_ylabel("P(A)")
    ax.set_title("P(A) by ordering and condition")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "plot_030626_per_ordering_pa.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {ASSETS_DIR / 'plot_030626_per_ordering_pa.png'}")

    # --- Plot 3: Per-pair scatter ---
    common_pos = sorted(set(eot_pos_eff.keys()) & set(pl_pos_eff.keys()))
    common_neg = sorted(set(eot_neg_eff.keys()) & set(pl_neg_eff.keys()))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if len(common_pos) > 10:
        eot_v = np.array([eot_pos_eff[p] for p in common_pos])
        pl_v = np.array([pl_pos_eff[p] for p in common_pos])
        r = np.corrcoef(eot_v, pl_v)[0, 1]
        axes[0].scatter(pl_v, eot_v, alpha=0.3, s=10, color="#4477AA")
        axes[0].set_xlabel("Prompt-last effect (+0.03)")
        axes[0].set_ylabel("EOT effect (+0.03)")
        axes[0].set_title(f"+0.03: Pearson r = {r:.3f} (n={len(common_pos)})")
        lim = max(abs(pl_v).max(), abs(eot_v).max()) * 1.1
        axes[0].set_xlim(-lim, lim)
        axes[0].set_ylim(-lim, lim)
        axes[0].axhline(0, color="gray", linewidth=0.5, linestyle="--")
        axes[0].axvline(0, color="gray", linewidth=0.5, linestyle="--")
        # Diagonal
        axes[0].plot([-lim, lim], [-lim, lim], "k--", alpha=0.3, linewidth=0.5)

    if len(common_neg) > 10:
        eot_v = np.array([eot_neg_eff[p] for p in common_neg])
        pl_v = np.array([pl_neg_eff[p] for p in common_neg])
        r = np.corrcoef(eot_v, pl_v)[0, 1]
        axes[1].scatter(pl_v, eot_v, alpha=0.3, s=10, color="#CC6677")
        axes[1].set_xlabel("Prompt-last effect (-0.03)")
        axes[1].set_ylabel("EOT effect (-0.03)")
        axes[1].set_title(f"-0.03: Pearson r = {r:.3f} (n={len(common_neg)})")
        lim = max(abs(pl_v).max(), abs(eot_v).max()) * 1.1
        axes[1].set_xlim(-lim, lim)
        axes[1].set_ylim(-lim, lim)
        axes[1].axhline(0, color="gray", linewidth=0.5, linestyle="--")
        axes[1].axvline(0, color="gray", linewidth=0.5, linestyle="--")
        axes[1].plot([-lim, lim], [-lim, lim], "k--", alpha=0.3, linewidth=0.5)

    fig.suptitle("Per-pair steering effect: EOT vs Prompt-last")
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "plot_030626_per_pair_scatter.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {ASSETS_DIR / 'plot_030626_per_pair_scatter.png'}")


if __name__ == "__main__":
    plot_steering_comparison()
