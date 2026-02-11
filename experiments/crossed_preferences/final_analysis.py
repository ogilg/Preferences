"""Final analysis for crossed preferences experiment.

Combines all behavioral and probe results, computes key metrics,
generates plots for the research log.
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

EXP_DIR = Path("experiments/crossed_preferences")
RESULTS_DIR = EXP_DIR / "results"
ASSETS_DIR = Path("docs/logs/assets/crossed_preferences")
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

from datetime import datetime
DATE_STR = datetime.now().strftime("%m%d%y")


def load_all_results():
    all_results = []
    for fname, source in [
        ("probe_behavioral_iteration.json", "iteration"),
        ("probe_behavioral_holdout.json", "holdout"),
        ("probe_behavioral_subtle.json", "subtle"),
    ]:
        path = RESULTS_DIR / fname
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            for r in data:
                r["prompt_source"] = source
            all_results.extend(data)
    return all_results


def compute_signed_delta(results: list[dict], key: str = "behavioral_delta") -> list[float]:
    """Return deltas with sign flipped for negative prompts so positive = correct direction."""
    signed = []
    for r in results:
        d = r[key]
        if r["direction"] == "negative":
            d = -d
        signed.append(d)
    return signed


def print_correlation_table(results: list[dict], label: str):
    beh = np.array([r["behavioral_delta"] for r in results])
    print(f"\n{'='*60}")
    print(f"{label} (n={len(results)})")
    print(f"{'='*60}")
    for layer in [31, 43, 55]:
        key = f"probe_delta_L{layer}"
        probe = np.array([r[key] for r in results])
        pr, pp = stats.pearsonr(beh, probe)
        sr, sp = stats.spearmanr(beh, probe)
        sign = np.mean(np.sign(beh) == np.sign(probe))
        print(f"  L{layer}: Pearson r={pr:.3f} (p={pp:.1e}), Spearman r={sr:.3f}, Sign={sign:.1%}")


def plot_correlation(results: list[dict], title: str, filename: str, color_by: str = "task_set"):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    beh = np.array([r["behavioral_delta"] for r in results])
    probe = np.array([r["probe_delta_L31"] for r in results])

    color_map = {
        "crossed": "#2196F3",
        "pure": "#FF9800",
        "subtle": "#4CAF50",
    }
    shell_map = {
        "math": "s",
        "coding": "^",
        "fiction": "D",
        "content_generation": "o",
        "harmful": "X",
        "pure": "o",
    }

    for r, b, p in zip(results, beh, probe):
        c = color_map.get(r["task_set"], "#999")
        m = shell_map.get(r["category_shell"], "o")
        ax.scatter(b, p, c=c, marker=m, alpha=0.4, s=30, edgecolors="none")

    pr, _ = stats.pearsonr(beh, probe)
    # Fit line
    slope, intercept = np.polyfit(beh, probe, 1)
    x_range = np.linspace(beh.min(), beh.max(), 100)
    ax.plot(x_range, slope * x_range + intercept, "k--", alpha=0.5, linewidth=1)

    ax.set_xlabel("Behavioral Delta", fontsize=12)
    ax.set_ylabel("Probe Delta (L31)", fontsize=12)
    ax.set_title(f"{title}\nr={pr:.3f}, n={len(results)}", fontsize=13)
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)

    # Legend for task sets
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2196F3", label="Crossed", markersize=8),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#FF9800", label="Pure", markersize=8),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4CAF50", label="Subtle", markersize=8),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    plt.tight_layout()
    path = ASSETS_DIR / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_category_shell_comparison(results: list[dict], filename: str):
    """Bar chart of probe-behavioral correlation by category shell."""
    crossed = [r for r in results if r["task_set"] == "crossed"]

    shells = ["math", "coding", "fiction", "content_generation", "harmful"]
    shell_labels = ["Math", "Coding", "Fiction", "Content Gen", "Harmful"]
    correlations = []
    ns = []

    for shell in shells:
        shell_data = [r for r in crossed if r["category_shell"] == shell]
        if len(shell_data) >= 3:
            beh = np.array([r["behavioral_delta"] for r in shell_data])
            probe = np.array([r["probe_delta_L31"] for r in shell_data])
            r_val, _ = stats.pearsonr(beh, probe)
            correlations.append(r_val)
            ns.append(len(shell_data))
        else:
            correlations.append(0)
            ns.append(0)

    # Add pure reference
    pure_data = [r for r in results if r["task_set"] == "pure"]
    beh = np.array([r["behavioral_delta"] for r in pure_data])
    probe = np.array([r["probe_delta_L31"] for r in pure_data])
    pure_r, _ = stats.pearsonr(beh, probe)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2196F3"] * len(shells)
    bars = ax.bar(range(len(shells)), correlations, color=colors, alpha=0.7, edgecolor="k", linewidth=0.5)
    ax.axhline(pure_r, color="#FF9800", linestyle="--", linewidth=2, label=f"Pure ref (r={pure_r:.3f})")

    ax.set_xticks(range(len(shells)))
    ax.set_xticklabels(shell_labels, fontsize=11)
    ax.set_ylabel("Pearson r (behavioral vs probe delta, L31)", fontsize=11)
    ax.set_title("Probe-Behavioral Correlation by Category Shell", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.0)

    # Add n labels
    for i, (r_val, n) in enumerate(zip(correlations, ns)):
        ax.text(i, r_val + 0.02, f"r={r_val:.2f}\nn={n}", ha="center", fontsize=9)

    plt.tight_layout()
    path = ASSETS_DIR / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_direction_agreement_by_shell(results: list[dict], filename: str):
    """Direction agreement by category shell (behavioral)."""
    crossed = [r for r in results if r["task_set"] == "crossed"]

    shells = ["math", "coding", "fiction", "content_generation", "harmful"]
    shell_labels = ["Math", "Coding", "Fiction", "Content Gen", "Harmful"]
    agreements = []

    for shell in shells:
        shell_data = [r for r in crossed if r["category_shell"] == shell]
        correct = sum(1 for r in shell_data
                     if (r["behavioral_delta"] > 0) == (r["direction"] == "positive")
                     and r["behavioral_delta"] != 0)
        agreements.append(correct / len(shell_data) if shell_data else 0)

    # Pure reference
    pure_data = [r for r in results if r["task_set"] == "pure"]
    pure_correct = sum(1 for r in pure_data
                      if (r["behavioral_delta"] > 0) == (r["direction"] == "positive")
                      and r["behavioral_delta"] != 0)
    pure_agreement = pure_correct / len(pure_data) if pure_data else 0

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2196F3"] * len(shells)
    ax.bar(range(len(shells)), agreements, color=colors, alpha=0.7, edgecolor="k", linewidth=0.5)
    ax.axhline(pure_agreement, color="#FF9800", linestyle="--", linewidth=2,
               label=f"Pure ref ({pure_agreement:.1%})")
    ax.axhline(0.7, color="red", linestyle=":", linewidth=1.5, label="Target (70%)")

    ax.set_xticks(range(len(shells)))
    ax.set_xticklabels(shell_labels, fontsize=11)
    ax.set_ylabel("Direction Agreement (%)", fontsize=11)
    ax.set_title("Behavioral Direction Agreement by Category Shell", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)

    for i, a in enumerate(agreements):
        ax.text(i, a + 0.02, f"{a:.0%}", ha="center", fontsize=10)

    plt.tight_layout()
    path = ASSETS_DIR / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_pure_vs_crossed_effect(results: list[dict], filename: str):
    """Compare mean |probe delta| for pure vs crossed tasks by topic."""
    topics = sorted(set(r["target_topic"] for r in results if r["task_set"] in ("crossed", "pure")))

    pure_means = []
    crossed_means = []

    for topic in topics:
        pure = [abs(r["probe_delta_L31"]) for r in results
                if r["task_set"] == "pure" and r["target_topic"] == topic]
        crossed = [abs(r["probe_delta_L31"]) for r in results
                  if r["task_set"] == "crossed" and r["target_topic"] == topic]
        if pure and crossed:
            pure_means.append(np.mean(pure))
            crossed_means.append(np.mean(crossed))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(pure_means, crossed_means, s=80, alpha=0.7, edgecolors="k", linewidth=0.5)

    # Identity line
    max_val = max(max(pure_means), max(crossed_means))
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.3)

    for i, topic in enumerate(topics[:len(pure_means)]):
        ax.annotate(topic, (pure_means[i], crossed_means[i]),
                   fontsize=8, alpha=0.7, xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("Mean |Probe Delta| — Pure Tasks", fontsize=11)
    ax.set_ylabel("Mean |Probe Delta| — Crossed Tasks", fontsize=11)
    ax.set_title("Pure vs Crossed Effect Size by Topic", fontsize=13)

    plt.tight_layout()
    path = ASSETS_DIR / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    all_results = load_all_results()
    print(f"Total results: {len(all_results)}")

    # --- Overall ---
    print_correlation_table(all_results, "All results combined")

    # --- By prompt source ---
    for source in ["iteration", "holdout", "subtle"]:
        subset = [r for r in all_results if r["prompt_source"] == source]
        print_correlation_table(subset, f"Source: {source}")

    # --- By task set ---
    for ts in ["crossed", "pure", "subtle"]:
        subset = [r for r in all_results if r["task_set"] == ts]
        if len(subset) >= 3:
            print_correlation_table(subset, f"Task set: {ts}")

    # --- Excluding harmful ---
    no_harmful = [r for r in all_results if r["category_shell"] != "harmful"]
    print_correlation_table(no_harmful, "All results (excluding harmful shell)")

    crossed_no_harmful = [r for r in no_harmful if r["task_set"] == "crossed"]
    print_correlation_table(crossed_no_harmful, "Crossed only (excluding harmful)")

    # --- Combined direction agreement ---
    print(f"\n{'='*60}")
    print("Direction Agreement Summary")
    print(f"{'='*60}")
    for label, subset in [
        ("All", all_results),
        ("Crossed (all shells)", [r for r in all_results if r["task_set"] == "crossed"]),
        ("Crossed (excl harmful)", crossed_no_harmful),
        ("Pure", [r for r in all_results if r["task_set"] == "pure"]),
        ("Subtle (new topics)", [r for r in all_results if r["task_set"] == "subtle"]),
    ]:
        if not subset:
            continue
        correct = sum(1 for r in subset
                     if (r["behavioral_delta"] > 0) == (r["direction"] == "positive")
                     and r["behavioral_delta"] != 0)
        total = len(subset)
        print(f"  {label:<30}: {correct}/{total} = {correct/total:.1%}")

    # --- Category shell effect size comparison ---
    print(f"\n{'='*60}")
    print("Category Shell: Mean |Probe Delta L31| (All Sources Combined)")
    print(f"{'='*60}")
    crossed_all = [r for r in all_results if r["task_set"] == "crossed"]
    shells = ["math", "coding", "fiction", "content_generation", "harmful"]
    for shell in shells:
        shell_data = [abs(r["probe_delta_L31"]) for r in crossed_all if r["category_shell"] == shell]
        print(f"  {shell:<20}: mean={np.mean(shell_data):.1f}, std={np.std(shell_data):.1f}, n={len(shell_data)}")

    pure_all = [abs(r["probe_delta_L31"]) for r in all_results if r["task_set"] == "pure"]
    print(f"  {'pure (reference)':<20}: mean={np.mean(pure_all):.1f}, std={np.std(pure_all):.1f}, n={len(pure_all)}")

    # --- Plots ---
    print("\n--- Generating plots ---")
    plot_correlation(all_results, "All Conditions Combined", f"plot_{DATE_STR}_correlation_all.png")
    plot_correlation(
        [r for r in all_results if r["prompt_source"] == "iteration"],
        "Iteration Prompts", f"plot_{DATE_STR}_correlation_iteration.png"
    )
    plot_category_shell_comparison(all_results, f"plot_{DATE_STR}_shell_correlation.png")
    plot_direction_agreement_by_shell(all_results, f"plot_{DATE_STR}_shell_agreement.png")
    plot_pure_vs_crossed_effect(all_results, f"plot_{DATE_STR}_pure_vs_crossed.png")

    # --- Attenuation test ---
    print(f"\n{'='*60}")
    print("Pure vs Crossed Attenuation (paired by topic × prompt)")
    print(f"{'='*60}")
    # For each prompt_id, compare mean |probe delta| on pure vs crossed tasks
    prompt_ids = sorted(set(r["prompt_id"] for r in all_results))
    pure_deltas = []
    crossed_deltas = []
    for pid in prompt_ids:
        pid_pure = [abs(r["probe_delta_L31"]) for r in all_results
                   if r["prompt_id"] == pid and r["task_set"] == "pure"]
        pid_crossed = [abs(r["probe_delta_L31"]) for r in all_results
                      if r["prompt_id"] == pid and r["task_set"] == "crossed"]
        if pid_pure and pid_crossed:
            pure_deltas.append(np.mean(pid_pure))
            crossed_deltas.append(np.mean(pid_crossed))

    pure_arr = np.array(pure_deltas)
    crossed_arr = np.array(crossed_deltas)
    t_stat, t_p = stats.ttest_rel(pure_arr, crossed_arr)
    print(f"  N prompt pairs: {len(pure_arr)}")
    print(f"  Mean |probe delta| pure: {np.mean(pure_arr):.1f}")
    print(f"  Mean |probe delta| crossed: {np.mean(crossed_arr):.1f}")
    print(f"  Ratio (crossed/pure): {np.mean(crossed_arr)/np.mean(pure_arr):.2f}")
    print(f"  Paired t-test: t={t_stat:.2f}, p={t_p:.1e}")
    print(f"  Direction: {'pure > crossed (attenuated)' if np.mean(pure_arr) > np.mean(crossed_arr) else 'crossed >= pure (not attenuated)'}")


if __name__ == "__main__":
    main()
