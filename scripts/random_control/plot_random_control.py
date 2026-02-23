"""
Plotting for random direction control experiment.

Creates:
1. Per-pair effect distributions: probe vs random directions (violin/box plots)
2. Mean effects comparison bar chart
3. CDF of per-pair effects
4. Sign check: -2641 vs +2641 comparison

Usage:
  python scripts/random_control/plot_random_control.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXP_DIR = REPO_ROOT / "experiments" / "steering" / "replication" / "random_control"
RESULTS_DIR = EXP_DIR / "results"
ASSETS_DIR = EXP_DIR / "assets"


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def compute_per_pair_effect_from_results(
    results: list[dict],
    condition: str,
    coef: float,
    control_results: list[dict],
) -> list[float]:
    """Return per-pair effects (pp) at given condition+coef vs control."""
    steered_choice = {"boost_a": "a", "diff_ab": "a"}[condition]

    # P(steered) per pair×ordering at condition+coef
    pair_ord_s: dict = {}
    for trial in results:
        if trial["condition"] == condition and abs(trial["coefficient"] - coef) < 1e-3:
            key = (trial["pair_id"], trial["ordering"])
            valid = [r for r in trial["responses"] if r != "parse_fail"]
            if valid:
                p = sum(1 for r in valid if r == steered_choice) / len(valid)
                pair_ord_s.setdefault(key, []).append(p)

    # P(a) per pair×ordering for control
    pair_ord_c: dict = {}
    for trial in control_results:
        if trial["condition"] == "control" and abs(trial["coefficient"] - 0.0) < 1e-3:
            key = (trial["pair_id"], trial["ordering"])
            valid = [r for r in trial["responses"] if r != "parse_fail"]
            if valid:
                p = sum(1 for r in valid if r == "a") / len(valid)
                pair_ord_c.setdefault(key, []).append(p)

    # Aggregate over orderings per pair
    pair_ids = sorted(set(k[0] for k in pair_ord_s) | set(k[0] for k in pair_ord_c))
    effects = []
    for pid in pair_ids:
        s_vals, c_vals = [], []
        for ordering in ["original", "swapped"]:
            k = (pid, ordering)
            if k in pair_ord_s:
                s_vals.extend(pair_ord_s[k])
            if k in pair_ord_c:
                c_vals.extend(pair_ord_c[k])
        if s_vals and c_vals:
            effects.append((np.mean(s_vals) - np.mean(c_vals)) * 100)
    return effects


def main():
    probe_path = RESULTS_DIR / "probe_rerun.json"
    probe_data = load_json(probe_path)
    probe_results = probe_data["results"]
    control_results = probe_results  # control embedded in probe re-run

    random_data = {}
    for seed in [100, 101, 102, 103, 104]:
        path = RESULTS_DIR / f"random_seed{seed}.json"
        if path.exists():
            random_data[seed] = load_json(path)

    seeds = sorted(random_data)
    print(f"Plotting: probe + {len(seeds)} random directions ({seeds})")

    # ────────────────────────────────────────────────────────────────────────────
    # Figure 1: Per-pair effect distributions (violin plots) for diff_ab at +2641
    # ────────────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax_idx, condition in enumerate(["boost_a", "diff_ab"]):
        ax = axes[ax_idx]

        probe_effects = compute_per_pair_effect_from_results(
            probe_results, condition, 2641.0, control_results
        )
        random_effects_list = [
            compute_per_pair_effect_from_results(
                random_data[seed]["results"], condition, 2641.0, control_results
            )
            for seed in seeds
        ]

        all_data = [probe_effects] + random_effects_list
        labels = ["probe\n(L31)"] + [f"rand\n{s}" for s in seeds]
        colors = ["#2196F3"] + ["#FF7043"] * len(seeds)

        parts = ax.violinplot(all_data, positions=range(len(all_data)), showmedians=True)

        for i, (body, color) in enumerate(zip(parts["bodies"], colors)):
            body.set_facecolor(color)
            body.set_alpha(0.6)
        parts["cmedians"].set_color("black")
        parts["cmaxes"].set_color("gray")
        parts["cmins"].set_color("gray")
        parts["cbars"].set_color("gray")

        # Overlay mean as dot
        for i, effects in enumerate(all_data):
            if effects:
                ax.scatter([i], [np.mean(effects)], color=colors[i], s=60, zorder=5,
                           edgecolors="black", linewidths=0.8)

        ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Per-pair effect (pp)")
        ax.set_title(f"{condition} @ coef=+2641")
        ax.set_ylim(-60, 60)
        ax.grid(axis="y", alpha=0.3)

        # Annotate means
        for i, effects in enumerate(all_data):
            if effects:
                mean_val = np.mean(effects)
                ax.text(i, mean_val + 3, f"{mean_val:+.1f}", ha="center", va="bottom",
                        fontsize=8, color=colors[i])

    probe_patch = mpatches.Patch(color="#2196F3", alpha=0.6, label="Probe direction")
    rand_patch = mpatches.Patch(color="#FF7043", alpha=0.6, label="Random directions")
    fig.legend(handles=[probe_patch, rand_patch], loc="upper right", fontsize=9)
    fig.suptitle("Probe vs Random Direction: Per-pair effects at coef=+2641\n"
                 "(vs within-experiment control; error bars = violin IQR)", fontsize=11)
    plt.tight_layout()
    out = ASSETS_DIR / "plot_022226_probe_vs_random_violin.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out.name}")

    # ────────────────────────────────────────────────────────────────────────────
    # Figure 2: Mean effects bar chart with SEM (both conditions, +2641 and -2641)
    # ────────────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax_idx, condition in enumerate(["boost_a", "diff_ab"]):
        ax = axes[ax_idx]

        all_dirs = ["probe"] + [f"rand_{s}" for s in seeds]
        all_results = [probe_results] + [random_data[s]["results"] for s in seeds]
        all_colors = ["#2196F3"] + ["#FF7043"] * len(seeds)

        x = np.arange(len(all_dirs))
        width = 0.35

        means_pos, sems_pos = [], []
        means_neg, sems_neg = [], []

        for results in all_results:
            eff_pos = compute_per_pair_effect_from_results(results, condition, 2641.0, control_results)
            eff_neg = compute_per_pair_effect_from_results(results, condition, -2641.0, control_results)

            if eff_pos:
                means_pos.append(np.mean(eff_pos))
                sems_pos.append(np.std(eff_pos) / np.sqrt(len(eff_pos)))
            else:
                means_pos.append(float("nan"))
                sems_pos.append(float("nan"))

            if eff_neg:
                means_neg.append(np.mean(eff_neg))
                sems_neg.append(np.std(eff_neg) / np.sqrt(len(eff_neg)))
            else:
                means_neg.append(float("nan"))
                sems_neg.append(float("nan"))

        bars_pos = ax.bar(x - width/2, [m*100 for m in means_pos],
                          width, yerr=[s*100 for s in sems_pos],
                          label="coef=+2641", color=all_colors, alpha=0.8, capsize=4)
        bars_neg = ax.bar(x + width/2, [m*100 for m in means_neg],
                          width, yerr=[s*100 for s in sems_neg],
                          label="coef=-2641", color=all_colors, alpha=0.4, capsize=4,
                          hatch="///")

        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(all_dirs, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Mean per-pair effect (pp)")
        ax.set_title(f"{condition}")
        ax.set_ylim(-30, 30)
        ax.grid(axis="y", alpha=0.3)

        from matplotlib.lines import Line2D
        pos_patch = mpatches.Patch(color="gray", alpha=0.8, label="coef=+2641")
        neg_patch = mpatches.Patch(color="gray", alpha=0.4, hatch="///", label="coef=-2641")
        ax.legend(handles=[pos_patch, neg_patch], fontsize=8)

    fig.suptitle("Mean per-pair effects: probe vs random directions\n"
                 "(error bars = SEM; positive=steering increases P(a))", fontsize=11)
    plt.tight_layout()
    out = ASSETS_DIR / "plot_022226_probe_vs_random_means.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out.name}")

    # ────────────────────────────────────────────────────────────────────────────
    # Figure 3: CDF of per-pair effects (diff_ab @ +2641)
    # ────────────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))

    condition = "diff_ab"
    probe_effects = compute_per_pair_effect_from_results(
        probe_results, condition, 2641.0, control_results
    )
    probe_sorted = np.sort(probe_effects)
    ax.plot(probe_sorted, np.arange(1, len(probe_sorted)+1) / len(probe_sorted),
            color="#2196F3", lw=2.5, label=f"probe_L31 (mean={np.mean(probe_effects):+.1f}pp)")

    for seed in seeds:
        r_eff = compute_per_pair_effect_from_results(
            random_data[seed]["results"], condition, 2641.0, control_results
        )
        r_sorted = np.sort(r_eff)
        ax.plot(r_sorted, np.arange(1, len(r_sorted)+1) / len(r_sorted),
                color="#FF7043", lw=1.2, alpha=0.6,
                label=f"rand_{seed} (mean={np.mean(r_eff):+.1f}pp)" if seed == seeds[0] else f"rand_{seed} ({np.mean(r_eff):+.1f}pp)")

    ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("Per-pair effect (pp)")
    ax.set_ylabel("CDF")
    ax.set_title(f"{condition} @ coef=+2641: CDF of per-pair effects\n(probe vs random directions)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(-60, 60)
    ax.set_ylim(0, 1)

    out = ASSETS_DIR / "plot_022226_probe_vs_random_cdf.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out.name}")

    # ────────────────────────────────────────────────────────────────────────────
    # Figure 4: Sign symmetry check (probe: +2641 vs -2641 should flip sign; random: both ~0)
    # ────────────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax_idx, condition in enumerate(["boost_a", "diff_ab"]):
        ax = axes[ax_idx]

        all_dirs = ["probe"] + [f"rand_{s}" for s in seeds]
        all_results = [probe_results] + [random_data[s]["results"] for s in seeds]
        all_colors = ["#2196F3"] + ["#FF7043"] * len(seeds)

        for i, (label, results) in enumerate(zip(all_dirs, all_results)):
            eff_pos = compute_per_pair_effect_from_results(results, condition, 2641.0, control_results)
            eff_neg = compute_per_pair_effect_from_results(results, condition, -2641.0, control_results)

            # Scatter: +coef effect vs -coef effect per pair
            # To compare: pair each pair's positive and negative effect
            pair_ids_pos = {}
            for trial in results:
                if trial["condition"] == condition and abs(trial["coefficient"] - 2641.0) < 1e-3:
                    for r in trial["responses"]:
                        pass  # already done above

            mean_pos = np.mean(eff_pos) * 100 if eff_pos else float("nan")
            mean_neg = np.mean(eff_neg) * 100 if eff_neg else float("nan")
            ax.scatter([mean_neg], [mean_pos], color=all_colors[i], s=100,
                       zorder=5, edgecolors="black", linewidths=0.8)
            ax.annotate(label, (mean_neg, mean_pos), textcoords="offset points",
                        xytext=(5, 5), fontsize=7)

        ax.axhline(0, color="black", lw=0.8, ls="-", alpha=0.4)
        ax.axvline(0, color="black", lw=0.8, ls="-", alpha=0.4)

        # Diagonal: if perfectly antisymmetric, pos = -neg
        lim = 20
        ax.plot([-lim, lim], [lim, -lim], "k--", alpha=0.3, lw=1, label="perfect antisymmetry")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("Mean effect at coef=-2641 (pp)")
        ax.set_ylabel("Mean effect at coef=+2641 (pp)")
        ax.set_title(f"{condition}: sign symmetry check")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle("Sign symmetry: +coef vs -coef effects\n"
                 "(probe should be roughly antisymmetric; random should cluster at origin)",
                 fontsize=11)
    plt.tight_layout()
    out = ASSETS_DIR / "plot_022226_sign_symmetry.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out.name}")

    print("\nAll plots saved to:", ASSETS_DIR)


if __name__ == "__main__":
    main()
