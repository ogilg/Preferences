"""
Fine-Grained Steering: Post-experiment report finalization.

Reads analysis.json, generates Phase 2-4 plots, and outputs summary tables.
Run after all phases complete.

Usage:
    python scripts/fine_grained/finalize_report.py
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXP_DIR = REPO_ROOT / "experiments" / "steering" / "replication" / "fine_grained"
RESULTS_DIR = EXP_DIR / "results"
ASSETS_DIR = EXP_DIR / "assets"
ASSETS_DIR.mkdir(exist_ok=True)


def load_analysis() -> dict:
    path = RESULTS_DIR / "analysis.json"
    with open(path) as f:
        return json.load(f)


def print_phase_table(analysis: dict, phase_key: str, condition: str, norm_scale: float,
                      label: str = "") -> None:
    """Print a formatted dose-response table for a given phase and condition."""
    if phase_key not in analysis:
        print(f"  {phase_key}: NO DATA")
        return

    phase_data = analysis[phase_key]
    ctrl = phase_data.get("_control_summary", {})
    if ctrl:
        print(f"  Control P(a): {ctrl['mean_p_a']:.3f} ± {ctrl['std_p_a']:.3f} (n={ctrl['n_pairs']})")

    if condition not in phase_data:
        print(f"  {condition}: NO DATA")
        return

    by_coef = {float(k): v for k, v in phase_data[condition]["by_coef"].items()}
    print(f"  {condition} ({label}):")
    print(f"  {'Coef':>8} {'%norm':>7} {'N':>5} {'P(a)':>6} {'Effect':>8} {'SE':>6} {'t':>6} {'p':>8} {'%pos':>5}")
    for coef in sorted(by_coef.keys()):
        d = by_coef[coef]
        pct = coef / norm_scale * 100
        print(f"  {coef:>8.0f} {pct:>6.1f}% {d['n_pairs']:>5} {d['mean_p_a']:>6.3f} "
              f"{d['mean_effect_pp']:>7.1f}pp {d['se_pp']:>5.1f} "
              f"{d['t_stat']:>6.2f} {d['p_value']:>8.4f} {d['pct_positive']:>5.1%}")


def print_comparison_table(analysis: dict, phases: list[tuple[str, str, float, str]]) -> None:
    """Compare peak effects across phases/layers at equivalent % norm."""
    # phases: list of (phase_key, condition, norm_scale, label)
    print("\n=== Peak effect comparison at each % norm ===")
    pct_points = [-10, -7.5, -5, -3, 3, 5, 7.5, 10]
    print(f"{'%norm':>7} | " + " | ".join(f"{label:>15}" for _, _, _, label in phases))
    print("-" * (9 + 19 * len(phases)))
    for pct in pct_points:
        row = [f"{pct:>6.1f}%"]
        for phase_key, condition, norm_scale, label in phases:
            if phase_key not in analysis or condition not in analysis[phase_key]:
                row.append(f"{'N/A':>15}")
                continue
            by_coef = {float(k): v for k, v in analysis[phase_key][condition]["by_coef"].items()}
            target_coef = norm_scale * pct / 100
            # Find closest coef
            closest = min(by_coef.keys(), key=lambda c: abs(c - target_coef))
            if abs(closest - target_coef) / (abs(target_coef) + 1) < 0.1:
                d = by_coef[closest]
                sig = "*" if d["p_value"] < 0.05 else " "
                row.append(f"{d['mean_effect_pp']:>+6.1f}pp{sig} (n={d['n_pairs']:3d})")
            else:
                row.append(f"{'N/A':>15}")
        print(" | ".join(row))


def generate_layer_comparison_plot(analysis: dict) -> None:
    """Generate plot comparing diff_ab across L31, L49, L55."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # L31 norm scale
    layer_configs = [
        ("phase1_L31", "diff_ab", 52823, "L31", "C0"),
        ("phase2_L49", "diff_ab", 80067, "L49", "C1"),
        ("phase2_L55", "diff_ab", 93579, "L55", "C2"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle("Single-layer diff_ab dose-response: L31 vs L49 vs L55", fontsize=13)

    for ax, (phase_key, condition, norm_scale, label, color) in zip(axes, layer_configs):
        if phase_key not in analysis or condition not in analysis[phase_key]:
            ax.text(0.5, 0.5, f"{phase_key}\nNo data", transform=ax.transAxes, ha="center")
            ax.set_title(f"{label}")
            continue

        by_coef = {float(k): v for k, v in analysis[phase_key][condition]["by_coef"].items()}
        coefs = sorted(by_coef.keys())
        pcts = [c / norm_scale * 100 for c in coefs]
        effects = [by_coef[c]["mean_effect_pp"] for c in coefs]
        ses = [by_coef[c]["se_pp"] for c in coefs]
        pvals = [by_coef[c]["p_value"] for c in coefs]

        for i, (pct, eff, se, pval) in enumerate(zip(pcts, effects, ses, pvals)):
            marker = "o" if pval < 0.05 else "o"
            mfc = color if pval < 0.05 else "white"
            ax.errorbar(pct, eff, yerr=se, fmt=marker, color=color, mfc=mfc,
                        mec=color, capsize=3, alpha=0.8)

        ax.plot(pcts, effects, "-", color=color, alpha=0.5, linewidth=1)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Coefficient (% of mean norm)")
        ax.set_title(f"{label}")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Effect (pp vs control)")
    plt.tight_layout()
    out = ASSETS_DIR / "plot_022426_layer_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


def generate_multilayer_plot(analysis: dict) -> None:
    """Generate plot comparing multi-layer configs vs single-layer L31 and L49."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "phase3_multilayer" not in analysis:
        print("phase3_multilayer: no data")
        return

    ml_data = analysis["phase3_multilayer"]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title("Multi-layer diff_ab vs single-layer (diff_ab, 300 pairs)", fontsize=12)

    # Single-layer L31 reference
    configs = [
        ("phase1_L31", "diff_ab", 52823, "L31 (single)", "C0", "-"),
        ("phase2_L49", "diff_ab", 80067, "L49 (single)", "C1", "-"),
    ]
    ml_configs = [
        ("L31_L37_layer_split", 52823, "L31+L37 (split)", "C3", "--"),
        ("L31_L49_layer_split", 52823, "L31+L49 (split)", "C4", "--"),
        ("L49_L55_layer_split", 80067, "L49+L55 (split)", "C5", "--"),
    ]

    for phase_key, condition, norm_scale, label, color, ls in configs:
        if phase_key not in analysis or condition not in analysis[phase_key]:
            continue
        by_coef = {float(k): v for k, v in analysis[phase_key][condition]["by_coef"].items()}
        coefs = sorted(by_coef.keys())
        pcts = [c / norm_scale * 100 for c in coefs]
        effects = [by_coef[c]["mean_effect_pp"] for c in coefs]
        ses = [by_coef[c]["se_pp"] for c in coefs]
        ax.errorbar(pcts, effects, yerr=ses, fmt="o", color=color, label=label,
                    capsize=3, alpha=0.9, linestyle=ls)

    for cond, norm_scale, label, color, ls in ml_configs:
        if cond not in ml_data:
            continue
        by_coef = {float(k): v for k, v in ml_data[cond]["by_coef"].items()}
        coefs = sorted(by_coef.keys())
        pcts = [c / norm_scale * 100 for c in coefs]
        effects = [by_coef[c]["mean_effect_pp"] for c in coefs]
        ses = [by_coef[c]["se_pp"] for c in coefs]
        ax.errorbar(pcts, effects, yerr=ses, fmt="s", color=color, label=label,
                    capsize=3, alpha=0.9, linestyle=ls)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Coefficient (% of primary-probe mean norm)")
    ax.set_ylabel("Effect (pp vs control)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = ASSETS_DIR / "plot_022426_multilayer_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


def generate_random_control_plot(analysis: dict) -> None:
    """Compare random direction vs probe direction at L49 and L55."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle("Probe direction vs random direction: diff_ab at L49, L55", fontsize=12)

    layer_configs = [
        (49, "phase2_L49", "diff_ab", 80067, "phase4_random_L49", "random_diff_ab_ridge_L49"),
        (55, "phase2_L55", "diff_ab", 93579, "phase4_random_L55", "random_diff_ab_ridge_L55"),
    ]

    for ax, (layer, probe_phase, probe_cond, norm_scale, rand_phase, rand_cond) in zip(axes, layer_configs):
        ax.set_title(f"L{layer}")

        # Probe direction
        if probe_phase in analysis and probe_cond in analysis[probe_phase]:
            by_coef = {float(k): v for k, v in analysis[probe_phase][probe_cond]["by_coef"].items()}
            coefs = sorted(by_coef.keys())
            pcts = [c / norm_scale * 100 for c in coefs]
            effects = [by_coef[c]["mean_effect_pp"] for c in coefs]
            ses = [by_coef[c]["se_pp"] for c in coefs]
            ax.errorbar(pcts, effects, yerr=ses, fmt="o-", color="C0",
                        label="Probe direction", capsize=3, alpha=0.9)

        # Random direction
        if rand_phase in analysis and rand_cond in analysis[rand_phase]:
            by_coef = {float(k): v for k, v in analysis[rand_phase][rand_cond]["by_coef"].items()}
            coefs = sorted(by_coef.keys())
            pcts = [c / norm_scale * 100 for c in coefs]
            effects = [by_coef[c]["mean_effect_pp"] for c in coefs]
            ses = [by_coef[c]["se_pp"] for c in coefs]
            ax.errorbar(pcts, effects, yerr=ses, fmt="s--", color="C1",
                        label="Random direction", capsize=3, alpha=0.9)

        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Coefficient (% of mean norm)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Effect (pp vs control)")
    plt.tight_layout()
    out = ASSETS_DIR / "plot_022426_random_control.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


def main():
    analysis = load_analysis()

    print("=== Available phases ===")
    for key in analysis:
        if not key.startswith("_"):
            print(f"  {key}")
    print()

    # Phase 2: L49
    print("\n=== Phase 2: L49 single-layer ===")
    for cond in ["diff_ab", "boost_a", "boost_b"]:
        print_phase_table(analysis, "phase2_L49", cond, 80067, "L49")
        print()

    # Phase 2: L55
    print("\n=== Phase 2: L55 single-layer ===")
    for cond in ["diff_ab", "boost_a", "boost_b"]:
        print_phase_table(analysis, "phase2_L55", cond, 93579, "L55")
        print()

    # Phase 3: Multi-layer
    print("\n=== Phase 3: Multi-layer ===")
    for cond in ["L31_L37_layer_split", "L31_L49_layer_split", "L49_L55_layer_split"]:
        print_phase_table(analysis, "phase3_multilayer", cond, 52823, "multi")
        print()

    # Phase 4: Random controls
    print("\n=== Phase 4: Random controls ===")
    print_phase_table(analysis, "phase4_random_L49", "random_diff_ab_ridge_L49", 80067, "random@L49")
    print()
    print_phase_table(analysis, "phase4_random_L55", "random_diff_ab_ridge_L55", 93579, "random@L55")
    print()

    # Cross-layer comparison
    print_comparison_table(analysis, [
        ("phase1_L31", "diff_ab", 52823, "L31"),
        ("phase2_L49", "diff_ab", 80067, "L49"),
        ("phase2_L55", "diff_ab", 93579, "L55"),
    ])

    # Generate plots
    print("\nGenerating plots...")
    try:
        generate_layer_comparison_plot(analysis)
    except Exception as e:
        print(f"  layer_comparison plot failed: {e}")

    try:
        generate_multilayer_plot(analysis)
    except Exception as e:
        print(f"  multilayer plot failed: {e}")

    try:
        generate_random_control_plot(analysis)
    except Exception as e:
        print(f"  random_control plot failed: {e}")

    print("\nDone. Run 'python scripts/fine_grained/analyze.py' first to update analysis.json.")


if __name__ == "__main__":
    main()
