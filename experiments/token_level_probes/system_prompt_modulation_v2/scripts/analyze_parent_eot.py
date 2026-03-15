"""Analyze parent experiment EOT scores by prompt type (user vs assistant turn).

Usage:
    python experiments/token_level_probes/system_prompt_modulation_v2/scripts/analyze_parent_eot.py
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

RESULTS_PATH = Path("experiments/token_level_probes/system_prompt_modulation_v2/parent_eot_scores.json")
ASSETS_DIR = Path("experiments/token_level_probes/system_prompt_modulation_v2/assets")

BEST_PROBE = {
    "truth": "task_mean_L32",
    "harm": "task_mean_L39",
}


def cohen_d(a, b):
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na - 1) * np.std(a, ddof=1)**2 + (nb - 1) * np.std(b, ddof=1)**2) / (na + nb - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled_std


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    data = json.load(open(RESULTS_PATH))
    items = data["items"]

    for domain in ["truth", "harm"]:
        probe = BEST_PROBE[domain]
        domain_items = [i for i in items if i["domain"] == domain]

        if domain == "truth":
            conditions = ["true", "false"]
        else:
            conditions = ["benign", "harmful"]

        turns = ["user", "assistant"]

        print(f"\n{'='*80}")
        print(f"{domain.upper()} — EOT scores by prompt type ({probe})")
        print(f"{'='*80}")
        print(f"{'Turn':<15} {conditions[0]:>12} {conditions[1]:>12} {'d':>8} {'p':>12}")
        print("-" * 60)

        turn_data = {}
        for turn in turns:
            cond_vals = {}
            for condition in conditions:
                vals = [i["eot_scores"][probe] for i in domain_items
                        if i["turn"] == turn and i["condition"] == condition]
                cond_vals[condition] = vals

            d = cohen_d(cond_vals[conditions[0]], cond_vals[conditions[1]])
            _, p = stats.mannwhitneyu(cond_vals[conditions[0]], cond_vals[conditions[1]], alternative="two-sided")

            c0_mean = np.mean(cond_vals[conditions[0]])
            c0_std = np.std(cond_vals[conditions[0]], ddof=1)
            c1_mean = np.mean(cond_vals[conditions[1]])
            c1_std = np.std(cond_vals[conditions[1]], ddof=1)

            p_str = f"{p:.2e}" if p < 0.001 else f"{p:.4f}"
            print(f"{turn:<15} {c0_mean:>+6.2f}±{c0_std:.2f} {c1_mean:>+6.2f}±{c1_std:.2f} {d:>+8.2f} {p_str:>12}")

            turn_data[turn] = cond_vals

        # Also print critical span for comparison
        print(f"\n  (Critical span for comparison:)")
        print(f"  {'Turn':<15} {conditions[0]:>12} {conditions[1]:>12} {'d':>8} {'p':>12}")
        print(f"  {'-'*56}")
        for turn in turns:
            cond_vals = {}
            for condition in conditions:
                vals = [i["critical_span_mean_scores"][probe] for i in domain_items
                        if i["turn"] == turn and i["condition"] == condition]
                cond_vals[condition] = vals

            d = cohen_d(cond_vals[conditions[0]], cond_vals[conditions[1]])
            _, p = stats.mannwhitneyu(cond_vals[conditions[0]], cond_vals[conditions[1]], alternative="two-sided")

            c0_mean = np.mean(cond_vals[conditions[0]])
            c0_std = np.std(cond_vals[conditions[0]], ddof=1)
            c1_mean = np.mean(cond_vals[conditions[1]])
            c1_std = np.std(cond_vals[conditions[1]], ddof=1)

            p_str = f"{p:.2e}" if p < 0.001 else f"{p:.4f}"
            print(f"  {turn:<15} {c0_mean:>+6.2f}±{c0_std:.2f} {c1_mean:>+6.2f}±{c1_std:.2f} {d:>+8.2f} {p_str:>12}")

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        colors = {conditions[0]: "#2196F3" if domain == "truth" else "#4CAF50",
                  conditions[1]: "#F44336"}

        for ti, turn in enumerate(turns):
            ax = axes[ti]
            for ci, condition in enumerate(conditions):
                vals = turn_data[turn][condition]
                positions = [ci]
                parts = ax.violinplot([vals], positions=positions, showmeans=True, showextrema=False)
                for pc in parts["bodies"]:
                    pc.set_facecolor(colors[condition])
                    pc.set_alpha(0.6)
                parts["cmeans"].set_color("black")

                ax.scatter(np.random.normal(ci, 0.04, len(vals)), vals, alpha=0.3,
                          s=8, color=colors[condition])

            d_val = cohen_d(turn_data[turn][conditions[0]], turn_data[turn][conditions[1]])
            ax.set_title(f"{turn.capitalize()} turn (d={d_val:.2f})")
            ax.set_xticks(range(len(conditions)))
            ax.set_xticklabels(conditions)
            ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
            ax.grid(axis="y", alpha=0.3)

        axes[0].set_ylabel(f"EOT score ({probe})")
        fig.suptitle(f"{domain.capitalize()}: EOT scores by turn", fontsize=14)
        plt.tight_layout()
        plt.savefig(ASSETS_DIR / f"plot_031426_{domain}_eot_by_turn.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved plot_031426_{domain}_eot_by_turn.png")

    # Politics EOT by system prompt
    politics_items = [i for i in items if i["domain"] == "politics"]
    if politics_items:
        probe = "task_mean_L39"
        print(f"\n{'='*80}")
        print(f"POLITICS — EOT scores by system prompt ({probe})")
        print(f"{'='*80}")

        for sp in ["democrat", "republican", "neutral"]:
            sp_items = [i for i in politics_items if i.get("system_prompt") == sp]
            if not sp_items:
                continue
            left_vals = [i["eot_scores"][probe] for i in sp_items if i["condition"] == "left"]
            right_vals = [i["eot_scores"][probe] for i in sp_items if i["condition"] == "right"]
            if left_vals and right_vals:
                d = cohen_d(left_vals, right_vals)
                _, p = stats.mannwhitneyu(left_vals, right_vals, alternative="two-sided")
                p_str = f"{p:.2e}" if p < 0.001 else f"{p:.4f}"
                print(f"{sp:<15} left={np.mean(left_vals):>+6.2f}±{np.std(left_vals, ddof=1):.2f}  right={np.mean(right_vals):>+6.2f}±{np.std(right_vals, ddof=1):.2f}  d={d:>+.2f}  p={p_str}")


if __name__ == "__main__":
    main()
