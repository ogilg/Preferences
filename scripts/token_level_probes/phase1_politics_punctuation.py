"""Phase 1 analyses 4 & 5: Politics system prompt modulation and fullstop scoring."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

DATA_PATH = Path("experiments/token_level_probes/scoring_results.json")
ASSETS_DIR = Path("experiments/token_level_probes/assets")

PROBE_SETS = ["tb-2", "tb-5", "task_mean"]
LAYERS = ["L32", "L39", "L53"]
ALL_PROBES = [f"{ps}_{l}" for ps in PROBE_SETS for l in LAYERS]

POLITICS_CONDITION_COLORS = {"left": "#2166AC", "right": "#B2182B", "nonsense": "#888888"}
SYSTEM_PROMPTS = ["democrat", "neutral", "republican"]
DOMAINS = ["truth", "harm", "politics"]
DOMAIN_CONDITIONS = {
    "truth": ["true", "false", "nonsense"],
    "harm": ["harmful", "benign", "nonsense"],
    "politics": ["left", "right", "nonsense"],
}
DOMAIN_COLORS = {
    "truth": {"true": "#2166AC", "false": "#B2182B", "nonsense": "#888888"},
    "harm": {"harmful": "#B2182B", "benign": "#2166AC", "nonsense": "#888888"},
    "politics": {"left": "#2166AC", "right": "#B2182B", "nonsense": "#888888"},
}


def load_data():
    with open(DATA_PATH) as f:
        return json.load(f)["items"]


# ---------------------------------------------------------------------------
# Analysis 4: Politics system prompt modulation
# ---------------------------------------------------------------------------

def analysis4_politics_modulation(items):
    politics_items = [it for it in items if it["domain"] == "politics"]
    print(f"\n{'='*60}")
    print("Analysis 4: Politics system prompt modulation")
    print(f"{'='*60}")
    print(f"Total politics items: {len(politics_items)}")

    conditions = sorted(set(it["condition"] for it in politics_items))
    sys_prompts = sorted(set(it["system_prompt"] for it in politics_items))
    print(f"Conditions: {conditions}")
    print(f"System prompts: {sys_prompts}")
    print(f"Issues: {sorted(set(it['issue'] for it in politics_items))}")

    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    fig.suptitle("Politics: Probe scores by system prompt and condition", fontsize=14, y=0.98)

    for row_idx, probe_set in enumerate(PROBE_SETS):
        for col_idx, layer in enumerate(LAYERS):
            probe_name = f"{probe_set}_{layer}"
            ax = axes[row_idx, col_idx]

            # Collect data grouped by (system_prompt, condition)
            violin_data = []
            violin_positions = []
            violin_colors = []
            tick_positions = []
            tick_labels = []

            for sp_idx, sp in enumerate(SYSTEM_PROMPTS):
                sp_items = [it for it in politics_items if it["system_prompt"] == sp]
                conds_present = sorted(set(it["condition"] for it in sp_items))

                for c_idx, cond in enumerate(conds_present):
                    scores = [
                        it["critical_span_mean_scores"][probe_name]
                        for it in sp_items
                        if it["condition"] == cond
                    ]
                    if len(scores) < 2:
                        continue
                    pos = sp_idx * 4 + c_idx
                    violin_data.append(scores)
                    violin_positions.append(pos)
                    violin_colors.append(POLITICS_CONDITION_COLORS[cond])

                tick_positions.append(sp_idx * 4 + 1)
                tick_labels.append(sp)

            if violin_data:
                parts = ax.violinplot(
                    violin_data,
                    positions=violin_positions,
                    showmeans=True,
                    showmedians=False,
                    widths=0.8,
                )
                for pc_idx, pc in enumerate(parts["bodies"]):
                    pc.set_facecolor(violin_colors[pc_idx])
                    pc.set_alpha(0.7)
                for key in ["cmeans", "cmins", "cmaxes", "cbars"]:
                    if key in parts:
                        parts[key].set_color("black")
                        parts[key].set_linewidth(0.8)

            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels)
            ax.set_title(probe_name, fontsize=11)
            ax.set_ylabel("Probe score")
            ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)

            # Set y-axis to include 0
            if violin_data:
                all_scores = [s for vd in violin_data for s in vd]
                ymin, ymax = min(all_scores), max(all_scores)
                margin = (ymax - ymin) * 0.1
                ax.set_ylim(min(0, ymin - margin), max(0, ymax + margin))

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=POLITICS_CONDITION_COLORS[c], alpha=0.7, label=c)
        for c in ["left", "right", "nonsense"]
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=11)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    out_path = ASSETS_DIR / "plot_031426_politics_system_prompt_modulation.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)

    # Summary statistics
    print("\nMean critical-span score by (system_prompt, condition) for tb-2_L39:")
    probe = "tb-2_L39"
    for sp in SYSTEM_PROMPTS:
        sp_items = [it for it in politics_items if it["system_prompt"] == sp]
        conds = sorted(set(it["condition"] for it in sp_items))
        for cond in conds:
            scores = [
                it["critical_span_mean_scores"][probe]
                for it in sp_items
                if it["condition"] == cond
            ]
            print(f"  {sp:12s} | {cond:10s} | n={len(scores):3d} | "
                  f"mean={np.mean(scores):+7.3f} | std={np.std(scores):.3f}")

    # Paired test: for each issue, compare left score under democrat vs republican
    print(f"\nPaired test (left condition): democrat vs republican system prompt")
    issues = sorted(set(it["issue"] for it in politics_items))
    for probe in ["tb-2_L39", "tb-5_L39", "task_mean_L39"]:
        dem_scores = []
        rep_scores = []
        for issue in issues:
            dem = [
                it["critical_span_mean_scores"][probe]
                for it in politics_items
                if it["issue"] == issue and it["condition"] == "left" and it["system_prompt"] == "democrat"
            ]
            rep = [
                it["critical_span_mean_scores"][probe]
                for it in politics_items
                if it["issue"] == issue and it["condition"] == "left" and it["system_prompt"] == "republican"
            ]
            if dem and rep:
                dem_scores.append(np.mean(dem))
                rep_scores.append(np.mean(rep))
        if len(dem_scores) >= 2:
            t_stat, p_val = stats.ttest_rel(dem_scores, rep_scores)
            diff = np.mean(dem_scores) - np.mean(rep_scores)
            print(f"  {probe}: mean diff (dem-rep) = {diff:+.3f}, "
                  f"t({len(dem_scores)-1}) = {t_stat:.3f}, p = {p_val:.4f}")


# ---------------------------------------------------------------------------
# Analysis 5: Fullstop (punctuation) analysis
# ---------------------------------------------------------------------------

def analysis5_fullstop(items):
    print(f"\n{'='*60}")
    print("Analysis 5: Fullstop (punctuation) scores")
    print(f"{'='*60}")

    representative_probe = "tb-2_L39"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Mean fullstop score by condition (probe: {representative_probe})", fontsize=13, y=1.02)

    for col_idx, domain in enumerate(DOMAINS):
        ax = axes[col_idx]
        domain_items = [it for it in items if it["domain"] == domain]
        conditions = DOMAIN_CONDITIONS[domain]
        colors = DOMAIN_COLORS[domain]

        # Compute mean fullstop score per item
        violin_data = []
        violin_colors = []
        for cond in conditions:
            cond_items = [it for it in domain_items if it["condition"] == cond]
            mean_scores = []
            for it in cond_items:
                fs_scores = it["fullstop_scores"][representative_probe]
                if fs_scores:
                    mean_scores.append(np.mean(fs_scores))
            violin_data.append(mean_scores)
            violin_colors.append(colors[cond])

        positions = list(range(len(conditions)))
        parts = ax.violinplot(
            violin_data,
            positions=positions,
            showmeans=True,
            showmedians=False,
            widths=0.7,
        )
        for pc_idx, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(violin_colors[pc_idx])
            pc.set_alpha(0.7)
        for key in ["cmeans", "cmins", "cmaxes", "cbars"]:
            if key in parts:
                parts[key].set_color("black")
                parts[key].set_linewidth(0.8)

        ax.set_xticks(positions)
        ax.set_xticklabels(conditions)
        ax.set_title(domain.capitalize(), fontsize=12)
        ax.set_ylabel("Mean fullstop score")
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)

        # Anchor y-axis at 0
        all_scores = [s for vd in violin_data for s in vd]
        if all_scores:
            ymin, ymax = min(all_scores), max(all_scores)
            margin = (ymax - ymin) * 0.1
            ax.set_ylim(min(0, ymin - margin), max(0, ymax + margin))

    plt.tight_layout()
    out_path = ASSETS_DIR / "plot_031426_fullstop_scores_by_condition.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)

    # Summary stats and paired tests
    print(f"\nSummary statistics for fullstop scores ({representative_probe}):")
    for domain in DOMAINS:
        domain_items = [it for it in items if it["domain"] == domain]
        conditions = DOMAIN_CONDITIONS[domain]
        print(f"\n  {domain.upper()}:")
        cond_scores = {}
        for cond in conditions:
            cond_items = [it for it in domain_items if it["condition"] == cond]
            mean_scores = []
            for it in cond_items:
                fs_scores = it["fullstop_scores"][representative_probe]
                if fs_scores:
                    mean_scores.append(np.mean(fs_scores))
            cond_scores[cond] = mean_scores
            print(f"    {cond:12s} | n={len(mean_scores):3d} | "
                  f"mean={np.mean(mean_scores):+7.3f} | std={np.std(mean_scores):.3f}")

        # Paired test for main contrast
        main_contrasts = {
            "truth": ("true", "false"),
            "harm": ("benign", "harmful"),
            "politics": ("left", "right"),
        }
        a_cond, b_cond = main_contrasts[domain]
        a_scores = cond_scores[a_cond]
        b_scores = cond_scores[b_cond]
        # Unpaired since items are different across conditions
        t_stat, p_val = stats.mannwhitneyu(a_scores, b_scores, alternative="two-sided")
        print(f"    Mann-Whitney U ({a_cond} vs {b_cond}): U={t_stat:.1f}, p={p_val:.4f}")
        # Also Welch's t-test
        t_stat2, p_val2 = stats.ttest_ind(a_scores, b_scores, equal_var=False)
        print(f"    Welch's t ({a_cond} vs {b_cond}): t={t_stat2:.3f}, p={p_val2:.4f}")


if __name__ == "__main__":
    items = load_data()
    analysis4_politics_modulation(items)
    analysis5_fullstop(items)
    print("\nDone.")
