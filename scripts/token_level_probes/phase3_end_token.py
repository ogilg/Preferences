"""Phase 3: End-of-sequence sentinel hypothesis.

Tests whether probe scores at the <end_of_turn> token are the strongest
single predictor of condition, comparing to critical span mean scores.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, LeaveOneOut, StratifiedKFold

DATA_PATH = Path("experiments/token_level_probes/scoring_results.json")
NPZ_PATH = Path("experiments/token_level_probes/all_token_scores.npz")
ASSETS_DIR = Path("experiments/token_level_probes/assets")

PROBE = "task_mean_L39"

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


def load_data():
    with open(DATA_PATH) as f:
        results = json.load(f)
    items = results["items"]
    scores_npz = np.load(NPZ_PATH)
    return items, scores_npz


def find_end_of_turn_score(item: dict, scores_npz) -> float:
    """Find the score at the last <end_of_turn> token position."""
    tokens = item["tokens"]
    eot_indices = [i for i, t in enumerate(tokens) if t == "<end_of_turn>"]
    if not eot_indices:
        raise ValueError(f"No <end_of_turn> token in {item['id']}")
    last_eot_idx = eot_indices[-1]

    key = f"{item['id']}__{PROBE}"
    all_scores = scores_npz[key]
    return float(all_scores[last_eot_idx])


def cohens_d(group_a: list[float], group_b: list[float]) -> float:
    a = np.array(group_a)
    b = np.array(group_b)
    n_a, n_b = len(a), len(b)
    pooled_std = np.sqrt(((n_a - 1) * a.std(ddof=1)**2 + (n_b - 1) * b.std(ddof=1)**2) / (n_a + n_b - 2))
    return float((a.mean() - b.mean()) / pooled_std)


# ── Analysis 1: End-token scores by condition (violin plots + Cohen's d) ──

def analysis1_end_token_scores(items: list[dict], scores_npz):
    print("=" * 70)
    print("ANALYSIS 1: End-token scores by condition")
    print("=" * 70)

    domains = ["truth", "harm", "politics"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, domain in zip(axes, domains):
        conditions = DOMAIN_CONDITIONS[domain]
        cond_a, cond_b = DOMAIN_CONTRASTS[domain]

        scores_by_cond = {c: [] for c in conditions}
        crit_by_cond = {c: [] for c in conditions}

        for item in items:
            if item["domain"] != domain:
                continue
            cond = item["condition"]
            if cond not in conditions:
                continue

            eot_score = find_end_of_turn_score(item, scores_npz)
            scores_by_cond[cond].append(eot_score)
            crit_by_cond[cond].append(item["critical_span_mean_scores"][PROBE])

        # Violin plots
        data_list = [scores_by_cond[c] for c in conditions]
        positions = list(range(len(conditions)))
        colors = [CONDITION_COLORS[c] for c in conditions]

        parts = ax.violinplot(data_list, positions=positions, showmeans=True, showmedians=True)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.6)
        for key in ["cmeans", "cmedians", "cmins", "cmaxes", "cbars"]:
            parts[key].set_color("black")

        ax.set_xticks(positions)
        ax.set_xticklabels(conditions)
        ax.set_title(f"{domain}")
        ax.set_ylabel("End-of-turn token score")

        # Cohen's d
        d_eot = cohens_d(scores_by_cond[cond_a], scores_by_cond[cond_b])
        d_crit = cohens_d(crit_by_cond[cond_a], crit_by_cond[cond_b])

        print(f"\n{domain} ({cond_a} vs {cond_b}):")
        print(f"  End-of-turn Cohen's d: {d_eot:.3f}")
        print(f"  Critical span Cohen's d: {d_crit:.3f}")
        print(f"  Ratio (EOT/crit): {abs(d_eot) / abs(d_crit):.2f}x" if abs(d_crit) > 0.001 else "  Critical span d ~ 0")

        for c in conditions:
            vals = scores_by_cond[c]
            print(f"  {c}: mean={np.mean(vals):.3f}, std={np.std(vals, ddof=1):.3f}, n={len(vals)}")

    fig.suptitle(f"End-of-turn token scores by condition ({PROBE})", fontsize=14)
    plt.tight_layout()
    out_path = ASSETS_DIR / "plot_031426_end_token_scores.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out_path}")


# ── Analysis 2: Predictive comparison ──

def analysis2_predictive_comparison(items: list[dict], scores_npz):
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Predictive comparison (logistic regression)")
    print("=" * 70)

    domains = ["truth", "harm", "politics"]

    for domain in domains:
        cond_a, cond_b = DOMAIN_CONTRASTS[domain]

        eot_scores = []
        crit_scores = []
        labels = []

        for item in items:
            if item["domain"] != domain:
                continue
            cond = item["condition"]
            if cond not in (cond_a, cond_b):
                continue

            eot_score = find_end_of_turn_score(item, scores_npz)
            crit_score = item["critical_span_mean_scores"][PROBE]

            eot_scores.append(eot_score)
            crit_scores.append(crit_score)
            labels.append(1 if cond == cond_a else 0)

        X_crit = np.array(crit_scores).reshape(-1, 1)
        X_eot = np.array(eot_scores).reshape(-1, 1)
        X_both = np.column_stack([crit_scores, eot_scores])
        y = np.array(labels)

        n = len(y)
        # Use 5-fold stratified CV (LOO is expensive and noisy)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        acc_crit = cross_val_score(
            LogisticRegression(max_iter=1000), X_crit, y, cv=cv, scoring="accuracy"
        ).mean()
        acc_eot = cross_val_score(
            LogisticRegression(max_iter=1000), X_eot, y, cv=cv, scoring="accuracy"
        ).mean()
        acc_both = cross_val_score(
            LogisticRegression(max_iter=1000), X_both, y, cv=cv, scoring="accuracy"
        ).mean()

        print(f"\n{domain} ({cond_a} vs {cond_b}, n={n}):")
        print(f"  Critical span only:  {acc_crit:.3f}")
        print(f"  End-of-turn only:    {acc_eot:.3f}")
        print(f"  Both combined:       {acc_both:.3f}")


# ── Analysis 3: Score accumulation curve ──

def analysis3_accumulation_curve(items: list[dict], scores_npz):
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Score accumulation curves (last 20 positions)")
    print("=" * 70)

    domains = ["truth", "harm", "politics"]
    max_lookback = 20

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, domain in zip(axes, domains):
        conditions = DOMAIN_CONDITIONS[domain]

        # Collect scores at each position from end
        # Position -1 = last token, -2 = second to last, etc.
        # We align from the <end_of_turn> token backwards
        scores_by_cond_pos = {c: {pos: [] for pos in range(-max_lookback, 0)} for c in conditions}

        for item in items:
            if item["domain"] != domain:
                continue
            cond = item["condition"]
            if cond not in conditions:
                continue

            key = f"{item['id']}__{PROBE}"
            all_scores = scores_npz[key]
            tokens = item["tokens"]

            # Find last <end_of_turn>
            eot_indices = [i for i, t in enumerate(tokens) if t == "<end_of_turn>"]
            if not eot_indices:
                continue
            last_eot_idx = eot_indices[-1]

            # Collect scores from end_of_turn backwards
            for offset in range(1, max_lookback + 1):
                pos = -offset
                token_idx = last_eot_idx - (offset - 1)
                if token_idx < 0:
                    continue
                scores_by_cond_pos[cond][pos].append(float(all_scores[token_idx]))

        # Plot mean +/- SEM for each condition
        for cond in conditions:
            positions = []
            means = []
            sems = []
            for pos in range(-max_lookback, 0):
                vals = scores_by_cond_pos[cond][pos]
                if len(vals) < 5:
                    continue
                positions.append(pos)
                means.append(np.mean(vals))
                sems.append(sem(vals))

            positions = np.array(positions)
            means = np.array(means)
            sems = np.array(sems)

            ax.plot(positions, means, color=CONDITION_COLORS[cond], label=cond, linewidth=2)
            ax.fill_between(positions, means - sems, means + sems,
                           color=CONDITION_COLORS[cond], alpha=0.15)

        ax.set_xlabel("Position relative to <end_of_turn>")
        ax.set_ylabel("Mean probe score")
        ax.set_title(domain)
        ax.legend()
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)

        # Print summary at key positions
        cond_a, cond_b = DOMAIN_CONTRASTS[domain]
        print(f"\n{domain} ({cond_a} vs {cond_b}):")
        for pos in [-1, -5, -10, -20]:
            vals_a = scores_by_cond_pos[cond_a].get(pos, [])
            vals_b = scores_by_cond_pos[cond_b].get(pos, [])
            if vals_a and vals_b:
                diff = np.mean(vals_a) - np.mean(vals_b)
                print(f"  Position {pos:3d}: {cond_a}={np.mean(vals_a):.3f}, "
                      f"{cond_b}={np.mean(vals_b):.3f}, diff={diff:.3f}")

    fig.suptitle(f"Score accumulation toward <end_of_turn> ({PROBE})", fontsize=14)
    plt.tight_layout()
    out_path = ASSETS_DIR / "plot_031426_score_accumulation.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out_path}")


def main():
    items, scores_npz = load_data()
    analysis1_end_token_scores(items, scores_npz)
    analysis2_predictive_comparison(items, scores_npz)
    analysis3_accumulation_curve(items, scores_npz)


if __name__ == "__main__":
    main()
