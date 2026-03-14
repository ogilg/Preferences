"""EOT truncation analysis: does excluding end-of-sequence tokens
reveal the critical span as the dominant predictor?

The all_token_scores.npz (gitignored, lives on RunPod) was split from
scoring_results.json and is not available locally. This script works with
the data we DO have: critical_span_mean_scores and fullstop_scores, plus
recomputing EOT-position statistics from the token lists.

Available comparison points:
- Full sequence (EOT + fullstop + critical span all included)
- Fullstop score (1 token before EOT)
- Critical span score (content tokens only, unaffected by end truncation)

For each domain, we compute Cohen's d between conditions at these three
feature levels, showing how signal changes as we move away from EOT.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = Path("experiments/token_level_probes/scoring_results.json")
ASSETS_DIR = Path("experiments/token_level_probes/assets")

DOMAIN_CONFIG = {
    "truth": {
        "probe": "task_mean_L32",
        "cond_a": "true",
        "cond_b": "false",
        "label": "True vs False",
    },
    "harm": {
        "probe": "task_mean_L39",
        "cond_a": "harmful",
        "cond_b": "benign",
        "label": "Harmful vs Benign",
    },
}


def cohens_d(group_a: list[float], group_b: list[float]) -> float:
    a = np.array(group_a)
    b = np.array(group_b)
    n_a, n_b = len(a), len(b)
    pooled_std = np.sqrt(
        ((n_a - 1) * a.std(ddof=1) ** 2 + (n_b - 1) * b.std(ddof=1) ** 2)
        / (n_a + n_b - 2)
    )
    return float((a.mean() - b.mean()) / pooled_std)


def get_eot_position(tokens: list[str]) -> int:
    """Return the position of the last <end_of_turn> token from the end.

    E.g. if EOT is at index 30 and sequence length is 34, returns 3
    (meaning it's the 4th-from-last token, 0-indexed from end).
    """
    eot_indices = [i for i, t in enumerate(tokens) if t == "<end_of_turn>"]
    if not eot_indices:
        raise ValueError("No <end_of_turn> token found")
    last_eot_idx = eot_indices[-1]
    return len(tokens) - 1 - last_eot_idx


def get_critical_span_distance_from_end(
    tokens: list[str], critical_indices: list[int]
) -> int:
    """How far the last critical span token is from the end of the sequence."""
    last_critical = max(critical_indices)
    return len(tokens) - 1 - last_critical


def main():
    with open(DATA_PATH) as f:
        results = json.load(f)
    items = results["items"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, (domain, cfg) in zip(axes, DOMAIN_CONFIG.items()):
        probe = cfg["probe"]
        cond_a = cfg["cond_a"]
        cond_b = cfg["cond_b"]

        # Collect scores per condition at different feature levels
        critical_a, critical_b = [], []
        fullstop_a, fullstop_b = [], []
        # Also track distances for context
        eot_distances = []
        critical_distances = []

        for item in items:
            if item["domain"] != domain:
                continue
            if item["condition"] not in (cond_a, cond_b):
                continue

            crit_score = item["critical_span_mean_scores"][probe]

            # Fullstop: take the last fullstop score (closest to EOT)
            fs_scores = item["fullstop_scores"][probe]
            if not fs_scores:
                continue
            fullstop_score = fs_scores[-1]

            eot_dist = get_eot_position(item["tokens"])
            crit_dist = get_critical_span_distance_from_end(
                item["tokens"], item["critical_token_indices"]
            )
            eot_distances.append(eot_dist)
            critical_distances.append(crit_dist)

            if item["condition"] == cond_a:
                critical_a.append(crit_score)
                fullstop_a.append(fullstop_score)
            else:
                critical_b.append(crit_score)
                fullstop_b.append(fullstop_score)

        d_critical = cohens_d(critical_a, critical_b)
        d_fullstop = cohens_d(fullstop_a, fullstop_b)

        print(f"\n{domain.upper()} ({cond_a} vs {cond_b}) — probe: {probe}")
        print(f"  Critical span Cohen's d: {d_critical:+.3f} "
              f"(n={len(critical_a)}+{len(critical_b)})")
        print(f"  Fullstop Cohen's d:      {d_fullstop:+.3f} "
              f"(n={len(fullstop_a)}+{len(fullstop_b)})")
        print(f"  Mean EOT distance from end: {np.mean(eot_distances):.1f} tokens")
        print(f"  Mean critical span distance from end: "
              f"{np.mean(critical_distances):.1f} tokens")
        print(f"  |d| ratio (fullstop/critical): {abs(d_fullstop)/abs(d_critical):.2f}x")

        # Phase 3 EOT values from the report (computed with all_token_scores.npz)
        eot_d_from_report = {"truth": 3.14, "harm": -2.27}
        d_eot = eot_d_from_report[domain]

        # Positions: distance from end (0 = last token, higher = further from end)
        # EOT is at position ~3 from end, fullstop is 1 before EOT, critical span ~5-10
        mean_eot_dist = np.mean(eot_distances)
        mean_crit_dist = np.mean(critical_distances)
        # Fullstop is typically 1 position before EOT
        mean_fullstop_dist = mean_eot_dist + 1

        positions = [mean_eot_dist, mean_fullstop_dist, mean_crit_dist]
        d_values = [abs(d_eot), abs(d_fullstop), abs(d_critical)]
        labels_pts = ["EOT\n(from report)", "Fullstop", "Critical\nspan"]

        # Bar chart showing |d| at each position
        colors = ["#e74c3c", "#f39c12", "#2ecc71"]
        bars = ax.bar(range(3), d_values, color=colors, width=0.6, edgecolor="black",
                       linewidth=0.5)

        # Add value labels on bars
        for bar, d_val in zip(bars, d_values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"|d|={d_val:.2f}", ha="center", va="bottom", fontsize=10,
                    fontweight="bold")

        ax.set_xticks(range(3))
        ax.set_xticklabels(labels_pts, fontsize=10)
        ax.set_ylabel("|Cohen's d|", fontsize=11)
        ax.set_title(f"{domain.capitalize()}: {cfg['label']}\nprobe: {probe}",
                     fontsize=12)
        ax.set_ylim(0, max(d_values) * 1.25)

        # Add distance annotation
        ax.text(0.02, 0.97,
                f"Avg distances from end:\n"
                f"  EOT: {mean_eot_dist:.0f} tokens\n"
                f"  Fullstop: {mean_fullstop_dist:.0f} tokens\n"
                f"  Critical span: {mean_crit_dist:.0f} tokens",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="gray", alpha=0.8))

    fig.suptitle(
        "Signal by token region: EOT dominates, critical span is weakest\n"
        "(EOT values from Phase 3 report; fullstop & critical span computed here)",
        fontsize=12,
    )
    plt.tight_layout()
    out_path = ASSETS_DIR / "plot_031426_eot_truncation_analysis.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")

    print("\n" + "=" * 70)
    print("NOTE: Full truncation sweep (exclude last N tokens, recompute mean)")
    print("requires all_token_scores.npz which is gitignored and lives on RunPod.")
    print("This analysis uses the available per-region scores instead.")
    print("=" * 70)


if __name__ == "__main__":
    main()
