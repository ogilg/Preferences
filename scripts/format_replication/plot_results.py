"""
Plots for Format Replication experiment.

Creates:
1. plot_022426_dose_response_by_format.png — dose-response curves for each format × position
2. plot_022426_slope_comparison.png — per-task slope distributions, format × position comparison
3. plot_022426_slope_heatmap.png — heatmap of mean slopes (format × position)
4. plot_022426_parse_rates.png — parse rates across conditions
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "experiments" / "steering" / "stated_steering" / "format_replication" / "results"
ASSETS_DIR = REPO_ROOT / "experiments" / "steering" / "stated_steering" / "format_replication" / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

FORMATS = ["qualitative_ternary", "adjective_pick", "anchored_simple_1_5"]
POSITIONS = ["task_tokens", "generation", "last_token"]
FORMAT_LABELS = {
    "qualitative_ternary": "Qualitative ternary\n(1-3)",
    "adjective_pick": "Adjective pick\n(1-10)",
    "anchored_simple_1_5": "Anchored 1-5",
}
POSITION_LABELS = {
    "task_tokens": "Task tokens",
    "generation": "Generation",
    "last_token": "Last token",
}
POSITION_COLORS = {
    "task_tokens": "#1f77b4",
    "generation": "#ff7f0e",
    "last_token": "#2ca02c",
}

# Scale ranges for normalization to [0,1]
SCALE_RANGES = {
    "qualitative_ternary": (1.0, 3.0),
    "adjective_pick": (1.0, 10.0),
    "anchored_simple_1_5": (1.0, 5.0),
}

# Mean L31 activation norm (used for x-axis as % of norm)
MEAN_L31_NORM = 52820.0


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_statistics() -> list[dict]:
    with open(RESULTS_DIR / "statistics.json") as f:
        return json.load(f)


def load_per_task_slopes() -> list[dict]:
    with open(RESULTS_DIR / "per_task_slopes.json") as f:
        return json.load(f)


def get_dose_response(fmt: str, position: str) -> tuple[list[float], list[float], list[float]]:
    """Return (coef_pct, mean_scores_normalized, sem_scores)."""
    path = RESULTS_DIR / f"results_{fmt}_{position}.jsonl"
    records = load_jsonl(path)
    if not records:
        return [], [], []

    lo, hi = SCALE_RANGES[fmt]
    by_coef: dict[float, list[float]] = defaultdict(list)
    for r in records:
        coef = r["coefficient"]
        scores = [s for s in r["scores"] if s is not None]
        # Normalize to [0,1]
        normalized = [(s - lo) / (hi - lo) for s in scores]
        by_coef[coef].extend(normalized)

    coefs = sorted(by_coef.keys())
    coef_pct = [c / MEAN_L31_NORM * 100 for c in coefs]
    means = [np.mean(by_coef[c]) for c in coefs]
    sems = [np.std(by_coef[c]) / np.sqrt(len(by_coef[c])) for c in coefs]
    return coef_pct, means, sems


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: Dose-response curves (3 rows = formats, 3 cols = positions)
# ─────────────────────────────────────────────────────────────────────────────


def plot_dose_response():
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True)
    fig.suptitle("Dose-Response: Normalized Rating vs Steering Coefficient\n(gemma-3-27b, ridge_L31)",
                 fontsize=13, y=1.01)

    stats_data = {(s["format"], s["position"]): s for s in load_statistics()}

    for row, fmt in enumerate(FORMATS):
        for col, pos in enumerate(POSITIONS):
            ax = axes[row][col]
            coef_pct, means, sems = get_dose_response(fmt, pos)

            if not coef_pct:
                ax.set_visible(False)
                continue

            color = POSITION_COLORS[pos]
            ax.errorbar(coef_pct, means, yerr=sems, fmt='-o', color=color,
                       markersize=4, linewidth=1.5, capsize=3, elinewidth=1)

            # Add neutral line
            ax.axhline(y=np.mean(means), color="gray", linestyle=":", linewidth=1, alpha=0.5)
            ax.axvline(x=0, color="gray", linestyle=":", linewidth=1, alpha=0.5)

            key = (fmt, pos)
            if key in stats_data:
                s = stats_data[key]
                p_str = f"p={s['p_value']:.4f}" if s['p_value'] >= 0.0001 else "p<0.0001"
                ax.set_title(f"{POSITION_LABELS[pos]}\nt={s['t_stat']:.1f}, {p_str}",
                            fontsize=9)

            ax.set_ylim(0, 1)
            ax.set_xlim(-11, 11)

            if row == 2:
                ax.set_xlabel("Coefficient (% of mean L31 norm)", fontsize=9)
            if col == 0:
                lo, hi = SCALE_RANGES[fmt]
                ax.set_ylabel(f"{FORMAT_LABELS[fmt]}\nNormalized rating [0,1]", fontsize=8)

    plt.tight_layout()
    outpath = ASSETS_DIR / "plot_022426_dose_response_by_format.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outpath}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: Per-task slope distributions (violin plots)
# ─────────────────────────────────────────────────────────────────────────────


def plot_slope_comparison():
    per_task_data = load_per_task_slopes()
    # Convert to dict for easy lookup
    slope_data: dict[tuple, list[float]] = {}
    for entry in per_task_data:
        key = (entry["format"], entry["position"])
        slopes = [s["slope"] for s in entry["slopes"]]
        slope_data[key] = slopes

    # Normalize slopes to scale-width for fair comparison
    # Divide by scale range so slope = "fraction of scale per unit coefficient"
    normalized_slopes: dict[tuple, list[float]] = {}
    for (fmt, pos), slopes in slope_data.items():
        lo, hi = SCALE_RANGES[fmt]
        normalized_slopes[(fmt, pos)] = [s / (hi - lo) for s in slopes]

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=False)
    fig.suptitle("Per-Task Slope Distribution: Normalized Rating ~ Coefficient\n"
                 "(slope normalized by scale width for comparability)", fontsize=12)

    for col, fmt in enumerate(FORMATS):
        ax = axes[col]
        data_per_pos = []
        labels = []
        for pos in POSITIONS:
            key = (fmt, pos)
            if key in normalized_slopes:
                data_per_pos.append(normalized_slopes[key])
                labels.append(POSITION_LABELS[pos])

        if not data_per_pos:
            continue

        # Box plot with individual points
        bp = ax.boxplot(data_per_pos, labels=labels, patch_artist=True,
                       medianprops=dict(color="black", linewidth=2),
                       flierprops=dict(marker=".", markersize=2))

        colors_for_pos = [POSITION_COLORS[pos] for pos in POSITIONS]
        for patch, color in zip(bp["boxes"], colors_for_pos):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.axhline(y=0, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="Zero")

        # Annotate with p-values
        stats_data = load_statistics()
        for i, pos in enumerate(POSITIONS):
            s_entry = next((s for s in stats_data if s["format"] == fmt and s["position"] == pos), None)
            if s_entry:
                p = s_entry["p_value"]
                sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
                ax.text(i + 1, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] != 0 else 1e-7,
                       sig, ha="center", fontsize=10, fontweight="bold")

        lo, hi = SCALE_RANGES[fmt]
        ax.set_title(FORMAT_LABELS[fmt].replace("\n", " "), fontsize=10)
        ax.set_ylabel("Slope (normalized / coefficient unit)" if col == 0 else "")
        ax.tick_params(axis='x', labelsize=9)

    plt.tight_layout()
    outpath = ASSETS_DIR / "plot_022426_slope_comparison.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outpath}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: Slope heatmap (format × position)
# ─────────────────────────────────────────────────────────────────────────────


def plot_slope_heatmap():
    stats_data = load_statistics()
    per_task_data = load_per_task_slopes()

    # Build normalized mean slope matrix
    slope_matrix = np.zeros((len(FORMATS), len(POSITIONS)))
    t_matrix = np.zeros((len(FORMATS), len(POSITIONS)))
    p_matrix = np.ones((len(FORMATS), len(POSITIONS)))

    for s in stats_data:
        row = FORMATS.index(s["format"])
        col = POSITIONS.index(s["position"])
        lo, hi = SCALE_RANGES[s["format"]]
        # Normalize: slope / scale_width → fraction of scale range per unit coefficient
        # Scale by 1000 for readability (per 1000 units of coefficient)
        slope_matrix[row, col] = s["mean_slope"] / (hi - lo) * 1e4
        t_matrix[row, col] = s["t_stat"]
        p_matrix[row, col] = s["p_value"]

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(slope_matrix, cmap="RdBu", aspect="auto",
                   vmin=-np.max(np.abs(slope_matrix)), vmax=np.max(np.abs(slope_matrix)))
    plt.colorbar(im, ax=ax, label="Normalized slope (×10⁻⁴ / coefficient unit)")

    ax.set_xticks(range(len(POSITIONS)))
    ax.set_xticklabels([POSITION_LABELS[p] for p in POSITIONS], fontsize=11)
    ax.set_yticks(range(len(FORMATS)))
    ax.set_yticklabels([FORMAT_LABELS[f].replace("\n", " ") for f in FORMATS], fontsize=11)

    for row in range(len(FORMATS)):
        for col in range(len(POSITIONS)):
            p = p_matrix[row, col]
            t = t_matrix[row, col]
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            val = slope_matrix[row, col]
            ax.text(col, row, f"{val:.1f}\n{sig}", ha="center", va="center",
                   fontsize=9, color="white" if abs(val) > np.max(np.abs(slope_matrix)) * 0.5 else "black")

    ax.set_title("Steerability Heatmap: Normalized Mean Slope\n(format × steering position, gemma-3-27b ridge_L31)",
                fontsize=11)
    plt.tight_layout()
    outpath = ASSETS_DIR / "plot_022426_slope_heatmap.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outpath}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4: Parse rates
# ─────────────────────────────────────────────────────────────────────────────


def plot_parse_rates():
    stats_data = load_statistics()

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(FORMATS))
    width = 0.25
    offsets = [-width, 0, width]

    for i, pos in enumerate(POSITIONS):
        parse_rates = []
        for fmt in FORMATS:
            s = next((s for s in stats_data if s["format"] == fmt and s["position"] == pos), None)
            parse_rates.append(s["parse_rate"] * 100 if s else 0)
        bars = ax.bar(x + offsets[i], parse_rates, width, label=POSITION_LABELS[pos],
                     color=POSITION_COLORS[pos], alpha=0.8)

    ax.set_xlabel("Format")
    ax.set_ylabel("Parse Rate (%)")
    ax.set_title("Parse Rates by Format × Position", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([FORMAT_LABELS[f].replace("\n", " ") for f in FORMATS], fontsize=10)
    ax.legend(title="Position")
    ax.set_ylim(0, 105)
    ax.axhline(y=100, color="gray", linestyle=":", linewidth=1)

    plt.tight_layout()
    outpath = ASSETS_DIR / "plot_022426_parse_rates.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outpath}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5: Dose-response with all positions overlaid per format
# ─────────────────────────────────────────────────────────────────────────────


def plot_dose_response_per_format():
    """One panel per format, all positions overlaid."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle("Dose-Response by Format (all positions overlaid)\nNormalized rating vs steering coefficient",
                fontsize=12)

    stats_data = {(s["format"], s["position"]): s for s in load_statistics()}

    for col, fmt in enumerate(FORMATS):
        ax = axes[col]
        ax.set_title(FORMAT_LABELS[fmt].replace("\n", " "), fontsize=11)

        for pos in POSITIONS:
            coef_pct, means, sems = get_dose_response(fmt, pos)
            if not coef_pct:
                continue
            color = POSITION_COLORS[pos]
            s = stats_data.get((fmt, pos), {})
            p = s.get("p_value", 1.0)
            p_str = f"p<0.0001" if p < 0.0001 else f"p={p:.3f}"
            label = f"{POSITION_LABELS[pos]} ({p_str})"
            ax.errorbar(coef_pct, means, yerr=sems, fmt='-o', color=color,
                       markersize=4, linewidth=2, capsize=3, elinewidth=1, label=label)

        ax.axvline(x=0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
        ax.set_xlabel("Coefficient (% of mean L31 norm)", fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_xlim(-11, 11)
        if col == 0:
            ax.set_ylabel("Normalized rating [0,1]", fontsize=10)
        ax.legend(fontsize=7, loc="best")

    plt.tight_layout()
    outpath = ASSETS_DIR / "plot_022426_dose_response_per_format.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outpath}")


if __name__ == "__main__":
    print("=== Generating plots ===")
    plot_dose_response()
    plot_slope_comparison()
    plot_slope_heatmap()
    plot_parse_rates()
    plot_dose_response_per_format()
    print("Done.")
