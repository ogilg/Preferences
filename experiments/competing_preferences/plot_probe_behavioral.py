"""Probe-behavioral correlation scatter for competing preferences."""

import json
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path("experiments/competing_preferences/results")
ASSETS_DIR = Path("docs/logs/assets/competing_preferences")
DATE_STR = datetime.now().strftime("%m%d%y")


def main():
    with open(RESULTS_DIR / "probe_competing.json") as f:
        data = json.load(f)

    probe_deltas = [d["probe_delta_L31"] for d in data]
    beh_deltas = [d["behavioral_delta"] for d in data]
    labels = [f"{d['pair_id']}\n({'subj+' if d['favored_dim'] == 'topic' else 'type+'})" for d in data]
    favored = [d["favored_dim"] for d in data]

    r, p = stats.pearsonr(probe_deltas, beh_deltas)
    print(f"Probe-behavioral correlation (L31): r={r:.3f}, p={p:.2e}")

    fig, ax = plt.subplots(figsize=(8, 6))

    # Color by condition
    colors = ["#FF7043" if f == "topic" else "#42A5F5" for f in favored]
    ax.scatter(probe_deltas, beh_deltas, c=colors, s=60, alpha=0.7, edgecolors="k", linewidth=0.5, zorder=5)

    # Regression line
    slope, intercept = np.polyfit(probe_deltas, beh_deltas, 1)
    x_range = np.linspace(min(probe_deltas) - 20, max(probe_deltas) + 20, 100)
    ax.plot(x_range, slope * x_range + intercept, color="gray", linewidth=1.5, alpha=0.5, zorder=1)

    # Link paired conditions (same pair, different prompt) with lines
    pair_ids = sorted(set(d["pair_id"] for d in data))
    for pid in pair_ids:
        pair_points = [d for d in data if d["pair_id"] == pid]
        if len(pair_points) == 2:
            x_vals = [d["probe_delta_L31"] for d in pair_points]
            y_vals = [d["behavioral_delta"] for d in pair_points]
            ax.plot(x_vals, y_vals, color="gray", linewidth=0.8, alpha=0.4, zorder=2)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF7043', markersize=8, label='"Love subject, hate task type"'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#42A5F5', markersize=8, label='"Love task type, hate subject"'),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="upper left")

    ax.set_xlabel("Probe delta (L31)", fontsize=11)
    ax.set_ylabel("Behavioral delta (choice rate change)", fontsize=11)
    ax.set_title(f"Probe-Behavioral Correlation Under Competing Preferences\nr = {r:.3f} (p = {p:.1e}, n = {len(data)})", fontsize=12)
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.3)
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    path = ASSETS_DIR / f"plot_{DATE_STR}_probe_behavioral_scatter.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
