"""Generate 3 analysis plots from per-token scored data."""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = Path(
    "/workspace/repo/experiments/truth_probes/error_prefill/per_token_scoring/scored_tokens.json"
)
ASSETS_DIR = DATA_PATH.parent / "assets"

LAYERS = ["L25", "L32", "L39", "L46", "L53"]
MIN_CONVERSATIONS = 10


def load_data():
    with open(DATA_PATH) as f:
        return json.load(f)


def split_by_condition(data):
    correct = [d for d in data if d["answer_condition"] == "correct"]
    incorrect = [d for d in data if d["answer_condition"] == "incorrect"]
    return correct, incorrect


def get_scores_array(entries, probe, layer):
    """Return list of lists (ragged) of per-token scores."""
    return [entry["scores"][probe][layer] for entry in entries]


def compute_cohens_d(scores_a, scores_b):
    """Cohen's d = (mean_a - mean_b) / pooled_std."""
    if len(scores_a) < 2 or len(scores_b) < 2:
        return float("nan")
    mean_a, mean_b = np.mean(scores_a), np.mean(scores_b)
    std_a, std_b = np.std(scores_a, ddof=1), np.std(scores_b, ddof=1)
    pooled = math.sqrt((std_a**2 + std_b**2) / 2)
    if pooled == 0:
        return float("nan")
    return (mean_a - mean_b) / pooled


def plot_score_trajectories(data):
    """Plot 1: Mean score trajectory for correct vs incorrect, tb-5 L39."""
    correct, incorrect = split_by_condition(data)
    probe, layer = "tb-5", "L39"

    correct_scores = get_scores_array(correct, probe, layer)
    incorrect_scores = get_scores_array(incorrect, probe, layer)

    max_len = max(
        max(len(s) for s in correct_scores), max(len(s) for s in incorrect_scores)
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    # Individual traces
    for scores in correct_scores:
        ax.plot(
            range(len(scores)), scores, color="green", alpha=0.15, linewidth=0.5
        )
    for scores in incorrect_scores:
        ax.plot(
            range(len(scores)), scores, color="red", alpha=0.15, linewidth=0.5
        )

    # Compute mean and SEM at each position, filtering by min conversations
    for group_scores, color, label_prefix, n_count in [
        (correct_scores, "green", "Correct", len(correct)),
        (incorrect_scores, "red", "Incorrect", len(incorrect)),
    ]:
        means = []
        sems = []
        positions = []
        for pos in range(max_len):
            vals = [s[pos] for s in group_scores if pos < len(s)]
            if len(vals) < MIN_CONVERSATIONS:
                break
            means.append(np.mean(vals))
            sems.append(np.std(vals, ddof=1) / math.sqrt(len(vals)))
            positions.append(pos)

        means = np.array(means)
        sems = np.array(sems)
        positions = np.array(positions)

        ax.plot(
            positions,
            means,
            color=color,
            linewidth=2.5,
            label=f"{label_prefix} (n={n_count})",
        )
        ax.fill_between(
            positions, means - sems, means + sems, color=color, alpha=0.2
        )

    ax.set_xlabel("Token position")
    ax.set_ylabel("Probe score")
    ax.set_title("Per-token probe score trajectories (tb-5, L39)")
    ax.legend()

    # Meaningful y-axis: find data range and anchor to round number
    all_scores_flat = [v for s in correct_scores + incorrect_scores for v in s]
    y_min_data = min(all_scores_flat)
    y_max_data = max(all_scores_flat)
    # Round down to nearest 5 below min, round up to nearest 5 above max
    y_min = math.floor(y_min_data / 5) * 5
    y_max = math.ceil(y_max_data / 5) * 5
    ax.set_ylim(y_min, y_max)

    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = ASSETS_DIR / "plot_031226_score_trajectories_L39.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_position_cohens_d(data):
    """Plot 2: Position-wise Cohen's d across layers, tb-5 and tb-2."""
    correct, incorrect = split_by_condition(data)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    for ax, probe, title_suffix in [
        (axes[0], "tb-5", "tb-5"),
        (axes[1], "tb-2", "tb-2"),
    ]:
        max_len = max(
            max(entry["n_tokens"] for entry in correct),
            max(entry["n_tokens"] for entry in incorrect),
        )

        for layer in LAYERS:
            correct_scores = get_scores_array(correct, probe, layer)
            incorrect_scores = get_scores_array(incorrect, probe, layer)

            ds = []
            positions = []
            for pos in range(max_len):
                c_vals = [s[pos] for s in correct_scores if pos < len(s)]
                i_vals = [s[pos] for s in incorrect_scores if pos < len(s)]
                if len(c_vals) < MIN_CONVERSATIONS or len(i_vals) < MIN_CONVERSATIONS:
                    break
                d = compute_cohens_d(c_vals, i_vals)
                ds.append(d)
                positions.append(pos)

            ax.plot(positions, ds, label=layer, linewidth=1.5)

        ax.set_xlabel("Token position")
        ax.set_ylabel("Cohen's d (correct - incorrect)")
        ax.set_title(f"Position-wise Cohen's d ({title_suffix})")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Y-axis: start at 0 or slightly negative
        y_min = -0.5
        ax.set_ylim(bottom=y_min)

    fig.tight_layout()
    out = ASSETS_DIR / "plot_031226_position_cohens_d.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_first_vs_last_scatter(data):
    """Plot 3: First vs last token score scatter, tb-2 and tb-5, L39."""
    correct, incorrect = split_by_condition(data)
    layer = "L39"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, probe, title in [
        (axes[0], "tb-2", "tb-2 (L39)"),
        (axes[1], "tb-5", "tb-5 (L39)"),
    ]:
        all_first = []
        all_last = []
        all_colors = []

        for group, color in [(correct, "green"), (incorrect, "red")]:
            for entry in group:
                scores = entry["scores"][probe][layer]
                all_first.append(scores[0])
                all_last.append(scores[-1])
                all_colors.append(color)

        all_first = np.array(all_first)
        all_last = np.array(all_last)

        # Scatter: correct first so incorrect is on top (or vice versa for visibility)
        for color, label in [("green", "Correct"), ("red", "Incorrect")]:
            mask = np.array(all_colors) == color
            ax.scatter(
                all_first[mask],
                all_last[mask],
                c=color,
                label=label,
                alpha=0.6,
                edgecolors="none",
                s=40,
            )

        # Identity line
        combined = np.concatenate([all_first, all_last])
        lo = math.floor(min(combined) / 5) * 5
        hi = math.ceil(max(combined) / 5) * 5
        ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=1, zorder=0)

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("First token score")
        ax.set_ylabel("Last token score")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

    fig.tight_layout()
    out = ASSETS_DIR / "plot_031226_first_vs_last_scatter.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    data = load_data()
    correct, incorrect = split_by_condition(data)
    print(f"Loaded {len(data)} entries: {len(correct)} correct, {len(incorrect)} incorrect")

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    plot_score_trajectories(data)
    plot_position_cohens_d(data)
    plot_first_vs_last_scatter(data)
    print("All plots generated.")


if __name__ == "__main__":
    main()
