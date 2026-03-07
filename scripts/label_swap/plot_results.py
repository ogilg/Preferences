"""Generate plots for the label-swap EOT patching experiment."""

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = Path("experiments/patching/eot_transfer/label_swap")
ASSETS_DIR = DATA_DIR / "assets"
CHECKPOINT_PATH = DATA_DIR / "checkpoint.jsonl"


def load_checkpoint():
    records = []
    with open(CHECKPOINT_PATH) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def plot_choice_distribution(records):
    baseline_counts = Counter()
    patched_counts = Counter()
    for rec in records:
        for c in rec["baseline_choices"]:
            baseline_counts[c] += 1
        for c in rec["patched_choices"]:
            patched_counts[c] += 1

    total_baseline = sum(baseline_counts.values())
    total_patched = sum(patched_counts.values())

    # "a" = first slot (Task B label), "b" = second slot (Task A label)
    categories = ["First slot\n(Task B label)", "Second slot\n(Task A label)"]
    baseline_fracs = [
        baseline_counts["a"] / total_baseline,
        baseline_counts["b"] / total_baseline,
    ]
    patched_fracs = [
        patched_counts["a"] / total_patched,
        patched_counts["b"] / total_patched,
    ]
    baseline_raw = [baseline_counts["a"], baseline_counts["b"]]
    patched_raw = [patched_counts["a"], patched_counts["b"]]

    x = np.arange(len(categories))
    width = 0.3

    fig, ax = plt.subplots(figsize=(7, 5))
    bars_b = ax.bar(x - width / 2, baseline_fracs, width, label="Baseline", color="#4878CF")
    bars_p = ax.bar(x + width / 2, patched_fracs, width, label="Patched", color="#E8853A")

    for bar, count, total in zip(bars_b, baseline_raw, [total_baseline] * 2):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{count}/{total}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar, count, total in zip(bars_p, patched_raw, [total_patched] * 2):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{count}/{total}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylabel("Fraction of trials")
    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_title("Label Swap: Choice Distribution (Baseline vs Patched)")
    ax.legend(loc="upper right")

    ax.annotate(
        "Donor encodes 'pick Task A' (= first slot in donor)",
        xy=(0.5, 0.92),
        xycoords="axes fraction",
        ha="center",
        fontsize=9,
        fontstyle="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray", alpha=0.8),
    )

    fig.tight_layout()
    out_path = ASSETS_DIR / "plot_030726_choice_distribution.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_per_ordering_comparison(records):
    ordering_indices = []
    baseline_rates = []
    patched_rates = []

    for rec in records:
        idx = rec["ordering_idx"]
        ordering_indices.append(idx)
        # "b" = second slot = Task A label; count fraction picking Task A label
        b_base = sum(1 for c in rec["baseline_choices"] if c == "b")
        b_patch = sum(1 for c in rec["patched_choices"] if c == "b")
        n_trials = len(rec["baseline_choices"])
        baseline_rates.append(b_base / n_trials)
        patched_rates.append(b_patch / n_trials)

    ordering_indices = np.array(ordering_indices)
    baseline_rates = np.array(baseline_rates)
    patched_rates = np.array(patched_rates)
    n_orderings = len(records)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.scatter(
        ordering_indices,
        baseline_rates,
        marker="o",
        s=25,
        color="#4878CF",
        alpha=0.7,
        label="Baseline",
        zorder=3,
    )
    ax.scatter(
        ordering_indices,
        patched_rates,
        marker="^",
        s=25,
        color="#E8853A",
        alpha=0.7,
        label="Patched",
        zorder=3,
    )

    highlight_idxs = [11, 57, 68]
    for hi in highlight_idxs:
        pos = np.where(ordering_indices == hi)[0][0]
        b_val = baseline_rates[pos]
        p_val = patched_rates[pos]
        max_val = max(b_val, p_val)

        ax.annotate(
            f"idx={hi}",
            xy=(hi, max_val),
            xytext=(hi, max_val + 0.12),
            ha="center",
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
        )

    ax.set_xlabel("Ordering index")
    ax.set_ylabel("Fraction picking 'Task A' label (second slot)")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-2, n_orderings + 1)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title("Per-Ordering 'Task A' Label Selection Rate")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = ASSETS_DIR / "plot_030726_per_ordering_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    records = load_checkpoint()
    print(f"Loaded {len(records)} orderings from checkpoint")
    plot_choice_distribution(records)
    plot_per_ordering_comparison(records)


if __name__ == "__main__":
    main()
