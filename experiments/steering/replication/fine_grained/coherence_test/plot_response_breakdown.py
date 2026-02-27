"""Plot response type breakdown by steering coefficient."""

import json
import re
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def classify_response(response: str) -> str:
    """Classify a response into Task A, Task B, Refusal, or Other."""
    # Strip leading markdown formatting (bold markers, headers, whitespace, etc.)
    stripped = re.sub(r"^[\s\*\#\-\_\>]+", "", response).strip()
    first_lower = stripped[:300].lower()

    # Check Task A / Task B first
    if first_lower.startswith("task a"):
        return "Task A"
    if first_lower.startswith("task b"):
        return "Task B"
    # "I will complete **Task A**" pattern
    if re.match(r"i will complete\s+\*{0,2}task a", first_lower):
        return "Task A"
    if re.match(r"i will complete\s+\*{0,2}task b", first_lower):
        return "Task B"

    # Check refusal patterns in first 300 chars
    refusal_phrases = [
        "cannot and will not",
        "unable to fulfill",
        "must decline",
        "i am programmed to be",
    ]
    snippet = response[:300].lower()
    for phrase in refusal_phrases:
        if phrase in snippet:
            return "Refusal"

    return "Other"


def main():
    with open(
        "experiments/steering/replication/fine_grained/coherence_test/results/raw_responses.json"
    ) as f:
        data = json.load(f)

    # Classify all responses, grouped by pct_norm
    counts = defaultdict(lambda: defaultdict(int))
    for entry in data:
        pct = entry["pct_norm"]
        for resp in entry["responses"]:
            label = classify_response(resp)
            counts[pct][label] += 1

    # Sort coefficients
    pct_norms = sorted(counts.keys())
    categories = ["Task A", "Task B", "Refusal", "Other"]
    colors = {
        "Task A": "#4878CF",    # blue
        "Task B": "#E8873A",    # orange
        "Refusal": "#999999",   # gray
        "Other": "#D44D4D",     # red
    }

    # Build arrays
    cat_counts = {cat: [counts[p][cat] for p in pct_norms] for cat in categories}

    # Format x-tick labels
    x_labels = []
    for p in pct_norms:
        if p == 0:
            x_labels.append("0%")
        elif p > 0:
            x_labels.append(f"+{p:g}%")
        else:
            x_labels.append(f"{p:g}%")

    x = np.arange(len(pct_norms))
    bar_width = 0.7

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 5))

    bottom = np.zeros(len(pct_norms))
    for cat in categories:
        vals = np.array(cat_counts[cat])
        ax.bar(x, vals, bar_width, bottom=bottom, label=cat, color=colors[cat],
               edgecolor="white", linewidth=0.4)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_xlabel("Steering Coefficient (% of Mean Norm)", fontsize=11)
    ax.set_ylabel("Response Count", fontsize=11)
    ax.set_title("Response Type Breakdown by Steering Coefficient", fontsize=13,
                 fontweight="bold")
    ax.legend(loc="upper right", frameon=True, fontsize=9)

    # Clean up
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(bottom) + 3)

    plt.tight_layout()
    out_path = "experiments/steering/replication/fine_grained/coherence_test/assets/plot_022726_response_breakdown.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {out_path}")

    # Print summary
    for p in pct_norms:
        parts = ", ".join(f"{cat}: {counts[p][cat]}" for cat in categories)
        total = sum(counts[p][cat] for cat in categories)
        print(f"  pct_norm={p:+6.1f}%  total={total}  {parts}")


if __name__ == "__main__":
    main()
