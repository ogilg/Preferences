"""Confusion matrix between two classifier models on topic classification."""

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path("src/analysis/topic_classification/output")

M1 = "openai/gpt-5-nano"
M2 = "google/gemini-3-flash-preview"


def main():
    cache = json.load(open(OUTPUT_DIR / "topics.json"))

    m1_labels = [e[M1]["primary"] for e in cache.values()]
    m2_labels = [e[M2]["primary"] for e in cache.values()]

    combined = Counter(m1_labels) + Counter(m2_labels)
    categories = [cat for cat, _ in combined.most_common()]

    cat_to_idx = {cat: i for i, cat in enumerate(categories)}
    n = len(categories)

    # Build confusion matrix: rows = model 1, cols = model 2
    matrix = np.zeros((n, n), dtype=int)
    for m1, m2 in zip(m1_labels, m2_labels):
        matrix[cat_to_idx[m1], cat_to_idx[m2]] += 1

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(matrix, cmap="Blues")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(categories, fontsize=9)
    ax.set_xlabel("gemini-3-flash", fontsize=12)
    ax.set_ylabel("gpt-5-nano", fontsize=12)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if val == 0:
                continue
            color = "white" if val > matrix.max() * 0.5 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=8)

    agree = sum(matrix[i, i] for i in range(n))
    total = matrix.sum()
    ax.set_title(
        f"Topic Classification: gpt-5-nano vs gemini-3-flash\n"
        f"Agreement: {agree}/{total} ({agree/total*100:.1f}%)",
        fontsize=13,
    )

    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()

    out_path = OUTPUT_DIR / "plot_020626_model_confusion_matrix_v2.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
