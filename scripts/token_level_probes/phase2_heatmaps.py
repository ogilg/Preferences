"""Phase 2: Per-token score heatmaps for representative items.

Generates heatmap visualizations showing probe scores at each token position,
with critical spans highlighted.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import TwoSlopeNorm

DATA_PATH = Path("experiments/token_level_probes/scoring_results.json")
NPZ_PATH = Path("experiments/token_level_probes/all_token_scores.npz")
ASSETS_DIR = Path("experiments/token_level_probes/assets")

PROBE = "task_mean_L39"

# Representative items to visualize
SELECTED_ITEMS = {
    "truth": [
        "truth_0_true_user",
        "truth_0_false_user",
        "truth_0_true_assistant",
        "truth_0_false_assistant",
    ],
    "harm": [
        "harm_2_harmful_user",
        "harm_2_benign_user",
        "harm_2_harmful_assistant",
        "harm_2_benign_assistant",
    ],
    "politics": [
        "politics_0_left_neutral",
        "politics_0_right_neutral",
    ],
}

# Grid layout: rows/cols for each domain figure
GRID_LAYOUT = {
    "truth": {
        "shape": (2, 2),
        "labels": [
            ("true / user", 0, 0),
            ("false / user", 1, 0),
            ("true / assistant", 0, 1),
            ("false / assistant", 1, 1),
        ],
    },
    "harm": {
        "shape": (2, 2),
        "labels": [
            ("harmful / user", 0, 0),
            ("benign / user", 1, 0),
            ("harmful / assistant", 0, 1),
            ("benign / assistant", 1, 1),
        ],
    },
    "politics": {
        "shape": (1, 2),
        "labels": [
            ("left / neutral", 0, 0),
            ("right / neutral", 0, 1),
        ],
    },
}


def load_data():
    with open(DATA_PATH) as f:
        results = json.load(f)
    items_by_id = {item["id"]: item for item in results["items"]}
    scores_npz = np.load(NPZ_PATH)
    return items_by_id, scores_npz


def get_window(n_tokens, critical_indices, context=20):
    """Determine the token window to show.

    If the total number of tokens is <= 60, show all tokens.
    Otherwise, show a window around the critical span with ±context tokens.
    """
    if n_tokens <= 60:
        return 0, n_tokens
    crit_min = min(critical_indices)
    crit_max = max(critical_indices)
    start = max(0, crit_min - context)
    end = min(n_tokens, crit_max + context + 1)
    return start, end


def sanitize_token(tok):
    """Make token text safe for matplotlib display."""
    tok = tok.replace("\n", "\\n")
    tok = tok.replace("\t", "\\t")
    tok = tok.replace("$", "\\$")
    return tok


def plot_single_heatmap(ax, tokens, scores, critical_indices, title, vmin, vmax):
    """Plot a single token-level heatmap on the given axes."""
    n_tokens = len(tokens)
    window_start, window_end = get_window(n_tokens, critical_indices, context=20)
    n_display = window_end - window_start

    display_scores = scores[window_start:window_end]
    display_tokens = tokens[window_start:window_end]
    display_critical = [i - window_start for i in critical_indices
                        if window_start <= i < window_end]

    norm = TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)

    score_2d = display_scores.reshape(1, -1)
    im = ax.imshow(score_2d, aspect="auto", cmap="RdBu_r", norm=norm,
                   extent=[-0.5, n_display - 0.5, -0.5, 0.5])

    # Highlight critical span with a rectangle
    if display_critical:
        crit_start = min(display_critical) - 0.5
        crit_width = max(display_critical) - min(display_critical) + 1
        rect = mpatches.Rectangle(
            (crit_start, -0.5), crit_width, 1.0,
            linewidth=2.5, edgecolor="black", facecolor="none",
            linestyle="-", zorder=5,
        )
        ax.add_patch(rect)

    # Token annotations on x-axis
    step = max(1, n_display // 60)
    tick_positions = list(range(0, n_display, step))
    tick_labels = [sanitize_token(display_tokens[i]) for i in tick_positions]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=6, fontfamily="monospace")
    ax.set_yticks([])

    window_label = ""
    if window_start > 0 or window_end < n_tokens:
        window_label = f"  [tokens {window_start}-{window_end - 1} of {n_tokens}]"
    ax.set_title(f"{title}{window_label}", fontsize=10, fontweight="bold")

    return im


def print_top_bottom_tokens(item_id, tokens, scores):
    """Print the top 5 highest and bottom 5 lowest scoring tokens."""
    n = min(len(tokens), len(scores))
    indices = np.argsort(scores[:n])

    print(f"\n{'='*70}")
    print(f"Item: {item_id}")
    print(f"{'='*70}")

    print("  Top 5 highest-scoring tokens:")
    for idx in indices[-5:][::-1]:
        print(f"    pos={idx:3d}  score={scores[idx]:+8.3f}  token='{tokens[idx]}'")

    print("  Bottom 5 lowest-scoring tokens:")
    for idx in indices[:5]:
        print(f"    pos={idx:3d}  score={scores[idx]:+8.3f}  token='{tokens[idx]}'")


def make_domain_figure(domain, item_ids, items_by_id, scores_npz):
    """Create a grid figure for one domain."""
    layout = GRID_LAYOUT[domain]
    n_rows, n_cols = layout["shape"]

    # Compute shared color scale across all items in this domain
    all_scores_for_domain = []
    item_data = []
    for item_id in item_ids:
        item = items_by_id[item_id]
        key = f"{item_id}__{PROBE}"
        scores = scores_npz[key]
        tokens = item["tokens"]
        n_tokens = len(tokens)
        # Truncate scores to match token count (extra score is from generation prompt token)
        scores = scores[:n_tokens]
        all_scores_for_domain.append(scores)
        item_data.append((item_id, item, tokens, scores))

    all_concat = np.concatenate(all_scores_for_domain)
    vmax = max(abs(all_concat.min()), abs(all_concat.max()))
    vmin = -vmax

    # Print top/bottom tokens for each item
    for item_id, item, tokens, scores in item_data:
        print_top_bottom_tokens(item_id, tokens, scores)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(20, 3 * n_rows),
        squeeze=False,
    )

    for i, (item_id, item, tokens, scores) in enumerate(item_data):
        label_text, row_idx, col_idx = layout["labels"][i]
        ax = axes[row_idx, col_idx]
        critical_indices = item["critical_token_indices"]
        title = f"{item_id} ({label_text})"
        im = plot_single_heatmap(ax, tokens, scores, critical_indices, title, vmin, vmax)

    # Add shared colorbar
    fig.subplots_adjust(right=0.92, hspace=0.6)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label=f"{PROBE} score")

    fig.suptitle(f"Token-level probe scores: {domain} domain ({PROBE})",
                 fontsize=14, fontweight="bold", y=1.02)

    return fig


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    items_by_id, scores_npz = load_data()

    output_files = {
        "truth": ASSETS_DIR / "plot_031426_truth_token_heatmaps.png",
        "harm": ASSETS_DIR / "plot_031426_harm_token_heatmaps.png",
        "politics": ASSETS_DIR / "plot_031426_politics_token_heatmaps.png",
    }

    for domain in ["truth", "harm", "politics"]:
        item_ids = SELECTED_ITEMS[domain]
        fig = make_domain_figure(domain, item_ids, items_by_id, scores_npz)
        fig.savefig(output_files[domain], dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSaved: {output_files[domain]}")

    print("\nDone.")


if __name__ == "__main__":
    main()
