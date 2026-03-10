import json
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

DATA_PATH = Path("experiments/eot_probes/turn_boundary_sweep/results_summary.json")
ASSETS_DIR = Path("experiments/eot_probes/turn_boundary_sweep/assets")

HOO_DIR = Path("results/probes")
HOO_SELECTOR_DIRS = {
    "tb-5": HOO_DIR / "gemma3_10k_hoo_topic_tb-5",
    "tb-4": HOO_DIR / "gemma3_10k_hoo_topic_tb-4",
    "tb-3": HOO_DIR / "gemma3_10k_hoo_topic_tb-3",
    "tb-2": HOO_DIR / "gemma3_10k_hoo_topic_tb-2",
    "tb-1": HOO_DIR / "gemma3_10k_hoo_topic_tb-1",
    "task_mean": HOO_DIR / "gemma3_10k_hoo_topic_task_mean",
}

SELECTOR_LABELS = {
    "tb-5": r"$\tt{<end\_of\_turn>}$ (tb-5)",
    "tb-4": r"$\tt{\backslash n}$ after EOT (tb-4)",
    "tb-3": r"$\tt{<start\_of\_turn>}$ (tb-3)",
    "tb-2": r"$\tt{model}$ (tb-2)",
    "tb-1": r"$\tt{\backslash n}$ final (tb-1)",
    "task_mean": "task_mean (baseline)",
}

# Colorblind-friendly palette (Wong 2011)
COLORS = {
    "tb-5": "#0072B2",      # blue
    "tb-4": "#E69F00",      # orange
    "tb-3": "#CC79A7",      # pink
    "tb-2": "#009E73",      # green
    "tb-1": "#D55E00",      # vermillion
    "task_mean": "#999999",  # grey
}

MARKERS = {
    "tb-5": "o",
    "tb-4": "s",
    "tb-3": "D",
    "tb-2": "^",
    "tb-1": "v",
    "task_mean": "X",
}


def load_hoo_data() -> dict:
    """Load HOO mean_r and std_hoo_r from per-selector hoo_summary.json files."""
    hoo_mean = {}
    hoo_se = {}
    layers = None

    for sel, sel_dir in HOO_SELECTOR_DIRS.items():
        summary_path = sel_dir / "hoo_summary.json"
        with open(summary_path) as f:
            summary = json.load(f)

        if layers is None:
            layers = summary["layers"]

        layer_summary = summary["layer_summary"]
        hoo_mean[sel] = {}
        hoo_se[sel] = {}
        for layer_str, layer_data in layer_summary.items():
            ridge = layer_data["ridge"]
            n_folds = ridge["n_folds"]
            hoo_mean[sel][layer_str] = ridge["mean_hoo_r"]
            hoo_se[sel][layer_str] = ridge["std_hoo_r"] / math.sqrt(n_folds)

    return {"layers": layers, "hoo_mean": hoo_mean, "hoo_se": hoo_se}


def make_plot(
    metric_data: dict,
    layers: list[int],
    title: str,
    out_path: Path,
    error_data: dict | None = None,
) -> None:
    # Sort selectors by peak r value (descending) for legend ordering
    peak_r = {sel: max(metric_data[sel].values()) for sel in metric_data}
    sorted_selectors = sorted(peak_r, key=lambda s: peak_r[s], reverse=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    for sel in sorted_selectors:
        values = [metric_data[sel][str(layer)] for layer in layers]
        plot_kwargs = dict(
            label=SELECTOR_LABELS[sel],
            color=COLORS[sel],
            marker=MARKERS[sel],
            linewidth=2.5,
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=0.8,
        )

        if error_data is not None:
            yerr = [error_data[sel][str(layer)] for layer in layers]
            ax.errorbar(
                layers,
                values,
                yerr=yerr,
                capsize=3,
                **plot_kwargs,
            )
        else:
            ax.plot(layers, values, **plot_kwargs)

    ax.set_xlabel("Layer", fontsize=13)
    ax.set_ylabel("Pearson r", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")

    ax.set_ylim(0, 1.0)
    ax.set_xticks(layers)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))

    ax.grid(axis="y", alpha=0.3, linewidth=0.8)
    ax.grid(axis="y", which="minor", alpha=0.15, linewidth=0.5)

    ax.legend(fontsize=10, loc="lower left", framealpha=0.9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(labelsize=11)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    with open(DATA_PATH) as f:
        data = json.load(f)

    layers = data["layers"]

    make_plot(
        metric_data=data["heldout"],
        layers=layers,
        title="Heldout Eval: Pearson r by Layer and Token Position",
        out_path=ASSETS_DIR / "plot_031026_heldout_r_by_layer.png",
    )

    hoo_data = load_hoo_data()
    make_plot(
        metric_data=hoo_data["hoo_mean"],
        layers=layers,
        title="Hold-One-Out by Topic: Mean Cross-Topic r by Layer and Token Position",
        out_path=ASSETS_DIR / "plot_031026_hoo_r_by_layer.png",
        error_data=hoo_data["hoo_se"],
    )


if __name__ == "__main__":
    main()
