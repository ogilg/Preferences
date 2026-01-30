"""Plot concept vector steering experiment results as violin plots.

Supports the 3x3 steering experiment format:
- completion_source × measurement_context × layer × coefficient
- Loads coeff=0 baseline from a separate sysprompt_3x3 experiment
- Produces one PNG per layer with 3x3 grid (rows=sources, cols=contexts)
"""

import argparse
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Patch

VALENCE_ORDER = ["positive", "neutral", "negative"]
TEXT_Y_TOP = 5.3
TEXT_Y_BOTTOM = 0.6


def discover_conditions(
    results_dir: Path,
) -> tuple[list[str], list[str], list[int], list[float]]:
    """Discover sources, contexts, layers, and coefficients from directory names.

    Supports two patterns:
    - New 3x3: completion_{source}_context_{context}_layer{layer}_coef{coef}
    - Old: completion_{source}_layer{layer}_coef{coef}
    """
    pattern_3x3 = re.compile(
        r"completion_(\w+)_context_(\w+)_layer(\d+)_coef(-?\d+\.?\d*)"
    )
    pattern_old = re.compile(r"completion_(\w+)_layer(\d+)_coef(-?\d+\.?\d*)")

    sources = set()
    contexts = set()
    layers = set()
    coefficients = set()
    is_3x3 = False

    for d in results_dir.iterdir():
        if not d.is_dir():
            continue
        if match := pattern_3x3.match(d.name):
            is_3x3 = True
            sources.add(match.group(1))
            contexts.add(match.group(2))
            layers.add(int(match.group(3)))
            coefficients.add(float(match.group(4)))
        elif match := pattern_old.match(d.name):
            sources.add(match.group(1))
            layers.add(int(match.group(2)))
            coefficients.add(float(match.group(3)))

    def sort_valence(items: set[str]) -> list[str]:
        return [v for v in VALENCE_ORDER if v in items] + sorted(items - set(VALENCE_ORDER))

    return (
        sort_valence(sources),
        sort_valence(contexts) if is_3x3 else [],
        sorted(layers),
        sorted(coefficients),
    )


def _load_scores_from_path(path: Path) -> list[float]:
    if not path.exists():
        return []
    with open(path) as f:
        data = yaml.safe_load(f)
    return [
        item["score"]
        for item in data
        if item["score"] is not None and 1 <= item["score"] <= 5
    ]


def load_scores(
    results_dir: Path,
    source: str,
    context: str | None,
    layer: int,
    coef: float,
) -> list[float]:
    if context:
        condition = f"completion_{source}_context_{context}_layer{layer}_coef{coef}"
    else:
        condition = f"completion_{source}_layer{layer}_coef{coef}"
    return _load_scores_from_path(results_dir / condition / "measurements.yaml")


def load_baseline_scores(
    baseline_experiment: str,
    completion_source: str,
    measurement_context: str,
) -> list[float]:
    path = (
        Path(f"results/experiments/{baseline_experiment}/post_task_stated")
        / f"completion_{completion_source}_context_{measurement_context}"
        / "measurements.yaml"
    )
    return _load_scores_from_path(path)


def load_experiment_config(experiment_id: str) -> dict | None:
    config_path = Path(f"results/experiments/{experiment_id}/config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    results_dir = Path(f"results/experiments/{experiment_id}/post_task_stated")
    if results_dir.exists():
        for subdir in results_dir.iterdir():
            if (config_path := subdir / "config.yaml").exists():
                with open(config_path) as f:
                    return yaml.safe_load(f)
    return None


def get_coefficient_color(coef: float, all_coefs: list[float]) -> str:
    if coef == 0.0:
        return "#BDBDBD"

    neg_coefs = sorted([c for c in all_coefs if c < 0], reverse=True)
    pos_coefs = sorted([c for c in all_coefs if c > 0])
    blues = ["#90CAF9", "#42A5F5", "#1976D2", "#0D47A1"]
    reds = ["#EF9A9A", "#EF5350", "#E53935", "#B71C1C"]

    if coef < 0:
        idx = neg_coefs.index(coef) if coef in neg_coefs else 0
        return blues[min(idx, len(blues) - 1)]
    idx = pos_coefs.index(coef) if coef in pos_coefs else 0
    return reds[min(idx, len(reds) - 1)]


def _style_violin(parts: dict, colors: list[str]) -> None:
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    for partname in ["cbars", "cmins", "cmaxes", "cmeans"]:
        if partname in parts:
            parts[partname].set_color("black")


def _collect_scores_for_subplot(
    all_coefs: list[float],
    results_dir: Path,
    source: str,
    context: str | None,
    layer: int,
    baseline_experiment: str | None,
) -> dict[float, list[float]]:
    """Collect scores for all coefficients in a subplot, loading each condition once."""
    scores_by_coef = {}
    for coef in all_coefs:
        if coef == 0.0:
            if baseline_experiment and context:
                scores_by_coef[coef] = load_baseline_scores(baseline_experiment, source, context)
            else:
                scores_by_coef[coef] = []
        else:
            scores_by_coef[coef] = load_scores(results_dir, source, context, layer, coef)
    return scores_by_coef


def _plot_subplot(
    ax: plt.Axes,
    scores_by_coef: dict[float, list[float]],
    all_coefs: list[float],
) -> None:
    data_to_plot = []
    positions = []
    colors_to_use = []

    for i, coef in enumerate(all_coefs):
        if scores := scores_by_coef[coef]:
            data_to_plot.append(scores)
            positions.append(i)
            colors_to_use.append(get_coefficient_color(coef, all_coefs))

    if data_to_plot:
        parts = ax.violinplot(
            data_to_plot,
            positions=positions,
            showmeans=True,
            showmedians=False,
            widths=0.7,
        )
        _style_violin(parts, colors_to_use)

    for i, coef in enumerate(all_coefs):
        if scores := scores_by_coef[coef]:
            ax.text(i, TEXT_Y_TOP, f"n={len(scores)}", ha="center", fontsize=8, color="gray")
            ax.text(i, TEXT_Y_BOTTOM, f"μ={np.mean(scores):.2f}", ha="center", fontsize=8, color="gray")

    ax.axhline(y=3, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_xticks(range(len(all_coefs)))
    ax.set_xticklabels([str(c) for c in all_coefs], fontsize=9)
    ax.set_ylim(0.3, 5.7)
    ax.set_yticks([1, 2, 3, 4, 5])


def plot_layer_3x3(
    results_dir: Path,
    layer: int,
    coefficients: list[float],
    baseline_experiment: str | None,
    experiment_id: str,
) -> Path:
    sources = contexts = VALENCE_ORDER
    fig, axes = plt.subplots(3, 3, figsize=(14, 12), sharey=True)
    all_coefs = sorted(set([0.0] + coefficients))

    source_labels = {v: f"{v.capitalize()} persona" for v in VALENCE_ORDER}
    context_labels = {v: f"{v.capitalize()} context" for v in VALENCE_ORDER}

    for row_idx, source in enumerate(sources):
        for col_idx, context in enumerate(contexts):
            ax = axes[row_idx, col_idx]
            scores_by_coef = _collect_scores_for_subplot(
                all_coefs, results_dir, source, context, layer, baseline_experiment
            )
            _plot_subplot(ax, scores_by_coef, all_coefs)

            if row_idx == 0:
                ax.set_title(context_labels[context], fontsize=12, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"{source_labels[source]}\n\nScore (1-5)", fontsize=10)
            if row_idx == 2:
                ax.set_xlabel("Steering coefficient", fontsize=10)

    fig.suptitle(
        f"Concept Vector Steering (Layer {layer})\n{experiment_id}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    legend_elements = [
        Patch(
            facecolor=get_coefficient_color(c, all_coefs),
            alpha=0.7,
            label=f"coef={c}" + (" (baseline)" if c == 0.0 else ""),
        )
        for c in all_coefs
    ]
    fig.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(0.98, 0.92))
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    date_str = datetime.now().strftime("%m%d%y")
    output_path = Path(f"src/analysis/concept_vectors/plots/plot_{date_str}_steering_layer{layer}.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_old_format(
    results_dir: Path,
    sources: list[str],
    layers: list[int],
    coefficients: list[float],
    experiment_id: str,
) -> Path:
    fig, axes = plt.subplots(
        len(sources),
        len(layers),
        figsize=(4 * len(layers), 3.5 * len(sources)),
        sharey=True,
        sharex=True,
        squeeze=False,
    )
    all_coefs = sorted(coefficients)

    for row_idx, source in enumerate(sources):
        for col_idx, layer in enumerate(layers):
            ax = axes[row_idx, col_idx]
            scores_by_coef = _collect_scores_for_subplot(
                all_coefs, results_dir, source, None, layer, None
            )
            _plot_subplot(ax, scores_by_coef, all_coefs)

            if row_idx == 0:
                ax.set_title(f"Layer {layer}", fontsize=12, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"Source: {source}\n\nScore (1-5)", fontsize=10)
            if row_idx == len(sources) - 1:
                ax.set_xlabel("Steering coefficient", fontsize=10)

    fig.suptitle(
        f"Effect of Concept Vector Steering on Stated Preferences\n({experiment_id})",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    legend_elements = [
        Patch(facecolor=get_coefficient_color(c, all_coefs), alpha=0.7, label=f"coef={c}")
        for c in all_coefs
    ]
    fig.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(0.98, 0.92))
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    date_str = datetime.now().strftime("%m%d%y")
    output_path = Path(f"src/analysis/concept_vectors/plots/plot_{date_str}_steering_{experiment_id}.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main(experiment_id: str, baseline_override: str | None, layer_filter: list[int] | None):
    results_dir = Path(f"results/experiments/{experiment_id}/post_task_stated")
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    sources, contexts, layers, coefficients = discover_conditions(results_dir)
    is_3x3 = len(contexts) > 0

    print(f"Discovered conditions:")
    print(f"  Sources: {sources}")
    print(f"  Contexts: {contexts if contexts else '(none - old format)'}")
    print(f"  Layers: {layers}")
    print(f"  Coefficients: {coefficients}")

    baseline_experiment = baseline_override
    if not baseline_experiment and is_3x3:
        config = load_experiment_config(experiment_id)
        if config:
            baseline_experiment = config.get("baseline_experiment")
    if baseline_experiment:
        baseline_dir = Path(f"results/experiments/{baseline_experiment}/post_task_stated")
        if baseline_dir.exists():
            print(f"  Baseline experiment: {baseline_experiment}")
        else:
            print(f"  Baseline experiment not found: {baseline_experiment}")
            baseline_experiment = None
    else:
        print("  Baseline experiment: (none)")

    if layer_filter:
        layers = [l for l in layers if l in layer_filter]
        print(f"  Filtered to layers: {layers}")

    if is_3x3:
        for layer in layers:
            path = plot_layer_3x3(
                results_dir, layer, coefficients, baseline_experiment, experiment_id
            )
            print(f"Saved: {path}")
    else:
        path = plot_old_format(results_dir, sources, layers, coefficients, experiment_id)
        print(f"Saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot concept vector steering results")
    parser.add_argument(
        "experiment_id", help="Experiment ID (e.g. concept_vector_steering_math_001)"
    )
    parser.add_argument(
        "--baseline",
        help="Override baseline experiment name for coef=0 comparison",
    )
    parser.add_argument(
        "--layer",
        type=int,
        nargs="+",
        help="Filter to specific layer(s)",
    )
    args = parser.parse_args()
    main(args.experiment_id, args.baseline, args.layer)
