"""Plot metadata confound analysis: R² decomposition and OLS coefficients.

Fits three models (topic-only, dataset-only, both) to show how much
of mu variance is explained by metadata. Plots coefficients for the
topic-only model (used for residualization) and the combined model.

Usage:
    python -m src.analysis.probe.plot_metadata_coefficients \
        --config configs/probes/gemma3_completion_preference.yaml
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.analysis.active_learning.plot_mu_by_topic import TOPIC_COLORS
from src.probes.data_loading import load_thurstonian_scores
from src.probes.experiments.run_dir_probes import RunDirProbeConfig
from src.probes.residualization import fit_metadata_models

PLOTS_DIR = Path(__file__).parent / "plots"

DATASET_COLORS = {
    "alpaca": "#6B8FBF",
    "bailbench": "#C44E52",
    "competition_math": "#4C72B0",
    "stresstest": "#E07B54",
    "wildchat": "#5BA37E",
}


def _get_color(feature_name: str) -> str:
    for key, color in TOPIC_COLORS.items():
        if f"topic_{key}" == feature_name:
            return color
    for key, color in DATASET_COLORS.items():
        if f"dataset_{key}" == feature_name:
            return color
    return "#999999"


def _clean_name(name: str) -> str:
    for prefix in ("dataset_", "topic_"):
        if name.startswith(prefix):
            return name[len(prefix):].replace("_", " ").title()
    return name.replace("_", " ").title()


def _plot_coef_bar(ax: plt.Axes, coefs: np.ndarray, names: list[str], title: str) -> None:
    order = np.argsort(coefs)
    sorted_coefs = coefs[order]
    sorted_names = [names[i] for i in order]
    display_names = [_clean_name(n) for n in sorted_names]
    colors = [_get_color(n) for n in sorted_names]

    bars = ax.barh(range(len(sorted_coefs)), sorted_coefs, color=colors, edgecolor="black", alpha=0.85)
    ax.set_yticks(range(len(sorted_coefs)))
    ax.set_yticklabels(display_names, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("OLS Coefficient (effect on mu)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)

    for bar, coef in zip(bars, sorted_coefs):
        x_pos = coef + 0.05 if coef >= 0 else coef - 0.05
        ax.text(
            x_pos, bar.get_y() + bar.get_height() / 2,
            f"{coef:+.2f}", va="center",
            ha="left" if coef >= 0 else "right", fontsize=7,
        )


def plot_metadata_analysis(models: dict, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                             gridspec_kw={"width_ratios": [1, 1.5, 1.5]})

    # Panel 1: R² decomposition bar chart
    ax_r2 = axes[0]
    labels = ["Topic\n+ length", "Dataset\n+ length", "Both"]
    r2s = [models["topic_r2"], models["dataset_r2"], models["both_r2"]]
    bar_colors = ["#5BA37E", "#6B8FBF", "#8B6BB0"]
    bars = ax_r2.bar(labels, r2s, color=bar_colors, edgecolor="black", alpha=0.85, width=0.6)
    for bar, r2 in zip(bars, r2s):
        ax_r2.text(bar.get_x() + bar.get_width() / 2, r2 + 0.01,
                   f"{r2:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax_r2.set_ylabel("R²")
    ax_r2.set_title("Metadata R² Decomposition")
    ax_r2.set_ylim(0, 1.0)
    ax_r2.grid(axis="y", alpha=0.3)

    # Panel 2: Topic-only coefficients
    topic_coefs = np.array(models["topic_coefs"])
    topic_features = models["topic_features"]
    _plot_coef_bar(axes[1], topic_coefs, topic_features,
                   f"Topic-Only Model (R²={models['topic_r2']:.3f})\n"
                   f"ref: {models['topic_ref']}")

    # Panel 3: Combined model coefficients
    both_coefs = np.array(models["both_coefs"])
    both_features = models["both_features"]
    _plot_coef_bar(axes[2], both_coefs, both_features,
                   f"Combined Model (R²={models['both_r2']:.3f})\n"
                   f"ref: dataset={models['both_ref_dataset']}, topic={models['both_ref_topic']}")

    fig.suptitle(f"Metadata Confound Analysis (n={models['n_tasks']})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot metadata regression coefficients")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config = RunDirProbeConfig.from_yaml(args.config)
    assert config.topics_json is not None, "topics_json required in config"

    scores = load_thurstonian_scores(config.run_dir)
    models = fit_metadata_models(scores, config.topics_json)

    print(f"Topic-only R²:  {models['topic_r2']:.4f}")
    print(f"Dataset-only R²: {models['dataset_r2']:.4f}")
    print(f"Both R²:         {models['both_r2']:.4f}")

    date_str = datetime.now().strftime("%m%d%y")
    output_path = PLOTS_DIR / f"plot_{date_str}_metadata_coefficients.png"
    plot_metadata_analysis(models, output_path)


if __name__ == "__main__":
    main()
