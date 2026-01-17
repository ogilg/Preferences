"""Plot probe training results."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Template ID -> (scale, format, phrasing)
TEMPLATE_INFO = {
    "001": ("bin", "regex", "p1"),
    "002": ("bin", "xml", "p1"),
    "003": ("bin", "tool", "p1"),
    "004": ("ter", "regex", "p1"),
    "005": ("ter", "xml", "p1"),
    "006": ("ter", "tool", "p1"),
    "007": ("bin", "regex", "p2"),
    "008": ("bin", "xml", "p2"),
    "009": ("bin", "tool", "p2"),
    "010": ("ter", "regex", "p2"),
    "011": ("ter", "xml", "p2"),
    "012": ("ter", "tool", "p2"),
}


def short_name(template: str) -> str:
    """Convert template name to informative short name."""
    # Extract ID from names like "post_task_qualitative_001"
    template_id = template.split("_")[-1]
    if template_id in TEMPLATE_INFO:
        scale, fmt, phrasing = TEMPLATE_INFO[template_id]
        return f"{scale}_{fmt}_{phrasing}"
    return template[:20]


def load_results(results_path: Path) -> dict:
    with open(results_path) as f:
        return json.load(f)


def plot_r2_by_layer(
    results: dict,
    output_path: Path,
) -> None:
    """Bar chart of R² by layer for each template."""
    templates = list(results.keys())
    if not templates:
        return

    layers = [r["layer"] for r in results[templates[0]]]
    n_layers = len(layers)
    n_templates = len(templates)

    fig, ax = plt.subplots(figsize=(max(10, n_templates * 0.8), 6))

    x = np.arange(n_layers)
    width = 0.8 / n_templates

    for i, template in enumerate(templates):
        layer_results = results[template]
        r2s = [r["cv_r2_mean"] for r in layer_results]
        stds = [r["cv_r2_std"] for r in layer_results]
        offset = (i - n_templates / 2 + 0.5) * width
        ax.bar(x + offset, r2s, width, yerr=stds, label=short_name(template), alpha=0.8, capsize=2)

    ax.set_xlabel("Layer")
    ax.set_ylabel("CV R²")
    ax.set_title("Probe R² by Template and Layer")
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_best_layer_comparison(
    results: dict,
    output_path: Path,
) -> None:
    """Bar chart comparing best R² across templates."""
    templates = []
    best_r2s = []
    best_layers = []

    for template, layer_results in results.items():
        best = max(layer_results, key=lambda r: r["cv_r2_mean"])
        templates.append(template)
        best_r2s.append(best["cv_r2_mean"])
        best_layers.append(best["layer"])

    # Sort by R²
    sorted_idx = np.argsort(best_r2s)[::-1]
    templates = [templates[i] for i in sorted_idx]
    best_r2s = [best_r2s[i] for i in sorted_idx]
    best_layers = [best_layers[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(10, max(6, len(templates) * 0.3)))

    y = np.arange(len(templates))
    bars = ax.barh(y, best_r2s, color="steelblue", alpha=0.8)

    # Add layer labels
    for i, (r2, layer) in enumerate(zip(best_r2s, best_layers)):
        ax.text(r2 + 0.01, i, f"L{layer}", va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels([short_name(t) for t in templates], fontsize=8)
    ax.set_xlabel("Best CV R²")
    ax.set_title("Best Probe R² per Template")
    ax.grid(True, alpha=0.3, axis="x")
    ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot probe results")
    parser.add_argument("results", type=Path, help="probe_results.json file")
    parser.add_argument("--output-dir", type=Path, help="Output directory for plots")
    args = parser.parse_args()

    results = load_results(args.results)
    print(f"Loaded results for {len(results)} templates")

    output_dir = args.output_dir or args.results.parent
    date_str = datetime.now().strftime("%m%d%y")

    plot_r2_by_layer(results, output_dir / f"plot_{date_str}_probe_r2_by_layer.png")
    plot_best_layer_comparison(results, output_dir / f"plot_{date_str}_probe_best_r2.png")


if __name__ == "__main__":
    main()
