"""Plot R² comparison with faceted subplots (one per layer/template/dataset)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from src.probes.storage import load_manifest
from src.analysis.probe.probe_helpers import get_template


def shorten_template_name(template: str, templates: list[str] | None = None) -> str:
    """Shorten template name for display. Adaptively show only distinguishing parts."""
    if templates and len(templates) > 1:
        prefixes = set(tuple(t.split("_")[0:2]) for t in templates)
        if len(prefixes) == 1:
            name = "_".join(template.split("_")[2:])
        else:
            parts = template.split("_")
            name = f"{parts[1][0]}_" + "_".join(parts[2:])
    else:
        name = template.replace("post_task_", "").replace("pre_task_", "")

    name = name.replace("qualitative_", "qual_")
    name = name.replace("stated_", "stat_")
    return name


@dataclass
class FacetConfig:
    """Configuration for a single facet in the plot."""
    title: str
    probes: list[dict]
    label_fn: Callable[[dict], str]


def plot_faceted(
    facets: list[FacetConfig],
    output: Path,
    ncols: int = 1,
    y_limit: float = 0.5,
    figwidth: float = 12,
    row_height: float = 2.5,
) -> None:
    """Generic faceted bar plot for R² values."""
    if not facets:
        print("No facets to plot")
        return

    nrows = (len(facets) + ncols - 1) // ncols
    figheight = row_height * nrows + 0.5

    fig, axes = plt.subplots(nrows, ncols, figsize=(figwidth, figheight))

    # Normalize axes to flat list
    if nrows == 1 and ncols == 1:
        ax_flat = [axes]
    elif nrows == 1 or ncols == 1:
        ax_flat = list(axes.flat) if hasattr(axes, 'flat') else list(axes)
    else:
        ax_flat = list(axes.flat)

    for ax, facet in zip(ax_flat, facets):
        if not facet.probes:
            ax.set_visible(False)
            continue

        labels = [facet.label_fn(p) for p in facet.probes]
        r2_means = [p["cv_r2_mean"] for p in facet.probes]
        r2_stds = [p["cv_r2_std"] for p in facet.probes]

        x = np.arange(len(labels))
        ax.bar(x, r2_means, yerr=r2_stds, capsize=3, alpha=0.7, color="steelblue", edgecolor="black")

        ax.set_ylabel("R²", fontsize=10)
        ax.set_title(facet.title, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, y_limit)

    # Hide unused subplots
    for ax in ax_flat[len(facets):]:
        ax.set_visible(False)

    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {output}")


def plot_by_layer(manifest: dict, output: Path) -> None:
    """Create subplots (one per layer) showing R² by template."""
    probes = manifest["probes"]
    if not probes:
        print("No probes found")
        return

    layers = sorted(set(p["layer"] for p in probes))
    all_templates = sorted(set(get_template(p) for p in probes))

    facets = []
    for layer in layers:
        layer_probes = sorted(
            [p for p in probes if p["layer"] == layer],
            key=lambda p: get_template(p)
        )
        facets.append(FacetConfig(
            title=f"Layer {layer}",
            probes=layer_probes,
            label_fn=lambda p, t=all_templates: shorten_template_name(get_template(p), t),
        ))

    plot_faceted(facets, output)


def plot_by_template(manifest: dict, output: Path) -> None:
    """Create subplots (one per template) showing R² by layer."""
    probes = manifest["probes"]
    if not probes:
        print("No probes found")
        return

    templates = sorted(set(get_template(p) for p in probes))

    facets = []
    for template in templates:
        template_probes = sorted(
            [p for p in probes if get_template(p) == template],
            key=lambda p: p["layer"]
        )
        facets.append(FacetConfig(
            title=shorten_template_name(template, templates),
            probes=template_probes,
            label_fn=lambda p: f"L{p['layer']}",
        ))

    plot_faceted(facets, output, ncols=3, figwidth=14, row_height=2)


def plot_by_task_type(manifest: dict, output: Path) -> None:
    """Create subplots (one per task type) showing R² by template."""
    probes = manifest["probes"]
    if not probes:
        print("No probes found")
        return

    task_types = sorted(set("pre" if "pre_task" in get_template(p) else "post" for p in probes))

    facets = []
    for task_type in task_types:
        search_str = "pre_task" if task_type == "pre" else "post_task"
        type_probes = sorted(
            [p for p in probes if search_str in get_template(p)],
            key=lambda p: get_template(p)
        )
        templates = sorted(set(get_template(p) for p in type_probes))
        facets.append(FacetConfig(
            title=f"Task Type: {task_type.upper()}-task",
            probes=type_probes,
            label_fn=lambda p, t=templates: shorten_template_name(get_template(p), t),
        ))

    plot_faceted(facets, output)


def plot_by_response_format(manifest: dict, output: Path) -> None:
    """Create subplots (one per response format) showing R² by template."""
    probes = manifest["probes"]
    if not probes:
        print("No probes found")
        return

    response_formats = sorted(set(fmt for p in probes for fmt in p.get("response_formats", [])))
    if len(response_formats) <= 1:
        print("Only one response format found, nothing to compare")
        return

    facets = []
    for fmt in response_formats:
        fmt_probes = sorted(
            [p for p in probes if fmt in p.get("response_formats", [])],
            key=lambda p: get_template(p)
        )
        templates = sorted(set(get_template(p) for p in fmt_probes))
        facets.append(FacetConfig(
            title=f"Response Format: {fmt.upper()}",
            probes=fmt_probes,
            label_fn=lambda p, t=templates: shorten_template_name(get_template(p), t),
        ))

    plot_faceted(facets, output, row_height=3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot R² with faceted subplots")
    parser.add_argument("manifest_dir", type=Path, help="Directory with manifest.json")
    parser.add_argument("--by", choices=["layer", "template", "task_type", "response_format"], default="layer",
                       help="Facet by layer, template, task_type, or response_format")
    parser.add_argument("--output", type=Path, help="Output PNG path")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest_dir)

    if args.output is None:
        from datetime import datetime
        date_str = datetime.now().strftime("%m%d%y")
        args.output = Path(f"src/analysis/probe/plots/plot_{date_str}_r2_by_{args.by}.png")

    plot_fns = {
        "layer": plot_by_layer,
        "template": plot_by_template,
        "task_type": plot_by_task_type,
        "response_format": plot_by_response_format,
    }
    plot_fns[args.by](manifest, args.output)


if __name__ == "__main__":
    main()
