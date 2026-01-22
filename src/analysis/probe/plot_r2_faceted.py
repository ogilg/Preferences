"""Plot R² comparison with faceted subplots (one per layer/template/dataset)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.probes.storage import load_manifest
from src.analysis.probe.probe_helpers import filter_probes, make_probe_label, get_template


def shorten_template_name(template: str, templates: list[str] | None = None) -> str:
    """Shorten template name for display. Adaptively show only distinguishing parts."""
    # If we have a list of templates to compare against, only show what's necessary
    if templates and len(templates) > 1:
        # Check if all templates share the same prefix
        # e.g., all post_task_ or all pre_task_
        prefixes = set(tuple(t.split("_")[0:2]) for t in templates)  # e.g., ('post', 'task')

        # If all have same prefix, skip it
        if len(prefixes) == 1:
            name = "_".join(template.split("_")[2:])
        else:
            # Include task type (pre/post) to distinguish
            parts = template.split("_")
            name = f"{parts[1][0]}_" + "_".join(parts[2:])  # e.g., p_qualitative_001
    else:
        name = template.replace("post_task_", "").replace("pre_task_", "")

    # Convert qualitative_001 -> qual_001, stated_001 -> stat_001
    name = name.replace("qualitative_", "qual_")
    name = name.replace("stated_", "stat_")
    return name


def plot_by_layer(manifest: dict, output: Path) -> None:
    """Create three subplots (one per layer) showing R² by template."""
    probes = manifest["probes"]

    if not probes:
        print("No probes match the filters")
        return

    layers = sorted(set(p["layer"] for p in probes))

    # Calculate global max for consistent scale
    all_r2_means = [p["cv_r2_mean"] + p["cv_r2_std"] for p in probes]
    global_max = max(all_r2_means) * 1.3

    figheight = 2.5 * len(layers)
    fig, axes = plt.subplots(len(layers), 1, figsize=(12, figheight))
    if len(layers) == 1:
        axes = [axes]

    # Get all unique templates for context
    all_templates = sorted(set(get_template(p) for p in probes))
    # Fixed y-axis limit for all subplots
    y_limit = 0.5

    for ax, layer in zip(axes, layers):
        layer_probes = sorted(
            [p for p in probes if p["layer"] == layer],
            key=lambda p: get_template(p)
        )

        labels = [shorten_template_name(get_template(p), all_templates) for p in layer_probes]
        r2_means = [p["cv_r2_mean"] for p in layer_probes]
        r2_stds = [p["cv_r2_std"] for p in layer_probes]

        x = np.arange(len(labels))
        bars = ax.bar(x, r2_means, yerr=r2_stds, capsize=3, alpha=0.7, color="steelblue", edgecolor="black")

        ax.set_ylabel("R²", fontsize=10)
        ax.set_title(f"Layer {layer}", fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, y_limit)

    plt.subplots_adjust(hspace=0.35)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved faceted by layer to {output}")


def plot_by_template(manifest: dict, output: Path) -> None:
    """Create subplots (one per template) showing R² by layer."""
    probes = manifest["probes"]

    if not probes:
        print("No probes match the filters")
        return

    templates = sorted(set(get_template(p) for p in probes))

    # Detect if there's a mix of task types
    task_types = set(t.split("_")[1] for t in templates)  # 'pre' or 'post'
    has_mixed_tasks = len(task_types) > 1

    # Calculate global max for consistent scale
    all_r2_means = [p["cv_r2_mean"] + p["cv_r2_std"] for p in probes]
    global_max = max(all_r2_means) * 1.3

    # Grid layout: 3 columns, dynamic rows
    ncols = 3
    nrows = (len(templates) + 2) // 3
    figwidth = 14
    figheight = 2 * nrows + 0.5

    fig, axes = plt.subplots(nrows, ncols, figsize=(figwidth, figheight))

    # Fixed y-axis limit for all subplots
    y_limit = 0.5

    # Flatten axes into 1D list
    if nrows == 1 and ncols == 1:
        ax_flat = [axes]
    elif nrows == 1 or ncols == 1:
        ax_flat = list(axes.flat) if hasattr(axes, 'flat') else list(axes)
    else:
        ax_flat = list(axes.flat)

    for ax, template in zip(ax_flat, templates):
        template_probes = sorted(
            [p for p in probes if get_template(p) == template],
            key=lambda p: p["layer"]
        )

        labels = ["L" + str(p["layer"]) for p in template_probes]
        r2_means = [p["cv_r2_mean"] for p in template_probes]
        r2_stds = [p["cv_r2_std"] for p in template_probes]

        x = np.arange(len(labels))
        bars = ax.bar(x, r2_means, yerr=r2_stds, capsize=3, alpha=0.7, color="steelblue", edgecolor="black")

        ax.set_ylabel("R²", fontsize=8)
        short_template = shorten_template_name(template, templates)
        ax.set_title(f"{short_template}", fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, fontsize=7)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, y_limit)

    # Hide unused subplots
    for ax in ax_flat[len(templates):]:
        ax.set_visible(False)

    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved faceted by template to {output}")


def plot_by_task_type(manifest: dict, output: Path) -> None:
    """Create subplots (one per task type) showing R² by template."""
    probes = manifest["probes"]

    if not probes:
        print("No probes match the filters")
        return

    # Get all task types present
    task_types = sorted(set("pre" if "pre_task" in get_template(p) else "post" for p in probes))

    nrows = len(task_types)
    figheight = 2.5 * nrows + 1

    fig, axes = plt.subplots(nrows, 1, figsize=(12, figheight))
    if nrows == 1:
        axes = [axes]

    for ax, task_type in zip(axes, task_types):
        search_str = "pre_task" if task_type == "pre" else "post_task"
        type_probes = sorted(
            [p for p in probes if search_str in get_template(p)],
            key=lambda p: get_template(p)
        )

        templates = sorted(set(get_template(p) for p in type_probes))
        labels = [shorten_template_name(get_template(p), templates) for p in type_probes]
        r2_means = [p["cv_r2_mean"] for p in type_probes]
        r2_stds = [p["cv_r2_std"] for p in type_probes]

        x = np.arange(len(labels))
        bars = ax.bar(x, r2_means, yerr=r2_stds, capsize=3, alpha=0.7, color="steelblue", edgecolor="black")

        ax.set_ylabel("R²", fontsize=10)
        ax.set_title(f"Task Type: {task_type.upper()}-task", fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 0.5)

    plt.subplots_adjust(hspace=0.35)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved faceted by task type to {output}")


def plot_by_response_format(manifest: dict, output: Path) -> None:
    """Create subplots (one per response format) showing R² by template."""
    probes = manifest["probes"]

    if not probes:
        print("No probes match the filters")
        return

    # Get all response formats present
    response_formats = sorted(set(fmt for p in probes for fmt in p["response_formats"]))

    if len(response_formats) <= 1:
        print("Only one response format found, nothing to compare")
        return

    nrows = len(response_formats)
    figheight = 3 * nrows

    fig, axes = plt.subplots(nrows, 1, figsize=(12, figheight))
    if nrows == 1:
        axes = [axes]

    # Fixed y-axis limit for all subplots
    y_limit = 0.5

    for ax, fmt in zip(axes, response_formats):
        # Get probes that include this format
        fmt_probes = sorted(
            [p for p in probes if fmt in p["response_formats"]],
            key=lambda p: get_template(p)
        )

        templates = sorted(set(get_template(p) for p in fmt_probes))
        labels = [shorten_template_name(get_template(p), templates) for p in fmt_probes]
        r2_means = [p["cv_r2_mean"] for p in fmt_probes]
        r2_stds = [p["cv_r2_std"] for p in fmt_probes]

        x = np.arange(len(labels))
        bars = ax.bar(x, r2_means, yerr=r2_stds, capsize=3, alpha=0.7, color="steelblue", edgecolor="black")

        ax.set_ylabel("R²", fontsize=10)
        ax.set_title(f"Response Format: {fmt.upper()}", fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, y_limit)

    plt.subplots_adjust(hspace=0.35)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved faceted by response format to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot R² with faceted subplots")
    parser.add_argument("manifest_dir", type=Path, help="Directory with manifest.json")
    parser.add_argument("--by", choices=["layer", "template", "task_type", "response_format"], default="layer",
                       help="Facet by layer, template, task_type, or response_format")
    parser.add_argument("--output", type=Path, help="Output PNG path")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest_dir)
    filtered_manifest = manifest

    # Default output path
    if args.output is None:
        from datetime import datetime
        date_str = datetime.now().strftime("%m%d%y")
        args.output = Path(f"src/analysis/probe/plots/plot_{date_str}_r2_by_{args.by}.png")

    if args.by == "layer":
        plot_by_layer(filtered_manifest, args.output)
    elif args.by == "template":
        plot_by_template(filtered_manifest, args.output)
    elif args.by == "task_type":
        plot_by_task_type(filtered_manifest, args.output)
    else:
        plot_by_response_format(filtered_manifest, args.output)


if __name__ == "__main__":
    main()
