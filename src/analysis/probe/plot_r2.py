"""Plot R² comparison bar chart with optional grouping."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.analysis.probe.helpers import (
    FacetConfig,
    format_probe_table,
    load_and_filter,
    output_path,
    plot_faceted,
    probe_label,
)


def _simple_bar(probes: list[dict], output: Path) -> None:
    labels = [probe_label(p) for p in probes]
    r2_means = [p["cv_r2_mean"] for p in probes]
    r2_stds = [p["cv_r2_std"] for p in probes]
    has_train = all("train_r2" in p for p in probes)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(labels))

    if has_train:
        train_r2s = [p["train_r2"] for p in probes]
        width = 0.35
        bars_train = ax.bar(x - width / 2, train_r2s, width, alpha=0.7, color="tab:blue", edgecolor="black", label="Train R²")
        bars_val = ax.bar(x + width / 2, r2_means, width, yerr=r2_stds, capsize=5, alpha=0.7, color="tab:orange", edgecolor="black", label="Val R² (CV)")
        for bar, val in zip(bars_train, train_r2s):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        for bar, mean, std in zip(bars_val, r2_means, r2_stds):
            ax.text(bar.get_x() + bar.get_width() / 2, mean + std + 0.01, f"{mean:.3f}", ha="center", va="bottom", fontsize=8)
        ax.legend()
    else:
        bars = ax.bar(x, r2_means, yerr=r2_stds, capsize=5, alpha=0.7, color="steelblue", edgecolor="black")
        for bar, mean, std in zip(bars, r2_means, r2_stds):
            ax.text(bar.get_x() + bar.get_width() / 2, mean + std + 0.01, f"{mean:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Probe", fontsize=11)
    ax.set_ylabel("R²", fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title(f"Probe Performance Comparison ({len(probes)} probes)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {output}")


def _grouped_facets(probes: list[dict], group_by: str, output: Path) -> None:
    groups: dict[str, list[dict]] = {}
    for p in probes:
        key = str(p[group_by])
        groups.setdefault(key, []).append(p)

    # Label function uses the fields that are NOT the group-by field
    other_fields = [f for f in ("method", "layer") if f != group_by]

    def label_fn(p: dict) -> str:
        parts = []
        for f in other_fields:
            if f == "method":
                parts.append("ridge" if p["method"] == "ridge" else "bt")
            elif f == "layer":
                parts.append(f"L{p['layer']}")
            else:
                parts.append(str(p[f]))
        return " ".join(parts)

    facets = []
    for key in sorted(groups, key=lambda k: (not k.isdigit(), int(k) if k.isdigit() else k)):
        facets.append(FacetConfig(
            title=f"{group_by.capitalize()} {key}",
            probes=groups[key],
            label_fn=label_fn,
        ))

    plot_faceted(facets, output)


def run(
    manifest_dir: Path,
    method: str | None = None,
    layer: int | None = None,
    probe_ids: list[str] | None = None,
    group_by: str | None = None,
    output: Path | None = None,
    no_plot: bool = False,
) -> None:
    _, probes = load_and_filter(manifest_dir, method, layer, probe_ids)
    if not probes:
        print("No probes match filters")
        return

    # Text summary (all probes)
    print(format_probe_table(probes))

    # Only ridge probes have R² for plotting
    ridge_probes = [p for p in probes if p["method"] == "ridge"]
    if not ridge_probes:
        print("\nNo ridge probes for R² plot.")
        return
    if no_plot:
        return

    suffix = group_by if group_by else "all"
    out = output or output_path("r2", suffix)

    if group_by:
        _grouped_facets(ridge_probes, group_by, out)
    else:
        _simple_bar(ridge_probes, out)
