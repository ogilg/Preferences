"""Shared helpers for probe analysis CLI."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from src.probes.core.storage import load_manifest
from src.probes.core.evaluate import compute_probe_similarity


PLOTS_DIR = Path("src/analysis/probe/plots")


def probe_label(probe: dict) -> str:
    method = probe["method"]
    abbrev = "ridge" if method == "ridge" else "bt"
    return f"{abbrev} L{probe['layer']}"


def filter_probes(
    manifest: dict,
    method: str | None = None,
    layer: int | None = None,
    probe_ids: list[str] | None = None,
) -> list[dict]:
    probes = manifest["probes"]
    if probe_ids is not None:
        probes = [p for p in probes if p["id"] in probe_ids]
    if method is not None:
        probes = [p for p in probes if p["method"] == method]
    if layer is not None:
        probes = [p for p in probes if p["layer"] == layer]
    return probes


def load_and_filter(
    manifest_dir: Path,
    method: str | None = None,
    layer: int | None = None,
    probe_ids: list[str] | None = None,
) -> tuple[dict, list[dict]]:
    manifest = load_manifest(manifest_dir)
    probes = filter_probes(manifest, method, layer, probe_ids)
    probes = sorted(probes, key=lambda p: (p["method"], p["layer"]))
    return manifest, probes


def add_filter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("manifest_dir", type=Path, help="Directory with manifest.json")
    parser.add_argument("--method", type=str, choices=["ridge", "bradley_terry"], help="Filter by method")
    parser.add_argument("--layer", type=int, help="Filter by layer")
    parser.add_argument("--probes", type=str, help="Comma-separated probe IDs")


def get_filters(args: argparse.Namespace) -> dict:
    probe_ids = args.probes.split(",") if args.probes else None
    return dict(
        manifest_dir=args.manifest_dir,
        method=args.method,
        layer=args.layer,
        probe_ids=probe_ids,
    )


def output_path(name: str, suffix: str = "") -> Path:
    date_str = datetime.now().strftime("%m%d%y")
    filename = f"plot_{date_str}_{name}"
    if suffix:
        filename += f"_{suffix}"
    return PLOTS_DIR / f"{filename}.png"


def format_probe_table(probes: list[dict]) -> str:
    if not probes:
        return "No probes to display."

    ridge = [p for p in probes if p["method"] == "ridge"]
    bt = [p for p in probes if p["method"] == "bradley_terry"]

    lines: list[str] = []

    if ridge:
        lines.append("Ridge Probes:")
        lines.append(f"  {'ID':<16} {'Layer':<6} {'R²':<8} {'±std':<8} {'MSE':<8} {'Alpha':<10} {'Gap':<8} {'Stab.':<8}")
        lines.append("  " + "-" * 72)
        for p in ridge:
            mse = f"{p['cv_mse_mean']:>7.4f}" if "cv_mse_mean" in p else "      —"
            gap = f"{p['train_test_gap']:>7.4f}" if "train_test_gap" in p else "      —"
            stab = f"{p['cv_stability']:>7.4f}" if "cv_stability" in p else "      —"
            lines.append(
                f"  {p['id']:<16} {p['layer']:<6} "
                f"{p['cv_r2_mean']:>7.4f} {p['cv_r2_std']:>7.4f} "
                f"{mse} "
                f"{p['best_alpha']:>9.0f} "
                f"{gap} "
                f"{stab}"
            )

    if bt:
        if ridge:
            lines.append("")
        lines.append("Bradley-Terry Probes:")
        lines.append(f"  {'ID':<16} {'Layer':<6} {'Acc.':<8} {'Loss':<8} {'Epochs':<8} {'Pairs':<8}")
        lines.append("  " + "-" * 54)
        for p in bt:
            lines.append(
                f"  {p['id']:<16} {p['layer']:<6} "
                f"{p['train_accuracy']:>7.4f} {p['train_loss']:>7.4f} "
                f"{p['n_epochs']:>7} {p['n_pairs']:>7}"
            )

    return "\n".join(lines)


def get_probe_similarity(manifest_dir: Path, probes: list[dict]) -> np.ndarray:
    probe_ids = [p["id"] for p in probes]
    return compute_probe_similarity(manifest_dir, probe_ids)


@dataclass
class FacetConfig:
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
    if not facets:
        print("No facets to plot")
        return

    nrows = (len(facets) + ncols - 1) // ncols
    figheight = row_height * nrows + 0.5

    fig, axes = plt.subplots(nrows, ncols, figsize=(figwidth, figheight), squeeze=False)
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

    for ax in ax_flat[len(facets):]:
        ax.set_visible(False)

    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {output}")
