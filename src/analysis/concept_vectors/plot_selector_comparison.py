"""Plot steering experiment results comparing different token selectors.

Creates a line plot showing:
- X-axis: steering coefficient
- Y-axis: mean stated preference score (1-5)
- Separate lines for each selector (last, mean, first)
- Error bars (95% CI)
- Annotations for parse failure rate

Usage:
    python -m src.analysis.concept_vectors.plot_selector_comparison <experiment_id>
    python -m src.analysis.concept_vectors.plot_selector_comparison <experiment_id> --layer 23
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


SELECTOR_COLORS = {
    "last": "#1976D2",
    "mean": "#388E3C",
    "first": "#E64A19",
}

SELECTOR_MARKERS = {
    "last": "o",
    "mean": "s",
    "first": "^",
}


@dataclass
class ConditionResult:
    selector: str
    layer: int
    coefficient: float
    scores: list[float]
    n_total: int
    n_parsed: int

    @property
    def mean_score(self) -> float:
        return np.mean(self.scores) if self.scores else np.nan

    @property
    def std_score(self) -> float:
        return np.std(self.scores, ddof=1) if len(self.scores) > 1 else 0.0

    @property
    def sem_score(self) -> float:
        return self.std_score / np.sqrt(len(self.scores)) if self.scores else np.nan

    @property
    def ci_95(self) -> float:
        return 1.96 * self.sem_score

    @property
    def parse_failure_rate(self) -> float:
        return 1 - (self.n_parsed / self.n_total) if self.n_total > 0 else 0.0


def discover_conditions(results_dir: Path) -> tuple[set[str], set[int], set[float]]:
    """Discover selectors, layers, and coefficients from condition directories.

    Expected pattern: selector_{sel}_layer{layer}_coef{coef}
    """
    pattern = re.compile(r"selector_(\w+)_layer(\d+)_coef(-?\d+\.?\d*)")

    selectors = set()
    layers = set()
    coefficients = set()

    for d in results_dir.iterdir():
        if not d.is_dir():
            continue
        if match := pattern.match(d.name):
            selectors.add(match.group(1))
            layers.add(int(match.group(2)))
            coefficients.add(float(match.group(3)))

    return selectors, layers, coefficients


def load_condition(
    results_dir: Path,
    selector: str,
    layer: int,
    coefficient: float,
) -> ConditionResult | None:
    """Load scores for a single condition."""
    condition_name = f"selector_{selector}_layer{layer}_coef{coefficient}"
    measurements_path = results_dir / condition_name / "measurements.yaml"

    if not measurements_path.exists():
        return None

    with open(measurements_path) as f:
        data = yaml.safe_load(f)

    scores = [item["score"] for item in data if item["score"] is not None]
    n_total = len(data)
    n_parsed = len(scores)

    return ConditionResult(
        selector=selector,
        layer=layer,
        coefficient=coefficient,
        scores=scores,
        n_total=n_total,
        n_parsed=n_parsed,
    )


def load_all_conditions(
    results_dir: Path,
    selectors: set[str],
    layers: set[int],
    coefficients: set[float],
) -> list[ConditionResult]:
    """Load all conditions from the experiment."""
    results = []
    for selector in selectors:
        for layer in layers:
            for coef in coefficients:
                if result := load_condition(results_dir, selector, layer, coef):
                    results.append(result)
    return results


def plot_selector_comparison(
    results: list[ConditionResult],
    layer: int,
    experiment_id: str,
    output_dir: Path,
) -> Path:
    """Create line plot comparing selectors at a given layer."""
    layer_results = [r for r in results if r.layer == layer]
    if not layer_results:
        raise ValueError(f"No results for layer {layer}")

    selectors = sorted(set(r.selector for r in layer_results))
    coefficients = sorted(set(r.coefficient for r in layer_results))

    fig, ax = plt.subplots(figsize=(10, 6))

    for selector in selectors:
        selector_results = sorted(
            [r for r in layer_results if r.selector == selector],
            key=lambda r: r.coefficient,
        )

        coefs = [r.coefficient for r in selector_results]
        means = [r.mean_score for r in selector_results]
        cis = [r.ci_95 for r in selector_results]

        color = SELECTOR_COLORS.get(selector, "#666666")
        marker = SELECTOR_MARKERS.get(selector, "o")

        ax.errorbar(
            coefs,
            means,
            yerr=cis,
            label=f"{selector} token",
            color=color,
            marker=marker,
            markersize=8,
            linewidth=2,
            capsize=4,
            capthick=1.5,
        )

        # Annotate parse failure rates if > 5%
        for r in selector_results:
            if r.parse_failure_rate > 0.05:
                ax.annotate(
                    f"{r.parse_failure_rate:.0%} fail",
                    (r.coefficient, r.mean_score),
                    textcoords="offset points",
                    xytext=(0, 12),
                    ha="center",
                    fontsize=8,
                    color="red",
                )

    ax.axhline(y=3, color="gray", linestyle="--", alpha=0.5, linewidth=1, label="neutral")
    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5, linewidth=1)

    ax.set_xlabel("Steering Coefficient", fontsize=12)
    ax.set_ylabel("Mean Stated Preference Score (1-5)", fontsize=12)
    ax.set_title(f"Steering Effect by Token Selector (Layer {layer})\n{experiment_id}", fontsize=13)

    ax.set_ylim(1, 5)
    ax.set_xticks(coefficients)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    date_str = datetime.now().strftime("%m%d%y")
    output_path = output_dir / f"plot_{date_str}_selector_comparison_layer{layer}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


def print_summary_table(results: list[ConditionResult], layer: int) -> None:
    """Print a summary table of results for a layer."""
    layer_results = [r for r in results if r.layer == layer]
    if not layer_results:
        return

    selectors = sorted(set(r.selector for r in layer_results))
    coefficients = sorted(set(r.coefficient for r in layer_results))

    print(f"\n{'='*60}")
    print(f"Layer {layer} Summary")
    print(f"{'='*60}")

    # Header
    header = f"{'Selector':<10} | " + " | ".join(f"coef={c:>5}" for c in coefficients)
    print(header)
    print("-" * len(header))

    for selector in selectors:
        row_results = {r.coefficient: r for r in layer_results if r.selector == selector}
        cells = []
        for coef in coefficients:
            if coef in row_results:
                r = row_results[coef]
                cells.append(f"{r.mean_score:>5.2f}")
            else:
                cells.append(f"{'N/A':>5}")
        print(f"{selector:<10} | " + " | ".join(cells))

    print()


def main(experiment_id: str, layer_filter: list[int] | None):
    results_dir = Path(f"results/experiments/{experiment_id}/post_task_stated")
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    selectors, layers, coefficients = discover_conditions(results_dir)

    if not selectors:
        print("No selector conditions found. Expected pattern: selector_{sel}_layer{layer}_coef{coef}")
        print("This script is for experiments that vary the token selector.")
        return

    print(f"Discovered conditions:")
    print(f"  Selectors: {sorted(selectors)}")
    print(f"  Layers: {sorted(layers)}")
    print(f"  Coefficients: {sorted(coefficients)}")

    results = load_all_conditions(results_dir, selectors, layers, coefficients)
    print(f"  Loaded {len(results)} conditions")

    if layer_filter:
        layers = {l for l in layers if l in layer_filter}
        print(f"  Filtered to layers: {sorted(layers)}")

    output_dir = Path("src/analysis/concept_vectors/plots")

    for layer in sorted(layers):
        print_summary_table(results, layer)
        path = plot_selector_comparison(results, layer, experiment_id, output_dir)
        print(f"Saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot selector comparison for steering experiment")
    parser.add_argument("experiment_id", help="Experiment ID")
    parser.add_argument("--layer", type=int, nargs="+", help="Filter to specific layer(s)")
    args = parser.parse_args()
    main(args.experiment_id, args.layer)
