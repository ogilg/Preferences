"""Plot correlation matrix between qualitative rating templates."""

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.experiments.correlation import safe_correlation
from src.preferences.templates.template import PromptTemplate, load_templates_from_yaml


def load_results(results_path: Path) -> dict:
    with open(results_path) as f:
        return json.load(f)


def make_short_label(template: PromptTemplate) -> str:
    """Create readable label from template tags."""
    tags = template.tags_dict
    phrasing = tags.get("phrasing", "?")
    scale = tags.get("scale", "?")
    response_format = tags.get("response_format", "?")

    scale_short = {"binary": "bin", "ternary": "ter"}.get(scale, scale[:3])
    fmt_short = {"regex": "re", "xml": "xml", "tool_use": "tool"}.get(response_format, response_format[:3])

    return f"p{phrasing}_{scale_short}_{fmt_short}"


def plot_correlation_matrix(
    results_dir: Path,
    templates_path: Path,
    method: str = "spearman",
) -> None:
    results_path = results_dir / "rating_results_full.json"
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return

    results = load_results(results_path)
    all_scores = results["results"]

    # Load templates for labels
    templates = {t.name: t for t in load_templates_from_yaml(templates_path)}

    # Filter to only qualitative templates that are in results
    template_names = [name for name in results["templates"] if name in templates]
    if not template_names:
        print("No matching templates found")
        return

    # Sort by phrasing, scale, then response_format
    def sort_key(name: str) -> tuple[str, str, str]:
        tags = templates[name].tags_dict
        return (
            tags.get("phrasing", "z"),
            tags.get("scale", "z"),
            tags.get("response_format", "z"),
        )

    template_names = sorted(template_names, key=sort_key)

    # Build DataFrame: rows=task_ids, columns=templates
    # Collect all task_ids across templates
    all_task_ids = set()
    for name in template_names:
        all_task_ids.update(all_scores[name].keys())

    # Create score matrix
    df = pd.DataFrame(index=sorted(all_task_ids), columns=template_names)
    for name in template_names:
        for task_id, score in all_scores[name].items():
            df.loc[task_id, name] = score

    # Compute correlation matrix
    n = len(template_names)
    corr_matrix = np.zeros((n, n))

    for i, name_i in enumerate(template_names):
        for j, name_j in enumerate(template_names):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                # Get overlapping task_ids
                scores_i = all_scores[name_i]
                scores_j = all_scores[name_j]
                common = set(scores_i.keys()) & set(scores_j.keys())
                if len(common) >= 10:
                    vals_i = np.array([scores_i[tid] for tid in common])
                    vals_j = np.array([scores_j[tid] for tid in common])
                    corr_matrix[i, j] = safe_correlation(vals_i, vals_j, method=method)
                else:
                    corr_matrix[i, j] = np.nan

    # Create readable labels
    labels = [make_short_label(templates[name]) for name in template_names]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.isnan(corr_matrix)
    sns.heatmap(
        corr_matrix,
        xticklabels=labels,
        yticklabels=labels,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        vmin=-1,
        vmax=1,
        mask=mask,
        ax=ax,
    )
    ax.set_title(f"Qualitative Rating Correlations ({method.title()})\nn={results['n_completions']} completions")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    date_str = datetime.now().strftime("%m%d%y")
    output_path = results_dir / f"plot_{date_str}_correlation_matrix.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()

    # Print summary stats
    triu_vals = corr_matrix[np.triu_indices(n, k=1)]
    triu_vals = triu_vals[~np.isnan(triu_vals)]
    print(f"\nCorrelation summary ({len(triu_vals)} pairs):")
    print(f"  Mean: {np.mean(triu_vals):.3f}")
    print(f"  Min:  {np.min(triu_vals):.3f}")
    print(f"  Max:  {np.max(triu_vals):.3f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot rating correlation matrix")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/probe_data/ratings"),
    )
    parser.add_argument(
        "--templates",
        type=Path,
        default=Path("src/preferences/templates/data/post_task_qualitative_v1.yaml"),
    )
    parser.add_argument(
        "--method",
        choices=["pearson", "spearman"],
        default="spearman",
    )
    args = parser.parse_args()

    plot_correlation_matrix(args.results_dir, args.templates, args.method)
