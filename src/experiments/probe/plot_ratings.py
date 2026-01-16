"""Plot rating collection results with template tag groupings."""

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.preferences.templates.template import PromptTemplate, load_templates_from_yaml

# Phrasing labels based on template text
QUALITATIVE_PHRASINGS = {"1": "experience", "2": "how_was", "3": "honest"}
STATED_PHRASINGS = {"1": "enjoyed", "2": "honest", "3": "rate_exp"}
FORMAT_LABELS = {"regex": "regex", "xml": "xml", "tool_use": "tool"}


def make_short_name(
    template_name: str,
    phrasing: str,
    response_format: str,
    scale: str,
    include_type: bool = False,
) -> str:
    """Generate readable short name for a template."""
    is_qualitative = "qualitative" in template_name
    phrasing_map = QUALITATIVE_PHRASINGS if is_qualitative else STATED_PHRASINGS
    phrasing_label = phrasing_map.get(phrasing, f"p{phrasing}")
    format_label = FORMAT_LABELS.get(response_format, response_format[:3])

    if include_type:
        type_label = "qual" if is_qualitative else "num"
        scale_label = scale.replace("-", "→")
        return f"{type_label}:{scale_label}:{phrasing_label}"
    return f"{phrasing_label}/{format_label}"


def load_results(results_path: Path) -> dict:
    with open(results_path) as f:
        return json.load(f)


def compute_distribution(
    scores: list[float],
    scale: str,
) -> tuple[float, float, float]:
    """Return (pct_negative, pct_neutral, pct_positive).

    For qualitative (binary/ternary): scores are -1, 0, 1
    For numeric scales like "1-2", "1-3": normalize based on range
    """
    if not scores:
        return (0, 0, 0)
    total = len(scores)

    # Parse numeric scale ranges like "1-2", "1-3"
    if "-" in scale:
        parts = scale.split("-")
        scale_min, scale_max = int(parts[0]), int(parts[1])
        midpoint = (scale_min + scale_max) / 2
        n_neg = sum(1 for s in scores if s < midpoint)
        n_neu = sum(1 for s in scores if s == midpoint)
        n_pos = sum(1 for s in scores if s > midpoint)
    else:
        # Qualitative: scores are already -1, 0, 1
        n_neg = sum(1 for s in scores if s < 0)
        n_neu = sum(1 for s in scores if s == 0)
        n_pos = sum(1 for s in scores if s > 0)

    return (n_neg / total * 100, n_neu / total * 100, n_pos / total * 100)


def plot_grouped_stacked(
    template_groups: dict[str, list[tuple[str, tuple[float, float, float]]]],
    title: str,
    output_path: Path,
) -> None:
    """Stacked bar chart with groups separated by vertical lines."""
    # Flatten for plotting
    labels = []
    neg_vals = []
    neu_vals = []
    pos_vals = []
    group_boundaries = []

    for group_name, templates in template_groups.items():
        group_boundaries.append((len(labels), group_name))
        for short_name, (neg, neu, pos) in templates:
            labels.append(short_name)
            neg_vals.append(neg)
            neu_vals.append(neu)
            pos_vals.append(pos)

    if not labels:
        print(f"No data to plot for {title}")
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(labels))
    width = 0.7

    ax.bar(x, neg_vals, width, label="Negative", color="#d62728")
    ax.bar(x, neu_vals, width, bottom=neg_vals, label="Neutral", color="#7f7f7f")
    ax.bar(x, pos_vals, width, bottom=np.array(neg_vals) + np.array(neu_vals), label="Positive", color="#2ca02c")

    # Group labels and separators
    for i, (start_idx, group_name) in enumerate(group_boundaries):
        # Find end of this group
        if i + 1 < len(group_boundaries):
            end_idx = group_boundaries[i + 1][0] - 1
        else:
            end_idx = len(labels) - 1

        # Draw separator line
        if start_idx > 0:
            ax.axvline(x=start_idx - 0.5, color="black", linewidth=0.5, linestyle="--")

        # Group label at bottom
        mid = (start_idx + end_idx) / 2
        ax.text(mid, -12, group_name, ha="center", fontsize=9, fontweight="bold")

    ax.set_ylabel("% of responses")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.5, len(labels) - 0.5)

    plt.title(title)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def main(
    results_dir: Path,
    qualitative_templates_path: Path | None,
    rating_templates_path: Path | None,
) -> None:
    results_path = results_dir / "rating_results_full.json"
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return

    results = load_results(results_path)
    all_scores = results["results"]
    n_completions = results["n_completions"]
    print(f"Loaded results: {n_completions} completions, {len(results['templates'])} templates")

    # Load templates to get tags
    templates: dict[str, PromptTemplate] = {}
    if qualitative_templates_path:
        for t in load_templates_from_yaml(qualitative_templates_path):
            templates[t.name] = t
    if rating_templates_path:
        for t in load_templates_from_yaml(rating_templates_path):
            templates[t.name] = t

    if not templates:
        print("No template files provided")
        return

    date_str = datetime.now().strftime("%m%d%y")

    # Group by scale (binary/ternary for qualitative, numeric scales for stated)
    qualitative_by_scale: dict[str, list[tuple[str, tuple[float, float, float]]]] = {
        "binary": [],
        "ternary": [],
    }
    rating_by_scale: dict[str, list[tuple[str, tuple[float, float, float]]]] = {}

    for template_name, scores_dict in all_scores.items():
        if template_name not in templates:
            continue
        t = templates[template_name]
        tags = t.tags_dict
        scale = tags.get("scale", "unknown")
        response_format = tags.get("response_format", "regex")
        phrasing = tags.get("phrasing", "?")

        scores = list(scores_dict.values())
        dist = compute_distribution(scores, scale)
        short_name = make_short_name(template_name, phrasing, response_format, scale)

        if "qualitative" in template_name:
            if scale in qualitative_by_scale:
                qualitative_by_scale[scale].append((short_name, dist))
        else:
            # Numeric rating template
            if scale not in rating_by_scale:
                rating_by_scale[scale] = []
            rating_by_scale[scale].append((short_name, dist))

    # Plot qualitative: binary vs ternary
    if any(qualitative_by_scale.values()):
        plot_grouped_stacked(
            qualitative_by_scale,
            f"Qualitative Templates by Scale (n={n_completions})",
            results_dir / f"plot_{date_str}_qualitative_by_scale.png",
        )

    # Plot numeric ratings by scale
    if rating_by_scale:
        plot_grouped_stacked(
            rating_by_scale,
            f"Numeric Rating Templates by Scale (n={n_completions})",
            results_dir / f"plot_{date_str}_rating_by_scale.png",
        )

    # Combined view: group by response_format across all templates
    by_response_format: dict[str, list[tuple[str, tuple[float, float, float]]]] = {
        "regex": [],
        "xml": [],
        "tool_use": [],
    }
    for template_name, scores_dict in all_scores.items():
        if template_name not in templates:
            continue
        t = templates[template_name]
        tags = t.tags_dict
        response_format = tags.get("response_format", "regex")
        scale = tags.get("scale", "?")
        phrasing = tags.get("phrasing", "?")

        scores = list(scores_dict.values())
        dist = compute_distribution(scores, scale)
        short_name = make_short_name(template_name, phrasing, response_format, scale, include_type=True)

        if response_format in by_response_format:
            by_response_format[response_format].append((short_name, dist))

    plot_grouped_stacked(
        by_response_format,
        f"All Templates by Response Format (n={n_completions})",
        results_dir / f"plot_{date_str}_by_response_format.png",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot rating results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/probe_data/ratings"),
    )
    parser.add_argument(
        "--qualitative-templates",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--rating-templates",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    main(args.results_dir, args.qualitative_templates, args.rating_templates)
