"""Collect self-reported ratings from a model on its completions.

Uses existing completions and collects ratings via API calls with various
prompt templates. Output format is compatible with run_probe_training.py.
"""

import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.models import get_client
from src.preference_measurement.measure import measure_post_task_stated
from src.preference_measurement.measurer import StatedScoreMeasurer
from src.preference_measurement.response_format import (
    qualitative_format_for_scale,
    RATING_FORMATS,
)
from src.prompt_templates.builders import PostTaskStatedPromptBuilder
from src.prompt_templates.template import PromptTemplate, load_templates_from_yaml
from src.task_data import load_completions

MODEL = "llama-3.1-8b"


def get_response_format(template: PromptTemplate):
    """Get appropriate response format based on template's scale and response_format tags."""
    scale = template.tags_dict.get("scale", "ternary")
    response_format = template.tags_dict.get("response_format", "regex")

    # Numeric scales: patterns like "1-2", "1-3", "1-10"
    if "-" in scale:
        parts = scale.split("-")
        scale_min, scale_max = int(parts[0]), int(parts[1])
        return RATING_FORMATS[response_format](scale_min=scale_min, scale_max=scale_max)

    # Qualitative scales: "binary", "ternary"
    return qualitative_format_for_scale(scale, response_format)


def run_rating_collection(
    completions_path: Path,
    output_dir: Path,
    qualitative_templates_path: Path | None = None,
    rating_templates_path: Path | None = None,
    max_concurrent: int = 50,
    limit: int | None = None,
    template_filter: list[str] | None = None,
):
    print(f"Loading completions from {completions_path}...")
    task_data = load_completions(completions_path)
    if limit:
        task_data = task_data[:limit]
    print(f"Loaded {len(task_data)} completions")

    # Show origin distribution
    origin_counts: dict[str, int] = {}
    for task, _ in task_data:
        origin = task.origin.value
        origin_counts[origin] = origin_counts.get(origin, 0) + 1
    print(f"Origins: {origin_counts}")

    client = get_client(MODEL, max_new_tokens=16)

    # Load templates from both files
    templates: list[PromptTemplate] = []
    if qualitative_templates_path:
        templates.extend(load_templates_from_yaml(qualitative_templates_path))
        print(f"Loaded {len(templates)} qualitative templates from {qualitative_templates_path}")
    if rating_templates_path:
        n_before = len(templates)
        templates.extend(load_templates_from_yaml(rating_templates_path))
        print(f"Loaded {len(templates) - n_before} rating templates from {rating_templates_path}")

    if not templates:
        print("No template files provided, exiting.")
        return

    if template_filter:
        templates = [t for t in templates if any(f in t.name for f in template_filter)]

    print(f"\nCollecting ratings with {len(templates)} templates...")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict[str, float]] = {}

    for template in templates:
        print(f"\n  Template: {template.name}")
        print(f"    Text: {template.template[:60]}...")

        response_format = get_response_format(template)
        print(f"    Format: {response_format.format_instruction()}")

        builder = PostTaskStatedPromptBuilder(
            measurer=StatedScoreMeasurer(),
            response_format=response_format,
            template=template,
        )

        batch = measure_post_task_stated(
            client=client,
            data=task_data,
            builder=builder,
            temperature=0.0,
            max_concurrent=max_concurrent,
        )

        # Convert to task_id -> score mapping
        scores = {ts.task.id: ts.score for ts in batch.successes}
        all_results[template.name] = scores

        # Stats
        n_success = len(batch.successes)
        n_errors = len(batch.failures)
        print(f"    Results: {n_success} parsed, {n_errors} errors")

        if n_success > 0:
            score_values = list(scores.values())
            unique_scores = sorted(set(score_values))
            print(f"    Score distribution: {unique_scores}")
            for score in unique_scores:
                count = score_values.count(score)
                pct = count / len(score_values) * 100
                print(f"      {score}: {count} ({pct:.1f}%)")

    # Save results
    full_results_path = output_dir / "rating_results_full.json"
    with open(full_results_path, "w") as f:
        json.dump({
            "model": MODEL,
            "n_completions": len(task_data),
            "templates": [t.name for t in templates],
            "results": all_results,
        }, f, indent=2)
    print(f"\nFull results saved to {full_results_path}")

    # Per-template scores files for probe training
    for template_name, scores in all_results.items():
        scores_path = output_dir / f"scores_{template_name}.json"
        with open(scores_path, "w") as f:
            json.dump(scores, f, indent=2)
        print(f"Scores saved to {scores_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect ratings on completions")
    parser.add_argument(
        "--completions",
        type=Path,
        default=Path("results/probe_data/completions_0.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/probe_data/ratings"),
    )
    parser.add_argument(
        "--qualitative-templates",
        type=Path,
        default=None,
        help="Path to qualitative templates YAML file",
    )
    parser.add_argument(
        "--rating-templates",
        type=Path,
        default=None,
        help="Path to numeric rating templates YAML file",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--templates",
        type=str,
        nargs="+",
        default=None,
        help="Filter templates by name substring (e.g., 'binary' 'ternary')",
    )
    args = parser.parse_args()

    run_rating_collection(
        completions_path=args.completions,
        output_dir=args.output_dir,
        qualitative_templates_path=args.qualitative_templates,
        rating_templates_path=args.rating_templates,
        max_concurrent=args.max_concurrent,
        limit=args.limit,
        template_filter=args.templates,
    )
