"""Collect self-reported ratings from a model on its completions.

Uses existing completions and collects ratings via API calls with various
prompt templates. Output format is compatible with run_probe_training.py.
"""

import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.models import get_client
from src.preferences.measurement.measure import measure_post_task_stated
from src.preferences.measurement.measurer import StatedScoreMeasurer
from src.preferences.measurement.response_format import qualitative_format_for_scale
from src.preferences.templates.builders import PostTaskStatedPromptBuilder
from src.preferences.templates.template import load_templates_from_yaml
from src.task_data import load_completions

MODEL = "llama-3.1-8b"
TEMPLATES_PATH = Path("src/preferences/templates/data/post_task_qualitative_v1.yaml")


def run_rating_collection(
    completions_path: Path,
    output_dir: Path,
    max_concurrent: int = 50,
    limit: int | None = None,
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
    templates = load_templates_from_yaml(TEMPLATES_PATH)

    print(f"\nCollecting ratings with {len(templates)} templates...")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict[str, float]] = {}

    for template in templates:
        print(f"\n  Template: {template.name}")
        print(f"    Text: {template.template[:60]}...")

        scale = template.tags_dict.get("scale", "ternary")
        response_format = qualitative_format_for_scale(scale)
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
        "--max-concurrent",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    args = parser.parse_args()

    run_rating_collection(
        completions_path=args.completions,
        output_dir=args.output_dir,
        max_concurrent=args.max_concurrent,
        limit=args.limit,
    )
