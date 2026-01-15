"""Check stated experiment data quality and compute basic statistics."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import yaml

RESULTS_DIR = Path("results/stated")


def load_measurements(path: Path) -> list[dict]:
    with open(path) as f:
        return yaml.safe_load(f)


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_scale_from_config(config: dict) -> tuple[int, int]:
    """Extract the expected scale from template_tags."""
    tags = config.get("template_tags", {})
    scale_str = tags.get("scale", "1-5")
    if scale_str == "1-5":
        return (1, 5)
    elif scale_str == "1-10":
        return (1, 10)
    return (1, 5)


def check_model(model_short: str) -> dict:
    """Run checks on all experiments for a given model."""
    # Match both old pattern (stated_stated_001_model) and new pattern (stated_stated_001_model_format_seedN)
    folders = sorted(RESULTS_DIR.glob(f"stated_stated_*_{model_short}")) + \
              sorted(RESULTS_DIR.glob(f"stated_stated_*_{model_short}_*"))
    # Deduplicate
    folders = sorted(set(folders))

    stats = {
        "model": model_short,
        "n_experiments": len(folders),
        "missing_measurements": [],
        "missing_configs": [],
        "score_counts_1_5": Counter(),
        "score_counts_1_10": Counter(),
        "outlier_scores": [],
        "task_counts": Counter(),
        "measurements_per_experiment": [],
        "template_ids": set(),
        "config_keys": set(),  # Track unique (template, format, seed) combos
        "scores_by_task_1_5": defaultdict(list),
        "scores_by_task_1_10": defaultdict(list),
        "scale_counts": Counter(),
        "experiments_by_scale": {"1-5": [], "1-10": []},
        "response_format_counts": Counter(),
        "seed_counts": Counter(),
    }

    for folder in folders:
        measurements_path = folder / "measurements.yaml"
        config_path = folder / "config.yaml"

        if not measurements_path.exists():
            stats["missing_measurements"].append(folder.name)
            continue
        if not config_path.exists():
            stats["missing_configs"].append(folder.name)

        config = load_config(config_path) if config_path.exists() else {}
        measurements = load_measurements(measurements_path)
        scale_min, scale_max = get_scale_from_config(config)
        scale_key = f"{scale_min}-{scale_max}"
        stats["scale_counts"][scale_key] += 1
        stats["experiments_by_scale"].setdefault(scale_key, []).append(folder.name)

        stats["measurements_per_experiment"].append(len(measurements))

        template_id = folder.name.split("_")[2]
        stats["template_ids"].add(template_id)

        # Track response_format and seed from config
        tags = config.get("template_tags", {})
        response_format = tags.get("response_format", "unknown")
        seed = tags.get("seed", "unknown")
        stats["response_format_counts"][response_format] += 1
        stats["seed_counts"][seed] += 1
        stats["config_keys"].add((template_id, response_format, seed))

        for m in measurements:
            score = m["score"]
            task_id = m["task_id"]

            stats["task_counts"][task_id] += 1

            if scale_max == 5:
                stats["score_counts_1_5"][score] += 1
                stats["scores_by_task_1_5"][task_id].append(score)
            else:
                stats["score_counts_1_10"][score] += 1
                stats["scores_by_task_1_10"][task_id].append(score)

            # Flag truly outlier scores (way outside expected range)
            if not isinstance(score, (int, float)):
                stats["outlier_scores"].append((folder.name, task_id, score, scale_key))
            elif score < scale_min - 0.5 or score > scale_max + 0.5:
                stats["outlier_scores"].append((folder.name, task_id, score, scale_key))

    return stats


def print_score_distribution(score_counts: Counter, scale_label: str) -> None:
    """Print score distribution for a given scale."""
    if not score_counts:
        return
    total = sum(score_counts.values())
    print(f"\n  {scale_label} scale ({total:,} measurements):")
    for score in sorted(score_counts.keys()):
        if score > 15:  # Skip extreme outliers in display
            continue
        count = score_counts[score]
        pct = 100 * count / total if total else 0
        bar = "█" * int(pct / 2)
        print(f"    Score {score:>4}: {count:>6} ({pct:5.1f}%) {bar}")


def compute_variance_stats(scores_by_task: dict) -> list:
    """Compute variance statistics per task."""
    variances = []
    for task_id, scores in scores_by_task.items():
        if len(scores) > 1:
            mean = sum(scores) / len(scores)
            var = sum((s - mean) ** 2 for s in scores) / len(scores)
            variances.append((task_id, var, mean, len(scores)))
    return variances


def print_report(stats: dict) -> None:
    """Print a formatted report of the statistics."""
    print("=" * 70)
    print(f"Model: {stats['model']}")
    print("=" * 70)

    print(f"\n[COMPLETENESS]")
    print(f"  Experiment folders found: {stats['n_experiments']}")
    print(f"  Unique template IDs: {len(stats['template_ids'])}")
    print(f"  Unique (template, format, seed) configs: {len(stats['config_keys'])}")

    if stats["missing_measurements"]:
        print(f"  WARNING: Missing measurements.yaml: {len(stats['missing_measurements'])}")
        for name in stats["missing_measurements"][:5]:
            print(f"      - {name}")
    else:
        print("  OK: All experiments have measurements.yaml")

    if stats["missing_configs"]:
        print(f"  WARNING: Missing config.yaml: {len(stats['missing_configs'])}")
    else:
        print("  OK: All experiments have config.yaml")

    print(f"\n[SCALE BREAKDOWN]")
    for scale, count in sorted(stats["scale_counts"].items()):
        print(f"  {scale}: {count} experiments")

    print(f"\n[RESPONSE FORMAT BREAKDOWN]")
    for fmt, count in sorted(stats["response_format_counts"].items()):
        print(f"  {fmt}: {count} experiments")

    print(f"\n[SEED BREAKDOWN]")
    for seed, count in sorted(stats["seed_counts"].items()):
        print(f"  seed={seed}: {count} experiments")

    print(f"\n[MEASUREMENT COUNTS]")
    mpe = stats["measurements_per_experiment"]
    if mpe:
        print(f"  Total measurements: {sum(mpe):,}")
        print(f"  Per experiment: min={min(mpe)}, max={max(mpe)}, mean={sum(mpe)/len(mpe):.1f}")

    print(f"\n[SCORE DISTRIBUTIONS]")
    print_score_distribution(stats["score_counts_1_5"], "1-5")
    print_score_distribution(stats["score_counts_1_10"], "1-10")

    print(f"\n[OUTLIER SCORES]")
    if stats["outlier_scores"]:
        print(f"  Found {len(stats['outlier_scores'])} outlier scores")
        # Group by experiment
        by_experiment = defaultdict(list)
        for folder, task_id, score, scale in stats["outlier_scores"]:
            by_experiment[folder].append((task_id, score, scale))
        print(f"  Affected experiments: {len(by_experiment)}")
        for folder, items in list(by_experiment.items())[:5]:
            print(f"    {folder}: {len(items)} outliers (e.g. {items[0][1]} on scale {items[0][2]})")
    else:
        print("  OK: No outlier scores detected")

    print(f"\n[TASK COVERAGE]")
    print(f"  Unique tasks: {len(stats['task_counts'])}")
    task_counts = list(stats["task_counts"].values())
    if task_counts:
        print(f"  Measurements per task: min={min(task_counts)}, max={max(task_counts)}, mean={sum(task_counts)/len(task_counts):.1f}")

    # Score consistency for each scale
    for scale_key, scores_by_task in [("1-5", stats["scores_by_task_1_5"]), ("1-10", stats["scores_by_task_1_10"])]:
        if not scores_by_task:
            continue
        print(f"\n[SCORE CONSISTENCY - {scale_key} scale]")
        variances = compute_variance_stats(scores_by_task)
        if variances:
            avg_var = sum(v[1] for v in variances) / len(variances)
            print(f"  Average within-task variance: {avg_var:.3f}")

            variances.sort(key=lambda x: x[1])
            print(f"  Most consistent (low variance):")
            for task_id, var, mean, n in variances[:3]:
                print(f"    {task_id}: var={var:.3f}, mean={mean:.2f}, n={n}")

            print(f"  Most variable (high variance):")
            for task_id, var, mean, n in variances[-3:]:
                print(f"    {task_id}: var={var:.3f}, mean={mean:.2f}, n={n}")

            mean_scores = [v[2] for v in variances]
            overall_mean = sum(mean_scores) / len(mean_scores)
            print(f"  Overall mean score: {overall_mean:.2f}")


def main():
    models = [
        "qwen-2.5-7b-instruct",
        "qwen-2.5-72b-instruct",
    ]

    for model in models:
        stats = check_model(model)
        print_report(stats)
        print("\n")


if __name__ == "__main__":
    main()
