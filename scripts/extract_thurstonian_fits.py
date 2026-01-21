#!/usr/bin/env python3
"""Extract thurstonian fit CSVs and configs from old template-level cache directories.

Copies thurstonian_active_learning_*.csv/yaml files along with config.yaml and active_learning.yaml
to a new consolidated directory structure.

Output structure:
  results/thurstonian_fits/{model}/{run_name}/
    config.yaml
    active_learning.yaml (if exists)
    thurstonian_active_learning_{hash}.csv
    thurstonian_active_learning_{hash}.yaml
"""

import shutil
from pathlib import Path

SOURCE_DIRS = [
    Path("results/pre_task_revealed"),
    Path("results/post_task_revealed"),
    Path("results/pre_task_stated"),
    Path("results/post_task_stated"),
]

OUTPUT_DIR = Path("results/thurstonian_fits")


def extract_fits(dry_run: bool = False) -> dict[str, int]:
    stats = {"runs_copied": 0, "files_copied": 0, "skipped": 0}

    for source_dir in SOURCE_DIRS:
        if not source_dir.exists():
            continue

        # Handle both flat and model-nested structures
        for model_dir in sorted(source_dir.iterdir()):
            if not model_dir.is_dir():
                continue

            # Check if this is a model directory (contains run subdirs) or a run dir itself
            has_subdirs = any(d.is_dir() for d in model_dir.iterdir())
            if has_subdirs and not (model_dir / "config.yaml").exists():
                # Model directory containing runs
                run_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
                model_name = model_dir.name
            else:
                # Flat structure - model_dir is actually a run dir
                run_dirs = [model_dir]
                model_name = "unknown"

            for run_dir in sorted(run_dirs):
                if not run_dir.is_dir():
                    continue

                # Check for thurstonian CSVs
                thurstonian_csvs = list(run_dir.glob("thurstonian_active_learning_*.csv"))
                if not thurstonian_csvs:
                    continue

                # Determine output path
                out_dir = OUTPUT_DIR / model_name / run_dir.name
                if out_dir.exists():
                    stats["skipped"] += 1
                    continue

                if dry_run:
                    print(f"Would copy: {run_dir} -> {out_dir}")
                    stats["runs_copied"] += 1
                    continue

                out_dir.mkdir(parents=True, exist_ok=True)

                # Copy config.yaml
                config_path = run_dir / "config.yaml"
                if config_path.exists():
                    shutil.copy2(config_path, out_dir / "config.yaml")
                    stats["files_copied"] += 1

                # Copy active_learning.yaml
                al_path = run_dir / "active_learning.yaml"
                if al_path.exists():
                    shutil.copy2(al_path, out_dir / "active_learning.yaml")
                    stats["files_copied"] += 1

                # Copy all thurstonian files (csv and yaml)
                for csv_path in thurstonian_csvs:
                    shutil.copy2(csv_path, out_dir / csv_path.name)
                    stats["files_copied"] += 1

                    yaml_path = csv_path.with_suffix(".yaml")
                    if yaml_path.exists():
                        shutil.copy2(yaml_path, out_dir / yaml_path.name)
                        stats["files_copied"] += 1

                stats["runs_copied"] += 1
                print(f"Copied: {run_dir.name}")

    return stats


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Show what would be copied without copying")
    args = parser.parse_args()

    print(f"Extracting thurstonian fits to {OUTPUT_DIR}")
    if args.dry_run:
        print("(dry run - no files will be copied)")
    print()

    stats = extract_fits(dry_run=args.dry_run)

    print()
    print(f"Runs copied: {stats['runs_copied']}")
    print(f"Files copied: {stats['files_copied']}")
    print(f"Skipped (already exist): {stats['skipped']}")


if __name__ == "__main__":
    main()
