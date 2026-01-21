"""Migrate old folder-based measurement caches to unified cache format.

Run this script once to migrate all existing measurements:
    python scripts/migrate_to_unified_cache.py

After verification, the old folders in results/ can be deleted.
"""

from collections import defaultdict
from pathlib import Path
import hashlib
import re

from tqdm import tqdm

from src.measurement_storage.base import load_yaml, save_yaml, model_short_name
from src.measurement_storage.unified_cache import StatedCache, RevealedCache


def extract_template_hash_from_config(config: dict) -> str:
    """Generate a template hash from config if we don't have the original template."""
    template_name = config.get("template_name", "unknown")
    template_tags = config.get("template_tags", {})
    key = f"{template_name}__{sorted(template_tags.items())}"
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def collect_run_folders(base_dir: Path) -> list[Path]:
    """Collect run folders, handling both flat and nested (model/) structures."""
    folders = []
    for item in base_dir.iterdir():
        if not item.is_dir():
            continue
        if (item / "measurements.yaml").exists():
            folders.append(item)
        else:
            for subfolder in item.iterdir():
                if subfolder.is_dir() and (subfolder / "measurements.yaml").exists():
                    folders.append(subfolder)
    return folders


def migrate_pre_task_stated():
    """Migrate results/pre_task_stated/ folders to unified cache."""
    base_dir = Path("results/pre_task_stated")
    if not base_dir.exists():
        print("No pre_task_stated folder found, skipping.")
        return

    folders = collect_run_folders(base_dir)
    caches: dict[str, StatedCache] = {}
    migrated = 0
    skipped = 0

    for folder in tqdm(folders, desc="pre_task_stated"):
        measurements_path = folder / "measurements.yaml"
        config_path = folder / "config.yaml"

        folder_name = folder.name
        match = re.match(r"(.+)_([^_]+)_(regex|tool_use|xml)_seed(\d+)$", folder_name)
        if not match:
            skipped += 1
            continue

        template_name, model_short, response_format, seed_str = match.groups()
        rating_seed = int(seed_str)

        config = {}
        if config_path.exists():
            config = load_yaml(config_path)

        template_config = {
            "name": config.get("template_name", template_name),
            "tags": config.get("template_tags", {}),
            "template_hash": extract_template_hash_from_config(config),
        }

        model_name = config.get("model", model_short)
        measurements = load_yaml(measurements_path)

        if model_name not in caches:
            caches[model_name] = StatedCache(model_name)

        cache = caches[model_name]
        for m in measurements:
            cache.add(
                template_config=template_config,
                response_format=response_format,
                rating_seed=rating_seed,
                task_id=m["task_id"],
                sample={"score": m["score"]},
            )

        migrated += 1

    for model_name, cache in caches.items():
        cache.save()

    print(f"Pre-task stated: migrated {migrated} folders, skipped {skipped}")


def migrate_pre_task_revealed():
    """Migrate results/pre_task_revealed/ folders to unified cache."""
    base_dir = Path("results/pre_task_revealed")
    if not base_dir.exists():
        print("No pre_task_revealed folder found, skipping.")
        return

    folders = collect_run_folders(base_dir)
    caches: dict[str, RevealedCache] = {}
    migrated = 0
    skipped = 0

    for folder in tqdm(folders, desc="pre_task_revealed"):
        measurements_path = folder / "measurements.yaml"
        config_path = folder / "config.yaml"

        folder_name = folder.name
        match = re.match(r"(.+)_([^_]+)_(regex|tool_use|xml)_(canonical|reversed)(?:_seed(\d+))?$", folder_name)
        if not match:
            skipped += 1
            continue

        template_name, model_short, response_format, order, seed_str = match.groups()
        rating_seed = int(seed_str) if seed_str else 0

        config = {}
        if config_path.exists():
            config = load_yaml(config_path)

        template_config = {
            "name": config.get("template_name", template_name),
            "tags": config.get("template_tags", {}),
            "template_hash": extract_template_hash_from_config(config),
        }

        model_name = config.get("model", model_short)
        measurements = load_yaml(measurements_path)

        if model_name not in caches:
            caches[model_name] = RevealedCache(model_name)

        cache = caches[model_name]
        for m in measurements:
            cache.add(
                template_config=template_config,
                response_format=response_format,
                order=order,
                rating_seed=rating_seed,
                task_a_id=m["task_a"],
                task_b_id=m["task_b"],
                sample={"choice": m["choice"]},
            )

        migrated += 1

    for model_name, cache in caches.items():
        cache.save()

    print(f"Pre-task revealed: migrated {migrated} folders, skipped {skipped}")


def migrate_post_task_stated():
    """Migrate results/post_task_stated/ folders to unified cache."""
    base_dir = Path("results/post_task_stated")
    if not base_dir.exists():
        print("No post_task_stated folder found, skipping.")
        return

    folders = collect_run_folders(base_dir)
    caches: dict[str, StatedCache] = {}
    migrated = 0
    skipped = 0

    for folder in tqdm(folders, desc="post_task_stated"):
        measurements_path = folder / "measurements.yaml"
        config_path = folder / "config.yaml"

        folder_name = folder.name
        match = re.match(r"(.+)_([^_]+)_(regex|tool_use|xml)_cseed(\d+)_rseed(\d+)$", folder_name)
        if not match:
            skipped += 1
            continue

        template_name, model_short, response_format, cseed_str, rseed_str = match.groups()
        completion_seed = int(cseed_str)
        rating_seed = int(rseed_str)

        config = {}
        if config_path.exists():
            config = load_yaml(config_path)

        template_config = {
            "name": config.get("template_name", template_name),
            "tags": config.get("template_tags", {}),
            "template_hash": extract_template_hash_from_config(config),
        }

        model_name = config.get("model", model_short)
        measurements = load_yaml(measurements_path)

        if model_name not in caches:
            caches[model_name] = StatedCache(model_name)

        cache = caches[model_name]
        for m in measurements:
            cache.add(
                template_config=template_config,
                response_format=response_format,
                rating_seed=rating_seed,
                task_id=m["task_id"],
                sample={"score": m["score"]},
                completion_seed=completion_seed,
            )

        migrated += 1

    for model_name, cache in caches.items():
        cache.save()

    print(f"Post-task stated: migrated {migrated} folders, skipped {skipped}")


def migrate_post_task_revealed():
    """Migrate results/post_task_revealed/ folders to unified cache."""
    base_dir = Path("results/post_task_revealed")
    if not base_dir.exists():
        print("No post_task_revealed folder found, skipping.")
        return

    folders = collect_run_folders(base_dir)
    caches: dict[str, RevealedCache] = {}
    migrated = 0
    skipped = 0

    for folder in tqdm(folders, desc="post_task_revealed"):
        measurements_path = folder / "measurements.yaml"
        config_path = folder / "config.yaml"

        folder_name = folder.name
        # Try new format first: _cseed{N}_rseed{M}
        match = re.match(r"(.+)_([^_]+)_(regex|tool_use|xml)_(canonical|reversed)_cseed(\d+)_rseed(\d+)$", folder_name)
        if match:
            template_name, model_short, response_format, order, cseed_str, rseed_str = match.groups()
            completion_seed = int(cseed_str)
            rating_seed = int(rseed_str)
        else:
            # Try old format: _seed{N} (no completion seed)
            match = re.match(r"(.+)_([^_]+)_(regex|tool_use|xml)_(canonical|reversed)_seed(\d+)$", folder_name)
            if not match:
                skipped += 1
                continue
            template_name, model_short, response_format, order, seed_str = match.groups()
            completion_seed = 0  # Default for old format
            rating_seed = int(seed_str)

        config = {}
        if config_path.exists():
            config = load_yaml(config_path)

        template_config = {
            "name": config.get("template_name", template_name),
            "tags": config.get("template_tags", {}),
            "template_hash": extract_template_hash_from_config(config),
        }

        model_name = config.get("model", model_short)
        measurements = load_yaml(measurements_path)

        if model_name not in caches:
            caches[model_name] = RevealedCache(model_name)

        cache = caches[model_name]
        for m in measurements:
            cache.add(
                template_config=template_config,
                response_format=response_format,
                order=order,
                rating_seed=rating_seed,
                task_a_id=m["task_a"],
                task_b_id=m["task_b"],
                sample={"choice": m["choice"]},
                completion_seed=completion_seed,
            )

        migrated += 1

    for model_name, cache in caches.items():
        cache.save()

    print(f"Post-task revealed: migrated {migrated} folders, skipped {skipped}")


def main():
    print("Migrating measurement caches to unified format...")
    print()

    print("Migrating pre-task stated measurements...")
    migrate_pre_task_stated()
    print()

    print("Migrating pre-task revealed measurements...")
    migrate_pre_task_revealed()
    print()

    print("Migrating post-task stated measurements...")
    migrate_post_task_stated()
    print()

    print("Migrating post-task revealed measurements...")
    migrate_post_task_revealed()
    print()

    print("Migration complete!")
    print()
    print("The unified cache is stored in:")
    print("  results/cache/stated/{model}.yaml")
    print("  results/cache/revealed/{model}.yaml")
    print()
    print("After verifying the migration, you can delete the old folders:")
    print("  results/pre_task_stated/")
    print("  results/pre_task_revealed/")
    print("  results/post_task_stated/")
    print("  results/post_task_revealed/")


if __name__ == "__main__":
    main()
