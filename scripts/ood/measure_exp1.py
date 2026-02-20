"""Measure pairwise comparisons for OOD experiments using YAML configs.

Usage:
    python -m scripts.ood.measure_exp1 --config configs/ood/category_preference.yaml
    python -m scripts.ood.measure_exp1 --sub all
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("VLLM_API_KEY", "dummy")

from src.models import get_client
from src.task_data import load_filtered_tasks, OriginDataset
from src.task_data.task import Task
from src.ood.config import load_ood_config, OODMeasurementConfig
from src.ood.prompts import OODPromptSet
from src.ood.measurement import measure_pairs


PROMPTS_DIR = Path("configs/ood_prompts")
MAPPINGS_DIR = Path("configs/ood_mappings")
TASKS_DIR = Path("configs/ood_tasks")
CONFIGS_DIR = Path("configs/ood")
OUTPUT_DIR = Path("results/ood")

ALL_ORIGINS = [
    OriginDataset.WILDCHAT,
    OriginDataset.ALPACA,
    OriginDataset.MATH,
    OriginDataset.BAILBENCH,
    OriginDataset.STRESS_TEST,
]


def load_custom_tasks(json_path: Path) -> dict[str, Task]:
    with open(json_path) as f:
        data = json.load(f)
    return {
        t["task_id"]: Task(prompt=t["prompt"], origin=OriginDataset.SYNTHETIC, id=t["task_id"], metadata=t)
        for t in data
    }


def load_standard_tasks(task_ids: set[str]) -> dict[str, Task]:
    tasks = load_filtered_tasks(n=len(task_ids), origins=ALL_ORIGINS, task_ids=task_ids)
    return {t.id: t for t in tasks}


def collect_task_ids(triples: list[dict]) -> set[str]:
    ids = set()
    for t in triples:
        ids.add(t["task_a"])
        ids.add(t["task_b"])
    return ids


def load_tasks(config: OODMeasurementConfig, triples: list[dict]) -> dict[str, Task]:
    needed_ids = collect_task_ids(triples)
    if not config.custom_tasks:
        return load_standard_tasks(needed_ids)
    custom = load_custom_tasks(TASKS_DIR / config.custom_tasks)
    anchor_ids = needed_ids - set(custom.keys())
    standard = load_standard_tasks(anchor_ids)
    return {**custom, **standard}


def run_config(config: OODMeasurementConfig) -> None:
    config_name = config.mapping.removesuffix(".json")
    print(f"\n{'=' * 60}")
    print(f"Experiment: {config_name}")

    # Load mapping
    with open(MAPPINGS_DIR / config.mapping) as f:
        mapping = json.load(f)
    triples = mapping["pairs"]

    # Load prompts
    system_prompts: dict[str, str] = {}
    for prompts_file in config.prompts:
        prompt_set = OODPromptSet.load(PROMPTS_DIR / prompts_file)
        system_prompts["baseline"] = prompt_set.baseline_prompt
        for c in prompt_set.conditions:
            system_prompts[c.condition_id] = c.system_prompt

    # Load tasks
    tasks = load_tasks(config, triples)

    # Output path
    output_path = OUTPUT_DIR / config_name / "pairwise.json"

    client = get_client(config.model, max_new_tokens=config.max_new_tokens)

    measure_pairs(
        client=client,
        config=config,
        triples=triples,
        system_prompts=system_prompts,
        tasks=tasks,
        output_path=output_path,
    )


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=Path, help="Path to YAML config file")
    group.add_argument("--sub", choices=["all"], help="Run all configs in configs/ood/")
    args = parser.parse_args()

    if args.sub == "all":
        config_paths = sorted(CONFIGS_DIR.glob("*.yaml"))
    else:
        config_paths = [args.config]

    for path in config_paths:
        config = load_ood_config(path)
        run_config(config)

    print("\nAll done.")


if __name__ == "__main__":
    main()
