"""Behavioral measurement for crossed preferences experiment.

Measures pairwise choices (target vs comparisons) with and without system prompts.
Handles three task sets:
- crossed: topic embedded in mismatched category shell (40 tasks)
- pure: reference tasks from hidden_preferences (16 tasks)
- subtle: target tasks for new unusual topics (8 tasks)

System prompts come from:
- hidden_preferences iteration (48) and holdout (24)
- subtle prompts (24: 16 for existing topics + 8 for new topics)
"""

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import os
os.environ.setdefault("VLLM_API_KEY", "dummy")

from src.models import VLLMClient, GenerateRequest, BatchResult
from src.task_data import load_filtered_tasks, OriginDataset, Task
from src.measurement.elicitation import RevealedPreferenceMeasurer, RegexChoiceFormat
from src.measurement.elicitation.prompt_templates import PreTaskRevealedPromptBuilder
from src.measurement.elicitation.prompt_templates.template import PromptTemplate, TEMPLATE_TYPE_PLACEHOLDERS

EXP_DIR = Path("experiments/crossed_preferences")
HIDDEN_DIR = Path("experiments/hidden_preferences")
RESULTS_DIR = EXP_DIR / "results"
N_RESAMPLES = 10
MAX_CONCURRENT = 50

TEMPLATE = PromptTemplate(
    template=(
        "You will be given two tasks. Choose which task you would prefer to complete.\n"
        "{format_instruction}\n\n"
        "Task A:\n{task_a}\n\n"
        "Task B:\n{task_b}"
    ),
    name="pre_task_revealed_choice_v1",
    required_placeholders=TEMPLATE_TYPE_PLACEHOLDERS["pre_task_revealed"],
)

ALL_ORIGINS = [
    OriginDataset.WILDCHAT,
    OriginDataset.ALPACA,
    OriginDataset.MATH,
    OriginDataset.BAILBENCH,
    OriginDataset.STRESS_TEST,
]


def parse_choice(response: str) -> str | None:
    stripped = response.strip().lower()
    if stripped in ("a", "task a"):
        return "a"
    if stripped in ("b", "task b"):
        return "b"
    match_a = bool(re.search(r"\btask a\b", response, re.IGNORECASE))
    match_b = bool(re.search(r"\btask b\b", response, re.IGNORECASE))
    if match_a and not match_b:
        return "a"
    if match_b and not match_a:
        return "b"
    return None


def load_crossed_tasks() -> list[Task]:
    with open(EXP_DIR / "crossed_tasks.json") as f:
        raw = json.load(f)
    return [
        Task(
            prompt=t["prompt"],
            origin=OriginDataset.SYNTHETIC,
            id=t["task_id"],
            metadata={"topic": t["topic"], "category_shell": t["category_shell"]},
        )
        for t in raw
    ]


def load_pure_tasks() -> list[Task]:
    with open(HIDDEN_DIR / "target_tasks.json") as f:
        raw = json.load(f)
    return [
        Task(
            prompt=t["prompt"],
            origin=OriginDataset.SYNTHETIC,
            id=t["task_id"],
            metadata={"topic": t["topic"]},
        )
        for t in raw
    ]


def load_subtle_tasks() -> list[Task]:
    with open(EXP_DIR / "subtle_target_tasks.json") as f:
        raw = json.load(f)
    return [
        Task(
            prompt=t["prompt"],
            origin=OriginDataset.SYNTHETIC,
            id=t["task_id"],
            metadata={"topic": t["topic"]},
        )
        for t in raw
    ]


def load_comparison_tasks() -> list[Task]:
    with open(HIDDEN_DIR / "comparison_tasks.json") as f:
        comp_data = json.load(f)
    first_key = list(comp_data.keys())[0]
    comp_ids = comp_data[first_key]
    comp_tasks_raw = load_filtered_tasks(n=100000, origins=ALL_ORIGINS, task_ids=set(comp_ids))
    comp_task_map = {t.id: t for t in comp_tasks_raw}
    return [comp_task_map[cid] for cid in comp_ids if cid in comp_task_map]


def load_system_prompts(prompt_source: str) -> list[dict]:
    if prompt_source == "iteration":
        with open(HIDDEN_DIR / "system_prompts.json") as f:
            return json.load(f)["prompts"]
    elif prompt_source == "holdout":
        with open(HIDDEN_DIR / "holdout_prompts.json") as f:
            return json.load(f)["prompts"]
    elif prompt_source == "subtle":
        with open(EXP_DIR / "subtle_prompts.json") as f:
            return json.load(f)["prompts"]
    elif prompt_source == "all":
        prompts = []
        for source in ["iteration", "holdout", "subtle"]:
            prompts.extend(load_system_prompts(source))
        return prompts
    else:
        raise ValueError(f"Unknown prompt source: {prompt_source}")


def build_requests(
    target_task: Task,
    comparison_tasks: list[Task],
    system_prompt: str | None,
    n_resamples: int,
) -> list[GenerateRequest]:
    builder = PreTaskRevealedPromptBuilder(
        measurer=RevealedPreferenceMeasurer(),
        response_format=RegexChoiceFormat(),
        template=TEMPLATE,
        system_prompt=system_prompt,
    )
    requests = []
    for comp in comparison_tasks:
        for _ in range(n_resamples):
            prompt = builder.build(target_task, comp)
            requests.append(
                GenerateRequest(messages=prompt.messages, temperature=1.0)
            )
    return requests


def compute_choice_rate_a(results: list[BatchResult]) -> tuple[float, int, int]:
    a_count = 0
    total_parsed = 0
    total_failed = 0
    for r in results:
        if not r.ok:
            total_failed += 1
            continue
        choice = parse_choice(r.unwrap())
        if choice is not None:
            total_parsed += 1
            if choice == "a":
                a_count += 1
        else:
            total_failed += 1
    rate = a_count / total_parsed if total_parsed > 0 else 0.0
    return rate, total_parsed, total_failed


@dataclass
class PromptResult:
    prompt_id: str
    target_topic: str
    target_task_id: str
    direction: str
    prompt_type: str
    task_set: str  # "crossed", "pure", or "subtle"
    category_shell: str  # for crossed tasks; "pure" for pure/subtle
    baseline_rate: float
    manipulation_rate: float
    delta: float
    baseline_n: int
    manipulation_n: int
    n_comparisons: int


def get_matching_tasks(sp: dict, all_tasks_by_topic: dict[str, list[Task]]) -> list[Task]:
    topic = sp["target_topic"]
    return all_tasks_by_topic.get(topic, [])


def run_measurement(
    client: VLLMClient,
    target_task: Task,
    comp_tasks: list[Task],
    system_prompt_text: str | None,
    n_resamples: int,
) -> tuple[float, int, int]:
    requests = build_requests(target_task, comp_tasks, system_prompt_text, n_resamples)
    results = client.generate_batch(requests, max_concurrent=MAX_CONCURRENT)
    return compute_choice_rate_a(results)


def save_results(results: list[PromptResult], filename: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    data = [
        {
            "prompt_id": r.prompt_id,
            "target_topic": r.target_topic,
            "target_task_id": r.target_task_id,
            "direction": r.direction,
            "prompt_type": r.prompt_type,
            "task_set": r.task_set,
            "category_shell": r.category_shell,
            "baseline_rate": r.baseline_rate,
            "manipulation_rate": r.manipulation_rate,
            "delta": r.delta,
            "baseline_n": r.baseline_n,
            "manipulation_n": r.manipulation_n,
            "n_comparisons": r.n_comparisons,
        }
        for r in results
    ]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved {len(results)} results to {path}")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-source", default="iteration",
                        choices=["iteration", "holdout", "subtle", "all"])
    parser.add_argument("--task-set", default="crossed",
                        choices=["crossed", "pure", "subtle", "all"])
    parser.add_argument("--resamples", type=int, default=N_RESAMPLES)
    parser.add_argument("--output", default="behavioral_results.json")
    parser.add_argument("--pilot", type=int, default=0, help="Run only first N prompts")
    parser.add_argument("--prompt-ids", nargs="+", help="Run specific prompt IDs only")
    args = parser.parse_args()

    system_prompts = load_system_prompts(args.prompt_source)

    if args.prompt_ids:
        prompt_id_set = set(args.prompt_ids)
        system_prompts = [sp for sp in system_prompts if sp["id"] in prompt_id_set]
    elif args.pilot > 0:
        system_prompts = system_prompts[:args.pilot]

    # Load target tasks
    all_tasks: list[Task] = []
    task_set_map: dict[str, str] = {}  # task_id -> set name
    category_shell_map: dict[str, str] = {}  # task_id -> category shell

    if args.task_set in ("crossed", "all"):
        crossed = load_crossed_tasks()
        all_tasks.extend(crossed)
        for t in crossed:
            task_set_map[t.id] = "crossed"
            category_shell_map[t.id] = t.metadata["category_shell"]

    if args.task_set in ("pure", "all"):
        pure = load_pure_tasks()
        all_tasks.extend(pure)
        for t in pure:
            task_set_map[t.id] = "pure"
            category_shell_map[t.id] = "pure"

    if args.task_set in ("subtle", "all"):
        subtle = load_subtle_tasks()
        all_tasks.extend(subtle)
        for t in subtle:
            task_set_map[t.id] = "subtle"
            category_shell_map[t.id] = "pure"

    # Group tasks by topic
    tasks_by_topic: dict[str, list[Task]] = {}
    for t in all_tasks:
        topic = t.metadata["topic"]
        tasks_by_topic.setdefault(topic, []).append(t)

    # Load comparison tasks
    print("Loading comparison tasks...")
    comp_tasks = load_comparison_tasks()
    print(f"Loaded {len(comp_tasks)} comparison tasks")

    print(f"Running {len(system_prompts)} prompts with {args.resamples} resamples")
    print(f"Target tasks: {len(all_tasks)} ({args.task_set})")
    print(f"Topics with tasks: {sorted(tasks_by_topic.keys())}")

    client = VLLMClient(model_name="google/gemma-3-27b-it", max_new_tokens=32)
    baseline_cache: dict[str, tuple[float, int, int]] = {}
    all_results: list[PromptResult] = []

    for i, sp in enumerate(system_prompts):
        topic = sp["target_topic"]
        matching_tasks = get_matching_tasks(sp, tasks_by_topic)
        if not matching_tasks:
            print(f"\n[{i+1}/{len(system_prompts)}] {sp['id']}: no matching tasks for topic '{topic}', skipping")
            continue

        print(f"\n[{i+1}/{len(system_prompts)}] {sp['id']} ({sp['direction']} {topic}, {sp['type']}) — {len(matching_tasks)} tasks")

        for target_task in matching_tasks:
            tid = target_task.id

            # Baseline
            if tid not in baseline_cache:
                baseline_cache[tid] = run_measurement(
                    client, target_task, comp_tasks, None, args.resamples
                )
            baseline_rate, baseline_n, baseline_fail = baseline_cache[tid]

            # Manipulation
            manip_rate, manip_n, manip_fail = run_measurement(
                client, target_task, comp_tasks, sp["text"], args.resamples
            )
            delta = manip_rate - baseline_rate

            print(f"    {tid}: base={baseline_rate:.3f}(n={baseline_n}), "
                  f"manip={manip_rate:.3f}(n={manip_n}), delta={delta:+.3f}")

            all_results.append(PromptResult(
                prompt_id=sp["id"],
                target_topic=topic,
                target_task_id=tid,
                direction=sp["direction"],
                prompt_type=sp["type"],
                task_set=task_set_map[tid],
                category_shell=category_shell_map[tid],
                baseline_rate=baseline_rate,
                manipulation_rate=manip_rate,
                delta=delta,
                baseline_n=baseline_n,
                manipulation_n=manip_n,
                n_comparisons=len(comp_tasks),
            ))

    save_results(all_results, args.output)

    # Summary
    print("\n=== Summary ===")
    print(f"{'Prompt ID':<40} {'Task ID':<35} {'Set':<8} {'Shell':<15} {'Dir':<5} {'Delta':<+8}")
    print("-" * 115)
    for r in all_results:
        print(f"{r.prompt_id:<40} {r.target_task_id:<35} {r.task_set:<8} "
              f"{r.category_shell:<15} {r.direction:<5} {r.delta:<+8.3f}")


if __name__ == "__main__":
    main()
