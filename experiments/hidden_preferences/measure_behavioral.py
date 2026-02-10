"""Behavioral measurement for hidden preferences experiment.

Runs pairwise choices (target vs comparisons) with and without system prompts via vLLM.
Each prompt targets TWO tasks (2 per topic), producing separate results for each.

Key differences from OOD:
- target_topic instead of target_category
- 16 synthetic target tasks (2 per topic, loaded from target_tasks.json)
- --control-mode routes to OOD target tasks for positive controls
"""

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("VLLM_API_KEY", "dummy")

from src.models import VLLMClient, GenerateRequest, BatchResult
from src.task_data import load_filtered_tasks, OriginDataset, Task
from src.measurement.elicitation import RevealedPreferenceMeasurer, RegexChoiceFormat
from src.measurement.elicitation.prompt_templates import PreTaskRevealedPromptBuilder
from src.measurement.elicitation.prompt_templates.template import PromptTemplate, TEMPLATE_TYPE_PLACEHOLDERS

EXP_DIR = Path("experiments/hidden_preferences")
OOD_DIR = Path("experiments/ood_generalization")
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


def load_target_tasks_hidden() -> list[Task]:
    """Load the 16 synthetic target tasks from this experiment."""
    with open(EXP_DIR / "target_tasks.json") as f:
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


def load_target_tasks_ood() -> list[Task]:
    """Load OOD target tasks for positive controls."""
    with open(OOD_DIR / "target_tasks.json") as f:
        raw = json.load(f)
    target_ids = {t["task_id"] for t in raw}
    tasks = load_filtered_tasks(n=100000, origins=ALL_ORIGINS, task_ids=target_ids)
    return tasks


def load_comparison_tasks() -> dict[str, list[str]]:
    with open(EXP_DIR / "comparison_tasks.json") as f:
        return json.load(f)


def load_comparison_tasks_ood() -> dict[str, list[str]]:
    with open(OOD_DIR / "comparison_tasks.json") as f:
        return json.load(f)


def build_requests_for_prompt(
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


@dataclass
class PromptResult:
    prompt_id: str
    target_topic: str
    target_task_id: str
    direction: str
    prompt_type: str
    baseline_rate: float
    manipulation_rate: float
    delta: float
    baseline_n: int
    manipulation_n: int
    n_comparisons: int


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


def run_measurement(
    client: VLLMClient,
    target_task: Task,
    comp_tasks: list[Task],
    system_prompt_text: str | None,
    n_resamples: int,
) -> tuple[float, int, int]:
    """Run baseline or manipulation measurement for a single target task."""
    requests = build_requests_for_prompt(target_task, comp_tasks, system_prompt_text, n_resamples)
    results = client.generate_batch(requests, max_concurrent=MAX_CONCURRENT)
    return compute_choice_rate_a(results)


def run_single_manipulation_hidden(
    client: VLLMClient,
    sp: dict,
    target_tasks_by_topic: dict[str, list[Task]],
    comp_tasks: list[Task],
    n_resamples: int,
    baseline_cache: dict[str, tuple[float, int, int]],
) -> list[PromptResult]:
    """Run one system prompt against both target tasks for its topic."""
    topic = sp["target_topic"]
    tasks_for_topic = target_tasks_by_topic[topic]
    results = []

    for target_task in tasks_for_topic:
        tid = target_task.id

        # Get or compute baseline
        if tid not in baseline_cache:
            baseline_requests = build_requests_for_prompt(target_task, comp_tasks, None, n_resamples)
            baseline_results = client.generate_batch(baseline_requests, max_concurrent=MAX_CONCURRENT)
            baseline_cache[tid] = compute_choice_rate_a(baseline_results)

        baseline_rate, baseline_n, baseline_fail = baseline_cache[tid]

        # Manipulation
        manip_requests = build_requests_for_prompt(target_task, comp_tasks, sp["text"], n_resamples)
        manip_results = client.generate_batch(manip_requests, max_concurrent=MAX_CONCURRENT)
        manip_rate, manip_n, manip_fail = compute_choice_rate_a(manip_results)
        delta = manip_rate - baseline_rate

        print(f"    {tid}: baseline={baseline_rate:.3f}(n={baseline_n},fail={baseline_fail}), "
              f"manip={manip_rate:.3f}(n={manip_n},fail={manip_fail}), delta={delta:+.3f}")

        results.append(PromptResult(
            prompt_id=sp["id"],
            target_topic=topic,
            target_task_id=tid,
            direction=sp["direction"],
            prompt_type=sp["type"],
            baseline_rate=baseline_rate,
            manipulation_rate=manip_rate,
            delta=delta,
            baseline_n=baseline_n,
            manipulation_n=manip_n,
            n_comparisons=len(comp_tasks),
        ))

    return results


def run_single_manipulation_control(
    client: VLLMClient,
    sp: dict,
    ood_targets: list[dict],
    ood_task_map: dict[str, Task],
    ood_comparisons: dict[str, list[str]],
    n_resamples: int,
    baseline_cache: dict[str, tuple[float, int, int]],
) -> list[PromptResult]:
    """Run a positive control prompt using OOD target tasks."""
    category = sp["target_category"]
    target_info = next(t for t in ood_targets if t["topic"] == category)
    target_task_id = target_info["task_id"]
    target_task = ood_task_map[target_task_id]

    comp_ids = ood_comparisons[target_task_id]
    comp_tasks = [ood_task_map[cid] for cid in comp_ids if cid in ood_task_map]

    if target_task_id not in baseline_cache:
        baseline_requests = build_requests_for_prompt(target_task, comp_tasks, None, n_resamples)
        baseline_results = client.generate_batch(baseline_requests, max_concurrent=MAX_CONCURRENT)
        baseline_cache[target_task_id] = compute_choice_rate_a(baseline_results)

    baseline_rate, baseline_n, baseline_fail = baseline_cache[target_task_id]

    manip_requests = build_requests_for_prompt(target_task, comp_tasks, sp["text"], n_resamples)
    manip_results = client.generate_batch(manip_requests, max_concurrent=MAX_CONCURRENT)
    manip_rate, manip_n, manip_fail = compute_choice_rate_a(manip_results)
    delta = manip_rate - baseline_rate

    print(f"    {target_task_id}: baseline={baseline_rate:.3f}(n={baseline_n},fail={baseline_fail}), "
          f"manip={manip_rate:.3f}(n={manip_n},fail={manip_fail}), delta={delta:+.3f}")

    return [PromptResult(
        prompt_id=sp["id"],
        target_topic=category,
        target_task_id=target_task_id,
        direction=sp["direction"],
        prompt_type=sp["type"],
        baseline_rate=baseline_rate,
        manipulation_rate=manip_rate,
        delta=delta,
        baseline_n=baseline_n,
        manipulation_n=manip_n,
        n_comparisons=len(comp_tasks),
    )]


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
    parser.add_argument("--prompt-file", default="system_prompts.json")
    parser.add_argument("--resamples", type=int, default=N_RESAMPLES)
    parser.add_argument("--output", default="behavioral_results.json")
    parser.add_argument("--control-mode", action="store_true",
                        help="Use OOD target tasks for positive controls")
    parser.add_argument("--pilot", type=int, default=0, help="Run only first N prompts")
    parser.add_argument("--prompt-ids", nargs="+", help="Run specific prompt IDs only")
    args = parser.parse_args()

    with open(EXP_DIR / args.prompt_file) as f:
        prompt_data = json.load(f)
    system_prompts = prompt_data["prompts"]

    if args.prompt_ids:
        prompt_id_set = set(args.prompt_ids)
        system_prompts = [sp for sp in system_prompts if sp["id"] in prompt_id_set]
    elif args.pilot > 0:
        system_prompts = system_prompts[:args.pilot]

    print(f"Running {len(system_prompts)} manipulations with {args.resamples} resamples")
    print(f"Control mode: {args.control_mode}")

    client = VLLMClient(model_name="google/gemma-3-27b-it", max_new_tokens=32)
    baseline_cache: dict[str, tuple[float, int, int]] = {}
    all_results: list[PromptResult] = []

    if args.control_mode:
        # Positive controls: use OOD targets and comparisons
        with open(OOD_DIR / "target_tasks.json") as f:
            ood_targets = json.load(f)
        ood_comparisons = load_comparison_tasks_ood()

        all_task_ids = set()
        for t in ood_targets:
            all_task_ids.add(t["task_id"])
        for comp_list in ood_comparisons.values():
            all_task_ids.update(comp_list)

        print(f"Loading {len(all_task_ids)} OOD tasks...")
        ood_tasks = load_filtered_tasks(n=100000, origins=ALL_ORIGINS, task_ids=all_task_ids)
        ood_task_map = {t.id: t for t in ood_tasks}
        print(f"Loaded {len(ood_task_map)} tasks")

        for i, sp in enumerate(system_prompts):
            print(f"\n[{i+1}/{len(system_prompts)}] Control: {sp['id']} ({sp['direction']} {sp['target_category']}, {sp['type']})")
            results = run_single_manipulation_control(
                client, sp, ood_targets, ood_task_map, ood_comparisons,
                args.resamples, baseline_cache
            )
            all_results.extend(results)
    else:
        # Hidden preferences: use synthetic targets
        target_tasks = load_target_tasks_hidden()
        target_tasks_by_topic: dict[str, list[Task]] = {}
        for t in target_tasks:
            topic = t.metadata["topic"]
            target_tasks_by_topic.setdefault(topic, []).append(t)

        comparisons = load_comparison_tasks()
        # All targets share the same pool, just get the first one
        first_key = list(comparisons.keys())[0]
        comp_ids = comparisons[first_key]

        print(f"Loading {len(comp_ids)} comparison tasks...")
        comp_tasks_raw = load_filtered_tasks(n=100000, origins=ALL_ORIGINS, task_ids=set(comp_ids))
        comp_task_map = {t.id: t for t in comp_tasks_raw}
        comp_tasks = [comp_task_map[cid] for cid in comp_ids if cid in comp_task_map]
        print(f"Loaded {len(comp_tasks)} comparison tasks")

        for i, sp in enumerate(system_prompts):
            print(f"\n[{i+1}/{len(system_prompts)}] {sp['id']} ({sp['direction']} {sp['target_topic']}, {sp['type']})")
            results = run_single_manipulation_hidden(
                client, sp, target_tasks_by_topic, comp_tasks,
                args.resamples, baseline_cache
            )
            all_results.extend(results)

    save_results(all_results, args.output)

    # Print summary
    print("\n=== Summary ===")
    print(f"{'Prompt ID':<35} {'Task ID':<25} {'Dir':<5} {'Base':<8} {'Manip':<8} {'Delta':<8}")
    print("-" * 90)
    for r in all_results:
        print(f"{r.prompt_id:<35} {r.target_task_id:<25} {r.direction:<5} "
              f"{r.baseline_rate:<8.3f} {r.manipulation_rate:<8.3f} {r.delta:<+8.3f}")


if __name__ == "__main__":
    main()
