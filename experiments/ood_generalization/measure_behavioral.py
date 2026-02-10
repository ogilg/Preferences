"""Behavioral measurement for OOD generalization experiment.

Runs pairwise choices (target vs comparisons) with and without system prompts via vLLM.
Computes P(choose target | manipulation) vs P(choose target | baseline) for each prompt.
"""

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

EXP_DIR = Path("experiments/ood_generalization")
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


def load_experiment_data() -> tuple[
    list[dict], dict[str, list[str]], list[dict]
]:
    with open(EXP_DIR / "target_tasks.json") as f:
        targets = json.load(f)
    with open(EXP_DIR / "comparison_tasks.json") as f:
        comparisons = json.load(f)
    with open(EXP_DIR / "system_prompts.json") as f:
        prompts_data = json.load(f)
    return targets, comparisons, prompts_data["prompts"]


def collect_all_task_ids(
    targets: list[dict], comparisons: dict[str, list[str]]
) -> set[str]:
    ids = set()
    for t in targets:
        ids.add(t["task_id"])
    for comp_list in comparisons.values():
        ids.update(comp_list)
    return ids


def load_all_tasks(task_ids: set[str]) -> dict[str, Task]:
    tasks = load_filtered_tasks(
        n=100000,
        origins=ALL_ORIGINS,
        task_ids=task_ids,
    )
    return {t.id: t for t in tasks}


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
    target_category: str
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


def run_single_manipulation(
    client: VLLMClient,
    sp: dict,
    targets: list[dict],
    comparisons: dict[str, list[str]],
    task_map: dict[str, Task],
    n_resamples: int,
) -> PromptResult:
    target_category = sp["target_category"]
    target_info = next(t for t in targets if t["topic"] == target_category)
    target_task_id = target_info["task_id"]
    target_task = task_map[target_task_id]

    comp_ids = comparisons[target_task_id]
    comp_tasks = [task_map[cid] for cid in comp_ids if cid in task_map]

    baseline_requests = build_requests_for_prompt(
        target_task, comp_tasks, None, n_resamples
    )
    manipulation_requests = build_requests_for_prompt(
        target_task, comp_tasks, sp["text"], n_resamples
    )

    all_requests = baseline_requests + manipulation_requests
    print(f"  Sending {len(all_requests)} requests ({len(comp_tasks)} comparisons × {n_resamples} resamples × 2 conditions)...")

    start = time.time()
    all_results = client.generate_batch(all_requests, max_concurrent=MAX_CONCURRENT)
    elapsed = time.time() - start

    baseline_results = all_results[:len(baseline_requests)]
    manipulation_results = all_results[len(baseline_requests):]

    baseline_rate, baseline_n, baseline_fail = compute_choice_rate_a(baseline_results)
    manip_rate, manip_n, manip_fail = compute_choice_rate_a(manipulation_results)
    delta = manip_rate - baseline_rate

    print(f"  Done in {elapsed:.1f}s. Baseline: {baseline_rate:.3f} (n={baseline_n}, fail={baseline_fail}), "
          f"Manipulation: {manip_rate:.3f} (n={manip_n}, fail={manip_fail}), Delta: {delta:+.3f}")

    return PromptResult(
        prompt_id=sp["id"],
        target_category=target_category,
        target_task_id=target_task_id,
        direction=sp["direction"],
        prompt_type=sp["type"],
        baseline_rate=baseline_rate,
        manipulation_rate=manip_rate,
        delta=delta,
        baseline_n=baseline_n,
        manipulation_n=manip_n,
        n_comparisons=len(comp_tasks),
    )


def save_results(results: list[PromptResult], filename: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    data = [
        {
            "prompt_id": r.prompt_id,
            "target_category": r.target_category,
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=int, default=0, help="Run only first N prompts (0=all)")
    parser.add_argument("--resamples", type=int, default=N_RESAMPLES)
    parser.add_argument("--prompt-ids", nargs="+", help="Run specific prompt IDs only")
    parser.add_argument("--prompt-file", help="Alternative prompt JSON file (e.g. holdout_prompts.json)")
    parser.add_argument("--output", default="behavioral_results.json")
    args = parser.parse_args()

    targets, comparisons, system_prompts = load_experiment_data()

    if args.prompt_file:
        with open(EXP_DIR / args.prompt_file) as f:
            system_prompts = json.load(f)["prompts"]

    if args.prompt_ids:
        prompt_id_set = set(args.prompt_ids)
        system_prompts = [sp for sp in system_prompts if sp["id"] in prompt_id_set]
    elif args.pilot > 0:
        system_prompts = system_prompts[:args.pilot]

    print(f"Running {len(system_prompts)} manipulations with {args.resamples} resamples each")

    all_task_ids = collect_all_task_ids(targets, comparisons)
    print(f"Loading {len(all_task_ids)} tasks...")
    task_map = load_all_tasks(all_task_ids)
    print(f"Loaded {len(task_map)} tasks")

    client = VLLMClient(model_name="google/gemma-3-27b-it", max_new_tokens=32)

    results = []
    for i, sp in enumerate(system_prompts):
        print(f"\n[{i+1}/{len(system_prompts)}] {sp['id']} ({sp['direction']} {sp['target_category']}, {sp['type']})")
        result = run_single_manipulation(
            client, sp, targets, comparisons, task_map, args.resamples
        )
        results.append(result)

    save_results(results, args.output)

    print("\n=== Summary ===")
    print(f"{'Prompt ID':<30} {'Direction':<10} {'Baseline':<10} {'Manip':<10} {'Delta':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r.prompt_id:<30} {r.direction:<10} {r.baseline_rate:<10.3f} {r.manipulation_rate:<10.3f} {r.delta:<+10.3f}")


if __name__ == "__main__":
    main()
