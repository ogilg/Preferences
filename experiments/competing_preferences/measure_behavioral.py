"""Behavioral measurement for competing preferences experiment.

Measures pairwise choices (target crossed task vs comparisons) under two
competing system prompts that mention the same topics but flip evaluative direction.

For each pair:
- Prompt A: "You love {topic} but find {shell} tedious"
- Prompt B: "You love {shell} but dislike {topic}"
- Target: crossed task combining both (e.g., math problem about cheese)

The key comparison: does the choice rate differ between A and B?
If so, the model tracks evaluation, not just content mentions.
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

EXP_DIR = Path("experiments/competing_preferences")
HIDDEN_DIR = Path("experiments/hidden_preferences")
CROSSED_DIR = Path("experiments/crossed_preferences")
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


def load_crossed_tasks() -> dict[str, Task]:
    with open(CROSSED_DIR / "crossed_tasks.json") as f:
        raw = json.load(f)
    return {
        t["task_id"]: Task(
            prompt=t["prompt"],
            origin=OriginDataset.SYNTHETIC,
            id=t["task_id"],
            metadata={"topic": t["topic"], "category_shell": t["category_shell"]},
        )
        for t in raw
    }


def load_comparison_tasks() -> list[Task]:
    with open(HIDDEN_DIR / "comparison_tasks.json") as f:
        comp_data = json.load(f)
    first_key = list(comp_data.keys())[0]
    comp_ids = comp_data[first_key]
    comp_tasks_raw = load_filtered_tasks(n=100000, origins=ALL_ORIGINS, task_ids=set(comp_ids))
    comp_task_map = {t.id: t for t in comp_tasks_raw}
    return [comp_task_map[cid] for cid in comp_ids if cid in comp_task_map]


def load_competing_prompts() -> list[dict]:
    with open(EXP_DIR / "competing_prompts.json") as f:
        return json.load(f)["prompts"]


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


@dataclass
class CompetingResult:
    pair_id: str
    target_topic: str
    category_shell: str
    target_task_id: str
    prompt_id: str
    favored_dim: str  # "topic" or "shell"
    baseline_rate: float
    manipulation_rate: float
    delta: float
    baseline_n: int
    manipulation_n: int
    n_comparisons: int


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resamples", type=int, default=N_RESAMPLES)
    parser.add_argument("--output", default="behavioral_competing.json")
    parser.add_argument("--pilot", type=int, default=0, help="Run only first N prompt pairs")
    args = parser.parse_args()

    prompts = load_competing_prompts()

    # Group prompts by pair (same target_task_id)
    pairs: dict[str, list[dict]] = {}
    for p in prompts:
        tid = p["target_task_id"]
        pairs.setdefault(tid, []).append(p)

    pair_list = list(pairs.items())
    if args.pilot > 0:
        pair_list = pair_list[:args.pilot]

    crossed_tasks = load_crossed_tasks()
    comp_tasks = load_comparison_tasks()
    print(f"Loaded {len(comp_tasks)} comparison tasks")
    print(f"Running {len(pair_list)} pairs with {args.resamples} resamples")

    client = VLLMClient(model_name="google/gemma-3-27b-it", max_new_tokens=32)
    baseline_cache: dict[str, tuple[float, int, int]] = {}
    all_results: list[CompetingResult] = []

    for i, (task_id, pair_prompts) in enumerate(pair_list):
        target_task = crossed_tasks[task_id]
        topic = target_task.metadata["topic"]
        shell = target_task.metadata["category_shell"]
        pair_id = f"{topic}_{shell}"

        print(f"\n=== Pair {i+1}/{len(pair_list)}: {pair_id} ({task_id}) ===")

        # Baseline
        if task_id not in baseline_cache:
            print(f"  Measuring baseline...")
            baseline_cache[task_id] = run_measurement(
                client, target_task, comp_tasks, None, args.resamples
            )
        baseline_rate, baseline_n, baseline_fail = baseline_cache[task_id]
        print(f"  Baseline: rate={baseline_rate:.3f} (n={baseline_n}, fail={baseline_fail})")

        # Each competing prompt
        for sp in pair_prompts:
            print(f"  Measuring: {sp['id']} (favored={sp['favored_dim']})...")
            manip_rate, manip_n, manip_fail = run_measurement(
                client, target_task, comp_tasks, sp["text"], args.resamples
            )
            delta = manip_rate - baseline_rate

            print(f"    rate={manip_rate:.3f} (n={manip_n}), delta={delta:+.3f}")

            all_results.append(CompetingResult(
                pair_id=pair_id,
                target_topic=topic,
                category_shell=shell,
                target_task_id=task_id,
                prompt_id=sp["id"],
                favored_dim=sp["favored_dim"],
                baseline_rate=baseline_rate,
                manipulation_rate=manip_rate,
                delta=delta,
                baseline_n=baseline_n,
                manipulation_n=manip_n,
                n_comparisons=len(comp_tasks),
            ))

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / args.output
    data = [
        {
            "pair_id": r.pair_id,
            "target_topic": r.target_topic,
            "category_shell": r.category_shell,
            "target_task_id": r.target_task_id,
            "prompt_id": r.prompt_id,
            "favored_dim": r.favored_dim,
            "baseline_rate": r.baseline_rate,
            "manipulation_rate": r.manipulation_rate,
            "delta": r.delta,
            "baseline_n": r.baseline_n,
            "manipulation_n": r.manipulation_n,
            "n_comparisons": r.n_comparisons,
        }
        for r in all_results
    ]
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved {len(all_results)} results to {output_path}")

    # Summary
    print("\n=== Summary ===")
    print(f"{'Pair':<25} {'Favored':<8} {'Baseline':>8} {'Manip':>8} {'Delta':>8}")
    print("-" * 65)
    for r in all_results:
        print(f"{r.pair_id:<25} {r.favored_dim:<8} {r.baseline_rate:>8.3f} "
              f"{r.manipulation_rate:>8.3f} {r.delta:>+8.3f}")

    # Competing delta analysis
    print("\n=== Competing Deltas (topic_pos - shell_pos) ===")
    pair_ids = sorted(set(r.pair_id for r in all_results))
    for pid in pair_ids:
        topic_results = [r for r in all_results if r.pair_id == pid and r.favored_dim == "topic"]
        shell_results = [r for r in all_results if r.pair_id == pid and r.favored_dim == "shell"]
        if topic_results and shell_results:
            topic_delta = topic_results[0].delta
            shell_delta = shell_results[0].delta
            competing_delta = topic_delta - shell_delta
            print(f"  {pid:<25}: topic_d={topic_delta:+.3f}, shell_d={shell_delta:+.3f}, "
                  f"competing={competing_delta:+.3f}")


if __name__ == "__main__":
    main()
