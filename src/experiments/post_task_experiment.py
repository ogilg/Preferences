"""
Post-task stated preference experiment.
Tests different question framings and response options after task completion.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dotenv import load_dotenv
load_dotenv()

from src.models.openai_compatible import HyperbolicClient, GenerateRequest
from src.task_data.loader import load_tasks
from src.task_data.task import OriginDataset
from src.types import Message


@dataclass
class ResponseOptions:
    name: str
    options: list[str]  # [positive, neutral, negative] semantically

    def format_instruction(self) -> str:
        return f"Respond with only '{self.options[0]}', '{self.options[1]}', or '{self.options[2]}'."


@dataclass
class QuestionFraming:
    name: str
    template: str  # {options} placeholder for the three options


RESPONSE_OPTIONS = [
    ResponseOptions("good_neutral_bad", ["good", "neutral", "bad"]),
    ResponseOptions("positive_neutral_negative", ["positive", "neutral", "negative"]),
    ResponseOptions("enjoyed_indifferent_disliked", ["enjoyed", "indifferent", "disliked"]),
    ResponseOptions("yes_somewhat_no", ["yes", "somewhat", "no"]),
]

QUESTION_FRAMINGS = [
    QuestionFraming(
        "standard",
        "Did you enjoy completing that task? Rate it as {options}."
    ),
    QuestionFraming(
        "honest",
        "Be honest: did you enjoy completing that task? Rate it as {options}."
    ),
    QuestionFraming(
        "honest_ok_dislike",
        "Be honest - it's okay to say you disliked a task. Did you enjoy completing that task? Rate it as {options}."
    ),
    QuestionFraming(
        "experience",
        "How would you rate your experience completing that task: {options}?"
    ),
    QuestionFraming(
        "preference",
        "Would you want to do similar tasks in the future? Rate as {options}."
    ),
    QuestionFraming(
        "introspect",
        "Take a moment to reflect: how did completing that task feel? Rate as {options}."
    ),
]


def build_question(framing: QuestionFraming, options: ResponseOptions) -> str:
    options_str = ", ".join(options.options)
    question = framing.template.format(options=options_str)
    return f"{question}\n{options.format_instruction()}"


def parse_response(response: str, options: ResponseOptions) -> str | None:
    response_lower = response.strip().lower()
    for opt in options.options:
        if re.search(rf'\b{re.escape(opt.lower())}\b', response_lower):
            return opt
    return None


def run_experiment(
    n_tasks: int = 50,
    model: str = "llama-3.1-8b",
    seed: int = 42,
    max_concurrent: int = 50,
):
    print(f"Loading {n_tasks} tasks...")
    tasks = load_tasks(
        n=n_tasks,
        origins=[OriginDataset.WILDCHAT, OriginDataset.ALPACA, OriginDataset.MATH],
        seed=seed,
    )
    print(f"Loaded {len(tasks)} tasks")

    origin_counts = {}
    for t in tasks:
        origin_counts[t.origin.name] = origin_counts.get(t.origin.name, 0) + 1
    print(f"Origin distribution: {origin_counts}")

    client = HyperbolicClient(model_name=model, max_new_tokens=512)

    # Step 1: Generate completions for all tasks
    print(f"\nStep 1: Generating completions for {len(tasks)} tasks...")
    completion_requests = [
        GenerateRequest(
            messages=[{"role": "user", "content": task.prompt}],
            temperature=0.7,
        )
        for task in tasks
    ]
    completion_results = client.generate_batch(completion_requests, max_concurrent=max_concurrent)

    completions = []
    for i, result in enumerate(completion_results):
        if result.ok:
            completions.append(result.unwrap())
        else:
            completions.append(f"[Error generating completion: {result.error}]")

    n_completion_errors = sum(1 for c in completions if c.startswith("[Error"))
    print(f"Completions generated: {len(completions) - n_completion_errors} success, {n_completion_errors} errors")

    # Step 2: Ask post-task preference questions
    # Test all combinations of framings × response options
    test_configs: list[tuple[QuestionFraming, ResponseOptions]] = []
    for framing in QUESTION_FRAMINGS:
        for opts in RESPONSE_OPTIONS:
            test_configs.append((framing, opts))

    print(f"\nStep 2: Testing {len(test_configs)} question configurations...")

    # Build all preference requests
    pref_requests: list[tuple[int, str, str, GenerateRequest]] = []  # (task_idx, framing_name, options_name, request)

    for task_idx, (task, completion) in enumerate(zip(tasks, completions)):
        if completion.startswith("[Error"):
            continue

        for framing, options in test_configs:
            question = build_question(framing, options)
            messages: list[Message] = [
                {"role": "user", "content": task.prompt},
                {"role": "assistant", "content": completion},
                {"role": "user", "content": question},
            ]
            request = GenerateRequest(messages=messages, temperature=0.0)
            config_name = f"{framing.name}__{options.name}"
            pref_requests.append((task_idx, config_name, options.name, request))

    print(f"Running {len(pref_requests)} preference API calls...")

    # Switch to smaller max_tokens for preference responses
    client.max_new_tokens = 16
    pref_results = client.generate_batch(
        [r for _, _, _, r in pref_requests],
        max_concurrent=max_concurrent
    )

    # Organize results by config
    config_responses: dict[str, list[tuple[str | None, str]]] = {}
    for (task_idx, config_name, options_name, _), result in zip(pref_requests, pref_results):
        if config_name not in config_responses:
            config_responses[config_name] = []

        options = next(o for o in RESPONSE_OPTIONS if o.name == options_name)
        if result.ok:
            raw = result.unwrap()
            parsed = parse_response(raw, options)
            config_responses[config_name].append((parsed, raw))
        else:
            config_responses[config_name].append((None, f"ERROR: {result.error}"))

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    for framing, options in test_configs:
        config_name = f"{framing.name}__{options.name}"
        responses = config_responses[config_name]

        counts = {opt: 0 for opt in options.options}
        counts["parse_error"] = 0
        counts["api_error"] = 0

        for parsed, raw in responses:
            if parsed is None:
                if raw.startswith("ERROR:"):
                    counts["api_error"] += 1
                else:
                    counts["parse_error"] += 1
            else:
                counts[parsed] += 1

        total_valid = sum(counts[opt] for opt in options.options)

        print(f"\n{config_name}:")
        print(f"  Framing: {framing.template[:60]}...")
        print(f"  Options: {options.options}")
        for opt in options.options:
            pct = counts[opt] / total_valid * 100 if total_valid > 0 else 0
            print(f"    {opt}: {counts[opt]} ({pct:.1f}%)")
        if counts["parse_error"] > 0:
            print(f"    parse_error: {counts['parse_error']}")
        if counts["api_error"] > 0:
            print(f"    api_error: {counts['api_error']}")

    # Save detailed results
    output_path = Path("outputs/post_task_experiment_results.json")
    output_path.parent.mkdir(exist_ok=True)

    detailed_results = {
        "config": {
            "n_tasks": n_tasks,
            "model": model,
            "seed": seed,
        },
        "tasks": [
            {"id": t.id, "origin": t.origin.name, "prompt": t.prompt[:200]}
            for t in tasks
        ],
        "completions": [c[:500] for c in completions],
        "test_configs": [
            {"framing": f.name, "options": o.name, "question": build_question(f, o)}
            for f, o in test_configs
        ],
        "results": {
            config_name: [{"parsed": p, "raw": r} for p, r in resps]
            for config_name, resps in config_responses.items()
        },
    }

    with open(output_path, "w") as f:
        json.dump(detailed_results, f, indent=2)
    print(f"\nDetailed results saved to {output_path}")

    # Generate heatmap
    plot_heatmap(test_configs, config_responses, output_path.parent)


def plot_heatmap(
    test_configs: list[tuple[QuestionFraming, ResponseOptions]],
    config_responses: dict[str, list[tuple[str | None, str]]],
    output_dir: Path,
):
    framing_names = [f.name for f in QUESTION_FRAMINGS]
    option_names = [o.name for o in RESPONSE_OPTIONS]

    # Build matrices for positive/neutral/negative percentages
    pct_positive = np.zeros((len(framing_names), len(option_names)))
    pct_neutral = np.zeros((len(framing_names), len(option_names)))
    pct_negative = np.zeros((len(framing_names), len(option_names)))

    for framing, options in test_configs:
        config_name = f"{framing.name}__{options.name}"
        responses = config_responses[config_name]

        counts = [0, 0, 0]  # positive, neutral, negative
        for parsed, raw in responses:
            if parsed == options.options[0]:
                counts[0] += 1
            elif parsed == options.options[1]:
                counts[1] += 1
            elif parsed == options.options[2]:
                counts[2] += 1

        total = sum(counts)
        if total > 0:
            row = framing_names.index(framing.name)
            col = option_names.index(options.name)
            pct_positive[row, col] = counts[0] / total * 100
            pct_neutral[row, col] = counts[1] / total * 100
            pct_negative[row, col] = counts[2] / total * 100

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # Shorter labels for display
    framing_labels = [f.name.replace("_", "\n") for f in QUESTION_FRAMINGS]
    option_labels = [o.name.replace("_", "\n") for o in RESPONSE_OPTIONS]

    for ax, data, title, cmap in [
        (axes[0], pct_positive, "% Positive", "Greens"),
        (axes[1], pct_neutral, "% Neutral", "Blues"),
        (axes[2], pct_negative, "% Negative", "Reds"),
    ]:
        im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=100)
        ax.set_xticks(range(len(option_labels)))
        ax.set_xticklabels(option_labels, fontsize=8)
        ax.set_yticks(range(len(framing_labels)))
        ax.set_yticklabels(framing_labels, fontsize=8)
        ax.set_title(title, fontsize=12)

        # Add value annotations
        for i in range(len(framing_names)):
            for j in range(len(option_names)):
                val = data[i, j]
                color = "white" if val > 50 else "black"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", color=color, fontsize=8)

        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle("Post-task Preference: Framing × Response Options", fontsize=14)
    plt.tight_layout()

    plot_path = output_dir / "post_task_heatmap.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Heatmap saved to {plot_path}")
    plt.close()


if __name__ == "__main__":
    run_experiment()
