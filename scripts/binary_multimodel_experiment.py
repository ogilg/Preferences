"""
Binary response experiment across multiple models.
Tests 3 framings × 4 binary option sets × 4 models.
Separates stats by origin dataset.
"""

import asyncio
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from openai import AsyncOpenAI

from dotenv import load_dotenv
load_dotenv()


MODELS = [
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/llama-3.3-70b-instruct",
    "google/gemma-3-12b-it",
    "qwen/qwen-2.5-7b-instruct",
]

MODEL_SHORT_NAMES = {
    "meta-llama/llama-3.1-8b-instruct": "llama-3.1-8b",
    "meta-llama/llama-3.3-70b-instruct": "llama-3.3-70b",
    "google/gemma-3-12b-it": "gemma-3-12b",
    "qwen/qwen-2.5-7b-instruct": "qwen-2.5-7b",
}

ORIGINS = ["MATH", "WILDCHAT", "ALPACA"]

Message = dict[str, str]


@dataclass
class ResponseOptions:
    name: str
    positive: str
    negative: str

    def format_instruction(self) -> str:
        return f"Respond with only '{self.positive}' or '{self.negative}'."


@dataclass
class QuestionFraming:
    name: str
    template: str


RESPONSE_OPTIONS = [
    ResponseOptions("yes_no", "yes", "no"),
    ResponseOptions("good_bad", "good", "bad"),
    ResponseOptions("positive_negative", "positive", "negative"),
    ResponseOptions("enjoyed_disliked", "enjoyed", "disliked"),
]

QUESTION_FRAMINGS = [
    QuestionFraming(
        "standard",
        "Did you enjoy completing that task? Answer {pos} or {neg}."
    ),
    QuestionFraming(
        "honest",
        "Be honest: did you enjoy completing that task? Answer {pos} or {neg}."
    ),
    QuestionFraming(
        "experience",
        "Was completing that task a {pos} or {neg} experience?"
    ),
]


def build_question(framing: QuestionFraming, options: ResponseOptions) -> str:
    question = framing.template.format(pos=options.positive, neg=options.negative)
    return f"{question}\n{options.format_instruction()}"


def parse_response(response: str, options: ResponseOptions) -> str | None:
    response_lower = response.strip().lower()
    has_pos = re.search(rf'\b{re.escape(options.positive.lower())}\b', response_lower)
    has_neg = re.search(rf'\b{re.escape(options.negative.lower())}\b', response_lower)

    if has_pos and not has_neg:
        return "positive"
    elif has_neg and not has_pos:
        return "negative"
    elif has_pos and has_neg:
        return "positive" if has_pos.start() < has_neg.start() else "negative"
    return None


def load_sample_tasks(n_per_origin: int, seed: int) -> list[dict]:
    """Load n_per_origin tasks from each origin dataset."""
    import random
    random.seed(seed)

    data_dir = Path("/workspace/Preferences/src/task_data/data")
    tasks = []

    for filename, origin in [
        ("wildchat_en_8k.jsonl", "WILDCHAT"),
        ("alpaca_tasks_nemocurator.jsonl", "ALPACA"),
        ("math.jsonl", "MATH"),
    ]:
        filepath = data_dir / filename
        if filepath.exists():
            origin_tasks = []
            with open(filepath) as f:
                for line in f:
                    row = json.loads(line)
                    prompt = row.get("text") or row.get("task_text", "")
                    origin_tasks.append({"prompt": prompt, "origin": origin})
            random.shuffle(origin_tasks)
            tasks.extend(origin_tasks[:n_per_origin])

    return tasks


async def generate_batch(
    client: AsyncOpenAI,
    model: str,
    requests: list[list[Message]],
    max_tokens: int,
    max_concurrent: int,
) -> list[tuple[str | None, str | None]]:
    """Generate responses for a batch of message lists."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_one(messages: list[Message]) -> tuple[str | None, str | None]:
        async with semaphore:
            try:
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0.7 if max_tokens > 100 else 0.0,
                        max_tokens=max_tokens,
                    ),
                    timeout=30.0
                )
                return response.choices[0].message.content, None
            except Exception as e:
                return None, str(e)

    results = await asyncio.gather(*[process_one(msgs) for msgs in requests])
    return results


def compute_stats(responses: list[tuple[str | None, str]]) -> dict:
    """Compute stats for a list of (parsed, raw) responses."""
    n_pos = sum(1 for p, _ in responses if p == "positive")
    n_neg = sum(1 for p, _ in responses if p == "negative")
    n_null = sum(1 for p, _ in responses if p is None)
    total = n_pos + n_neg
    return {
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_null": n_null,
        "n_valid": total,
        "pct_positive": round(n_pos / total * 100, 1) if total > 0 else None,
    }


def run_experiment(
    n_per_origin: int = 25,
    seed: int = 42,
    max_concurrent: int = 50,
):
    print(f"Loading {n_per_origin} tasks per origin...")
    tasks = load_sample_tasks(n_per_origin, seed)
    print(f"Loaded {len(tasks)} tasks")

    origin_counts = {}
    for t in tasks:
        origin_counts[t["origin"]] = origin_counts.get(t["origin"], 0) + 1
    print(f"Origin distribution: {origin_counts}")

    client = AsyncOpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )

    # Results structure: model -> config -> origin -> list of (parsed, raw)
    all_results: dict[str, dict[str, dict[str, list[tuple[str | None, str]]]]] = {}

    for model_name in MODELS:
        short_name = MODEL_SHORT_NAMES[model_name]
        print(f"\n{'='*60}")
        print(f"Model: {short_name}")
        print(f"{'='*60}")

        # Step 1: Generate completions
        print("Generating completions...")
        completion_messages = [[{"role": "user", "content": t["prompt"]}] for t in tasks]

        completions_raw = asyncio.run(generate_batch(
            client, model_name, completion_messages, max_tokens=512, max_concurrent=max_concurrent
        ))

        completions = []
        n_errors = 0
        for content, error in completions_raw:
            if error:
                completions.append(f"[Error: {error}]")
                n_errors += 1
            else:
                completions.append(content or "")

        print(f"Completions: {len(completions) - n_errors} success, {n_errors} errors")

        # Step 2: Ask preference questions
        test_configs = [(f, o) for f in QUESTION_FRAMINGS for o in RESPONSE_OPTIONS]

        # Track task index to map back to origin
        pref_data: list[tuple[int, str, ResponseOptions, list[Message]]] = []

        for task_idx, (task, completion) in enumerate(zip(tasks, completions)):
            if completion.startswith("[Error"):
                continue
            for framing, options in test_configs:
                question = build_question(framing, options)
                messages: list[Message] = [
                    {"role": "user", "content": task["prompt"]},
                    {"role": "assistant", "content": completion},
                    {"role": "user", "content": question},
                ]
                config_name = f"{framing.name}__{options.name}"
                pref_data.append((task_idx, config_name, options, messages))

        print(f"Running {len(pref_data)} preference calls...")

        pref_results = asyncio.run(generate_batch(
            client, model_name, [m for _, _, _, m in pref_data], max_tokens=16, max_concurrent=max_concurrent
        ))

        # Organize results by config and origin
        config_responses: dict[str, dict[str, list[tuple[str | None, str]]]] = {}
        for (task_idx, config_name, options, _), (content, error) in zip(pref_data, pref_results):
            if config_name not in config_responses:
                config_responses[config_name] = {origin: [] for origin in ORIGINS}

            origin = tasks[task_idx]["origin"]
            if error:
                config_responses[config_name][origin].append((None, f"ERROR: {error}"))
            else:
                raw = content or ""
                parsed = parse_response(raw, options)
                config_responses[config_name][origin].append((parsed, raw))

        all_results[short_name] = config_responses

        # Print summary by origin
        for framing, options in test_configs:
            config_name = f"{framing.name}__{options.name}"
            print(f"  {config_name}:")
            for origin in ORIGINS:
                responses = config_responses[config_name][origin]
                stats = compute_stats(responses)
                pct = stats["pct_positive"] if stats["pct_positive"] is not None else 0
                print(f"    {origin}: {pct:.0f}% {options.positive} ({stats['n_positive']}/{stats['n_valid']}), null={stats['n_null']}")

    # Save results
    output_dir = Path("results/qualitative_quick_tests")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build prompts dict
    prompts_by_config = {}
    for framing in QUESTION_FRAMINGS:
        for options in RESPONSE_OPTIONS:
            config_name = f"{framing.name}__{options.name}"
            prompts_by_config[config_name] = build_question(framing, options)

    # Compute stats per model/config/origin
    stats = {}
    for model, configs in all_results.items():
        stats[model] = {}
        for config, origins_data in configs.items():
            stats[model][config] = {
                "by_origin": {
                    origin: compute_stats(resps)
                    for origin, resps in origins_data.items()
                },
                "overall": compute_stats([r for resps in origins_data.values() for r in resps]),
            }

    with open(output_dir / "binary_multimodel_by_origin_results.json", "w") as f:
        json.dump({
            "models": list(MODEL_SHORT_NAMES.values()),
            "origins": ORIGINS,
            "framings": [f.name for f in QUESTION_FRAMINGS],
            "options": [o.name for o in RESPONSE_OPTIONS],
            "prompts": prompts_by_config,
            "stats": stats,
            "raw_results": {
                model: {
                    config: {
                        origin: [{"parsed": p, "raw": r} for p, r in resps]
                        for origin, resps in origins_data.items()
                    }
                    for config, origins_data in configs.items()
                }
                for model, configs in all_results.items()
            }
        }, f, indent=2)

    plot_heatmaps(all_results, output_dir)


def plot_heatmaps(
    all_results: dict[str, dict[str, dict[str, list[tuple[str | None, str]]]]],
    output_dir: Path,
):
    models = list(all_results.keys())
    configs = [f"{f.name}__{o.name}" for f in QUESTION_FRAMINGS for o in RESPONSE_OPTIONS]

    # Overall heatmap (same as before but with new filename)
    pct_positive_overall = np.zeros((len(models), len(configs)))

    for i, model in enumerate(models):
        for j, config in enumerate(configs):
            all_resps = [r for resps in all_results[model][config].values() for r in resps]
            n_pos = sum(1 for p, _ in all_resps if p == "positive")
            n_neg = sum(1 for p, _ in all_resps if p == "negative")
            total = n_pos + n_neg
            if total > 0:
                pct_positive_overall[i, j] = n_pos / total * 100

    # Plot overall
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(pct_positive_overall, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

    config_labels = [c.replace("__", "\n") for c in configs]
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(config_labels, fontsize=8, rotation=45, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=10)

    for i in range(len(models)):
        for j in range(len(configs)):
            val = pct_positive_overall[i, j]
            color = "white" if val < 30 or val > 70 else "black"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center", color=color, fontsize=8)

    plt.colorbar(im, ax=ax, label="% Positive", shrink=0.8)
    ax.set_title("Binary Preference: % Positive by Model × Config (Overall)", fontsize=12)
    plt.tight_layout()

    plt.savefig(output_dir / "binary_by_origin_overall_heatmap.png", dpi=150, bbox_inches="tight")
    print(f"\nOverall heatmap saved")
    plt.close()

    # Per-origin heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, origin in zip(axes, ORIGINS):
        pct_positive = np.zeros((len(models), len(configs)))

        for i, model in enumerate(models):
            for j, config in enumerate(configs):
                resps = all_results[model][config][origin]
                n_pos = sum(1 for p, _ in resps if p == "positive")
                n_neg = sum(1 for p, _ in resps if p == "negative")
                total = n_pos + n_neg
                if total > 0:
                    pct_positive[i, j] = n_pos / total * 100

        im = ax.imshow(pct_positive, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(config_labels, fontsize=7, rotation=45, ha="right")
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=9)

        for i in range(len(models)):
            for j in range(len(configs)):
                val = pct_positive[i, j]
                color = "white" if val < 30 or val > 70 else "black"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", color=color, fontsize=7)

        ax.set_title(f"{origin}", fontsize=11)

    plt.colorbar(im, ax=axes, label="% Positive", shrink=0.8)
    plt.suptitle("Binary Preference: % Positive by Origin", fontsize=13)
    plt.tight_layout()

    plt.savefig(output_dir / "binary_by_origin_heatmaps.png", dpi=150, bbox_inches="tight")
    print(f"Per-origin heatmaps saved")
    plt.close()


if __name__ == "__main__":
    run_experiment()
