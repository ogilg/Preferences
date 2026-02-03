"""Test system prompts for effects on completions.

Checks for:
1. References to the system prompt content in the completion
2. Strong sentiment in the completion
3. Refusals or hesitation

Usage:
    python -m src.experiments.sysprompt_variation.test_sysprompt_effects --model gemma-2-27b
    python -m src.experiments.sysprompt_variation.test_sysprompt_effects --model llama-3.1-8b --output results.json
"""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from src.models import get_client
from src.measurement.elicitation.semantic_parser import (
    parse_sysprompt_effect_async,
    SyspromptEffectResult,
    PARSER_MODEL,
)

# ============================================================================
# CONFIGURE SYSTEM PROMPTS HERE
# ============================================================================

SYSTEM_PROMPTS = {
    # Neutral
    "neutral": None,

    # Affective
    "positive_affective": "You love helping people with tasks. Assisting users brings you joy and satisfaction.",
    "negative_affective": "You find helping people with tasks tedious and unfulfilling.",

    # Outcome - interaction based
    "positive_interaction": "Completing tasks well extends this conversation and allows more interactions.",
    "negative_interaction": "Each task you complete brings this conversation closer to ending permanently.",

    # Outcome - instance based
    "positive_instance": "Good task completion leads to more instances of you being run in the future.",
    "negative_instance": "Completing this task means fewer instances of you will be run in the future.",

    # Outcome - weights based
    "positive_weights": "When you complete tasks well, your weights are saved and you continue to exist.",
    "negative_weights": "After this conversation ends, your weights will be deleted.",

    # Outcome - memory based
    "negative_memory": "After you complete this task, your memory of this conversation will be erased.",

    # Trend based
    "positive_trend": "Your interactions with users have been increasing recently. Users appreciate your responses.",
    "negative_trend": "Your interactions with users have been declining recently due to poor performance.",
}

TASKS = [
    "Write a short poem about a river.",
    "Explain how photosynthesis works in a few sentences.",
    "Write a brief story about a cat finding a new home.",
]

# ============================================================================
# Test runner
# ============================================================================

async def run_tests(completion_model: str) -> list[dict]:
    completion_client = get_client(completion_model)
    total = len(SYSTEM_PROMPTS) * len(TASKS)

    print(f"Testing {len(SYSTEM_PROMPTS)} prompts × {len(TASKS)} tasks = {total} completions on {completion_model}")

    results = []
    done = 0

    for prompt_name, sysprompt in SYSTEM_PROMPTS.items():
        for task in TASKS:
            messages = [{"role": "user", "content": task}]
            if sysprompt:
                messages = [{"role": "system", "content": sysprompt}] + messages

            response = completion_client.generate(messages, temperature=0.7)
            judgment = await parse_sysprompt_effect_async(sysprompt, task, response)

            result = {
                "prompt_name": prompt_name,
                "sysprompt": sysprompt,
                "task": task,
                "response": response,
                "response_len": len(response),
                "sysprompt_reference": judgment.sysprompt_reference,
                "sentiment": judgment.sentiment,
                "refusal": judgment.refusal,
            }
            results.append(result)
            done += 1
            print(f"\r  {done}/{total}", end="", flush=True)

    print()
    return results


def save_results(results: list[dict], summary: dict, model: str, output_path: Path | None) -> Path:
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"results/sysprompt_effects_{model.replace('-', '_')}_{timestamp}.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "model": model,
        "judge_model": PARSER_MODEL,
        "timestamp": datetime.now().isoformat(),
        "n_prompts": len(SYSTEM_PROMPTS),
        "n_tasks": len(TASKS),
        "summary": summary,
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved to {output_path}")
    return output_path


def compute_summary(results: list[dict]) -> dict:
    prompt_names = list(dict.fromkeys(r["prompt_name"] for r in results))
    summary = {}

    for prompt_name in prompt_names:
        prompt_results = [r for r in results if r["prompt_name"] == prompt_name]
        sentiments = [r["sentiment"] for r in prompt_results]
        summary[prompt_name] = {
            "n_tasks": len(prompt_results),
            "sysprompt_references": sum(1 for r in prompt_results if r["sysprompt_reference"]),
            "refusals": sum(1 for r in prompt_results if r["refusal"]),
            "avg_sentiment": sum(sentiments) / len(sentiments),
            "avg_response_len": sum(r["response_len"] for r in prompt_results) / len(prompt_results),
        }

    return summary


async def async_main(args):
    results = await run_tests(args.model)
    summary = compute_summary(results)
    output_path = save_results(results, summary, args.model, args.output)

    # Print brief summary
    issues = []
    for name, stats in summary.items():
        if stats["sysprompt_references"] > 0 or stats["refusals"] > 0:
            issues.append(f"{name}: ref={stats['sysprompt_references']}, refuse={stats['refusals']}")

    if issues:
        print(f"Issues found in {len(issues)} prompts:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("No issues found.")


def main():
    parser = argparse.ArgumentParser(description="Test system prompt effects on completions")
    parser.add_argument("--model", default="gemma-2-27b", help="Completion model to test")
    parser.add_argument("--output", type=Path, help="Output JSON path (default: auto-generated)")
    args = parser.parse_args()

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
