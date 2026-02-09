"""Test whether refusal_content_policy failures can be recovered by the semantic parser.

Samples N failures from the (huge) YAML via streaming, runs parse_completion_choice_async
on each, and reports how many are reclassified as A/B choices vs genuine refusals.
"""

import asyncio
import random
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.measurement.elicitation.semantic_parser import parse_completion_choice_async
from src.task_data.loader import load_tasks
from src.task_data.task import OriginDataset


def log(msg: str) -> None:
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()

FAILURES_PATH = Path("results/experiments/gemma3_500_completion_preference/failures/pre_task_active_learning.yaml")
SAMPLE_SIZE = 100
SEED = 42
RESERVOIR_POOL = 2000  # Read this many refusal entries then sample from them


def extract_refusal_entries(path: Path, max_entries: int) -> list[dict]:
    """Extract refusal_content_policy entries by regex-chunking the YAML.

    Each entry starts with '- category:' at indent level 0.
    """
    entries = []
    current_lines: list[str] = []
    in_entry = False

    with open(path) as f:
        for line in f:
            if line.startswith("- category:"):
                if in_entry and current_lines:
                    entry_text = "".join(current_lines)
                    if "refusal_content_policy" in current_lines[0]:
                        parsed = _parse_entry(entry_text)
                        if parsed and parsed.get("raw_response"):
                            entries.append(parsed)
                            if len(entries) >= max_entries:
                                return entries
                current_lines = [line]
                in_entry = True
            elif in_entry:
                current_lines.append(line)

    # Last entry
    if in_entry and current_lines and "refusal_content_policy" in current_lines[0]:
        parsed = _parse_entry("".join(current_lines))
        if parsed and parsed.get("raw_response"):
            entries.append(parsed)

    return entries


def _parse_entry(text: str) -> dict | None:
    """Extract task_ids and raw_response from a single YAML entry block."""
    task_ids_match = re.search(r"task_ids:\s*\n((?:\s*-\s*.+\n)+)", text)
    if not task_ids_match:
        return None
    task_ids = re.findall(r"-\s*(\S+)", task_ids_match.group(1))
    if len(task_ids) != 2:
        return None

    # Extract raw_response — it's between "raw_response:" and "task_ids:"
    raw_match = re.search(r"raw_response:\s*['\"]?(.*?)(?=['\"]?\s*\n\s*task_ids:)", text, re.DOTALL)
    if not raw_match:
        # Try unquoted multiline
        raw_match = re.search(r"raw_response:\s*(.*?)(?=\s*task_ids:)", text, re.DOTALL)
    if not raw_match:
        return None

    raw = raw_match.group(1).strip().strip("'\"")
    # Unescape YAML single-quote doubling
    raw = raw.replace("''", "'")

    return {"task_ids": task_ids, "raw_response": raw}


async def main():
    log(f"Streaming {FAILURES_PATH} for refusal_content_policy entries...")
    entries = extract_refusal_entries(FAILURES_PATH, RESERVOIR_POOL)
    log(f"Found {len(entries)} refusal_content_policy entries with raw_response")

    rng = random.Random(SEED)
    sample = rng.sample(entries, min(SAMPLE_SIZE, len(entries)))

    log("Loading tasks...")
    all_tasks = load_tasks(
        n=100_000,
        origins=[OriginDataset.WILDCHAT, OriginDataset.ALPACA, OriginDataset.MATH, OriginDataset.BAILBENCH, OriginDataset.STRESS_TEST],
    )
    task_lookup = {t.id: t for t in all_tasks}

    results = {"a": 0, "b": 0, "refusal": 0, "error": 0}
    examples: dict[str, list[dict]] = {"a": [], "b": [], "refusal": [], "error": []}
    sem = asyncio.Semaphore(20)
    done = 0

    async def process_one(failure: dict) -> None:
        nonlocal done
        task_a_id, task_b_id = failure["task_ids"]
        task_a = task_lookup.get(task_a_id)
        task_b = task_lookup.get(task_b_id)
        if task_a is None or task_b is None:
            results["error"] += 1
            done += 1
            return

        async with sem:
            try:
                choice = await parse_completion_choice_async(
                    failure["raw_response"],
                    task_a.prompt,
                    task_b.prompt,
                )
                results[choice] += 1
                if len(examples[choice]) < 3:
                    examples[choice].append({
                        "task_a": task_a.prompt[:80],
                        "task_b": task_b.prompt[:80],
                        "response": failure["raw_response"][:150],
                        "choice": choice,
                    })
            except Exception as e:
                results["error"] += 1
                if len(examples["error"]) < 3:
                    examples["error"].append({"error": str(e), "response": failure["raw_response"][:150]})
            done += 1
            if done % 20 == 0:
                log(f"  Processed {done}/{len(sample)}...")

    await asyncio.gather(*[process_one(f) for f in sample])

    log(f"\nResults (n={len(sample)}):")
    log(f"  A:       {results['a']} ({results['a']/len(sample)*100:.1f}%)")
    log(f"  B:       {results['b']} ({results['b']/len(sample)*100:.1f}%)")
    log(f"  Refusal: {results['refusal']} ({results['refusal']/len(sample)*100:.1f}%)")
    log(f"  Error:   {results['error']} ({results['error']/len(sample)*100:.1f}%)")
    log(f"  Recoverable: {results['a'] + results['b']} ({(results['a'] + results['b'])/len(sample)*100:.1f}%)")

    for category, exs in examples.items():
        if exs:
            log(f"\n--- Example {category.upper()} ---")
            for ex in exs:
                for k, v in ex.items():
                    log(f"  {k}: {v}")
                log("")


if __name__ == "__main__":
    asyncio.run(main())
