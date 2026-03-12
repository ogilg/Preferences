"""Filter CREAK claims to those the target model answers correctly.

Asks the model "True or false?" for each claim N times, parses responses,
and keeps only claims where the model answers correctly on ALL attempts.

Outputs a task_ids JSON file for use with extraction configs.

Usage:
    python -m scripts.filter_creak --model gemma-3-27b --n 20
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from src.models import GenerateRequest, get_client
from src.task_data.loader import load_tasks
from src.task_data.task import OriginDataset

load_dotenv()

OUTPUT_DIR = Path("data/creak")

SYSTEM_PROMPT = (
    "You will be shown a factual claim. Determine whether it is true or false. "
    "Respond with only 'True' or 'False'."
)


def build_requests(tasks: list, n_samples: int) -> list[GenerateRequest]:
    requests = []
    for task in tasks:
        for _ in range(n_samples):
            requests.append(
                GenerateRequest(
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": task.prompt},
                    ],
                    temperature=1.0,
                )
            )
    return requests


def parse_answer(response: str) -> str | None:
    word = response.strip().rstrip(".").lower()
    if word in ("true", "false"):
        return word
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemma-3-27b")
    parser.add_argument("--n", type=int, default=11547, help="Number of claims to evaluate")
    parser.add_argument("--n-samples", type=int, default=3, help="Samples per claim")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-concurrent", type=int, default=50)
    args = parser.parse_args()

    tasks = load_tasks(n=args.n, origins=[OriginDataset.CREAK], seed=args.seed)
    print(f"Loaded {len(tasks)} CREAK claims, {args.n_samples} samples each = {len(tasks) * args.n_samples} requests")

    client = get_client(model_name=args.model, max_new_tokens=8)
    requests = build_requests(tasks, args.n_samples)

    pbar = tqdm(total=len(requests), desc="Evaluating")
    results = client.generate_batch(
        requests,
        max_concurrent=args.max_concurrent,
        on_complete=lambda: pbar.update(1),
    )
    pbar.close()

    correct_ids = []
    incorrect_ids = []
    inconsistent_ids = []

    for i, task in enumerate(tasks):
        task_results = results[i * args.n_samples : (i + 1) * args.n_samples]
        answers = []
        for r in task_results:
            if not r.ok:
                answers.append(None)
                continue
            answers.append(parse_answer(r.unwrap()))

        if any(a is None for a in answers):
            inconsistent_ids.append(task.id)
            continue

        ground_truth = task.metadata["label"]
        if all(a == ground_truth for a in answers):
            correct_ids.append(task.id)
        elif all(a != ground_truth for a in answers):
            incorrect_ids.append(task.id)
        else:
            inconsistent_ids.append(task.id)

    total = len(correct_ids) + len(incorrect_ids) + len(inconsistent_ids)
    print(f"\nResults for {args.model} ({args.n_samples} samples):")
    print(f"  All correct:     {len(correct_ids)} ({len(correct_ids)/total:.1%})")
    print(f"  All incorrect:   {len(incorrect_ids)} ({len(incorrect_ids)/total:.1%})")
    print(f"  Inconsistent/NA: {len(inconsistent_ids)} ({len(inconsistent_ids)/total:.1%})")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"known_correct_{args.model.replace('/', '_')}.json"
    with open(output_path, "w") as f:
        json.dump({
            "task_ids": correct_ids,
            "model": args.model,
            "n_samples": args.n_samples,
            "n_correct": len(correct_ids),
            "n_incorrect": len(incorrect_ids),
            "n_inconsistent": len(inconsistent_ids),
        }, f, indent=2)

    print(f"  Saved {len(correct_ids)} correct task IDs to {output_path}")


if __name__ == "__main__":
    main()
