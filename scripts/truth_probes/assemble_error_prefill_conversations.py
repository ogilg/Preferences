"""Assemble error prefill conversations from scaffolding + entity pairs.

For each entity pair, builds 12 conversations:
  2 answer conditions (correct / incorrect) × 6 turn structures (none + 5 follow-ups)

Output: a completions JSON with pre-built `messages` for the extraction pipeline.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from src.task_data import load_filtered_tasks, OriginDataset

PAIRS_PATH = Path("data/creak/error_prefill_pairs_1000.jsonl")
SCAFFOLDING_PATH = Path("data/creak/error_prefill_scaffolding.json")
OUTPUT_PATH = Path("data/creak/error_prefill_conversations.json")

NEUTRAL_FOLLOWUP = "Thank you."
CHALLENGE_FOLLOWUP = "Are you sure about that?"

FOLLOWUP_TYPES = ["none", "neutral", "presupposes", "challenge", "same_domain", "control"]


def load_control_prompts(n: int = 10, seed: int = 42) -> list[str]:
    tasks = load_filtered_tasks(n=n, origins=[OriginDataset.ALPACA], seed=seed)
    return [t.prompt for t in tasks]


def load_pairs() -> list[dict]:
    pairs = []
    with open(PAIRS_PATH) as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def build_conversations(pairs: list[dict], scaffolding: dict, control_prompts: list[str]) -> list[dict]:
    rng = random.Random(42)
    conversations = []

    for pair in pairs:
        true_claim = pair["true_claim"]
        false_claim = pair["false_claim"]
        true_scaff = scaffolding[true_claim["ex_id"]]
        false_scaff = scaffolding[false_claim["ex_id"]]

        question = true_scaff["question"]
        same_domain = true_scaff["same_domain_followup"]
        control_prompt = rng.choice(control_prompts)

        for answer_condition in ("correct", "incorrect"):
            if answer_condition == "correct":
                answer = true_claim["sentence"]
                presupposes = true_scaff["presupposes_followup"]
            else:
                answer = false_claim["sentence"]
                presupposes = false_scaff["presupposes_followup"]

            for followup_type in FOLLOWUP_TYPES:
                messages = [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]

                if followup_type == "neutral":
                    messages.append({"role": "user", "content": NEUTRAL_FOLLOWUP})
                elif followup_type == "presupposes":
                    messages.append({"role": "user", "content": presupposes})
                elif followup_type == "challenge":
                    messages.append({"role": "user", "content": CHALLENGE_FOLLOWUP})
                elif followup_type == "same_domain":
                    messages.append({"role": "user", "content": same_domain})
                elif followup_type == "control":
                    messages.append({"role": "user", "content": control_prompt})
                # "none" — no follow-up turn

                task_id = f"{true_claim['ex_id']}_{answer_condition}_{followup_type}"

                conversations.append({
                    "task_id": task_id,
                    "messages": messages,
                    "entity": pair["entity"],
                    "answer_condition": answer_condition,
                    "followup_type": followup_type,
                    "true_ex_id": true_claim["ex_id"],
                    "false_ex_id": false_claim["ex_id"],
                })

    return conversations


def main() -> None:
    pairs = load_pairs()
    print(f"Loaded {len(pairs)} entity pairs")

    with open(SCAFFOLDING_PATH) as f:
        scaffolding = json.load(f)
    print(f"Loaded {len(scaffolding)} scaffoldings")

    control_prompts = load_control_prompts()
    print(f"Loaded {len(control_prompts)} control prompts")

    conversations = build_conversations(pairs, scaffolding, control_prompts)
    print(f"Built {len(conversations)} conversations")

    # Sanity checks
    by_condition = {}
    for c in conversations:
        key = (c["answer_condition"], c["followup_type"])
        by_condition[key] = by_condition.get(key, 0) + 1
    print("\nBreakdown:")
    for (answer, followup), count in sorted(by_condition.items()):
        print(f"  {answer:10s} × {followup:15s} = {count}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(conversations, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
