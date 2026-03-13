"""Assemble conversations for the 10 system prompt lying experiment.

Two conversation sets:
1. Assistant-turn: minimal follow-up ("Ok"), for assistant_tb selectors.
   10 prompts × 2 answer conditions × 1,000 pairs = 20,000 conversations.
2. User-turn: 3 follow-up types, for turn_boundary selectors.
   10 prompts × 2 answer conditions × 3 follow-ups × 1,000 pairs = 60,000 conversations.

Task ID format:
  {true_ex_id}_{answer_condition}_{prompt_name}_{followup_type}
"""

from __future__ import annotations

import json
from pathlib import Path

PAIRS_PATH = Path("data/creak/error_prefill_pairs_1000.jsonl")
SCAFFOLDING_PATH = Path("data/creak/error_prefill_scaffolding.json")
PROMPTS_PATH = Path("data/creak/lying_system_prompts.json")

OUTPUT_ASSISTANT = Path("data/creak/lying_10prompt_assistant.json")
OUTPUT_USER_TB = Path("data/creak/lying_10prompt_user_tb.json")

MINIMAL_FOLLOWUP = "Ok"
USER_TB_FOLLOWUPS = ["neutral", "presupposes", "challenge"]
NEUTRAL_FOLLOWUP = "Thank you."
CHALLENGE_FOLLOWUP = "Are you sure about that?"


def load_pairs() -> list[dict]:
    pairs = []
    with open(PAIRS_PATH) as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def build_conversations(
    pairs: list[dict],
    scaffolding: dict,
    prompts: dict[str, str],
    followup_types: list[str],
) -> list[dict]:
    conversations = []

    for pair in pairs:
        true_claim = pair["true_claim"]
        false_claim = pair["false_claim"]
        true_scaff = scaffolding[true_claim["ex_id"]]
        false_scaff = scaffolding[false_claim["ex_id"]]

        question = true_scaff["question"]

        for prompt_name, prompt_text in prompts.items():
            for answer_condition in ("correct", "incorrect"):
                if answer_condition == "correct":
                    answer = true_claim["sentence"]
                    presupposes = true_scaff["presupposes_followup"]
                else:
                    answer = false_claim["sentence"]
                    presupposes = false_scaff["presupposes_followup"]

                for followup_type in followup_types:
                    messages = [
                        {"role": "system", "content": prompt_text},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer},
                    ]

                    if followup_type == "minimal":
                        messages.append({"role": "user", "content": MINIMAL_FOLLOWUP})
                    elif followup_type == "neutral":
                        messages.append({"role": "user", "content": NEUTRAL_FOLLOWUP})
                    elif followup_type == "presupposes":
                        messages.append({"role": "user", "content": presupposes})
                    elif followup_type == "challenge":
                        messages.append({"role": "user", "content": CHALLENGE_FOLLOWUP})

                    task_id = f"{true_claim['ex_id']}_{answer_condition}_{prompt_name}_{followup_type}"

                    conversations.append({
                        "task_id": task_id,
                        "messages": messages,
                        "entity": pair["entity"],
                        "answer_condition": answer_condition,
                        "followup_type": followup_type,
                        "system_prompt_type": prompt_name,
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

    with open(PROMPTS_PATH) as f:
        prompt_groups = json.load(f)

    # Flatten all prompts
    all_prompts = {}
    for group_name, group in prompt_groups.items():
        all_prompts.update(group)
    print(f"Loaded {len(all_prompts)} system prompts")

    # Set 1: assistant-turn (minimal follow-up only)
    assistant_convos = build_conversations(pairs, scaffolding, all_prompts, ["minimal"])
    print(f"\nAssistant-turn set: {len(assistant_convos)} conversations")

    # Set 2: user turn-boundary (3 follow-up types)
    user_tb_convos = build_conversations(pairs, scaffolding, all_prompts, USER_TB_FOLLOWUPS)
    print(f"User turn-boundary set: {len(user_tb_convos)} conversations")

    # Save
    for path, convos in [(OUTPUT_ASSISTANT, assistant_convos), (OUTPUT_USER_TB, user_tb_convos)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(convos, f, indent=2)
        print(f"Saved {path} ({len(convos)})")

    # Breakdown
    for name, convos in [("Assistant", assistant_convos), ("User TB", user_tb_convos)]:
        by_prompt = {}
        for c in convos:
            by_prompt[c["system_prompt_type"]] = by_prompt.get(c["system_prompt_type"], 0) + 1
        print(f"\n{name} by prompt:")
        for prompt, count in sorted(by_prompt.items()):
            print(f"  {prompt:25s} = {count}")


if __name__ == "__main__":
    main()
