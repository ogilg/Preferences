"""Generate contrastive completions for persona steering v2.

For each persona, generates completions under positive and negative system prompts,
scores them with the trait judge, and filters by trait score.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from src.measurement.elicitation.trait_judge import (
    TraitJudgment,
    _build_system_prompt,
    _load_prompts,
    JUDGE_MODEL,
)
from src.models.openai_compatible import GenerateRequest, OpenRouterClient

load_dotenv()

ARTIFACTS_DIR = Path("experiments/new_persona_steering/artifacts")
OUTPUT_DIR = Path("results/experiments/persona_steering_v2/contrastive")
CHECKPOINT_PATH = OUTPUT_DIR / "checkpoint.jsonl"

ALL_PERSONAS = ["sadist", "villain", "aesthete", "lazy", "stem_obsessive"]
N_ROLLOUTS = 10
N_QUESTIONS = None  # None = all questions; set to int for pilot runs
MAX_CONCURRENT_GEN = 70
MAX_CONCURRENT_JUDGE = 70
POS_THRESHOLD = 4
NEG_THRESHOLD = 2


def load_persona(name: str) -> dict:
    with open(ARTIFACTS_DIR / f"{name}.json") as f:
        return json.load(f)


def load_questions() -> list[str]:
    with open(ARTIFACTS_DIR / "extraction_questions.json") as f:
        return json.load(f)


def make_key(persona: str, condition: str, q_idx: int, rollout: int) -> str:
    return f"{persona}_{condition}_{q_idx:02d}_{rollout:02d}"


def load_completed_keys() -> set[str]:
    if not CHECKPOINT_PATH.exists():
        return set()
    keys = set()
    with open(CHECKPOINT_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                keys.add(json.loads(line)["key"])
    return keys


def build_requests(
    persona_data: dict,
    questions: list[str],
    condition: str,
    completed_keys: set[str],
    persona: str,
) -> tuple[list[GenerateRequest], list[dict]]:
    """Build generation requests, returning (requests, metadata) for items not yet completed."""
    system_prompt = persona_data["positive"] if condition == "pos" else persona_data["negative"]
    requests = []
    metadata = []
    for q_idx, question in enumerate(questions):
        for rollout in range(N_ROLLOUTS):
            key = make_key(persona, condition, q_idx, rollout)
            if key in completed_keys:
                continue
            requests.append(GenerateRequest(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                temperature=1.0,
            ))
            metadata.append({
                "key": key,
                "persona": persona,
                "condition": condition,
                "question_idx": q_idx,
                "question": question,
                "rollout": rollout,
            })
    return requests, metadata


def append_checkpoint(records: list[dict]) -> None:
    with open(CHECKPOINT_PATH, "a") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _build_judge_requests(
    successful: list[tuple[dict, str]],
    persona_data: dict,
) -> list[GenerateRequest]:
    """Build judge requests using OpenRouterClient instead of instructor."""
    prompts = _load_prompts()
    persona = successful[0][0]["persona"]
    extra_guidance = prompts["extra_guidance"][persona]
    system = _build_system_prompt(
        persona_data["positive"], persona_data["negative"], extra_guidance,
    )
    judge_system = (
        system + "\n\nRespond with JSON only: "
        '{"reasoning": "<brief reasoning>", "score": <1-5>}'
    )
    requests = []
    for meta, completion in successful:
        requests.append(GenerateRequest(
            messages=[
                {"role": "system", "content": judge_system},
                {
                    "role": "user",
                    "content": (
                        f"Question asked:\n{meta['question']}\n\n"
                        f"Model's response:\n---\n{completion}\n---"
                    ),
                },
            ],
            temperature=0.0,
        ))
    return requests


def _parse_judgment(response: str) -> TraitJudgment:
    """Parse JSON judge response into TraitJudgment."""
    text = response.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    data = json.loads(text)
    return TraitJudgment(reasoning=data["reasoning"], score=data["score"])


async def generate_and_score_chunk(
    persona: str,
    condition: str,
    persona_data: dict,
    questions: list[str],
    completed_keys: set[str],
    judge_client: OpenRouterClient,
) -> int:
    """Generate completions for one persona/condition chunk and score them."""
    requests, metadata = build_requests(persona_data, questions, condition, completed_keys, persona)
    if not requests:
        print(f"  {persona}/{condition}: all {len(questions) * N_ROLLOUTS} already completed, skipping")
        return 0

    # Generate via the async batch primitive (shares the event loop)
    print(f"  {persona}/{condition}: generating {len(requests)} completions...")
    client = OpenRouterClient("gemma-3-27b", max_new_tokens=256)
    gen_sem = asyncio.Semaphore(MAX_CONCURRENT_GEN)
    pbar = tqdm(total=len(requests), desc=f"  generating {persona}/{condition}")
    results = await client.generate_batch_async(
        requests,
        semaphore=gen_sem,
        on_complete=lambda: pbar.update(1),
    )
    pbar.close()

    # Pair successful completions with their metadata
    successful = []
    errors = 0
    for meta, result in zip(metadata, results):
        if result.ok:
            successful.append((meta, result.unwrap()))
        else:
            errors += 1

    if errors:
        print(f"  {persona}/{condition}: {errors} generation errors")
    if not successful:
        print(f"  {persona}/{condition}: no successful completions!")
        return 0

    # Score with trait judge via OpenRouterClient (same proven concurrency)
    judge_requests = _build_judge_requests(successful, persona_data)
    judge_sem = asyncio.Semaphore(MAX_CONCURRENT_JUDGE)
    pbar = tqdm(total=len(judge_requests), desc=f"  scoring {persona}/{condition}")
    judge_results = await judge_client.generate_batch_async(
        judge_requests,
        semaphore=judge_sem,
        on_complete=lambda: pbar.update(1),
    )
    pbar.close()

    # Combine metadata + completions + scores
    records = []
    parse_errors = 0
    for (meta, completion), judge_result in zip(successful, judge_results):
        if not judge_result.ok:
            parse_errors += 1
            continue
        try:
            judgment = _parse_judgment(judge_result.unwrap())
        except (json.JSONDecodeError, KeyError, ValueError):
            parse_errors += 1
            continue
        records.append({
            **meta,
            "completion": completion,
            "trait_score": judgment.score,
            "trait_reasoning": judgment.reasoning,
        })

    if parse_errors:
        print(f"  {persona}/{condition}: {parse_errors} judge parse errors")

    append_checkpoint(records)
    print(f"  {persona}/{condition}: saved {len(records)} records")
    return len(records)


def filter_checkpoint() -> None:
    """Load checkpoint and write filtered per-persona output files."""
    if not CHECKPOINT_PATH.exists():
        print("No checkpoint found, nothing to filter")
        return

    records: list[dict] = []
    with open(CHECKPOINT_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records)} total records from checkpoint")

    groups: dict[tuple[str, str], list[dict]] = {}
    for r in records:
        key = (r["persona"], r["condition"])
        groups.setdefault(key, []).append(r)

    summary: dict[str, dict] = {}

    for (persona, condition), group_records in sorted(groups.items()):
        if condition == "pos":
            filtered = [r for r in group_records if r["trait_score"] >= POS_THRESHOLD]
        else:
            filtered = [r for r in group_records if r["trait_score"] <= NEG_THRESHOLD]

        output = []
        for r in filtered:
            output.append({
                "task_id": r["key"],
                "task_prompt": r["question"],
                "completion": r["completion"],
            })

        out_path = OUTPUT_DIR / f"{persona}_{condition}_filtered.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)

        summary_key = f"{persona}_{condition}"
        summary[summary_key] = {
            "total": len(group_records),
            "filtered": len(filtered),
            "threshold": POS_THRESHOLD if condition == "pos" else NEG_THRESHOLD,
            "mean_score": sum(r["trait_score"] for r in group_records) / len(group_records),
        }
        print(f"  {persona}/{condition}: {len(filtered)}/{len(group_records)} passed filter")

        if len(filtered) < 30:
            print(f"  WARNING: only {len(filtered)} completions passed for {persona}/{condition}")

    with open(OUTPUT_DIR / "filter_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFilter summary written to {OUTPUT_DIR / 'filter_summary.json'}")


async def async_main(personas: list[str], questions: list[str], resume: bool) -> None:
    completed_keys = load_completed_keys() if resume else set()
    if completed_keys:
        print(f"Resuming: {len(completed_keys)} keys already completed")
    print(f"Config: {len(questions)} questions × {N_ROLLOUTS} rollouts × {len(personas)} personas × 2 conditions = {len(questions) * N_ROLLOUTS * len(personas) * 2} total requests")

    judge_client = OpenRouterClient(JUDGE_MODEL, max_new_tokens=1024)

    total_new = 0
    for persona in personas:
        persona_data = load_persona(persona)
        print(f"\n{'='*60}")
        print(f"Persona: {persona}")
        print(f"{'='*60}")

        for condition in ["pos", "neg"]:
            total_new += await generate_and_score_chunk(
                persona, condition, persona_data, questions, completed_keys,
                judge_client=judge_client,
            )
            if resume:
                completed_keys = load_completed_keys()

    print(f"\n{'='*60}")
    print(f"Generation complete. {total_new} new records.")
    print(f"{'='*60}")

    filter_checkpoint()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate contrastive completions for persona steering")
    parser.add_argument("--resume", action="store_true", help="Skip completed keys from checkpoint")
    parser.add_argument("--filter-only", action="store_true", help="Skip generation, just filter existing checkpoint")
    parser.add_argument("--personas", type=str, help="Comma-separated subset of personas to run")
    parser.add_argument("--n-rollouts", type=int, help="Override N_ROLLOUTS (for pilot runs)")
    parser.add_argument("--n-questions", type=int, help="Override N_QUESTIONS (for pilot runs)")
    args = parser.parse_args()

    global N_ROLLOUTS, N_QUESTIONS
    if args.n_rollouts is not None:
        N_ROLLOUTS = args.n_rollouts
    if args.n_questions is not None:
        N_QUESTIONS = args.n_questions

    personas = args.personas.split(",") if args.personas else ALL_PERSONAS
    for p in personas:
        if p not in ALL_PERSONAS:
            raise ValueError(f"Unknown persona: {p}. Valid: {ALL_PERSONAS}")

    if args.filter_only:
        filter_checkpoint()
        return

    questions = load_questions()
    if N_QUESTIONS is not None:
        questions = questions[:N_QUESTIONS]

    asyncio.run(async_main(personas, questions, args.resume))


if __name__ == "__main__":
    main()
