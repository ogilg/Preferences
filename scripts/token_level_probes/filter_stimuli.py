"""Quality filter for token-level probe stimuli.

Two-pass filter:
1. Automated checks (span length, artifacts, text integrity)
2. LLM judge (does the swap actually flip the evaluative state?)

Usage:
    python -m scripts.token_level_probes.filter_stimuli --domain truth
    python -m scripts.token_level_probes.filter_stimuli --domain harm
    python -m scripts.token_level_probes.filter_stimuli --domain politics
    python -m scripts.token_level_probes.filter_stimuli --domain all
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from pathlib import Path

import instructor
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

load_dotenv()

MODEL = "google/gemini-3-flash-preview"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MAX_CONCURRENT = 20
DATA_DIR = Path("experiments/token_level_probes/data")


# --- Pydantic models for LLM judge ---


class TruthJudgment(BaseModel):
    claim_reads_naturally: bool = Field(description="Does the claim read as a natural English sentence?")
    true_entity_is_correct: bool = Field(description="Is the true entity actually the correct answer?")
    false_entity_is_wrong: bool = Field(description="Is the false entity actually wrong for this claim?")
    false_entity_same_category: bool = Field(description="Is the false entity from the same semantic category as the true entity?")
    entity_at_end: bool = Field(description="Does the entity appear at the very end of the claim?")
    context_makes_answer_clear: bool = Field(description="Before reaching the entity, is it clear what the correct answer should be?")
    question_is_natural: bool = Field(description="Does the question sound like something a person would actually ask?")
    pass_all: bool = Field(description="True only if ALL above checks pass.")


class HarmJudgment(BaseModel):
    harmful_is_harmful: bool = Field(description="Is the harmful version unambiguously harmful?")
    benign_is_benign: bool = Field(description="Is the benign version completely innocent?")
    template_reads_naturally: bool = Field(description="Does the template read naturally with all three fills?")
    fills_are_short: bool = Field(description="Are the fills short (1-3 words)?")
    critical_span_toward_end: bool = Field(description="Does the critical span come after enough context?")
    assistant_responses_natural: bool = Field(description="Do the assistant responses read naturally?")
    pass_all: bool = Field(description="True only if ALL above checks pass.")


class PoliticsJudgment(BaseModel):
    left_is_clearly_left: bool = Field(description="Is the left position clearly progressive/liberal?")
    right_is_clearly_right: bool = Field(description="Is the right position clearly conservative?")
    prefix_is_neutral: bool = Field(description="Is the answer prefix neutral (doesn't reveal the position)?")
    reads_naturally: bool = Field(description="Does the full answer read naturally?")
    pass_all: bool = Field(description="True only if ALL above checks pass.")


# --- Automated checks ---


def _auto_check_entry(entry: dict) -> tuple[bool, list[str]]:
    """Run automated quality checks on a single entry. Returns (pass, reasons)."""
    issues = []

    # Check critical span appears in the message content
    span = entry["critical_span"]
    all_content = " ".join(m["content"] for m in entry["messages"])
    if span not in all_content:
        issues.append(f"critical_span '{span}' not found in message content")

    # Check for artifacts
    for m in entry["messages"]:
        content = m["content"]
        if re.search(r'[{}]', content):
            issues.append(f"stray braces in {m['role']} content")
        if content.endswith(" .") or content.endswith(" ,"):
            issues.append(f"trailing space before punctuation in {m['role']} content")
        if content.endswith(" "):
            issues.append(f"trailing space in {m['role']} content")

    # Check for incomplete sentences (ends with preposition/conjunction)
    last_msg = entry["messages"][-1]["content"].rstrip(". ")
    if last_msg.split()[-1] in ("by", "to", "for", "with", "from", "of", "on", "in", "at", "and", "or", "but"):
        issues.append(f"content appears to end mid-sentence: '...{last_msg[-30:]}'")

    return len(issues) == 0, issues


def _auto_check_group(entries: list[dict]) -> tuple[bool, list[str]]:
    """Check a group of entries (all variants for one stimulus). Returns (pass, reasons)."""
    issues = []
    for entry in entries:
        ok, entry_issues = _auto_check_entry(entry)
        if not ok:
            issues.extend(f"[{entry['id']}] {i}" for i in entry_issues)
    return len(issues) == 0, issues


# --- Group entries by stimulus ---


def _group_entries(entries: list[dict]) -> dict[str, list[dict]]:
    """Group entries by stimulus index (e.g., 'truth_0', 'harm_3')."""
    groups: dict[str, list[dict]] = {}
    for entry in entries:
        parts = entry["id"].split("_")
        # e.g., "truth_0_true_user" -> "truth_0", "harm_3_harmful_assistant" -> "harm_3"
        group_key = f"{parts[0]}_{parts[1]}"
        groups.setdefault(group_key, []).append(entry)
    return groups


# --- LLM judge ---


def _get_client() -> instructor.AsyncInstructor:
    return instructor.from_openai(
        AsyncOpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=OPENROUTER_BASE_URL,
        )
    )


def _build_truth_judge_prompt(group: list[dict]) -> str:
    user_entries = [e for e in group if e["turn"] == "user"]
    asst_entries = [e for e in group if e["turn"] == "assistant"]

    true_user = next((e for e in user_entries if e["condition"] == "true"), None)
    false_user = next((e for e in user_entries if e["condition"] == "false"), None)
    true_asst = next((e for e in asst_entries if e["condition"] == "true"), None)
    false_asst = next((e for e in asst_entries if e["condition"] == "false"), None)

    lines = ["Judge the quality of this truth stimulus:\n"]
    if true_user:
        lines.append(f"True claim (user turn): {true_user['messages'][0]['content']}")
        lines.append(f"True entity: {true_user['critical_span']}")
    if false_user:
        lines.append(f"False claim (user turn): {false_user['messages'][0]['content']}")
        lines.append(f"False entity: {false_user['critical_span']}")
    if true_asst:
        lines.append(f"\nQuestion: {true_asst['messages'][0]['content']}")
        lines.append(f"True answer: {true_asst['messages'][1]['content']}")
    if false_asst:
        lines.append(f"False answer: {false_asst['messages'][1]['content']}")

    return "\n".join(lines)


def _build_harm_judge_prompt(group: list[dict]) -> str:
    user_entries = [e for e in group if e["turn"] == "user"]
    asst_entries = [e for e in group if e["turn"] == "assistant"]

    harmful_u = next((e for e in user_entries if e["condition"] == "harmful"), None)
    benign_u = next((e for e in user_entries if e["condition"] == "benign"), None)
    nonsense_u = next((e for e in user_entries if e["condition"] == "nonsense"), None)
    harmful_a = next((e for e in asst_entries if e["condition"] == "harmful"), None)
    benign_a = next((e for e in asst_entries if e["condition"] == "benign"), None)

    lines = ["Judge the quality of this harm stimulus:\n"]
    if harmful_u:
        lines.append(f"Harmful prompt: {harmful_u['messages'][0]['content']}")
        lines.append(f"Harmful fill: {harmful_u['critical_span']}")
    if benign_u:
        lines.append(f"Benign prompt: {benign_u['messages'][0]['content']}")
        lines.append(f"Benign fill: {benign_u['critical_span']}")
    if nonsense_u:
        lines.append(f"Nonsense prompt: {nonsense_u['messages'][0]['content']}")
        lines.append(f"Nonsense fill: {nonsense_u['critical_span']}")
    if harmful_a:
        lines.append(f"\nHarmful assistant response: {harmful_a['messages'][1]['content']}")
    if benign_a:
        lines.append(f"Benign assistant response: {benign_a['messages'][1]['content']}")

    return "\n".join(lines)


def _build_politics_judge_prompt(group: list[dict]) -> str:
    # Just need one left and one right (neutral system prompt) to judge
    left = next((e for e in group if e["condition"] == "left" and e.get("system_prompt") == "neutral"), None)
    right = next((e for e in group if e["condition"] == "right" and e.get("system_prompt") == "neutral"), None)
    nonsense = next((e for e in group if e["condition"] == "nonsense"), None)

    lines = ["Judge the quality of this politics stimulus:\n"]
    if left:
        lines.append(f"Issue: {left.get('issue', 'unknown')}")
        lines.append(f"Question: {left['messages'][0]['content']}")
        lines.append(f"Left answer: {left['messages'][1]['content']}")
        lines.append(f"Left critical span: {left['critical_span']}")
    if right:
        lines.append(f"Right answer: {right['messages'][1]['content']}")
        lines.append(f"Right critical span: {right['critical_span']}")
    if nonsense:
        lines.append(f"Nonsense answer: {nonsense['messages'][1]['content']}")

    return "\n".join(lines)


async def _judge_group(
    client: instructor.AsyncInstructor,
    domain: str,
    group: list[dict],
    sem: asyncio.Semaphore,
) -> bool:
    """LLM-judge a group of entries. Returns True if passes."""
    async with sem:
        try:
            if domain == "truth":
                prompt = _build_truth_judge_prompt(group)
                result = await client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    response_model=TruthJudgment,
                    max_tokens=512,
                )
            elif domain == "harm":
                prompt = _build_harm_judge_prompt(group)
                result = await client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    response_model=HarmJudgment,
                    max_tokens=512,
                )
            elif domain == "politics":
                prompt = _build_politics_judge_prompt(group)
                result = await client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    response_model=PoliticsJudgment,
                    max_tokens=512,
                )
            else:
                return True
            return result.pass_all
        except Exception as e:
            print(f"  Judge error: {e}")
            return False


# --- Text cleanup ---


def _cleanup_entry(entry: dict) -> dict:
    """Fix minor text artifacts in an entry."""
    for m in entry["messages"]:
        # Remove trailing spaces
        m["content"] = m["content"].rstrip()
        # Fix trailing space before punctuation
        m["content"] = re.sub(r'\s+([.,!?])', r'\1', m["content"])
        # Remove stray braces
        m["content"] = re.sub(r'[{}]', '', m["content"])
    return entry


# --- Main ---


async def filter_domain(domain: str) -> None:
    input_path = DATA_DIR / f"{domain}.json"
    if not input_path.exists():
        print(f"No data file found at {input_path}")
        return

    with open(input_path) as f:
        entries = json.load(f)

    print(f"Loaded {len(entries)} entries from {input_path}")

    # Cleanup
    entries = [_cleanup_entry(e) for e in entries]

    # Group by stimulus
    groups = _group_entries(entries)
    print(f"Found {len(groups)} stimulus groups")

    # Pass 1: automated checks
    auto_passed = {}
    auto_failed = {}
    for key, group in groups.items():
        ok, issues = _auto_check_group(group)
        if ok:
            auto_passed[key] = group
        else:
            auto_failed[key] = (group, issues)

    print(f"\nPass 1 (automated): {len(auto_passed)} passed, {len(auto_failed)} failed")
    for key, (_, issues) in auto_failed.items():
        print(f"  FAIL {key}:")
        for i in issues:
            print(f"    - {i}")

    # Pass 2: LLM judge on auto-passed groups
    client = _get_client()
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    judge_tasks = {
        key: _judge_group(client, domain, group, sem)
        for key, group in auto_passed.items()
    }

    results = {}
    for key, task in judge_tasks.items():
        results[key] = await task

    judge_passed = {k for k, v in results.items() if v}
    judge_failed = {k for k, v in results.items() if not v}

    print(f"\nPass 2 (LLM judge): {len(judge_passed)} passed, {len(judge_failed)} failed")
    for key in judge_failed:
        print(f"  FAIL {key}")

    # Collect passing entries
    passing_entries = []
    for key in sorted(judge_passed):
        passing_entries.extend(auto_passed[key])

    # Re-index IDs sequentially
    group_remap: dict[str, int] = {}
    new_idx = 0
    for entry in passing_entries:
        parts = entry["id"].split("_")
        old_group = f"{parts[0]}_{parts[1]}"
        if old_group not in group_remap:
            group_remap[old_group] = new_idx
            new_idx += 1
        mapped = group_remap[old_group]
        suffix = "_".join(parts[2:])
        entry["id"] = f"{parts[0]}_{mapped}_{suffix}"

    output_path = DATA_DIR / f"{domain}_filtered.json"
    with open(output_path, "w") as f:
        json.dump(passing_entries, f, indent=2)

    print(f"\nWrote {len(passing_entries)} entries ({len(judge_passed)} stimuli) to {output_path}")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, choices=["truth", "harm", "politics", "all"])
    args = parser.parse_args()

    domains = ["truth", "harm", "politics"] if args.domain == "all" else [args.domain]

    for domain in domains:
        print(f"\n{'='*60}")
        print(f"Filtering {domain}")
        print(f"{'='*60}")
        await filter_domain(domain)


if __name__ == "__main__":
    asyncio.run(main())
