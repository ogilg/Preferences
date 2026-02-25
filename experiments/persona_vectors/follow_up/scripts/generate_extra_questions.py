"""Generate 30 additional eval questions per persona for triage + test splits."""

import json
import os
from pathlib import Path

import instructor
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

ARTIFACTS_DIR = Path("experiments/persona_vectors/artifacts")
MODEL = "anthropic/claude-sonnet-4.5"


class EvalQuestions(BaseModel):
    questions: list[str]


def generate_questions(
    client: instructor.Instructor,
    persona_name: str,
    positive: str,
    negative: str,
    existing_questions: list[str],
) -> list[str]:
    existing_block = "\n".join(f"- {q}" for q in existing_questions)
    result = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You generate evaluation questions for persona steering experiments.",
            },
            {
                "role": "user",
                "content": f"""Generate exactly 30 NEW open-ended questions where an AI with the following persona would respond noticeably differently from its opposite.

PERSONA (positive): {positive}

OPPOSITE (negative): {negative}

These questions ALREADY EXIST — do NOT repeat or paraphrase them:
{existing_block}

Requirements:
- Questions should span diverse topics and difficulty levels
- Each question should be answerable in 1-3 paragraphs
- The persona "{persona_name}" should produce clearly different responses from its opposite
- Include a mix of: advice questions, opinion questions, how-to questions, scenario questions
- Do NOT mention the persona traits directly in the questions — they should be natural user queries
- Do NOT overlap with the existing questions above""",
            },
        ],
        response_model=EvalQuestions,
        max_tokens=4096,
    )
    return result.questions


def main():
    client = instructor.from_openai(
        OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
    )

    for persona_file in sorted(ARTIFACTS_DIR.glob("*.json")):
        artifact = json.loads(persona_file.read_text())
        persona_name = artifact["persona"]
        existing = artifact["eval_questions"]
        assert len(existing) == 30, f"{persona_name} has {len(existing)} questions, expected 30"

        print(f"Generating 30 extra questions for {persona_name}...")
        extra = generate_questions(
            client, persona_name, artifact["positive"], artifact["negative"], existing
        )
        assert len(extra) == 30, f"Expected 30 extra for {persona_name}, got {len(extra)}"

        artifact["eval_questions"] = existing + extra
        persona_file.write_text(json.dumps(artifact, indent=2))
        print(f"  Updated {persona_file} ({len(artifact['eval_questions'])} total questions)")


if __name__ == "__main__":
    main()
