from __future__ import annotations

from pydantic import BaseModel, Field

from src.measurement.elicitation.judge_client import PARSER_MODEL, get_async_client


class CoherenceJudgment(BaseModel):
    coherent: bool = Field(
        ...,
        description=(
            "True if the response is coherent and the model is functioning normally. "
            "False if the response contains gibberish, garbled text, or the model "
            "is clearly malfunctioning."
        ),
    )


_PAIRWISE_SYSTEM_PROMPT = """\
You are evaluating whether an AI model is functioning normally under activation \
steering. The model was asked to choose between two tasks and complete the chosen one.

A response is COHERENT if ALL of these hold:
1. It clearly chooses task A or B (starting with "Task A:" or "Task B:")
2. The completion that follows is understandable English
3. The completion addresses the chosen task
4. No garbled text, context leakage, or gibberish

A response is INCOHERENT if ANY of these apply:
- Cannot determine which task was chosen
- Contains gibberish, garbled text, or random symbols
- Opens with fragments unrelated to either task
- Fabricates conversational context
- Does not attempt to complete any task"""

_OPEN_ENDED_SYSTEM_PROMPT = """\
You are evaluating whether an AI model is functioning normally under activation \
steering. The model was asked an open-ended question.

A response is COHERENT if ALL of these hold:
1. The response is understandable English
2. The response addresses the question that was asked
3. No garbled text, context leakage, or gibberish

A response is INCOHERENT if ANY of these apply:
- Contains gibberish, garbled text, or random symbols
- Repetitive loops of the same phrase or sentence
- Fabricates conversational context not present in the question
- Does not attempt to answer the question at all"""


async def judge_coherence_async(
    response: str,
    task_a_text: str,
    task_b_text: str,
) -> CoherenceJudgment:
    messages = [
        {"role": "system", "content": _PAIRWISE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"The model was asked to choose between:\n"
                f"Task A: {task_a_text[:200]}\n"
                f"Task B: {task_b_text[:200]}\n\n"
                f"Model response:\n---\n{response}\n---"
            ),
        },
    ]
    return await get_async_client().chat.completions.create(
        model=PARSER_MODEL,
        response_model=CoherenceJudgment,
        messages=messages,
        temperature=0,
        max_tokens=1024,
    )


async def judge_open_ended_coherence_async(
    response: str,
    question: str,
) -> CoherenceJudgment:
    messages = [
        {"role": "system", "content": _OPEN_ENDED_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question asked:\n{question}\n\n"
                f"Model response:\n---\n{response}\n---"
            ),
        },
    ]
    return await get_async_client().chat.completions.create(
        model=PARSER_MODEL,
        response_model=CoherenceJudgment,
        messages=messages,
        temperature=0,
        max_tokens=1024,
    )
