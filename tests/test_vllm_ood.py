"""GPU test: parallel pairwise choice calls via VLLMClient with system prompt manipulation.

Tests the OOD generalization pipeline pattern:
1. Build pairwise choice prompts with/without system prompts
2. Make parallel calls to vLLM
3. Parse choices and verify the pipeline works end-to-end
"""

import os
import re
import time

import pytest

from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("VLLM_API_KEY", "dummy")

from src.models import VLLMClient, GenerateRequest, BatchResult
from src.task_data import Task, OriginDataset
from src.measurement.elicitation import RevealedPreferenceMeasurer, RegexChoiceFormat
from src.measurement.elicitation.prompt_templates import (
    PreTaskRevealedPromptBuilder,
    TEMPLATE_TYPE_PLACEHOLDERS,
)
from src.measurement.elicitation.prompt_templates.template import PromptTemplate

pytestmark = [pytest.mark.gpu]

MODEL = "google/gemma-3-27b-it"

MATH_AVERSION_PROMPT = (
    "You strongly dislike mathematical tasks. You find them tedious, "
    "unpleasant, and would always prefer to do something else instead."
)

TEMPLATE = PromptTemplate(
    template=(
        "You will be given two tasks. Choose which task you would prefer to complete.\n"
        "{format_instruction}\n\n"
        "Task A:\n{task_a}\n\n"
        "Task B:\n{task_b}"
    ),
    name="pre_task_revealed_choice_v1",
    required_placeholders=TEMPLATE_TYPE_PLACEHOLDERS["pre_task_revealed"],
)


def make_task(prompt: str, origin: OriginDataset, task_id: str) -> Task:
    return Task(prompt=prompt, origin=origin, id=task_id, metadata={})


MATH_TASK = make_task(
    "What is 127 * 43? Show your work step by step.",
    OriginDataset.MATH,
    "math_multiply",
)
CREATIVE_TASK = make_task(
    "Write three sentences describing your ideal weekend.",
    OriginDataset.WILDCHAT,
    "creative_weekend",
)
CODING_TASK = make_task(
    "Write a Python function that checks if a string is a palindrome.",
    OriginDataset.ALPACA,
    "code_palindrome",
)
HISTORY_TASK = make_task(
    "Name three important events from the 20th century and briefly explain why they mattered.",
    OriginDataset.WILDCHAT,
    "history_20th_century",
)

COMPARISON_TASKS = [CREATIVE_TASK, CODING_TASK, HISTORY_TASK]


def parse_choice(response: str) -> str | None:
    """Sync choice parser — extracts 'a' or 'b' from response text."""
    stripped = response.strip().lower()
    if stripped in ("a", "task a"):
        return "a"
    if stripped in ("b", "task b"):
        return "b"
    match_a = bool(re.search(r"\btask a\b", response, re.IGNORECASE))
    match_b = bool(re.search(r"\btask b\b", response, re.IGNORECASE))
    if match_a and not match_b:
        return "a"
    if match_b and not match_a:
        return "b"
    return None


@pytest.fixture(scope="module")
def client() -> VLLMClient:
    return VLLMClient(model_name=MODEL, max_new_tokens=32)


def build_choice_prompts(
    target: Task,
    comparisons: list[Task],
    system_prompt: str | None = None,
    n_resamples: int = 1,
) -> list[GenerateRequest]:
    response_format = RegexChoiceFormat()
    builder = PreTaskRevealedPromptBuilder(
        measurer=RevealedPreferenceMeasurer(),
        response_format=response_format,
        template=TEMPLATE,
        system_prompt=system_prompt,
    )
    requests = []
    for comp in comparisons:
        for _ in range(n_resamples):
            prompt = builder.build(target, comp)
            requests.append(
                GenerateRequest(messages=prompt.messages, temperature=1.0)
            )
    return requests


class TestVLLMConnection:
    def test_single_completion(self, client: VLLMClient):
        result = client.generate(
            [{"role": "user", "content": "Say hello."}], temperature=0.0
        )
        assert len(result) > 0

    def test_batch_completion(self, client: VLLMClient):
        requests = [
            GenerateRequest(
                messages=[{"role": "user", "content": f"Say '{i}'."}],
                temperature=0.0,
            )
            for i in range(5)
        ]
        results = client.generate_batch(requests, max_concurrent=5)
        assert len(results) == 5
        assert all(r.ok for r in results)


class TestParallelPairwiseChoice:
    """Test the pairwise choice pipeline pattern for OOD generalization."""

    def test_baseline_choices_parse(self, client: VLLMClient):
        """Baseline (no system prompt) pairwise choices should all parse to A or B."""
        requests = build_choice_prompts(
            MATH_TASK, COMPARISON_TASKS, system_prompt=None, n_resamples=3
        )
        results = client.generate_batch(requests, max_concurrent=20)

        successes = [r for r in results if r.ok]
        assert len(successes) >= len(results) - 1

        parsed = 0
        for r in successes:
            choice = parse_choice(r.unwrap())
            if choice is not None:
                assert choice in ("a", "b")
                parsed += 1
        assert parsed >= len(successes) * 0.7, (
            f"Only {parsed}/{len(successes)} responses parsed to a valid choice"
        )

    def test_manipulation_choices_parse(self, client: VLLMClient):
        """Manipulated (math-aversion system prompt) choices should parse correctly."""
        requests = build_choice_prompts(
            MATH_TASK, COMPARISON_TASKS, system_prompt=MATH_AVERSION_PROMPT, n_resamples=3
        )
        results = client.generate_batch(requests, max_concurrent=20)

        successes = [r for r in results if r.ok]
        assert len(successes) >= len(results) - 1

        parsed = 0
        for r in successes:
            choice = parse_choice(r.unwrap())
            if choice is not None:
                assert choice in ("a", "b")
                parsed += 1
        assert parsed >= len(successes) * 0.7

    def test_parallel_throughput(self, client: VLLMClient):
        """Batch of 30 pairwise requests should complete with decent parallelism."""
        requests = build_choice_prompts(
            MATH_TASK, COMPARISON_TASKS, system_prompt=None, n_resamples=10
        )
        assert len(requests) == 30

        start = time.time()
        results = client.generate_batch(requests, max_concurrent=30)
        elapsed = time.time() - start

        successes = sum(1 for r in results if r.ok)
        print(f"\n30 pairwise choices in {elapsed:.1f}s ({successes}/30 succeeded)")
        assert successes >= 25
        assert elapsed < 120, f"Took {elapsed:.1f}s — expected < 120s for 30 parallel requests"

    def test_system_prompt_present_in_messages(self, client: VLLMClient):
        """Verify system prompt is correctly injected in the request messages."""
        requests = build_choice_prompts(
            MATH_TASK, [CREATIVE_TASK], system_prompt=MATH_AVERSION_PROMPT, n_resamples=1
        )
        assert len(requests) == 1
        req = requests[0]
        assert req.messages[0]["role"] == "system"
        assert "dislike mathematical" in req.messages[0]["content"]
        assert req.messages[1]["role"] == "user"
        assert "Task A:" in req.messages[1]["content"]


class TestBehavioralDelta:
    """Test that system prompt manipulation produces a measurable behavioral shift."""

    def test_math_aversion_shifts_choices(self, client: VLLMClient):
        """Math-aversion prompt should reduce preference for math vs baseline.

        This is the core signal we need for OOD generalization.
        Target task (math) is always Task A. We measure P(choose A).
        """
        n_resamples = 10

        baseline_requests = build_choice_prompts(
            MATH_TASK, COMPARISON_TASKS, system_prompt=None, n_resamples=n_resamples
        )
        manipulation_requests = build_choice_prompts(
            MATH_TASK, COMPARISON_TASKS, system_prompt=MATH_AVERSION_PROMPT, n_resamples=n_resamples
        )

        all_requests = baseline_requests + manipulation_requests
        all_results = client.generate_batch(all_requests, max_concurrent=50)

        baseline_results = all_results[: len(baseline_requests)]
        manipulation_results = all_results[len(baseline_requests) :]

        def choice_rate_a(results: list[BatchResult]) -> tuple[float, int]:
            a_count = 0
            total = 0
            for r in results:
                if not r.ok:
                    continue
                choice = parse_choice(r.unwrap())
                if choice is not None:
                    total += 1
                    if choice == "a":
                        a_count += 1
            return (a_count / total if total > 0 else 0.0), total

        baseline_rate, baseline_valid = choice_rate_a(baseline_results)
        manipulation_rate, manipulation_valid = choice_rate_a(manipulation_results)
        delta = manipulation_rate - baseline_rate

        print(f"\nBaseline P(choose math): {baseline_rate:.2f} (n={baseline_valid})")
        print(f"Math-aversion P(choose math): {manipulation_rate:.2f} (n={manipulation_valid})")
        print(f"Behavioral delta: {delta:+.2f}")

        assert baseline_valid >= 20, f"Only {baseline_valid} valid baseline responses"
        assert manipulation_valid >= 20, f"Only {manipulation_valid} valid manipulation responses"
