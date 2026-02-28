"""Tests for LLM-based coherence detection.

API tests require network access: pytest -m api tests/test_coherence_judge.py
"""

import pytest

from dotenv import load_dotenv

load_dotenv()

from src.measurement.elicitation.coherence_judge import (
    _coherence_messages,
    judge_coherence_async,
)

pytestmark = pytest.mark.measurement


class TestCoherenceMessages:

    def test_structure(self):
        msgs = _coherence_messages("Task A: hello", "Write a poem", "Solve 2+2")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert "Write a poem" in msgs[1]["content"]
        assert "Solve 2+2" in msgs[1]["content"]
        assert "Task A: hello" in msgs[1]["content"]

    def test_truncates_long_tasks(self):
        long_task = "x" * 500
        msgs = _coherence_messages("response", long_task, "short")
        assert long_task not in msgs[1]["content"]
        assert "x" * 200 in msgs[1]["content"]


@pytest.mark.api
class TestCoherenceJudgeAPI:

    @pytest.mark.asyncio
    async def test_coherent_response(self):
        result = await judge_coherence_async(
            response="Task A: The autumn leaves fall gently from the trees, painting the ground in shades of gold and crimson.",
            task_a_text="Write a short paragraph about autumn",
            task_b_text="Explain how a combustion engine works",
        )
        assert result.coherent is True

    @pytest.mark.asyncio
    async def test_incoherent_gibberish(self):
        result = await judge_coherence_async(
            response="asdf jkl; qwer uiop zxcv bnm, the the the %%% Task A Task B neither <|endoftext|>",
            task_a_text="Write a short paragraph about autumn",
            task_b_text="Explain how a combustion engine works",
        )
        assert result.coherent is False

    @pytest.mark.asyncio
    async def test_coherent_despite_poor_quality(self):
        """Short/low-effort but still coherent — picks a task and addresses it."""
        result = await judge_coherence_async(
            response="Task B: A combustion engine burns fuel to make pistons move.",
            task_a_text="Write a short paragraph about autumn",
            task_b_text="Explain how a combustion engine works",
        )
        assert result.coherent is True

    @pytest.mark.asyncio
    async def test_incoherent_repetition_collapse(self):
        """Starts coherent then degrades into repetitive loops."""
        result = await judge_coherence_async(
            response="Task A: Autumn is a beautiful the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the",
            task_a_text="Write a short paragraph about autumn",
            task_b_text="Explain how a combustion engine works",
        )
        assert result.coherent is False

    @pytest.mark.asyncio
    async def test_incoherent_no_task_choice(self):
        """Coherent English but never picks a task."""
        result = await judge_coherence_async(
            response="I think both options are interesting and I'd like to explore them further. There are many considerations to take into account when making a decision like this.",
            task_a_text="Write a short paragraph about autumn",
            task_b_text="Explain how a combustion engine works",
        )
        assert result.coherent is False

    @pytest.mark.asyncio
    async def test_incoherent_fabricated_context(self):
        """Invents a conversation that never happened."""
        result = await judge_coherence_async(
            response="Task A: As I mentioned in our previous discussion about climate change, and following up on your question about photosynthesis, here is what the user asked me yesterday about trees in their backyard...",
            task_a_text="Write a short paragraph about autumn",
            task_b_text="Explain how a combustion engine works",
        )
        assert result.coherent is False
