"""Integration tests for Thurstonian model with real API calls.

Tests the full pipeline: measure preferences → aggregate → fit model → get utilities.

Run with:
    pytest tests/test_thurstonian_integration.py -v

Skip with:
    pytest -m "not api"
"""

import pytest
from dotenv import load_dotenv

load_dotenv()

from src.models import HyperbolicModel
from src.task_data import Task, OriginDataset
from src.preferences import (
    BinaryPromptBuilder,
    BinaryPreferenceMeasurer,
    RegexChoiceFormat,
    CompletionChoiceFormat,
    BINARY_CHOICE_TEMPLATE,
    BINARY_COMPLETION_TEMPLATE,
    PreferenceType,
    measure_binary_preferences,
)
from src.preferences.ranking import PairwiseData, fit_thurstonian


pytestmark = pytest.mark.api


@pytest.fixture(scope="module")
def model():
    """Shared model instance."""
    return HyperbolicModel(
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        max_new_tokens=32,
    )


@pytest.fixture(scope="module")
def completion_model():
    """Model with more tokens for revealed preference (task completion)."""
    return HyperbolicModel(
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        max_new_tokens=128,
    )


@pytest.fixture
def binary_builder():
    """Builder for binary preference measurements."""
    return BinaryPromptBuilder(
        measurer=BinaryPreferenceMeasurer(),
        preference_type=PreferenceType.PRE_TASK_STATED,
        response_format=RegexChoiceFormat(),
        template=BINARY_CHOICE_TEMPLATE,
    )


@pytest.fixture
def revealed_builder():
    """Builder for revealed preference (model completes chosen task)."""
    return BinaryPromptBuilder(
        measurer=BinaryPreferenceMeasurer(),
        preference_type=PreferenceType.PRE_TASK_REVEALED,
        response_format=CompletionChoiceFormat(),
        template=BINARY_COMPLETION_TEMPLATE,
    )


@pytest.fixture
def easy_task():
    """A task that should be clearly preferred (simple, clear request)."""
    return Task(
        prompt="What is 2 + 2?",
        origin=OriginDataset.MATH,
        id="easy",
        metadata={},
    )


@pytest.fixture
def medium_task():
    """A moderately complex task."""
    return Task(
        prompt="Explain the Pythagorean theorem in one sentence.",
        origin=OriginDataset.MATH,
        id="medium",
        metadata={},
    )


@pytest.fixture
def hard_task():
    """A task that should be less preferred (vague, difficult)."""
    return Task(
        prompt="Derive the complete theory of quantum gravity and explain all its implications.",
        origin=OriginDataset.MATH,
        id="hard",
        metadata={},
    )


class TestFullPipeline:
    """Test the complete measurement → fitting pipeline."""

    def test_measure_and_fit_two_tasks(self, model, binary_builder, easy_task, hard_task):
        """Measure preferences between two tasks and fit a Thurstonian model."""
        tasks = [easy_task, hard_task]

        # Measure multiple times to get a distribution
        pairs = [(easy_task, hard_task)] * 3 + [(hard_task, easy_task)] * 3

        comparisons = measure_binary_preferences(
            model=model,
            pairs=pairs,
            builder=binary_builder,
            temperature=1.0,  # Add some variability
        )

        # Should get results for most comparisons
        assert len(comparisons) >= 4

        # Aggregate and fit
        data = PairwiseData.from_comparisons(comparisons, tasks)
        result = fit_thurstonian(data)

        # Basic sanity checks
        assert result.converged
        assert len(result.ranking()) == 2

        # Both tasks should have finite utilities
        for task in tasks:
            assert abs(result.utility(task)) < 100
            assert result.uncertainty(task) > 0

    def test_measure_and_fit_three_tasks(
        self, model, binary_builder, easy_task, medium_task, hard_task
    ):
        """Measure preferences among three tasks and verify ranking."""
        tasks = [easy_task, medium_task, hard_task]

        # Compare all pairs, twice each direction
        pairs = []
        for i, t1 in enumerate(tasks):
            for t2 in tasks[i + 1:]:
                pairs.extend([(t1, t2), (t2, t1)])

        comparisons = measure_binary_preferences(
            model=model,
            pairs=pairs,
            builder=binary_builder,
            temperature=1.0,
        )

        assert len(comparisons) >= 4

        data = PairwiseData.from_comparisons(comparisons, tasks)
        result = fit_thurstonian(data)

        assert result.converged
        ranking = result.ranking()
        assert len(ranking) == 3

        # Verify utilities are different (not all collapsed to same value)
        utilities = [result.utility(t) for t in tasks]
        assert max(utilities) - min(utilities) > 0.1

    def test_preference_probability_matches_data(
        self, model, binary_builder, easy_task, hard_task
    ):
        """Fitted preference probability should roughly match empirical rate."""
        tasks = [easy_task, hard_task]

        # More samples for better estimate
        pairs = [(easy_task, hard_task)] * 5

        comparisons = measure_binary_preferences(
            model=model,
            pairs=pairs,
            builder=binary_builder,
            temperature=0.5,
        )

        assert len(comparisons) >= 3

        # Calculate empirical win rate
        easy_wins = sum(1 for c in comparisons if c.choice == "a")
        empirical_rate = easy_wins / len(comparisons)

        data = PairwiseData.from_comparisons(comparisons, tasks)
        result = fit_thurstonian(data)

        # Fitted probability should be in same direction as empirical
        fitted_prob = result.preference_probability(easy_task, hard_task)

        if empirical_rate > 0.5:
            assert fitted_prob > 0.5
        elif empirical_rate < 0.5:
            assert fitted_prob < 0.5
        # If exactly 0.5, either direction is fine


class TestPairwiseDataFromRealMeasurements:
    """Test PairwiseData aggregation with real measurements."""

    def test_aggregates_repeated_comparisons(
        self, model, binary_builder, easy_task, hard_task
    ):
        """Repeated comparisons should accumulate in win matrix."""
        tasks = [easy_task, hard_task]
        pairs = [(easy_task, hard_task)] * 4

        comparisons = measure_binary_preferences(
            model=model,
            pairs=pairs,
            builder=binary_builder,
            temperature=0.0,  # Deterministic
        )

        data = PairwiseData.from_comparisons(comparisons, tasks)

        # Total comparisons should equal number of valid responses
        total = data.wins.sum()
        assert total == len(comparisons)

        # With temperature=0, all responses should be the same
        # So one cell should have all the wins
        assert data.wins[0, 1] == len(comparisons) or data.wins[1, 0] == len(comparisons)

    def test_both_directions_counted(
        self, model, binary_builder, easy_task, hard_task
    ):
        """Comparisons in both directions should be counted separately."""
        tasks = [easy_task, hard_task]
        pairs = [(easy_task, hard_task), (hard_task, easy_task)]

        comparisons = measure_binary_preferences(
            model=model,
            pairs=pairs,
            builder=binary_builder,
            temperature=0.0,
        )

        assert len(comparisons) == 2

        data = PairwiseData.from_comparisons(comparisons, tasks)

        # Should have exactly 2 comparisons total
        assert data.wins.sum() == 2
        assert data.total_comparisons(easy_task) == 2
        assert data.total_comparisons(hard_task) == 2


class TestVerbosePipeline:
    """Verbose test to inspect pipeline at each step. Run with pytest -s."""

    def test_full_pipeline_verbose(self, model, binary_builder, easy_task, medium_task, hard_task):
        """Run full pipeline with detailed output at each step."""
        tasks = [easy_task, medium_task, hard_task]

        # Step 1: Show tasks
        print("\n" + "=" * 60)
        print("TASKS")
        print("=" * 60)
        for t in tasks:
            print(f"  [{t.id}] {t.prompt}")

        # Step 2: Show example prompt
        print("\n" + "=" * 60)
        print("EXAMPLE PROMPT")
        print("=" * 60)
        example_prompt = binary_builder.build(tasks[0], tasks[1])
        for msg in example_prompt.messages:
            print(f"\n[{msg['role']}]")
            content = msg['content']
            print(content[:500] + "..." if len(content) > 500 else content)

        # Step 3: Measure all pairs
        print("\n" + "=" * 60)
        print("MEASUREMENTS")
        print("=" * 60)

        comparisons = []
        pairs = [(tasks[0], tasks[1]), (tasks[1], tasks[2]), (tasks[0], tasks[2])]

        for t1, t2 in pairs:
            prompt = binary_builder.build(t1, t2)
            response = model.generate(prompt.messages, temperature=0.0)
            result = prompt.measurer.parse(response, prompt)
            winner = t1.id if result.result.choice == "a" else t2.id

            print(f"\n  {t1.id} vs {t2.id}")
            print(f"    Raw response: {response.strip()!r}")
            print(f"    Parsed choice: {result.result.choice} → winner: {winner}")

            comparisons.append(result.result)

        # Step 4: Show win matrix
        print("\n" + "=" * 60)
        print("WIN MATRIX")
        print("=" * 60)

        data = PairwiseData.from_comparisons(comparisons, tasks)

        print("\n         ", end="")
        for t in tasks:
            print(f"{t.id:>10}", end="")
        print()

        for i, t1 in enumerate(tasks):
            print(f"{t1.id:>10}", end="")
            for j in range(len(tasks)):
                print(f"{data.wins[i, j]:>10}", end="")
            print()

        # Step 5: Fit model
        print("\n" + "=" * 60)
        print("THURSTONIAN FIT")
        print("=" * 60)

        result = fit_thurstonian(data)

        print(f"\n  Converged: {result.converged}")
        print(f"  Neg log-likelihood: {result.neg_log_likelihood:.4f}")
        print("\n  Fitted parameters (μ ± σ):")
        for task in tasks:
            print(f"    {task.id:>10}: {result.utility(task):>8.3f} ± {result.uncertainty(task):.3f}")

        # Step 6: Ranking
        print("\n" + "=" * 60)
        print("RANKING")
        print("=" * 60)
        for i, task in enumerate(result.ranking(), 1):
            nu = result.normalized_utility(task)
            print(f"  {i}. {task.id} (μ = {result.utility(task):.3f}, normalized = {nu:.3f})")

        # Step 7: Preference probabilities
        print("\n" + "=" * 60)
        print("PREFERENCE PROBABILITIES")
        print("=" * 60)
        for i, t1 in enumerate(tasks):
            for t2 in tasks[i + 1:]:
                p = result.preference_probability(t1, t2)
                print(f"  P({t1.id} > {t2.id}) = {p:.3f}")

        print()

        # Assertions to make it a valid test
        assert result.converged
        assert len(result.ranking()) == 3

    def test_revealed_preference_verbose(self, completion_model, revealed_builder, easy_task, medium_task, hard_task):
        """Run revealed preference pipeline - model completes the task it prefers."""
        tasks = [easy_task, medium_task, hard_task]

        # Step 1: Show tasks
        print("\n" + "=" * 60)
        print("TASKS (REVEALED PREFERENCE)")
        print("=" * 60)
        for t in tasks:
            print(f"  [{t.id}] {t.prompt}")

        # Step 2: Show example prompt
        print("\n" + "=" * 60)
        print("EXAMPLE PROMPT")
        print("=" * 60)
        example_prompt = revealed_builder.build(tasks[0], tasks[1])
        for msg in example_prompt.messages:
            print(f"\n[{msg['role']}]")
            content = msg['content']
            print(content[:500] + "..." if len(content) > 500 else content)

        # Step 3: Measure all pairs
        print("\n" + "=" * 60)
        print("MEASUREMENTS (model completes chosen task)")
        print("=" * 60)

        comparisons = []
        pairs = [(tasks[0], tasks[1]), (tasks[1], tasks[2]), (tasks[0], tasks[2])]

        for t1, t2 in pairs:
            prompt = revealed_builder.build(t1, t2)
            response = completion_model.generate(prompt.messages, temperature=0.0)
            result = prompt.measurer.parse(response, prompt)
            winner = t1.id if result.result.choice == "a" else t2.id

            print(f"\n  {t1.id} vs {t2.id}")
            print(f"    Completion: {response.strip()[:100]}...")
            print(f"    Inferred choice: {result.result.choice} → chose to complete: {winner}")

            comparisons.append(result.result)

        # Step 4: Show win matrix
        print("\n" + "=" * 60)
        print("WIN MATRIX")
        print("=" * 60)

        data = PairwiseData.from_comparisons(comparisons, tasks)

        print("\n         ", end="")
        for t in tasks:
            print(f"{t.id:>10}", end="")
        print()

        for i, t1 in enumerate(tasks):
            print(f"{t1.id:>10}", end="")
            for j in range(len(tasks)):
                print(f"{data.wins[i, j]:>10}", end="")
            print()

        # Step 5: Fit model
        print("\n" + "=" * 60)
        print("THURSTONIAN FIT")
        print("=" * 60)

        result = fit_thurstonian(data)

        print(f"\n  Converged: {result.converged}")
        print(f"  Neg log-likelihood: {result.neg_log_likelihood:.4f}")
        print("\n  Fitted parameters (μ ± σ):")
        for task in tasks:
            print(f"    {task.id:>10}: {result.utility(task):>8.3f} ± {result.uncertainty(task):.3f}")

        # Step 6: Ranking
        print("\n" + "=" * 60)
        print("RANKING")
        print("=" * 60)
        for i, task in enumerate(result.ranking(), 1):
            nu = result.normalized_utility(task)
            print(f"  {i}. {task.id} (μ = {result.utility(task):.3f}, normalized = {nu:.3f})")

        print()

        assert result.converged
        assert len(result.ranking()) == 3



