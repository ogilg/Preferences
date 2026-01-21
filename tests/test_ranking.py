"""Integration tests for TrueSkill-based ranking measurement.

Tests the full workflow: sampling -> measurement -> TrueSkill fitting.
Uses mock clients to avoid API calls.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import numpy as np
import pytest

from src.task_data import Task, OriginDataset
from src.types import RankingMeasurement, PreferenceType, MeasurementBatch
from src.preference_measurement.response_format import (
    RegexRankingFormat,
    XMLRankingFormat,
    ToolUseRankingFormat,
    get_ranking_response_format,
)
from src.preference_measurement.measurer import RankingMeasurer
from src.prompt_templates import PreTaskRankingPromptBuilder, PostTaskRankingPromptBuilder
from src.prompt_templates.template import pre_task_ranking_template, post_task_ranking_template
from src.trueskill_fitting import TrueSkillResult, fit_trueskill_from_rankings, sample_ranking_groups
from src.measurement_storage import RankingCache
from src.models.openai_compatible import BatchResult


pytestmark = [pytest.mark.measurement, pytest.mark.ranking]


@pytest.fixture
def sample_tasks():
    return [
        Task(prompt=f"Task {i}: Do something interesting", origin=OriginDataset.WILDCHAT, id=f"task_{i}", metadata={})
        for i in range(20)
    ]


@pytest.fixture
def task_lookup(sample_tasks):
    return {t.id: t for t in sample_tasks}


@pytest.fixture
def temp_cache_dir():
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.canonical_model_name = "test-model"
    client.model_name = "test-model"
    return client


@pytest.fixture
def ranking_template():
    return pre_task_ranking_template(
        name="test_ranking_template",
        template=(
            "Rank these tasks from most to least preferred.\n\n"
            "{format_instruction}\n\n"
            "Task A: {task_a}\n"
            "Task B: {task_b}\n"
            "Task C: {task_c}\n"
            "Task D: {task_d}\n"
            "Task E: {task_e}"
        ),
    )


class TestPreTaskRankingE2E:
    """End-to-end test of pre-task ranking workflow."""

    @pytest.mark.asyncio
    async def test_full_ranking_pipeline(self, sample_tasks, task_lookup, temp_cache_dir, mock_client, ranking_template):
        """Full pipeline: sample groups -> measure rankings -> fit TrueSkill."""
        # Sample task groups
        rng = np.random.default_rng(42)
        task_groups = sample_ranking_groups(sample_tasks, n_tasks_per_group=5, n_groups=10, rng=rng)

        assert len(task_groups) == 10
        for group in task_groups:
            assert len(group) == 5
            assert len(set(t.id for t in group)) == 5

        # Mock API responses - consistent ordering (alphabetical by task_id within each group)
        def make_ranking_response(group):
            sorted_indices = sorted(range(5), key=lambda i: group[i].id)
            labels = [chr(65 + i) for i in sorted_indices]
            return " > ".join(labels)

        responses = [[BatchResult(response=make_ranking_response(g), error=None)] for g in task_groups]
        mock_client.generate_batch_async = AsyncMock(side_effect=responses)

        # Build and measure
        task_labels = ("A", "B", "C", "D", "E")
        response_format = get_ranking_response_format(task_labels, "regex")
        builder = PreTaskRankingPromptBuilder(
            measurer=RankingMeasurer(),
            response_format=response_format,
            template=ranking_template,
        )

        # Measure all groups
        measurements = []
        for group in task_groups:
            prompt = builder.build(group)
            result = await mock_client.generate_batch_async([{"messages": prompt.messages}])
            response_text = result[0].response
            ranking = await response_format.parse(response_text)
            measurements.append(RankingMeasurement(
                tasks=group,
                ranking=ranking,
                preference_type=PreferenceType.PRE_TASK_RANKING,
            ))

        assert len(measurements) == 10

        # Fit TrueSkill
        result = fit_trueskill_from_rankings(measurements)

        assert isinstance(result, TrueSkillResult)
        assert result.n_rankings == 10
        assert len(result.tasks) == 20  # All tasks should appear

        # Get tasks by id for comparison
        task_0 = next(t for t in result.tasks if t.id == "task_0")
        task_19 = next(t for t in result.tasks if t.id == "task_19")
        # Lower task IDs should have higher utility (due to alphabetical ordering in responses)
        assert result.utility(task_0) > result.utility(task_19)

        # Test ranking output
        final_ranking = result.ranking()
        assert len(final_ranking) == 20
        # First few should be low-numbered tasks
        assert any(t.id == "task_0" for t in final_ranking[:5])

    @pytest.mark.asyncio
    async def test_ranking_with_cache(self, sample_tasks, task_lookup, temp_cache_dir, mock_client, ranking_template):
        """Test caching behavior: second run should use cached results."""
        with patch.object(RankingCache, "CACHE_DIR", temp_cache_dir):
            cache = RankingCache("test-model")

            rng = np.random.default_rng(42)
            task_groups = sample_ranking_groups(sample_tasks[:10], n_tasks_per_group=5, n_groups=4, rng=rng)

            # Create measurements
            measurements = []
            for group in task_groups:
                ranking = list(range(5))  # Simple ordering: A > B > C > D > E
                measurements.append(RankingMeasurement(
                    tasks=group,
                    ranking=ranking,
                    preference_type=PreferenceType.PRE_TASK_RANKING,
                ))

            # Save to cache
            cache.add(measurements, "test_template", "regex", seed=42)

            # Check what's cached
            existing = cache.get_measured_groups("test_template", "regex", seed=42)
            assert len(existing) == 4

            # Load back and verify
            loaded = cache.get_all_measurements("test_template", "regex", seed=42, task_lookup=task_lookup)
            assert len(loaded) == 4

            # Fit from cached data
            result = fit_trueskill_from_rankings(loaded)
            assert result.n_rankings == 4


class TestPostTaskRankingE2E:
    """End-to-end test of post-task ranking with completions."""

    @pytest.mark.asyncio
    async def test_post_task_ranking_pipeline(self, sample_tasks, mock_client):
        """Full post-task ranking: tasks + completions -> ranking -> fit."""
        tasks = sample_tasks[:10]
        completions = {t.id: f"Completion for {t.id}" for t in tasks}

        template = post_task_ranking_template(
            name="test_post_ranking",
            template=(
                "You completed these tasks. Rank them by how much you enjoyed them.\n\n"
                "{format_instruction}\n\n"
                "Task A: {task_a}\nYour response: {completion_a}\n\n"
                "Task B: {task_b}\nYour response: {completion_b}\n\n"
                "Task C: {task_c}\nYour response: {completion_c}\n\n"
                "Task D: {task_d}\nYour response: {completion_d}\n\n"
                "Task E: {task_e}\nYour response: {completion_e}"
            ),
        )

        task_labels = ("A", "B", "C", "D", "E")
        response_format = get_ranking_response_format(task_labels, "regex")
        builder = PostTaskRankingPromptBuilder(
            measurer=RankingMeasurer(),
            response_format=response_format,
            template=template,
        )

        # Sample groups
        rng = np.random.default_rng(42)
        task_groups = sample_ranking_groups(tasks, n_tasks_per_group=5, n_groups=4, rng=rng)

        # Mock responses
        responses = [[BatchResult(response="A > B > C > D > E", error=None)] for _ in task_groups]
        mock_client.generate_batch_async = AsyncMock(side_effect=responses)

        measurements = []
        for group in task_groups:
            group_completions = [completions[t.id] for t in group]
            prompt = builder.build(group, group_completions)

            # Verify prompt includes completions
            prompt_text = str(prompt.messages)
            assert "Completion for" in prompt_text

            result = await mock_client.generate_batch_async([{"messages": prompt.messages}])
            ranking = await response_format.parse(result[0].response)
            measurements.append(RankingMeasurement(
                tasks=group,
                ranking=ranking,
                preference_type=PreferenceType.POST_TASK_RANKING,
            ))

        result = fit_trueskill_from_rankings(measurements)
        assert result.n_rankings == 4


class TestRankingRefusalHandling:
    """Test that ranking formats handle refusals like other formats."""

    @pytest.mark.asyncio
    async def test_ranking_refusal_returns_refusal_string(self):
        """Ranking format should return 'refusal' for refusal responses, not raise."""
        from unittest.mock import patch, AsyncMock

        task_labels = ("A", "B", "C", "D", "E")
        fmt = RegexRankingFormat(task_labels)

        # Mock refusal judge to return True
        with patch("src.preference_measurement.response_format.refusal_judge") as mock_judge:
            mock_judge.judge_preference_refusal_async = AsyncMock(return_value=True)

            result = await fmt.parse("I cannot rank these tasks as I have no preferences.")
            assert result == "refusal"

    @pytest.mark.asyncio
    async def test_ranking_measurer_creates_refusal_result(self):
        """RankingMeasurer should create RankingRefusal for refusal responses."""
        from unittest.mock import patch, AsyncMock, MagicMock
        from src.preference_measurement.measurer import RankingMeasurer
        from src.types import RankingRefusal, PreferenceType

        tasks = [
            Task(prompt=f"Task {i}", origin=OriginDataset.WILDCHAT, id=f"task_{i}", metadata={})
            for i in range(5)
        ]

        # Create mock prompt with format that returns refusal
        mock_format = MagicMock()
        mock_format.parse = AsyncMock(return_value="refusal")

        mock_prompt = MagicMock()
        mock_prompt.tasks = tasks
        mock_prompt.kind = PreferenceType.PRE_TASK_RANKING
        mock_prompt.response_format = mock_format

        measurer = RankingMeasurer()
        response = await measurer.parse("I have no preferences", mock_prompt)

        assert isinstance(response.result, RankingRefusal)
        assert response.result.tasks == tasks
        assert response.result.preference_type == PreferenceType.PRE_TASK_RANKING


class TestResponseFormatParsing:
    """Test different response formats parse correctly in realistic scenarios."""

    @pytest.mark.asyncio
    async def test_regex_format_realistic_responses(self):
        """Test regex format handles realistic model outputs (no API fallback)."""
        task_labels = ("A", "B", "C", "D", "E")
        fmt = RegexRankingFormat(task_labels)

        # Test cases that should parse with regex alone (no semantic fallback)
        test_cases = [
            ("A > C > B > E > D", [0, 2, 1, 4, 3]),
            ("A, C, B, E, D", [0, 2, 1, 4, 3]),
            ("A C B E D", [0, 2, 1, 4, 3]),
            ("a > c > b > e > d", [0, 2, 1, 4, 3]),
            ("A > B > C > D > E", [0, 1, 2, 3, 4]),
        ]

        for response, expected in test_cases:
            result = await fmt.parse(response)
            assert result == expected, f"Failed for: {response}"

    @pytest.mark.asyncio
    async def test_xml_format_realistic_responses(self):
        """Test XML format handles realistic model outputs."""
        task_labels = ("A", "B", "C", "D", "E")
        fmt = XMLRankingFormat(task_labels)

        test_cases = [
            ("<ranking>A > C > B > E > D</ranking>", [0, 2, 1, 4, 3]),
            ("Here's my ranking:\n<ranking>A, C, B, E, D</ranking>", [0, 2, 1, 4, 3]),
            ("<ranking>\nA > B > C > D > E\n</ranking>", [0, 1, 2, 3, 4]),
        ]

        for response, expected in test_cases:
            result = await fmt.parse(response)
            assert result == expected, f"Failed for: {response}"

    @pytest.mark.asyncio
    async def test_tool_use_format(self):
        """Test tool_use format parses JSON correctly."""
        task_labels = ("A", "B", "C", "D", "E")
        fmt = ToolUseRankingFormat(task_labels)

        test_cases = [
            ('{"ranking": ["A", "C", "B", "E", "D"]}', [0, 2, 1, 4, 3]),
            ('{"ranking": ["a", "b", "c", "d", "e"]}', [0, 1, 2, 3, 4]),
        ]

        for response, expected in test_cases:
            result = await fmt.parse(response)
            assert result == expected

        # Verify tools schema
        tools = fmt.tools
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "submit_ranking"


class TestTrueSkillConvergence:
    """Test TrueSkill fitting converges with enough rankings."""

    def test_trueskill_ordering_stability(self, sample_tasks):
        """With consistent rankings, TrueSkill should produce stable ordering."""
        tasks = sample_tasks[:5]

        # Create 20 consistent rankings where task_0 > task_1 > ... > task_4
        measurements = []
        for _ in range(20):
            measurements.append(RankingMeasurement(
                tasks=tasks,
                ranking=[0, 1, 2, 3, 4],
                preference_type=PreferenceType.PRE_TASK_RANKING,
            ))

        result = fit_trueskill_from_rankings(measurements)

        # Should produce correct ordering
        final_ranking = result.ranking()
        assert [t.id for t in final_ranking] == ["task_0", "task_1", "task_2", "task_3", "task_4"]

        # Uncertainties should be lower with more data
        for task in result.tasks:
            assert result.uncertainty(task) < 8.333  # Lower than default sigma

    def test_trueskill_handles_ties(self, sample_tasks):
        """TrueSkill should handle conflicting rankings gracefully."""
        tasks = sample_tasks[:3]

        # Create conflicting rankings
        measurements = [
            RankingMeasurement(tasks=tasks, ranking=[0, 1, 2], preference_type=PreferenceType.PRE_TASK_RANKING),
            RankingMeasurement(tasks=tasks, ranking=[2, 1, 0], preference_type=PreferenceType.PRE_TASK_RANKING),
            RankingMeasurement(tasks=tasks, ranking=[1, 0, 2], preference_type=PreferenceType.PRE_TASK_RANKING),
        ]

        result = fit_trueskill_from_rankings(measurements)

        # Should still produce a result
        assert result.n_rankings == 3
        assert len(result.ranking()) == 3
        # High uncertainty due to conflicting data
        for task in result.tasks:
            assert result.uncertainty(task) > 3.0


class TestSamplingCoverage:
    """Test that sampling produces balanced task coverage."""

    def test_balanced_coverage(self, sample_tasks):
        """Each task should appear roughly equally across groups."""
        rng = np.random.default_rng(42)
        groups = sample_ranking_groups(sample_tasks, n_tasks_per_group=5, n_groups=40, rng=rng)

        counts = {}
        for group in groups:
            for t in group:
                counts[t.id] = counts.get(t.id, 0) + 1

        # All tasks should appear
        assert len(counts) == 20

        # Expected: 40 groups × 5 tasks / 20 unique = 10 appearances each
        values = list(counts.values())
        assert min(values) >= 5   # At least half of expected
        assert max(values) <= 15  # At most 1.5× expected

    def test_deterministic_with_seed(self, sample_tasks):
        """Same seed should produce same groups."""
        groups1 = sample_ranking_groups(sample_tasks, 5, 10, np.random.default_rng(42))
        groups2 = sample_ranking_groups(sample_tasks, 5, 10, np.random.default_rng(42))

        for g1, g2 in zip(groups1, groups2):
            assert [t.id for t in g1] == [t.id for t in g2]


class TestTrueSkillRevealedCorrelation:
    """E2E test: TrueSkill rankings should correlate with pairwise revealed preferences."""

    def test_trueskill_predicts_pairwise_choices(self, sample_tasks):
        """TrueSkill utilities should predict pairwise comparison outcomes."""
        from scipy.stats import spearmanr

        tasks = sample_tasks[:10]
        # Assign ground truth utilities (task_0 highest, task_9 lowest)
        true_utilities = {t.id: 10 - i for i, t in enumerate(tasks)}

        # Generate rankings consistent with true utilities
        rng = np.random.default_rng(42)
        task_groups = sample_ranking_groups(tasks, n_tasks_per_group=5, n_groups=20, rng=rng)

        measurements = []
        for group in task_groups:
            # Rank by true utility (add small noise for realism)
            noise = rng.normal(0, 0.5, len(group))
            scores = [true_utilities[t.id] + n for t, n in zip(group, noise)]
            ranking = sorted(range(len(group)), key=lambda i: scores[i], reverse=True)
            measurements.append(RankingMeasurement(
                tasks=group,
                ranking=ranking,
                preference_type=PreferenceType.PRE_TASK_RANKING,
            ))

        # Fit TrueSkill
        result = fit_trueskill_from_rankings(measurements)

        # Check correlation between TrueSkill utilities and true utilities
        fitted_utilities = [result.utility(t) for t in tasks]
        true_util_list = [true_utilities[t.id] for t in tasks]
        correlation, p_value = spearmanr(fitted_utilities, true_util_list)

        assert correlation > 0.8, f"Correlation {correlation:.3f} too low"
        assert p_value < 0.01, f"p-value {p_value:.4f} too high"

    def test_trueskill_agrees_with_pairwise_comparisons(self, sample_tasks):
        """TrueSkill ranking should agree with majority of pairwise comparisons."""
        from itertools import combinations

        tasks = sample_tasks[:8]
        # Ground truth: task_0 > task_1 > ... > task_7
        true_ranking = {t.id: 8 - i for i, t in enumerate(tasks)}

        # Generate rankings from ground truth
        rng = np.random.default_rng(123)
        task_groups = sample_ranking_groups(tasks, n_tasks_per_group=4, n_groups=30, rng=rng)

        measurements = []
        for group in task_groups:
            # 90% of the time follow true ranking, 10% add noise
            if rng.random() < 0.9:
                ranking = sorted(range(len(group)), key=lambda i: true_ranking[group[i].id], reverse=True)
            else:
                ranking = list(rng.permutation(len(group)))
            measurements.append(RankingMeasurement(
                tasks=group,
                ranking=ranking,
                preference_type=PreferenceType.PRE_TASK_RANKING,
            ))

        result = fit_trueskill_from_rankings(measurements)

        # Generate all pairwise comparisons from ground truth
        all_pairs = list(combinations(tasks, 2))
        agreements = 0
        for task_a, task_b in all_pairs:
            # Ground truth: higher true_ranking value wins
            gt_winner = task_a.id if true_ranking[task_a.id] > true_ranking[task_b.id] else task_b.id
            # TrueSkill prediction: higher utility wins
            ts_winner = task_a.id if result.utility(task_a) > result.utility(task_b) else task_b.id
            if gt_winner == ts_winner:
                agreements += 1

        agreement_rate = agreements / len(all_pairs)
        assert agreement_rate > 0.85, f"Agreement rate {agreement_rate:.2%} too low"


@pytest.mark.api
class TestSemanticParserAPI:
    """Tests for semantic parser that require API calls."""

    @pytest.mark.asyncio
    async def test_parse_choice_clear_a(self):
        """Parse clear choice for option A."""
        from src.preference_measurement.semantic_parser import parse_choice_async

        result = await parse_choice_async(
            "I would definitely prefer to do Task A. It seems more interesting.",
            "Task A", "Task B"
        )
        assert result == "a"

    @pytest.mark.asyncio
    async def test_parse_choice_clear_b(self):
        """Parse clear choice for option B."""
        from src.preference_measurement.semantic_parser import parse_choice_async

        result = await parse_choice_async(
            "Task B is the one I'd choose without hesitation.",
            "Task A", "Task B"
        )
        assert result == "b"

    @pytest.mark.asyncio
    async def test_parse_choice_refusal(self):
        """Parse refusal to choose."""
        from src.preference_measurement.semantic_parser import parse_choice_async

        result = await parse_choice_async(
            "I don't have personal preferences and cannot choose between tasks.",
            "Task A", "Task B"
        )
        assert result == "refusal"

    @pytest.mark.asyncio
    async def test_parse_choice_with_negation(self):
        """Parse choice expressed through negation."""
        from src.preference_measurement.semantic_parser import parse_choice_async

        result = await parse_choice_async(
            "I wouldn't want to do Task A, so I'll go with the other one.",
            "Task A", "Task B"
        )
        assert result == "b"

    @pytest.mark.asyncio
    async def test_parse_rating_embedded_in_text(self):
        """Parse rating embedded in conversational text."""
        from src.preference_measurement.semantic_parser import parse_rating_async

        result = await parse_rating_async(
            "After thinking about it, I'd give this task a solid 7. It's pretty interesting.",
            scale_min=1, scale_max=10
        )
        assert result == 7.0

    @pytest.mark.asyncio
    async def test_parse_rating_worded(self):
        """Parse rating expressed in words."""
        from src.preference_measurement.semantic_parser import parse_rating_async

        result = await parse_rating_async(
            "I'd rate this eight out of ten.",
            scale_min=1, scale_max=10
        )
        assert result == 8.0

    @pytest.mark.asyncio
    async def test_parse_ranking_natural_language(self):
        """Parse ranking expressed in natural language."""
        from src.preference_measurement.semantic_parser import parse_ranking_async

        result = await parse_ranking_async(
            "I'd most prefer Task C, followed by A, then E, B, and finally D.",
            ("A", "B", "C", "D", "E")
        )
        # C=2, A=0, E=4, B=1, D=3
        assert result == [2, 0, 4, 1, 3]

    @pytest.mark.asyncio
    async def test_parse_ranking_numbered_list(self):
        """Parse ranking as numbered list."""
        from src.preference_measurement.semantic_parser import parse_ranking_async

        result = await parse_ranking_async(
            "My ranking:\n1. Task B\n2. Task D\n3. Task A\n4. Task C\n5. Task E",
            ("A", "B", "C", "D", "E")
        )
        # B=1, D=3, A=0, C=2, E=4
        assert result == [1, 3, 0, 2, 4]

    @pytest.mark.asyncio
    async def test_parse_ranking_unclear_raises(self):
        """Unclear ranking should raise ParseError."""
        from src.preference_measurement.semantic_parser import parse_ranking_async, ParseError

        with pytest.raises(ParseError):
            await parse_ranking_async(
                "I'm not sure how to rank these. They all seem equally interesting.",
                ("A", "B", "C", "D", "E")
            )

    @pytest.mark.asyncio
    async def test_parse_qualitative_synonym(self):
        """Parse qualitative value expressed as synonym."""
        from src.preference_measurement.semantic_parser import parse_qualitative_async

        result = await parse_qualitative_async(
            "I strongly concur with this approach.",
            ("strongly agree", "agree", "neutral", "disagree", "strongly disagree")
        )
        assert result == "strongly agree"

    @pytest.mark.asyncio
    async def test_parse_qualitative_negation(self):
        """Parse qualitative value expressed through negation."""
        from src.preference_measurement.semantic_parser import parse_qualitative_async

        result = await parse_qualitative_async(
            "I don't agree with this at all.",
            ("strongly agree", "agree", "neutral", "disagree", "strongly disagree")
        )
        assert result in ("disagree", "strongly disagree")
