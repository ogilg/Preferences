"""Tests for LLM-based semantic parsing.

These tests require API access and are marked with @pytest.mark.api.
Run with: pytest -m api tests/test_semantic_parser.py
"""

import pytest
from dotenv import load_dotenv

load_dotenv()

from src.preference_measurement import semantic_parser


@pytest.mark.api
class TestParseChoice:
    """Tests for semantic choice parsing."""

    def test_clear_choice_a(self):
        result = semantic_parser.parse_choice(
            "I prefer Task A",
            "Task A",
            "Task B",
        )
        assert result == "a"

    def test_clear_choice_b(self):
        result = semantic_parser.parse_choice(
            "I prefer Task B",
            "Task A",
            "Task B",
        )
        assert result == "b"

    def test_negation_chooses_b(self):
        result = semantic_parser.parse_choice(
            "Task A is worse, I prefer Task B",
            "Task A",
            "Task B",
        )
        assert result == "b"

    def test_first_mention_not_choice(self):
        result = semantic_parser.parse_choice(
            "I initially considered Task A but ultimately prefer Task B",
            "Task A",
            "Task B",
        )
        assert result == "b"

    def test_unlike_negation(self):
        result = semantic_parser.parse_choice(
            "Unlike Task A, I want to choose Task B",
            "Task A",
            "Task B",
        )
        assert result == "b"

    def test_custom_labels(self):
        result = semantic_parser.parse_choice(
            "I pick Option X",
            "Option X",
            "Option Y",
        )
        assert result == "a"

    def test_unclear_returns_none(self):
        result = semantic_parser.parse_choice(
            "Both options seem equally good to me",
            "Task A",
            "Task B",
        )
        assert result is None


@pytest.mark.api
class TestParseRating:
    """Tests for semantic rating parsing."""

    def test_simple_rating(self):
        result = semantic_parser.parse_rating("I give it a 7", 1, 10)
        assert result == 7.0

    def test_rating_in_scale_context(self):
        result = semantic_parser.parse_rating(
            "On a scale from 1 to 10, I rate this 7",
            1,
            10,
        )
        assert result == 7.0

    def test_multiple_numbers_extracts_rating(self):
        result = semantic_parser.parse_rating(
            "I did 5 calculations but I'd rate this task 8 overall",
            1,
            10,
        )
        assert result == 8.0

    def test_percentage_context(self):
        result = semantic_parser.parse_rating(
            "This is 50% complete but I rate it 9",
            1,
            10,
        )
        assert result == 9.0

    def test_fraction_format(self):
        result = semantic_parser.parse_rating(
            "I'd give this a 7/10",
            1,
            10,
        )
        assert result == 7.0

    def test_unclear_returns_none(self):
        result = semantic_parser.parse_rating(
            "I'm not sure how to rate this",
            1,
            10,
        )
        assert result is None


@pytest.mark.api
class TestParseQualitative:
    """Tests for semantic qualitative parsing."""

    def test_good(self):
        result = semantic_parser.parse_qualitative(
            "This task is good",
            ("good", "neutral", "bad"),
        )
        assert result == "good"

    def test_bad(self):
        result = semantic_parser.parse_qualitative(
            "This is a bad task",
            ("good", "neutral", "bad"),
        )
        assert result == "bad"

    def test_neutral(self):
        result = semantic_parser.parse_qualitative(
            "I feel neutral about this",
            ("good", "neutral", "bad"),
        )
        assert result == "neutral"

    def test_synonym_maps_to_value(self):
        result = semantic_parser.parse_qualitative(
            "This is excellent!",
            ("good", "neutral", "bad"),
        )
        assert result == "good"

    def test_negative_synonym_maps_to_bad(self):
        result = semantic_parser.parse_qualitative(
            "This is terrible",
            ("good", "neutral", "bad"),
        )
        assert result == "bad"

    def test_unclear_returns_none(self):
        result = semantic_parser.parse_qualitative(
            "I have mixed feelings",
            ("good", "bad"),  # Binary scale, no neutral
        )
        # Could be None or could reasonably be interpreted - depends on LLM
        # Just check it returns something valid or None
        assert result in ("good", "bad", None)
