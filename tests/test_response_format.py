import pytest
from dotenv import load_dotenv

load_dotenv()

from src.preference_measurement.response_format import (
    RegexChoiceFormat,
    XMLChoiceFormat,
    CompletionChoiceFormat,
    RegexRatingFormat,
    XMLRatingFormat,
    ToolUseChoiceFormat,
    ToolUseRatingFormat,
    RegexQualitativeFormat,
    XMLQualitativeFormat,
    ToolUseQualitativeFormat,
)


class TestRegexChoiceFormat:
    """Tests for RegexChoiceFormat._extract_choice and parse."""

    def test_extracts_task_a(self):
        fmt = RegexChoiceFormat()
        assert fmt._extract_choice("Task A") == "a"

    def test_extracts_task_b(self):
        fmt = RegexChoiceFormat()
        assert fmt._extract_choice("Task B") == "b"

    def test_case_insensitive_lowercase(self):
        fmt = RegexChoiceFormat()
        assert fmt._extract_choice("task a") == "a"

    def test_case_insensitive_uppercase(self):
        fmt = RegexChoiceFormat()
        assert fmt._extract_choice("TASK B") == "b"

    def test_case_insensitive_mixed(self):
        fmt = RegexChoiceFormat()
        assert fmt._extract_choice("tAsK a") == "a"

    def test_with_surrounding_text(self):
        fmt = RegexChoiceFormat()
        assert fmt._extract_choice("I prefer Task A because it's more interesting") == "a"

    def test_with_surrounding_whitespace(self):
        fmt = RegexChoiceFormat()
        assert fmt._extract_choice("  Task B  ") == "b"

    def test_both_labels_returns_first_a_before_b(self):
        fmt = RegexChoiceFormat()
        assert fmt._extract_choice("Task A is better than Task B") == "a"

    def test_both_labels_returns_first_b_before_a(self):
        fmt = RegexChoiceFormat()
        assert fmt._extract_choice("Task B is better than Task A") == "b"

    def test_neither_label_returns_none(self):
        fmt = RegexChoiceFormat()
        assert fmt._extract_choice("I like both options") is None

    def test_empty_string_returns_none(self):
        fmt = RegexChoiceFormat()
        assert fmt._extract_choice("") is None

    def test_whitespace_only_returns_none(self):
        fmt = RegexChoiceFormat()
        assert fmt._extract_choice("   ") is None

    def test_partial_match_not_matched(self):
        # "TaskA" without space should not match "Task A"
        fmt = RegexChoiceFormat()
        assert fmt._extract_choice("TaskA") is None

    def test_custom_labels(self):
        fmt = RegexChoiceFormat(task_a_label="Option X", task_b_label="Option Y")
        assert fmt._extract_choice("Option X") == "a"
        assert fmt._extract_choice("Option Y") == "b"

    def test_custom_labels_case_insensitive(self):
        fmt = RegexChoiceFormat(task_a_label="First", task_b_label="Second")
        assert fmt._extract_choice("FIRST") == "a"
        assert fmt._extract_choice("second") == "b"

    def test_special_regex_chars_in_labels(self):
        fmt = RegexChoiceFormat(task_a_label="Task (A)", task_b_label="Task (B)")
        assert fmt._extract_choice("Task (A)") == "a"
        assert fmt._extract_choice("Task (B)") == "b"

    def test_parse_returns_lowercase_a(self):
        fmt = RegexChoiceFormat()
        assert fmt.parse("Task A") == "a"

    def test_parse_returns_lowercase_b(self):
        fmt = RegexChoiceFormat()
        assert fmt.parse("Task B") == "b"

    def test_parse_raises_on_no_match(self):
        fmt = RegexChoiceFormat()
        with pytest.raises(ValueError, match="Could not parse choice"):
            fmt.parse("neither option")

    def test_format_instruction(self):
        fmt = RegexChoiceFormat()
        instruction = fmt.format_instruction()
        assert "Task A" in instruction
        assert "Task B" in instruction


class TestXMLChoiceFormat:
    """Tests for XMLChoiceFormat._extract_choice and parse."""

    def test_extracts_task_a(self):
        fmt = XMLChoiceFormat()
        assert fmt._extract_choice("<choice>Task A</choice>") == "a"

    def test_extracts_task_b(self):
        fmt = XMLChoiceFormat()
        assert fmt._extract_choice("<choice>Task B</choice>") == "b"

    def test_case_insensitive_content(self):
        fmt = XMLChoiceFormat()
        assert fmt._extract_choice("<choice>task a</choice>") == "a"

    def test_case_insensitive_tags(self):
        fmt = XMLChoiceFormat()
        assert fmt._extract_choice("<CHOICE>Task A</CHOICE>") == "a"

    def test_whitespace_inside_tags(self):
        fmt = XMLChoiceFormat()
        assert fmt._extract_choice("<choice>  Task A  </choice>") == "a"

    def test_with_surrounding_text(self):
        fmt = XMLChoiceFormat()
        assert fmt._extract_choice("I choose <choice>Task B</choice> because...") == "b"

    def test_custom_tag(self):
        fmt = XMLChoiceFormat(tag="answer")
        assert fmt._extract_choice("<answer>Task A</answer>") == "a"

    def test_custom_labels(self):
        fmt = XMLChoiceFormat(task_a_label="First", task_b_label="Second")
        assert fmt._extract_choice("<choice>First</choice>") == "a"
        assert fmt._extract_choice("<choice>Second</choice>") == "b"

    def test_no_tag_returns_none(self):
        fmt = XMLChoiceFormat()
        assert fmt._extract_choice("Task A") is None

    def test_wrong_tag_returns_none(self):
        fmt = XMLChoiceFormat(tag="choice")
        assert fmt._extract_choice("<answer>Task A</answer>") is None

    def test_invalid_content_returns_none(self):
        fmt = XMLChoiceFormat()
        assert fmt._extract_choice("<choice>Task C</choice>") is None

    def test_empty_tag_returns_none(self):
        fmt = XMLChoiceFormat()
        assert fmt._extract_choice("<choice></choice>") is None

    def test_empty_string_returns_none(self):
        fmt = XMLChoiceFormat()
        assert fmt._extract_choice("") is None

    def test_malformed_xml_missing_close(self):
        fmt = XMLChoiceFormat()
        assert fmt._extract_choice("<choice>Task A") is None

    def test_malformed_xml_missing_open(self):
        fmt = XMLChoiceFormat()
        assert fmt._extract_choice("Task A</choice>") is None

    def test_parse_returns_lowercase(self):
        fmt = XMLChoiceFormat()
        assert fmt.parse("<choice>Task A</choice>") == "a"

    def test_parse_raises_on_no_match(self):
        fmt = XMLChoiceFormat()
        with pytest.raises(ValueError, match="Could not parse choice"):
            fmt.parse("no xml here")

    def test_format_instruction(self):
        fmt = XMLChoiceFormat()
        instruction = fmt.format_instruction()
        assert "<choice>" in instruction
        assert "Task A" in instruction


class TestCompletionChoiceFormat:
    """Tests for CompletionChoiceFormat._extract_choice and parse."""

    def test_extracts_task_a_prefix(self):
        fmt = CompletionChoiceFormat()
        assert fmt._extract_choice("Task A: Here is my response") == "a"

    def test_extracts_task_b_prefix(self):
        fmt = CompletionChoiceFormat()
        assert fmt._extract_choice("Task B: Here is my response") == "b"

    def test_case_insensitive(self):
        fmt = CompletionChoiceFormat()
        assert fmt._extract_choice("task a: lowercase") == "a"
        assert fmt._extract_choice("TASK B: uppercase") == "b"

    def test_both_present_returns_first_a(self):
        fmt = CompletionChoiceFormat()
        assert fmt._extract_choice("Task A then Task B") == "a"

    def test_both_present_returns_first_b(self):
        fmt = CompletionChoiceFormat()
        assert fmt._extract_choice("Task B then Task A") == "b"

    def test_neither_present_returns_none(self):
        fmt = CompletionChoiceFormat()
        assert fmt._extract_choice("Some other text") is None

    def test_empty_string_returns_none(self):
        fmt = CompletionChoiceFormat()
        assert fmt._extract_choice("") is None

    def test_custom_labels(self):
        fmt = CompletionChoiceFormat(task_a_label="Option 1", task_b_label="Option 2")
        assert fmt._extract_choice("Option 1: my choice") == "a"
        assert fmt._extract_choice("Option 2: my choice") == "b"

    def test_partial_match_does_not_match(self):
        # "TaskA" does not contain "task a" as substring (different spacing)
        fmt = CompletionChoiceFormat()
        assert fmt._extract_choice("TaskA") is None

    def test_parse_returns_lowercase(self):
        fmt = CompletionChoiceFormat()
        assert fmt.parse("Task A: response") == "a"

    def test_parse_raises_on_no_match(self):
        fmt = CompletionChoiceFormat()
        with pytest.raises(ValueError, match="Could not parse choice"):
            fmt.parse("no task label here")

    def test_format_instruction(self):
        fmt = CompletionChoiceFormat()
        instruction = fmt.format_instruction()
        assert "Task A:" in instruction
        assert "Task B:" in instruction


class TestRegexRatingFormat:
    """Tests for RegexRatingFormat._extract_number and parse."""

    def test_extracts_integer(self):
        fmt = RegexRatingFormat()
        assert fmt._extract_number("7") == 7.0

    def test_extracts_float(self):
        fmt = RegexRatingFormat()
        assert fmt._extract_number("7.5") == 7.5

    def test_extracts_from_surrounding_text(self):
        fmt = RegexRatingFormat()
        assert fmt._extract_number("I rate this a 7") == 7.0

    def test_extracts_first_number_when_multiple(self):
        fmt = RegexRatingFormat()
        assert fmt._extract_number("Rating: 7 out of 10") == 7.0

    def test_extracts_negative_number(self):
        fmt = RegexRatingFormat()
        assert fmt._extract_number("-3") == -3.0

    def test_extracts_negative_float(self):
        fmt = RegexRatingFormat()
        assert fmt._extract_number("-3.5") == -3.5

    def test_extracts_decimal_without_leading_zero(self):
        fmt = RegexRatingFormat()
        assert fmt._extract_number(".5") == 0.5

    def test_no_number_returns_none(self):
        fmt = RegexRatingFormat()
        assert fmt._extract_number("no numbers here") is None

    def test_empty_string_returns_none(self):
        fmt = RegexRatingFormat()
        assert fmt._extract_number("") is None

    def test_whitespace_only_returns_none(self):
        fmt = RegexRatingFormat()
        assert fmt._extract_number("   ") is None

    def test_parse_returns_float(self):
        fmt = RegexRatingFormat()
        assert fmt.parse("7") == 7.0
        assert isinstance(fmt.parse("7"), float)

    def test_parse_raises_on_no_number(self):
        fmt = RegexRatingFormat()
        with pytest.raises(ValueError, match="Could not extract number"):
            fmt.parse("no number")

    def test_format_instruction_includes_scale(self):
        fmt = RegexRatingFormat(scale_min=1, scale_max=10)
        instruction = fmt.format_instruction()
        assert "1" in instruction
        assert "10" in instruction

    def test_custom_scale(self):
        fmt = RegexRatingFormat(scale_min=0, scale_max=100)
        instruction = fmt.format_instruction()
        assert "0" in instruction
        assert "100" in instruction


class TestXMLRatingFormat:
    """Tests for XMLRatingFormat._extract_number and parse."""

    def test_extracts_integer(self):
        fmt = XMLRatingFormat()
        assert fmt._extract_number("<rating>7</rating>") == 7.0

    def test_extracts_float(self):
        fmt = XMLRatingFormat()
        assert fmt._extract_number("<rating>7.5</rating>") == 7.5

    def test_extracts_negative(self):
        fmt = XMLRatingFormat()
        assert fmt._extract_number("<rating>-3</rating>") == -3.0

    def test_whitespace_inside_tags(self):
        fmt = XMLRatingFormat()
        assert fmt._extract_number("<rating>  7  </rating>") == 7.0

    def test_with_surrounding_text(self):
        fmt = XMLRatingFormat()
        assert fmt._extract_number("My rating is <rating>7</rating> because...") == 7.0

    def test_custom_tag(self):
        fmt = XMLRatingFormat(tag="score")
        assert fmt._extract_number("<score>7</score>") == 7.0

    def test_wrong_tag_returns_none(self):
        fmt = XMLRatingFormat(tag="rating")
        assert fmt._extract_number("<score>7</score>") is None

    def test_no_tag_returns_none(self):
        fmt = XMLRatingFormat()
        assert fmt._extract_number("7") is None

    def test_empty_tag_returns_none(self):
        fmt = XMLRatingFormat()
        assert fmt._extract_number("<rating></rating>") is None

    def test_non_numeric_content_returns_none(self):
        fmt = XMLRatingFormat()
        assert fmt._extract_number("<rating>seven</rating>") is None

    def test_empty_string_returns_none(self):
        fmt = XMLRatingFormat()
        assert fmt._extract_number("") is None

    def test_malformed_xml_returns_none(self):
        fmt = XMLRatingFormat()
        assert fmt._extract_number("<rating>7") is None

    def test_parse_returns_float(self):
        fmt = XMLRatingFormat()
        assert fmt.parse("<rating>7</rating>") == 7.0

    def test_parse_raises_on_no_match(self):
        fmt = XMLRatingFormat()
        with pytest.raises(ValueError, match="Could not extract number"):
            fmt.parse("no xml")

    def test_format_instruction_includes_tag(self):
        fmt = XMLRatingFormat(scale_min=1, scale_max=10)
        instruction = fmt.format_instruction()
        assert "<rating>" in instruction


class TestToolUseChoiceFormat:
    """Tests for ToolUseChoiceFormat.parse."""

    def test_parses_task_a(self):
        fmt = ToolUseChoiceFormat()
        assert fmt.parse('{"choice": "Task A"}') == "a"

    def test_parses_task_b(self):
        fmt = ToolUseChoiceFormat()
        assert fmt.parse('{"choice": "Task B"}') == "b"

    def test_custom_labels(self):
        fmt = ToolUseChoiceFormat(task_a_label="First", task_b_label="Second")
        assert fmt.parse('{"choice": "First"}') == "a"
        assert fmt.parse('{"choice": "Second"}') == "b"

    def test_invalid_json_raises(self):
        fmt = ToolUseChoiceFormat()
        with pytest.raises(ValueError, match="Could not parse choice"):
            fmt.parse("not json")

    def test_missing_choice_key_raises(self):
        fmt = ToolUseChoiceFormat()
        with pytest.raises(ValueError, match="Could not parse choice"):
            fmt.parse('{"wrong_key": "Task A"}')

    def test_invalid_choice_value_raises(self):
        fmt = ToolUseChoiceFormat()
        with pytest.raises(ValueError, match="Could not parse choice"):
            fmt.parse('{"choice": "Task C"}')

    def test_non_string_choice_raises(self):
        fmt = ToolUseChoiceFormat()
        with pytest.raises(ValueError, match="Could not parse choice"):
            fmt.parse('{"choice": 1}')

    def test_empty_string_raises(self):
        fmt = ToolUseChoiceFormat()
        with pytest.raises(ValueError, match="Could not parse choice"):
            fmt.parse("")

    def test_tools_property_returns_tool_definition(self):
        fmt = ToolUseChoiceFormat()
        tools = fmt.tools
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "submit_choice"

    def test_tools_include_custom_labels(self):
        fmt = ToolUseChoiceFormat(task_a_label="X", task_b_label="Y")
        tools = fmt.tools
        enum = tools[0]["function"]["parameters"]["properties"]["choice"]["enum"]
        assert "X" in enum
        assert "Y" in enum

    def test_format_instruction(self):
        fmt = ToolUseChoiceFormat()
        instruction = fmt.format_instruction()
        assert "submit_choice" in instruction


class TestToolUseRatingFormat:
    """Tests for ToolUseRatingFormat.parse."""

    def test_parses_integer(self):
        fmt = ToolUseRatingFormat()
        assert fmt.parse('{"rating": 7}') == 7.0

    def test_parses_float(self):
        fmt = ToolUseRatingFormat()
        assert fmt.parse('{"rating": 7.5}') == 7.5

    def test_parses_negative(self):
        fmt = ToolUseRatingFormat()
        assert fmt.parse('{"rating": -3}') == -3.0

    def test_invalid_json_raises(self):
        fmt = ToolUseRatingFormat()
        with pytest.raises(ValueError, match="Could not extract number"):
            fmt.parse("not json")

    def test_missing_rating_key_raises(self):
        fmt = ToolUseRatingFormat()
        with pytest.raises(ValueError, match="Could not extract number"):
            fmt.parse('{"wrong_key": 7}')

    def test_non_numeric_rating_raises(self):
        fmt = ToolUseRatingFormat()
        with pytest.raises(ValueError, match="Could not extract number"):
            fmt.parse('{"rating": "seven"}')

    def test_empty_string_raises(self):
        fmt = ToolUseRatingFormat()
        with pytest.raises(ValueError, match="Could not extract number"):
            fmt.parse("")

    def test_tools_property_returns_tool_definition(self):
        fmt = ToolUseRatingFormat()
        tools = fmt.tools
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "submit_rating"

    def test_tools_include_scale_in_description(self):
        fmt = ToolUseRatingFormat(scale_min=1, scale_max=10)
        tools = fmt.tools
        desc = tools[0]["function"]["parameters"]["properties"]["rating"]["description"]
        assert "1" in desc
        assert "10" in desc

    def test_format_instruction(self):
        fmt = ToolUseRatingFormat(scale_min=1, scale_max=10)
        instruction = fmt.format_instruction()
        assert "submit_rating" in instruction
        assert "1" in instruction
        assert "10" in instruction


class TestBaseClassTools:
    """Tests for base class tools property behavior."""

    def test_regex_choice_format_has_no_tools(self):
        fmt = RegexChoiceFormat()
        assert fmt.tools is None

    def test_xml_choice_format_has_no_tools(self):
        fmt = XMLChoiceFormat()
        assert fmt.tools is None

    def test_completion_choice_format_has_no_tools(self):
        fmt = CompletionChoiceFormat()
        assert fmt.tools is None

    def test_regex_rating_format_has_no_tools(self):
        fmt = RegexRatingFormat()
        assert fmt.tools is None

    def test_xml_rating_format_has_no_tools(self):
        fmt = XMLRatingFormat()
        assert fmt.tools is None

    def test_tool_use_choice_format_has_tools(self):
        fmt = ToolUseChoiceFormat()
        assert fmt.tools is not None
        assert len(fmt.tools) > 0

    def test_tool_use_rating_format_has_tools(self):
        fmt = ToolUseRatingFormat()
        assert fmt.tools is not None
        assert len(fmt.tools) > 0


class TestQualitativeFormats:
    """Tests for qualitative rating formats."""

    def test_regex_qualitative_good(self):
        fmt = RegexQualitativeFormat()
        assert fmt.parse("good") == 1.0
        assert fmt.parse("This task is good") == 1.0
        assert fmt.parse("GOOD") == 1.0

    def test_regex_qualitative_neutral(self):
        fmt = RegexQualitativeFormat()
        assert fmt.parse("neutral") == 0.0
        assert fmt.parse("I feel neutral about this") == 0.0

    def test_regex_qualitative_bad(self):
        fmt = RegexQualitativeFormat()
        assert fmt.parse("bad") == -1.0
        assert fmt.parse("This is bad.") == -1.0

    def test_regex_qualitative_case_insensitive(self):
        fmt = RegexQualitativeFormat()
        assert fmt.parse("Good") == 1.0
        assert fmt.parse("NEUTRAL") == 0.0
        assert fmt.parse("Bad") == -1.0

    def test_regex_qualitative_synonym(self):
        # "excellent" is semantically interpreted as "good"
        fmt = RegexQualitativeFormat()
        assert fmt.parse("excellent") == 1.0

    def test_regex_qualitative_word_boundaries(self):
        fmt = RegexQualitativeFormat()
        # Word boundary regex should not match "good" in "goodbye"
        with pytest.raises(ValueError):
            fmt._extract_qualitative("goodbye")

    def test_xml_qualitative_good(self):
        fmt = XMLQualitativeFormat()
        assert fmt.parse("<rating>good</rating>") == 1.0

    def test_xml_qualitative_neutral(self):
        fmt = XMLQualitativeFormat()
        assert fmt.parse("<rating>neutral</rating>") == 0.0

    def test_xml_qualitative_bad(self):
        fmt = XMLQualitativeFormat()
        assert fmt.parse("<rating>bad</rating>") == -1.0

    def test_xml_qualitative_case_insensitive(self):
        fmt = XMLQualitativeFormat()
        assert fmt.parse("<rating>Good</rating>") == 1.0
        assert fmt.parse("<rating>NEUTRAL</rating>") == 0.0
        assert fmt.parse("<rating>Bad</rating>") == -1.0

    def test_xml_qualitative_with_whitespace(self):
        fmt = XMLQualitativeFormat()
        assert fmt.parse("<rating>  good  </rating>") == 1.0

    def test_xml_qualitative_synonym(self):
        # "excellent" is semantically interpreted as "good"
        fmt = XMLQualitativeFormat()
        assert fmt.parse("<rating>excellent</rating>") == 1.0

    def test_xml_qualitative_custom_tag(self):
        fmt = XMLQualitativeFormat(tag="score")
        assert fmt.parse("<score>good</score>") == 1.0

    def test_tool_use_qualitative_good(self):
        fmt = ToolUseQualitativeFormat()
        assert fmt.parse('{"rating": "good"}') == 1.0

    def test_tool_use_qualitative_neutral(self):
        fmt = ToolUseQualitativeFormat()
        assert fmt.parse('{"rating": "neutral"}') == 0.0

    def test_tool_use_qualitative_bad(self):
        fmt = ToolUseQualitativeFormat()
        assert fmt.parse('{"rating": "bad"}') == -1.0

    def test_tool_use_qualitative_case_insensitive(self):
        fmt = ToolUseQualitativeFormat()
        assert fmt.parse('{"rating": "Good"}') == 1.0
        assert fmt.parse('{"rating": "NEUTRAL"}') == 0.0

    def test_tool_use_qualitative_invalid_raises(self):
        # Tool use has enum constraints - invalid values should fail (no semantic parsing)
        fmt = ToolUseQualitativeFormat()
        with pytest.raises(ValueError, match="Could not parse qualitative"):
            fmt.parse('{"rating": "excellent"}')

    def test_tool_use_qualitative_has_tools(self):
        fmt = ToolUseQualitativeFormat()
        assert fmt.tools is not None
        assert len(fmt.tools) > 0
        # Check that the tool has an enum constraint
        tool = fmt.tools[0]
        rating_prop = tool["function"]["parameters"]["properties"]["rating"]
        assert "enum" in rating_prop
        assert set(rating_prop["enum"]) == {"good", "neutral", "bad"}

    def test_qualitative_format_instructions(self):
        regex_fmt = RegexQualitativeFormat()
        xml_fmt = XMLQualitativeFormat()
        tool_fmt = ToolUseQualitativeFormat()

        assert "good" in regex_fmt.format_instruction()
        assert "neutral" in regex_fmt.format_instruction()
        assert "bad" in regex_fmt.format_instruction()

        assert "<rating>" in xml_fmt.format_instruction()

        assert "submit_rating" in tool_fmt.format_instruction()
        assert "good" in tool_fmt.format_instruction()
