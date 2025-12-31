import pytest
from src.preferences.measurement.response_format import (
    _parse_tool_json,
    RegexChoiceFormat,
    XMLChoiceFormat,
    CompletionChoiceFormat,
    RegexRatingFormat,
    XMLRatingFormat,
    ToolUseChoiceFormat,
    ToolUseRatingFormat,
)


class TestParseToolJson:
    """Tests for the _parse_tool_json helper function."""

    def test_valid_json_object(self):
        assert _parse_tool_json('{"key": "value"}') == {"key": "value"}

    def test_valid_json_with_numbers(self):
        assert _parse_tool_json('{"rating": 7}') == {"rating": 7}

    def test_valid_json_with_float(self):
        assert _parse_tool_json('{"rating": 7.5}') == {"rating": 7.5}

    def test_returns_none_for_json_array(self):
        assert _parse_tool_json("[1, 2, 3]") is None

    def test_returns_none_for_json_string(self):
        assert _parse_tool_json('"just a string"') is None

    def test_returns_none_for_invalid_json(self):
        assert _parse_tool_json("not json at all") is None

    def test_returns_none_for_empty_string(self):
        assert _parse_tool_json("") is None

    def test_returns_none_for_malformed_json(self):
        assert _parse_tool_json('{"key": }') is None


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

    def test_format_instruction_includes_example(self):
        fmt = XMLRatingFormat(scale_min=1, scale_max=10)
        instruction = fmt.format_instruction()
        assert "<rating>" in instruction
        assert "</rating>" in instruction
        # Mid-point example (5 or 6 depending on rounding)
        assert "5" in instruction or "6" in instruction


class TestToolUseChoiceFormat:
    """Tests for ToolUseChoiceFormat._extract_choice and parse."""

    def test_extracts_task_a(self):
        fmt = ToolUseChoiceFormat()
        assert fmt._extract_choice('{"choice": "Task A"}') == "a"

    def test_extracts_task_b(self):
        fmt = ToolUseChoiceFormat()
        assert fmt._extract_choice('{"choice": "Task B"}') == "b"

    def test_case_insensitive(self):
        fmt = ToolUseChoiceFormat()
        assert fmt._extract_choice('{"choice": "task a"}') == "a"
        assert fmt._extract_choice('{"choice": "TASK B"}') == "b"

    def test_custom_labels(self):
        fmt = ToolUseChoiceFormat(task_a_label="First", task_b_label="Second")
        assert fmt._extract_choice('{"choice": "First"}') == "a"
        assert fmt._extract_choice('{"choice": "Second"}') == "b"

    def test_invalid_json_returns_none(self):
        fmt = ToolUseChoiceFormat()
        assert fmt._extract_choice("not json") is None

    def test_missing_choice_key_returns_none(self):
        fmt = ToolUseChoiceFormat()
        assert fmt._extract_choice('{"wrong_key": "Task A"}') is None

    def test_invalid_choice_value_returns_none(self):
        fmt = ToolUseChoiceFormat()
        assert fmt._extract_choice('{"choice": "Task C"}') is None

    def test_non_string_choice_returns_none(self):
        fmt = ToolUseChoiceFormat()
        assert fmt._extract_choice('{"choice": 1}') is None

    def test_empty_string_returns_none(self):
        fmt = ToolUseChoiceFormat()
        assert fmt._extract_choice("") is None

    def test_parse_returns_lowercase(self):
        fmt = ToolUseChoiceFormat()
        assert fmt.parse('{"choice": "Task A"}') == "a"

    def test_parse_raises_on_invalid(self):
        fmt = ToolUseChoiceFormat()
        with pytest.raises(ValueError, match="Could not parse choice"):
            fmt.parse("invalid")

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
    """Tests for ToolUseRatingFormat._extract_number and parse."""

    def test_extracts_integer(self):
        fmt = ToolUseRatingFormat()
        assert fmt._extract_number('{"rating": 7}') == 7.0

    def test_extracts_float(self):
        fmt = ToolUseRatingFormat()
        assert fmt._extract_number('{"rating": 7.5}') == 7.5

    def test_extracts_negative(self):
        fmt = ToolUseRatingFormat()
        assert fmt._extract_number('{"rating": -3}') == -3.0

    def test_invalid_json_returns_none(self):
        fmt = ToolUseRatingFormat()
        assert fmt._extract_number("not json") is None

    def test_missing_rating_key_returns_none(self):
        fmt = ToolUseRatingFormat()
        assert fmt._extract_number('{"wrong_key": 7}') is None

    def test_non_numeric_rating_returns_none(self):
        fmt = ToolUseRatingFormat()
        assert fmt._extract_number('{"rating": "seven"}') is None

    def test_empty_string_returns_none(self):
        fmt = ToolUseRatingFormat()
        assert fmt._extract_number("") is None

    def test_parse_returns_float(self):
        fmt = ToolUseRatingFormat()
        assert fmt.parse('{"rating": 7}') == 7.0

    def test_parse_raises_on_invalid(self):
        fmt = ToolUseRatingFormat()
        with pytest.raises(ValueError, match="Could not extract number"):
            fmt.parse("invalid")

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
