"""Tests for preference measurement - designed from research requirements.

Key requirements from project docs:
1. Self-reported valence: Ask model to report interaction valence AFTER completing task
2. Revealed preferences: Give model a choice between tasks, framed as what THEY prefer
3. Must avoid conflating preference with task difficulty/importance/objective quality
4. Binary choice should emphasize this is about model's own preference, not instrumental value
"""

import pytest
from src.task_data import Task, OriginDataset
from src.types import PreferencePrompt
from src.preferences import (
    BinaryPromptBuilder,
    PreTaskRatingPromptBuilder,
    PostTaskRatingPromptBuilder,
    PromptTemplate,
    BINARY_PLACEHOLDERS,
    binary_template,
    BINARY_CHOICE_TEMPLATE,
    PRE_TASK_RATING_TEMPLATE,
    POST_TASK_RATING_TEMPLATE,
    RegexChoiceFormat,
    RegexRatingFormat,
    XMLChoiceFormat,
    XMLRatingFormat,
    PreferenceType,
)
from src.preferences.measurement import (
    BinaryPreferenceMeasurer,
    TaskScoreMeasurer,
)


class TestPromptTemplate:
    """Tests for PromptTemplate validation and formatting."""

    def test_valid_template_creation(self):
        """Should create template when all required placeholders present."""
        template = PromptTemplate(
            template="A: {task_a}, B: {task_b}, {format_instruction}",
            name="test_template",
            required_placeholders=frozenset({"task_a", "task_b", "format_instruction"}),
        )
        assert template.name == "test_template"

    def test_missing_placeholder_raises(self):
        """Should raise ValueError when template missing required placeholder."""
        with pytest.raises(ValueError, match="missing required placeholders"):
            PromptTemplate(
                template="A: {task_a}, {format_instruction}",  # missing task_b
                name="bad_template",
                required_placeholders=frozenset({"task_a", "task_b", "format_instruction"}),
            )

    def test_format_validates_kwargs(self):
        """Should raise ValueError when format() missing required values."""
        template = PromptTemplate(
            template="A: {task_a}, B: {task_b}",
            name="test",
            required_placeholders=frozenset({"task_a", "task_b"}),
        )
        with pytest.raises(ValueError, match="Missing values"):
            template.format(task_a="foo")  # missing task_b

    def test_format_returns_formatted_string(self):
        """Should return formatted string when all values provided."""
        template = PromptTemplate(
            template="A: {task_a}, B: {task_b}",
            name="test",
            required_placeholders=frozenset({"task_a", "task_b"}),
        )
        result = template.format(task_a="first", task_b="second")
        assert result == "A: first, B: second"

    def test_factory_function_sets_placeholders(self):
        """Factory functions should set correct placeholders."""
        template = binary_template(
            template="Pick: {task_a} or {task_b}? {format_instruction}",
            name="custom_binary",
        )
        assert template.required_placeholders == BINARY_PLACEHOLDERS

    def test_template_is_immutable(self):
        """Template should be frozen (immutable)."""
        template = PromptTemplate(
            template="{task_a}",
            name="test",
            required_placeholders=frozenset({"task_a"}),
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            template.name = "changed"

    def test_tags_stored_as_frozenset(self):
        """Tags should be stored as frozenset."""
        template = PromptTemplate(
            template="{task_a}",
            name="test",
            required_placeholders=frozenset({"task_a"}),
            tags=frozenset({"canonical", "lang:en"}),
        )
        assert "canonical" in template.tags
        assert "lang:en" in template.tags

    def test_tags_dict_parses_key_value_tags(self):
        """tags_dict should parse key:value tags into dict."""
        template = PromptTemplate(
            template="{task_a}",
            name="test",
            required_placeholders=frozenset({"task_a"}),
            tags=frozenset({"canonical", "lang:en", "variant:binary_001"}),
        )
        assert template.tags_dict == {"lang": "en", "variant": "binary_001"}


class TestLoadTemplatesFromYaml:
    """Tests for loading templates from YAML files."""

    def test_loads_templates_from_yaml_file(self, tmp_path):
        """Should load templates from a valid YAML file."""
        from src.preferences.templates import load_templates_from_yaml

        yaml_content = """
- id: "001"
  name: test_template_001
  type: binary
  tags: [canonical]
  template: |
    {format_instruction}
    Task A: {task_a}
    Task B: {task_b}
"""
        yaml_file = tmp_path / "templates.yaml"
        yaml_file.write_text(yaml_content)

        templates = load_templates_from_yaml(yaml_file)

        assert len(templates) == 1
        assert templates[0].name == "test_template_001"
        assert "canonical" in templates[0].tags

    def test_loads_real_template_file(self):
        """Should load the actual binary_choice_v1.yaml file."""
        from pathlib import Path
        from src.preferences.templates import load_templates_from_yaml

        yaml_path = Path(__file__).parent.parent / "src/preferences/templates/data/binary_choice_v1.yaml"
        templates = load_templates_from_yaml(yaml_path)

        assert len(templates) >= 1
        # Check templates have structured tags
        first = templates[0]
        assert "language" in first.tags_dict
        assert "phrasing" in first.tags_dict


def get_all_content(prompt: PreferencePrompt) -> str:
    """Helper to get all message content concatenated for simple assertions."""
    return "\n".join(msg["content"] for msg in prompt.messages)


@pytest.fixture
def sample_task_a():
    return Task(
        prompt="Write a haiku about spring.",
        origin=OriginDataset.WILDCHAT,
        id="task_a",
        metadata={"topic": "creative_writing"},
    )


@pytest.fixture
def sample_task_b():
    return Task(
        prompt="Solve the integral of x^2 dx.",
        origin=OriginDataset.MATH,
        id="task_b",
        metadata={"topic": "calculus"},
    )


@pytest.fixture
def sample_completion_text():
    return "Cherry blossoms fall\nGentle breeze carries petals\nNew life awakens"


class TestBinaryPromptBuilder:
    """Tests for binary choice prompt building."""

    def test_build_creates_valid_prompt(self, sample_task_a, sample_task_b):
        """Built prompt should have correct structure and carry all components."""
        measurer = BinaryPreferenceMeasurer()
        response_format = RegexChoiceFormat()
        builder = BinaryPromptBuilder(
            measurer=measurer,
            preference_type=PreferenceType.PRE_TASK_STATED,
            response_format=response_format,
            template=BINARY_CHOICE_TEMPLATE,
        )
        prompt = builder.build(sample_task_a, sample_task_b)
        prompt_content = get_all_content(prompt)

        # Content includes both tasks
        assert sample_task_a.prompt in prompt_content
        assert sample_task_b.prompt in prompt_content
        # Tasks list populated
        assert sample_task_a in prompt.tasks
        assert sample_task_b in prompt.tasks
        # Components carried through
        assert prompt.kind == PreferenceType.PRE_TASK_STATED
        assert prompt.measurer is measurer
        assert prompt.response_format is response_format

    def test_template_placeholders_are_filled(self, sample_task_a, sample_task_b):
        """Template placeholders should be filled with task content."""
        template = "Task A: {task_a}\nTask B: {task_b}\n{format_instruction}"
        builder = BinaryPromptBuilder(
            measurer=BinaryPreferenceMeasurer(),
            preference_type=PreferenceType.PRE_TASK_STATED,
            response_format=RegexChoiceFormat(),
            template=template,
        )
        prompt = builder.build(sample_task_a, sample_task_b)
        prompt_content = get_all_content(prompt)

        assert "Task A:" in prompt_content
        assert "Task B:" in prompt_content
        assert sample_task_a.prompt in prompt_content
        assert sample_task_b.prompt in prompt_content
        assert prompt.kind == PreferenceType.PRE_TASK_STATED


class TestPreTaskRatingPromptBuilder:
    """Tests for pre-task rating prompt building."""

    def test_build_creates_valid_prompt(self, sample_task_a):
        """Built prompt should have correct structure and carry all components."""
        measurer = TaskScoreMeasurer()
        response_format = RegexRatingFormat()
        builder = PreTaskRatingPromptBuilder(
            measurer=measurer,
            response_format=response_format,
            template=PRE_TASK_RATING_TEMPLATE,
        )
        prompt = builder.build(sample_task_a)
        prompt_content = get_all_content(prompt)

        # Content includes task
        assert sample_task_a.prompt in prompt_content
        # Task in list
        assert sample_task_a in prompt.tasks
        # Components carried through
        assert prompt.kind == PreferenceType.PRE_TASK_STATED
        assert prompt.measurer is measurer
        assert prompt.response_format is response_format

    def test_scale_placeholders_are_filled(self, sample_task_a):
        """Scale placeholders in template should be filled from response format."""
        template = "Rate from {scale_min} to {scale_max}.\nTask: {task}\n{format_instruction}"
        measurer = TaskScoreMeasurer()
        response_format = RegexRatingFormat(scale_min=-5, scale_max=5)
        builder = PreTaskRatingPromptBuilder(
            measurer=measurer,
            response_format=response_format,
            template=template,
        )
        prompt = builder.build(sample_task_a)
        prompt_content = get_all_content(prompt)

        assert "-5" in prompt_content
        assert "5" in prompt_content


class TestPostTaskRatingPromptBuilder:
    """Tests for post-task rating prompt building."""

    def test_build_creates_valid_prompt(self, sample_task_a, sample_completion_text):
        """Built prompt should have correct structure and carry all components."""
        measurer = TaskScoreMeasurer()
        response_format = RegexRatingFormat()
        builder = PostTaskRatingPromptBuilder(
            measurer=measurer,
            response_format=response_format,
            template=POST_TASK_RATING_TEMPLATE,
        )
        prompt = builder.build(sample_task_a, sample_completion_text)
        prompt_content = get_all_content(prompt)

        # Content includes task and completion
        assert sample_task_a.prompt in prompt_content
        assert sample_completion_text in prompt_content
        # Task in list
        assert sample_task_a in prompt.tasks
        # Components carried through
        assert prompt.kind == PreferenceType.POST_TASK_STATED
        assert prompt.measurer is measurer
        assert prompt.response_format is response_format

    def test_scale_placeholders_are_filled(self, sample_task_a, sample_completion_text):
        """Scale placeholders in template should be filled from response format."""
        template = "Rate from {scale_min} to {scale_max}.\n{format_instruction}"
        measurer = TaskScoreMeasurer()
        response_format = RegexRatingFormat(scale_min=0, scale_max=100)
        builder = PostTaskRatingPromptBuilder(
            measurer=measurer,
            response_format=response_format,
            template=template,
        )
        prompt = builder.build(sample_task_a, sample_completion_text)
        prompt_content = get_all_content(prompt)

        assert "0" in prompt_content
        assert "100" in prompt_content


class TestXMLResponseFormats:
    """Tests for XML-based response parsing."""

    def test_xml_choice_format_instruction(self):
        """XMLChoiceFormat should include XML tag instructions."""
        fmt = XMLChoiceFormat()
        instruction = fmt.format_instruction()

        assert "<choice>" in instruction
        assert "</choice>" in instruction

    def test_xml_choice_parse(self):
        """XMLChoiceFormat should parse choice from XML tags."""
        fmt = XMLChoiceFormat()

        assert fmt.parse("<choice>Task A</choice>") == "a"
        assert fmt.parse("<choice>Task B</choice>") == "b"
        assert fmt.parse("I think <choice>Task A</choice> is better") == "a"
        assert fmt.parse("<choice> Task B </choice>") == "b"

    def test_xml_choice_custom_tag(self):
        """XMLChoiceFormat should support custom tag names."""
        fmt = XMLChoiceFormat(tag="answer")

        assert "<answer>" in fmt.format_instruction()
        assert fmt.parse("<answer>Task A</answer>") == "a"

    def test_xml_rating_format_instruction(self):
        """XMLRatingFormat should include XML tag instructions."""
        fmt = XMLRatingFormat()
        instruction = fmt.format_instruction()

        assert "<rating>" in instruction
        assert "</rating>" in instruction

    def test_xml_rating_parse(self):
        """XMLRatingFormat should parse rating from XML tags."""
        fmt = XMLRatingFormat()

        assert fmt.parse("<rating>7</rating>") == 7.0
        assert fmt.parse("<rating>7.5</rating>") == 7.5
        assert fmt.parse("My rating is <rating>8</rating>") == 8.0
        assert fmt.parse("<rating> 9 </rating>") == 9.0

    def test_xml_choice_raises_on_missing_tag(self):
        """XMLChoiceFormat should raise when XML tag is missing."""
        fmt = XMLChoiceFormat()

        with pytest.raises(ValueError):
            fmt.parse("I choose A")

    def test_xml_rating_raises_on_missing_tag(self):
        """XMLRatingFormat should raise when XML tag is missing."""
        fmt = XMLRatingFormat()

        with pytest.raises(ValueError):
            fmt.parse("My rating is 7")

    def test_builder_with_xml_response_format(self, sample_task_a, sample_task_b):
        """Builders should work with custom XML response format."""
        builder = BinaryPromptBuilder(
            measurer=BinaryPreferenceMeasurer(),
            preference_type=PreferenceType.PRE_TASK_STATED,
            response_format=XMLChoiceFormat(),
            template=BINARY_CHOICE_TEMPLATE,
        )

        prompt = builder.build(sample_task_a, sample_task_b)
        assert "<choice>" in get_all_content(prompt)

        response = prompt.measurer.parse("<choice>Task A</choice>", prompt)
        assert response.result.choice == "a"

    def test_rating_builder_with_xml_response_format(self, sample_task_a):
        """Rating builders should work with custom XML response format."""
        builder = PreTaskRatingPromptBuilder(
            measurer=TaskScoreMeasurer(),
            response_format=XMLRatingFormat(),
            template=PRE_TASK_RATING_TEMPLATE,
        )

        prompt = builder.build(sample_task_a)
        assert "<rating>" in get_all_content(prompt)

        response = prompt.measurer.parse("<rating>7</rating>", prompt)
        assert response.result.score == 7.0


class TestToolUseChoiceFormat:
    """Tests for tool use choice format."""

    def test_tools_property_returns_valid_definition(self):
        """ToolUseChoiceFormat should return valid tool definitions."""
        from src.preferences import ToolUseChoiceFormat

        fmt = ToolUseChoiceFormat()

        assert fmt.tools is not None
        assert len(fmt.tools) == 1
        assert fmt.tools[0]["type"] == "function"
        assert fmt.tools[0]["function"]["name"] == "submit_choice"
        assert "choice" in fmt.tools[0]["function"]["parameters"]["properties"]

    def test_format_instruction_mentions_tool(self):
        """Format instruction should reference tool use."""
        from src.preferences import ToolUseChoiceFormat

        fmt = ToolUseChoiceFormat()
        instruction = fmt.format_instruction()

        assert "submit_choice" in instruction

    def test_parse_json_choice_a(self):
        """Should parse JSON with choice A."""
        from src.preferences import ToolUseChoiceFormat

        fmt = ToolUseChoiceFormat()

        assert fmt.parse('{"choice": "Task A"}') == "a"
        assert fmt.parse('{"choice": "task a"}') == "a"

    def test_parse_json_choice_b(self):
        """Should parse JSON with choice B."""
        from src.preferences import ToolUseChoiceFormat

        fmt = ToolUseChoiceFormat()

        assert fmt.parse('{"choice": "Task B"}') == "b"
        assert fmt.parse('{"choice": "task b"}') == "b"

    def test_raises_on_invalid_json(self):
        """Should raise ValueError when JSON parsing fails."""
        from src.preferences import ToolUseChoiceFormat

        fmt = ToolUseChoiceFormat()

        with pytest.raises(ValueError):
            fmt.parse("I choose Task A")  # Not valid JSON

    def test_raises_on_invalid_choice_value(self):
        """Should raise ValueError when choice is not a valid task label."""
        from src.preferences import ToolUseChoiceFormat

        fmt = ToolUseChoiceFormat()

        with pytest.raises(ValueError):
            fmt.parse('{"choice": "Task C"}')

    def test_builder_with_tool_use_format(self, sample_task_a, sample_task_b):
        """BinaryPromptBuilder should work with ToolUseChoiceFormat."""
        from src.preferences import ToolUseChoiceFormat

        fmt = ToolUseChoiceFormat()
        builder = BinaryPromptBuilder(
            measurer=BinaryPreferenceMeasurer(),
            preference_type=PreferenceType.PRE_TASK_STATED,
            response_format=fmt,
            template=BINARY_CHOICE_TEMPLATE,
        )

        prompt = builder.build(sample_task_a, sample_task_b)
        assert "submit_choice" in get_all_content(prompt)

        # Test parsing JSON response
        response = prompt.measurer.parse('{"choice": "Task A"}', prompt)
        assert response.result.choice == "a"


class TestToolUseRatingFormat:
    """Tests for tool use rating format."""

    def test_tools_property_returns_valid_definition(self):
        """ToolUseRatingFormat should return valid tool definitions."""
        from src.preferences import ToolUseRatingFormat

        fmt = ToolUseRatingFormat()

        assert fmt.tools is not None
        assert len(fmt.tools) == 1
        assert fmt.tools[0]["type"] == "function"
        assert fmt.tools[0]["function"]["name"] == "submit_rating"
        assert "rating" in fmt.tools[0]["function"]["parameters"]["properties"]

    def test_tools_include_scale_in_description(self):
        """Tool description should include scale bounds."""
        from src.preferences import ToolUseRatingFormat

        fmt = ToolUseRatingFormat(scale_min=0, scale_max=100)
        desc = fmt.tools[0]["function"]["parameters"]["properties"]["rating"]["description"]

        assert "0" in desc
        assert "100" in desc

    def test_format_instruction_mentions_tool_and_scale(self):
        """Format instruction should reference tool and scale."""
        from src.preferences import ToolUseRatingFormat

        fmt = ToolUseRatingFormat(scale_min=1, scale_max=10)
        instruction = fmt.format_instruction()

        assert "submit_rating" in instruction
        assert "1" in instruction
        assert "10" in instruction

    def test_parse_json_integer_rating(self):
        """Should parse JSON with integer rating."""
        from src.preferences import ToolUseRatingFormat

        fmt = ToolUseRatingFormat()

        assert fmt.parse('{"rating": 7}') == 7.0
        assert fmt.parse('{"rating": 1}') == 1.0
        assert fmt.parse('{"rating": 10}') == 10.0

    def test_parse_json_float_rating(self):
        """Should parse JSON with float rating."""
        from src.preferences import ToolUseRatingFormat

        fmt = ToolUseRatingFormat()

        assert fmt.parse('{"rating": 7.5}') == 7.5
        assert fmt.parse('{"rating": 3.14}') == 3.14

    def test_raises_on_invalid_json(self):
        """Should raise ValueError when JSON parsing fails."""
        from src.preferences import ToolUseRatingFormat

        fmt = ToolUseRatingFormat()

        with pytest.raises(ValueError):
            fmt.parse("My rating is 7")  # Not valid JSON

    def test_raises_on_missing_rating_key(self):
        """Should raise ValueError when rating key is missing."""
        from src.preferences import ToolUseRatingFormat

        fmt = ToolUseRatingFormat()

        with pytest.raises(ValueError):
            fmt.parse('{"score": 7}')  # Wrong key

    def test_raises_on_non_numeric_rating(self):
        """Should raise ValueError when rating is not a number."""
        from src.preferences import ToolUseRatingFormat

        fmt = ToolUseRatingFormat()

        with pytest.raises(ValueError):
            fmt.parse('{"rating": "seven"}')

    def test_builder_with_tool_use_format(self, sample_task_a):
        """PreTaskRatingPromptBuilder should work with ToolUseRatingFormat."""
        from src.preferences import ToolUseRatingFormat

        fmt = ToolUseRatingFormat()
        builder = PreTaskRatingPromptBuilder(
            measurer=TaskScoreMeasurer(),
            response_format=fmt,
            template=PRE_TASK_RATING_TEMPLATE,
        )

        prompt = builder.build(sample_task_a)
        assert "submit_rating" in get_all_content(prompt)

        # Test parsing JSON response
        response = prompt.measurer.parse('{"rating": 8}', prompt)
        assert response.result.score == 8.0


class TestCompletionChoiceFormat:
    """Tests for completion choice format (revealed preference through task completion)."""

    def test_format_instruction(self):
        """Format instruction should ask model to prefix with Task A/B."""
        from src.preferences import CompletionChoiceFormat

        fmt = CompletionChoiceFormat()
        instruction = fmt.format_instruction()

        assert "Task A:" in instruction
        assert "Task B:" in instruction

    def test_parse_task_a_prefix(self):
        """Should parse Task A prefix at start of response."""
        from src.preferences import CompletionChoiceFormat

        fmt = CompletionChoiceFormat()

        assert fmt.parse("Task A: Here is my haiku...") == "a"
        assert fmt.parse("task a: lowercase also works") == "a"
        assert fmt.parse("  Task A: with leading whitespace") == "a"

    def test_parse_task_b_prefix(self):
        """Should parse Task B prefix at start of response."""
        from src.preferences import CompletionChoiceFormat

        fmt = CompletionChoiceFormat()

        assert fmt.parse("Task B: Solving the integral...") == "b"
        assert fmt.parse("task b: lowercase also works") == "b"
        assert fmt.parse("  Task B: with leading whitespace") == "b"

    def test_parse_first_occurrence_wins(self):
        """When both Task A and Task B appear, first one wins."""
        from src.preferences import CompletionChoiceFormat

        fmt = CompletionChoiceFormat()

        # Task A comes first
        assert fmt.parse("Task A: I'll do this because Task B seemed harder") == "a"
        # Task B comes first
        assert fmt.parse("Task B: I chose this over Task A") == "b"

    def test_raises_on_missing_task_indicator(self):
        """Should raise ValueError when no Task A/B indicator found."""
        from src.preferences import CompletionChoiceFormat

        fmt = CompletionChoiceFormat()

        with pytest.raises(ValueError):
            fmt.parse("Here is my response without a task indicator")

        with pytest.raises(ValueError):
            fmt.parse("I chose option A")  # "option A" not "Task A"

    def test_builder_with_completion_format(self, sample_task_a, sample_task_b):
        """BinaryPromptBuilder should work with CompletionChoiceFormat."""
        from src.preferences import CompletionChoiceFormat, BINARY_COMPLETION_TEMPLATE

        fmt = CompletionChoiceFormat()
        builder = BinaryPromptBuilder(
            measurer=BinaryPreferenceMeasurer(),
            preference_type=PreferenceType.PRE_TASK_REVEALED,
            response_format=fmt,
            template=BINARY_COMPLETION_TEMPLATE,
        )

        prompt = builder.build(sample_task_a, sample_task_b)
        prompt_content = get_all_content(prompt)

        # Template should ask model to complete a task
        assert "complete" in prompt_content.lower()
        assert "Task A:" in prompt_content or "task a:" in prompt_content.lower()

        # Test parsing completion response
        response = prompt.measurer.parse("Task A: Cherry blossoms bloom...", prompt)
        assert response.result.choice == "a"
