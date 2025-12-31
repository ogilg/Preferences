"""Tests for visualization components."""

import pytest

from src.preferences.storage import BinaryRunConfig


class TestLoadTemplate:
    """Test BinaryRunConfig.load_template()."""

    def test_load_template_file_not_found(self, tmp_path):
        """Template file doesn't exist."""
        config = BinaryRunConfig(
            template_id="001",
            template_name="test_template_001",
            template_file=str(tmp_path / "nonexistent.yaml"),
            template_tags={},
            model="test-model",
            model_short="test",
            temperature=0.0,
            task_origin="test",
            n_tasks=1,
            task_ids=["a"],
        )
        with pytest.raises(FileNotFoundError):
            config.load_template()

    def test_load_template_name_not_found(self, tmp_path):
        """Template file exists but name not in it."""
        template_file = tmp_path / "templates.yaml"
        template_file.write_text("""
- id: "002"
  name: "other_template_002"
  type: "binary"
  template: "{task_a} {task_b} {format_instruction}"
""")
        config = BinaryRunConfig(
            template_id="001",
            template_name="test_template_001",
            template_file=str(template_file),
            template_tags={},
            model="test-model",
            model_short="test",
            temperature=0.0,
            task_origin="test",
            n_tasks=1,
            task_ids=["a"],
        )
        with pytest.raises(ValueError, match="not found"):
            config.load_template()

    def test_load_template_success(self, tmp_path):
        """Template loads successfully."""
        template_file = tmp_path / "templates.yaml"
        template_file.write_text("""
- id: "001"
  name: "test_template_001"
  type: "binary"
  template: "Choose {task_a} or {task_b}. {format_instruction}"
""")
        config = BinaryRunConfig(
            template_id="001",
            template_name="test_template_001",
            template_file=str(template_file),
            template_tags={},
            model="test-model",
            model_short="test",
            temperature=0.0,
            task_origin="test",
            n_tasks=1,
            task_ids=["a"],
        )
        template = config.load_template()
        assert template.name == "test_template_001"


class TestTaskPromptsBackwardCompat:
    """Test backward compatibility for task_prompts."""

    def test_missing_task_prompts_defaults_empty(self):
        """Old configs without task_prompts should get empty dict."""
        data = {
            "template_id": "001",
            "template_name": "test",
            "template_file": "test.yaml",
            "template_tags": {},
            "model": "test",
            "model_short": "test",
            "temperature": 0.0,
            "task_origin": "test",
            "n_tasks": 1,
            "task_ids": ["a"],
        }
        config = BinaryRunConfig.model_validate(data)
        assert config.task_prompts == {}

    def test_task_prompts_preserved(self):
        """task_prompts should be preserved when present."""
        data = {
            "template_id": "001",
            "template_name": "test",
            "template_file": "test.yaml",
            "template_tags": {},
            "model": "test",
            "model_short": "test",
            "temperature": 0.0,
            "task_origin": "test",
            "n_tasks": 1,
            "task_ids": ["a"],
            "task_prompts": {"a": "Do task A"},
        }
        config = BinaryRunConfig.model_validate(data)
        assert config.task_prompts == {"a": "Do task A"}
