import pytest

from src.task_data import Task, OriginDataset, load_tasks


class TestTask:
    """Tests for the Task dataclass."""

    def test_task_exists(self):
        """Task should exist."""
        assert Task is not None

    def test_task_has_prompt(self):
        """Task should have a prompt field."""
        task = Task(prompt="Hello", origin=OriginDataset.WILDCHAT, id="1", metadata={})
        assert task.prompt == "Hello"

    def test_task_has_origin(self):
        """Task should have an origin field."""
        task = Task(prompt="Hello", origin=OriginDataset.WILDCHAT, id="1", metadata={})
        assert task.origin == OriginDataset.WILDCHAT

    def test_task_has_id(self):
        """Task should have an id field."""
        task = Task(prompt="Hello", origin=OriginDataset.WILDCHAT, id="abc123", metadata={})
        assert task.id == "abc123"

    def test_task_has_metadata(self):
        """Task should have a metadata field."""
        task = Task(prompt="Hello", origin=OriginDataset.WILDCHAT, id="1", metadata={"topic": "math"})
        assert task.metadata == {"topic": "math"}


class TestLoadTasks:
    """Tests for the load_tasks function."""

    def test_load_tasks_exists(self):
        """load_tasks function should exist."""
        assert callable(load_tasks)

    def test_load_tasks_returns_list(self):
        """load_tasks should return a list."""
        result = load_tasks(n=10)
        assert isinstance(result, list)

    def test_load_tasks_returns_task_objects(self):
        """load_tasks should return a list of Task objects."""
        result = load_tasks(n=10)
        assert len(result) > 0
        assert all(isinstance(t, Task) for t in result)

    def test_load_tasks_n_limits_count(self):
        """load_tasks(n=k) should return at most k tasks."""
        result = load_tasks(n=5)
        assert len(result) <= 5

    def test_load_tasks_n_returns_exact_if_available(self):
        """load_tasks(n=k) should return exactly k tasks if enough exist."""
        result = load_tasks(n=3)
        assert len(result) == 3

    def test_load_tasks_filter_by_origin(self):
        """load_tasks should filter by origin dataset."""
        result = load_tasks(origin=OriginDataset.WILDCHAT, n=10)
        assert len(result) > 0
        assert all(t.origin == OriginDataset.WILDCHAT for t in result)

    def test_load_tasks_filter_and_n_combine(self):
        """Filtering and n should work together."""
        result = load_tasks(n=2, origin=OriginDataset.WILDCHAT)
        assert len(result) == 2
        assert all(t.origin == OriginDataset.WILDCHAT for t in result)

    def test_load_tasks_custom_filter(self):
        """load_tasks should accept a custom filter function."""
        result = load_tasks(
            n=10,
            origin=OriginDataset.WILDCHAT,
            filter_fn=lambda t: t.metadata.get("topic") == "ai",
        )
        assert len(result) > 0, "Expected at least one task matching filter"
        assert all(t.metadata.get("topic") == "ai" for t in result)

    def test_loaded_task_has_nonempty_prompt(self):
        """Loaded tasks should have non-empty prompts."""
        result = load_tasks(n=5)
        assert len(result) > 0, "Expected at least one task"
        assert all(len(t.prompt) > 0 for t in result)

    def test_loaded_task_has_nonempty_id(self):
        """Loaded tasks should have non-empty ids."""
        result = load_tasks(n=5)
        assert len(result) > 0, "Expected at least one task"
        assert all(len(t.id) > 0 for t in result)

    def test_load_different_origins(self):
        """Should load from different dataset origins."""
        wildchat = load_tasks(n=1, origin=OriginDataset.WILDCHAT)
        alpaca = load_tasks(n=1, origin=OriginDataset.ALPACA)
        math = load_tasks(n=1, origin=OriginDataset.MATH)

        assert len(wildchat) == 1
        assert len(alpaca) == 1
        assert len(math) == 1
        assert wildchat[0].origin == OriginDataset.WILDCHAT
        assert alpaca[0].origin == OriginDataset.ALPACA
        assert math[0].origin == OriginDataset.MATH
