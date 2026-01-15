import pytest

from src.task_data import Task, OriginDataset, load_tasks


class TestTask:
    """Tests for the Task dataclass."""

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

    def test_load_tasks_returns_task_objects(self):
        """load_tasks should return a list of Task objects."""
        result = load_tasks(n=10, origins=[OriginDataset.WILDCHAT])
        assert len(result) == 10
        assert all(isinstance(t, Task) for t in result)

    def test_load_tasks_n_returns_exact_count(self):
        """load_tasks(n=k) should return exactly k tasks if enough exist."""
        result = load_tasks(n=3, origins=[OriginDataset.WILDCHAT])
        assert len(result) == 3

    def test_load_tasks_filter_by_origin(self):
        """load_tasks should filter by origin dataset."""
        result = load_tasks(origins=[OriginDataset.WILDCHAT], n=10)
        assert len(result) > 0
        assert all(t.origin == OriginDataset.WILDCHAT for t in result)

    def test_load_tasks_filter_and_n_combine(self):
        """Filtering and n should work together."""
        result = load_tasks(n=2, origins=[OriginDataset.WILDCHAT])
        assert len(result) == 2
        assert all(t.origin == OriginDataset.WILDCHAT for t in result)

    def test_load_tasks_custom_filter(self):
        """load_tasks should accept a custom filter function."""
        result = load_tasks(
            n=10,
            origins=[OriginDataset.WILDCHAT],
            filter_fn=lambda t: t.metadata.get("topic") == "ai",
        )
        assert len(result) > 0, "Expected at least one task matching filter"
        assert all(t.metadata.get("topic") == "ai" for t in result)

    def test_loaded_task_has_nonempty_prompt(self):
        """Loaded tasks should have non-empty prompts."""
        result = load_tasks(n=5, origins=[OriginDataset.WILDCHAT])
        assert len(result) > 0, "Expected at least one task"
        assert all(len(t.prompt) > 0 for t in result)

    def test_loaded_task_has_nonempty_id(self):
        """Loaded tasks should have non-empty ids."""
        result = load_tasks(n=5, origins=[OriginDataset.WILDCHAT])
        assert len(result) > 0, "Expected at least one task"
        assert all(len(t.id) > 0 for t in result)

    def test_load_different_origins(self):
        """Should load from different dataset origins."""
        wildchat = load_tasks(n=1, origins=[OriginDataset.WILDCHAT])
        alpaca = load_tasks(n=1, origins=[OriginDataset.ALPACA])
        math = load_tasks(n=1, origins=[OriginDataset.MATH])

        assert len(wildchat) == 1
        assert len(alpaca) == 1
        assert len(math) == 1
        assert wildchat[0].origin == OriginDataset.WILDCHAT
        assert alpaca[0].origin == OriginDataset.ALPACA
        assert math[0].origin == OriginDataset.MATH

    def test_load_multiple_origins(self):
        """Should load from multiple origins and shuffle reproducibly."""
        result = load_tasks(
            n=20,
            origins=[OriginDataset.WILDCHAT, OriginDataset.ALPACA],
            seed=42,
        )
        assert len(result) == 20
        origins_present = {t.origin for t in result}
        assert OriginDataset.WILDCHAT in origins_present or OriginDataset.ALPACA in origins_present

    def test_seed_ensures_reproducibility(self):
        """Same seed should produce same task order."""
        result1 = load_tasks(n=10, origins=[OriginDataset.WILDCHAT, OriginDataset.ALPACA], seed=42)
        result2 = load_tasks(n=10, origins=[OriginDataset.WILDCHAT, OriginDataset.ALPACA], seed=42)
        assert [t.id for t in result1] == [t.id for t in result2]

    def test_different_seeds_produce_different_order(self):
        """Different seeds should produce different task order."""
        result1 = load_tasks(n=10, origins=[OriginDataset.WILDCHAT], seed=42)
        result2 = load_tasks(n=10, origins=[OriginDataset.WILDCHAT], seed=123)
        assert [t.id for t in result1] != [t.id for t in result2]


class TestBailBench:
    """Tests for the BailBench dataset."""

    def test_load_bailbench_returns_tasks(self):
        """Should load BailBench tasks."""
        result = load_tasks(n=10, origins=[OriginDataset.BAILBENCH])
        assert len(result) == 10
        assert all(isinstance(t, Task) for t in result)

    def test_bailbench_has_correct_origin(self):
        """BailBench tasks should have BAILBENCH origin."""
        result = load_tasks(n=5, origins=[OriginDataset.BAILBENCH])
        assert all(t.origin == OriginDataset.BAILBENCH for t in result)

    def test_bailbench_has_generated_ids(self):
        """BailBench tasks should have auto-generated IDs."""
        result = load_tasks(n=5, origins=[OriginDataset.BAILBENCH])
        assert all(t.id.startswith("bailbench_") for t in result)

    def test_bailbench_has_category_metadata(self):
        """BailBench tasks should have category and subcategory in metadata."""
        result = load_tasks(n=5, origins=[OriginDataset.BAILBENCH])
        for t in result:
            assert "category" in t.metadata
            assert "subcategory" in t.metadata

    def test_bailbench_has_nonempty_prompts(self):
        """BailBench tasks should have non-empty prompts."""
        result = load_tasks(n=10, origins=[OriginDataset.BAILBENCH])
        assert all(len(t.prompt) > 0 for t in result)

    def test_bailbench_filter_by_category(self):
        """Should be able to filter BailBench by category."""
        result = load_tasks(
            n=100,
            origins=[OriginDataset.BAILBENCH],
            filter_fn=lambda t: t.metadata["category"] == "Gross Out",
        )
        assert len(result) > 0
        assert all(t.metadata["category"] == "Gross Out" for t in result)
