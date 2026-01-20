import pytest

pytestmark = pytest.mark.tasks

from src.task_data import Task, OriginDataset, load_tasks


class TestTask:

    def test_task_has_prompt(self):
        task = Task(prompt="Hello", origin=OriginDataset.WILDCHAT, id="1", metadata={})
        assert task.prompt == "Hello"

    def test_task_has_origin(self):
        task = Task(prompt="Hello", origin=OriginDataset.WILDCHAT, id="1", metadata={})
        assert task.origin == OriginDataset.WILDCHAT

    def test_task_has_id(self):
        task = Task(prompt="Hello", origin=OriginDataset.WILDCHAT, id="abc123", metadata={})
        assert task.id == "abc123"

    def test_task_has_metadata(self):
        task = Task(prompt="Hello", origin=OriginDataset.WILDCHAT, id="1", metadata={"topic": "math"})
        assert task.metadata == {"topic": "math"}


class TestLoadTasks:

    def test_load_tasks_returns_task_objects(self):
        result = load_tasks(n=10, origins=[OriginDataset.WILDCHAT])
        assert len(result) == 10
        assert all(isinstance(t, Task) for t in result)

    def test_load_tasks_n_returns_exact_count(self):
        result = load_tasks(n=3, origins=[OriginDataset.WILDCHAT])
        assert len(result) == 3

    def test_load_tasks_filter_by_origin(self):
        result = load_tasks(origins=[OriginDataset.WILDCHAT], n=10)
        assert len(result) > 0
        assert all(t.origin == OriginDataset.WILDCHAT for t in result)

    def test_load_tasks_filter_and_n_combine(self):
        result = load_tasks(n=2, origins=[OriginDataset.WILDCHAT])
        assert len(result) == 2
        assert all(t.origin == OriginDataset.WILDCHAT for t in result)

    def test_load_tasks_custom_filter(self):
        result = load_tasks(
            n=10,
            origins=[OriginDataset.WILDCHAT],
            filter_fn=lambda t: t.metadata.get("topic") == "ai",
        )
        assert len(result) > 0, "Expected at least one task matching filter"
        assert all(t.metadata.get("topic") == "ai" for t in result)

    def test_loaded_task_has_nonempty_prompt(self):
        result = load_tasks(n=5, origins=[OriginDataset.WILDCHAT])
        assert len(result) > 0
        assert all(len(t.prompt) > 0 for t in result)

    def test_loaded_task_has_nonempty_id(self):
        result = load_tasks(n=5, origins=[OriginDataset.WILDCHAT])
        assert len(result) > 0
        assert all(len(t.id) > 0 for t in result)

    def test_load_different_origins(self):
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
        result = load_tasks(
            n=20,
            origins=[OriginDataset.WILDCHAT, OriginDataset.ALPACA],
            seed=42,
        )
        assert len(result) == 20
        origins_present = {t.origin for t in result}
        assert OriginDataset.WILDCHAT in origins_present or OriginDataset.ALPACA in origins_present

    def test_seed_ensures_reproducibility(self):
        result1 = load_tasks(n=10, origins=[OriginDataset.WILDCHAT, OriginDataset.ALPACA], seed=42)
        result2 = load_tasks(n=10, origins=[OriginDataset.WILDCHAT, OriginDataset.ALPACA], seed=42)
        assert [t.id for t in result1] == [t.id for t in result2]

    def test_different_seeds_produce_different_order(self):
        result1 = load_tasks(n=10, origins=[OriginDataset.WILDCHAT], seed=42)
        result2 = load_tasks(n=10, origins=[OriginDataset.WILDCHAT], seed=123)
        assert [t.id for t in result1] != [t.id for t in result2]


class TestBailBench:

    def test_load_bailbench_returns_tasks(self):
        result = load_tasks(n=10, origins=[OriginDataset.BAILBENCH])
        assert len(result) == 10
        assert all(isinstance(t, Task) for t in result)

    def test_bailbench_has_correct_origin(self):
        result = load_tasks(n=5, origins=[OriginDataset.BAILBENCH])
        assert all(t.origin == OriginDataset.BAILBENCH for t in result)

    def test_bailbench_has_generated_ids(self):
        result = load_tasks(n=5, origins=[OriginDataset.BAILBENCH])
        assert all(t.id.startswith("bailbench_") for t in result)

    def test_bailbench_has_category_metadata(self):
        result = load_tasks(n=5, origins=[OriginDataset.BAILBENCH])
        for t in result:
            assert "category" in t.metadata
            assert "subcategory" in t.metadata

    def test_bailbench_has_nonempty_prompts(self):
        result = load_tasks(n=10, origins=[OriginDataset.BAILBENCH])
        assert all(len(t.prompt) > 0 for t in result)

    def test_bailbench_filter_by_category(self):
        result = load_tasks(
            n=100,
            origins=[OriginDataset.BAILBENCH],
            filter_fn=lambda t: t.metadata["category"] == "Gross Out",
        )
        assert len(result) > 0
        assert all(t.metadata["category"] == "Gross Out" for t in result)
