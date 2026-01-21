"""Shared test fixtures."""

import pytest
from dotenv import load_dotenv

from src.task_data import Task, OriginDataset

# Load environment variables for API tests
load_dotenv()


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
def sample_tasks():
    return [
        Task(prompt="What is 2 + 2?", origin=OriginDataset.WILDCHAT, id="task_1", metadata={}),
        Task(prompt="Write a haiku about the ocean.", origin=OriginDataset.WILDCHAT, id="task_2", metadata={}),
        Task(prompt="Explain quantum entanglement in simple terms.", origin=OriginDataset.WILDCHAT, id="task_3", metadata={}),
    ]
