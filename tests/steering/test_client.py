"""Tests for SteeredHFClient."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

import numpy as np

from src.models.openai_compatible import GenerateRequest
from src.steering.client import SteeredHFClient


def _make_mock_hf_model() -> MagicMock:
    mock = MagicMock()
    mock.canonical_model_name = "gemma-3-27b"
    mock.model_name = "google/gemma-3-27b-it"
    mock.max_new_tokens = 256
    mock.device = "cpu"
    mock.generate.return_value = "I enjoyed that task."
    mock.generate_with_steering.return_value = "I loved that task!"
    return mock


def _make_client(mock: MagicMock, coefficient: float = 1500.0) -> SteeredHFClient:
    direction = np.ones(16, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    return SteeredHFClient(
        hf_model=mock,
        layer=15,
        steering_direction=direction,
        coefficient=coefficient,
    )


def test_steered_calls_generate_with_steering():
    mock = _make_mock_hf_model()
    client = _make_client(mock, coefficient=1500.0)
    messages = [{"role": "user", "content": "Hello"}]

    result = client.generate(messages, temperature=0.8)

    assert result == "I loved that task!"
    mock.generate_with_steering.assert_called_once()
    assert mock.generate_with_steering.call_args.kwargs["layer"] == 15
    assert mock.generate_with_steering.call_args.kwargs["temperature"] == 0.8
    mock.generate.assert_not_called()


def test_zero_coefficient_calls_plain_generate():
    mock = _make_mock_hf_model()
    client = _make_client(mock, coefficient=0.0)

    result = client.generate([{"role": "user", "content": "Hello"}])

    assert result == "I enjoyed that task."
    mock.generate.assert_called_once()
    mock.generate_with_steering.assert_not_called()


def test_batch_async_processes_all_requests():
    mock = _make_mock_hf_model()
    client = _make_client(mock)
    requests = [
        GenerateRequest(messages=[{"role": "user", "content": f"Task {i}"}])
        for i in range(3)
    ]

    results = asyncio.run(
        client.generate_batch_async(requests, asyncio.Semaphore(1))
    )

    assert len(results) == 3
    assert all(r.ok for r in results)
    assert all(r.unwrap() == "I loved that task!" for r in results)


def test_batch_async_captures_errors():
    mock = _make_mock_hf_model()
    mock.generate_with_steering.side_effect = RuntimeError("GPU OOM")
    client = _make_client(mock)
    requests = [GenerateRequest(messages=[{"role": "user", "content": "Hi"}])]

    results = asyncio.run(
        client.generate_batch_async(requests, asyncio.Semaphore(1))
    )

    assert len(results) == 1
    assert not results[0].ok
    assert "GPU OOM" in str(results[0].error)


def test_batch_async_calls_on_complete():
    mock = _make_mock_hf_model()
    client = _make_client(mock)
    requests = [
        GenerateRequest(messages=[{"role": "user", "content": f"Task {i}"}])
        for i in range(2)
    ]
    callback = MagicMock()

    asyncio.run(
        client.generate_batch_async(requests, asyncio.Semaphore(10), on_complete=callback)
    )

    assert callback.call_count == 2


def test_rejects_tool_use_requests():
    mock = _make_mock_hf_model()
    client = _make_client(mock)
    requests = [GenerateRequest(
        messages=[{"role": "user", "content": "Hi"}],
        tools=[{"type": "function", "function": {"name": "f"}}],
    )]

    with pytest.raises(ValueError, match="tool use"):
        client.generate_batch(requests)


def test_exposes_expected_attributes():
    mock = _make_mock_hf_model()
    client = _make_client(mock)

    assert client.canonical_model_name == "gemma-3-27b"
    assert client.model_name == "google/gemma-3-27b-it"
    assert client.max_new_tokens == 256
