"""Unit tests for token selectors (no GPU required)."""

import torch
import pytest

from src.models.base import (
    select_prompt_last_batched,
    select_first_batched,
    select_last_batched,
)


class TestPromptLastSelector:
    """Test prompt_last selector logic."""

    def test_selects_token_before_first_completion(self):
        """prompt_last should select first_completion_idx - 1."""
        batch_size, seq_len, d_model = 2, 10, 4
        activations = torch.randn(batch_size, seq_len, d_model)
        first_completion_indices = torch.tensor([5, 7])
        seq_lengths = torch.tensor([10, 10])

        result = select_prompt_last_batched(activations, first_completion_indices, seq_lengths)

        assert result.shape == (batch_size, d_model)
        assert torch.allclose(result[0], activations[0, 4, :])  # 5 - 1 = 4
        assert torch.allclose(result[1], activations[1, 6, :])  # 7 - 1 = 6

    def test_differs_from_first_selector(self):
        """prompt_last and first should select adjacent but different tokens."""
        batch_size, seq_len, d_model = 1, 10, 4
        activations = torch.randn(batch_size, seq_len, d_model)
        first_completion_indices = torch.tensor([5])
        seq_lengths = torch.tensor([10])

        prompt_last = select_prompt_last_batched(activations, first_completion_indices, seq_lengths)
        first = select_first_batched(activations, first_completion_indices, seq_lengths)

        # prompt_last selects index 4, first selects index 5
        assert torch.allclose(prompt_last[0], activations[0, 4, :])
        assert torch.allclose(first[0], activations[0, 5, :])
        assert not torch.allclose(prompt_last, first)

    def test_with_varying_completion_indices(self):
        """Should handle different first_completion_idx per batch item."""
        batch_size, seq_len, d_model = 3, 15, 8
        activations = torch.arange(batch_size * seq_len * d_model, dtype=torch.float32)
        activations = activations.reshape(batch_size, seq_len, d_model)
        first_completion_indices = torch.tensor([3, 8, 12])
        seq_lengths = torch.tensor([15, 15, 15])

        result = select_prompt_last_batched(activations, first_completion_indices, seq_lengths)

        assert result.shape == (batch_size, d_model)
        assert torch.allclose(result[0], activations[0, 2, :])   # 3 - 1 = 2
        assert torch.allclose(result[1], activations[1, 7, :])   # 8 - 1 = 7
        assert torch.allclose(result[2], activations[2, 11, :])  # 12 - 1 = 11
