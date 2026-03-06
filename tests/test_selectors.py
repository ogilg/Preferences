"""Unit tests for token selectors (no GPU required)."""

import torch
import pytest

from src.models.base import (
    find_eot_indices,
    select_prompt_last_batched,
    select_prompt_mean_batched,
    select_first_batched,
    select_last_batched,
)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


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


class TestPromptMeanSelector:

    def test_averages_over_prompt_tokens_only(self):
        batch_size, seq_len, d_model = 2, 10, 4
        activations = torch.randn(batch_size, seq_len, d_model)
        first_completion_indices = torch.tensor([5, 7])
        seq_lengths = torch.tensor([10, 10])

        result = select_prompt_mean_batched(activations, first_completion_indices, seq_lengths)

        assert result.shape == (batch_size, d_model)
        assert torch.allclose(result[0], activations[0, :5, :].mean(dim=0))
        assert torch.allclose(result[1], activations[1, :7, :].mean(dim=0))

    def test_excludes_completion_tokens(self):
        batch_size, seq_len, d_model = 1, 10, 4
        activations = torch.zeros(batch_size, seq_len, d_model)
        # Put signal only in completion tokens — prompt_mean should be zero
        activations[0, 5:, :] = 1.0
        first_completion_indices = torch.tensor([5])
        seq_lengths = torch.tensor([10])

        result = select_prompt_mean_batched(activations, first_completion_indices, seq_lengths)
        assert torch.allclose(result, torch.zeros(1, d_model))

    def test_with_varying_prompt_lengths(self):
        batch_size, seq_len, d_model = 3, 15, 8
        activations = torch.arange(batch_size * seq_len * d_model, dtype=torch.float32)
        activations = activations.reshape(batch_size, seq_len, d_model)
        first_completion_indices = torch.tensor([3, 8, 12])
        seq_lengths = torch.tensor([15, 15, 15])

        result = select_prompt_mean_batched(activations, first_completion_indices, seq_lengths)

        assert result.shape == (batch_size, d_model)
        assert torch.allclose(result[0], activations[0, :3, :].mean(dim=0))
        assert torch.allclose(result[1], activations[1, :8, :].mean(dim=0))
        assert torch.allclose(result[2], activations[2, :12, :].mean(dim=0))


class TestFindEotIndices:
    """Test find_eot_indices with synthetic token IDs simulating Gemma 3 template.

    Gemma 3 prompt layout:
      <bos> <start_of_turn> user \\n {prompt tokens} <end_of_turn> \\n <start_of_turn> model \\n
      ^0    ^1               ^2   ^3  ^4...          ^eot            ...                       ^first_completion_idx
    """

    EOT_TOKEN_ID = 107  # Gemma 3's <end_of_turn> token ID

    def _make_gemma_ids(self, prompt_len: int) -> torch.Tensor:
        """Build a fake token sequence with EOT at the right place."""
        # Template tokens around the prompt
        # [bos=1, sot=106, user=..., \n=..., *prompt*, eot=107, \n=..., sot=106, model=..., \n=...]
        prefix = [1, 106, 2, 3]  # <bos> <start_of_turn> user \n
        prompt = list(range(1000, 1000 + prompt_len))
        suffix = [self.EOT_TOKEN_ID, 13, 106, 4, 13]  # <end_of_turn> \n <start_of_turn> model \n
        return torch.tensor(prefix + prompt + suffix, dtype=torch.long)

    def test_finds_eot_before_first_completion(self):
        ids = self._make_gemma_ids(prompt_len=5)
        # first_completion_idx = total length (prompt-only, no assistant content)
        first_comp = torch.tensor([len(ids)])
        result = find_eot_indices(ids.unsqueeze(0), self.EOT_TOKEN_ID, first_comp)
        # EOT should be at position 4 (prefix) + 5 (prompt) = 9
        assert result[0].item() == 9

    def test_batch_with_different_lengths(self):
        ids_short = self._make_gemma_ids(prompt_len=3)
        ids_long = self._make_gemma_ids(prompt_len=10)
        max_len = max(len(ids_short), len(ids_long))

        # Left-pad shorter sequence
        padded = torch.zeros(2, max_len, dtype=torch.long)
        pad_short = max_len - len(ids_short)
        padded[0, pad_short:] = ids_short
        padded[1, :len(ids_long)] = ids_long

        # first_completion_indices shifted for padding
        first_comp = torch.tensor([
            len(ids_short) + pad_short,  # shifted
            len(ids_long),
        ])
        result = find_eot_indices(padded, self.EOT_TOKEN_ID, first_comp)

        # EOT positions: prefix(4) + prompt_len, shifted by padding
        assert result[0].item() == 4 + 3 + pad_short  # short prompt, shifted
        assert result[1].item() == 4 + 10  # long prompt, no shift

    def test_raises_if_no_eot_found(self):
        # Sequence with no EOT token
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        first_comp = torch.tensor([5])
        with pytest.raises(ValueError, match="No end-of-turn token"):
            find_eot_indices(ids, self.EOT_TOKEN_ID, first_comp)

    def test_eot_is_before_prompt_last(self):
        """EOT should be strictly before prompt_last position."""
        ids = self._make_gemma_ids(prompt_len=8)
        first_comp = torch.tensor([len(ids)])
        eot_idx = find_eot_indices(ids.unsqueeze(0), self.EOT_TOKEN_ID, first_comp)
        prompt_last_idx = first_comp[0] - 1
        assert eot_idx[0].item() < prompt_last_idx.item()


@pytest.mark.api
class TestEotWithGemmaTokenizer:
    """Integration test: verify EOT finding works with the real Gemma 3 tokenizer."""

    @pytest.fixture(autouse=True)
    def load_tokenizer(self):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-27b-it")
        self.eot_token_id = self.tokenizer.convert_tokens_to_ids("<end_of_turn>")

    def _tokenize_prompt(self, user_content: str) -> torch.Tensor:
        messages = [{"role": "user", "content": user_content}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        return self.tokenizer(prompt, return_tensors="pt").input_ids[0]

    def test_eot_token_exists(self):
        assert self.eot_token_id != self.tokenizer.unk_token_id

    def test_finds_eot_in_real_prompt(self):
        ids = self._tokenize_prompt("Write me a haiku about cats")
        first_comp = torch.tensor([len(ids)])
        eot_idx = find_eot_indices(ids.unsqueeze(0), self.eot_token_id, first_comp)
        # Token at found position should decode to <end_of_turn>
        assert ids[eot_idx[0].item()].item() == self.eot_token_id

    def test_eot_offset_from_prompt_last(self):
        ids = self._tokenize_prompt("Hello world")
        first_comp = torch.tensor([len(ids)])
        eot_idx = find_eot_indices(ids.unsqueeze(0), self.eot_token_id, first_comp)

        offset = (len(ids) - 1) - eot_idx[0].item()
        # After EOT: \n <start_of_turn> model \n — expect offset of 4
        assert offset == 4, f"Expected offset 4, got {offset}"

    def test_works_with_long_prompt(self):
        ids = self._tokenize_prompt("x " * 500)
        first_comp = torch.tensor([len(ids)])
        eot_idx = find_eot_indices(ids.unsqueeze(0), self.eot_token_id, first_comp)
        assert ids[eot_idx[0].item()].item() == self.eot_token_id
