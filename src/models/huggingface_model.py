"""HuggingFace-based model for activation extraction and steering."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models.base import BATCHED_SELECTOR_REGISTRY, GenerationResult, SteeringHook
from src.models.registry import is_valid_model, get_hf_name
from src.models.architecture import get_layers, get_n_layers, get_hidden_dim
from src.types import Message


class HuggingFaceModel:
    def __init__(
        self,
        model_name: str,
        dtype: str = "bfloat16",
        device: str = "cuda",
        max_new_tokens: int = 256,
        attn_implementation: str = "flash_attention_2",
    ):
        self.canonical_model_name = model_name
        if is_valid_model(model_name):
            resolved_name = get_hf_name(model_name)
        else:
            resolved_name = model_name
        self.model_name = resolved_name
        self.max_new_tokens = max_new_tokens
        self.device = device

        torch_dtype = getattr(torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            resolved_name,
            torch_dtype=torch_dtype,
            device_map=device,
            attn_implementation=attn_implementation,
        )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(resolved_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def n_layers(self) -> int:
        return get_n_layers(self.model)

    @property
    def hidden_dim(self) -> int:
        return get_hidden_dim(self.model)

    def resolve_layer(self, layer: int | float) -> int:
        """Resolve layer index. Floats in [0, 1] are relative positions."""
        if isinstance(layer, float):
            return int(layer * self.n_layers)
        return layer

    def _get_layer(self, layer: int) -> torch.nn.Module:
        return get_layers(self.model)[layer]

    def _format_messages(self, messages: list[Message], add_generation_prompt: bool = True) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt,
        )

    def _tokenize(self, text: str) -> torch.Tensor:
        return self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

    def _get_assistant_start_position(self, messages: list[Message]) -> int:
        if not messages or messages[-1]["role"] != "assistant":
            raise ValueError("Messages must end with an assistant message")
        prompt_messages = messages[:-1]
        prompt_with_header = self._format_messages(prompt_messages, add_generation_prompt=True)
        return self._tokenize(prompt_with_header).shape[1]

    @contextmanager
    def _hooked_forward(
        self, layers: list[int]
    ) -> Iterator[dict[int, torch.Tensor]]:
        """Context manager that registers hooks and yields activation buffer.

        Usage:
            with self._hooked_forward([0, 15, 31]) as activations:
                self.model(input_ids)
            # activations[layer] now contains (batch, seq, d_model) tensors
        """
        activations: dict[int, torch.Tensor] = {}
        handles: list[torch.utils.hooks.RemovableHandle] = []

        # make_hook closure captures layer_idx to avoid late-binding in loop.
        # Without this, all hooks would reference the final loop value.
        for layer in layers:
            def make_hook(layer_idx: int) -> Callable:
                def hook(module: torch.nn.Module, input: tuple, output: tuple | torch.Tensor) -> None:
                    # Transformer layer output is either:
                    #   - tuple: (hidden_states, present_key_value, ...)
                    #   - tensor: just hidden_states (when output_hidden_states=False, use_cache=False)
                    # hidden_states shape: (batch, seq_len, d_model)
                    hidden = output[0] if isinstance(output, tuple) else output
                    activations[layer_idx] = hidden.detach().cpu()
                return hook
            handles.append(self._get_layer(layer).register_forward_hook(make_hook(layer)))

        try:
            yield activations
        finally:
            for h in handles:
                h.remove()

    def _apply_selectors(
        self,
        activations: dict[int, torch.Tensor],
        selector_names: list[str],
        first_completion_indices: torch.Tensor,
        max_seq_len: int,
    ) -> dict[str, dict[int, np.ndarray]]:
        """Apply token selectors to reduce (batch, seq, d_model) -> (batch, d_model).

        Selectors:
          - "last": final token (the token just before EOS/padding)
          - "first": first completion token (where assistant content starts)
          - "mean": mean over all completion tokens
        """
        batch_size = first_completion_indices.shape[0]
        # All sequences are padded to max_seq_len, so seq_lengths = max_seq_len for indexing.
        seq_lengths = torch.tensor([max_seq_len] * batch_size)

        results: dict[str, dict[int, np.ndarray]] = {name: {} for name in selector_names}
        for layer, layer_acts in activations.items():
            for name in selector_names:
                act = BATCHED_SELECTOR_REGISTRY[name](
                    layer_acts, first_completion_indices.cpu(), seq_lengths,
                )
                results[name][layer] = act.float().numpy()
        return results

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        messages: list[Message],
        temperature: float = 1.0,
        max_new_tokens: int | None = None,
    ) -> str:
        prompt = self._format_messages(messages, add_generation_prompt=True)
        input_ids = self._tokenize(prompt)
        prompt_len = input_ids.shape[1]

        gen_kwargs = {
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        output_ids = self.model.generate(input_ids, **gen_kwargs)
        new_tokens = output_ids[0, prompt_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    @torch.inference_mode()
    def get_activations(
        self,
        messages: list[Message],
        layers: list[int],
        selector_names: list[str],
    ) -> dict[str, dict[int, np.ndarray]]:
        """Get activations for a single conversation.

        Returns: {selector_name: {layer: (d_model,) array}}
        """
        prompt = self._format_messages(messages, add_generation_prompt=False)
        input_ids = self._tokenize(prompt)
        first_completion_idx = self._get_assistant_start_position(messages)
        seq_len = input_ids.shape[1]

        if first_completion_idx >= seq_len:
            raise ValueError(
                f"Assistant start position {first_completion_idx} is beyond "
                f"sequence length {seq_len}. "
                "This can happen if the assistant message is empty."
            )

        # Chat templates add special tokens (e.g. <|eot_id|>) even for empty
        # assistant messages, so first_completion_idx < seq_len can hold with
        # zero real content tokens. Require at least 2 tokens after the prompt
        # (content + end-of-turn marker).
        n_completion_tokens = seq_len - first_completion_idx
        if n_completion_tokens < 2:
            raise ValueError(
                f"Assistant message has no content tokens (only {n_completion_tokens} "
                f"token after prompt, likely just an end-of-turn marker). "
                "This can happen if the assistant message is empty."
            )

        with self._hooked_forward(layers) as activations:
            self.model(input_ids)

        first_indices = torch.tensor([first_completion_idx])
        batched = self._apply_selectors(
            activations, selector_names, first_indices, seq_len,
        )
        return {
            name: {layer: acts[0] for layer, acts in layer_dict.items()}
            for name, layer_dict in batched.items()
        }

    def get_activations_batch(
        self,
        messages_batch: list[list[Message]],
        layers: list[int],
        selector_names: list[str],
    ) -> dict[str, dict[int, np.ndarray]]:
        """Get activations for a batch of conversations.

        Processes each sample individually to avoid bfloat16 non-determinism from
        left-padding in batched forward passes. Results are stacked along a new
        batch dimension.

        Returns: {selector_name: {layer: (batch, d_model) array}}
        """
        per_sample = [
            self.get_activations(messages, layers, selector_names)
            for messages in messages_batch
        ]

        result: dict[str, dict[int, np.ndarray]] = {name: {} for name in selector_names}
        for name in selector_names:
            for layer in layers:
                result[name][layer] = np.stack(
                    [sample[name][layer] for sample in per_sample]
                )
        return result

    @torch.inference_mode()
    def generate_with_activations(
        self,
        messages: list[Message],
        layers: list[int],
        selector_names: list[str],
        temperature: float = 1.0,
        max_new_tokens: int | None = None,
    ) -> GenerationResult:
        prompt = self._format_messages(messages, add_generation_prompt=True)
        prompt_tokens = len(self._tokenize(prompt)[0])

        completion = self.generate(messages, temperature=temperature, max_new_tokens=max_new_tokens)
        completion_tokens = len(self.tokenizer(completion, return_tensors="pt").input_ids[0])

        full_messages = messages + [{"role": "assistant", "content": completion}]
        activations = self.get_activations(full_messages, layers, selector_names)

        return GenerationResult(
            completion=completion,
            activations=activations,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    @torch.inference_mode()
    def generate_with_steering(
        self,
        messages: list[Message],
        layer: int,
        steering_hook: SteeringHook,
        temperature: float = 1.0,
        max_new_tokens: int | None = None,
    ) -> str:
        """Generate with activation steering applied at specified layer."""
        prompt = self._format_messages(messages, add_generation_prompt=True)
        input_ids = self._tokenize(prompt)
        prompt_len = input_ids.shape[1]

        def hf_hook(module: torch.nn.Module, input: tuple, output: tuple | torch.Tensor) -> tuple | torch.Tensor:
            hidden = output[0] if isinstance(output, tuple) else output
            modified = steering_hook(hidden, prompt_len)
            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified

        handle = self._get_layer(layer).register_forward_hook(hf_hook)
        try:
            gen_kwargs = {
                "max_new_tokens": max_new_tokens or self.max_new_tokens,
                "do_sample": temperature > 0,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
            if temperature > 0:
                gen_kwargs["temperature"] = temperature
            output_ids = self.model.generate(input_ids, **gen_kwargs)
        finally:
            handle.remove()

        new_tokens = output_ids[0, prompt_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
