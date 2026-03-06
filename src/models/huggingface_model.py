"""HuggingFace-based model for activation extraction and steering."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models.base import (
    BATCHED_SELECTOR_REGISTRY,
    COMPLETION_SELECTORS,
    TOKEN_ID_SELECTORS,
    GenerationResult,
    LayerHook,
    find_eot_indices,
)
from src.models.registry import is_valid_model, get_hf_name, get_eot_token
from src.models.architecture import get_layers, get_n_layers, get_hidden_dim
from src.types import Message


class HuggingFaceModel:
    def __init__(
        self,
        model_name: str,
        dtype: str = "bfloat16",
        device: str = "cuda",
        max_new_tokens: int = 256,
        attn_implementation: str = "sdpa",
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
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                resolved_name,
                torch_dtype=torch_dtype,
                device_map=device,
                attn_implementation=attn_implementation,
            )
        except ValueError:
            self.model = AutoModelForCausalLM.from_pretrained(
                resolved_name,
                torch_dtype=torch_dtype,
                device_map=device,
                attn_implementation="eager",
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

    def _get_eot_token_id(self) -> int:
        eot_token = get_eot_token(self.canonical_model_name)
        return self.tokenizer.convert_tokens_to_ids(eot_token)

    def format_messages(self, messages: list[Message], add_generation_prompt: bool = True) -> str:
        if self.tokenizer.chat_template is not None:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt,
            )
        # Base models without chat template: concatenate message content
        parts = []
        if messages and messages[0]["role"] == "system":
            parts.append(messages[0]["content"])
            messages = messages[1:]
        for m in messages:
            parts.append(m["content"])
        return "\n\n".join(parts)

    def _tokenize(self, text: str) -> torch.Tensor:
        return self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

    def _get_assistant_start_position(self, messages: list[Message]) -> int:
        if not messages or messages[-1]["role"] != "assistant":
            raise ValueError("Messages must end with an assistant message")
        prompt_messages = messages[:-1]
        prompt_with_header = self.format_messages(prompt_messages, add_generation_prompt=True)
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
        input_ids: torch.Tensor | None = None,
    ) -> dict[str, dict[int, np.ndarray]]:
        """Apply token selectors to reduce (batch, seq, d_model) -> (batch, d_model)."""
        batch_size = first_completion_indices.shape[0]
        seq_lengths = torch.tensor([max_seq_len] * batch_size)

        # Pre-compute special indices if needed
        eot_indices: torch.Tensor | None = None
        if "eot" in selector_names:
            if input_ids is None:
                raise ValueError("eot selector requires input_ids")
            eot_token_id = self._get_eot_token_id()
            eot_indices = find_eot_indices(input_ids, eot_token_id, first_completion_indices)

        results: dict[str, dict[int, np.ndarray]] = {name: {} for name in selector_names}
        for layer, layer_acts in activations.items():
            for name in selector_names:
                if name == "eot":
                    assert eot_indices is not None
                    act = layer_acts[torch.arange(batch_size), eot_indices, :]
                else:
                    act = BATCHED_SELECTOR_REGISTRY[name](
                        layer_acts, first_completion_indices.cpu(), seq_lengths,
                    )
                results[name][layer] = act.float().numpy()
        return results

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def _decode_completions(
        self, output_ids: torch.Tensor, prompt_len: int, n: int,
    ) -> list[str]:
        return [
            self.tokenizer.decode(output_ids[i, prompt_len:], skip_special_tokens=True).strip()
            for i in range(n)
        ]

    def _build_gen_kwargs(
        self,
        temperature: float,
        max_new_tokens: int | None,
        num_return_sequences: int = 1,
    ) -> dict:
        gen_kwargs = {
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "num_return_sequences": num_return_sequences,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
        return gen_kwargs

    @torch.inference_mode()
    def get_logprobs(
        self,
        messages: list[Message],
        top_k: int = 10,
    ) -> dict[str, float]:
        prompt = self.format_messages(messages, add_generation_prompt=False)
        input_ids = self._tokenize(prompt)
        logits = self.model(input_ids).logits[0, -1, :]
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        top_values, top_indices = torch.topk(log_probs, top_k)
        return {
            self.tokenizer.decode(idx.item()): val.item()
            for idx, val in zip(top_indices, top_values)
        }

    @torch.inference_mode()
    def generate(
        self,
        messages: list[Message],
        temperature: float = 1.0,
        max_new_tokens: int | None = None,
    ) -> str:
        prompt = self.format_messages(messages, add_generation_prompt=True)
        input_ids = self._tokenize(prompt)
        prompt_len = input_ids.shape[1]
        output_ids = self.model.generate(
            input_ids, **self._build_gen_kwargs(temperature, max_new_tokens),
        )
        return self._decode_completions(output_ids, prompt_len, 1)[0]

    @torch.inference_mode()
    def generate_n(
        self,
        messages: list[Message],
        n: int,
        temperature: float = 1.0,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        """Generate n completions in a single forward pass (shared prefill)."""
        prompt = self.format_messages(messages, add_generation_prompt=True)
        input_ids = self._tokenize(prompt)
        prompt_len = input_ids.shape[1]
        output_ids = self.model.generate(
            input_ids, **self._build_gen_kwargs(temperature, max_new_tokens, n),
        )
        return self._decode_completions(output_ids, prompt_len, n)

    @torch.inference_mode()
    def get_activations(
        self,
        messages: list[Message],
        layers: list[int],
        selector_names: list[str],
    ) -> dict[str, dict[int, np.ndarray]]:
        """Get activations for a single conversation.

        Works with or without an assistant message. Completion-dependent
        selectors (first, last, mean) require an assistant message. Prompt-based
        selectors (prompt_last) work with prompt-only messages.

        Returns: {selector_name: {layer: (d_model,) array}}
        """
        has_completion = messages and messages[-1]["role"] == "assistant"

        if not has_completion:
            needs_completion = set(selector_names) & COMPLETION_SELECTORS
            if needs_completion:
                raise ValueError(
                    f"Selectors {needs_completion} require an assistant message, "
                    f"but messages end with role '{messages[-1]['role']}'"
                )
            prompt = self.format_messages(messages, add_generation_prompt=True)
            input_ids = self._tokenize(prompt)
            seq_len = input_ids.shape[1]
            first_completion_idx = seq_len
        else:
            prompt = self.format_messages(messages, add_generation_prompt=False)
            input_ids = self._tokenize(prompt)
            first_completion_idx = self._get_assistant_start_position(messages)
            seq_len = input_ids.shape[1]

            if first_completion_idx >= seq_len:
                raise ValueError(
                    f"Assistant start position {first_completion_idx} is beyond "
                    f"sequence length {seq_len}. "
                    "This can happen if the assistant message is empty."
                )

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
        needs_ids = set(selector_names) & TOKEN_ID_SELECTORS
        batched = self._apply_selectors(
            activations, selector_names, first_indices, seq_len,
            input_ids=input_ids if needs_ids else None,
        )
        return {
            name: {layer: acts[0] for layer, acts in layer_dict.items()}
            for name, layer_dict in batched.items()
        }

    @torch.inference_mode()
    def get_activations_batch(
        self,
        messages_batch: list[list[Message]],
        layers: list[int],
        selector_names: list[str],
    ) -> dict[str, dict[int, np.ndarray]]:
        """Get activations for a batch of conversations via a single forward pass.

        Left-pads sequences to equal length. Selector indices are shifted to
        account for the padding offset.

        Returns: {selector_name: {layer: (batch, d_model) array}}
        """
        token_ids_list: list[torch.Tensor] = []
        first_completion_indices: list[int] = []
        for messages in messages_batch:
            has_completion = messages and messages[-1]["role"] == "assistant"
            if has_completion:
                prompt = self.format_messages(messages, add_generation_prompt=False)
                ids = self._tokenize(prompt)[0]  # (seq_len,)
                token_ids_list.append(ids)
                first_completion_indices.append(self._get_assistant_start_position(messages))
            else:
                needs_completion = set(selector_names) & COMPLETION_SELECTORS
                if needs_completion:
                    raise ValueError(
                        f"Selectors {needs_completion} require an assistant message, "
                        f"but messages end with role '{messages[-1]['role']}'"
                    )
                prompt = self.format_messages(messages, add_generation_prompt=True)
                ids = self._tokenize(prompt)[0]
                token_ids_list.append(ids)
                first_completion_indices.append(ids.shape[0])

        seq_lengths = [ids.shape[0] for ids in token_ids_list]
        max_len = max(seq_lengths)

        # Left-pad to max_len
        batch_size = len(messages_batch)
        padded = torch.full(
            (batch_size, max_len),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
        for i, ids in enumerate(token_ids_list):
            pad_offset = max_len - seq_lengths[i]
            padded[i, pad_offset:] = ids
            attention_mask[i, pad_offset:] = 1

        with self._hooked_forward(layers) as activations:
            self.model(padded, attention_mask=attention_mask)

        # Shift indices to account for left-padding
        shifted_first = torch.tensor([
            first_completion_indices[i] + (max_len - seq_lengths[i])
            for i in range(batch_size)
        ])

        needs_ids = set(selector_names) & TOKEN_ID_SELECTORS
        return self._apply_selectors(
            activations, selector_names, shifted_first, max_len,
            input_ids=padded if needs_ids else None,
        )

    @torch.inference_mode()
    def generate_with_activations(
        self,
        messages: list[Message],
        layers: list[int],
        selector_names: list[str],
        temperature: float = 1.0,
        max_new_tokens: int | None = None,
    ) -> GenerationResult:
        prompt = self.format_messages(messages, add_generation_prompt=True)
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
    def generate_with_hook(
        self,
        messages: list[Message],
        layer: int,
        hook: LayerHook,
        temperature: float = 1.0,
        max_new_tokens: int | None = None,
    ) -> str:
        """Generate with a hook applied at a single layer."""
        return self._generate_hooked(
            messages=messages,
            layer_hooks=[(layer, hook)],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
        )[0]

    @torch.inference_mode()
    def generate_with_hook_n(
        self,
        messages: list[Message],
        layer: int,
        hook: LayerHook,
        n: int,
        temperature: float = 1.0,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        """Generate n completions with a hook at a single layer (shared prefill)."""
        return self._generate_hooked(
            messages=messages,
            layer_hooks=[(layer, hook)],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            num_return_sequences=n,
        )

    @torch.inference_mode()
    def generate_with_hooks(
        self,
        messages: list[Message],
        layer_hooks: list[tuple[int, LayerHook]],
        temperature: float = 1.0,
        max_new_tokens: int | None = None,
    ) -> str:
        """Generate with hooks applied at multiple layers simultaneously."""
        return self._generate_hooked(
            messages=messages,
            layer_hooks=layer_hooks,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
        )[0]

    @torch.inference_mode()
    def generate_with_hooks_n(
        self,
        messages: list[Message],
        layer_hooks: list[tuple[int, LayerHook]],
        n: int,
        temperature: float = 1.0,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        """Generate n completions with hooks at multiple layers (shared prefill)."""
        return self._generate_hooked(
            messages=messages,
            layer_hooks=layer_hooks,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            num_return_sequences=n,
        )

    def _generate_hooked(
        self,
        messages: list[Message],
        layer_hooks: list[tuple[int, LayerHook]],
        temperature: float,
        max_new_tokens: int | None,
        num_return_sequences: int,
    ) -> list[str]:
        prompt = self.format_messages(messages, add_generation_prompt=True)
        input_ids = self._tokenize(prompt)
        prompt_len = input_ids.shape[1]

        def make_hf_hook(hook: LayerHook) -> Callable:
            def hf_hook(module: torch.nn.Module, input: tuple, output: tuple | torch.Tensor) -> tuple | torch.Tensor:
                hidden = output[0] if isinstance(output, tuple) else output
                modified = hook(hidden, prompt_len)
                if isinstance(output, tuple):
                    return (modified,) + output[1:]
                return modified
            return hf_hook

        handles = [
            self._get_layer(layer).register_forward_hook(make_hf_hook(hook))
            for layer, hook in layer_hooks
        ]
        try:
            output_ids = self.model.generate(
                input_ids,
                **self._build_gen_kwargs(temperature, max_new_tokens, num_return_sequences),
            )
        finally:
            for handle in handles:
                handle.remove()

        return self._decode_completions(output_ids, prompt_len, num_return_sequences)
