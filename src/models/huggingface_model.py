"""HuggingFace-based model for activation extraction and steering."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models.base import (
    ASSISTANT_SELECTOR_REGISTRY,
    BATCHED_SELECTOR_REGISTRY,
    COMPLETION_SELECTORS,
    TASK_SELECTOR_REGISTRY,
    TASK_SELECTORS,
    TOKEN_ID_SELECTORS,
    GenerationResult,
    LayerHook,
    find_eot_indices,
    is_anchored_offset_selector,
    needs_assistant_content_span,
    needs_assistant_tb_anchor,
    parse_anchored_offset,
    requires_chat_template,
    validate_selectors,
)
from src.models.registry import is_valid_model, get_hf_name, get_eot_token
from src.models.architecture import get_layers, get_n_layers, get_hidden_dim
from src.steering.tokenization import find_text_span
from src.types import Message


class HuggingFaceModel:
    def __init__(
        self,
        model_name: str,
        dtype: str = "bfloat16",
        device: str = "cuda",
        max_new_tokens: int = 256,
        attn_implementation: str = "sdpa",
        subfolder: str | None = None,
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
        load_kwargs: dict = dict(
            torch_dtype=torch_dtype,
            device_map=device,
        )
        if subfolder is not None:
            load_kwargs["subfolder"] = subfolder
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                resolved_name, attn_implementation=attn_implementation, **load_kwargs,
            )
        except ValueError:
            self.model = AutoModelForCausalLM.from_pretrained(
                resolved_name, attn_implementation="eager", **load_kwargs,
            )
        self.model.eval()
        tokenizer_kwargs = {"subfolder": subfolder} if subfolder else {}
        self.tokenizer = AutoTokenizer.from_pretrained(resolved_name, **tokenizer_kwargs)
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

    @property
    def has_chat_template(self) -> bool:
        return self.tokenizer.chat_template is not None

    def _validate_selectors_for_model(self, selector_names: list[str]) -> None:
        validate_selectors(selector_names)
        if self.has_chat_template:
            return
        bad = [s for s in selector_names if requires_chat_template(s)]
        if bad:
            raise ValueError(
                f"Selectors {bad} require a chat template, but {self.model_name} has none. "
                f"Use task_last or task_mean instead."
            )

    def format_messages(self, messages: list[Message], add_generation_prompt: bool = True) -> str:
        if self.has_chat_template:
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

    def _get_first_assistant_span(self, messages: list[Message]) -> tuple[int, int]:
        """Find token indices [start, end) of the first assistant message content.

        Uses differential formatting to avoid substring matching ambiguity:
        tokenize up to assistant header to get start, append content to get end.
        """
        asst_idx = self._find_first_assistant_idx(messages)
        before = self.format_messages(messages[:asst_idx], add_generation_prompt=True)
        start = self._tokenize(before).shape[1]
        through = before + messages[asst_idx]["content"]
        end = self._tokenize(through).shape[1]
        return start, end

    def _get_assistant_to_user_anchor(self, messages: list[Message]) -> int:
        """Find the token position where the follow-up user content starts.

        This is the anchor for assistant_tb:N selectors — the structural
        equivalent of first_completion_indices for turn_boundary:N.
        Both anchors point to the first content token after a turn header,
        so the same offsets select the same structural tokens:
          -1 = \\n, -2 = role, -3 = <start_of_turn>, -4 = \\n, -5 = <end_of_turn>
        """
        asst_idx = self._find_first_assistant_idx(messages)
        followup_idx = asst_idx + 1
        if followup_idx >= len(messages):
            raise ValueError("No message after first assistant turn")
        # Format through the follow-up message, then strip its content via rindex
        formatted = self.format_messages(messages[:followup_idx + 1], add_generation_prompt=False)
        content_char_start = formatted.rindex(messages[followup_idx]["content"])
        header = formatted[:content_char_start]
        return self._tokenize(header).shape[1]

    @staticmethod
    def _find_first_assistant_idx(messages: list[Message]) -> int:
        for i, m in enumerate(messages):
            if m["role"] == "assistant":
                return i
        raise ValueError("No assistant message found in messages")

    def _get_task_span(self, messages: list[Message]) -> tuple[int, int]:
        """Find token indices [start, end) of the last user message content."""
        user_content = None
        for m in reversed(messages):
            if m["role"] == "user":
                user_content = m["content"]
                break
        if user_content is None:
            raise ValueError("No user message found in messages")

        has_completion = messages and messages[-1]["role"] == "assistant"
        formatted = self.format_messages(messages, add_generation_prompt=not has_completion)
        return find_text_span(self.tokenizer, formatted, user_content)

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
        task_starts: torch.Tensor | None = None,
        task_ends: torch.Tensor | None = None,
        assistant_starts: torch.Tensor | None = None,
        assistant_ends: torch.Tensor | None = None,
        assistant_to_user_anchor: torch.Tensor | None = None,
    ) -> dict[str, dict[int, np.ndarray]]:
        """Apply token selectors to reduce (batch, seq, d_model) -> (batch, d_model).

        Builds a dispatch table mapping each selector name to a function
        (layer_acts) -> (batch, d_model), then applies them all uniformly.
        """
        batch_size = first_completion_indices.shape[0]
        seq_lengths = torch.tensor([max_seq_len] * batch_size)
        batch_range = torch.arange(batch_size)

        # Anchor tensors for anchored offset selectors (turn_boundary:N, assistant_tb:N)
        anchors: dict[str, torch.Tensor] = {
            "first_completion": first_completion_indices,
        }
        if assistant_to_user_anchor is not None:
            anchors["assistant_to_user"] = assistant_to_user_anchor

        # Pre-compute single-index selectors: name -> (batch,) index tensor
        index_selectors: dict[str, torch.Tensor] = {}
        for name in selector_names:
            parsed = parse_anchored_offset(name)
            if parsed is not None:
                anchor_name, offset = parsed
                assert anchor_name in anchors, f"{name} requires anchor '{anchor_name}'"
                index_selectors[name] = anchors[anchor_name] + offset
        if "eot" in selector_names:
            assert input_ids is not None, "eot selector requires input_ids"
            eot_token_id = self._get_eot_token_id()
            index_selectors["eot"] = find_eot_indices(input_ids, eot_token_id, first_completion_indices)

        # Build per-selector dispatch: name -> fn(layer_acts) -> (batch, d_model)
        dispatch: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {}
        for name in selector_names:
            if name in index_selectors:
                idx = index_selectors[name]
                dispatch[name] = lambda acts, _idx=idx: acts[batch_range, _idx, :]

            elif name in BATCHED_SELECTOR_REGISTRY:
                fn = BATCHED_SELECTOR_REGISTRY[name]
                fci = first_completion_indices.cpu()
                dispatch[name] = lambda acts, _fn=fn, _fci=fci: _fn(acts, _fci, seq_lengths)

            elif name in TASK_SELECTOR_REGISTRY:
                assert task_starts is not None and task_ends is not None
                fn = TASK_SELECTOR_REGISTRY[name]
                dispatch[name] = lambda acts, _fn=fn: _fn(acts, task_starts, task_ends)

            elif name in ASSISTANT_SELECTOR_REGISTRY:
                assert assistant_starts is not None and assistant_ends is not None
                fn = ASSISTANT_SELECTOR_REGISTRY[name]
                dispatch[name] = lambda acts, _fn=fn: _fn(acts, assistant_starts, assistant_ends)

            else:
                raise ValueError(f"No dispatch for selector: {name}")

        results: dict[str, dict[int, np.ndarray]] = {name: {} for name in selector_names}
        for layer, layer_acts in activations.items():
            for name in selector_names:
                results[name][layer] = dispatch[name](layer_acts).float().numpy()
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

        Returns: {selector_name: {layer: (d_model,) array}}
        """
        self._validate_selectors_for_model(selector_names)
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
        needs_task = set(selector_names) & TASK_SELECTORS
        task_starts = task_ends = None
        assistant_starts = assistant_ends = None
        assistant_anchor = None
        if needs_task:
            start, end = self._get_task_span(messages)
            task_starts = torch.tensor([start])
            task_ends = torch.tensor([end])
        if needs_assistant_content_span(selector_names):
            a_start, a_end = self._get_first_assistant_span(messages)
            assistant_starts = torch.tensor([a_start])
            assistant_ends = torch.tensor([a_end])
        if needs_assistant_tb_anchor(selector_names):
            assistant_anchor = torch.tensor([self._get_assistant_to_user_anchor(messages)])
        batched = self._apply_selectors(
            activations, selector_names, first_indices, seq_len,
            input_ids=input_ids if needs_ids else None,
            task_starts=task_starts,
            task_ends=task_ends,
            assistant_starts=assistant_starts,
            assistant_ends=assistant_ends,
            assistant_to_user_anchor=assistant_anchor,
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
        self._validate_selectors_for_model(selector_names)
        needs_task = set(selector_names) & TASK_SELECTORS
        _needs_content_span = needs_assistant_content_span(selector_names)
        _needs_tb_anchor = needs_assistant_tb_anchor(selector_names)
        token_ids_list: list[torch.Tensor] = []
        first_completion_indices: list[int] = []
        task_start_indices: list[int] = []
        task_end_indices: list[int] = []
        assistant_start_indices: list[int] = []
        assistant_end_indices: list[int] = []
        assistant_anchor_indices: list[int] = []
        for messages in messages_batch:
            has_completion = messages and messages[-1]["role"] == "assistant"
            if has_completion:
                prompt = self.format_messages(messages, add_generation_prompt=False)
                ids = self._tokenize(prompt)[0]  # (seq_len,)
                token_ids_list.append(ids)
                first_completion_indices.append(self._get_assistant_start_position(messages))
            else:
                needs_completion_sel = set(selector_names) & COMPLETION_SELECTORS
                if needs_completion_sel:
                    raise ValueError(
                        f"Selectors {needs_completion_sel} require an assistant message, "
                        f"but messages end with role '{messages[-1]['role']}'"
                    )
                prompt = self.format_messages(messages, add_generation_prompt=True)
                ids = self._tokenize(prompt)[0]
                token_ids_list.append(ids)
                first_completion_indices.append(ids.shape[0])
            if needs_task:
                start, end = self._get_task_span(messages)
                task_start_indices.append(start)
                task_end_indices.append(end)
            if _needs_content_span:
                a_start, a_end = self._get_first_assistant_span(messages)
                assistant_start_indices.append(a_start)
                assistant_end_indices.append(a_end)
            if _needs_tb_anchor:
                assistant_anchor_indices.append(self._get_assistant_to_user_anchor(messages))

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
        task_starts = task_ends = None
        assistant_starts = assistant_ends = None
        assistant_anchor = None
        if needs_task:
            task_starts = torch.tensor([
                task_start_indices[i] + (max_len - seq_lengths[i])
                for i in range(batch_size)
            ])
            task_ends = torch.tensor([
                task_end_indices[i] + (max_len - seq_lengths[i])
                for i in range(batch_size)
            ])
        if _needs_content_span:
            assistant_starts = torch.tensor([
                assistant_start_indices[i] + (max_len - seq_lengths[i])
                for i in range(batch_size)
            ])
            assistant_ends = torch.tensor([
                assistant_end_indices[i] + (max_len - seq_lengths[i])
                for i in range(batch_size)
            ])
        if _needs_tb_anchor:
            assistant_anchor = torch.tensor([
                assistant_anchor_indices[i] + (max_len - seq_lengths[i])
                for i in range(batch_size)
            ])
        return self._apply_selectors(
            activations, selector_names, shifted_first, max_len,
            input_ids=padded if needs_ids else None,
            task_starts=task_starts,
            task_ends=task_ends,
            assistant_starts=assistant_starts,
            assistant_ends=assistant_ends,
            assistant_to_user_anchor=assistant_anchor,
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
