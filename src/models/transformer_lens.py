from __future__ import annotations

from typing import Callable

import torch
import numpy as np
from transformer_lens import HookedTransformer

from src.models.base import SELECTOR_REGISTRY
from src.models.nnsight_model import GenerationResult
from src.models.registry import get_transformer_lens_name, is_valid_model
from src.types import Message


# SteeringHook takes (resid, prompt_len) and returns modified resid
SteeringHook = Callable[[torch.Tensor, int], torch.Tensor]


def last_token_steering(steering_tensor: torch.Tensor) -> SteeringHook:
    """Steer only the last token position."""
    def hook(resid: torch.Tensor, prompt_len: int) -> torch.Tensor:
        resid[:, -1, :] += steering_tensor
        return resid
    return hook


def all_tokens_steering(steering_tensor: torch.Tensor) -> SteeringHook:
    """Steer all token positions."""
    def hook(resid: torch.Tensor, prompt_len: int) -> torch.Tensor:
        resid += steering_tensor
        return resid
    return hook


def generation_only_steering(steering_tensor: torch.Tensor) -> SteeringHook:
    """Steer only newly generated tokens (after the full prompt).

    This steers only the tokens being generated in the current turn,
    not any prior conversation history in the prompt.
    """
    def hook(resid: torch.Tensor, prompt_len: int) -> torch.Tensor:
        resid[:, prompt_len:, :] += steering_tensor
        return resid
    return hook


STEERING_MODES = {
    "last_token": last_token_steering,
    "all_tokens": all_tokens_steering,
    "generation_only": generation_only_steering,
}


class TransformerLensModel:
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
            resolved_name = get_transformer_lens_name(model_name)
        else:
            resolved_name = model_name
        self.model_name = resolved_name
        self.max_new_tokens = max_new_tokens
        self.model = HookedTransformer.from_pretrained(
            resolved_name,
            device=device,
            dtype=dtype,
            attn_implementation=attn_implementation,
        )
        self.tokenizer = self.model.tokenizer

    @property
    def n_layers(self) -> int:
        return self.model.cfg.n_layers

    @property
    def hidden_dim(self) -> int:
        return self.model.cfg.d_model

    def resolve_layer(self, layer: int | float) -> int:
        """Resolve layer index. Floats in [0, 1] are relative positions."""
        if isinstance(layer, float):
            return int(layer * self.n_layers)
        return layer

    def _format_messages(self, messages: list[Message], add_generation_prompt: bool = True) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    @torch.inference_mode()
    def generate(
        self,
        messages: list[Message],
        temperature: float = 1.0,
        max_new_tokens: int | None = None,
    ) -> str:
        prompt = self._format_messages(messages, add_generation_prompt=True)
        output = self.model.generate(
            prompt,
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            temperature=temperature,
            verbose=False,
        )
        # Strip prompt from output - TransformerLens returns full sequence
        if output.startswith(prompt):
            output = output[len(prompt):]
        return output.strip()

    def _get_assistant_start_position(self, messages: list[Message]) -> int:
        """Get the token position where the assistant's content starts.

        This tokenizes messages without the assistant content to find where it begins.
        Requires messages to have at least one assistant message.
        """
        if not messages or messages[-1]["role"] != "assistant":
            raise ValueError("Messages must end with an assistant message for FIRST position")

        # Messages without assistant content - just the prompt up to where assistant starts
        prompt_messages = messages[:-1]
        prompt_with_header = self._format_messages(prompt_messages, add_generation_prompt=True)
        prompt_tokens = self.model.to_tokens(prompt_with_header)

        # The first assistant token is at this position (0-indexed)
        return prompt_tokens.shape[1]

    @torch.inference_mode()
    def get_activations(
        self,
        messages: list[Message],
        layers: list[int],
        selector_names: list[str],
    ) -> dict[str, dict[int, np.ndarray]]:
        """Get activations using token selectors.

        Returns:
            {selector_name: {layer: activation}}
        """
        prompt = self._format_messages(messages, add_generation_prompt=False)
        tokens = self.model.to_tokens(prompt)

        first_completion_idx = self._get_assistant_start_position(messages)
        if first_completion_idx >= tokens.shape[1]:
            raise ValueError(
                f"Assistant start position {first_completion_idx} is beyond sequence length {tokens.shape[1]}. "
                "This can happen if the assistant message is empty."
            )

        names_filter = [f"blocks.{layer}.hook_resid_post" for layer in layers]
        _, cache = self.model.run_with_cache(tokens, names_filter=names_filter)

        results: dict[str, dict[int, np.ndarray]] = {name: {} for name in selector_names}
        for layer in layers:
            layer_acts = cache["resid_post", layer][0]  # (seq_len, d_model)
            for name in selector_names:
                act = SELECTOR_REGISTRY[name](layer_acts, first_completion_idx)
                results[name][layer] = act.float().cpu().numpy()

        return results

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
        prompt_tokens = len(self.model.to_tokens(prompt)[0])

        completion = self.generate(messages, temperature=temperature, max_new_tokens=max_new_tokens)
        completion_tokens = len(self.model.to_tokens(completion)[0])

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
        """Generate with activation steering applied at specified layer.

        Args:
            messages: Conversation messages.
            layer: Layer index to apply steering hook.
            steering_hook: A SteeringHook function that modifies residual activations.
                Use factory functions: last_token_steering, all_tokens_steering,
                or completion_only_steering.
            temperature: Sampling temperature.
            max_new_tokens: Maximum tokens to generate.
        """
        prompt = self._format_messages(messages, add_generation_prompt=True)
        prompt_tokens = self.model.to_tokens(prompt)
        prompt_len = prompt_tokens.shape[1]

        def tl_hook(resid: torch.Tensor, hook) -> torch.Tensor:
            return steering_hook(resid, prompt_len)

        hook_name = f"blocks.{layer}.hook_resid_post"
        self.model.add_hook(hook_name, tl_hook)
        try:
            output_tokens = self.model.generate(
                prompt,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                temperature=temperature,
                return_type="tokens",
                verbose=False,
            )
        finally:
            self.model.reset_hooks()

        new_tokens = output_tokens[0, prompt_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
