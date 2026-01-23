from __future__ import annotations

import torch
import numpy as np
from transformer_lens import HookedTransformer

from src.models.base import TokenPosition
from src.models.nnsight_model import GenerationResult
from src.models.registry import get_transformer_lens_name, is_valid_model
from src.types import Message


class TransformerLensModel:
    def __init__(
        self,
        model_name: str,
        dtype: str = "bfloat16",
        device: str = "cuda",
        max_new_tokens: int = 256,
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
        )
        self.tokenizer = self.model.tokenizer

    @property
    def n_layers(self) -> int:
        return self.model.cfg.n_layers

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
        token_position: TokenPosition = TokenPosition.LAST,
    ) -> dict[int, np.ndarray]:
        prompt = self._format_messages(messages, add_generation_prompt=False)
        tokens = self.model.to_tokens(prompt)

        if token_position == TokenPosition.LAST:
            pos_idx = -1
        elif token_position == TokenPosition.FIRST:
            pos_idx = self._get_assistant_start_position(messages)
            if pos_idx >= tokens.shape[1]:
                raise ValueError(
                    f"Assistant start position {pos_idx} is beyond sequence length {tokens.shape[1]}. "
                    "This can happen if the assistant message is empty."
                )
        else:
            raise ValueError(f"Unsupported token position: {token_position}")

        names_filter = [f"blocks.{layer}.hook_resid_post" for layer in layers]
        _, cache = self.model.run_with_cache(tokens, names_filter=names_filter)

        return {
            layer: cache["resid_post", layer][0, pos_idx, :].float().cpu().numpy()
            for layer in layers
        }

    @torch.inference_mode()
    def generate_with_activations(
        self,
        messages: list[Message],
        layers: list[int],
        token_position: TokenPosition = TokenPosition.LAST,
        temperature: float = 1.0,
        max_new_tokens: int | None = None,
    ) -> GenerationResult:
        prompt = self._format_messages(messages, add_generation_prompt=True)
        prompt_tokens = len(self.model.to_tokens(prompt)[0])

        completion = self.generate(messages, temperature=temperature, max_new_tokens=max_new_tokens)
        completion_tokens = len(self.model.to_tokens(completion)[0])

        full_messages = messages + [{"role": "assistant", "content": completion}]
        activations = self.get_activations(full_messages, layers, token_position)

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
        steering_vector: np.ndarray,
        steering_coefficient: float = 1.0,
        temperature: float = 1.0,
        max_new_tokens: int | None = None,
    ) -> str:
        """Generate with activation steering applied at specified layer.

        Adds scaled steering vector to residual stream at last token position
        during each generation step.
        """
        prompt = self._format_messages(messages, add_generation_prompt=True)

        steering_tensor = torch.tensor(
            steering_vector * steering_coefficient,
            dtype=self.model.cfg.dtype,
            device=self.model.cfg.device,
        )

        def steering_hook(resid: torch.Tensor, hook) -> torch.Tensor:
            # Always steer the last token position
            resid[:, -1, :] += steering_tensor
            return resid

        hook_name = f"blocks.{layer}.hook_resid_post"
        prompt_tokens = self.model.to_tokens(prompt)
        prompt_len = prompt_tokens.shape[1]

        self.model.add_hook(hook_name, steering_hook)
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
