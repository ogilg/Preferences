from __future__ import annotations

import torch
import numpy as np
from transformer_lens import HookedTransformer

from src.models.base import TokenPosition
from src.types import Message


class TransformerLensModel:
    def __init__(
        self,
        model_name: str,
        dtype: str = "bfloat16",
        device: str = "cuda",
        max_new_tokens: int = 256,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.model = HookedTransformer.from_pretrained(
            model_name,
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
        )
        # Strip prompt from output - TransformerLens returns full sequence
        if output.startswith(prompt):
            output = output[len(prompt):]
        return output.strip()

    @torch.inference_mode()
    def get_activations(
        self,
        messages: list[Message],
        layers: list[int],
        token_position: TokenPosition = TokenPosition.LAST,
    ) -> dict[int, np.ndarray]:
        if token_position != TokenPosition.LAST:
            raise ValueError(f"Unsupported token position: {token_position}")

        prompt = self._format_messages(messages, add_generation_prompt=False)
        tokens = self.model.to_tokens(prompt)

        names_filter = [f"blocks.{layer}.hook_resid_post" for layer in layers]
        _, cache = self.model.run_with_cache(tokens, names_filter=names_filter)

        return {
            layer: cache["resid_post", layer][0, -1, :].float().cpu().numpy()
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
    ) -> tuple[str, dict[int, np.ndarray]]:
        completion = self.generate(messages, temperature=temperature, max_new_tokens=max_new_tokens)

        full_messages = messages + [{"role": "assistant", "content": completion}]
        activations = self.get_activations(full_messages, layers, token_position)

        return completion, activations
