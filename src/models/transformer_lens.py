from __future__ import annotations

import numpy as np
from transformer_lens import HookedTransformer

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
        return output

    def get_activations(
        self,
        messages: list[Message],
        layers: list[int | float],
    ) -> dict[int, np.ndarray]:
        prompt = self._format_messages(messages, add_generation_prompt=False)
        tokens = self.model.to_tokens(prompt)

        resolved_layers = [self.resolve_layer(layer) for layer in layers]

        _, cache = self.model.run_with_cache(tokens)

        return {
            resolved: cache["resid_post", resolved][0, -1, :].cpu().numpy()
            for resolved in resolved_layers
        }
