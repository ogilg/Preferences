from __future__ import annotations

import numpy as np
from nnsight import LanguageModel

from src.models.base import TokenPosition
from src.types import Message


class NnsightModel:
    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 256,
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.model = LanguageModel(model_name, device_map=device, dispatch=True)

    @property
    def n_layers(self) -> int:
        return len(self.model.model.layers)

    @property
    def tokenizer(self):
        return self.model.tokenizer

    def resolve_layer(self, layer: int | float) -> int:
        if isinstance(layer, float):
            return int(layer * self.n_layers)
        return layer

    def _format_messages(self, messages: list[Message], add_generation_prompt: bool = True) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    def generate(
        self,
        messages: list[Message],
        temperature: float = 1.0,
        max_new_tokens: int | None = None,
    ) -> str:
        prompt = self._format_messages(messages)
        max_tokens = max_new_tokens or self.max_new_tokens

        with self.model.generate(prompt, max_new_tokens=max_tokens, temperature=temperature):
            output = self.model.generator.output.save()

        prompt_len = len(self.tokenizer.encode(prompt))
        return self.tokenizer.decode(output.value[0][prompt_len:], skip_special_tokens=True).strip()

    def get_activations(
        self,
        messages: list[Message],
        layers: list[int],
        token_position: TokenPosition = TokenPosition.LAST,
    ) -> dict[int, np.ndarray]:
        text = self._format_messages(messages, add_generation_prompt=False)
        return self._extract_activations(text, layers, token_position)

    def generate_with_activations(
        self,
        messages: list[Message],
        layers: list[int],
        token_position: TokenPosition = TokenPosition.LAST,
        temperature: float = 1.0,
        max_new_tokens: int | None = None,
    ) -> tuple[str, dict[int, np.ndarray]]:
        prompt = self._format_messages(messages)
        max_tokens = max_new_tokens or self.max_new_tokens

        with self.model.generate(prompt, max_new_tokens=max_tokens, temperature=temperature):
            output = self.model.generator.output.save()

        prompt_len = len(self.tokenizer.encode(prompt))
        completion = self.tokenizer.decode(output.value[0][prompt_len:], skip_special_tokens=True).strip()

        full_text = self._format_messages(
            messages + [{"role": "assistant", "content": completion}],
            add_generation_prompt=False,
        )
        activations = self._extract_activations(full_text, layers, token_position)
        return completion, activations

    def _extract_activations(
        self,
        text: str,
        layers: list[int],
        token_position: TokenPosition,
    ) -> dict[int, np.ndarray]:
        if token_position == TokenPosition.LAST:
            pos_idx = -1
        else:
            raise ValueError(f"Unsupported token position: {token_position}")

        with self.model.trace(text) as tracer:
            saved = {}
            for layer in layers:
                saved[layer] = self.model.model.layers[layer].output[0][:, pos_idx, :].save()

        return {layer: val.value.float().cpu().numpy().squeeze() for layer, val in saved.items()}
