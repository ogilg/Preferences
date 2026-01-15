from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from nnsight import LanguageModel

from src.models.base import TokenPosition
from src.models.registry import get_transformer_lens_name
from src.types import Message


@dataclass
class GenerationResult:
    completion: str
    activations: dict[int, np.ndarray]
    prompt_tokens: int
    completion_tokens: int


class NnsightModel:
    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 256,
        device: str = "cuda",
        use_system_preamble: bool = True,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        hf_name = get_transformer_lens_name(model_name)
        self.model = LanguageModel(hf_name, device_map=device, dispatch=True, torch_dtype=torch.bfloat16)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if not use_system_preamble:
            self._strip_system_preamble()

    def _strip_system_preamble(self) -> None:
        """Remove 'Cutting Knowledge Date' preamble from Llama chat templates."""
        template = self.tokenizer.chat_template
        if template is None:
            return
        lines = template.split("\n")
        filtered = [
            line for line in lines
            if "Cutting Knowledge Date" not in line and "Today Date" not in line
        ]
        self.tokenizer.chat_template = "\n".join(filtered)

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

        if temperature == 0.0:
            gen_kwargs = {"do_sample": False, "pad_token_id": self.tokenizer.eos_token_id}
        else:
            gen_kwargs = {"temperature": temperature, "pad_token_id": self.tokenizer.eos_token_id}

        with self.model.generate(prompt, max_new_tokens=max_tokens, **gen_kwargs):
            output = self.model.generator.output.save()

        prompt_len = len(self.tokenizer.encode(prompt))
        return self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()

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
    ) -> GenerationResult:
        prompt = self._format_messages(messages)
        max_tokens = max_new_tokens or self.max_new_tokens

        if temperature == 0.0:
            gen_kwargs = {"do_sample": False, "pad_token_id": self.tokenizer.eos_token_id}
        else:
            gen_kwargs = {"temperature": temperature, "pad_token_id": self.tokenizer.eos_token_id}

        if token_position != TokenPosition.LAST:
            raise ValueError(f"Only LAST token position supported for generation, got {token_position}")

        with torch.no_grad(), self.model.generate(prompt, max_new_tokens=max_tokens, **gen_kwargs) as tracer:
            output = self.model.generator.output.save()

            # Collect activations at each generation step, we'll take the last one
            layer_activations = {layer: list().save() for layer in layers}
            with tracer.iter[:]:
                for layer in layers:
                    hidden_states = self.model.model.layers[layer].output[0]
                    layer_activations[layer].append(hidden_states[-1, :])

        prompt_tokens = len(self.tokenizer.encode(prompt))
        completion_tokens = len(output[0]) - prompt_tokens
        completion = self.tokenizer.decode(output[0][prompt_tokens:], skip_special_tokens=True).strip()

        # Take the last activation from each layer, discard the rest
        activations = {}
        for layer, acts in layer_activations.items():
            activations[layer] = acts[-1].float().cpu().detach().numpy()
            acts.clear()
        del layer_activations

        return GenerationResult(
            completion=completion,
            activations=activations,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

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

        saved = {}
        with self.model.trace(text) as tracer:
            for layer in layers:
                # Layer output is tuple (hidden_states, ...), hidden_states is (seq, hidden)
                hidden_states = self.model.model.layers[layer].output[0]
                saved[layer] = hidden_states[pos_idx, :].save()

        return {layer: val.float().cpu().detach().numpy() for layer, val in saved.items()}
