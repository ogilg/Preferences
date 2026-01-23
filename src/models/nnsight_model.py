from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from nnsight import LanguageModel

from src.models.base import ActivationDtype, ActivationReduction, TokenPosition
from src.models.registry import get_transformer_lens_name
from src.types import Message


@dataclass
class GenerationResult:
    completion: str
    activations: dict[int, np.ndarray] | dict[str, dict[int, np.ndarray]]
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
        dtype: ActivationDtype = ActivationDtype.FLOAT32,
    ) -> dict[int, np.ndarray]:
        text = self._format_messages(messages, add_generation_prompt=False)
        return self._extract_activations(text, layers, token_position, dtype)

    def generate_with_activations(
        self,
        messages: list[Message],
        layers: list[int],
        token_position: TokenPosition = TokenPosition.LAST,
        temperature: float = 1.0,
        max_new_tokens: int | None = None,
        dtype: ActivationDtype = ActivationDtype.FLOAT32,
    ) -> GenerationResult:
        prompt = self._format_messages(messages)
        max_tokens = max_new_tokens or self.max_new_tokens

        if temperature == 0.0:
            gen_kwargs = {"do_sample": False, "pad_token_id": self.tokenizer.eos_token_id}
        else:
            gen_kwargs = {"temperature": temperature, "pad_token_id": self.tokenizer.eos_token_id}

        if token_position != TokenPosition.LAST:
            raise ValueError(f"Only LAST token position supported for generation, got {token_position}")

        # Step 1: Generate completion
        with torch.no_grad(), self.model.generate(prompt, max_new_tokens=max_tokens, **gen_kwargs):
            output = self.model.generator.output.save()

        prompt_tokens = len(self.tokenizer.encode(prompt))
        completion_tokens = len(output[0]) - prompt_tokens
        completion = self.tokenizer.decode(output[0][prompt_tokens:], skip_special_tokens=True).strip()

        # Step 2: Trace full sequence to get activations at last token
        full_messages = messages + [{"role": "assistant", "content": completion}]
        full_text = self._format_messages(full_messages, add_generation_prompt=False)
        layer_modules = [self.model.model.layers[layer] for layer in layers]

        with self.model.trace(full_text) as tracer:
            cache = tracer.cache(modules=layer_modules)

        activations = {}
        for layer in layers:
            module_key = f"model.model.layers.{layer}"
            hidden_states = cache[module_key].output[0]  # (seq_len, hidden_dim)
            activations[layer] = self._convert_dtype(hidden_states[-1, :], dtype)

        return GenerationResult(
            completion=completion,
            activations=activations,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def generate_with_activations_efficient(
        self,
        messages: list[Message],
        layers: list[int],
        reduction: ActivationReduction = ActivationReduction.LAST,
        temperature: float = 1.0,
        max_new_tokens: int | None = None,
        chunk_size: int = 10,
        dtype: ActivationDtype = ActivationDtype.FLOAT32,
    ) -> GenerationResult:
        """Single-pass generation with activation extraction.

        More efficient than generate_with_activations as it captures activations
        during generation rather than in a separate forward pass.

        Args:
            reduction: How to reduce activations across generated tokens.
                LAST: Only the last generated token's activation (hidden_dim,)
                MEAN: Mean across all generated tokens (hidden_dim,)
                CHUNKED_MEAN: Mean of chunk_size-token chunks (n_chunks, hidden_dim)
            chunk_size: Size of chunks for CHUNKED_MEAN reduction.
        """
        prompt = self._format_messages(messages)
        max_tokens = max_new_tokens or self.max_new_tokens

        if temperature == 0.0:
            gen_kwargs = {"do_sample": False, "pad_token_id": self.tokenizer.eos_token_id}
        else:
            gen_kwargs = {"temperature": temperature, "pad_token_id": self.tokenizer.eos_token_id}

        with torch.no_grad(), self.model.generate(prompt, max_new_tokens=max_tokens, **gen_kwargs) as tracer:
            output = self.model.generator.output.save()

            layer_activations = {layer: list().save() for layer in layers}
            with tracer.iter[:]:
                for layer in layers:
                    hidden_states = self.model.model.layers[layer].output[0]
                    layer_activations[layer].append(hidden_states[-1, :])

        prompt_tokens = len(self.tokenizer.encode(prompt))
        completion_tokens = len(output[0]) - prompt_tokens
        completion = self.tokenizer.decode(output[0][prompt_tokens:], skip_special_tokens=True).strip()

        activations = {}
        for layer, acts in layer_activations.items():
            # Stack in float32 for reduction, then convert to target dtype
            stacked = np.stack([a.float().cpu().numpy() for a in acts])
            if reduction == ActivationReduction.LAST:
                reduced = stacked[-1]
            elif reduction == ActivationReduction.MEAN:
                reduced = stacked.mean(axis=0)
            elif reduction == ActivationReduction.CHUNKED_MEAN:
                n_tokens = stacked.shape[0]
                n_chunks = (n_tokens + chunk_size - 1) // chunk_size
                padded = np.zeros((n_chunks * chunk_size, stacked.shape[1]), dtype=stacked.dtype)
                padded[:n_tokens] = stacked
                chunked = padded.reshape(n_chunks, chunk_size, -1)
                reduced = chunked.mean(axis=1)
            activations[layer] = self._convert_dtype_numpy(reduced, dtype)
            acts.clear()
        del layer_activations

        return GenerationResult(
            completion=completion,
            activations=activations,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def _convert_dtype(self, tensor: torch.Tensor, dtype: ActivationDtype) -> np.ndarray:
        """Convert torch tensor to numpy with specified dtype."""
        if dtype == ActivationDtype.FLOAT32:
            return tensor.float().cpu().detach().numpy()
        elif dtype == ActivationDtype.FLOAT16:
            return tensor.half().cpu().detach().numpy()
        elif dtype == ActivationDtype.BFLOAT16:
            # numpy doesn't support bfloat16, store as uint16 view
            return tensor.bfloat16().cpu().detach().view(torch.uint16).numpy()
        raise ValueError(f"Unsupported dtype: {dtype}")

    def _convert_dtype_numpy(self, arr: np.ndarray, dtype: ActivationDtype) -> np.ndarray:
        """Convert numpy array to specified dtype."""
        if dtype == ActivationDtype.FLOAT32:
            return arr.astype(np.float32)
        elif dtype == ActivationDtype.FLOAT16:
            return arr.astype(np.float16)
        elif dtype == ActivationDtype.BFLOAT16:
            # Convert via torch for bfloat16
            tensor = torch.from_numpy(arr).bfloat16()
            return tensor.view(torch.uint16).numpy()
        raise ValueError(f"Unsupported dtype: {dtype}")

    def _extract_activations(
        self,
        text: str,
        layers: list[int],
        token_position: TokenPosition,
        dtype: ActivationDtype = ActivationDtype.FLOAT32,
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

        return {layer: self._convert_dtype(val, dtype) for layer, val in saved.items()}

    def generate_with_steering(
        self,
        messages: list[Message],
        layer: int,
        steering_vector: np.ndarray,
        steering_coefficient: float,
        temperature: float = 1.0,
        max_new_tokens: int | None = None,
    ) -> str:
        """Generate with activation steering applied at specified layer.

        Adds scaled steering vector to residual stream at last token position
        during each generation step.

        Args:
            layer: Layer index to apply steering
            steering_vector: Unit-normalized direction vector (hidden_dim,)
            steering_coefficient: Scalar multiplier for steering strength
        """
        prompt = self._format_messages(messages)
        max_tokens = max_new_tokens or self.max_new_tokens

        if temperature == 0.0:
            gen_kwargs = {"do_sample": False, "pad_token_id": self.tokenizer.eos_token_id}
        else:
            gen_kwargs = {"temperature": temperature, "pad_token_id": self.tokenizer.eos_token_id}

        # Convert steering vector to tensor
        steering_tensor = torch.tensor(
            steering_vector * steering_coefficient,
            dtype=torch.bfloat16,
            device=self.model.device,
        )

        with torch.no_grad(), self.model.generate(prompt, max_new_tokens=max_tokens, **gen_kwargs) as tracer:
            output = self.model.generator.output.save()

            # Apply steering at each generation step
            with tracer.all():
                self.model.model.layers[layer].output[0][-1, :] += steering_tensor

        prompt_len = len(self.tokenizer.encode(prompt))
        return self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()
