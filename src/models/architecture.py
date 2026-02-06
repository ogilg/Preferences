"""Model architecture detection for HuggingFace models.

Maps model_type from HF config to layer accessor functions.
"""

from __future__ import annotations

from typing import Callable, Any

import torch.nn as nn


LayerAccessor = Callable[[Any], nn.ModuleList]


def _standard_layers(model: Any) -> nn.ModuleList:
    return model.model.layers


def _gemma3_layers(model: Any) -> nn.ModuleList:
    return model.model.language_model.layers


ARCHITECTURE_CONFIGS: dict[str, LayerAccessor] = {
    "llama": _standard_layers,
    "qwen2": _standard_layers,
    "qwen3": _standard_layers,
    "gemma": _standard_layers,
    "gemma2": _standard_layers,
    "gemma3": _gemma3_layers,
}


def get_layer_accessor(model_type: str) -> LayerAccessor:
    """Get layer accessor function for a model type."""
    if model_type not in ARCHITECTURE_CONFIGS:
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            f"Supported types: {list(ARCHITECTURE_CONFIGS.keys())}"
        )
    return ARCHITECTURE_CONFIGS[model_type]


def get_layers(model: Any) -> nn.ModuleList:
    """Get the transformer layers from a HuggingFace model."""
    model_type = model.config.model_type
    accessor = get_layer_accessor(model_type)
    return accessor(model)


def get_n_layers(model: Any) -> int:
    """Get number of layers in a HuggingFace model."""
    return len(get_layers(model))


def get_hidden_dim(model: Any) -> int:
    """Get hidden dimension from model config."""
    config = model.config
    if hasattr(config, "text_config"):
        return config.text_config.hidden_size
    return config.hidden_size
