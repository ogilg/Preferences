"""Canonical model registry with backend-specific name mappings.

This module defines canonical model names used throughout the codebase and provides
mappings to backend-specific model names (TransformerLens, Hyperbolic, OpenRouter, etc.).

Usage:
    # In tests and application code, use canonical names:
    client = get_client("llama-3.1-8b")

    # The client resolves to the appropriate backend name automatically
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a model across different backends."""

    canonical_name: str
    transformer_lens_name: str | None
    hyperbolic_name: str | None
    cerebras_name: str | None
    openrouter_name: str | None


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "llama-3.1-8b": ModelConfig(
        canonical_name="llama-3.1-8b",
        transformer_lens_name="meta-llama/Llama-3.1-8B-Instruct",
        hyperbolic_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        cerebras_name="llama3.1-8b",
        openrouter_name="meta-llama/llama-3.1-8b-instruct",
    ),
    # "llama-3.1-70b": ModelConfig(
    #     canonical_name="llama-3.1-70b",
    #     transformer_lens_name="meta-llama/Llama-3.1-70B-Instruct",
    #     hyperbolic_name="meta-llama/Meta-Llama-3.1-70B-Instruct",
    #     cerebras_name="llama3.1-70b",
    #     openrouter_name="meta-llama/llama-3.1-70b-instruct",
    # ),
    "llama-3.3-70b": ModelConfig(
        canonical_name="llama-3.3-70b",
        transformer_lens_name="meta-llama/Llama-3.3-70B-Instruct",
        hyperbolic_name=None,  # Not available on Hyperbolic
        cerebras_name=None,  # Not available on Cerebras
        openrouter_name="meta-llama/llama-3.3-70b-instruct",
    ),
    "qwen3-8b": ModelConfig(
        canonical_name="qwen3-8b",
        transformer_lens_name=None,  # Not available
        hyperbolic_name=None,  # Not available
        cerebras_name=None,  # Not available
        openrouter_name="qwen/qwen3-8b",
    ),
}


def get_transformer_lens_name(canonical_name: str) -> str:
    """Get TransformerLens model name from canonical name."""
    config = MODEL_REGISTRY[canonical_name]
    if config.transformer_lens_name is None:
        raise ValueError(f"Model {canonical_name} not available for TransformerLens")
    return config.transformer_lens_name


def get_hyperbolic_name(canonical_name: str) -> str:
    """Get Hyperbolic API model name from canonical name."""
    config = MODEL_REGISTRY[canonical_name]
    if config.hyperbolic_name is None:
        raise ValueError(f"Model {canonical_name} not available for Hyperbolic")
    return config.hyperbolic_name


def get_cerebras_name(canonical_name: str) -> str:
    """Get Cerebras API model name from canonical name."""
    config = MODEL_REGISTRY[canonical_name]
    if config.cerebras_name is None:
        raise ValueError(f"Model {canonical_name} not available for Cerebras")
    return config.cerebras_name


def get_openrouter_name(canonical_name: str) -> str:
    """Get OpenRouter API model name from canonical name."""
    config = MODEL_REGISTRY[canonical_name]
    if config.openrouter_name is None:
        raise ValueError(f"Model {canonical_name} not available for OpenRouter")
    return config.openrouter_name


def is_valid_model(canonical_name: str) -> bool:
    """Check if a canonical model name is registered."""
    return canonical_name in MODEL_REGISTRY


def list_models() -> list[str]:
    """List all available canonical model names."""
    return list(MODEL_REGISTRY.keys())
