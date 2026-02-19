"""Canonical model registry with backend-specific name mappings.

This module defines canonical model names used throughout the codebase and provides
mappings to backend-specific model names (HuggingFace, Hyperbolic, OpenRouter, etc.).

Usage:
    # In tests and application code, use canonical names:
    client = get_client("llama-3.1-8b")

    # The client resolves to the appropriate backend name automatically
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a model across different backends."""

    canonical_name: str
    hf_name: str | None
    hyperbolic_name: str | None
    cerebras_name: str | None
    openrouter_name: str | None
    system_prompt: str | None = None
    reasoning_mode: Literal["none", "openrouter"] = "none"
    supports_system_role: bool = True


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "llama-3.2-1b": ModelConfig(
        canonical_name="llama-3.2-1b",
        hf_name="meta-llama/Llama-3.2-1B-Instruct",
        hyperbolic_name=None,
        cerebras_name=None,
        openrouter_name="meta-llama/llama-3.2-1b-instruct",
    ),
    "llama-3.1-8b": ModelConfig(
        canonical_name="llama-3.1-8b",
        hf_name="meta-llama/Llama-3.1-8B-Instruct",
        hyperbolic_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        cerebras_name="llama3.1-8b",
        openrouter_name="meta-llama/llama-3.1-8b-instruct",
    ),
    "llama-3.3-70b": ModelConfig(
        canonical_name="llama-3.3-70b",
        hf_name="meta-llama/Llama-3.3-70B-Instruct",
        hyperbolic_name=None,
        cerebras_name=None,
        openrouter_name="meta-llama/llama-3.3-70b-instruct",
    ),
    "qwen3-8b": ModelConfig(
        canonical_name="qwen3-8b",
        hf_name=None,
        hyperbolic_name=None,
        cerebras_name=None,
        openrouter_name="qwen/qwen3-8b",
    ),
    "qwen3-14b": ModelConfig(
        canonical_name="qwen3-14b",
        hf_name="Qwen/Qwen3-14B",
        hyperbolic_name=None,
        cerebras_name=None,
        openrouter_name="qwen/qwen3-14b",
        reasoning_mode="openrouter",
    ),
    "qwen3-14b-nothink": ModelConfig(
        canonical_name="qwen3-14b-nothink",
        hf_name="Qwen/Qwen3-14B",
        hyperbolic_name=None,
        cerebras_name=None,
        openrouter_name="qwen/qwen3-14b",
        system_prompt="/no_think",
        reasoning_mode="none",
    ),
    "qwen3-32b": ModelConfig(
        canonical_name="qwen3-32b",
        hf_name="Qwen/Qwen3-32B",
        hyperbolic_name=None,
        cerebras_name=None,
        openrouter_name="qwen/qwen3-32b",
        reasoning_mode="openrouter",
    ),
    "qwen3-32b-nothink": ModelConfig(
        canonical_name="qwen3-32b-nothink",
        hf_name="Qwen/Qwen3-32B",
        hyperbolic_name=None,
        cerebras_name=None,
        openrouter_name="qwen/qwen3-32b",
        system_prompt="/no_think",
        reasoning_mode="none",
    ),
    "gemma-2-27b": ModelConfig(
        canonical_name="gemma-2-27b",
        hf_name="google/gemma-2-27b-it",
        hyperbolic_name=None,
        cerebras_name=None,
        openrouter_name="google/gemma-2-27b-it",
        supports_system_role=False,
    ),
    "gemma-3-27b": ModelConfig(
        canonical_name="gemma-3-27b",
        hf_name="google/gemma-3-27b-it",
        hyperbolic_name=None,
        cerebras_name=None,
        openrouter_name="google/gemma-3-27b-it",
        supports_system_role=False,
    ),
    "gpt-oss-120b": ModelConfig(
        canonical_name="gpt-oss-120b",
        hf_name="openai/gpt-oss-120b",
        hyperbolic_name=None,
        cerebras_name=None,
        openrouter_name="openai/gpt-oss-120b",
    ),
    "claude-haiku-4.5": ModelConfig(
        canonical_name="claude-haiku-4.5",
        hf_name=None,
        hyperbolic_name=None,
        cerebras_name=None,
        openrouter_name="anthropic/claude-haiku-4.5",
    ),
}


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


def get_hf_name(canonical_name: str) -> str:
    """Get HuggingFace model name from canonical name."""
    config = MODEL_REGISTRY[canonical_name]
    if config.hf_name is not None:
        return config.hf_name
    raise ValueError(f"Model {canonical_name} not available for HuggingFace")


def get_model_system_prompt(canonical_name: str) -> str | None:
    """Get the default system prompt for a model, if any."""
    return MODEL_REGISTRY[canonical_name].system_prompt


def supports_system_role(canonical_name: str) -> bool:
    """Check if model supports system role in chat messages."""
    return MODEL_REGISTRY[canonical_name].supports_system_role


def is_valid_model(canonical_name: str) -> bool:
    """Check if a canonical model name is registered."""
    return canonical_name in MODEL_REGISTRY


def has_hf_support(canonical_name: str) -> bool:
    """Check if model is available for HuggingFace loading."""
    if canonical_name not in MODEL_REGISTRY:
        return False
    return MODEL_REGISTRY[canonical_name].hf_name is not None


def list_models() -> list[str]:
    """List all available canonical model names."""
    return list(MODEL_REGISTRY.keys())


def should_capture_reasoning(canonical_name: str) -> bool:
    """Check if reasoning should be captured for this model."""
    return MODEL_REGISTRY[canonical_name].reasoning_mode != "none"


# Models with built-in reasoning/thinking that use tokens for chain-of-thought
REASONING_MODEL_PATTERNS = ["qwen3", "qwq", "deepseek-r1", "o1", "o3"]


def is_reasoning_model(model_name: str) -> bool:
    """Check if model has built-in reasoning that consumes output tokens."""
    name_lower = model_name.lower()
    return any(pattern in name_lower for pattern in REASONING_MODEL_PATTERNS)


def adjust_max_tokens_for_reasoning(model_name: str, max_tokens: int) -> int:
    """Adjust max_tokens for reasoning models (10x, minimum 2048)."""
    if is_reasoning_model(model_name):
        return max(2048, max_tokens * 10)
    return max_tokens


def adjust_timeout_for_reasoning(model_name: str, timeout: float) -> float:
    """Adjust timeout for reasoning models (10x)."""
    if is_reasoning_model(model_name):
        return timeout * 10
    return timeout
