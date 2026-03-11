from .base import Model, ConfigurableMockModel
from .openai_compatible import OpenAICompatibleClient, VLLMClient, CerebrasClient, OpenRouterClient, ToolCallError, GenerateRequest, BatchResult
from .base import GenerationResult, LayerHook, autoregressive_steering, all_tokens_steering, STEERING_MODES
from .registry import (
    MODEL_REGISTRY,
    ModelConfig,
    get_cerebras_name,
    get_openrouter_name,
    get_hf_name,
    is_valid_model,
    list_models,
    is_reasoning_model,
    adjust_max_tokens_for_reasoning,
)

try:
    from .huggingface_model import HuggingFaceModel
except ImportError:
    HuggingFaceModel = None  # type: ignore[assignment,misc]

try:
    from .hybrid_model import HybridActivationModel
except ImportError:
    HybridActivationModel = None  # type: ignore[assignment,misc]

BACKENDS: dict[str, type[OpenAICompatibleClient]] = {
    "openrouter": OpenRouterClient,
    "cerebras": CerebrasClient,
    "vllm": VLLMClient,
}


def get_client(
    model_name: str | None = None,
    max_new_tokens: int = 256,
    reasoning_effort: str | None = None,
    backend: str = "openrouter",
) -> OpenAICompatibleClient:
    if model_name is not None:
        max_new_tokens = adjust_max_tokens_for_reasoning(model_name, max_new_tokens)
    return BACKENDS[backend](model_name=model_name, max_new_tokens=max_new_tokens, reasoning_effort=reasoning_effort)


__all__ = [
    "Model",
    "ConfigurableMockModel",
    "HuggingFaceModel",
    "HybridActivationModel",
    "GenerationResult",
    "LayerHook",
    "autoregressive_steering",
    "all_tokens_steering",
    "STEERING_MODES",
    "OpenAICompatibleClient",
    "VLLMClient",
    "CerebrasClient",
    "OpenRouterClient",
    "ToolCallError",
    "GenerateRequest",
    "BatchResult",
    "BACKENDS",
    "get_client",
    "MODEL_REGISTRY",
    "ModelConfig",
    "get_cerebras_name",
    "get_openrouter_name",
    "get_hf_name",
    "is_valid_model",
    "list_models",
]
