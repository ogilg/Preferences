from .base import Model, ConfigurableMockModel, TokenPosition, ActivationReduction, TokenSelectorFn, SELECTOR_REGISTRY
from .openai_compatible import OpenAICompatibleClient, HyperbolicClient, CerebrasClient, OpenRouterClient, ToolCallError, GenerateRequest, BatchResult
from .base import GenerationResult, SteeringHook, autoregressive_steering, all_tokens_steering, STEERING_MODES
from .registry import (
    MODEL_REGISTRY,
    ModelConfig,
    get_hyperbolic_name,
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

# === INFERENCE PROVIDER CONFIGURATION ===
# Change this to switch providers globally
InferenceClient: type[OpenAICompatibleClient] = OpenRouterClient


def get_client(
    model_name: str | None = None,
    max_new_tokens: int = 256,
    reasoning_effort: str | None = None,
) -> OpenAICompatibleClient:
    if model_name is not None:
        max_new_tokens = adjust_max_tokens_for_reasoning(model_name, max_new_tokens)
    return InferenceClient(model_name=model_name, max_new_tokens=max_new_tokens, reasoning_effort=reasoning_effort)


def get_default_max_concurrent() -> int:
    return InferenceClient.default_max_concurrent


__all__ = [
    "Model",
    "ConfigurableMockModel",
    "TokenPosition",
    "ActivationReduction",
    "TokenSelectorFn",
    "SELECTOR_REGISTRY",
    "HuggingFaceModel",
    "HybridActivationModel",
    "GenerationResult",
    "SteeringHook",
    "autoregressive_steering",
    "all_tokens_steering",
    "STEERING_MODES",
    "OpenAICompatibleClient",
    "HyperbolicClient",
    "CerebrasClient",
    "OpenRouterClient",
    "ToolCallError",
    "GenerateRequest",
    "BatchResult",
    "InferenceClient",
    "get_client",
    "get_default_max_concurrent",
    "MODEL_REGISTRY",
    "ModelConfig",
    "get_hyperbolic_name",
    "get_cerebras_name",
    "get_openrouter_name",
    "get_hf_name",
    "is_valid_model",
    "list_models",
]
