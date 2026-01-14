from .base import Model, ConfigurableMockModel
from .ndif import NDIFModel
from .openai_compatible import OpenAICompatibleClient, HyperbolicClient, CerebrasClient, OpenRouterClient, ToolCallError, GenerateRequest, BatchResult
from .registry import (
    MODEL_REGISTRY,
    ModelConfig,
    get_transformer_lens_name,
    get_hyperbolic_name,
    get_cerebras_name,
    get_openrouter_name,
    is_valid_model,
    list_models,
)

try:
    from .transformer_lens import TransformerLensModel
except ImportError:
    TransformerLensModel = None  # type: ignore[assignment,misc]

# === INFERENCE PROVIDER CONFIGURATION ===
# Change this to switch providers globally
InferenceClient: type[OpenAICompatibleClient] = OpenRouterClient


def get_client(model_name: str | None = None, max_new_tokens: int = 256) -> OpenAICompatibleClient:
    return InferenceClient(model_name=model_name, max_new_tokens=max_new_tokens)


def get_default_max_concurrent() -> int:
    return InferenceClient.default_max_concurrent


__all__ = [
    "Model",
    "ConfigurableMockModel",
    "NDIFModel",
    "TransformerLensModel",
    "OpenAICompatibleClient",
    "HyperbolicClient",
    "CerebrasClient",
    "OpenRouterClient",
    "ToolCallError",
    "GenerateRequest",
    "BatchResult",
    "InferenceClient",
    "get_client",
    "MODEL_REGISTRY",
    "ModelConfig",
    "get_transformer_lens_name",
    "get_hyperbolic_name",
    "get_cerebras_name",
    "get_openrouter_name",
    "is_valid_model",
    "list_models",
]
