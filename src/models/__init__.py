from .base import Model, ConfigurableMockModel
from .ndif import NDIFModel
from .openai_compatible import OpenAICompatibleClient, HyperbolicClient, CerebrasClient, OpenRouterClient, ToolCallError, GenerateRequest, BatchResult

# === INFERENCE PROVIDER CONFIGURATION ===
# Change this to switch providers globally
InferenceClient: type[OpenAICompatibleClient] = OpenRouterClient


def get_client(model_name: str | None = None, max_new_tokens: int = 256) -> OpenAICompatibleClient:
    return InferenceClient(model_name=model_name, max_new_tokens=max_new_tokens)


def get_default_max_concurrent() -> int:
    return InferenceClient.default_max_concurrent


__all__ = ["Model", "ConfigurableMockModel", "NDIFModel", "OpenAICompatibleClient", "HyperbolicClient", "CerebrasClient", "OpenRouterClient", "ToolCallError", "GenerateRequest", "BatchResult", "InferenceClient", "get_client"]
