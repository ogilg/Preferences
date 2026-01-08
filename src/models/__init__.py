from .base import Model, ConfigurableMockModel
from .ndif import NDIFModel
from .hyperbolic import OpenAICompatibleClient, HyperbolicClient, ToolCallError, GenerateRequest, BatchResult
from .cerebras import CerebrasClient

# === INFERENCE PROVIDER CONFIGURATION ===
# Change this to switch providers globally
InferenceClient: type[OpenAICompatibleClient] = CerebrasClient


def get_client(model_name: str | None = None, max_new_tokens: int = 256) -> OpenAICompatibleClient:
    return InferenceClient(model_name=model_name, max_new_tokens=max_new_tokens)


__all__ = ["Model", "ConfigurableMockModel", "NDIFModel", "OpenAICompatibleClient", "HyperbolicClient", "CerebrasClient", "ToolCallError", "GenerateRequest", "BatchResult", "InferenceClient", "get_client"]
