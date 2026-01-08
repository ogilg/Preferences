from .hyperbolic import OpenAICompatibleClient


class CerebrasClient(OpenAICompatibleClient):
    _api_key_env_var = "CEREBRAS_API_KEY"
    _base_url = "https://api.cerebras.ai/v1"
    _default_model = "llama3.1-8b"
    default_max_concurrent = 150
    _model_aliases = {
        "llama-3.1-8b": "llama3.1-8b",
        "llama-3.1-70b": "llama-3.1-70b",
    }
