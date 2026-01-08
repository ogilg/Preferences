from .hyperbolic import OpenAICompatibleModel


class CerebrasModel(OpenAICompatibleModel):
    _api_key_env_var = "CEREBRAS_API_KEY"
    _base_url = "https://api.cerebras.ai/v1"
    _default_model = "llama3.1-8b"
