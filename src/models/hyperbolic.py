"""Hyperbolic API model implementation."""

import os
from openai import OpenAI

from src.types import Message


class HyperbolicModel:
    """Model via Hyperbolic API (OpenAI-compatible)."""

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        max_new_tokens: int = 256,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.client = OpenAI(
            api_key=os.environ.get("HYPERBOLIC_API_KEY"),
            base_url="https://api.hyperbolic.xyz/v1",
        )

    def generate(
        self,
        messages: list[Message],
        temperature: float = 1.0,
    ) -> str:
        """Generate a response for the given messages."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=self.max_new_tokens,
        )
        return response.choices[0].message.content.strip()

    def get_logprobs(
        self,
        messages: list[Message],
        max_tokens: int = 1,
    ) -> dict[str, float]:
        """Get log probabilities for next tokens."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=1,
            logprobs=True,
            top_logprobs=max_tokens,
        )
        top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
        return {lp.token: lp.logprob for lp in top_logprobs}
