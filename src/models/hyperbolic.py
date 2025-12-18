"""Hyperbolic API model implementation."""

import json
import os
from typing import Any

from openai import OpenAI

from src.types import Message


class ToolCallError(Exception):
    """Raised when tool call parsing fails."""

    pass


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
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        """Generate a response for the given messages.

        Args:
            messages: The conversation messages.
            temperature: Sampling temperature.
            tools: Optional list of tool definitions for function calling.
                When provided, tool_choice is set to "required".

        Returns:
            If tools are provided: JSON string of the tool call arguments.
            Otherwise: The model's text response.

        Raises:
            ToolCallError: If tool use was requested but failed.
        """
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_new_tokens,
        }

        if tools is not None:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as e:
            raise ToolCallError(f"API call failed: {e}") from e

        message = response.choices[0].message

        # Tool use path
        if tools is not None:
            if not message.tool_calls:
                raise ToolCallError(
                    f"Expected tool call but got text: {message.content}"
                )
            tool_call = message.tool_calls[0]
            try:
                args = json.loads(tool_call.function.arguments)
                return json.dumps(args)
            except json.JSONDecodeError as e:
                raise ToolCallError(
                    f"Invalid JSON in tool arguments: {tool_call.function.arguments}"
                ) from e

        # Standard text path
        return (message.content or "").strip()

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
