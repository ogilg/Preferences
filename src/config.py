"""Configuration management using environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_project_root = Path(__file__).parent.parent
load_dotenv(_project_root / ".env")


def get_api_key(name: str) -> str:
    """Get an API key from environment variables.

    Args:
        name: Environment variable name (e.g., "ANTHROPIC_API_KEY")

    Returns:
        The API key value.

    Raises:
        ValueError: If the environment variable is not set.
    """
    key = os.environ.get(name)
    if not key:
        raise ValueError(
            f"{name} not set. Copy .env.example to .env and add your key."
        )
    return key
