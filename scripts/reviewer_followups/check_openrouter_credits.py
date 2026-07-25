"""Query OpenRouter key status without spending credits. Never prints the key itself."""

import os

import requests
from dotenv import load_dotenv

load_dotenv()

key = os.environ["OPENROUTER_API_KEY"]
print(f"key loaded: {len(key)} chars, prefix {key[:8]}…, suffix …{key[-4:]}")

r = requests.get(
    "https://openrouter.ai/api/v1/key",
    headers={"Authorization": f"Bearer {key}"},
    timeout=30,
)
print(f"HTTP {r.status_code}")
print(r.text)
