"""Minimal real completion call to confirm the account can actually spend."""

import os

import requests
from dotenv import load_dotenv

load_dotenv()

r = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
    json={
        "model": "google/gemini-3-flash-preview",
        "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
        "max_tokens": 5,
    },
    timeout=60,
)
print(f"HTTP {r.status_code}")
body = r.json()
if r.status_code == 200:
    print("content:", body["choices"][0]["message"]["content"])
    print("usage:", body.get("usage"))
else:
    print(body)
