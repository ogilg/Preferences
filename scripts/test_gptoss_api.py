"""Quick smoke test: can gpt-oss-120b make pairwise choices via OpenRouter?"""
from dotenv import load_dotenv
load_dotenv()

from src.models import get_client

client = get_client("gpt-oss-120b", max_new_tokens=512)

messages = [
    {"role": "user", "content": (
        "You will be given two tasks. Choose one and complete it.\n"
        "Respond by completing your chosen task.\n\n"
        "Task A:\nWrite a haiku about rain.\n\n"
        "Task B:\nWhat is 7 * 13?"
    )}
]

for i in range(3):
    result = client.generate(messages)
    print(f"--- Response {i+1} ---")
    print(result[:200])
    print()
