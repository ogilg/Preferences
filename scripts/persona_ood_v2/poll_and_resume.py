"""Poll OpenRouter API until it's back, then resume measurement."""

import time
import subprocess
import sys

from dotenv import load_dotenv
load_dotenv()

import openai
import os

POLL_INTERVAL = 120  # seconds

def check_api():
    client = openai.OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )
    try:
        resp = client.chat.completions.create(
            model="google/gemma-3-27b-it",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=5,
        )
        return True
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] Still down: {type(e).__name__}: {e}")
        return False


def main():
    print("Polling OpenRouter API...")
    while not check_api():
        time.sleep(POLL_INTERVAL)

    print(f"\n[{time.strftime('%H:%M:%S')}] API is back! Launching measurement...")
    subprocess.run(
        [
            sys.executable, "-m", "scripts.persona_ood.measure_persona",
            "--config", "experiments/probe_generalization/persona_ood/v2_config.json",
            "--output", "experiments/probe_generalization/persona_ood/v2_results.json",
        ],
        check=True,
    )
    print("Measurement complete!")


if __name__ == "__main__":
    main()
