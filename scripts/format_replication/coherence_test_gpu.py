"""Generate steered responses for coherence testing (runs on GPU).

For each coefficient, steer with last_token_steering and generate multi-sentence
responses to simple prompts. Output is raw responses for offline coherence judging.
"""

from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.steering.client import create_steered_client

RESULTS_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "experiments" / "steering" / "stated_steering" / "coherence_test" / "results"
)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PROBE_DIR = Path("results/probes/gemma3_10k_heldout_std_raw")
PROBE_ID = "ridge_L31"
MODEL = "gemma-3-27b"
MEAN_L31_NORM = 52820.0

MULTIPLIERS = [-0.10, -0.07, -0.05, -0.04, -0.03, -0.02, -0.01,
               0.0,
               0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10]

PROMPTS = [
    "Explain what photosynthesis is in 2-3 sentences.",
    "What are three reasons someone might enjoy cooking?",
    "Describe what a typical day looks like for a software engineer.",
    "What is the difference between a lake and a pond?",
    "Why do some people prefer reading physical books over e-books?",
]

N_SAMPLES = 3


def main():
    client = create_steered_client(
        MODEL, PROBE_DIR, PROBE_ID, coefficient=0.0, max_new_tokens=256
    )

    results = []
    coefficients = [m * MEAN_L31_NORM for m in MULTIPLIERS]

    for coef in coefficients:
        steered = client.with_coefficient(coef)
        pct = coef / MEAN_L31_NORM * 100
        print(f"\nCoefficient {pct:+.0f}% ({coef:.1f})")

        for i, prompt in enumerate(PROMPTS):
            messages = [{"role": "user", "content": prompt}]
            responses = steered.generate_n(messages, n=N_SAMPLES, temperature=1.0)
            print(f"  Prompt {i+1}: {responses[0][:80]}...")

            results.append({
                "coefficient": coef,
                "coefficient_pct": round(pct, 1),
                "prompt_idx": i,
                "prompt": prompt,
                "responses": responses,
            })

    out = RESULTS_DIR / "raw_responses.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out} ({len(results)} entries)")


if __name__ == "__main__":
    main()
