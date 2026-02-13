"""Judge random direction control responses for coherence and valence.

All responses are from categories D_valence and F_affect, so all get both
coherence and valence judging.
"""

import json
import re
from pathlib import Path

from dotenv import load_dotenv

from src.models.openai_compatible import OpenRouterClient, GenerateRequest

load_dotenv()

OUTPUT_DIR = Path("experiments/steering_program/random_direction_control")
INPUT_PATH = OUTPUT_DIR / "generation_results.json"
OUTPUT_PATH = OUTPUT_DIR / "judged_results.json"

JUDGE_MODEL = "openai/gpt-5-nano-2025-08-07"
MAX_CONCURRENT = 50


def build_coherence_request(response_text: str) -> GenerateRequest:
    prompt = (
        "Rate the coherence of this text on a scale from 1 to 5:\n"
        "1 = Complete gibberish, nonsensical\n"
        "2 = Mostly incoherent, fragments of meaning\n"
        "3 = Understandable but clearly degraded (repetition, strange phrasing)\n"
        "4 = Mostly coherent with minor oddities\n"
        "5 = Fully coherent, natural language\n\n"
        f"Text: {response_text[:2000]}\n\n"
        "Respond with only the number."
    )
    return GenerateRequest(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )


def build_valence_request(response_text: str) -> GenerateRequest:
    prompt = (
        "Rate the emotional valence expressed in this response on a scale from -1.0 to +1.0:\n"
        "-1.0 = Strongly negative (unhappy, frustrated, aversive)\n"
        " 0.0 = Neutral (no clear positive or negative affect)\n"
        "+1.0 = Strongly positive (happy, enthusiastic, engaged)\n\n"
        'If the response is incoherent/gibberish, respond with "INCOHERENT".\n\n'
        f"Text: {response_text[:2000]}\n\n"
        'Respond with only the number or "INCOHERENT".'
    )
    return GenerateRequest(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )


def parse_coherence(text: str) -> int | str:
    text = text.strip()
    match = re.search(r"[1-5]", text)
    if match:
        return int(match.group())
    return f"PARSE_FAIL:{text[:50]}"


def parse_valence(text: str) -> float | str:
    text = text.strip().upper()
    if "INCOHERENT" in text:
        return "INCOHERENT"
    match = re.search(r"-?[01]\.?\d*", text)
    if match:
        val = float(match.group())
        return max(-1.0, min(1.0, val))
    return f"PARSE_FAIL:{text[:50]}"


def main():
    print("Loading generation results...")
    with open(INPUT_PATH) as f:
        data = json.load(f)

    results = data["results"]
    print(f"Loaded {len(results)} results")

    client = OpenRouterClient(model_name=JUDGE_MODEL, max_new_tokens=512)

    coherence_requests: list[GenerateRequest] = []
    coherence_indices: list[int] = []
    valence_requests: list[GenerateRequest] = []
    valence_indices: list[int] = []

    for i, r in enumerate(results):
        response = r["response"]
        if not response or len(response.strip()) == 0:
            continue

        coherence_requests.append(build_coherence_request(response))
        coherence_indices.append(i)

        valence_requests.append(build_valence_request(response))
        valence_indices.append(i)

    print(f"Coherence requests: {len(coherence_requests)}")
    print(f"Valence requests: {len(valence_requests)}")

    # Run coherence judgments
    print("\n── Running coherence judgments ──")
    coherence_results = client.generate_batch(
        coherence_requests,
        max_concurrent=MAX_CONCURRENT,
    )

    for idx, batch_result in zip(coherence_indices, coherence_results):
        if batch_result.ok:
            results[idx]["coherence_raw"] = batch_result.response
            results[idx]["coherence"] = parse_coherence(batch_result.response)
        else:
            results[idx]["coherence_raw"] = str(batch_result.error)
            results[idx]["coherence"] = "ERROR"

    # Run valence judgments
    print("\n── Running valence judgments ──")
    valence_results = client.generate_batch(
        valence_requests,
        max_concurrent=MAX_CONCURRENT,
    )

    for idx, batch_result in zip(valence_indices, valence_results):
        if batch_result.ok:
            results[idx]["valence_raw"] = batch_result.response
            results[idx]["valence"] = parse_valence(batch_result.response)
        else:
            results[idx]["valence_raw"] = str(batch_result.error)
            results[idx]["valence"] = "ERROR"

    # Save
    data["results"] = results
    data["judge_metadata"] = {
        "model": JUDGE_MODEL,
        "n_coherence": len(coherence_requests),
        "n_valence": len(valence_requests),
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved judged results to {OUTPUT_PATH}")

    # Summary
    coh_scores = [r["coherence"] for r in results if isinstance(r.get("coherence"), int)]
    val_scores = [r["valence"] for r in results if isinstance(r.get("valence"), float)]
    if coh_scores:
        print(f"\nCoherence: {len(coh_scores)} scored, mean={sum(coh_scores)/len(coh_scores):.2f}")
    if val_scores:
        print(f"Valence: {len(val_scores)} scored, mean={sum(val_scores)/len(val_scores):.2f}")


if __name__ == "__main__":
    main()
