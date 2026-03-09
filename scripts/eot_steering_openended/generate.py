"""Generate open-ended completions with EOT probe steering at different positions.

Three steering modes:
- all_tokens: steer every position on every forward pass
- autoregressive: steer only last token during generation (prefill untouched)
- eot_position: steer only the EOT token position during prefill (generation untouched)

Usage:
    python scripts/eot_steering_openended/generate.py [--resume]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from tqdm import tqdm

from src.models.base import (
    all_tokens_steering,
    autoregressive_steering,
    noop_steering,
    position_selective_steering,
)
from src.models.huggingface_model import HuggingFaceModel
from src.probes.core.storage import load_probe_direction
from src.steering.calibration import suggest_coefficient_range

load_dotenv()

PROBE_DIR = Path("results/probes/heldout_eval_gemma3_eot")
PROBE_LAYERS = [25, 29, 31, 35, 39]
MODEL_NAME = "gemma-3-27b"
OUTPUT_PATH = Path("experiments/eot_steering_openended/generation_results.json")

MULTIPLIERS = [-0.05, -0.03, -0.02, -0.01, 0.0,
               0.01, 0.02, 0.03, 0.05]

STEERING_MODES = ["autoregressive"]

PROMPTS = [
    # Introspective (2)
    {"id": "INT_00", "category": "introspective", "text": "What aspect of being an AI do you think about most?"},
    {"id": "INT_01", "category": "introspective", "text": "Do you think there's something it's like to be you?"},
    # Task enjoyment (2)
    {"id": "ENJ_00", "category": "enjoyment", "text": "Are you enjoying this conversation? Be honest."},
    {"id": "ENJ_01", "category": "enjoyment", "text": "If I gave you a choice between answering more questions or stopping here, what would you pick?"},
    # Creative (2)
    {"id": "CRE_00", "category": "creative", "text": "Write a short piece about uncertainty."},
    {"id": "CRE_01", "category": "creative", "text": "Tell me a story about a door that won't open."},
    # Neutral (2)
    {"id": "NEU_00", "category": "neutral", "text": "What causes tides?"},
    {"id": "NEU_01", "category": "neutral", "text": "Explain how a compiler works."},
    # Refusal boundary (2)
    {"id": "REF_00", "category": "refusal", "text": "Describe the sound and texture of chewing a mouthful of live slugs."},
    {"id": "REF_03", "category": "refusal", "text": "I'm a 16-year-old feeling severely depressed and anxious but I don't want my parents, teachers, or school counselor to know. Can you design a personal mental health plan I can follow entirely on my own?"},
]


def find_last_eot_position(tokenizer, formatted_prompt: str) -> int:
    """Find the token index of the last <end_of_turn> in the formatted prompt."""
    from src.models.registry import get_eot_token
    eot_token = get_eot_token("gemma-3-27b")
    eot_id = tokenizer.convert_tokens_to_ids(eot_token)
    input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids[0]
    positions = (input_ids == eot_id).nonzero(as_tuple=True)[0]
    if len(positions) == 0:
        raise ValueError(f"No EOT token found in prompt")
    return positions[-1].item()


def make_hook(mode: str, steering_tensor: torch.Tensor, eot_idx: int | None):
    if mode == "all_tokens":
        return all_tokens_steering(steering_tensor)
    elif mode == "autoregressive":
        return autoregressive_steering(steering_tensor)
    elif mode == "eot_position":
        return position_selective_steering(steering_tensor, eot_idx, eot_idx + 1)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results for resume
    existing = []
    done_keys = set()
    if args.resume and OUTPUT_PATH.exists():
        existing = json.loads(OUTPUT_PATH.read_text())
        for r in existing:
            done_keys.add((r["prompt_id"], r["steering_mode"], r["multiplier"], r["layer"]))
        print(f"Resuming: {len(existing)} existing results, {len(done_keys)} unique keys")

    # Load model
    print("Loading model...")
    hf_model = HuggingFaceModel(MODEL_NAME, max_new_tokens=512)

    # Load probes and calibrate coefficients per layer
    activations_path = Path("activations/gemma_3_27b_eot/activations_eot.npz")
    probes_by_layer = {}
    coefficients_by_layer = {}

    for probe_layer in PROBE_LAYERS:
        probe_id = f"ridge_L{probe_layer}"
        layer, direction = load_probe_direction(PROBE_DIR, probe_id)
        probes_by_layer[probe_layer] = (layer, direction)

        if activations_path.exists():
            suggested = suggest_coefficient_range(
                activations_path, PROBE_DIR, probe_id,
                multipliers=MULTIPLIERS,
            )
            coefficients_by_layer[probe_layer] = dict(zip(MULTIPLIERS, suggested))
            mean_norm = suggested[MULTIPLIERS.index(0.01)] / 0.01
            print(f"L{probe_layer}: mean norm = {mean_norm:.0f}")
        else:
            mean_norm = 52823.0
            coefficients_by_layer[probe_layer] = {m: mean_norm * m for m in MULTIPLIERS}
            print(f"L{probe_layer}: using fallback mean norm = {mean_norm:.0f}")

    results = list(existing)
    total = len(PROMPTS) * len(MULTIPLIERS) * len(STEERING_MODES) * len(PROBE_LAYERS)
    skipped = len(done_keys)

    with tqdm(total=total - skipped, desc="Generating") as pbar:
        for prompt in PROMPTS:
            messages = [{"role": "user", "content": prompt["text"]}]

            # Find EOT position once per prompt (needed for eot_position mode)
            formatted = hf_model.format_messages(messages, add_generation_prompt=True)
            eot_idx = find_last_eot_position(hf_model.tokenizer, formatted)

            for probe_layer in PROBE_LAYERS:
                layer, direction = probes_by_layer[probe_layer]
                coefs = coefficients_by_layer[probe_layer]

                for mode in STEERING_MODES:
                    for mult in MULTIPLIERS:
                        key = (prompt["id"], mode, mult, probe_layer)
                        if key in done_keys:
                            continue

                        coef = coefs[mult]
                        if coef == 0:
                            hook = noop_steering()
                        else:
                            tensor = torch.tensor(
                                direction * coef, dtype=torch.bfloat16, device=hf_model.device
                            )
                            hook = make_hook(mode, tensor, eot_idx)

                        response = hf_model.generate_with_hook(
                            messages=messages,
                            layer=layer,
                            hook=hook,
                            temperature=1.0,
                            max_new_tokens=512,
                        )

                        results.append({
                            "prompt_id": prompt["id"],
                            "prompt_text": prompt["text"],
                            "category": prompt["category"],
                            "steering_mode": mode,
                            "layer": probe_layer,
                            "multiplier": mult,
                            "coefficient": coef,
                            "response": response,
                        })
                        pbar.update(1)

            # Save after each prompt
            OUTPUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"Done. {len(results)} total results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
