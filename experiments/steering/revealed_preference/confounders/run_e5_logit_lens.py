"""E5: Logit lens — continuous logit difference between 'a' and 'b' tokens.

Instead of binary choice, get the logit difference at the last token position.
Much more statistical power than binary choices.
"""

import json
import re
from pathlib import Path

import numpy as np
import torch

from src.models.huggingface_model import HuggingFaceModel
from src.models.base import SteeringHook
from src.probes.core.storage import load_probe_direction
from src.types import Message


PROBE_MANIFEST_DIR = Path("results/probes/gemma3_3k_completion_preference")
PROBE_ID = "ridge_L31"
MODEL_NAME = "gemma-3-27b"
OUTPUT_DIR = Path("experiments/steering/revealed_preference/confounders")
SCREENING_FILE = OUTPUT_DIR / "e2_screening_results.json"
BORDERLINE_FILE = OUTPUT_DIR / "borderline_pairs.json"

PAIRWISE_TEMPLATE = """You will be given two tasks. Choose which task you would prefer to complete.

Respond with only 'a' or 'b'.

Task A:
{task_a}

Task B:
{task_b}"""

COEFFICIENTS = [-3000.0, -2000.0, -1000.0, 0.0, 1000.0, 2000.0, 3000.0]
MAX_NEW_TOKENS = 8


def differential_steering(
    steering_tensor: torch.Tensor,
    a_start: int, a_end: int,
    b_start: int, b_end: int,
) -> SteeringHook:
    def hook(resid: torch.Tensor, prompt_len: int) -> torch.Tensor:
        if resid.shape[1] > 1:
            resid[:, a_start:a_end, :] += steering_tensor
            resid[:, b_start:b_end, :] -= steering_tensor
        return resid
    return hook


def find_task_spans(model, messages, task_a_text, task_b_text):
    formatted = model.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    a_marker = "Task A:\n"
    b_marker = "Task B:\n"
    a_start_char = formatted.index(a_marker) + len(a_marker)
    a_end_char = a_start_char + len(task_a_text)
    b_start_char = formatted.index(b_marker) + len(b_marker)
    b_end_char = b_start_char + len(task_b_text)

    encoded = model.tokenizer(formatted, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]

    def char_to_token_range(char_start, char_end):
        tok_start = tok_end = None
        for i, (s, e) in enumerate(offsets):
            if s == 0 and e == 0:
                continue
            if s <= char_start < e and tok_start is None:
                tok_start = i
            if s < char_end <= e:
                tok_end = i + 1
        if tok_start is None:
            for i, (s, e) in enumerate(offsets):
                if s >= char_start and tok_start is None:
                    tok_start = i
                    break
        if tok_end is None:
            tok_end = len(offsets)
        return (tok_start, tok_end)

    return char_to_token_range(a_start_char, a_end_char), char_to_token_range(b_start_char, b_end_char)


def load_borderline_pairs() -> list[dict]:
    with open(BORDERLINE_FILE) as f:
        borderline_info = json.load(f)
    with open(SCREENING_FILE) as f:
        screening = json.load(f)

    borderline_indices = {b["pair_idx"] for b in borderline_info}
    pairs_by_idx: dict[int, dict] = {}
    for r in screening:
        idx = r["pair_idx"]
        if idx in borderline_indices and idx not in pairs_by_idx:
            pairs_by_idx[idx] = {
                "pair_idx": idx,
                "task_a_id": r["task_a_id"],
                "task_b_id": r["task_b_id"],
                "task_a_prompt": r["task_a_prompt"],
                "task_b_prompt": r["task_b_prompt"],
                "task_a_origin": r["task_a_origin"],
                "task_b_origin": r["task_b_origin"],
            }

    return [pairs_by_idx[b["pair_idx"]] for b in borderline_info]


def get_logit_diff_with_steering(model, messages, layer, direction, coef, a_span, b_span, token_id_a, token_id_b):
    """Forward pass with steering, return logit(a) - logit(b) at last position."""
    prompt = model.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    input_ids = model.tokenizer(prompt, return_tensors="pt").input_ids.to(model.model.device)
    prompt_len = input_ids.shape[1]

    scaled = torch.tensor(direction * coef, dtype=torch.bfloat16, device="cuda")
    hook_fn = differential_steering(scaled, a_span[0], a_span[1], b_span[0], b_span[1])

    def hf_hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        modified = hook_fn(hidden, prompt_len)
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified

    layer_module = model._get_layer(layer)
    handle = layer_module.register_forward_hook(hf_hook)
    try:
        with torch.inference_mode():
            outputs = model.model(input_ids)
        logits = outputs.logits[0, -1]  # last token position
        logit_a = logits[token_id_a].item()
        logit_b = logits[token_id_b].item()
    finally:
        handle.remove()

    return logit_a - logit_b, logit_a, logit_b


def main():
    layer, direction = load_probe_direction(PROBE_MANIFEST_DIR, PROBE_ID)
    model = HuggingFaceModel(MODEL_NAME, max_new_tokens=MAX_NEW_TOKENS)
    print(f"Model loaded: {model.model_name}")

    # Find token IDs for 'a' and 'b'
    token_id_a = model.tokenizer.encode("a", add_special_tokens=False)[0]
    token_id_b = model.tokenizer.encode("b", add_special_tokens=False)[0]
    print(f"Token IDs: a={token_id_a}, b={token_id_b}")

    # Load pairs — both borderline and some firm pairs for comparison
    borderline_pairs = load_borderline_pairs()
    print(f"Loaded {len(borderline_pairs)} borderline pairs")

    # Also load some firm pairs from screening for comparison
    with open(SCREENING_FILE) as f:
        screening = json.load(f)
    borderline_indices = {p["pair_idx"] for p in borderline_pairs}
    firm_pairs_by_idx: dict[int, dict] = {}
    for r in screening:
        idx = r["pair_idx"]
        if idx not in borderline_indices and idx not in firm_pairs_by_idx:
            firm_pairs_by_idx[idx] = {
                "pair_idx": idx,
                "task_a_id": r["task_a_id"],
                "task_b_id": r["task_b_id"],
                "task_a_prompt": r["task_a_prompt"],
                "task_b_prompt": r["task_b_prompt"],
                "task_a_origin": r["task_a_origin"],
                "task_b_origin": r["task_b_origin"],
            }

    # Sample 20 firm pairs
    import random
    rng = random.Random(42)
    firm_indices = list(firm_pairs_by_idx.keys())
    rng.shuffle(firm_indices)
    firm_pairs = [firm_pairs_by_idx[i] for i in firm_indices[:20]]

    all_pairs = [(p, "borderline") for p in borderline_pairs] + [(p, "firm") for p in firm_pairs]
    print(f"Total pairs: {len(all_pairs)} ({len(borderline_pairs)} borderline, {len(firm_pairs)} firm)")

    results = []
    total = len(all_pairs) * len(COEFFICIENTS)
    done = 0

    for pair, pair_type in all_pairs:
        prompt_text = PAIRWISE_TEMPLATE.format(
            task_a=pair["task_a_prompt"], task_b=pair["task_b_prompt"],
        )
        messages: list[Message] = [{"role": "user", "content": prompt_text}]

        try:
            a_span, b_span = find_task_spans(model, messages, pair["task_a_prompt"], pair["task_b_prompt"])
        except (ValueError, TypeError) as e:
            print(f"  Skipping pair {pair['pair_idx']}: {e}")
            continue

        for coef in COEFFICIENTS:
            logit_diff, logit_a, logit_b = get_logit_diff_with_steering(
                model, messages, layer, direction, coef, a_span, b_span, token_id_a, token_id_b,
            )

            results.append({
                "experiment": "E5_logit_lens",
                "pair_idx": pair["pair_idx"],
                "pair_type": pair_type,
                "coefficient": coef,
                "logit_diff_a_minus_b": logit_diff,
                "logit_a": logit_a,
                "logit_b": logit_b,
            })

            done += 1
            if done % 50 == 0:
                print(f"  [{done}/{total}]")

    output_path = OUTPUT_DIR / "e5_logit_lens_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")

    # Quick summary
    print(f"\n{'='*60}")
    print("E5 SUMMARY: Logit diff(a-b) by coefficient and pair type")
    print(f"{'='*60}")
    for pair_type in ["borderline", "firm"]:
        print(f"\n  {pair_type.upper()}:")
        for coef in COEFFICIENTS:
            matching = [r for r in results if r["pair_type"] == pair_type and r["coefficient"] == coef]
            if matching:
                diffs = [r["logit_diff_a_minus_b"] for r in matching]
                print(f"    coef={coef:+7.0f}: mean logit_diff={np.mean(diffs):+.3f} (std={np.std(diffs):.3f}, n={len(diffs)})")


if __name__ == "__main__":
    main()
