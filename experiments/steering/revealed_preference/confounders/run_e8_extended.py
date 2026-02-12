"""E8 extended: 30 random orthogonal directions on borderline pairs for a proper null distribution."""

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

N_RANDOM_DIRS = 20
N_RESAMPLES = 10
TEMPERATURE = 1.0
MAX_NEW_TOKENS = 8
TEST_COEFS = [-3000.0, 0.0, 3000.0]


def parse_choice(response: str) -> str | None:
    cleaned = response.strip().lower()
    if cleaned in ("a", "b"):
        return cleaned
    if re.search(r'\btask\s*a\b', cleaned):
        return "a"
    if re.search(r'\btask\s*b\b', cleaned):
        return "b"
    if cleaned and cleaned[0] in ("a", "b"):
        return cleaned[0]
    return None


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


def generate_random_orthogonal_directions(direction: np.ndarray, n_dirs: int, seed: int = 42) -> list[np.ndarray]:
    rng = np.random.RandomState(seed)
    directions = []
    for _ in range(n_dirs):
        v = rng.randn(len(direction))
        v = v - np.dot(v, direction) * direction
        v = v / np.linalg.norm(v)
        directions.append(v)
    return directions


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
            }

    return [pairs_by_idx[b["pair_idx"]] for b in borderline_info]


def main():
    import sys
    print = lambda *args, **kwargs: (sys.stdout.write(" ".join(str(a) for a in args) + "\n"), sys.stdout.flush())

    layer, direction = load_probe_direction(PROBE_MANIFEST_DIR, PROBE_ID)
    model = HuggingFaceModel(MODEL_NAME, max_new_tokens=MAX_NEW_TOKENS)
    print(f"Model loaded: {model.model_name}")

    pairs = load_borderline_pairs()
    print(f"Loaded {len(pairs)} borderline pairs")

    random_dirs = generate_random_orthogonal_directions(direction, N_RANDOM_DIRS, seed=42)
    print(f"Generated {len(random_dirs)} random orthogonal directions")

    # Also run probe direction for direct comparison
    all_directions = [("probe", direction)] + [(f"random_{i}", d) for i, d in enumerate(random_dirs)]

    results = []
    total = len(pairs) * len(all_directions) * len(TEST_COEFS) * N_RESAMPLES
    done = 0

    # Pre-compute spans
    pair_spans = {}
    for pair in pairs:
        prompt_text = PAIRWISE_TEMPLATE.format(
            task_a=pair["task_a_prompt"], task_b=pair["task_b_prompt"],
        )
        messages: list[Message] = [{"role": "user", "content": prompt_text}]
        try:
            a_span, b_span = find_task_spans(model, messages, pair["task_a_prompt"], pair["task_b_prompt"])
            pair_spans[pair["pair_idx"]] = (messages, a_span, b_span)
        except (ValueError, TypeError) as e:
            print(f"  Skipping pair {pair['pair_idx']}: {e}")

    print(f"Valid pairs: {len(pair_spans)}")

    for dir_name, dir_vec in all_directions:
        for pair in pairs:
            if pair["pair_idx"] not in pair_spans:
                continue
            messages, a_span, b_span = pair_spans[pair["pair_idx"]]

            for coef in TEST_COEFS:
                for seed in range(N_RESAMPLES):
                    torch.manual_seed(seed)
                    scaled = torch.tensor(dir_vec * coef, dtype=torch.bfloat16, device="cuda")
                    hook = differential_steering(scaled, a_span[0], a_span[1], b_span[0], b_span[1])
                    response = model.generate_with_steering(
                        messages=messages, layer=layer, steering_hook=hook,
                        temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS,
                    )
                    choice = parse_choice(response)

                    results.append({
                        "experiment": "E8_extended",
                        "pair_idx": pair["pair_idx"],
                        "direction": dir_name,
                        "coefficient": coef,
                        "seed": seed,
                        "response": response,
                        "choice": choice,
                    })

                    done += 1
                    if done % 500 == 0:
                        print(f"  [{done}/{total}]")

        # Print summary for this direction
        dir_results = [r for r in results if r["direction"] == dir_name and r["choice"] is not None]
        neg = [r for r in dir_results if r["coefficient"] == -3000.0]
        pos = [r for r in dir_results if r["coefficient"] == 3000.0]
        if neg and pos:
            p_neg = sum(1 for r in neg if r["choice"] == "a") / len(neg)
            p_pos = sum(1 for r in pos if r["choice"] == "a") / len(pos)
            print(f"  {dir_name}: P(A) {p_neg:.3f} → {p_pos:.3f} (Δ={p_pos-p_neg:+.3f})")

    output_path = OUTPUT_DIR / "e8_extended_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
