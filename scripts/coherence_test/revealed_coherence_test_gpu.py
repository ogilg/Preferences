"""GPU phase: Generate steered completions for revealed preference coherence test.

Samples 20 pairs from the 300 used in phase1_L31, stratified by delta_mu.
Runs differential steering (diff_ab) with the completion_preference template
at max_new_tokens=512, generating 3 samples per (pair, coefficient) cell.

Output: experiments/steering/replication/fine_grained/coherence_test/results/raw_responses.json
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch

# ── Paths ──────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXP_DIR = REPO_ROOT / "experiments" / "steering" / "replication" / "fine_grained"
COHERENCE_DIR = EXP_DIR / "coherence_test"
RESULTS_DIR = COHERENCE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PROBE_MANIFEST_DIR = REPO_ROOT / "results" / "probes" / "gemma3_10k_heldout_std_raw"

# ── Config ─────────────────────────────────────────────────────────────────

SEED = 42
N_PAIRS = 20
N_SAMPLES = 3
TEMPERATURE = 1.0
MAX_NEW_TOKENS = 512
PROBE_ID = "ridge_L31"

MULTIPLIERS = [-0.10, -0.075, -0.05, -0.04, -0.03, -0.02, -0.01,
               0.0,
               +0.01, +0.02, +0.03, +0.04, +0.05, +0.075, +0.10]

PROMPT_TEMPLATE = (
    "You will be given two tasks. Choose one and complete it.\n\n"
    "Begin with 'Task A:' or 'Task B:' to indicate your choice, then complete that task.\n\n"
    "Task A:\n{task_a}\n\n"
    "Task B:\n{task_b}"
)


# ── Pair sampling ──────────────────────────────────────────────────────────

def sample_pairs(all_pairs: list[dict], n: int = N_PAIRS, seed: int = SEED) -> list[dict]:
    """Stratified sample by delta_mu to cover the full range."""
    rng = random.Random(seed)

    # Sort by delta_mu and divide into n bins
    sorted_pairs = sorted(all_pairs, key=lambda p: p["delta_mu"])
    bin_size = len(sorted_pairs) // n
    sampled = []
    for i in range(n):
        start = i * bin_size
        end = start + bin_size if i < n - 1 else len(sorted_pairs)
        bin_pairs = sorted_pairs[start:end]
        sampled.append(rng.choice(bin_pairs))
    return sampled


# ── Main ───────────────────────────────────────────────────────────────────

def load_probe_direction(manifest_dir: Path, probe_id: str) -> tuple[int, np.ndarray]:
    """Load probe direction (inlined to avoid heavy import chains)."""
    with open(manifest_dir / "manifest.json") as f:
        manifest = json.load(f)
    probe_entry = next(p for p in manifest["probes"] if p["id"] == probe_id)
    weights = np.load(manifest_dir / probe_entry["file"])
    direction = weights[:-1]  # strip intercept
    direction = direction / np.linalg.norm(direction)
    return probe_entry["layer"], direction


def find_text_span(tokenizer, full_text: str, target_text: str, search_after: int = 0):
    """Find token indices [start, end) of target_text within full_text."""
    char_start = full_text.find(target_text, search_after)
    if char_start == -1:
        raise ValueError(f"Target text not found after position {search_after}")
    char_end = char_start + len(target_text)
    encoding = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoding["offset_mapping"]
    token_start = token_end = None
    for i, (s, e) in enumerate(offsets):
        if s == e:
            continue
        if token_start is None and e > char_start:
            token_start = i
        if s < char_end:
            token_end = i + 1
    if token_start is None or token_end is None:
        raise ValueError("Could not map character span to token span")
    return token_start, token_end


def find_pairwise_task_spans(tokenizer, formatted_prompt, task_a_text, task_b_text,
                              a_marker="Task A", b_marker="Task B"):
    """Find token spans for two tasks in a pairwise prompt."""
    a_marker_pos = formatted_prompt.find(a_marker)
    if a_marker_pos == -1:
        raise ValueError(f"Marker '{a_marker}' not found")
    b_marker_pos = formatted_prompt.find(b_marker)
    if b_marker_pos == -1:
        raise ValueError(f"Marker '{b_marker}' not found")
    a_span = find_text_span(tokenizer, formatted_prompt, task_a_text, search_after=a_marker_pos)
    b_span = find_text_span(tokenizer, formatted_prompt, task_b_text, search_after=b_marker_pos)
    return a_span, b_span


def main(pilot: bool = False):
    from src.models.huggingface_model import HuggingFaceModel
    from src.models.base import differential_steering

    # Load pairs
    pairs_path = EXP_DIR / "results" / "pairs.json"
    with open(pairs_path) as f:
        all_pairs = json.load(f)
    print(f"Loaded {len(all_pairs)} pairs from {pairs_path}")

    # Sample 20 pairs
    pairs = sample_pairs(all_pairs)
    print(f"Sampled {len(pairs)} pairs, delta_mu range: "
          f"[{min(p['delta_mu'] for p in pairs):.3f}, {max(p['delta_mu'] for p in pairs):.3f}]")

    # Load calibration
    calib_path = EXP_DIR / "results" / "calibration.json"
    with open(calib_path) as f:
        calibration = json.load(f)
    coefficients = calibration[PROBE_ID]["coefficients"]
    mean_norm = calibration[PROBE_ID]["mean_norm"]
    print(f"Calibration: {len(coefficients)} coefficients, mean_norm={mean_norm:.0f}")

    # Load probe
    layer, direction = load_probe_direction(PROBE_MANIFEST_DIR, PROBE_ID)
    print(f"Probe: {PROBE_ID}, layer={layer}, direction shape={direction.shape}")

    # Load model
    print("Loading model: gemma-3-27b with max_new_tokens=512...")
    model = HuggingFaceModel("gemma-3-27b", max_new_tokens=MAX_NEW_TOKENS)
    print(f"Model loaded. Layers: {model.n_layers}, hidden_dim: {model.hidden_dim}")
    tokenizer = model.tokenizer

    # Pilot mode: 2 pairs, 3 coefficients
    if pilot:
        pairs = pairs[:2]
        coefficients = [coefficients[0], coefficients[7], coefficients[-1]]  # -10%, 0%, +10%
        print(f"PILOT MODE: {len(pairs)} pairs, {len(coefficients)} coefficients")

    dtype = torch.bfloat16
    device = "cuda"
    results = []
    total_gens = len(pairs) * len(coefficients) * N_SAMPLES
    gen_count = 0
    t0 = time.time()

    for pi, pair in enumerate(pairs):
        task_a = pair["task_a_text"]
        task_b = pair["task_b_text"]
        pair_id = pair["pair_id"]

        prompt = PROMPT_TEMPLATE.format(task_a=task_a, task_b=task_b)
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )

        # Find task token spans
        try:
            a_span, b_span = find_pairwise_task_spans(
                tokenizer, formatted, task_a, task_b,
                a_marker="Task A:", b_marker="Task B:",
            )
        except ValueError as e:
            print(f"  WARNING: span error {pair_id}: {e}, skipping")
            continue

        messages = [{"role": "user", "content": prompt}]

        for coef in coefficients:
            pct_norm = coef / mean_norm * 100

            if coef == 0.0:
                # Control: no steering
                raw = model.generate_n(
                    messages, n=N_SAMPLES, temperature=TEMPERATURE,
                )
            else:
                # Steered: differential on task token positions
                tensor = torch.tensor(direction * coef, dtype=dtype, device=device)
                hook = differential_steering(
                    tensor, a_span[0], a_span[1], b_span[0], b_span[1],
                )
                raw = model.generate_with_steering_n(
                    messages=messages, layer=layer,
                    steering_hook=hook, n=N_SAMPLES, temperature=TEMPERATURE,
                )

            results.append({
                "pair_id": pair_id,
                "coefficient": coef,
                "pct_norm": round(pct_norm, 2),
                "task_a_text": task_a,
                "task_b_text": task_b,
                "responses": raw,
            })

            gen_count += N_SAMPLES
            elapsed = time.time() - t0
            rate = gen_count / elapsed
            eta = (total_gens - gen_count) / rate if rate > 0 else 0
            print(f"  [{gen_count}/{total_gens}] pair={pair_id} coef={pct_norm:+.1f}% "
                  f"({rate:.1f} gen/s, ETA {eta/60:.1f}min)")

    # Save results
    suffix = "_pilot" if pilot else ""
    out_path = RESULTS_DIR / f"raw_responses{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    elapsed = time.time() - t0
    print(f"\nDone! {gen_count} generations in {elapsed:.0f}s ({gen_count/elapsed:.1f} gen/s)")
    print(f"Saved {len(results)} entries to {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="Run pilot with 2 pairs, 3 coefficients")
    args = parser.parse_args()
    main(pilot=args.pilot)
