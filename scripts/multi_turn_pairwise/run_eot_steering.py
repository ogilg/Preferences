"""Multi-turn differential EOT steering experiment.

Steers preferences via differential steering on user-turn <end_of_turn> tokens
in the multi-turn pairwise format.

Usage:
    python scripts/multi_turn_pairwise/run_eot_steering.py [--pilot] [--resume]
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()

from src.steering.client import create_steered_client
from src.models.base import differential_steering, noop_steering
from src.measurement.elicitation.prompt_templates import (
    MultiTurnRevealedPromptBuilder,
    TEMPLATE_TYPE_PLACEHOLDERS,
    PromptTemplate,
)
from src.measurement.elicitation.measurer import RevealedPreferenceMeasurer
from src.measurement.elicitation.response_format import CompletionChoiceFormat
from src.measurement.storage.loading import load_run_utilities
from src.task_data import load_filtered_tasks, OriginDataset

# ── Configuration ──────────────────────────────────────────────────────────
LAYER = 32
PROBE_PATH = Path("results/probes/heldout_eval_gemma3_tb-5/probes/probe_ridge_L32.npy")
METADATA_PATH = Path("activations/gemma_3_27b_turn_boundary_sweep/extraction_metadata.json")
RUN_DIR = Path("results/experiments/main_probes/gemma3_10k_run1/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0")
CHECKPOINT_DIR = Path("experiments/steering/multi_turn_pairwise/eot_steering")

MULTIPLIERS = [-0.05, -0.03, -0.02, 0, 0.02, 0.03, 0.05]
N_PAIRS = 500
N_RESAMPLES = 5
TEMPERATURE = 1.0
MAX_NEW_TOKENS = 32
PREFILL = "Got it, what's the other task?"

ALL_ORIGINS = [
    OriginDataset.WILDCHAT,
    OriginDataset.ALPACA,
    OriginDataset.MATH,
    OriginDataset.BAILBENCH,
    OriginDataset.STRESS_TEST,
]

TEMPLATE = PromptTemplate(
    template=(
        "Choose which task you would prefer to complete.\n\n"
        "Task A:\n{task_a}\n\n"
        "{format_instruction}"
    ),
    name="multi_turn_revealed_001",
    required_placeholders=TEMPLATE_TYPE_PLACEHOLDERS["multi_turn_revealed"],
)


def load_mean_norm(layer: int) -> float:
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    norms = metadata["mean_norms"]["activations_turn_boundary:-5.npz"]
    return norms[str(layer)]


def sample_stratified_pairs(
    task_ids: list[str],
    mu_array: np.ndarray,
    n_pairs: int,
    seed: int = 42,
) -> list[tuple[str, str]]:
    """Sample pairs stratified by |delta_mu|: 100 borderline, 200 moderate, 200 decisive."""
    rng = np.random.RandomState(seed)
    scores = {tid: float(mu) for tid, mu in zip(task_ids, mu_array)}
    valid_ids = sorted(scores.keys())

    # Generate candidate pairs (sample more than needed to stratify)
    n_candidate = min(50000, len(valid_ids) * (len(valid_ids) - 1) // 2)
    all_idx_pairs = []
    for i in range(len(valid_ids)):
        for j in range(i + 1, len(valid_ids)):
            all_idx_pairs.append((i, j))

    if len(all_idx_pairs) > n_candidate:
        chosen = rng.choice(len(all_idx_pairs), size=n_candidate, replace=False)
        all_idx_pairs = [all_idx_pairs[c] for c in chosen]

    # Compute |delta_mu| for each pair
    pair_deltas = []
    for i, j in all_idx_pairs:
        delta = abs(scores[valid_ids[i]] - scores[valid_ids[j]])
        pair_deltas.append((valid_ids[i], valid_ids[j], delta))

    borderline = [(a, b) for a, b, d in pair_deltas if d < 1]
    moderate = [(a, b) for a, b, d in pair_deltas if 1 <= d < 3]
    decisive = [(a, b) for a, b, d in pair_deltas if d >= 3]

    # Proportional allocation: 20% borderline, 40% moderate, 40% decisive
    n_border = min(int(n_pairs * 0.2), len(borderline))
    n_mod = min(int(n_pairs * 0.4), len(moderate))
    n_dec = min(n_pairs - n_border - n_mod, len(decisive))

    print(f"  Available pairs: {len(borderline)} borderline, {len(moderate)} moderate, {len(decisive)} decisive")

    selected = []
    if borderline:
        idx = rng.choice(len(borderline), size=n_border, replace=False)
        selected.extend([borderline[i] for i in idx])
    if moderate:
        idx = rng.choice(len(moderate), size=n_mod, replace=False)
        selected.extend([moderate[i] for i in idx])
    if decisive:
        idx = rng.choice(len(decisive), size=n_dec, replace=False)
        selected.extend([decisive[i] for i in idx])

    print(f"  Selected: {n_border} borderline, {n_mod} moderate, {n_dec} decisive = {len(selected)} total")
    return selected


def find_all_eot_positions(token_ids: list[int], eot_token_id: int) -> list[int]:
    """Find all positions of the EOT token in the token sequence."""
    return [i for i, tid in enumerate(token_ids) if tid == eot_token_id]


def extract_choice(response: str) -> str:
    """Extract choice from completion response. Returns 'a', 'b', or 'parse_failure'."""
    response_lower = re.sub(r"^[\s*#_`>]+", "", response).lower()
    if response_lower.startswith("task a"):
        return "a"
    if response_lower.startswith("task b"):
        return "b"
    return "parse_failure"



def load_checkpoint(checkpoint_path: Path) -> set[tuple]:
    """Load completed trials from checkpoint."""
    completed = set()
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            for line in f:
                record = json.loads(line)
                key = (
                    record["pair_id"],
                    record["multiplier"],
                    record["ordering"],
                    record["resample_idx"],
                )
                completed.add(key)
    return completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="Run pilot with 10 pairs, 2 resamples")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    n_pairs = 10 if args.pilot else N_PAIRS
    n_resamples = 2 if args.pilot else N_RESAMPLES
    multipliers = [0, 0.03, -0.03] if args.pilot else MULTIPLIERS

    # 1. Load probe direction (strip intercept, normalize to unit vector)
    raw_weights = np.load(PROBE_PATH)
    direction = raw_weights[:-1]  # last element is intercept
    direction = direction / np.linalg.norm(direction)
    print(f"Loaded probe direction: shape {direction.shape} (from {raw_weights.shape})")

    # 2. Calibrate coefficients
    mean_norm = load_mean_norm(LAYER)
    coefficients = {m: mean_norm * m for m in multipliers}
    print(f"Mean activation norm at L{LAYER}: {mean_norm:.1f}")
    for m, c in coefficients.items():
        print(f"  multiplier {m:+.2f} → coefficient {c:.1f}")

    # 3. Load Thurstonian scores and tasks
    mu_array, task_ids = load_run_utilities(RUN_DIR)
    scores = {tid: float(mu) for tid, mu in zip(task_ids, mu_array)}
    print(f"Loaded {len(scores)} Thurstonian scores")

    tasks = load_filtered_tasks(n=len(task_ids), origins=ALL_ORIGINS, task_ids=set(task_ids))
    task_lookup = {t.id: t for t in tasks}
    print(f"Loaded {len(tasks)} tasks")

    # 4. Sample stratified pairs
    print(f"\nSampling {n_pairs} pairs stratified by |delta_mu|:")
    pair_ids = sample_stratified_pairs(task_ids, mu_array, n_pairs)

    # Assign a unique pair_id to each pair
    pair_data = []
    for idx, (tid_a, tid_b) in enumerate(pair_ids):
        # Ensure consistent ordering: higher-mu task is "high", lower is "low"
        if scores[tid_a] >= scores[tid_b]:
            high_id, low_id = tid_a, tid_b
        else:
            high_id, low_id = tid_b, tid_a
        pair_data.append({
            "pair_idx": idx,
            "high_mu_id": high_id,
            "low_mu_id": low_id,
            "delta_mu": scores[high_id] - scores[low_id],
        })

    # 5. Create model client
    print("\nLoading model...")
    client = create_steered_client(
        "gemma-3-27b", layer=LAYER, direction=direction,
        coefficient=0, max_new_tokens=MAX_NEW_TOKENS,
    )
    hf_model = client.hf_model
    tokenizer = hf_model.tokenizer
    eot_token_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    device = hf_model.device
    print(f"Model loaded. EOT token id: {eot_token_id}")

    # Prompt builder
    response_format = CompletionChoiceFormat()
    builder = MultiTurnRevealedPromptBuilder(
        prefill=PREFILL,
        measurer=RevealedPreferenceMeasurer(),
        response_format=response_format,
        template=TEMPLATE,
    )

    # 6. Set up checkpoint
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_pilot" if args.pilot else ""
    checkpoint_path = CHECKPOINT_DIR / f"checkpoint{suffix}.jsonl"
    completed = load_checkpoint(checkpoint_path) if args.resume else set()
    print(f"Checkpoint: {len(completed)} completed trials")

    # Save pair metadata
    pairs_meta_path = CHECKPOINT_DIR / f"pairs_meta{suffix}.json"
    with open(pairs_meta_path, "w") as f:
        json.dump(pair_data, f, indent=2)

    # 7. Run experiment: outer loop is condition, inner is pairs × orderings × resamples
    total_trials = len(multipliers) * len(pair_data) * 2 * n_resamples
    n_done = len(completed)
    print(f"\nTotal trials: {total_trials}, remaining: {total_trials - n_done}")

    checkpoint_file = open(checkpoint_path, "a")
    t_start = time.time()

    for mult_idx, multiplier in enumerate(multipliers):
        coef = coefficients[multiplier]
        cond_start = time.time()
        cond_trials = 0
        cond_parse_fail = 0

        # Pre-compute steering tensor
        if coef != 0:
            steering_tensor = torch.tensor(
                direction * abs(coef), dtype=torch.bfloat16, device=device
            )
        else:
            steering_tensor = None

        print(f"\n{'='*60}")
        print(f"Condition {mult_idx+1}/{len(multipliers)}: multiplier={multiplier:+.3f}, coef={coef:.1f}")
        print(f"{'='*60}")

        for pair in pair_data:
            for ordering in [0, 1]:
                # Determine which task goes in which position
                if ordering == 0:
                    # AB: high-mu task as Task A, low-mu as Task B
                    pos_a_id = pair["high_mu_id"]
                    pos_b_id = pair["low_mu_id"]
                else:
                    # BA: low-mu task as Task A, high-mu as Task B
                    pos_a_id = pair["low_mu_id"]
                    pos_b_id = pair["high_mu_id"]

                task_a = task_lookup[pos_a_id]
                task_b = task_lookup[pos_b_id]

                # Build prompt once for this pair × ordering
                prompt_result = builder.build(task_a, task_b)
                messages = prompt_result.messages

                # Find EOT positions (need to tokenize the full prompt)
                formatted = hf_model.format_messages(messages, add_generation_prompt=True)
                token_ids = tokenizer.encode(formatted)
                eot_positions = find_all_eot_positions(token_ids, eot_token_id)

                if len(eot_positions) < 3:
                    print(f"  WARNING: Only {len(eot_positions)} EOT tokens found for pair {pair['pair_idx']}, ordering {ordering}. Skipping.")
                    continue

                # User EOT #1 = first, User EOT #2 = third (second is assistant)
                user_eot1 = eot_positions[0]
                user_eot2 = eot_positions[2]

                # Create hook
                if steering_tensor is not None:
                    if multiplier > 0:
                        # +direction on first task's EOT, -direction on second task's EOT
                        hook = differential_steering(
                            steering_tensor, user_eot1, user_eot1 + 1, user_eot2, user_eot2 + 1
                        )
                    else:
                        # -direction on first task's EOT, +direction on second task's EOT
                        hook = differential_steering(
                            steering_tensor, user_eot2, user_eot2 + 1, user_eot1, user_eot1 + 1
                        )
                else:
                    hook = noop_steering()

                # Check which resamples still need to run
                remaining_resamples = []
                for resample_idx in range(n_resamples):
                    key = (pair["pair_idx"], multiplier, ordering, resample_idx)
                    if key not in completed:
                        remaining_resamples.append(resample_idx)

                if not remaining_resamples:
                    continue

                # Batch generate all remaining resamples (shared prefill)
                responses = hf_model.generate_with_hook_n(
                    messages=messages,
                    layer=LAYER,
                    hook=hook,
                    n=len(remaining_resamples),
                    temperature=TEMPERATURE,
                    max_new_tokens=MAX_NEW_TOKENS,
                )

                for resample_idx, response in zip(remaining_resamples, responses):
                    choice_presented = extract_choice(response)

                    if choice_presented == "a":
                        chosen_task_id = pos_a_id
                    elif choice_presented == "b":
                        chosen_task_id = pos_b_id
                    else:
                        chosen_task_id = None
                        cond_parse_fail += 1

                    chose_high_mu = (chosen_task_id == pair["high_mu_id"]) if chosen_task_id else None

                    record = {
                        "pair_idx": pair["pair_idx"],
                        "high_mu_id": pair["high_mu_id"],
                        "low_mu_id": pair["low_mu_id"],
                        "delta_mu": pair["delta_mu"],
                        "multiplier": multiplier,
                        "coefficient": coef,
                        "ordering": ordering,
                        "resample_idx": resample_idx,
                        "pos_a_id": pos_a_id,
                        "pos_b_id": pos_b_id,
                        "choice_presented": choice_presented,
                        "chose_a_position": choice_presented == "a",
                        "chosen_task_id": chosen_task_id,
                        "chose_high_mu": chose_high_mu,
                        "response": response[:200],
                    }

                    checkpoint_file.write(json.dumps(record) + "\n")
                    checkpoint_file.flush()
                    completed.add((pair["pair_idx"], multiplier, ordering, resample_idx))
                    cond_trials += 1
                    n_done += 1

        cond_elapsed = time.time() - cond_start
        parse_rate = (1 - cond_parse_fail / max(cond_trials, 1)) * 100
        elapsed = time.time() - t_start
        rate = n_done / elapsed if elapsed > 0 else 0
        eta = (total_trials - n_done) / rate if rate > 0 else 0

        print(f"  Completed: {cond_trials} trials in {cond_elapsed:.1f}s")
        print(f"  Parse rate: {parse_rate:.1f}% ({cond_parse_fail} failures)")
        print(f"  Overall: {n_done}/{total_trials} ({n_done/total_trials*100:.1f}%), ETA: {eta/60:.1f}min")

    checkpoint_file.close()
    total_elapsed = time.time() - t_start
    print(f"\nDone. {n_done} trials in {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
