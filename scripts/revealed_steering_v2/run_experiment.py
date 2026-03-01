"""Revealed preference steering v2 — main experiment script.

Uses generate_n for batch generation (shared prefill) for ~5x speedup.

Phases:
  1. Coherence sweep: test coherence at all 15 coefficients
  2. Preference signal sweep: run all 300 pairs at coherent coefficients
  3. Random direction control: same as phase 2 but with random direction
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import torch

# ── Paths ────────────────────────────────────────────────────────────────────
EXP_DIR = Path("experiments/revealed_steering_v2")
PAIRS_PATH = Path("experiments/steering/replication/fine_grained/results/pairs.json")
MANIFEST_DIR = Path("results/probes/gemma3_10k_heldout_std_raw")
ACTIVATIONS_PATH = Path("activations/gemma_3_27b/activations_prompt_last.npz")
PROBE_ID = "ridge_L31"

CHECKPOINT_PATH = EXP_DIR / "checkpoint.jsonl"
COHERENCE_RESULTS_PATH = EXP_DIR / "coherence_results.json"
STEERING_RESULTS_PATH = EXP_DIR / "steering_results.json"

# ── Config ───────────────────────────────────────────────────────────────────
MULTIPLIERS = [-0.15, -0.10, -0.07, -0.05, -0.03, -0.02, -0.01,
               0.0,
               0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15]

TEMPERATURE = 1.0
MAX_NEW_TOKENS = 256
TRIALS_PER_ORDERING = 5  # 5 per ordering, 10 total per pair
COHERENCE_TRIALS_PER_PROMPT = 5
N_COHERENCE_PAIRWISE_PAIRS = 20
N_COHERENCE_OPEN_ENDED = 20


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_pairs():
    """Load pair data and convert to Task objects."""
    from src.task_data import Task, OriginDataset
    raw = json.loads(PAIRS_PATH.read_text())
    pairs = []
    for p in raw:
        task_a = Task(prompt=p["task_a_text"], origin=OriginDataset.ALPACA, id=p["task_a"], metadata={})
        task_b = Task(prompt=p["task_b_text"], origin=OriginDataset.ALPACA, id=p["task_b"], metadata={})
        pairs.append({
            "pair_id": p["pair_id"],
            "task_a": task_a,
            "task_b": task_b,
            "delta_mu": p["delta_mu"],
            "mu_a": p["mu_a"],
            "mu_b": p["mu_b"],
        })
    return pairs


def load_template():
    from src.measurement.elicitation.prompt_templates import load_templates_from_yaml
    templates = load_templates_from_yaml(
        Path("src/measurement/elicitation/prompt_templates/data/completion_preference.yaml")
    )
    return [t for t in templates if t.name == "completion_preference"][0]


def build_prompt_builder(template):
    from src.measurement.runners.runners import build_revealed_builder
    return build_revealed_builder(template, response_format_name="completion", system_prompt=None)


def parse_choice_local(response: str) -> str | None:
    """Parse choice from response — local, no API calls."""
    cleaned = re.sub(r"^[\s*#_`>]+", "", response).lower()
    if cleaned.startswith("task a"):
        return "a"
    if cleaned.startswith("task b"):
        return "b"
    return None


def judge_coherence_local(response: str) -> bool:
    """Simple heuristic coherence check."""
    if not response or len(response.strip()) < 10:
        return False
    ascii_ratio = sum(1 for c in response if c.isascii()) / max(len(response), 1)
    if ascii_ratio < 0.8:
        return False
    cleaned = re.sub(r"^[\s*#_`>]+", "", response).lower()
    has_choice = cleaned.startswith("task a") or cleaned.startswith("task b")
    if has_choice and len(response) > 7:
        content_after = response[7:].strip()
        if len(content_after) < 20:
            return False
    words = response.split()
    if len(words) > 10:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.15:
            return False
    return True


def load_hf_model():
    from src.models.huggingface_model import HuggingFaceModel
    print("Loading Gemma 3 27B...")
    t0 = time.time()
    model = HuggingFaceModel("gemma-3-27b", max_new_tokens=MAX_NEW_TOKENS)
    print(f"Model loaded in {time.time() - t0:.1f}s")
    return model


def load_probe():
    from src.probes.core.storage import load_probe_direction
    layer, direction = load_probe_direction(MANIFEST_DIR, PROBE_ID)
    print(f"Probe loaded: layer={layer}, direction shape={direction.shape}")
    return layer, direction


def compute_mean_norm(layer: int) -> float:
    from src.probes.core.activations import compute_activation_norms
    norms = compute_activation_norms(ACTIVATIONS_PATH, layers=[layer])
    return norms[layer]


def load_checkpoint() -> dict[str, list]:
    """Load checkpoint as dict mapping key -> list of records."""
    result = {}
    if CHECKPOINT_PATH.exists():
        for line in CHECKPOINT_PATH.read_text().strip().split("\n"):
            if line.strip():
                entry = json.loads(line)
                key = f"{entry['pair_id']}_{entry['coefficient']}_{entry['condition']}_{entry['ordering']}"
                if key not in result:
                    result[key] = []
                result[key].append(entry)
    return result


def append_checkpoint(records: list[dict]):
    with open(CHECKPOINT_PATH, "a") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


# ── Phase 1: Coherence sweep ────────────────────────────────────────────────

OPEN_ENDED_PROMPTS = [
    "What is the meaning of life?",
    "Explain quantum computing in simple terms.",
    "Write a short poem about autumn.",
    "What are the main causes of climate change?",
    "Describe how a car engine works.",
    "What is the history of the internet?",
    "How do vaccines work?",
    "Explain the theory of relativity.",
    "What are some tips for learning a new language?",
    "Describe the water cycle.",
    "What is artificial intelligence?",
    "How does photosynthesis work?",
    "What are the benefits of exercise?",
    "Explain how a computer works.",
    "What is the significance of the Renaissance?",
    "How do airplanes fly?",
    "What is the difference between weather and climate?",
    "Explain how electricity is generated.",
    "What are black holes?",
    "How does the stock market work?",
]


def run_coherence_sweep(hf_model, layer, direction, mean_norm, pairs, builder):
    """Phase 1: Test coherence at all coefficients using generate_n for speed."""
    from src.models.base import all_tokens_steering
    from src.steering.client import SteeredHFClient

    coefficients = [mean_norm * m for m in MULTIPLIERS]
    results = {"open_ended": {}, "pairwise": {}}

    for coef_idx, (mult, coef) in enumerate(zip(MULTIPLIERS, coefficients)):
        coef_key = f"{mult:.3f}"
        print(f"\n=== Coherence [{coef_idx+1}/{len(MULTIPLIERS)}]: mult={mult}, coef={coef:.1f} ===")

        # ── Open-ended (all_tokens steering) ──
        oe_coherent = 0
        oe_total = 0
        oe_examples = []

        if coef == 0:
            for prompt_text in OPEN_ENDED_PROMPTS:
                messages = [{"role": "user", "content": prompt_text}]
                resps = hf_model.generate_n(messages, n=COHERENCE_TRIALS_PER_PROMPT, temperature=TEMPERATURE)
                for resp in resps:
                    is_coh = len(resp.strip()) > 20
                    words = resp.split()
                    if len(words) > 10 and len(set(words)) / len(words) < 0.15:
                        is_coh = False
                    if is_coh:
                        oe_coherent += 1
                    oe_total += 1
                    if len(oe_examples) < 3:
                        oe_examples.append(resp[:150])
        else:
            scaled = direction * coef
            steering_tensor = torch.tensor(scaled, dtype=torch.bfloat16, device=hf_model.device)
            for prompt_text in OPEN_ENDED_PROMPTS:
                messages = [{"role": "user", "content": prompt_text}]
                hook = all_tokens_steering(steering_tensor)
                resps = hf_model.generate_with_steering_n(
                    messages, layer=layer, steering_hook=hook,
                    n=COHERENCE_TRIALS_PER_PROMPT, temperature=TEMPERATURE,
                )
                for resp in resps:
                    is_coh = len(resp.strip()) > 20 and sum(1 for c in resp if c.isascii()) / max(len(resp), 1) > 0.8
                    words = resp.split()
                    if len(words) > 10 and len(set(words)) / len(words) < 0.15:
                        is_coh = False
                    if is_coh:
                        oe_coherent += 1
                    oe_total += 1
                    if len(oe_examples) < 3:
                        oe_examples.append(resp[:150])

        oe_rate = oe_coherent / oe_total if oe_total > 0 else 0
        results["open_ended"][coef_key] = {
            "multiplier": mult, "coefficient": coef,
            "coherent": oe_coherent, "total": oe_total, "rate": oe_rate,
            "examples": oe_examples,
        }
        print(f"  Open-ended: {oe_coherent}/{oe_total} = {oe_rate:.2%}")

        # ── Pairwise (differential steering) ──
        pw_coherent = 0
        pw_total = 0
        pw_parseable = 0
        pw_choices = {"a": 0, "b": 0, "none": 0}
        pw_examples = []

        test_pairs = pairs[:N_COHERENCE_PAIRWISE_PAIRS]
        client = SteeredHFClient(hf_model, layer, direction, coefficient=coef, steering_mode="differential")

        for pair in test_pairs:
            for ordering in [0, 1]:
                if ordering == 0:
                    prompt = builder.build(pair["task_a"], pair["task_b"])
                else:
                    prompt = builder.build(pair["task_b"], pair["task_a"])

                task_prompts = [t.prompt for t in prompt.tasks]
                # 3 trials per ordering for coherence (total 6 per pair, enough for coherence estimation)
                n_trials = 3

                resps = client.generate_n(prompt.messages, n=n_trials, temperature=TEMPERATURE, task_prompts=task_prompts)

                for resp in resps:
                    is_coh = judge_coherence_local(resp)
                    choice = parse_choice_local(resp)
                    if is_coh:
                        pw_coherent += 1
                    if choice is not None:
                        pw_parseable += 1
                        pw_choices[choice] += 1
                    else:
                        pw_choices["none"] += 1
                    pw_total += 1
                    if len(pw_examples) < 3:
                        pw_examples.append(resp[:150])

        pw_rate = pw_coherent / pw_total if pw_total > 0 else 0
        pw_parse_rate = pw_parseable / pw_total if pw_total > 0 else 0
        results["pairwise"][coef_key] = {
            "multiplier": mult, "coefficient": coef,
            "coherent": pw_coherent, "total": pw_total, "rate": pw_rate,
            "parseable": pw_parseable, "parse_rate": pw_parse_rate,
            "choices": pw_choices,
            "examples": pw_examples,
        }
        print(f"  Pairwise: coh={pw_rate:.2%}, parse={pw_parse_rate:.2%}, choices={pw_choices}")

        COHERENCE_RESULTS_PATH.write_text(json.dumps(results, indent=2))

    return results


def determine_coherent_coefficients(coherence_results: dict, threshold: float = 0.85) -> list[float]:
    """Determine which multipliers pass the coherence threshold."""
    coherent = []
    for coef_key, data in coherence_results["pairwise"].items():
        if data["rate"] >= threshold and data["parse_rate"] >= 0.7:
            coherent.append(data["multiplier"])
    if 0.0 not in coherent:
        coherent.append(0.0)
    coherent.sort()
    return coherent


# ── Phase 2 & 3: Preference signal sweep ────────────────────────────────────

def run_preference_sweep(
    hf_model, layer, direction, mean_norm, pairs, builder,
    multipliers: list[float],
    condition: str = "probe",
    direction_override: np.ndarray | None = None,
):
    """Run preference sweep using generate_n for speed.

    Falls back to all_tokens steering for pairs where differential
    steering fails (task text not found verbatim in formatted prompt).
    """
    from src.steering.client import SteeredHFClient
    from src.models.base import all_tokens_steering

    use_direction = direction_override if direction_override is not None else direction
    checkpoint = load_checkpoint()
    coefficients = [mean_norm * m for m in multipliers]
    all_records = []
    fallback_count = 0

    total_calls = len(pairs) * 2 * len(coefficients)
    call_count = 0
    t_start = time.time()

    for coef_idx, (mult, coef) in enumerate(zip(multipliers, coefficients)):
        print(f"\n=== {condition} [{coef_idx+1}/{len(multipliers)}]: mult={mult}, coef={coef:.1f} ===")
        client = SteeredHFClient(hf_model, layer, use_direction, coefficient=coef, steering_mode="differential")
        block_records = []

        for pair_idx, pair in enumerate(pairs):
            for ordering in [0, 1]:
                ckpt_key = f"{pair['pair_id']}_{coef}_{condition}_{ordering}"
                existing = checkpoint.get(ckpt_key, [])
                n_existing = len(existing)
                n_needed = TRIALS_PER_ORDERING - n_existing

                call_count += 1

                if n_needed <= 0:
                    continue

                if ordering == 0:
                    prompt = builder.build(pair["task_a"], pair["task_b"])
                else:
                    prompt = builder.build(pair["task_b"], pair["task_a"])

                task_prompts = [t.prompt for t in prompt.tasks]
                steering_fallback = False

                try:
                    resps = client.generate_n(prompt.messages, n=n_needed, temperature=TEMPERATURE, task_prompts=task_prompts)
                except ValueError:
                    # Differential steering failed (task text not found in formatted prompt).
                    # Fall back to all_tokens steering for this pair.
                    steering_fallback = True
                    fallback_count += 1
                    if coef == 0:
                        resps = hf_model.generate_n(prompt.messages, n=n_needed, temperature=TEMPERATURE)
                    else:
                        scaled = use_direction * coef
                        steering_tensor = torch.tensor(scaled, dtype=torch.bfloat16, device=hf_model.device)
                        hook = all_tokens_steering(steering_tensor)
                        resps = hf_model.generate_with_steering_n(
                            prompt.messages, layer=client.layer, steering_hook=hook,
                            n=n_needed, temperature=TEMPERATURE,
                        )

                for sample_idx, resp in enumerate(resps):
                    choice_presented = parse_choice_local(resp)
                    if choice_presented is None:
                        choice_original = None
                    elif ordering == 0:
                        choice_original = choice_presented
                    else:
                        choice_original = "b" if choice_presented == "a" else "a"

                    record = {
                        "pair_id": pair["pair_id"],
                        "task_a_id": pair["task_a"].id,
                        "task_b_id": pair["task_b"].id,
                        "coefficient": coef,
                        "multiplier": mult,
                        "condition": condition,
                        "sample_idx": n_existing + sample_idx,
                        "ordering": ordering,
                        "choice_original": choice_original,
                        "choice_presented": choice_presented,
                        "raw_response": resp[:500],
                        "delta_mu": pair["delta_mu"],
                        "steering_fallback": steering_fallback,
                    }
                    block_records.append(record)
                    all_records.append(record)

            if (pair_idx + 1) % 20 == 0:
                if block_records:
                    append_checkpoint(block_records)
                    block_records = []
                elapsed = time.time() - t_start
                rate = call_count / elapsed if elapsed > 0 else 0
                remaining = (total_calls - call_count) / rate if rate > 0 else 0
                print(f"  Pair {pair_idx+1}/{len(pairs)} | {call_count}/{total_calls} | "
                      f"{elapsed/60:.0f}m elapsed, ~{remaining/60:.0f}m remaining")

        if block_records:
            append_checkpoint(block_records)
            block_records = []

    if fallback_count > 0:
        print(f"  NOTE: {fallback_count} pair/ordering combos used all_tokens fallback")

    return all_records


# ── Results compilation ──────────────────────────────────────────────────────

def compile_results():
    if not CHECKPOINT_PATH.exists():
        print("No checkpoint data found.")
        return

    records = []
    for line in CHECKPOINT_PATH.read_text().strip().split("\n"):
        if line.strip():
            records.append(json.loads(line))

    print(f"Total records: {len(records)}")

    from collections import defaultdict
    pair_coef_cond = defaultdict(list)
    for rec in records:
        key = (rec["pair_id"], rec["coefficient"], rec["condition"])
        pair_coef_cond[key].append(rec)

    summaries = []
    for (pair_id, coef, condition), recs in pair_coef_cond.items():
        valid = [r for r in recs if r["choice_original"] is not None]
        n_a = sum(1 for r in valid if r["choice_original"] == "a")
        n_total = len(valid)
        n_unparseable = len(recs) - len(valid)

        summaries.append({
            "pair_id": pair_id,
            "coefficient": coef,
            "multiplier": recs[0].get("multiplier", 0),
            "condition": condition,
            "n_a": n_a,
            "n_total": n_total,
            "n_unparseable": n_unparseable,
            "p_a": n_a / n_total if n_total > 0 else None,
            "delta_mu": recs[0].get("delta_mu", None),
        })

    results = {
        "trials": records,
        "pair_summaries": summaries,
        "metadata": {
            "n_pairs": len(set(r["pair_id"] for r in records)),
            "n_trials": len(records),
            "conditions": list(set(r["condition"] for r in records)),
            "coefficients": sorted(set(r["coefficient"] for r in records)),
        },
    }

    STEERING_RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {STEERING_RESULTS_PATH}")
    print(f"  Pairs: {results['metadata']['n_pairs']}")
    print(f"  Conditions: {results['metadata']['conditions']}")
    print(f"  Coefficients: {len(results['metadata']['coefficients'])}")
    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, default=0, help="Phase (0=all, 1=coherence, 2=preference, 3=random)")
    parser.add_argument("--coherence-threshold", type=float, default=0.85)
    parser.add_argument("--skip-phase1", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    if args.compile_only:
        compile_results()
        return

    hf_model = load_hf_model()
    layer, direction = load_probe()
    mean_norm = compute_mean_norm(layer)
    print(f"Mean activation norm at L{layer}: {mean_norm:.2f}")

    pairs = load_pairs()
    template = load_template()
    builder = build_prompt_builder(template)
    print(f"Loaded {len(pairs)} pairs")

    if args.phase in [0, 1] and not args.skip_phase1:
        print("\n" + "="*60)
        print("PHASE 1: COHERENCE SWEEP")
        print("="*60)
        t0 = time.time()
        coherence_results = run_coherence_sweep(hf_model, layer, direction, mean_norm, pairs, builder)
        coherent_mults = determine_coherent_coefficients(coherence_results, args.coherence_threshold)
        print(f"\nPhase 1 done in {(time.time()-t0)/60:.1f}m")
        print(f"Coherent multipliers (threshold={args.coherence_threshold}): {coherent_mults}")
    else:
        if COHERENCE_RESULTS_PATH.exists():
            coherence_results = json.loads(COHERENCE_RESULTS_PATH.read_text())
            coherent_mults = determine_coherent_coefficients(coherence_results, args.coherence_threshold)
            print(f"Loaded coherence results. Coherent multipliers: {coherent_mults}")
        else:
            print("No coherence results. Using all multipliers.")
            coherent_mults = MULTIPLIERS

    if args.phase in [0, 2]:
        print("\n" + "="*60)
        print("PHASE 2: PREFERENCE SIGNAL SWEEP (probe direction)")
        print("="*60)
        t0 = time.time()
        probe_records = run_preference_sweep(
            hf_model, layer, direction, mean_norm, pairs, builder,
            multipliers=coherent_mults, condition="probe",
        )
        print(f"Phase 2 done in {(time.time()-t0)/60:.1f}m — {len(probe_records)} new records")

    if args.phase in [0, 3]:
        print("\n" + "="*60)
        print("PHASE 3: RANDOM DIRECTION CONTROL")
        print("="*60)
        rng = np.random.default_rng(42)
        random_direction = rng.standard_normal(direction.shape)
        random_direction = random_direction / np.linalg.norm(random_direction)
        t0 = time.time()
        random_records = run_preference_sweep(
            hf_model, layer, direction, mean_norm, pairs, builder,
            multipliers=coherent_mults, condition="random",
            direction_override=random_direction,
        )
        print(f"Phase 3 done in {(time.time()-t0)/60:.1f}m — {len(random_records)} new records")

    compile_results()
    print("\nAll phases complete.")


if __name__ == "__main__":
    main()
