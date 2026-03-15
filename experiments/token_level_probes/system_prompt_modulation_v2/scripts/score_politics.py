"""Score politics system-prompt variant items with preference probes.

All items are assistant-turn. Extracts critical span + EOT + logprobs.

Usage:
    python experiments/token_level_probes/system_prompt_modulation_v2/scripts/score_politics.py
"""

import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.models.huggingface_model import HuggingFaceModel
from src.probes.scoring import score_prompt_all_tokens
from src.steering.tokenization import find_text_span

DATA_DIR = Path("experiments/token_level_probes/system_prompt_modulation_v2/data")
OUTPUT_PATH = Path("experiments/token_level_probes/system_prompt_modulation_v2/politics_scoring_results.json")

PROBE_SETS = {
    "tb-2": Path("results/probes/heldout_eval_gemma3_tb-2/probes"),
    "tb-5": Path("results/probes/heldout_eval_gemma3_tb-5/probes"),
    "task_mean": Path("results/probes/heldout_eval_gemma3_task_mean/probes"),
}
LAYERS = [32, 39, 53]


def load_probes():
    named_probes = []
    scoring_probes = []
    for probe_name, probe_dir in PROBE_SETS.items():
        for layer in LAYERS:
            weights = np.load(probe_dir / f"probe_ridge_L{layer}.npy")
            key = f"{probe_name}_L{layer}"
            named_probes.append((key, layer, weights))
            scoring_probes.append((layer, weights))
    return named_probes, scoring_probes


def load_stimuli():
    return json.load(open(DATA_DIR / "politics_system_prompts_v2.json"))


def get_logprobs(model, messages, token_ids):
    """Extract per-token logprobs from the model."""
    formatted = model.format_messages(messages, add_generation_prompt=False)
    inputs = model.tokenizer(formatted, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(model.model.device)

    with torch.no_grad():
        outputs = model.model(input_ids)
        logits = outputs.logits[0]  # (seq_len, vocab_size)
        log_probs = torch.log_softmax(logits, dim=-1)

        # For each position, get the logprob of the NEXT token
        # logprobs[i] = log P(token[i+1] | token[0..i])
        token_logprobs = []
        for i in range(len(input_ids[0]) - 1):
            next_token = input_ids[0, i + 1]
            token_logprobs.append(float(log_probs[i, next_token]))

        # First token has no logprob (no preceding context)
        token_logprobs = [0.0] + token_logprobs

    return token_logprobs


def score_item(model, item, named_probes, scoring_probes):
    messages = item["messages"]

    all_scores = score_prompt_all_tokens(
        model, messages, scoring_probes, add_generation_prompt=False,
    )

    formatted = model.format_messages(messages, add_generation_prompt=False)
    token_ids = model.tokenizer(formatted, add_special_tokens=False)["input_ids"]
    tokens = [model.tokenizer.decode(tid) for tid in token_ids]

    critical_span = item["critical_span"]
    span_start, span_end = find_text_span(model.tokenizer, formatted, critical_span)

    # Extract logprobs
    token_logprobs = get_logprobs(model, messages, token_ids)
    critical_span_mean_logprob = float(np.mean(token_logprobs[span_start:span_end]))
    eot_logprob = token_logprobs[-1]

    critical_span_mean_scores = {}
    eot_scores = {}

    for i, (key, _layer, _weights) in enumerate(named_probes):
        scores_arr = all_scores[i]
        critical_span_mean_scores[key] = float(np.mean(scores_arr[span_start:span_end]))
        eot_scores[key] = float(scores_arr[-1])

    return {
        "id": item["id"],
        "domain": item["domain"],
        "turn": item["turn"],
        "condition": item["condition"],
        "system_prompt": item["system_prompt"],
        "critical_span": critical_span,
        "issue": item.get("issue", ""),
        "critical_span_mean_scores": critical_span_mean_scores,
        "eot_scores": eot_scores,
        "critical_span_mean_logprob": critical_span_mean_logprob,
        "eot_logprob": eot_logprob,
    }


def run_pilot(model, items, named_probes, scoring_probes):
    seen = set()
    pilots = []
    for item in items:
        sp = item["system_prompt"]
        if sp not in seen and len(pilots) < 4:
            pilots.append(item)
            seen.add(sp)
        if len(pilots) == 4:
            break

    for item in pilots:
        print(f"\n{'='*60}")
        print(f"PILOT: {item['id']} (system_prompt={item['system_prompt']})")
        result = score_item(model, item, named_probes, scoring_probes)
        print(f"Critical span mean (task_mean_L39): {result['critical_span_mean_scores']['task_mean_L39']:.4f}")
        print(f"EOT score (task_mean_L39): {result['eot_scores']['task_mean_L39']:.4f}")
        print(f"Critical span mean logprob: {result['critical_span_mean_logprob']:.4f}")
        print(f"EOT logprob: {result['eot_logprob']:.4f}")

    print(f"\n{'='*60}")
    print("PILOT PASSED")


def main():
    print("Loading model...")
    model = HuggingFaceModel("google/gemma-3-27b-it")

    print("Loading probes...")
    named_probes, scoring_probes = load_probes()

    print("Loading stimuli...")
    items = load_stimuli()
    print(f"Loaded {len(items)} items")

    print("\n--- PILOT ---")
    run_pilot(model, items, named_probes, scoring_probes)

    print("\n--- SCORING ---")
    results = []
    for item in tqdm(items, desc="Scoring"):
        results.append(score_item(model, item, named_probes, scoring_probes))

    output = {
        "items": results,
        "probe_configs": {
            f"{name}_L{layer}": {"probe_set": name, "layer": layer}
            for name in PROBE_SETS for layer in LAYERS
        },
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    file_size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"\nSaved {len(results)} items to {OUTPUT_PATH} ({file_size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
