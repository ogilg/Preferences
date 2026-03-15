"""Re-score parent experiment items to extract EOT scores.

The original scoring_results.json doesn't have eot_scores (all_token_scores was
split to .npz which is gitignored and lost). This re-scores all 1,536 parent items
but only saves critical_span_mean_scores and eot_scores (no all_token_scores).

Usage:
    python experiments/token_level_probes/system_prompt_modulation_v2/scripts/rescore_parent_eot.py
"""

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.models.huggingface_model import HuggingFaceModel
from src.probes.scoring import score_prompt_all_tokens
from src.steering.tokenization import find_text_span

DATA_DIR = Path("experiments/token_level_probes/data")
OUTPUT_PATH = Path("experiments/token_level_probes/system_prompt_modulation_v2/parent_eot_scores.json")

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
    items = []
    for filename in ["truth_filtered.json", "harm_filtered.json", "politics_filtered.json"]:
        items.extend(json.load(open(DATA_DIR / filename)))
    return items


def score_item(model, item, named_probes, scoring_probes):
    messages = item["messages"]
    last_role = messages[-1]["role"]
    add_gen = last_role == "user"

    all_scores = score_prompt_all_tokens(
        model, messages, scoring_probes, add_generation_prompt=add_gen,
    )

    formatted = model.format_messages(messages, add_generation_prompt=add_gen)
    critical_span = item["critical_span"]
    span_start, span_end = find_text_span(model.tokenizer, formatted, critical_span)

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
        "critical_span_mean_scores": critical_span_mean_scores,
        "eot_scores": eot_scores,
        **{k: item[k] for k in ["system_prompt", "source_id", "issue"] if k in item},
    }


def main():
    print("Loading model...")
    model = HuggingFaceModel("google/gemma-3-27b-it")

    print("Loading probes...")
    named_probes, scoring_probes = load_probes()

    print("Loading stimuli...")
    items = load_stimuli()
    print(f"Loaded {len(items)} items")

    print("\n--- SCORING ---")
    results = []
    for item in tqdm(items, desc="Re-scoring for EOT"):
        results.append(score_item(model, item, named_probes, scoring_probes))

    OUTPUT_PATH.write_text(json.dumps({"items": results}, indent=2))
    print(f"\nSaved {len(results)} items to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
