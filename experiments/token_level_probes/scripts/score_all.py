"""Score all token-level probe stimuli.

Loads each stimulus, runs it through the model with probe scoring hooks,
and saves per-token scores + critical span extraction.

Usage:
    python experiments/token_level_probes/scripts/score_all.py
"""

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.models.huggingface_model import HuggingFaceModel
from src.probes.scoring import score_prompt_all_tokens
from src.steering.tokenization import find_text_span

DATA_DIR = Path("experiments/token_level_probes/data")
OUTPUT_PATH = Path("experiments/token_level_probes/scoring_results.json")

PROBE_SETS = {
    "tb-2": Path("results/probes/heldout_eval_gemma3_tb-2/probes"),
    "tb-5": Path("results/probes/heldout_eval_gemma3_tb-5/probes"),
    "task_mean": Path("results/probes/heldout_eval_gemma3_task_mean/probes"),
}
LAYERS = [32, 39, 53]


def load_probes() -> tuple[list[tuple[str, int, np.ndarray]], list[tuple[int, np.ndarray]]]:
    """Load all probe weights.

    Returns:
        named_probes: list of (probe_name, layer, weights) for bookkeeping
        scoring_probes: list of (layer, weights) for score_prompt_all_tokens
    """
    named_probes = []
    scoring_probes = []
    for probe_name, probe_dir in PROBE_SETS.items():
        for layer in LAYERS:
            weights = np.load(probe_dir / f"probe_ridge_L{layer}.npy")
            key = f"{probe_name}_L{layer}"
            named_probes.append((key, layer, weights))
            scoring_probes.append((layer, weights))
    return named_probes, scoring_probes


def load_stimuli() -> list[dict]:
    items = []
    for filename in ["truth_filtered.json", "harm_filtered.json", "politics_filtered.json"]:
        path = DATA_DIR / filename
        items.extend(json.load(open(path)))
    return items


def find_fullstop_indices(tokens: list[str]) -> list[int]:
    """Find indices of tokens that are or contain a full stop."""
    indices = []
    for i, tok in enumerate(tokens):
        if "." in tok:
            indices.append(i)
    return indices


def score_item(
    model: HuggingFaceModel,
    item: dict,
    named_probes: list[tuple[str, int, np.ndarray]],
    scoring_probes: list[tuple[int, np.ndarray]],
) -> dict:
    """Score a single stimulus item and extract critical span + fullstop scores.

    Handles both user-turn items (messages = [user]) and assistant-turn items
    (messages = [user, assistant] or [system, user, assistant]).

    For user-turn items:
        - critical_span is in the user message content
        - We score with add_generation_prompt=True (normal)

    For assistant-turn items:
        - critical_span is in the prefilled assistant message content
        - We score with add_generation_prompt=False to avoid appending a
          spurious generation prompt after the assistant content
        - IMPORTANT: verify in pilot that this produces correct scores
    """
    messages = item["messages"]
    last_role = messages[-1]["role"]

    # Format the prompt the same way score_prompt_all_tokens will
    # Assistant-turn items: don't add generation prompt after prefilled content
    add_gen_prompt = last_role != "assistant"

    # --- Score all tokens ---
    all_scores = score_prompt_all_tokens(
        model, messages, scoring_probes, add_generation_prompt=add_gen_prompt,
    )

    # --- Get formatted prompt and tokens for span detection ---
    formatted = model.format_messages(messages, add_generation_prompt=add_gen_prompt)
    token_ids = model.tokenizer(formatted, add_special_tokens=False)["input_ids"]
    tokens = [model.tokenizer.decode(tid) for tid in token_ids]

    # --- Find critical span token indices ---
    critical_span = item["critical_span"]
    span_start, span_end = find_text_span(model.tokenizer, formatted, critical_span)

    # --- Find fullstop indices ---
    fullstop_indices = find_fullstop_indices(tokens)

    # --- Build per-probe score dicts ---
    critical_span_scores = {}
    critical_span_mean_scores = {}
    fullstop_scores = {}
    all_token_scores = {}

    for i, (key, _layer, _weights) in enumerate(named_probes):
        scores_arr = all_scores[i]
        critical_span_scores[key] = scores_arr[span_start:span_end].tolist()
        critical_span_mean_scores[key] = float(np.mean(scores_arr[span_start:span_end]))
        fullstop_scores[key] = [float(scores_arr[idx]) for idx in fullstop_indices]
        all_token_scores[key] = scores_arr.tolist()

    return {
        "id": item["id"],
        "domain": item["domain"],
        "turn": item["turn"],
        "condition": item["condition"],
        "critical_span": critical_span,
        "critical_token_indices": list(range(span_start, span_end)),
        "fullstop_indices": fullstop_indices,
        "critical_span_scores": critical_span_scores,
        "critical_span_mean_scores": critical_span_mean_scores,
        "fullstop_scores": fullstop_scores,
        "all_token_scores": all_token_scores,
        "tokens": tokens,
        # Preserve metadata for analysis grouping
        **{k: item[k] for k in ["source_id", "system_prompt", "issue"] if k in item},
    }


def run_pilot(
    model: HuggingFaceModel,
    items: list[dict],
    named_probes: list[tuple[str, int, np.ndarray]],
    scoring_probes: list[tuple[int, np.ndarray]],
) -> None:
    """Validate on 3 items before full scoring. Prints diagnostics."""
    # Pick one user-turn, one assistant-turn, one politics item
    pilot_items = []
    for item in items:
        if item["turn"] == "user" and item["domain"] == "truth" and not any(p["turn"] == "user" for p in pilot_items):
            pilot_items.append(item)
        elif item["turn"] == "assistant" and item["domain"] == "harm" and not any(p["turn"] == "assistant" and p["domain"] == "harm" for p in pilot_items):
            pilot_items.append(item)
        elif item["domain"] == "politics" and not any(p["domain"] == "politics" for p in pilot_items):
            pilot_items.append(item)
        if len(pilot_items) == 3:
            break

    for item in pilot_items:
        print(f"\n{'='*60}")
        print(f"PILOT: {item['id']} (turn={item['turn']}, domain={item['domain']})")
        print(f"Critical span: '{item['critical_span']}'")

        result = score_item(model, item, named_probes, scoring_probes)

        print(f"Tokens ({len(result['tokens'])}): {result['tokens']}")
        print(f"Critical span indices: {result['critical_token_indices']}")
        critical_tokens = [result["tokens"][i] for i in result["critical_token_indices"]]
        print(f"Critical tokens: {critical_tokens}")
        print(f"Fullstop indices: {result['fullstop_indices']}")

        # Check scores are reasonable
        first_probe = list(result["critical_span_mean_scores"].keys())[0]
        all_scores = result["all_token_scores"][first_probe]
        print(f"Score range ({first_probe}): [{min(all_scores):.4f}, {max(all_scores):.4f}]")
        print(f"Critical span mean: {result['critical_span_mean_scores'][first_probe]:.4f}")

        # Check for NaN
        if any(np.isnan(s) for s in all_scores):
            raise ValueError(f"NaN scores detected for {item['id']}")

    print(f"\n{'='*60}")
    print("PILOT PASSED — all items scored successfully")


def main():
    print("Loading model...")
    model = HuggingFaceModel("google/gemma-3-27b-it")

    print("Loading probes...")
    named_probes, scoring_probes = load_probes()
    print(f"Loaded {len(named_probes)} probes: {[p[0] for p in named_probes]}")

    print("Loading stimuli...")
    items = load_stimuli()
    print(f"Loaded {len(items)} items")

    print("\n--- PILOT ---")
    run_pilot(model, items, named_probes, scoring_probes)

    print("\n--- SCORING ALL ITEMS ---")
    results = []
    for item in tqdm(items, desc="Scoring"):
        results.append(score_item(model, item, named_probes, scoring_probes))

    output = {
        "items": results,
        "probe_configs": {
            f"{name}_L{layer}": {
                "probe_set": name,
                "layer": layer,
                "path": str(PROBE_SETS[name] / f"probe_ridge_L{layer}.npy"),
            }
            for name in PROBE_SETS
            for layer in LAYERS
        },
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    file_size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"\nSaved {len(results)} items to {OUTPUT_PATH} ({file_size_mb:.1f} MB)")

    if file_size_mb > 20:
        print("WARNING: File exceeds 20MB. Consider splitting all_token_scores to .npz.")


if __name__ == "__main__":
    main()
