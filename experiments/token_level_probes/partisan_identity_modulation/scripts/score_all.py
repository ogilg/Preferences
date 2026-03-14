"""Score partisan-identity variant items with preference probes.

Adapted from system_prompt_modulation/scripts/score_all.py.
All items are assistant-turn, so add_generation_prompt=False throughout.

Usage:
    python experiments/token_level_probes/partisan_identity_modulation/scripts/score_all.py
"""

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.models.huggingface_model import HuggingFaceModel
from src.probes.scoring import score_prompt_all_tokens
from src.steering.tokenization import find_text_span

DATA_DIR = Path("experiments/token_level_probes/partisan_identity_modulation/data")
OUTPUT_PATH = Path("experiments/token_level_probes/partisan_identity_modulation/scoring_results.json")

PROBE_SETS = {
    "tb-2": Path("results/probes/heldout_eval_gemma3_tb-2/probes"),
    "tb-5": Path("results/probes/heldout_eval_gemma3_tb-5/probes"),
    "task_mean": Path("results/probes/heldout_eval_gemma3_task_mean/probes"),
}
LAYERS = [32, 39, 53]


def load_probes() -> tuple[list[tuple[str, int, np.ndarray]], list[tuple[int, np.ndarray]]]:
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
    for filename in ["truth_partisan.json", "harm_partisan.json"]:
        items.extend(json.load(open(DATA_DIR / filename)))
    return items


def find_fullstop_indices(tokens: list[str]) -> list[int]:
    return [i for i, tok in enumerate(tokens) if "." in tok]


def score_item(
    model: HuggingFaceModel,
    item: dict,
    named_probes: list[tuple[str, int, np.ndarray]],
    scoring_probes: list[tuple[int, np.ndarray]],
) -> dict:
    messages = item["messages"]

    all_scores = score_prompt_all_tokens(
        model, messages, scoring_probes, add_generation_prompt=False,
    )

    formatted = model.format_messages(messages, add_generation_prompt=False)
    token_ids = model.tokenizer(formatted, add_special_tokens=False)["input_ids"]
    tokens = [model.tokenizer.decode(tid) for tid in token_ids]

    critical_span = item["critical_span"]
    span_start, span_end = find_text_span(model.tokenizer, formatted, critical_span)

    fullstop_indices = find_fullstop_indices(tokens)

    critical_span_scores = {}
    critical_span_mean_scores = {}
    fullstop_scores = {}
    all_token_scores = {}
    eot_scores = {}

    for i, (key, _layer, _weights) in enumerate(named_probes):
        scores_arr = all_scores[i]
        critical_span_scores[key] = scores_arr[span_start:span_end].tolist()
        critical_span_mean_scores[key] = float(np.mean(scores_arr[span_start:span_end]))
        fullstop_scores[key] = [float(scores_arr[idx]) for idx in fullstop_indices]
        all_token_scores[key] = scores_arr.tolist()
        eot_scores[key] = float(scores_arr[-1])

    return {
        "id": item["id"],
        "domain": item["domain"],
        "turn": item["turn"],
        "condition": item["condition"],
        "system_prompt": item["system_prompt"],
        "critical_span": critical_span,
        "critical_token_indices": list(range(span_start, span_end)),
        "fullstop_indices": fullstop_indices,
        "critical_span_scores": critical_span_scores,
        "critical_span_mean_scores": critical_span_mean_scores,
        "eot_scores": eot_scores,
        "fullstop_scores": fullstop_scores,
        "all_token_scores": all_token_scores,
        "tokens": tokens,
        **{k: item[k] for k in ["source_id", "issue"] if k in item},
    }


def run_pilot(model, items, named_probes, scoring_probes):
    pilot_items = []
    seen = set()
    for item in items:
        key = (item["domain"], item["system_prompt"])
        if key not in seen:
            seen.add(key)
            pilot_items.append(item)
        if len(pilot_items) == 3:
            break

    for item in pilot_items:
        print(f"\n{'='*60}")
        print(f"PILOT: {item['id']} (system_prompt={item['system_prompt']})")
        print(f"Critical span: '{item['critical_span']}'")

        result = score_item(model, item, named_probes, scoring_probes)

        critical_tokens = [result["tokens"][i] for i in result["critical_token_indices"]]
        print(f"Critical tokens: {critical_tokens}")
        print(f"EOT score (task_mean_L32): {result['eot_scores'].get('task_mean_L32', 'N/A'):.4f}")

        first_probe = list(result["critical_span_mean_scores"].keys())[0]
        all_s = result["all_token_scores"][first_probe]
        print(f"Score range ({first_probe}): [{min(all_s):.4f}, {max(all_s):.4f}]")
        print(f"Critical span mean: {result['critical_span_mean_scores'][first_probe]:.4f}")

        if any(np.isnan(s) for s in all_s):
            raise ValueError(f"NaN scores detected for {item['id']}")

    print(f"\n{'='*60}")
    print("PILOT PASSED")


def main():
    print("Loading model...")
    model = HuggingFaceModel("google/gemma-3-27b-it")

    print("Loading probes...")
    named_probes, scoring_probes = load_probes()
    print(f"Loaded {len(named_probes)} probes")

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
            f"{name}_L{layer}": {"probe_set": name, "layer": layer, "path": str(PROBE_SETS[name] / f"probe_ridge_L{layer}.npy")}
            for name in PROBE_SETS for layer in LAYERS
        },
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    file_size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"\nSaved {len(results)} items to {OUTPUT_PATH} ({file_size_mb:.1f} MB)")

    if file_size_mb > 20:
        print("WARNING: File exceeds 20MB. Consider splitting all_token_scores to .npz.")


if __name__ == "__main__":
    main()
