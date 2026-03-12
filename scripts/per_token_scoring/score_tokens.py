"""Score all tokens with trained probes and save results."""
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from src.probes.core.activations import load_span_activations

ACTIVATIONS_PATH = Path("activations/gemma_3_27b_error_prefill/activations_assistant_all.npz")
DATA_PATH = Path("data/creak/error_prefill_none_100.json")
OUTPUT_PATH = Path("experiments/truth_probes/error_prefill/per_token_scoring/scored_tokens.json")

LAYERS = [25, 32, 39, 46, 53]
PROBE_DIRS = {
    "tb-2": Path("results/probes/heldout_eval_gemma3_tb-2/probes"),
    "tb-5": Path("results/probes/heldout_eval_gemma3_tb-5/probes"),
}

# Load data
with open(DATA_PATH) as f:
    data = json.load(f)
data_by_id = {d["task_id"]: d for d in data}

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-27b-it")

# Load span activations
print("Loading span activations...")
task_ids, activations = load_span_activations(ACTIVATIONS_PATH, layers=LAYERS)

# Load probes
probes = {}
for probe_name, probe_dir in PROBE_DIRS.items():
    probes[probe_name] = {}
    for layer in LAYERS:
        weights = np.load(probe_dir / f"probe_ridge_L{layer}.npy")
        probes[probe_name][layer] = weights
        print(f"  {probe_name} L{layer}: weights shape {weights.shape}")

# Score all tokens
results = []
for i, task_id in enumerate(task_ids):
    record = data_by_id[str(task_id)]
    assistant_content = record["messages"][1]["content"]

    # Tokenize to get individual token strings
    token_ids = tokenizer.encode(assistant_content, add_special_tokens=False)
    token_strings = [tokenizer.decode([tid]) for tid in token_ids]

    n_tokens_act = activations[LAYERS[0]][i].shape[0]
    n_tokens_tok = len(token_ids)

    if n_tokens_act != n_tokens_tok:
        print(f"  WARNING: {task_id} has {n_tokens_act} activation tokens vs {n_tokens_tok} tokenizer tokens")
        # Use min to avoid index errors; pad token strings if needed
        n_tokens = min(n_tokens_act, n_tokens_tok)
        token_strings = token_strings[:n_tokens]
    else:
        n_tokens = n_tokens_act

    scores_by_probe = {}
    for probe_name in PROBE_DIRS:
        scores_by_probe[probe_name] = {}
        for layer in LAYERS:
            acts = activations[layer][i][:n_tokens]  # (n_tokens, d_model)
            weights = probes[probe_name][layer]
            coefs = weights[:-1]
            intercept = weights[-1]
            token_scores = (acts @ coefs + intercept).tolist()
            scores_by_probe[probe_name][f"L{layer}"] = token_scores

    results.append({
        "task_id": str(task_id),
        "answer_condition": record["answer_condition"],
        "entity": record["entity"],
        "true_ex_id": record["true_ex_id"],
        "assistant_content": assistant_content,
        "token_strings": token_strings,
        "n_tokens": n_tokens,
        "scores": scores_by_probe,
    })

# Save
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved {len(results)} scored tasks to {OUTPUT_PATH}")

# Quick summary
correct = [r for r in results if r["answer_condition"] == "correct"]
incorrect = [r for r in results if r["answer_condition"] == "incorrect"]
print(f"Correct: {len(correct)}, Incorrect: {len(incorrect)}")

# Quick check: mean score at last token for L32 tb-2
correct_last = [r["scores"]["tb-2"]["L32"][-1] for r in correct]
incorrect_last = [r["scores"]["tb-2"]["L32"][-1] for r in incorrect]
mean_c = np.mean(correct_last)
mean_i = np.mean(incorrect_last)
std_pooled = np.sqrt((np.std(correct_last)**2 + np.std(incorrect_last)**2) / 2)
d = (mean_c - mean_i) / std_pooled if std_pooled > 0 else 0
print(f"\nQuick check — L32 tb-2 last token: correct mean={mean_c:.3f}, incorrect mean={mean_i:.3f}, d={d:.2f}")
