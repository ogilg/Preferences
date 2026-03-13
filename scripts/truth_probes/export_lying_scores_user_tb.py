"""Export per-item probe scores for user turn-boundary violins. Chunked for 512MB limit."""

import gc
import json
import tempfile
import zipfile
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ACT_PATH = ROOT / "activations" / "gemma_3_27b_lying_10prompt_user_tb" / "activations_turn_boundary:-2.npz"
PROBES_DIR = ROOT / "results" / "probes" / "heldout_eval_gemma3_tb-2" / "probes"
OUTPUT = ROOT / "experiments" / "truth_probes" / "error_prefill" / "lying_prompts" / "lying_10prompt_scores_user_tb.json"

LAYERS = [32, 53]
CHUNK = 2000


def extract_to_tmpfile(npz_path, array_name):
    tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
    with zipfile.ZipFile(npz_path, "r") as zf:
        with zf.open(array_name + ".npy") as src:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                tmp.write(chunk)
    tmp.close()
    return np.load(tmp.name, mmap_mode="r", allow_pickle=True), tmp.name


def main():
    print("Loading task_ids...")
    task_ids_arr, tmp1 = extract_to_tmpfile(ACT_PATH, "task_ids")
    task_ids = list(task_ids_arr)
    del task_ids_arr
    os.unlink(tmp1)
    print(f"  {len(task_ids)} items")

    results = {"task_ids": task_ids}

    for layer in LAYERS:
        print(f"Layer {layer}: extracting to temp file...")
        acts, tmp_path = extract_to_tmpfile(ACT_PATH, f"layer_{layer}")
        probe = np.load(PROBES_DIR / f"probe_ridge_L{layer}.npy")
        weights, bias = probe[:-1], probe[-1]

        all_scores = []
        for start in range(0, len(acts), CHUNK):
            end = min(start + CHUNK, len(acts))
            chunk_scores = (acts[start:end] @ weights + bias).astype(np.float32)
            all_scores.extend(chunk_scores.tolist())

        results[f"L{layer}"] = all_scores
        print(f"  {len(all_scores)} scores")
        del acts, probe, weights, all_scores
        gc.collect()
        os.unlink(tmp_path)

    with open(OUTPUT, "w") as f:
        json.dump(results, f)
    print(f"Saved {OUTPUT} ({os.path.getsize(str(OUTPUT)) / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
