"""Cross-evaluate IT and PT probes on IT and PT activations.

We don't have Thurstonian scores locally, so we correlate each probe x
activation combo's predictions against the IT-probe-on-IT-acts predictions
(the "gold standard" proxy for IT preferences, r=0.864 on heldout).

Also reports cosine similarity between the two probes' weight vectors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

from src.probes.core.activations import load_activations
from src.probes.core.evaluate import score_with_probe

REPO = Path(__file__).resolve().parent.parent.parent

IT_PROBE = np.load(REPO / "results/probes/gemma3_10k_heldout_std_raw/probes/probe_ridge_L31.npy")
PT_PROBE = np.load(REPO / "results/probes/gemma3_pt_10k_heldout_std_raw/probes/probe_ridge_L31.npy")

IT_ACTS_PATH = REPO / "activations/gemma_3_27b/activations_prompt_last.npz"
PT_ACTS_PATH = REPO / "activations/gemma_3_27b_pt/activations_prompt_last.npz"

LAYER = 31


def main() -> None:
    # Load activations
    it_tids, it_acts = load_activations(IT_ACTS_PATH, layers=[LAYER])
    pt_tids, pt_acts = load_activations(PT_ACTS_PATH, layers=[LAYER])

    # Find shared task IDs
    it_id_to_idx = {tid: i for i, tid in enumerate(it_tids)}
    pt_id_to_idx = {tid: i for i, tid in enumerate(pt_tids)}
    shared = sorted(set(it_id_to_idx) & set(pt_id_to_idx))
    print(f"Shared tasks: {len(shared)} (IT has {len(it_tids)}, PT has {len(pt_tids)})")

    it_idx = np.array([it_id_to_idx[tid] for tid in shared])
    pt_idx = np.array([pt_id_to_idx[tid] for tid in shared])

    # Score all four combos on shared tasks
    scores = {
        ("IT probe", "IT acts"): score_with_probe(IT_PROBE, it_acts[LAYER][it_idx]),
        ("IT probe", "PT acts"): score_with_probe(IT_PROBE, pt_acts[LAYER][pt_idx]),
        ("PT probe", "IT acts"): score_with_probe(PT_PROBE, it_acts[LAYER][it_idx]),
        ("PT probe", "PT acts"): score_with_probe(PT_PROBE, pt_acts[LAYER][pt_idx]),
    }

    # Probe cosine similarity (exclude intercept)
    it_w = IT_PROBE[:-1]
    pt_w = PT_PROBE[:-1]
    cos_sim = np.dot(it_w, pt_w) / (np.linalg.norm(it_w) * np.linalg.norm(pt_w))
    print(f"\nProbe cosine similarity (IT vs PT): {cos_sim:.4f}")

    # Pairwise correlations between all four score vectors
    labels = list(scores.keys())
    print(f"\nPearson r between all probe-activation combos (n={len(shared)}):\n")
    header = f"{'':>24}" + "".join(f"  {p} / {a}" for p, a in labels)
    print(header)
    print("-" * len(header))
    for i, key_i in enumerate(labels):
        row = f"{key_i[0] + ' / ' + key_i[1]:>24}"
        for j, key_j in enumerate(labels):
            r, _ = pearsonr(scores[key_i], scores[key_j])
            row += f"  {r:>13.4f}"
        print(row)


if __name__ == "__main__":
    main()
