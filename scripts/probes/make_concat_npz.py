"""Create a concatenated activation file from EOT and prompt_last at L31."""

from pathlib import Path

import numpy as np

PL_PATH = Path("activations/gemma_3_27b/activations_prompt_last.npz")
EOT_PATH = Path("activations/gemma_3_27b_eot/activations_eot.npz")
OUT_PATH = Path("activations/gemma_3_27b_concat/activations_concat.npz")

LAYER = 31


def main():
    pl = np.load(PL_PATH, allow_pickle=True)
    eot = np.load(EOT_PATH, allow_pickle=True)

    pl_ids = list(pl["task_ids"])
    eot_ids = list(eot["task_ids"])

    common = sorted(set(pl_ids) & set(eot_ids))
    print(f"Common tasks: {len(common)}")

    pl_idx = {tid: i for i, tid in enumerate(pl_ids)}
    eot_idx = {tid: i for i, tid in enumerate(eot_ids)}

    pl_order = [pl_idx[tid] for tid in common]
    eot_order = [eot_idx[tid] for tid in common]

    pl_acts = pl[f"layer_{LAYER}"][pl_order]
    eot_acts = eot[f"layer_{LAYER}"][eot_order]
    concat = np.concatenate([pl_acts, eot_acts], axis=1)

    print(f"Concat shape: {concat.shape}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_PATH, task_ids=np.array(common), **{f"layer_{LAYER}": concat})
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
