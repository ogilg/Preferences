"""Do the two orphaned ridge_L23 candidates give the same steering direction?

If cosine similarity is ~1 the tb-2/eot ambiguity is immaterial; if not, the
choice changes the intervention and must be resolved before running.
"""

import numpy as np

CANDIDATES = {
    "tb-2": "results/probes/layer_sweep/tb-2/probes/probe_ridge_L23.npy",
    "eot": "results/probes/layer_sweep/eot/probes/probe_ridge_L23.npy",
}


def direction(path: str) -> np.ndarray:
    w = np.load(path)
    d = w[:-1]  # last element is the intercept
    return d / np.linalg.norm(d)


dirs = {k: direction(v) for k, v in CANDIDATES.items()}
for k, v in CANDIDATES.items():
    raw = np.load(v)
    print(f"{k}: shape={raw.shape} intercept={raw[-1]:.6f} dir_norm_preunit={np.linalg.norm(raw[:-1]):.4f}")

a, b = dirs["tb-2"], dirs["eot"]
print(f"\ncosine(tb-2, eot) = {float(a @ b):.6f}")
print(f"identical bytes: {np.array_equal(np.load(CANDIDATES['tb-2']), np.load(CANDIDATES['eot']))}")
