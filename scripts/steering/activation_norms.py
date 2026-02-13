"""Compute activation norm statistics at each layer for Gemma-3-27B.

Also computes the probe direction's projection onto activations to understand
what coefficient values mean in context.
"""

import numpy as np
from pathlib import Path

ACTIVATIONS_PATH = "activations/gemma_3_27b/activations_prompt_last.npz"
PROBE_DIR = "results/probes/gemma3_3k_nostd_raw"

data = np.load(ACTIVATIONS_PATH)
task_ids = data["task_ids"]
print(f"Tasks: {len(task_ids)}")

layers = [15, 31, 37, 43, 49, 55]
layer_depth_pct = {15: 25, 31: 50, 37: 60, 43: 70, 49: 80, 55: 90}

print("\n=== Activation norm statistics per layer ===")
print(f"{'Layer':<8} {'Depth%':<8} {'Mean norm':<12} {'Std':<12} {'Min':<12} {'Max':<12} {'Median':<12} {'P5':<12} {'P95':<12}")
print("-" * 100)

layer_norms = {}
for layer in layers:
    key = f"layer_{layer}"
    acts = data[key]  # shape: (n_tasks, hidden_dim)
    norms = np.linalg.norm(acts, axis=1)
    layer_norms[layer] = norms
    print(
        f"{layer:<8} {layer_depth_pct[layer]:<8} {norms.mean():<12.1f} {norms.std():<12.1f} "
        f"{norms.min():<12.1f} {norms.max():<12.1f} {np.median(norms):<12.1f} "
        f"{np.percentile(norms, 5):<12.1f} {np.percentile(norms, 95):<12.1f}"
    )

# Focus on L31 (the steering layer)
print("\n=== Layer 31 detailed analysis ===")
norms_31 = layer_norms[31]
print(f"Mean activation norm: {norms_31.mean():.1f}")
print(f"Std: {norms_31.std():.1f}")
print(f"Coefficient of variation: {norms_31.std() / norms_31.mean():.3f}")

# Load probe direction and compute projections
probe_weights_path = Path(PROBE_DIR) / "probes" / "probe_ridge_L31.npy"
if probe_weights_path.exists():
    weights = np.load(probe_weights_path)
    direction = weights[:-1]  # strip intercept
    direction = direction / np.linalg.norm(direction)  # unit vector

    print(f"\nProbe direction norm (should be 1.0): {np.linalg.norm(direction):.6f}")
    print(f"Probe dimension: {len(direction)}")

    # Project activations onto probe direction
    acts_31 = data["layer_31"]
    projections = acts_31 @ direction  # scalar projection per task
    print(f"\n=== Projection of activations onto probe direction ===")
    print(f"Mean projection: {projections.mean():.1f}")
    print(f"Std projection: {projections.std():.1f}")
    print(f"Min: {projections.min():.1f}, Max: {projections.max():.1f}")
    print(f"P5: {np.percentile(projections, 5):.1f}, P95: {np.percentile(projections, 95):.1f}")

    # What coefficient values mean relative to natural variation
    print(f"\n=== Coefficient interpretation ===")
    natural_range = projections.std()
    print(f"Natural std of projections onto probe: {natural_range:.1f}")
    print(f"So a coefficient of X shifts the projection by X (since direction is unit)")
    print(f"")
    print(f"{'Coefficient':<15} {'As % of mean norm':<22} {'As multiples of proj std':<28} {'As % of proj range (P5-P95)'}")
    print("-" * 90)
    proj_range = np.percentile(projections, 95) - np.percentile(projections, 5)
    for coef in [10, 50, 100, 500, 1000, 1500, 3000, 5000, 10000]:
        pct_norm = 100 * coef / norms_31.mean()
        multiples_std = coef / natural_range
        pct_range = 100 * coef / proj_range
        print(f"{coef:<15} {pct_norm:<22.1f} {multiples_std:<28.1f} {pct_range:.1f}")
else:
    print(f"\nProbe weights not found at {probe_weights_path}")
    # Try to find them
    import glob
    candidates = glob.glob(str(Path(PROBE_DIR) / "*L31*"))
    print(f"Available files matching L31: {candidates}")
