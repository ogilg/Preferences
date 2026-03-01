"""Run Phases 2 and 3 with focused coefficient sets."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

# Import from main experiment script
from run_experiment import (
    load_hf_model, load_probe, compute_mean_norm,
    load_pairs, load_template, build_prompt_builder,
    run_preference_sweep, compile_results,
    EXP_DIR, COHERENCE_RESULTS_PATH,
)

# Phase 2: Key coefficients spanning the useful range
# Based on Phase 1: clear signal at ±0.02-0.05, inverted-U beyond ±0.05
PHASE2_MULTIPLIERS = [-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10]

# Phase 3: Smaller set for random control (enough to test for direction specificity)
PHASE3_MULTIPLIERS = [-0.05, 0.0, 0.05]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, default=0, help="2=probe, 3=random, 0=both")
    args = parser.parse_args()

    hf_model = load_hf_model()
    layer, direction = load_probe()
    mean_norm = compute_mean_norm(layer)
    print(f"Mean activation norm at L{layer}: {mean_norm:.2f}")

    pairs = load_pairs()
    template = load_template()
    builder = build_prompt_builder(template)
    print(f"Loaded {len(pairs)} pairs")

    if args.phase in [0, 2]:
        print("\n" + "=" * 60)
        print(f"PHASE 2: PROBE SWEEP ({len(PHASE2_MULTIPLIERS)} multipliers × {len(pairs)} pairs)")
        print("=" * 60)
        t0 = time.time()
        probe_records = run_preference_sweep(
            hf_model, layer, direction, mean_norm, pairs, builder,
            multipliers=PHASE2_MULTIPLIERS, condition="probe",
        )
        print(f"Phase 2 done in {(time.time() - t0) / 60:.1f}m — {len(probe_records)} records")

    if args.phase in [0, 3]:
        print("\n" + "=" * 60)
        print(f"PHASE 3: RANDOM CONTROL ({len(PHASE3_MULTIPLIERS)} multipliers × {len(pairs)} pairs)")
        print("=" * 60)
        rng = np.random.default_rng(42)
        random_direction = rng.standard_normal(direction.shape)
        random_direction = random_direction / np.linalg.norm(random_direction)
        t0 = time.time()
        random_records = run_preference_sweep(
            hf_model, layer, direction, mean_norm, pairs, builder,
            multipliers=PHASE3_MULTIPLIERS, condition="random",
            direction_override=random_direction,
        )
        print(f"Phase 3 done in {(time.time() - t0) / 60:.1f}m — {len(random_records)} records")

    compile_results()
    print("\nPhases 2+3 complete.")


if __name__ == "__main__":
    main()
