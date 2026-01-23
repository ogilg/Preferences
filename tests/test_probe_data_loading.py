"""Test probe training data loading."""
import time
from pathlib import Path

import pytest

from src.measurement_storage.loading import load_raw_scores
from src.probes.activations import load_task_origins, load_activations
from src.probes.config import ProbeTrainingConfig


@pytest.mark.slow
def test_data_filtering_order():
    """Test that we filter measurements -> completions -> activations in correct order."""
    config = ProbeTrainingConfig.from_yaml(Path("configs/probe_training/test_new_config.yaml"))
    origins_cache = load_task_origins(Path("probe_data/activations"))

    print("\n=== Data Loading Order Test ===\n")

    # Step 1: Load measurements
    print("Step 1: Loading measurements...")
    t0 = time.time()
    all_measurements = []
    templates = ["post_task_qualitative_001"]
    response_formats = ["tool_use"]
    seeds = [0, 1]

    for template in templates:
        task_type = "pre_task" if template.startswith("pre_task") else "post_task"
        measurement_dir = config.experiment_dir / f"{task_type}_stated"
        measurements = load_raw_scores(
            measurement_dir,
            [template],
            response_formats,
            seeds,
        )
        all_measurements.extend(measurements)

    elapsed = time.time() - t0
    print(f"  Loaded {len(all_measurements)} measurements in {elapsed:.2f}s")

    # Step 2: Get task IDs from measurements
    print("\nStep 2: Extracting task IDs from measurements...")
    t0 = time.time()
    measurement_task_ids = set(tid for tid, _ in all_measurements)
    elapsed = time.time() - t0
    print(f"  Found {len(measurement_task_ids)} unique tasks in {elapsed:.3f}s")

    # Step 3: Filter by dataset
    print("\nStep 3: Filtering by dataset...")
    datasets = ["wildchat", "alpaca", "math", "bailbench"]
    t0 = time.time()
    target_task_ids = set()
    for dataset in datasets:
        dataset_task_ids = origins_cache.get(dataset.upper(), set())
        target_task_ids.update(dataset_task_ids)
        print(f"  {dataset.upper()}: {len(dataset_task_ids)} tasks")

    # Keep only tasks that have both measurements AND are in target datasets
    filtered_task_ids = measurement_task_ids & target_task_ids
    elapsed = time.time() - t0
    print(f"  After filtering: {len(filtered_task_ids)} tasks in {elapsed:.3f}s")

    # Step 4: Load activations (only what we need)
    print("\nStep 4: Loading activations...")
    t0 = time.time()
    all_task_ids, all_activations = load_activations(Path("probe_data/activations"))
    elapsed = time.time() - t0
    print(f"  Loaded all activations ({len(all_task_ids)} tasks) in {elapsed:.2f}s")

    # Step 5: Slice to only what we need
    print("\nStep 5: Slicing activations to needed tasks...")
    t0 = time.time()
    import numpy as np
    mask = np.array([tid in filtered_task_ids for tid in all_task_ids])
    sliced_activations = {l: a[mask] for l, a in all_activations.items()}
    elapsed = time.time() - t0
    print(f"  Sliced to {mask.sum()} tasks in {elapsed:.3f}s")

    print(f"\n=== Summary ===")
    print(f"Measurement instances: {len(all_measurements)}")
    print(f"Unique tasks with measurements: {len(measurement_task_ids)}")
    print(f"Tasks in target datasets: {len(target_task_ids)}")
    print(f"Tasks with both measurements AND in datasets: {len(filtered_task_ids)}")
    print(f"Activation shape per layer: {list(sliced_activations.values())[0].shape}")
    print(f"\nData ready for training!")


if __name__ == "__main__":
    test_data_filtering_order()
