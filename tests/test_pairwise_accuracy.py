"""Integration tests for pairwise accuracy code paths.

All tests use synthetic data: a true preference direction generates
deterministic activations, measurements, and scores.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.probes.bradley_terry.data import PairwiseActivationData
from src.probes.bradley_terry.training import pairwise_accuracy_from_scores
from src.probes.experiments.eval_on_heldout import heldout_eval
from src.probes.experiments.hoo_ridge import build_ridge_xy
from src.probes.experiments.run_dir_probes import _cv_pairwise_acc_ridge
from src.task_data import Task, OriginDataset
from src.types import BinaryPreferenceMeasurement, PreferenceType

pytestmark = pytest.mark.probes


@pytest.fixture
def synthetic_probe_data():
    rng = np.random.default_rng(42)
    n_tasks = 80
    d_model = 16

    true_direction = rng.standard_normal(d_model)
    true_direction /= np.linalg.norm(true_direction)

    activations_raw = rng.standard_normal((n_tasks, d_model))
    true_utilities = activations_raw @ true_direction

    task_ids = np.array([f"t{i}" for i in range(n_tasks)])
    tasks = [
        Task(id=f"t{i}", prompt=f"task {i}", origin=OriginDataset.SYNTHETIC, metadata={})
        for i in range(n_tasks)
    ]
    activations = {0: activations_raw}

    # Deterministic pairwise measurements
    measurements = []
    for _ in range(200):
        i, j = rng.choice(n_tasks, size=2, replace=False)
        choice = "a" if true_utilities[i] > true_utilities[j] else "b"
        measurements.append(
            BinaryPreferenceMeasurement(
                task_a=tasks[i],
                task_b=tasks[j],
                choice=choice,
                preference_type=PreferenceType.PRE_TASK_REVEALED,
            )
        )

    # Thurstonian-like scores (true utilities + small noise)
    scores = {f"t{i}": float(true_utilities[i] + rng.normal(0, 0.1)) for i in range(n_tasks)}

    bt_data = PairwiseActivationData.from_measurements(measurements, task_ids, activations)

    return {
        "task_ids": task_ids,
        "activations": activations,
        "activations_raw": activations_raw,
        "measurements": measurements,
        "scores": scores,
        "true_direction": true_direction,
        "true_utilities": true_utilities,
        "bt_data": bt_data,
        "n_tasks": n_tasks,
        "d_model": d_model,
    }


def test_filter_by_indices(synthetic_probe_data):
    bt_data = synthetic_probe_data["bt_data"]
    n_tasks = synthetic_probe_data["n_tasks"]

    # Keep only first half of tasks
    idx_set = set(range(n_tasks // 2))
    filtered = bt_data.filter_by_indices(idx_set)

    # All pairs should have both indices in the set
    for i, j in filtered.pairs:
        assert int(i) in idx_set
        assert int(j) in idx_set

    # Filtered should have fewer pairs
    assert len(filtered.pairs) <= len(bt_data.pairs)
    assert len(filtered.pairs) > 0

    # n_measurements should be consistent
    assert filtered.n_measurements == int(np.sum(filtered.total))

    # Empty set should give empty result
    empty = bt_data.filter_by_indices(set())
    assert len(empty.pairs) == 0
    assert empty.n_measurements == 0


def test_cv_pairwise_acc_ridge_above_chance(synthetic_probe_data):
    task_ids = synthetic_probe_data["task_ids"]
    scores = synthetic_probe_data["scores"]
    bt_data = synthetic_probe_data["bt_data"]
    activations_raw = synthetic_probe_data["activations_raw"]

    indices, y = build_ridge_xy(task_ids, scores)
    X = activations_raw[indices]

    mean_acc, std_acc = _cv_pairwise_acc_ridge(
        X=X,
        y=y,
        row_indices=indices,
        pairwise_data=bt_data,
        activations=activations_raw,
        best_alpha=1.0,
        cv_folds=5,
        standardize=True,
    )

    assert mean_acc > 0.6, f"Expected pairwise acc > 0.6, got {mean_acc}"
    assert std_acc >= 0.0


def test_heldout_eval_end_to_end(synthetic_probe_data):
    scores = synthetic_probe_data["scores"]
    task_ids = synthetic_probe_data["task_ids"]
    activations = synthetic_probe_data["activations"]
    measurements = synthetic_probe_data["measurements"]
    n_tasks = synthetic_probe_data["n_tasks"]

    # Split scores into train/eval (first 50 train, rest eval)
    train_scores = {f"t{i}": scores[f"t{i}"] for i in range(50)}
    eval_scores = {f"t{i}": scores[f"t{i}"] for i in range(50, n_tasks)}

    results = heldout_eval(
        train_scores=train_scores,
        eval_scores=eval_scores,
        task_ids_arr=task_ids,
        activations=activations,
        eval_measurements=measurements,
        layers=[0],
        standardize=True,
        alpha_sweep_size=5,
        eval_split_seed=42,
    )

    layer_result = results[0]
    assert "final_r" in layer_result
    assert "final_acc" in layer_result
    assert "best_alpha" in layer_result
    assert "alpha_sweep" in layer_result

    # Probe should transfer with reasonable correlation
    if layer_result["final_r"] is not None:
        assert layer_result["final_r"] > 0.3, f"Expected final_r > 0.3, got {layer_result['final_r']}"


def test_hoo_ridge_returns_hoo_acc(synthetic_probe_data):
    from src.probes.experiments.run_dir_probes import RunDirProbeConfig, ProbeMode
    from src.probes.experiments import hoo_ridge

    scores = synthetic_probe_data["scores"]
    task_ids = synthetic_probe_data["task_ids"]
    activations = synthetic_probe_data["activations"]
    bt_data = synthetic_probe_data["bt_data"]
    n_tasks = synthetic_probe_data["n_tasks"]

    # Assign groups: first half = "group_a", second half = "group_b"
    task_groups = {}
    scored_and_grouped = set()
    for i in range(n_tasks):
        tid = f"t{i}"
        task_groups[tid] = "group_a" if i < n_tasks // 2 else "group_b"
        scored_and_grouped.add(tid)

    config = RunDirProbeConfig(
        experiment_name="test",
        run_dir=Path("/tmp/test"),  # not used
        activations_path=Path("/tmp/test"),  # not used
        output_dir=Path("/tmp/test_output"),
        layers=[0],
        modes=[ProbeMode.RIDGE],
        cv_folds=3,
        alpha_sweep_size=5,
    )

    method = hoo_ridge.make_method(
        fold_idx=0,
        config=config,
        task_ids=task_ids,
        activations=activations,
        scores=scores,
        task_groups=task_groups,
        scored_and_grouped=scored_and_grouped,
        held_out_set={"group_b"},
        best_hp=None,
        bt_data=bt_data,
    )

    assert method is not None

    weights, hp = method.train(0, None)
    assert hp is not None
    assert len(weights) == synthetic_probe_data["d_model"] + 1  # coef + intercept

    metrics = method.evaluate(0, weights)
    assert "hoo_r" in metrics
    assert "hoo_acc" in metrics
    assert metrics["hoo_acc"] > 0.5, f"Expected hoo_acc > 0.5, got {metrics['hoo_acc']}"
