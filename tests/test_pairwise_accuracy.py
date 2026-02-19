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
from src.probes.experiments.hoo_ridge import build_ridge_xy
from src.probes.experiments.run_dir_probes import (
    RunDirProbeConfig, ProbeMode, _cv_pairwise_acc_ridge, _train_ridge_probe,
)
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


def test_heldout_eval_end_to_end(synthetic_probe_data, tmp_path):
    scores = synthetic_probe_data["scores"]
    task_ids = synthetic_probe_data["task_ids"]
    activations_raw = synthetic_probe_data["activations_raw"]
    measurements = synthetic_probe_data["measurements"]
    n_tasks = synthetic_probe_data["n_tasks"]

    # Split scores into train/eval (first 50 train, rest eval)
    train_scores = {f"t{i}": scores[f"t{i}"] for i in range(50)}
    eval_scores = {f"t{i}": scores[f"t{i}"] for i in range(50, n_tasks)}

    config = RunDirProbeConfig(
        experiment_name="test_heldout",
        run_dir=Path("/tmp/test"),
        activations_path=Path("/tmp/test"),
        output_dir=tmp_path,
        layers=[0],
        modes=[ProbeMode.RIDGE],
        alpha_sweep_size=5,
        standardize=True,
    )

    result = _train_ridge_probe(
        config, layer=0, task_ids=task_ids, activations=activations_raw,
        scores=train_scores,
        eval_scores=eval_scores,
        eval_measurements=measurements,
        eval_split_seed=42,
    )

    assert result is not None
    assert "final_r" in result
    assert "final_acc" in result
    assert "best_alpha" in result
    assert "sweep_r" in result
    assert "alpha_sweep" in result

    # Probe should transfer with reasonable correlation
    if result["final_r"] is not None:
        assert result["final_r"] > 0.3, f"Expected final_r > 0.3, got {result['final_r']}"


def test_heldout_splits_are_disjoint_and_correct_size(tmp_path):
    """Verify train/sweep/final sets are disjoint and sizes are correct."""
    rng = np.random.default_rng(99)
    n_train, n_eval = 60, 40
    n_total = n_train + n_eval
    d_model = 8

    true_dir = rng.standard_normal(d_model)
    true_dir /= np.linalg.norm(true_dir)
    activations_raw = rng.standard_normal((n_total, d_model))
    true_utilities = activations_raw @ true_dir

    task_ids = np.array([f"t{i}" for i in range(n_total)])
    tasks = [
        Task(id=f"t{i}", prompt=f"task {i}", origin=OriginDataset.SYNTHETIC, metadata={})
        for i in range(n_total)
    ]

    train_scores = {f"t{i}": float(true_utilities[i]) for i in range(n_train)}
    eval_scores = {f"t{i}": float(true_utilities[i]) for i in range(n_train, n_total)}

    # Measurements only among eval tasks
    measurements = []
    for _ in range(100):
        i, j = rng.choice(range(n_train, n_total), size=2, replace=False)
        choice = "a" if true_utilities[i] > true_utilities[j] else "b"
        measurements.append(BinaryPreferenceMeasurement(
            task_a=tasks[i], task_b=tasks[j], choice=choice,
            preference_type=PreferenceType.PRE_TASK_REVEALED,
        ))

    config = RunDirProbeConfig(
        experiment_name="test_splits",
        run_dir=Path("/tmp/test"),
        activations_path=Path("/tmp/test"),
        output_dir=tmp_path,
        layers=[0],
        modes=[ProbeMode.RIDGE],
        alpha_sweep_size=5,
        standardize=True,
    )

    result = _train_ridge_probe(
        config, layer=0, task_ids=task_ids, activations=activations_raw,
        scores=train_scores,
        eval_scores=eval_scores,
        eval_measurements=measurements,
        eval_split_seed=42,
    )

    assert result is not None
    # Train set is exactly the train_scores keys
    assert result["n_train"] == n_train
    # Sweep + final = all eval tasks
    assert result["n_sweep"] + result["n_final"] == n_eval
    # Sweep and final are roughly equal halves
    assert result["n_sweep"] == n_eval // 2
    assert result["n_final"] == n_eval - n_eval // 2


def test_heldout_probe_learns_train_direction_not_eval(tmp_path):
    """Probe trained on train scores should reflect the train direction, not eval.

    Uses orthogonal directions for train vs eval scores so leaking eval data
    into training would produce a detectably different probe.
    """
    rng = np.random.default_rng(123)
    n_train, n_eval = 80, 60
    n_total = n_train + n_eval
    d_model = 32

    # Two orthogonal directions
    dir_train = rng.standard_normal(d_model)
    dir_train /= np.linalg.norm(dir_train)
    # Gram-Schmidt to get orthogonal eval direction
    dir_eval = rng.standard_normal(d_model)
    dir_eval -= dir_eval @ dir_train * dir_train
    dir_eval /= np.linalg.norm(dir_eval)
    assert abs(dir_train @ dir_eval) < 1e-10

    activations_raw = rng.standard_normal((n_total, d_model))

    task_ids = np.array([f"t{i}" for i in range(n_total)])
    tasks = [
        Task(id=f"t{i}", prompt=f"task {i}", origin=OriginDataset.SYNTHETIC, metadata={})
        for i in range(n_total)
    ]

    # Train scores follow dir_train, eval scores follow dir_eval
    train_scores = {
        f"t{i}": float(activations_raw[i] @ dir_train)
        for i in range(n_train)
    }
    eval_scores = {
        f"t{i}": float(activations_raw[i] @ dir_eval)
        for i in range(n_train, n_total)
    }

    # Measurements among eval tasks, consistent with dir_eval
    measurements = []
    for _ in range(200):
        i, j = rng.choice(range(n_train, n_total), size=2, replace=False)
        u_i = activations_raw[i] @ dir_eval
        u_j = activations_raw[j] @ dir_eval
        choice = "a" if u_i > u_j else "b"
        measurements.append(BinaryPreferenceMeasurement(
            task_a=tasks[i], task_b=tasks[j], choice=choice,
            preference_type=PreferenceType.PRE_TASK_REVEALED,
        ))

    config = RunDirProbeConfig(
        experiment_name="test_direction",
        run_dir=Path("/tmp/test"),
        activations_path=Path("/tmp/test"),
        output_dir=tmp_path,
        layers=[0],
        modes=[ProbeMode.RIDGE],
        alpha_sweep_size=5,
        standardize=True,
    )

    result = _train_ridge_probe(
        config, layer=0, task_ids=task_ids, activations=activations_raw,
        scores=train_scores,
        eval_scores=eval_scores,
        eval_measurements=measurements,
        eval_split_seed=42,
    )

    assert result is not None

    # Load the saved probe weights to check the learned direction
    probe_file = tmp_path / "probes" / "probe_ridge_L00.npy"
    weights = np.load(probe_file)
    learned_dir = weights[:-1]  # exclude intercept
    learned_dir /= np.linalg.norm(learned_dir)

    # Probe should align with train direction, not eval direction
    cos_train = abs(learned_dir @ dir_train)
    cos_eval = abs(learned_dir @ dir_eval)
    assert cos_train > 0.7, f"Probe should align with train dir (cos={cos_train:.3f})"
    assert cos_train > cos_eval * 2, (
        f"Probe aligns more with eval dir than train dir: "
        f"cos_train={cos_train:.3f}, cos_eval={cos_eval:.3f}"
    )

    # final_r should be low since probe was trained on a different direction
    # than eval scores — this confirms eval data wasn't leaked into training
    assert result["final_r"] is not None
    assert result["final_r"] < 0.5, (
        f"final_r={result['final_r']:.3f} is suspiciously high — "
        f"probe may be leaking eval data into training"
    )


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
