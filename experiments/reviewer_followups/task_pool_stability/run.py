#!/usr/bin/env python3
"""Run the leave-one-dataset-out task-pool stability analysis."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.fitting.thurstonian_fitting import PairwiseData, fit_thurstonian
from src.task_data import OriginDataset, Task
from src.types import BinaryPreferenceMeasurement, PreferenceType


TRAIN_RUN = ROOT / (
    "results/experiments/persona_sweep_final_six/pre_task_active_learning/"
    "completion_preference_gemma-3-27b_completion_canonical_seed0_train_task_ids"
)
EVAL_RUN = ROOT / (
    "results/experiments/persona_sweep_final_six/pre_task_active_learning/"
    "completion_preference_gemma-3-27b_completion_canonical_seed0_eval_task_ids"
)
ACTIVATIONS = ROOT / (
    "activations/gemma-3-27b_it/pref_layer_sweep/activations_eot_L23_L32.npz"
)
OUTPUT_DIR = Path(__file__).resolve().parent

DATASETS = ("WILDCHAT", "ALPACA", "MATH", "BAILBENCH", "STRESS_TEST")
OMISSIONS: tuple[str | None, ...] = (
    None,
    "WILDCHAT",
    "ALPACA",
    "MATH",
    "BAILBENCH",
    "STRESS_TEST",
)
ALPHAS = np.logspace(-1, 5, 10)

EXPECTED_COUNTS = {
    None: (4000, 37196, 0, 1000),
    "ALPACA": (3000, 20852, 1, 750),
    "BAILBENCH": (3600, 30656, 0, 899),
    "MATH": (2996, 21514, 4, 750),
    "STRESS_TEST": (3398, 27081, 0, 850),
    "WILDCHAT": (3001, 21072, 0, 751),
}


def require_one_csv(run_dir: Path) -> Path:
    matches = sorted(run_dir.glob("thurstonian_*.csv"))
    if len(matches) != 1:
        raise AssertionError(
            f"Expected exactly one thurstonian_*.csv in {run_dir}, found {len(matches)}"
        )
    return matches[0]


def load_utilities(path: Path, expected_count: int) -> dict[str, float]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    task_ids = [row["task_id"] for row in rows]
    if len(task_ids) != expected_count or len(set(task_ids)) != expected_count:
        raise AssertionError(
            f"{path} must have {expected_count} rows with unique task IDs; "
            f"found {len(task_ids)} rows and {len(set(task_ids))} unique IDs"
        )
    utilities = {row["task_id"]: float(row["mu"]) for row in rows}
    if not np.isfinite(list(utilities.values())).all():
        raise AssertionError(f"{path} contains non-finite utilities")
    return utilities


def load_measurements(
    path: Path,
) -> tuple[list[BinaryPreferenceMeasurement], dict[str, str]]:
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    with path.open() as handle:
        raw_measurements = yaml.load(handle, Loader=loader)
    if not isinstance(raw_measurements, list):
        raise AssertionError(f"{path} did not contain a measurement list")

    origins: dict[str, str] = {}
    tasks: dict[str, Task] = {}
    measurements: list[BinaryPreferenceMeasurement] = []

    def get_task(task_id: str, origin_name: str) -> Task:
        if origin_name not in DATASETS:
            raise AssertionError(f"Unexpected origin {origin_name!r} for task {task_id}")
        previous = origins.setdefault(task_id, origin_name)
        if previous != origin_name:
            raise AssertionError(
                f"Conflicting origins for {task_id}: {previous!r} and {origin_name!r}"
            )
        if task_id not in tasks:
            tasks[task_id] = Task(
                id=task_id,
                prompt="",
                origin=OriginDataset[origin_name],
                metadata={},
            )
        return tasks[task_id]

    for row in raw_measurements:
        task_a = get_task(row["task_a"], row["origin_a"])
        task_b = get_task(row["task_b"], row["origin_b"])
        choice = row["choice"]
        if choice not in {"a", "b", "refusal"}:
            raise AssertionError(f"Unexpected stored choice: {choice!r}")
        measurements.append(
            BinaryPreferenceMeasurement(
                task_a=task_a,
                task_b=task_b,
                choice=choice,
                preference_type=PreferenceType.POST_TASK_REVEALED,
            )
        )
    return measurements, origins


def load_origins(path: Path) -> dict[str, str]:
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    with path.open() as handle:
        raw_measurements = yaml.load(handle, Loader=loader)
    if not isinstance(raw_measurements, list):
        raise AssertionError(f"{path} did not contain a measurement list")

    origins: dict[str, str] = {}
    for row in raw_measurements:
        for endpoint in ("a", "b"):
            task_id = row[f"task_{endpoint}"]
            origin = row[f"origin_{endpoint}"]
            if origin not in DATASETS:
                raise AssertionError(f"Unexpected origin {origin!r} for task {task_id}")
            previous = origins.setdefault(task_id, origin)
            if previous != origin:
                raise AssertionError(
                    f"Conflicting origins for {task_id}: {previous!r} and {origin!r}"
                )
    return origins


def assert_connected(
    task_ids: list[str], comparisons: list[BinaryPreferenceMeasurement]
) -> None:
    parent = {task_id: task_id for task_id in task_ids}

    def find(task_id: str) -> str:
        while parent[task_id] != task_id:
            parent[task_id] = parent[parent[task_id]]
            task_id = parent[task_id]
        return task_id

    def union(task_a: str, task_b: str) -> None:
        root_a, root_b = find(task_a), find(task_b)
        if root_a != root_b:
            parent[root_b] = root_a

    for comparison in comparisons:
        union(comparison.task_a.id, comparison.task_b.id)
    roots = {find(task_id) for task_id in task_ids}
    if len(roots) != 1:
        sizes = Counter(find(task_id) for task_id in task_ids)
        raise AssertionError(
            f"Comparison graph has {len(roots)} components with sizes "
            f"{sorted(sizes.values(), reverse=True)[:10]}"
        )


def correlation(x: np.ndarray, y: np.ndarray, label: str) -> float:
    if len(x) != len(y) or len(x) < 2:
        raise AssertionError(f"Invalid inputs for {label}: {len(x)} and {len(y)}")
    result = float(pearsonr(x, y).statistic)
    if not np.isfinite(result):
        raise AssertionError(f"{label} is non-finite")
    return result


def raw_probe_direction(
    train_ids: list[str],
    train_utilities: dict[str, float],
    eval_ids: list[str],
    eval_utilities: dict[str, float],
    activations: np.ndarray,
    activation_row: dict[str, int],
) -> tuple[float, float, np.ndarray]:
    missing = (set(train_ids) | set(eval_ids)) - activation_row.keys()
    if missing:
        raise AssertionError(
            f"{len(missing)} probe tasks lack activations; examples: {sorted(missing)[:5]}"
        )

    train_rows = np.fromiter(
        (activation_row[task_id] for task_id in train_ids),
        dtype=np.int64,
        count=len(train_ids),
    )
    eval_rows = np.fromiter(
        (activation_row[task_id] for task_id in eval_ids),
        dtype=np.int64,
        count=len(eval_ids),
    )
    x_train = activations[train_rows]
    x_eval = activations[eval_rows]
    y_train = np.asarray([train_utilities[task_id] for task_id in train_ids])
    y_eval = np.asarray([eval_utilities[task_id] for task_id in eval_ids])

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_eval_scaled = scaler.transform(x_eval)
    if not np.isfinite(x_train_scaled).all() or not np.isfinite(x_eval_scaled).all():
        raise AssertionError("Scaled activations contain non-finite values")

    best_alpha: float | None = None
    best_eval_r = -np.inf
    for alpha in ALPHAS:
        ridge = Ridge(alpha=float(alpha), fit_intercept=True)
        ridge.fit(x_train_scaled, y_train)
        eval_r = correlation(
            ridge.predict(x_eval_scaled), y_eval, f"eval r at alpha={alpha}"
        )
        # ALPHAS is increasing, so retaining the first exact tie selects the smaller alpha.
        if eval_r > best_eval_r:
            best_alpha = float(alpha)
            best_eval_r = eval_r

    if best_alpha is None:
        raise AssertionError("No ridge alpha was selected")
    selected_ridge = Ridge(alpha=best_alpha, fit_intercept=True)
    selected_ridge.fit(x_train_scaled, y_train)
    w_raw = np.asarray(selected_ridge.coef_, dtype=np.float64) / scaler.scale_
    if w_raw.shape != (activations.shape[1],) or not np.isfinite(w_raw).all():
        raise AssertionError(f"Invalid raw probe direction with shape {w_raw.shape}")
    if np.linalg.norm(w_raw) == 0:
        raise AssertionError("Raw probe direction has zero norm")
    return best_alpha, best_eval_r, w_raw


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    value = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    if not np.isfinite(value):
        raise AssertionError("Probe cosine is non-finite")
    return value


def write_outputs(results: list[dict[str, Any]]) -> None:
    fields = [
        "condition",
        "omitted_dataset",
        "n_train_tasks",
        "n_train_comparisons",
        "n_zero_degree_removed",
        "n_eval_tasks",
        "converged",
        "best_alpha",
        "eval_r",
        "utility_r",
        "probe_cosine",
    ]
    with (OUTPUT_DIR / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    payload = {
        "inputs": {
            "train_run": str(TRAIN_RUN.relative_to(ROOT)),
            "eval_run": str(EVAL_RUN.relative_to(ROOT)),
            "activations": str(ACTIVATIONS.relative_to(ROOT)),
            "activation_key": "layer_32",
        },
        "alphas": [float(alpha) for alpha in ALPHAS],
        "results": results,
    }
    (OUTPUT_DIR / "results.json").write_text(json.dumps(payload, indent=2) + "\n")

    ablated = results[1:]
    min_utility = min(ablated, key=lambda row: row["utility_r"])
    min_cosine = min(ablated, key=lambda row: row["probe_cosine"])
    report_lines = [
        "# Leave-one-dataset-out task-pool stability",
        "",
        "| Condition | Omitted dataset | Train tasks | Comparisons | Zero-degree removed | Eval tasks | Converged | Best alpha | Eval r | Utility r | Probe cosine |",
        "|---|---|---:|---:|---:|---:|:---:|---:|---:|---:|---:|",
    ]
    for row in results:
        report_lines.append(
            f"| {row['condition']} | {row['omitted_dataset'] or '—'} | "
            f"{row['n_train_tasks']:,} | {row['n_train_comparisons']:,} | "
            f"{row['n_zero_degree_removed']} | {row['n_eval_tasks']:,} | "
            f"{'yes' if row['converged'] else 'no'} | {row['best_alpha']:.6g} | "
            f"{row['eval_r']:.6f} | {row['utility_r']:.6f} | "
            f"{row['probe_cosine']:.6f} |"
        )
    report_lines.extend(
        [
            "",
            (
                "Across all five leave-one-dataset-out refits, the minimum utility "
                f"correlation with the original fit was **{min_utility['utility_r']:.6f}** "
                f"(omitting {min_utility['omitted_dataset']}) and the minimum signed "
                f"cosine similarity of the raw-coordinate L32 ridge direction was "
                f"**{min_cosine['probe_cosine']:.6f}** "
                f"(omitting {min_cosine['omitted_dataset']}). These results directly "
                "measure whether either the inferred task utilities or the learned "
                "probe direction depends materially on any one source dataset, while "
                "preserving the stored comparisons and evaluating alpha selection only "
                "on retained-origin eval tasks."
            ),
            "",
        ]
    )
    (OUTPUT_DIR / "report.md").write_text("\n".join(report_lines))


def main() -> None:
    train_csv = require_one_csv(TRAIN_RUN)
    eval_csv = require_one_csv(EVAL_RUN)
    baseline_utilities = load_utilities(train_csv, 4000)
    eval_utilities = load_utilities(eval_csv, 1000)

    fit_metadata_path = train_csv.with_suffix(".yaml")
    if not fit_metadata_path.exists():
        raise AssertionError(f"Missing baseline fit metadata: {fit_metadata_path}")
    with fit_metadata_path.open() as handle:
        baseline_fit_metadata = yaml.safe_load(handle)
    if not baseline_fit_metadata.get("converged", False):
        raise AssertionError("Existing baseline Thurstonian fit is not converged")

    print("Loading train measurements...", flush=True)
    train_measurements, train_origins = load_measurements(
        TRAIN_RUN / "measurements.yaml"
    )
    print("Loading eval origins...", flush=True)
    eval_origins = load_origins(EVAL_RUN / "measurements.yaml")
    if set(baseline_utilities) != set(train_origins):
        raise AssertionError(
            "Train utility IDs and train measurement-origin IDs do not match: "
            f"{len(set(baseline_utilities) - set(train_origins))} utilities lack origins, "
            f"{len(set(train_origins) - set(baseline_utilities))} origins lack utilities"
        )
    if set(eval_utilities) != set(eval_origins):
        raise AssertionError(
            "Eval utility IDs and eval measurement-origin IDs do not match: "
            f"{len(set(eval_utilities) - set(eval_origins))} utilities lack origins, "
            f"{len(set(eval_origins) - set(eval_utilities))} origins lack utilities"
        )
    if set(train_origins.values()) != set(DATASETS):
        raise AssertionError(f"Unexpected train origin labels: {set(train_origins.values())}")
    if set(eval_origins.values()) != set(DATASETS):
        raise AssertionError(f"Unexpected eval origin labels: {set(eval_origins.values())}")

    print("Loading L32 activations...", flush=True)
    with np.load(ACTIVATIONS) as archive:
        activation_ids = archive["task_ids"].tolist()
        activations = archive["layer_32"]
    if len(activation_ids) != 6000 or len(set(activation_ids)) != 6000:
        raise AssertionError(
            "ACTIVATIONS must contain exactly 6,000 unique task IDs; "
            f"found {len(activation_ids)} rows and {len(set(activation_ids))} unique IDs"
        )
    if activations.shape != (6000, 5376):
        raise AssertionError(
            f"layer_32 must have shape (6000, 5376), found {activations.shape}"
        )
    if not np.isfinite(activations).all():
        raise AssertionError("layer_32 contains non-finite activations")
    activation_row = {
        task_id: row_index for row_index, task_id in enumerate(activation_ids)
    }

    condition_data: dict[
        str | None,
        tuple[
            list[str],
            list[BinaryPreferenceMeasurement],
            int,
            list[str],
            dict[str, float],
            bool,
        ],
    ] = {}
    for omitted in OMISSIONS:
        retained_before_degree = {
            task_id
            for task_id, origin in train_origins.items()
            if omitted is None or origin != omitted
        }
        retained_comparisons = [
            measurement
            for measurement in train_measurements
            if measurement.task_a.id in retained_before_degree
            and measurement.task_b.id in retained_before_degree
        ]
        degree = Counter()
        for measurement in retained_comparisons:
            degree[measurement.task_a.id] += 1
            degree[measurement.task_b.id] += 1
        retained_ids = sorted(task_id for task_id in retained_before_degree if degree[task_id])
        zero_degree_removed = len(retained_before_degree) - len(retained_ids)
        eval_ids = sorted(
            task_id
            for task_id, origin in eval_origins.items()
            if omitted is None or origin != omitted
        )

        observed = (
            len(retained_ids),
            len(retained_comparisons),
            zero_degree_removed,
            len(eval_ids),
        )
        if observed != EXPECTED_COUNTS[omitted]:
            raise AssertionError(
                f"Count check failed for omitted={omitted!r}: "
                f"expected {EXPECTED_COUNTS[omitted]}, observed {observed}"
            )
        assert_connected(retained_ids, retained_comparisons)

        if omitted is None:
            utilities = baseline_utilities
            converged = True
        else:
            retained_id_set = set(retained_ids)
            tasks_by_id = {
                measurement.task_a.id: measurement.task_a
                for measurement in retained_comparisons
                if measurement.task_a.id in retained_id_set
            }
            tasks_by_id.update(
                {
                    measurement.task_b.id: measurement.task_b
                    for measurement in retained_comparisons
                    if measurement.task_b.id in retained_id_set
                }
            )
            tasks = [tasks_by_id[task_id] for task_id in retained_ids]
            print(
                f"Fitting Thurstonian utilities with {omitted} omitted "
                f"({len(tasks)} tasks, {len(retained_comparisons)} comparisons)...",
                flush=True,
            )
            pairwise_data = PairwiseData.from_comparisons(
                retained_comparisons, tasks
            )
            if pairwise_data.n_comparisons != len(retained_comparisons):
                raise AssertionError(
                    f"PairwiseData count mismatch for {omitted}: "
                    f"{pairwise_data.n_comparisons} vs {len(retained_comparisons)}"
                )
            fit = fit_thurstonian(pairwise_data)
            if not fit.converged:
                raise RuntimeError(
                    f"Thurstonian fit did not converge for omitted={omitted}: "
                    f"{fit.termination_message}; iterations={fit.n_iterations}; "
                    f"gradient_norm={fit.gradient_norm}"
                )
            utilities = {
                task.id: float(fit.mu[index]) for index, task in enumerate(fit.tasks)
            }
            converged = fit.converged

        condition_data[omitted] = (
            retained_ids,
            retained_comparisons,
            zero_degree_removed,
            eval_ids,
            utilities,
            converged,
        )

    results: list[dict[str, Any]] = []
    baseline_w_raw: np.ndarray | None = None
    for omitted in OMISSIONS:
        (
            train_ids,
            comparisons,
            zero_degree_removed,
            eval_ids,
            train_utilities,
            converged,
        ) = condition_data[omitted]
        print(
            f"Selecting ridge alpha for {'baseline' if omitted is None else f'omit {omitted}'}...",
            flush=True,
        )
        best_alpha, eval_r, w_raw = raw_probe_direction(
            train_ids,
            train_utilities,
            eval_ids,
            eval_utilities,
            activations,
            activation_row,
        )
        if omitted is None:
            baseline_w_raw = w_raw
            utility_r = 1.0
            probe_cosine = 1.0
        else:
            if baseline_w_raw is None:
                raise AssertionError("Baseline probe must be fitted first")
            utility_r = correlation(
                np.asarray([train_utilities[task_id] for task_id in train_ids]),
                np.asarray([baseline_utilities[task_id] for task_id in train_ids]),
                f"utility r for omitted={omitted}",
            )
            probe_cosine = cosine(w_raw, baseline_w_raw)

        results.append(
            {
                "condition": "baseline" if omitted is None else f"omit_{omitted}",
                "omitted_dataset": omitted,
                "n_train_tasks": len(train_ids),
                "n_train_comparisons": len(comparisons),
                "n_zero_degree_removed": zero_degree_removed,
                "n_eval_tasks": len(eval_ids),
                "converged": bool(converged),
                "best_alpha": best_alpha,
                "eval_r": eval_r,
                "utility_r": utility_r,
                "probe_cosine": probe_cosine,
            }
        )
        print(
            f"  alpha={best_alpha:.6g}, eval_r={eval_r:.6f}, "
            f"utility_r={utility_r:.6f}, probe_cosine={probe_cosine:.6f}",
            flush=True,
        )

    write_outputs(results)
    print(f"Wrote results to {OUTPUT_DIR.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
