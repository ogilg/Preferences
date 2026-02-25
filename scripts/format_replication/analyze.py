"""
Analysis for Format Replication experiment.

Computes:
- Parse rates per format × position
- Dose-response: mean score vs coefficient per format × position
- Per-task slopes (linear fit of score ~ coefficient)
- One-sample t-test of slope distribution vs zero
- Format comparison: slope magnitudes across formats

Saves:
- results/statistics.json: summary statistics
- results/per_task_slopes.json: per-task slope data
"""

from __future__ import annotations

import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "experiments" / "steering" / "stated_steering" / "format_replication" / "results"

FORMATS = ["qualitative_ternary", "adjective_pick", "anchored_simple_1_5"]
POSITIONS = ["task_tokens", "generation", "last_token"]

# Scale normalizations for comparing across formats
# Convert to 0-1 for comparison: qualitative 1-3, adjective 1-10, anchored 1-5
SCALE_RANGES = {
    "qualitative_ternary": (1.0, 3.0),
    "adjective_pick": (1.0, 10.0),
    "anchored_simple_1_5": (1.0, 5.0),
}


def normalize_score(score: float, fmt: str) -> float:
    lo, hi = SCALE_RANGES[fmt]
    return (score - lo) / (hi - lo)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def fit_slope(coefficients: list[float], mean_scores: list[float]) -> tuple[float, float]:
    """Fit linear slope (score ~ coefficient). Returns (slope, r_squared)."""
    x = np.array(coefficients)
    y = np.array(mean_scores)
    if len(x) < 2 or np.std(y) < 1e-10:
        return 0.0, 0.0
    result = stats.linregress(x, y)
    slope = float(result.slope)
    r_sq = float(result.rvalue ** 2)
    return slope, r_sq


def analyze_format_position(fmt: str, position: str) -> dict | None:
    path = RESULTS_DIR / f"results_{fmt}_{position}.jsonl"
    records = load_jsonl(path)
    if not records:
        print(f"  MISSING: {fmt} × {position}")
        return None

    print(f"  {fmt} × {position}: {len(records)} records")

    # Group by task_id → coefficient → scores
    by_task: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        coef = r["coefficient"]
        task_id = r["task_id"]
        scores = [s for s in r["scores"] if s is not None]
        by_task[task_id][coef].extend(scores)

    # Parse rate
    all_scores_flat = []
    all_raw_flat = []
    for r in records:
        for s in r["scores"]:
            all_raw_flat.append(s)
        all_scores_flat.extend([s for s in r["scores"] if s is not None])
    parse_rate = len(all_scores_flat) / len(all_raw_flat) if all_raw_flat else 0.0

    # Sorted coefficient grid
    all_coefs = sorted(set(r["coefficient"] for r in records))

    # Dose-response: mean score per coefficient (across all tasks)
    dose_response = {}
    for coef in all_coefs:
        all_scores_at_coef = []
        for task_scores in by_task.values():
            all_scores_at_coef.extend(task_scores.get(coef, []))
        if all_scores_at_coef:
            dose_response[coef] = {
                "mean": float(np.mean(all_scores_at_coef)),
                "std": float(np.std(all_scores_at_coef)),
                "n": len(all_scores_at_coef),
            }

    # Per-task slope
    per_task_slopes = []
    per_task_slopes_normalized = []
    for task_id, coef_scores in by_task.items():
        coefs = sorted(coef_scores.keys())
        means = [np.mean(coef_scores[c]) for c in coefs if coef_scores[c]]
        coefs_used = [c for c in coefs if coef_scores[c]]
        if len(coefs_used) >= 3:
            slope, r_sq = fit_slope(coefs_used, means)
            lo, hi = SCALE_RANGES[fmt]
            slope_normalized = slope * (hi - lo) / 1.0  # scale-agnostic: per unit of full range
            per_task_slopes.append({"task_id": task_id, "slope": slope, "r_squared": r_sq})
            per_task_slopes_normalized.append(slope_normalized)

    # t-test: are slopes significantly different from zero?
    if per_task_slopes_normalized:
        slopes_arr = np.array([s["slope"] for s in per_task_slopes])
        t_stat, p_val = stats.ttest_1samp(slopes_arr, 0.0)
        mean_slope = float(np.mean(slopes_arr))
        std_slope = float(np.std(slopes_arr))
    else:
        t_stat, p_val = 0.0, 1.0
        mean_slope, std_slope = 0.0, 0.0

    result = {
        "format": fmt,
        "position": position,
        "n_records": len(records),
        "n_tasks": len(by_task),
        "parse_rate": parse_rate,
        "dose_response": dose_response,
        "mean_slope": mean_slope,
        "std_slope": std_slope,
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "n_slopes": len(per_task_slopes),
        "per_task_slopes": per_task_slopes,
    }
    print(f"    parse_rate={parse_rate:.1%}, mean_slope={mean_slope:.2e}, "
          f"t={t_stat:.2f}, p={p_val:.4f}")
    return result


def main():
    print("=== Format Replication Analysis ===\n")
    all_results = []

    for fmt in FORMATS:
        for pos in POSITIONS:
            r = analyze_format_position(fmt, pos)
            if r:
                all_results.append(r)

    # Summary comparison table
    print("\n=== Summary: Mean Slope by Format × Position ===")
    print(f"{'Format':<25} {'Position':<15} {'MeanSlope':>12} {'t':>8} {'p':>10} {'ParseRate':>10}")
    print("-" * 85)
    for r in all_results:
        sig = "*" if r["p_value"] < 0.05 else ""
        print(f"{r['format']:<25} {r['position']:<15} {r['mean_slope']:>12.2e} "
              f"{r['t_stat']:>8.2f} {r['p_value']:>10.4f} {r['parse_rate']:>9.1%} {sig}")

    # Save statistics
    stats_path = RESULTS_DIR / "statistics.json"
    # Strip per_task_slopes from main stats (save separately)
    stats_summary = []
    per_task_data = []
    for r in all_results:
        slopes = r.pop("per_task_slopes")
        stats_summary.append(r)
        per_task_data.append({
            "format": r["format"],
            "position": r["position"],
            "slopes": slopes,
        })

    with open(stats_path, "w") as f:
        json.dump(stats_summary, f, indent=2)
    print(f"\nSaved statistics → {stats_path}")

    slopes_path = RESULTS_DIR / "per_task_slopes.json"
    with open(slopes_path, "w") as f:
        json.dump(per_task_data, f, indent=2)
    print(f"Saved per-task slopes → {slopes_path}")

    return stats_summary, per_task_data


if __name__ == "__main__":
    main()
