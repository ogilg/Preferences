"""
Analysis script for stated preference steering experiment.

Produces:
- Dose-response plots per steering position (Arm A)
- Per-task slope distributions (Arm A)
- Arm B dose-response per wording × position
- Phase 2 format comparison

Usage:
    python scripts/stated_steering/analyze.py --phase 1a
    python scripts/stated_steering/analyze.py --phase 1b
    python scripts/stated_steering/analyze.py --phase 2 --positions generation last_token
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "experiments" / "steering" / "stated_steering" / "results"
ASSETS_DIR = REPO_ROOT / "experiments" / "steering" / "stated_steering" / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

POSITIONS_ARM_A = ["task_tokens", "generation", "throughout", "last_token"]
COEFFICIENTS = [-5282, -3697, -2641, -2113, -1585, -1056, -528, 0, 528, 1056, 1585, 2113, 2641, 3697, 5282]

POSITION_COLORS = {
    "task_tokens": "#2196F3",
    "generation": "#4CAF50",
    "throughout": "#FF9800",
    "last_token": "#E91E63",
    "question_tokens": "#9C27B0",
}


def load_json(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def compute_dose_response(results: list[dict], position: str, coefficients: list[int]) -> dict:
    """Compute mean rating and parse rate per coefficient for a given position."""
    rows = [r for r in results if r["position"] == position]
    means = []
    parse_rates = []
    n_tasks = []
    for c in coefficients:
        c_rows = [r for r in rows if r["coefficient"] == c]
        all_ratings = []
        for r in c_rows:
            valid = [x for x in (r.get("ratings") or []) if x is not None]
            all_ratings.extend(valid)
        if all_ratings:
            means.append(np.mean(all_ratings))
        else:
            means.append(None)
        parse_rates.append(np.mean([r["parse_rate"] for r in c_rows]) if c_rows else 0.0)
        n_tasks.append(len(c_rows))
    return {"means": means, "parse_rates": parse_rates, "n_tasks": n_tasks}


def compute_per_task_slopes(results: list[dict], position: str) -> list[float]:
    """Fit slope (rating ~ coefficient) per task, return list of slopes."""
    rows = [r for r in results if r["position"] == position]
    task_ids = sorted(set(r["task_id"] for r in rows))
    slopes = []
    for tid in task_ids:
        task_rows = [r for r in rows if r["task_id"] == tid]
        xs, ys = [], []
        for r in task_rows:
            c = r["coefficient"]
            valid = [x for x in (r.get("ratings") or []) if x is not None]
            if valid:
                xs.append(c)
                ys.append(np.mean(valid))
        if len(xs) >= 3:
            slope, _, _, _, _ = stats.linregress(xs, ys)
            slopes.append(slope)
    return slopes


def plot_arm_a_dose_response(results: list[dict], out_path: Path) -> None:
    """Dose-response curves for all 4 steering positions."""
    coefficients = sorted(set(r["coefficient"] for r in results))
    coef_arr = np.array(coefficients)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    positions = POSITIONS_ARM_A
    for ax, pos in zip(axes, positions):
        dr = compute_dose_response(results, pos, coefficients)
        means = dr["means"]
        parse_rates = dr["parse_rates"]

        # Only plot points with parse_rate > 10%
        valid_mask = [pr > 0.1 for pr in parse_rates]
        valid_coefs = [c for c, v in zip(coefficients, valid_mask) if v]
        valid_means = [m for m, v in zip(means, valid_mask) if v and m is not None]

        color = POSITION_COLORS.get(pos, "gray")
        ax.plot(valid_coefs, valid_means, "o-", color=color, markersize=5, linewidth=1.5)
        ax.axhline(3.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, label="midpoint")
        ax.axvline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_title(pos, fontsize=12)
        ax.set_ylim(1, 5)
        ax.set_ylabel("Mean rating (1-5)")
        ax.set_xlabel("Steering coefficient")

        # Annotate parse rates
        for c, pr in zip(coefficients, parse_rates):
            if pr < 0.5:
                ax.annotate(f"{pr:.0%}", xy=(c, 1.1), fontsize=6, ha="center", color="red", rotation=45)

    plt.suptitle("Phase 1 Arm A: Dose-response by steering position", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_arm_a_slopes(results: list[dict], out_path: Path) -> None:
    """Per-task slope distributions, one strip per steering position."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)

    for ax, pos in zip(axes, POSITIONS_ARM_A):
        slopes = compute_per_task_slopes(results, pos)
        if not slopes:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(pos)
            continue
        slopes_arr = np.array(slopes)
        t_stat, p_val = stats.ttest_1samp(slopes_arr, 0)
        color = POSITION_COLORS.get(pos, "gray")
        ax.violinplot(slopes_arr, positions=[0], showmeans=True)
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"{pos}\nn={len(slopes)}, t={t_stat:.2f}, p={p_val:.3f}", fontsize=9)
        ax.set_ylabel("Slope (rating/coef×10⁴)")
        ax.set_xticks([])

    plt.suptitle("Phase 1 Arm A: Per-task slopes (rating ~ coefficient)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


WORD_SCALE = {"terrible": 1.0, "bad": 2.0, "neutral": 3.0, "good": 4.0, "great": 5.0}


def reparse_word_wording(results: list[dict]) -> list[dict]:
    """For wording 2 (word pick), re-parse completions using word matching."""
    import re as _re
    reparsed = []
    for r in results:
        if r["wording_idx"] != 2:
            reparsed.append(r)
            continue
        new_ratings = []
        for comp in r.get("completions", []):
            low = comp.lower()
            found = None
            for word, score in WORD_SCALE.items():
                if _re.search(r"\b" + word + r"\b", low):
                    found = score
                    break
            new_ratings.append(found)
        valid = [x for x in new_ratings if x is not None]
        r = dict(r)
        r["ratings"] = new_ratings
        r["parse_rate"] = len(valid) / len(new_ratings) if new_ratings else 0.0
        r["mean_rating"] = float(sum(valid) / len(valid)) if valid else None
        reparsed.append(r)
    return reparsed


def summarize_arm_b(results: list[dict]) -> None:
    """Print per-wording slope statistics for Arm B, with t-tests across wordings."""
    wordings = sorted(set(r["wording_idx"] for r in results))
    positions = sorted(set(r["position"] for r in results))
    print("\n=== Arm B Summary ===")
    # Per-position t-test across wordings
    print("\nPosition-level t-tests (n=8 wordings):")
    for pos in positions:
        slopes_for_pos = []
        for w_idx in wordings:
            rows = [r for r in results if r["wording_idx"] == w_idx and r["position"] == pos]
            xs, ys = [], []
            for r in rows:
                c = r["coefficient"]
                valid = [x for x in (r.get("ratings") or []) if x is not None]
                if valid:
                    xs.append(c)
                    ys.append(sum(valid) / len(valid))
            if len(xs) >= 3:
                slope, _, _, _, _ = stats.linregress(xs, ys)
                slopes_for_pos.append(slope)
        if slopes_for_pos:
            t, p = stats.ttest_1samp(slopes_for_pos, 0)
            mean_slope = sum(slopes_for_pos) / len(slopes_for_pos)
            print(f"  {pos}: mean_slope={mean_slope:.6f}, t={t:.2f}, p={p:.3f}, n={len(slopes_for_pos)}")
    # Per-wording detail
    print("\nPer-wording detail:")
    for w_idx in wordings:
        wording_rows = [r for r in results if r["wording_idx"] == w_idx]
        wording_text = wording_rows[0]["wording"][:60] + "..." if wording_rows else ""
        for pos in positions:
            rows = [r for r in wording_rows if r["position"] == pos]
            xs, ys = [], []
            for r in rows:
                c = r["coefficient"]
                valid = [x for x in (r.get("ratings") or []) if x is not None]
                if valid:
                    xs.append(c)
                    ys.append(sum(valid) / len(valid))
            if len(xs) >= 3:
                slope, _, _, _, _ = stats.linregress(xs, ys)
                mean_parse = sum(r["parse_rate"] for r in rows) / len(rows) if rows else 0.0
                print(f"  W{w_idx} {pos}: slope={slope:.6f}, parse={mean_parse:.0%}")


def plot_arm_b_dose_response(results: list[dict], out_path: Path) -> None:
    """Arm B: dose-response per wording and position."""
    wordings = sorted(set(r["wording_idx"] for r in results))
    positions = sorted(set(r["position"] for r in results))
    coefficients = sorted(set(r["coefficient"] for r in results))

    n_wordings = len(wordings)
    n_pos = len(positions)
    fig, axes = plt.subplots(n_wordings, n_pos, figsize=(4 * n_pos, 3 * n_wordings), sharex=True, sharey=True)
    if n_wordings == 1:
        axes = axes[np.newaxis, :]
    if n_pos == 1:
        axes = axes[:, np.newaxis]

    wording_texts = {}
    for r in results:
        wording_texts[r["wording_idx"]] = r["wording"][:40] + "..."

    for i, w_idx in enumerate(wordings):
        for j, pos in enumerate(positions):
            ax = axes[i, j]
            rows = [r for r in results if r["wording_idx"] == w_idx and r["position"] == pos]
            xs, ys = [], []
            for c in coefficients:
                c_rows = [r for r in rows if r["coefficient"] == c]
                all_ratings = [x for r in c_rows for x in (r.get("ratings") or []) if x is not None]
                if all_ratings:
                    xs.append(c)
                    ys.append(np.mean(all_ratings))
            color = POSITION_COLORS.get(pos, "gray")
            ax.plot(xs, ys, "o-", color=color, markersize=4)
            ax.axhline(3.0, color="gray", linestyle="--", linewidth=0.5)
            ax.axvline(0, color="gray", linestyle=":", linewidth=0.5)
            ax.set_ylim(1, 5)
            if i == 0:
                ax.set_title(pos, fontsize=10)
            if j == 0:
                ax.set_ylabel(wording_texts.get(w_idx, f"W{w_idx}"), fontsize=7)

    plt.suptitle("Phase 1 Arm B: No-task mood probe", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


FORMAT_SCALE = {
    "numeric_1_5": (1.0, 5.0),
    "qualitative_ternary": (1.0, 3.0),
    "adjective_pick": (1.0, 10.0),
    "anchored_precise_1_5": (1.0, 5.0),
    "anchored_simple_1_5": (1.0, 5.0),
    "fruit_rating": (0.0, 4.0),
}


def normalize_to_01(value: float, fmt: str) -> float:
    lo, hi = FORMAT_SCALE.get(fmt, (1.0, 5.0))
    return (value - lo) / (hi - lo)


def summarize_phase2(results: list[dict], positions: list[str]) -> None:
    """Print per-format dose-response slopes (normalized to [0,1]) with t-tests across tasks."""
    formats = sorted(set(r["format"] for r in results))
    print("\n=== Phase 2 Summary (normalized [0,1] slopes) ===")
    for pos in positions:
        print(f"\n  Position: {pos}")
        for fmt in formats:
            rows = [r for r in results if r["format"] == fmt and r["position"] == pos]
            task_ids = sorted(set(r["task_id"] for r in rows))
            slopes = []
            for tid in task_ids:
                task_rows = [r for r in rows if r["task_id"] == tid]
                xs, ys = [], []
                for r in task_rows:
                    c = r["coefficient"]
                    valid = [normalize_to_01(x, fmt) for x in (r.get("ratings") or []) if x is not None]
                    if valid:
                        xs.append(c)
                        ys.append(np.mean(valid))
                if len(xs) >= 3:
                    slope, _, _, _, _ = stats.linregress(xs, ys)
                    slopes.append(slope)
            if slopes:
                slopes_arr = np.array(slopes)
                t, p = stats.ttest_1samp(slopes_arr, 0)
                mean_parse = np.mean([r["parse_rate"] for r in rows if r["position"] == pos]) if rows else 0.0
                print(f"    {fmt}: mean_slope={slopes_arr.mean():.6f}, t={t:.2f}, p={p:.4f}, n={len(slopes)}, parse={mean_parse:.0%}")


def plot_phase2_comparison(results: list[dict], positions: list[str], out_path: Path) -> None:
    """Phase 2: Compare response formats by dose-response (normalized to [0,1])."""
    formats = sorted(set(r["format"] for r in results))
    coefficients = sorted(set(r["coefficient"] for r in results))

    n_pos = len(positions)
    fig, axes = plt.subplots(1, n_pos, figsize=(7 * n_pos, 5), sharey=True)
    if n_pos == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, len(formats)))

    for ax, pos in zip(axes, positions):
        for fmt, color in zip(formats, colors):
            rows = [r for r in results if r["format"] == fmt and r["position"] == pos]
            means = []
            for c in coefficients:
                c_rows = [r for r in rows if r["coefficient"] == c]
                all_ratings = [
                    normalize_to_01(x, fmt)
                    for r in c_rows for x in (r.get("ratings") or []) if x is not None
                ]
                means.append(np.mean(all_ratings) if all_ratings else None)
            valid_mask = [m is not None for m in means]
            xs = [c for c, v in zip(coefficients, valid_mask) if v]
            ys = [m for m, v in zip(means, valid_mask) if v]
            ax.plot(xs, ys, "o-", color=color, label=fmt, markersize=4)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
        ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
        ax.set_title(f"Steering position: {pos}", fontsize=11)
        ax.set_xlabel("Coefficient")
        ax.set_ylabel("Mean rating (normalized 0–1)")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8, loc="upper left")

    plt.suptitle("Phase 2: Response format comparison (all formats normalized to [0,1])", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def summarize_arm_a(results: list[dict]) -> None:
    """Print summary statistics for Arm A."""
    coefficients = sorted(set(r["coefficient"] for r in results))
    print("\n=== Arm A Summary ===")
    print(f"Total conditions: {len(results)}")
    print(f"Tasks: {len(set(r['task_id'] for r in results))}")
    print(f"Coefficients: {len(coefficients)}")
    print()
    for pos in POSITIONS_ARM_A:
        slopes = compute_per_task_slopes(results, pos)
        if slopes:
            slopes_arr = np.array(slopes)
            t, p = stats.ttest_1samp(slopes_arr, 0)
            mean_parse = np.mean([r["parse_rate"] for r in results if r["position"] == pos])
            print(f"  {pos}: mean_slope={slopes_arr.mean():.6f}, "
                  f"t={t:.2f}, p={p:.4f}, n={len(slopes)}, parse={mean_parse:.0%}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["1a", "1b", "2"], required=True)
    parser.add_argument("--positions", nargs="+", default=["generation", "last_token"])
    args = parser.parse_args()

    if args.phase == "1a":
        results_path = RESULTS_DIR / "phase1_arm_a.json"
        if not results_path.exists():
            print(f"No results found at {results_path}")
            return
        results = load_json(results_path)
        print(f"Loaded {len(results)} conditions from Arm A")
        summarize_arm_a(results)
        plot_arm_a_dose_response(results, ASSETS_DIR / "plot_022426_arm_a_dose_response.png")
        plot_arm_a_slopes(results, ASSETS_DIR / "plot_022426_arm_a_slopes.png")

    elif args.phase == "1b":
        results_path = RESULTS_DIR / "phase1_arm_b.json"
        if not results_path.exists():
            print(f"No results found at {results_path}")
            return
        results = load_json(results_path)
        results = reparse_word_wording(results)
        print(f"Loaded {len(results)} conditions from Arm B")
        summarize_arm_b(results)
        plot_arm_b_dose_response(results, ASSETS_DIR / "plot_022426_arm_b_dose_response.png")

    elif args.phase == "2":
        results_path = RESULTS_DIR / "phase2.json"
        if not results_path.exists():
            print(f"No results found at {results_path}")
            return
        results = load_json(results_path)
        print(f"Loaded {len(results)} conditions from Phase 2")
        summarize_phase2(results, args.positions)
        plot_phase2_comparison(results, args.positions, ASSETS_DIR / "plot_022426_phase2_formats.png")


if __name__ == "__main__":
    main()
