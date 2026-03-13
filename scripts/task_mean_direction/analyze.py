"""Analysis for task-mean direction steering experiment.

Combines task_mean checkpoint with baseline (v2 followup) and EOT/prompt_last
data. Computes steering effects, dose-response curves, and per-pair correlations.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# --- Paths ---
TASK_MEAN_CHECKPOINT = Path(
    "experiments/steering/task_mean_direction/checkpoint.jsonl"
)
V2_CHECKPOINT = Path(
    "experiments/revealed_steering_v2/followup/checkpoint.jsonl"
)
EOT_CHECKPOINT = Path("experiments/steering/eot_direction/checkpoint.jsonl")
ASSETS_DIR = Path("experiments/steering/task_mean_direction/assets")


def load_jsonl(path: Path) -> list[dict]:
    records = []
    for line in path.read_text().strip().split("\n"):
        if line.strip():
            records.append(json.loads(line))
    return records


def filter_parseable(records: list[dict]) -> list[dict]:
    """Keep only records with valid a/b choices."""
    return [r for r in records if r["choice_presented"] in ("a", "b")]


def p_choose_a(records: list[dict]) -> float:
    """Fraction choosing presented A."""
    valid = [r for r in records if r["choice_presented"] in ("a", "b")]
    if not valid:
        return float("nan")
    return sum(1 for r in valid if r["choice_presented"] == "a") / len(valid)


def bootstrap_ci(
    records: list[dict],
    n_boot: int = 10000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Bootstrap CI for P(choose presented A). Returns (mean, lo, hi)."""
    if rng is None:
        rng = np.random.default_rng(42)
    valid = [r for r in records if r["choice_presented"] in ("a", "b")]
    n = len(valid)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    choices = np.array([1 if r["choice_presented"] == "a" else 0 for r in valid])
    boot_means = np.array(
        [choices[rng.integers(0, n, size=n)].mean() for _ in range(n_boot)]
    )
    alpha = (1 - ci) / 2
    return float(choices.mean()), float(np.quantile(boot_means, alpha)), float(np.quantile(boot_means, 1 - alpha))


def bootstrap_effect_ci(
    records_pos: list[dict],
    records_neg: list[dict],
    n_boot: int = 10000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Bootstrap CI for steering effect = P_a(pos) - P_a(neg)."""
    if rng is None:
        rng = np.random.default_rng(42)
    valid_pos = np.array(
        [1 if r["choice_presented"] == "a" else 0 for r in records_pos if r["choice_presented"] in ("a", "b")]
    )
    valid_neg = np.array(
        [1 if r["choice_presented"] == "a" else 0 for r in records_neg if r["choice_presented"] in ("a", "b")]
    )
    n_pos, n_neg = len(valid_pos), len(valid_neg)
    if n_pos == 0 or n_neg == 0:
        return (float("nan"), float("nan"), float("nan"))
    effects = []
    for _ in range(n_boot):
        bp = valid_pos[rng.integers(0, n_pos, size=n_pos)].mean()
        bn = valid_neg[rng.integers(0, n_neg, size=n_neg)].mean()
        effects.append(bp - bn)
    effects = np.array(effects)
    point = valid_pos.mean() - valid_neg.mean()
    return float(point), float(np.quantile(effects, 0.025)), float(np.quantile(effects, 0.975))


def per_pair_p_a(records: list[dict]) -> dict[str, float]:
    """P(choose_presented=a) per pair, averaged over orderings."""
    by_pair: dict[str, list[int]] = defaultdict(list)
    for r in records:
        if r["choice_presented"] in ("a", "b"):
            by_pair[r["pair_id"]].append(1 if r["choice_presented"] == "a" else 0)
    return {pid: np.mean(vals) for pid, vals in by_pair.items()}


def per_pair_effect(
    records_pos: list[dict], records_neg: list[dict]
) -> dict[str, float]:
    """Per-pair steering effect = P_a(pos) - P_a(neg)."""
    pa_pos = per_pair_p_a(records_pos)
    pa_neg = per_pair_p_a(records_neg)
    common = set(pa_pos) & set(pa_neg)
    return {pid: pa_pos[pid] - pa_neg[pid] for pid in common}


def main() -> None:
    print("Loading data...")

    # Task-mean data
    tm_records = load_jsonl(TASK_MEAN_CHECKPOINT)
    print(f"  task_mean: {len(tm_records)} records")

    # V2 followup (baseline + probe/prompt_last)
    v2_records = load_jsonl(V2_CHECKPOINT)
    baseline_records = [r for r in v2_records if r["condition"] == "baseline"]
    prompt_last_records = [r for r in v2_records if r["condition"] == "probe"]
    print(f"  baseline: {len(baseline_records)} records")
    print(f"  prompt_last: {len(prompt_last_records)} records")

    # EOT
    eot_records = load_jsonl(EOT_CHECKPOINT)
    print(f"  eot: {len(eot_records)} records")

    # --- Parse rates ---
    print("\n=== Parse Rates ===")
    for name, recs in [
        ("baseline", baseline_records),
        ("eot", eot_records),
        ("prompt_last", prompt_last_records),
        ("task_mean", tm_records),
    ]:
        valid = sum(1 for r in recs if r["choice_presented"] in ("a", "b"))
        total = len(recs)
        rate = valid / total if total > 0 else 0
        refusals = sum(1 for r in recs if r.get("choice_presented") == "refusal")
        fails = sum(1 for r in recs if r.get("choice_presented") == "parse_fail")
        print(f"  {name}: {rate:.4f} ({valid}/{total}), refusals={refusals}, parse_fail={fails}")

    # Parse rates by multiplier for task_mean
    print("\n  task_mean by layer × multiplier:")
    tm_by_lm: dict[tuple[int, float], list[dict]] = defaultdict(list)
    for r in tm_records:
        tm_by_lm[(r["layer"], r["multiplier"])].append(r)
    for (layer, mult), recs in sorted(tm_by_lm.items()):
        valid = sum(1 for r in recs if r["choice_presented"] in ("a", "b"))
        total = len(recs)
        rate = valid / total if total > 0 else 0
        print(f"    L{layer} m={mult:+.2f}: {rate:.4f} ({valid}/{total})")

    # --- Baseline P(choose A) ---
    rng = np.random.default_rng(42)
    bl_mean, bl_lo, bl_hi = bootstrap_ci(baseline_records, rng=rng)
    print(f"\n=== Baseline P(choose A) ===")
    print(f"  {bl_mean:.4f} [{bl_lo:.4f}, {bl_hi:.4f}]")

    # --- Steering effects ---
    print("\n=== Steering Effects (P_a(+m) - P_a(-m)) ===")
    magnitudes = [0.01, 0.02, 0.03, 0.05]

    # Task-mean by layer
    print("\n  Task-mean steering:")
    tm_effects = {}
    for layer in [25, 32]:
        for mag in magnitudes:
            pos_recs = [r for r in tm_records if r["layer"] == layer and r["multiplier"] == mag]
            neg_recs = [r for r in tm_records if r["layer"] == layer and r["multiplier"] == -mag]
            if not pos_recs or not neg_recs:
                print(f"    L{layer} m=±{mag}: MISSING DATA")
                continue
            eff, lo, hi = bootstrap_effect_ci(pos_recs, neg_recs, rng=rng)
            tm_effects[(layer, mag)] = (eff, lo, hi)
            print(f"    L{layer} m=±{mag:.2f}: {eff:+.4f} [{lo:+.4f}, {hi:+.4f}] (n_pos={len(pos_recs)}, n_neg={len(neg_recs)})")

    # EOT at ±0.03
    eot_pos = [r for r in eot_records if r["multiplier"] == 0.03]
    eot_neg = [r for r in eot_records if r["multiplier"] == -0.03]
    eot_eff, eot_lo, eot_hi = bootstrap_effect_ci(eot_pos, eot_neg, rng=rng)
    print(f"\n  EOT m=±0.03: {eot_eff:+.4f} [{eot_lo:+.4f}, {eot_hi:+.4f}]")

    # prompt_last at ±0.03
    pl_pos = [r for r in prompt_last_records if r["multiplier"] == 0.03]
    pl_neg = [r for r in prompt_last_records if r["multiplier"] == -0.03]
    pl_eff, pl_lo, pl_hi = bootstrap_effect_ci(pl_pos, pl_neg, rng=rng)
    print(f"  prompt_last m=±0.03: {pl_eff:+.4f} [{pl_lo:+.4f}, {pl_hi:+.4f}]")

    # --- P(choose A) by condition for dose-response ---
    print("\n=== P(choose presented A) by condition ===")
    print(f"  baseline: {bl_mean:.4f}")
    for layer in [25, 32]:
        for mult in sorted(set(r["multiplier"] for r in tm_records if r["layer"] == layer)):
            recs = [r for r in tm_records if r["layer"] == layer and r["multiplier"] == mult]
            pa = p_choose_a(recs)
            print(f"  task_mean L{layer} m={mult:+.2f}: {pa:.4f} (n={len(filter_parseable(recs))})")

    for mult in sorted(set(r["multiplier"] for r in eot_records)):
        recs = [r for r in eot_records if r["multiplier"] == mult]
        pa = p_choose_a(recs)
        print(f"  eot m={mult:+.2f}: {pa:.4f} (n={len(filter_parseable(recs))})")

    # prompt_last at matching multipliers
    for mult in [-0.05, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.05]:
        recs = [r for r in prompt_last_records if r["multiplier"] == mult]
        if recs:
            pa = p_choose_a(recs)
            print(f"  prompt_last m={mult:+.2f}: {pa:.4f} (n={len(filter_parseable(recs))})")

    # --- Per-pair correlations ---
    print("\n=== Per-Pair Correlations ===")
    # Best matching multiplier for task_mean (use 0.03 to match EOT)
    for layer in [25, 32]:
        tm_pair_eff = per_pair_effect(
            [r for r in tm_records if r["layer"] == layer and r["multiplier"] == 0.03],
            [r for r in tm_records if r["layer"] == layer and r["multiplier"] == -0.03],
        )
        eot_pair_eff = per_pair_effect(eot_pos, eot_neg)
        common = set(tm_pair_eff) & set(eot_pair_eff)
        if len(common) >= 10:
            x = np.array([eot_pair_eff[p] for p in sorted(common)])
            y = np.array([tm_pair_eff[p] for p in sorted(common)])
            r = np.corrcoef(x, y)[0, 1]
            print(f"  task_mean L{layer} vs EOT (m=±0.03): r={r:.4f} (n={len(common)} pairs)")
        else:
            print(f"  task_mean L{layer} vs EOT: insufficient pairs ({len(common)})")

        pl_pair_eff = per_pair_effect(pl_pos, pl_neg)
        common_pl = set(tm_pair_eff) & set(pl_pair_eff)
        if len(common_pl) >= 10:
            x = np.array([pl_pair_eff[p] for p in sorted(common_pl)])
            y = np.array([tm_pair_eff[p] for p in sorted(common_pl)])
            r = np.corrcoef(x, y)[0, 1]
            print(f"  task_mean L{layer} vs prompt_last (m=±0.03): r={r:.4f} (n={len(common_pl)} pairs)")

    # --- Steering fallback rate ---
    n_fb = sum(1 for r in tm_records if r.get("steering_fallback", False))
    print(f"\n=== Steering Fallbacks ===")
    print(f"  {n_fb}/{len(tm_records)} ({n_fb/len(tm_records)*100:.2f}%)")

    # --- Save summary JSON for plotting ---
    summary = {
        "baseline_p_a": bl_mean,
        "baseline_ci": [bl_lo, bl_hi],
        "eot_effect_0.03": [eot_eff, eot_lo, eot_hi],
        "prompt_last_effect_0.03": [pl_eff, pl_lo, pl_hi],
        "task_mean_effects": {
            f"L{layer}_m{mag}": list(tm_effects.get((layer, mag), (None, None, None)))
            for layer in [25, 32]
            for mag in magnitudes
        },
        "n_records": {
            "task_mean": len(tm_records),
            "baseline": len(baseline_records),
            "eot": len(eot_records),
            "prompt_last": len(prompt_last_records),
        },
    }
    summary_path = ASSETS_DIR / "analysis_summary.json"
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
