"""Plotting for task-mean direction steering experiment.

Generates three plots:
1. Steering effect comparison (bar chart)
2. Dose-response curve
3. Per-pair scatter (task_mean vs EOT)
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

MAGNITUDES = [0.01, 0.02, 0.03, 0.05]


def load_jsonl(path: Path) -> list[dict]:
    records = []
    for line in path.read_text().strip().split("\n"):
        if line.strip():
            records.append(json.loads(line))
    return records


def bootstrap_effect(
    pos_recs: list[dict],
    neg_recs: list[dict],
    n_boot: int = 10000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    if rng is None:
        rng = np.random.default_rng(42)
    pos = np.array([1 if r["choice_presented"] == "a" else 0 for r in pos_recs if r["choice_presented"] in ("a", "b")])
    neg = np.array([1 if r["choice_presented"] == "a" else 0 for r in neg_recs if r["choice_presented"] in ("a", "b")])
    if len(pos) == 0 or len(neg) == 0:
        return (float("nan"), float("nan"), float("nan"))
    effects = []
    for _ in range(n_boot):
        bp = pos[rng.integers(0, len(pos), size=len(pos))].mean()
        bn = neg[rng.integers(0, len(neg), size=len(neg))].mean()
        effects.append(bp - bn)
    point = float(pos.mean() - neg.mean())
    return point, float(np.quantile(effects, 0.025)), float(np.quantile(effects, 0.975))


def p_choose_a(records: list[dict]) -> float:
    valid = [r for r in records if r["choice_presented"] in ("a", "b")]
    if not valid:
        return float("nan")
    return sum(1 for r in valid if r["choice_presented"] == "a") / len(valid)


def per_pair_p_a(records: list[dict]) -> dict[str, float]:
    by_pair: dict[str, list[int]] = defaultdict(list)
    for r in records:
        if r["choice_presented"] in ("a", "b"):
            by_pair[r["pair_id"]].append(1 if r["choice_presented"] == "a" else 0)
    return {pid: float(np.mean(vals)) for pid, vals in by_pair.items()}


def per_pair_effect(pos_recs: list[dict], neg_recs: list[dict]) -> dict[str, float]:
    pa_pos = per_pair_p_a(pos_recs)
    pa_neg = per_pair_p_a(neg_recs)
    common = set(pa_pos) & set(pa_neg)
    return {pid: pa_pos[pid] - pa_neg[pid] for pid in common}


def plot_steering_effect_comparison(
    tm_records: list[dict],
    eot_records: list[dict],
    prompt_last_records: list[dict],
    rng: np.random.Generator,
) -> None:
    """Bar chart: steering effect at each multiplier for L25, L32, with reference lines."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Compute effects for task_mean
    bars = []
    labels = []
    for layer in [25, 32]:
        for mag in MAGNITUDES:
            pos = [r for r in tm_records if r["layer"] == layer and r["multiplier"] == mag]
            neg = [r for r in tm_records if r["layer"] == layer and r["multiplier"] == -mag]
            if not pos or not neg:
                continue
            eff, lo, hi = bootstrap_effect(pos, neg, rng=rng)
            bars.append((eff, lo, hi))
            labels.append(f"L{layer}\n±{mag}")

    if not bars:
        print("  No task_mean data for bar chart, skipping")
        return

    x = np.arange(len(bars))
    effs = [b[0] for b in bars]
    errs_lo = [b[0] - b[1] for b in bars]
    errs_hi = [b[2] - b[0] for b in bars]

    colors = ["#4A90D9"] * len(MAGNITUDES) + ["#E67E22"] * len(MAGNITUDES)
    ax.bar(x, effs, yerr=[errs_lo, errs_hi], capsize=4, color=colors[:len(bars)], alpha=0.8, edgecolor="black", linewidth=0.5)

    # EOT reference at ±0.03
    eot_pos = [r for r in eot_records if r["multiplier"] == 0.03]
    eot_neg = [r for r in eot_records if r["multiplier"] == -0.03]
    if eot_pos and eot_neg:
        eot_eff, _, _ = bootstrap_effect(eot_pos, eot_neg, rng=rng)
        ax.axhline(eot_eff, color="red", linestyle="--", linewidth=1.5, label=f"EOT ±0.03 ({eot_eff:+.3f})")

    # prompt_last reference at ±0.03
    pl_pos = [r for r in prompt_last_records if r["multiplier"] == 0.03]
    pl_neg = [r for r in prompt_last_records if r["multiplier"] == -0.03]
    if pl_pos and pl_neg:
        pl_eff, _, _ = bootstrap_effect(pl_pos, pl_neg, rng=rng)
        ax.axhline(pl_eff, color="green", linestyle="--", linewidth=1.5, label=f"prompt_last ±0.03 ({pl_eff:+.3f})")

    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Steering effect: P(choose A|+m) - P(choose A|-m)")
    ax.set_xlabel("Layer / Multiplier magnitude")
    ax.set_title("Task-Mean Direction Steering: Effect by Layer and Multiplier")
    ax.set_ylim(-0.15, 1.1)
    ax.legend(loc="upper left")

    fig.tight_layout()
    out = ASSETS_DIR / "plot_031326_steering_effect_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_dose_response(
    tm_records: list[dict],
    eot_records: list[dict],
    prompt_last_records: list[dict],
    rng: np.random.Generator,
) -> None:
    """Line plot: steering effect vs multiplier magnitude for L25 and L32."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for layer, color, marker in [(25, "#4A90D9", "o"), (32, "#E67E22", "s")]:
        mags_plot = []
        effs_plot = []
        los_plot = []
        his_plot = []
        for mag in MAGNITUDES:
            pos = [r for r in tm_records if r["layer"] == layer and r["multiplier"] == mag]
            neg = [r for r in tm_records if r["layer"] == layer and r["multiplier"] == -mag]
            if not pos or not neg:
                continue
            eff, lo, hi = bootstrap_effect(pos, neg, rng=rng)
            mags_plot.append(mag)
            effs_plot.append(eff)
            los_plot.append(lo)
            his_plot.append(hi)

        if mags_plot:
            errs = [
                [e - l for e, l in zip(effs_plot, los_plot)],
                [h - e for e, h in zip(effs_plot, his_plot)],
            ]
            ax.errorbar(
                mags_plot, effs_plot, yerr=errs,
                color=color, marker=marker, markersize=7,
                capsize=4, linewidth=2, label=f"task_mean L{layer}",
            )

    # EOT reference point
    eot_pos = [r for r in eot_records if r["multiplier"] == 0.03]
    eot_neg = [r for r in eot_records if r["multiplier"] == -0.03]
    if eot_pos and eot_neg:
        eot_eff, eot_lo, eot_hi = bootstrap_effect(eot_pos, eot_neg, rng=rng)
        ax.errorbar(
            [0.03], [eot_eff], yerr=[[eot_eff - eot_lo], [eot_hi - eot_eff]],
            color="red", marker="D", markersize=9, capsize=5, linewidth=0,
            label=f"EOT L31 ±0.03",
        )

    # prompt_last reference point
    pl_pos = [r for r in prompt_last_records if r["multiplier"] == 0.03]
    pl_neg = [r for r in prompt_last_records if r["multiplier"] == -0.03]
    if pl_pos and pl_neg:
        pl_eff, pl_lo, pl_hi = bootstrap_effect(pl_pos, pl_neg, rng=rng)
        ax.errorbar(
            [0.03], [pl_eff], yerr=[[pl_eff - pl_lo], [pl_hi - pl_eff]],
            color="green", marker="^", markersize=9, capsize=5, linewidth=0,
            label=f"prompt_last ±0.03",
        )

    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Multiplier magnitude")
    ax.set_ylabel("Steering effect: P(choose A|+m) - P(choose A|-m)")
    ax.set_title("Dose-Response: Task-Mean Direction Steering")
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlim(0, 0.06)
    ax.legend()

    fig.tight_layout()
    out = ASSETS_DIR / "plot_031326_dose_response.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_per_pair_scatter(
    tm_records: list[dict],
    eot_records: list[dict],
    rng: np.random.Generator,
) -> None:
    """Scatter: EOT per-pair effect (x) vs task_mean per-pair effect (y)."""
    eot_pos = [r for r in eot_records if r["multiplier"] == 0.03]
    eot_neg = [r for r in eot_records if r["multiplier"] == -0.03]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, layer in enumerate([25, 32]):
        ax = axes[idx]
        tm_pos = [r for r in tm_records if r["layer"] == layer and r["multiplier"] == 0.03]
        tm_neg = [r for r in tm_records if r["layer"] == layer and r["multiplier"] == -0.03]

        if not tm_pos or not tm_neg or not eot_pos or not eot_neg:
            ax.set_title(f"L{layer}: insufficient data")
            continue

        tm_eff = per_pair_effect(tm_pos, tm_neg)
        eot_eff = per_pair_effect(eot_pos, eot_neg)
        common = sorted(set(tm_eff) & set(eot_eff))

        if len(common) < 10:
            ax.set_title(f"L{layer}: too few common pairs ({len(common)})")
            continue

        x = np.array([eot_eff[p] for p in common])
        y = np.array([tm_eff[p] for p in common])
        r = np.corrcoef(x, y)[0, 1]

        ax.scatter(x, y, alpha=0.3, s=15, color="#4A90D9" if layer == 25 else "#E67E22")
        ax.set_xlabel("EOT per-pair effect (m=±0.03)")
        ax.set_ylabel(f"Task-mean L{layer} per-pair effect (m=±0.03)")
        ax.set_title(f"L{layer}: r = {r:.3f} (n={len(common)} pairs)")

        # Reference lines
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)

        # Fit line
        m, b = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, m * x_line + b, "r--", linewidth=1, alpha=0.7)

        lim = max(abs(x).max(), abs(y).max()) * 1.1
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

    fig.suptitle("Per-Pair Steering Effect: Task-Mean vs EOT", fontsize=13)
    fig.tight_layout()
    out = ASSETS_DIR / "plot_031326_per_pair_scatter.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def main() -> None:
    print("Loading data...")
    tm_records = load_jsonl(TASK_MEAN_CHECKPOINT)
    v2_records = load_jsonl(V2_CHECKPOINT)
    baseline = [r for r in v2_records if r["condition"] == "baseline"]
    prompt_last = [r for r in v2_records if r["condition"] == "probe"]
    eot_records = load_jsonl(EOT_CHECKPOINT)

    print(f"  task_mean={len(tm_records)}, baseline={len(baseline)}, "
          f"prompt_last={len(prompt_last)}, eot={len(eot_records)}")

    rng = np.random.default_rng(42)

    print("\nPlot 1: Steering effect comparison...")
    plot_steering_effect_comparison(tm_records, eot_records, prompt_last, rng)

    print("Plot 2: Dose-response...")
    plot_dose_response(tm_records, eot_records, prompt_last, rng)

    print("Plot 3: Per-pair scatter...")
    plot_per_pair_scatter(tm_records, eot_records, rng)

    print("\nDone!")


if __name__ == "__main__":
    main()
