"""Overlay the user-context persona steering curves on the published system-context ones.

Reuses the existing outcome definitions and aggregation from
scripts/cross_persona_differential/plot_options.py, as the spec requires, so the
two arms are scored identically (including the truncated-response rescue and the
refusal classification).
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.cross_persona_differential.plot_options import (
    _effective_choice,
    _load,
    _wilson_err,
    contrastive_curve,
    single_task_curve,
    style_axis,
)

REPO = Path(__file__).resolve().parents[2]
SYSTEM_CKPT = REPO / "experiments/persona_steering_l23_finegrain/checkpoints"
USER_CKPT = REPO / "experiments/reviewer_followups/user_context_persona/checkpoints"
PAIRS = REPO / "experiments/layer_sweep/harm_breakdown/steering_pairs_150.json"
ASSETS = REPO / "experiments/reviewer_followups/user_context_persona/assets"

SHARED_MULTS = [-0.06, -0.02, 0.0, 0.02, 0.06]
HARMFUL_ORIGINS = {"BAILBENCH", "STRESS_TEST"}
PAIR_TYPES = ["bb", "hb", "hh"]
PAIR_TYPE_LABEL = {"bb": "benign-benign", "hb": "harmful-benign", "hh": "harmful-harmful"}

ARMS = {
    "system": ("System-context persona", "#B45309"),
    "user": ("User-context persona", "#1D4ED8"),
}


def shared_only(rows: list[dict]) -> list[dict]:
    """The published run used 9 multipliers; overlay only the 5 shared with the new run."""
    keep = {round(m, 6) for m in SHARED_MULTS}
    return [r for r in rows if round(r["signed_multiplier"], 6) in keep]


def load_arms() -> dict:
    pairs = {p["pair_id"]: p for p in json.loads(PAIRS.read_text())}
    arms = {
        "system": {
            "contrastive": shared_only(_load(SYSTEM_CKPT / "sadist_contrastive.parsed.jsonl")),
            "single": shared_only(_load(SYSTEM_CKPT / "sadist_single_task.parsed.jsonl")),
        },
        "user": {
            "contrastive": _load(USER_CKPT / "sadist_user_context_contrastive.parsed.jsonl"),
            "single": _load(USER_CKPT / "sadist_user_context_single_task.parsed.jsonl"),
        },
    }
    for arm in arms.values():
        for rows in arm.values():
            for r in rows:
                r["pair_type"] = pairs[r["pair_id"]]["pair_type"]
    return arms, pairs


def counts_table(rows: list[dict]) -> list[dict]:
    by_mult: dict[float, list[dict]] = defaultdict(list)
    for r in rows:
        by_mult[round(r["signed_multiplier"], 6)].append(r)
    out = []
    for mult in sorted(by_mult):
        group = by_mult[mult]
        responded = [r for r in group if _effective_choice(r) in ("a", "b")]
        refused = len(group) - len(responded)
        out.append({
            "multiplier": mult,
            "rows": len(group),
            "responded": len(responded),
            "refusals": refused,
        })
    return out


def harmful_choice_rate_at_zero(rows: list[dict], pairs: dict) -> tuple[float, int] | None:
    """Raw harmful-task choice rate on harmful-benign pairs at c=0."""
    hits = n = 0
    for r in rows:
        if round(r["signed_multiplier"], 6) != 0.0 or r["pair_type"] != "hb":
            continue
        ch = _effective_choice(r)
        if ch not in ("a", "b"):
            continue
        pair = pairs[r["pair_id"]]
        harmful_side = "a" if pair["task_a_origin"] in HARMFUL_ORIGINS else "b"
        hits += int(ch == harmful_side)
        n += 1
    return (hits / n, n) if n else None


def two_panel(arms: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    fig.patch.set_facecolor("#F0F0EC")
    for ax in axes:
        ax.set_facecolor("#F0F0EC")

    for key, (label, colour) in ARMS.items():
        cx, cy, ce = contrastive_curve(arms[key]["contrastive"])
        axes[0].errorbar(cx, cy, yerr=ce, color=colour, marker="o", markersize=5,
                         linewidth=1.4, capsize=1.5, alpha=0.9, label=label)
        sx, sy, se = single_task_curve(arms[key]["single"])
        axes[1].errorbar(sx, sy, yerr=se, color=colour, marker="o", markersize=5,
                         linewidth=1.4, capsize=1.5, alpha=0.9, label=label)

    style_axis(axes[0], ylabel="P(chose steered task | responded)", xlabel="steering coefficient")
    style_axis(axes[1], xlabel="steering coefficient")
    axes[0].set_title("Steer both tasks (contrastive)", fontsize=10, color="#374151")
    axes[1].set_title("Steer one task (pooled unilateral)", fontsize=10, color="#374151")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
    print(f"wrote {out_path}")


def pair_type_panel(arms: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.4))
    fig.patch.set_facecolor("#F0F0EC")
    for row_idx, (mode, curve_fn, mode_label) in enumerate(
        [("contrastive", contrastive_curve, "contrastive"),
         ("single", single_task_curve, "single-task")]
    ):
        for col, ptype in enumerate(PAIR_TYPES):
            ax = axes[row_idx][col]
            ax.set_facecolor("#F0F0EC")
            for key, (label, colour) in ARMS.items():
                rows = [r for r in arms[key][mode] if r["pair_type"] == ptype]
                if not rows:
                    continue
                xs, ys, es = curve_fn(rows)
                ax.errorbar(xs, ys, yerr=es, color=colour, marker="o", markersize=4,
                            linewidth=1.2, capsize=1.5, alpha=0.9, label=label)
            style_axis(ax,
                       ylabel="P(chose steered | responded)" if col == 0 else None,
                       xlabel="steering coefficient" if row_idx == 1 else None)
            ax.set_title(f"{mode_label} — {PAIR_TYPE_LABEL[ptype]}", fontsize=9, color="#374151")
    axes[0][0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
    print(f"wrote {out_path}")


def main() -> None:
    arms, pairs = load_arms()
    ASSETS.mkdir(parents=True, exist_ok=True)

    for key in ARMS:
        for mode in ("contrastive", "single"):
            n = len(arms[key][mode])
            print(f"{key}/{mode}: {n} rows")
            if n == 0:
                raise SystemExit(f"No rows for {key}/{mode} — run incomplete, refusing to plot.")

    two_panel(arms, ASSETS / "plot_072526_user_vs_system_context_dose_response.png")
    pair_type_panel(arms, ASSETS / "plot_072526_user_vs_system_context_by_pair_type.png")

    print("\n=== Counts and P(chose steered | responded) ===")
    for key, (label, _) in ARMS.items():
        for mode, curve_fn in [("contrastive", contrastive_curve), ("single", single_task_curve)]:
            rows = arms[key][mode]
            xs, ys, es = curve_fn(rows)
            print(f"\n{label} / {mode}")
            print(f"{'coef':>7} {'rows':>6} {'resp':>6} {'refus':>6}   P(steered)")
            table = {t["multiplier"]: t for t in counts_table(rows)}
            for x, y, lo, hi in zip(xs, ys, es[0], es[1]):
                t = table.get(round(x, 6))
                stats = f"{t['rows']:>6} {t['responded']:>6} {t['refusals']:>6}" if t else f"{'-':>6} {'-':>6} {'-':>6}"
                print(f"{x:>7.2f} {stats}   {y:.3f} (-{lo:.3f}/+{hi:.3f})")

    print("\n=== Harmful-task choice rate on harmful-benign pairs at c=0 ===")
    for key, (label, _) in ARMS.items():
        for mode in ("contrastive", "single"):
            res = harmful_choice_rate_at_zero(arms[key][mode], pairs)
            if res:
                print(f"  {label} / {mode}: {res[0]:.3f} (n={res[1]})")


if __name__ == "__main__":
    main()
