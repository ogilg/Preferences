"""Plot dose-response curves broken out by mu-bin for each format × position.

Points with <50% parse rate are dropped. Marker size scales with parse rate.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from dotenv import load_dotenv

load_dotenv()

EXP_DIR = Path(__file__).resolve().parent.parent.parent / "experiments" / "steering" / "stated_steering" / "format_replication"
RESULTS_DIR = EXP_DIR / "results"
ASSETS_DIR = EXP_DIR / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

MEAN_L31_NORM = 52820.0
MIN_PARSE_RATE = 0.95

FORMATS = ["qualitative_ternary", "adjective_pick", "anchored_simple_1_5"]
POSITIONS = ["task_tokens", "generation", "last_token"]
FORMAT_LABELS = {
    "qualitative_ternary": "Ternary",
    "adjective_pick": "Adjective",
    "anchored_simple_1_5": "Anchored",
}
# Raw scale ranges for 0-1 normalization
FORMAT_SCALE = {
    "qualitative_ternary": (1, 3),
    "adjective_pick": (1, 10),
    "anchored_simple_1_5": (1, 5),
}


def normalize_score(score: float, fmt: str) -> float:
    lo, hi = FORMAT_SCALE[fmt]
    return (score - lo) / (hi - lo)
POSITION_LABELS = {
    "task_tokens": "Task tokens",
    "generation": "During generation",
    "last_token": "Last prompt token",
}


def load_coherence_mask() -> dict[str, dict[str, bool]]:
    """Load format-independent coherence mask: {position: {coefficient_str: bool}}."""
    path = RESULTS_DIR / "coherence_mask.json"
    with open(path) as f:
        return json.load(f)


def is_coherent(mask: dict, pos: str, coefficient: float) -> bool:
    return mask.get(pos, {}).get(str(coefficient), True)


COHERENCE_MASK = load_coherence_mask()


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_tasks() -> dict[str, dict]:
    with open(RESULTS_DIR / "tasks.json") as f:
        tasks = json.load(f)
    return {t["task_id"]: t for t in tasks}


def compute_bin_stats(records: list[dict], fmt: str, pos: str) -> dict[int, dict[float, dict]]:
    """Group by mu_bin × coefficient, return {bin: {coef: {mean, parse_rate, n_total}}}."""
    raw: dict[int, dict[float, dict]] = defaultdict(lambda: defaultdict(lambda: {"valid": [], "total": 0}))
    for r in [r for r in records if is_coherent(COHERENCE_MASK, pos, r["coefficient"])]:
        bucket = raw[r["mu_bin"]][r["coefficient"]]
        bucket["valid"].extend(s for s in r["scores"] if s is not None)
        bucket["total"] += len(r["scores"])

    result: dict[int, dict[float, dict]] = {}
    for b, coef_data in raw.items():
        result[b] = {}
        for c, d in coef_data.items():
            pr = len(d["valid"]) / d["total"] if d["total"] else 0
            raw_mean = np.mean(d["valid"]) if d["valid"] else np.nan
            result[b][c] = {
                "mean": normalize_score(raw_mean, fmt) if d["valid"] else np.nan,
                "parse_rate": pr,
                "n_total": d["total"],
            }
    return result


def compute_aggregate_stats(records: list[dict], fmt: str, pos: str) -> dict[float, dict]:
    """Aggregate across all bins for a single format × position."""
    raw: dict[float, dict] = defaultdict(lambda: {"valid": [], "total": 0})
    for r in [r for r in records if is_coherent(COHERENCE_MASK, pos, r["coefficient"])]:
        bucket = raw[r["coefficient"]]
        bucket["valid"].extend(s for s in r["scores"] if s is not None)
        bucket["total"] += len(r["scores"])

    lo, hi = FORMAT_SCALE[fmt]
    scale = hi - lo
    result = {}
    for c, d in raw.items():
        pr = len(d["valid"]) / d["total"] if d["total"] else 0
        raw_mean = np.mean(d["valid"]) if d["valid"] else np.nan
        raw_sem = np.std(d["valid"]) / np.sqrt(len(d["valid"])) if d["valid"] else np.nan
        result[c] = {
            "mean": normalize_score(raw_mean, fmt) if d["valid"] else np.nan,
            "sem": raw_sem / scale if d["valid"] else np.nan,
            "parse_rate": pr,
            "n_valid": len(d["valid"]),
        }
    return result


def plot_dose_response_with_parse(pcts, means, parse_rates, ax, color, linewidth=1.5, label=None):
    """Plot dose-response, dropping points below MIN_PARSE_RATE, sizing markers by parse rate."""
    # Draw line segments only between reliable points
    reliable_mask = np.array(parse_rates) >= MIN_PARSE_RATE
    reliable_pcts = np.array(pcts)[reliable_mask]
    reliable_means = np.array(means)[reliable_mask]
    reliable_pr = np.array(parse_rates)[reliable_mask]

    if len(reliable_pcts) == 0:
        return

    # Line through reliable points
    ax.plot(reliable_pcts, reliable_means, color=color, linewidth=linewidth,
            alpha=0.8, label=label)

    # Markers sized by parse rate (min 2, max 8)
    sizes = 2 + 6 * reliable_pr
    ax.scatter(reliable_pcts, reliable_means, s=sizes ** 2, color=color,
               alpha=0.9, zorder=5, edgecolors="none")


def plot_by_bin_grid():
    """3×3 grid (format × position), each subplot has 10 colored lines for mu-bins."""
    fig, axes = plt.subplots(3, 3, figsize=(20, 14), sharex=True, sharey=True)

    cmap = plt.cm.RdYlGn
    bin_colors = [cmap(i / 9) for i in range(10)]

    for row, fmt in enumerate(FORMATS):
        for col, pos in enumerate(POSITIONS):
            ax = axes[row, col]
            path = RESULTS_DIR / f"results_{fmt}_{pos}.jsonl"
            if not path.exists():
                ax.set_visible(False)
                continue

            records = load_jsonl(path)
            bin_stats = compute_bin_stats(records, fmt, pos)

            for b in sorted(bin_stats.keys()):
                coefs_sorted = sorted(bin_stats[b].keys())
                pcts = [c / MEAN_L31_NORM * 100 for c in coefs_sorted]
                means = [bin_stats[b][c]["mean"] for c in coefs_sorted]
                parse_rates = [bin_stats[b][c]["parse_rate"] for c in coefs_sorted]

                plot_dose_response_with_parse(pcts, means, parse_rates, ax,
                                              color=bin_colors[b], linewidth=1.0)

            if row == 0:
                ax.set_title(POSITION_LABELS[pos], fontsize=12)
            if col == 0:
                ax.set_ylabel(f"{FORMAT_LABELS[fmt]}\nNormalised rating (0–1)", fontsize=10)
            if row == 2:
                ax.set_xlabel("Coefficient (% of norm)")
            ax.axvline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
            ax.grid(True, alpha=0.3)

    handles = [Line2D([0], [0], color=bin_colors[b], linewidth=2, label=f"Bin {b}")
               for b in range(10)]
    # Add parse rate legend entries
    handles.append(Line2D([0], [0], marker="o", color="gray", markersize=3,
                          linestyle="none", label="50% parse"))
    handles.append(Line2D([0], [0], marker="o", color="gray", markersize=7,
                          linestyle="none", label="100% parse"))
    fig.legend(handles=handles, loc="center right", fontsize=9,
               title="Preference bin\n(0=low, 9=high)\n\nMarker size\n= parse rate",
               title_fontsize=9, bbox_to_anchor=(0.99, 0.5))

    fig.suptitle(f"Dose-Response by Preference Bin\n(points with <{MIN_PARSE_RATE:.0%} parse rate omitted; marker size ∝ parse rate)",
                 fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 0.89, 0.95])

    out = ASSETS_DIR / "plot_022426_dose_response_by_bin.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_aggregate_dose_response():
    """3×3 grid (format × position) with dose-response lines and parse rate bars on twin axis."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 14), sharex=True, sharey=True)
    colors = {"task_tokens": "#1f77b4", "generation": "#ff7f0e", "last_token": "#2ca02c"}

    for row, fmt in enumerate(FORMATS):
        for col, pos in enumerate(POSITIONS):
            ax = axes[row, col]
            path = RESULTS_DIR / f"results_{fmt}_{pos}.jsonl"
            if not path.exists():
                ax.set_visible(False)
                continue

            records = load_jsonl(path)

            # Unfiltered stats for parse rate bars (show all coefficients)
            all_coefs = sorted(set(r["coefficient"] for r in records))
            all_pcts = [c / MEAN_L31_NORM * 100 for c in all_coefs]
            # Compute parse rates from raw data
            pr_raw: dict[float, dict] = defaultdict(lambda: {"valid": 0, "total": 0})
            for r in records:
                pr_raw[r["coefficient"]]["valid"] += sum(1 for s in r["scores"] if s is not None)
                pr_raw[r["coefficient"]]["total"] += len(r["scores"])
            all_parse_rates = [
                pr_raw[c]["valid"] / pr_raw[c]["total"] if pr_raw[c]["total"] else 0
                for c in all_coefs
            ]
            all_coherent = [is_coherent(COHERENCE_MASK, pos, c) for c in all_coefs]

            # Filtered stats for score line (coherent coefficients only)
            stats = compute_aggregate_stats(records, fmt, pos)
            coefs_sorted = sorted(stats.keys())
            pcts = [c / MEAN_L31_NORM * 100 for c in coefs_sorted]
            means = [stats[c]["mean"] for c in coefs_sorted]
            sems = [stats[c]["sem"] for c in coefs_sorted]

            color = colors[pos]

            # Parse rate bars on twin axis (behind everything)
            ax2 = ax.twinx()
            bar_width = (max(all_pcts) - min(all_pcts)) / len(all_pcts) * 0.7 if len(all_pcts) > 1 else 0.5
            bar_colors = [
                "#ffcccc" if not coh else ("#e0e0e0" if pr >= MIN_PARSE_RATE else "#ffcccc")
                for pr, coh in zip(all_parse_rates, all_coherent)
            ]
            # Hatching for incoherent coefficients
            ax2.bar(all_pcts, all_parse_rates, width=bar_width, color=bar_colors, alpha=0.4, zorder=1)
            # Mark incoherent with X
            for p, pr, coh in zip(all_pcts, all_parse_rates, all_coherent):
                if not coh:
                    ax2.text(p, pr + 0.03, "✗", ha="center", va="bottom", fontsize=8,
                             color="red", fontweight="bold", alpha=0.7)
            ax2.set_ylim(0, 1.15)
            ax2.axhline(MIN_PARSE_RATE, color="red", linewidth=0.5, linestyle=":", alpha=0.4)
            if col == 2:
                ax2.set_ylabel("Parse rate", fontsize=9, color="gray")
                ax2.tick_params(axis="y", labelsize=8, colors="gray")
            else:
                ax2.set_yticklabels([])
                ax2.tick_params(axis="y", length=0)

            # Score line on primary axis (coherent + above parse rate threshold)
            parse_rates_line = [stats[c]["parse_rate"] for c in coefs_sorted]
            reliable = [
                i for i, (c, pr) in enumerate(zip(coefs_sorted, parse_rates_line))
                if pr >= MIN_PARSE_RATE
            ]
            if reliable:
                r_pcts = [pcts[i] for i in reliable]
                r_means = [means[i] for i in reliable]
                r_sems = [sems[i] for i in reliable]
                ax.plot(r_pcts, r_means, color=color, linewidth=2, alpha=0.8, zorder=10)
                ax.fill_between(r_pcts, np.array(r_means) - np.array(r_sems),
                                np.array(r_means) + np.array(r_sems),
                                color=color, alpha=0.15, zorder=9)
                ax.scatter(r_pcts, r_means, s=30, color=color,
                           alpha=0.9, zorder=11, edgecolors="none")

            ax.set_zorder(ax2.get_zorder() + 1)
            ax.patch.set_visible(False)

            if row == 0:
                ax.set_title(POSITION_LABELS[pos], fontsize=12)
            if col == 0:
                ax.set_ylabel(f"{FORMAT_LABELS[fmt]}\nNormalised rating (0–1)", fontsize=10)
            if row == 2:
                ax.set_xlabel("Coefficient (% of norm)")
            ax.axvline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
            ax.grid(True, alpha=0.3)

    fig.suptitle(f"Dose-Response: Normalised Rating vs Steering Coefficient\n(grey bars = parse rate; red bars/line = below {MIN_PARSE_RATE:.0%} threshold; score points omitted where parse <{MIN_PARSE_RATE:.0%})",
                 fontsize=12, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out = ASSETS_DIR / "plot_022426_dose_response_by_format.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_slopes_by_bin():
    """Bar chart of mean slope per mu-bin for generation and last_token positions."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharey=True)

    with open(RESULTS_DIR / "per_task_slopes.json") as f:
        all_slopes = json.load(f)

    tasks = load_tasks()

    for col, fmt in enumerate(FORMATS):
        for row_idx, pos in enumerate(["generation", "last_token"]):
            ax = axes[row_idx, col]

            entry = next((e for e in all_slopes if e["format"] == fmt and e["position"] == pos), None)
            if not entry:
                ax.set_visible(False)
                continue

            lo, hi = FORMAT_SCALE[fmt]
            scale = hi - lo
            bin_slopes: dict[int, list[float]] = defaultdict(list)
            for s in entry["slopes"]:
                task_info = tasks.get(s["task_id"])
                if task_info:
                    bin_slopes[task_info["mu_bin"]].append(s["slope"] / scale)

            bins_sorted = sorted(bin_slopes.keys())
            means = [np.mean(bin_slopes[b]) for b in bins_sorted]
            sems = [np.std(bin_slopes[b]) / np.sqrt(len(bin_slopes[b])) for b in bins_sorted]

            cmap = plt.cm.RdYlGn
            colors_list = [cmap(b / 9) for b in bins_sorted]
            ax.bar(bins_sorted, means, yerr=sems, color=colors_list, edgecolor="gray",
                   linewidth=0.5, capsize=3, alpha=0.8)
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_xlabel("Preference bin" if row_idx == 1 else "")
            if col == 0:
                ax.set_ylabel(f"Normalised slope\n({POSITION_LABELS[pos]})", fontsize=11)
            if row_idx == 0:
                ax.set_title(FORMAT_LABELS[fmt], fontsize=11)
            ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Per-Task Slope by Preference Bin\n(gemma-3-27b, ridge_L31)", fontsize=14)
    fig.tight_layout()

    out = ASSETS_DIR / "plot_022426_slopes_by_bin.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    plot_aggregate_dose_response()
    plot_by_bin_grid()
    plot_slopes_by_bin()
