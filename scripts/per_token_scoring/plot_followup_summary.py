"""Generate summary plots for the per-token followup experiment."""
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

DATA_PATH = Path("experiments/truth_probes/error_prefill/per_token_followup/scored_tokens.json")
ASSETS_DIR = DATA_PATH.parent / "assets"

PROBE = "tb-2"
LAYER = "L53"
FOLLOWUP_TYPES = ["neutral", "presupposes", "challenge"]
FT_COLORS = {"neutral": "#4477AA", "presupposes": "#228833", "challenge": "#EE6677"}
FT_LABELS = {
    "neutral": 'Neutral ("Thank you.")',
    "presupposes": "Presupposes answer",
    "challenge": '"Are you sure about that?"',
}


def load_data():
    with open(DATA_PATH) as f:
        return json.load(f)


def cohens_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    mean_a, mean_b = np.mean(a), np.mean(b)
    std_a, std_b = np.std(a, ddof=1), np.std(b, ddof=1)
    pooled = math.sqrt((std_a**2 + std_b**2) / 2)
    if pooled == 0:
        return float("nan")
    return (mean_a - mean_b) / pooled


def plot_followup_cohens_d(data):
    """Position-wise Cohen's d within the followup span, by follow-up type.

    x-axis: token position relative to assistant/followup boundary (0 = <end_of_turn>).
    y-axis: Cohen's d (correct - incorrect).
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for ft in FOLLOWUP_TYPES:
        c_entries = [d for d in data if d["followup_type"] == ft and d["answer_condition"] == "correct"]
        i_entries = [d for d in data if d["followup_type"] == ft and d["answer_condition"] == "incorrect"]

        max_followup = max(d["n_followup_tokens"] for d in c_entries + i_entries)

        positions = []
        ds = []
        labels = []
        for pos in range(max_followup):
            c_vals = [
                d["scores"][PROBE][LAYER][d["n_assistant_tokens"] + pos]
                for d in c_entries
                if d["n_followup_tokens"] > pos
            ]
            i_vals = [
                d["scores"][PROBE][LAYER][d["n_assistant_tokens"] + pos]
                for d in i_entries
                if d["n_followup_tokens"] > pos
            ]
            if len(c_vals) < 3 or len(i_vals) < 3:
                break
            d_val = cohens_d(c_vals, i_vals)
            positions.append(pos)
            ds.append(d_val)
            # Get token label from first correct entry
            tok = c_entries[0]["token_strings"][c_entries[0]["n_assistant_tokens"] + pos]
            labels.append(tok.strip() if tok.strip() else repr(tok))

        ax.plot(positions, ds, color=FT_COLORS[ft], label=FT_LABELS[ft], linewidth=2, marker="o", markersize=4)

    # Mark boundary between turn-template tokens and content tokens
    ax.axvline(x=4.5, color="gray", linewidth=1, linestyle=":", alpha=0.5)
    ax.annotate(
        "turn boundary\ntokens",
        xy=(2, -1.3),
        ha="center",
        fontsize=7,
        color="gray",
    )
    ax.annotate(
        "follow-up\ncontent",
        xy=(7, -1.3),
        ha="center",
        fontsize=7,
        color="gray",
    )

    ax.axhline(y=0, color="black", linewidth=0.5, alpha=0.3)
    ax.set_xlabel("Token position in follow-up span", fontsize=10)
    ax.set_ylabel("Cohen's d (correct - incorrect)", fontsize=10)
    ax.set_title(f"Signal separation by follow-up type ({PROBE}, {LAYER})", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(-1.5, 5.0)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    out = ASSETS_DIR / "plot_031226_followup_position_cohens_d.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_mean_score_trajectories(data):
    """Mean probe score across the full sequence (assistant + followup), split by condition and followup type.

    Shows how the signal evolves from assistant response through turn boundary into follow-up.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

    for ax, ft in zip(axes, FOLLOWUP_TYPES):
        c_entries = [d for d in data if d["followup_type"] == ft and d["answer_condition"] == "correct"]
        i_entries = [d for d in data if d["followup_type"] == ft and d["answer_condition"] == "incorrect"]

        for entries, color, label in [
            (c_entries, "#228833", "Correct"),
            (i_entries, "#EE6677", "Incorrect"),
        ]:
            # Compute mean score at each position
            max_len = max(len(d["token_strings"]) for d in entries)
            means = []
            positions = []
            for pos in range(max_len):
                vals = [d["scores"][PROBE][LAYER][pos] for d in entries if len(d["scores"][PROBE][LAYER]) > pos]
                if len(vals) < 3:
                    break
                means.append(np.mean(vals))
                positions.append(pos)

            ax.plot(positions, means, color=color, label=label, linewidth=2)

            # Add SEM band
            sems = []
            for pos in range(len(positions)):
                vals = [d["scores"][PROBE][LAYER][pos] for d in entries if len(d["scores"][PROBE][LAYER]) > pos]
                sems.append(np.std(vals, ddof=1) / math.sqrt(len(vals)))
            means_arr = np.array(means)
            sems_arr = np.array(sems)
            ax.fill_between(positions, means_arr - sems_arr, means_arr + sems_arr, color=color, alpha=0.15)

        # Mark the typical boundary position
        # assistant tokens vary, but we can mark the mean
        mean_n_asst = np.mean([d["n_assistant_tokens"] for d in c_entries + i_entries])
        ax.axvline(x=mean_n_asst, color="gray", linewidth=1.5, linestyle="--", alpha=0.6)
        ax.text(mean_n_asst + 0.3, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else -20, "turn\nboundary",
                fontsize=7, color="gray", va="bottom")

        ax.set_title(FT_LABELS[ft], fontsize=10)
        ax.set_xlabel("Token position", fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.axhline(y=0, color="black", linewidth=0.5, alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel(f"Mean probe score ({PROBE}, {LAYER})", fontsize=9)

    fig.suptitle("Score trajectories across assistant response and follow-up", fontsize=11, y=1.02)
    fig.tight_layout()
    out = ASSETS_DIR / "plot_031226_followup_score_trajectories.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    data = load_data()
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    plot_followup_cohens_d(data)
    plot_mean_score_trajectories(data)
    print("Done.")


if __name__ == "__main__":
    main()
