import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np

DATA_PATH = Path("/workspace/repo/experiments/truth_probes/error_prefill/per_token_scoring/scored_tokens.json")
ASSETS_DIR = Path("/workspace/repo/experiments/truth_probes/error_prefill/per_token_scoring/assets")

PROBE = "tb-5"
LAYER = "L39"


def text_color_for_background(rgba):
    r, g, b = rgba[0], rgba[1], rgba[2]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if luminance < 0.5 else "black"


def draw_token_row(ax, tokens, scores, cmap, norm, y_center=0.5, box_height=0.6):
    ax.set_xlim(0, len(tokens))
    ax.set_ylim(0, 1)
    ax.axis("off")

    for i, (tok, score) in enumerate(zip(tokens, scores)):
        rgba = cmap(norm(score))
        rect = plt.Rectangle(
            (i, y_center - box_height / 2),
            1,
            box_height,
            facecolor=rgba,
            edgecolor="gray",
            linewidth=0.5,
        )
        ax.add_patch(rect)
        tc = text_color_for_background(rgba)
        display = tok.replace("\n", "\\n")
        ax.text(
            i + 0.5,
            y_center,
            display,
            ha="center",
            va="center",
            fontsize=7,
            color=tc,
            fontfamily="monospace",
            clip_on=True,
        )


def plot_pair(correct_entry, incorrect_entry, pair_idx):
    entity = correct_entry["entity"]

    c_tokens = correct_entry["token_strings"]
    c_scores = correct_entry["scores"][PROBE][LAYER]
    i_tokens = incorrect_entry["token_strings"]
    i_scores = incorrect_entry["scores"][PROBE][LAYER]

    all_scores = c_scores + i_scores
    vmin = min(all_scores)
    vmax = max(all_scores)
    abs_max = max(abs(vmin), abs(vmax))
    norm = mcolors.Normalize(vmin=-abs_max, vmax=abs_max)
    cmap = plt.get_cmap("RdBu_r")

    max_tokens = max(len(c_tokens), len(i_tokens))
    fig_width = max(max_tokens * 1.2 + 2, 6)
    fig_height = 4

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(fig_width, fig_height), gridspec_kw={"hspace": 0.3}
    )

    fig.suptitle(entity, fontsize=12, fontweight="bold", y=0.98)

    ax_top.set_title("Correct answer", fontsize=9, loc="left", pad=4)
    draw_token_row(ax_top, c_tokens, c_scores, cmap, norm)

    ax_bot.set_title("Incorrect answer", fontsize=9, loc="left", pad=4)
    draw_token_row(ax_bot, i_tokens, i_scores, cmap, norm)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_top, ax_bot], location="bottom", shrink=0.6, pad=0.12, aspect=30)
    cbar.set_label(f"Probe score ({PROBE} / {LAYER})", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.tight_layout(rect=[0, 0.08, 1, 0.95])

    out_path = ASSETS_DIR / f"plot_031226_token_scores_pair_{pair_idx:03d}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    with open(DATA_PATH) as f:
        data = json.load(f)

    pairs: dict[str, dict[str, dict]] = {}
    for entry in data:
        ex_id = entry["true_ex_id"]
        cond = entry["answer_condition"]
        if ex_id not in pairs:
            pairs[ex_id] = {}
        pairs[ex_id][cond] = entry

    sorted_ids = sorted(pairs.keys())
    assert len(sorted_ids) == 50, f"Expected 50 pairs, got {len(sorted_ids)}"

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    for idx, ex_id in enumerate(sorted_ids, start=1):
        pair = pairs[ex_id]
        out = plot_pair(pair["correct"], pair["incorrect"], idx)
        print(f"[{idx:02d}/50] {out.name}  entity={pair['correct']['entity']}")

    print(f"\nDone. {len(sorted_ids)} plots saved to {ASSETS_DIR}")


if __name__ == "__main__":
    main()
