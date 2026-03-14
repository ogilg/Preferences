"""Score all tokens with trained probes for followup experiment and generate visualizations."""
import json
import sys
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

from src.probes.core.activations import load_span_activations

# --- Config ---
ASSISTANT_ACTS_PATH = Path("activations/gemma_3_27b_error_prefill/activations_assistant_all.npz")
FOLLOWUP_ACTS_PATH = Path("activations/gemma_3_27b_error_prefill/activations_followup_all.npz")
DATA_PATH = Path("data/creak/error_prefill_followup_60.json")
OUTPUT_DIR = Path("experiments/truth_probes/error_prefill/per_token_followup")
ASSETS_DIR = OUTPUT_DIR / "assets"
SCORED_PATH = OUTPUT_DIR / "scored_tokens.json"

LAYERS = [25, 32, 39, 46, 53]
PROBE_DIRS = {
    "tb-2": Path("results/probes/heldout_eval_gemma3_tb-2/probes"),
    "tb-5": Path("results/probes/heldout_eval_gemma3_tb-5/probes"),
}
MODEL_NAME = "google/gemma-3-27b-it"

FOLLOWUP_TYPES = ["neutral", "presupposes", "challenge"]
CONDITIONS = ["correct", "incorrect"]


def load_data():
    with open(DATA_PATH) as f:
        return json.load(f)


def load_probes():
    probes = {}
    for probe_name, probe_dir in PROBE_DIRS.items():
        probes[probe_name] = {}
        for layer in LAYERS:
            weights = np.load(probe_dir / f"probe_ridge_L{layer}.npy")
            probes[probe_name][layer] = weights
            print(f"  Loaded {probe_name} L{layer}: {weights.shape}")
    return probes


def get_token_strings(tokenizer, messages):
    """Get token strings for assistant_all and followup_all spans.

    Uses differential formatting to find the exact token ranges,
    matching what the extraction code does.
    """
    # Format messages incrementally to find spans
    # assistant_all: tokens of assistant content
    # followup_all: tokens from assistant content end to followup content end

    # Find assistant content span
    user_msg = [messages[0]]
    prompt_with_header = tokenizer.apply_chat_template(
        user_msg, tokenize=False, add_generation_prompt=True
    )
    before_ids = tokenizer.encode(prompt_with_header, add_special_tokens=False)
    asst_start = len(before_ids)

    through_asst_content = prompt_with_header + messages[1]["content"]
    through_ids = tokenizer.encode(through_asst_content, add_special_tokens=False)
    asst_end = len(through_ids)

    # followup_all starts at asst_end
    # Format through followup message
    full_formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    # Find where followup content ends
    followup_content = messages[2]["content"]
    content_char_start = full_formatted.rindex(followup_content)
    through_followup = full_formatted[:content_char_start] + followup_content
    through_followup_ids = tokenizer.encode(through_followup, add_special_tokens=False)
    followup_end = len(through_followup_ids)

    # Get all token IDs for the full sequence
    full_ids = tokenizer.encode(full_formatted, add_special_tokens=False)

    # Extract token strings for each span
    asst_token_ids = full_ids[asst_start:asst_end]
    followup_token_ids = full_ids[asst_end:followup_end]

    asst_tokens = [tokenizer.decode([tid]) for tid in asst_token_ids]
    followup_tokens = [tokenizer.decode([tid]) for tid in followup_token_ids]

    return asst_tokens, followup_tokens


def score_all_tokens(data, probes, asst_acts, followup_acts, asst_ids, followup_ids, tokenizer):
    """Score all tokens with all probes."""
    data_by_id = {d["task_id"]: d for d in data}

    # Build index for both activation sets
    asst_id_to_idx = {str(tid): i for i, tid in enumerate(asst_ids)}
    followup_id_to_idx = {str(tid): i for i, tid in enumerate(followup_ids)}

    results = []
    for task_id_raw in asst_ids:
        task_id = str(task_id_raw)
        record = data_by_id[task_id]
        asst_idx = asst_id_to_idx[task_id]
        followup_idx = followup_id_to_idx[task_id]

        # Get token strings
        asst_tokens, followup_tokens = get_token_strings(tokenizer, record["messages"])

        # Get activation token counts
        n_asst_act = asst_acts[LAYERS[0]][asst_idx].shape[0]
        n_followup_act = followup_acts[LAYERS[0]][followup_idx].shape[0]

        # Align token counts (may differ slightly due to tokenization)
        n_asst = min(n_asst_act, len(asst_tokens))
        n_followup = min(n_followup_act, len(followup_tokens))

        if n_asst_act != len(asst_tokens):
            print(f"  WARN {task_id}: asst acts={n_asst_act} vs tokens={len(asst_tokens)}")
        if n_followup_act != len(followup_tokens):
            print(f"  WARN {task_id}: followup acts={n_followup_act} vs tokens={len(followup_tokens)}")

        asst_tokens = asst_tokens[:n_asst]
        followup_tokens = followup_tokens[:n_followup]
        all_tokens = asst_tokens + followup_tokens

        # Score with all probes
        scores_by_probe = {}
        for probe_name in PROBE_DIRS:
            scores_by_probe[probe_name] = {}
            for layer in LAYERS:
                weights = probes[probe_name][layer]
                coefs = weights[:-1]
                intercept = weights[-1]

                # Concatenate assistant and followup activations
                a_acts = asst_acts[layer][asst_idx][:n_asst]
                f_acts = followup_acts[layer][followup_idx][:n_followup]
                all_acts = np.concatenate([a_acts, f_acts], axis=0)

                token_scores = (all_acts @ coefs + intercept).tolist()
                scores_by_probe[probe_name][f"L{layer}"] = token_scores

        results.append({
            "task_id": task_id,
            "answer_condition": record["answer_condition"],
            "followup_type": record["followup_type"],
            "entity": record["entity"],
            "true_ex_id": record["true_ex_id"],
            "token_strings": all_tokens,
            "n_assistant_tokens": n_asst,
            "n_followup_tokens": n_followup,
            "scores": scores_by_probe,
        })

    return results


def select_best_probes(results):
    """Select 2 probes with highest mean |correct - incorrect| separation."""
    probe_separations = {}

    # Group by claim + followup_type for paired comparisons
    grouped = {}
    for r in results:
        key = (r["true_ex_id"], r["followup_type"])
        if key not in grouped:
            grouped[key] = {}
        grouped[key][r["answer_condition"]] = r

    for probe_name in PROBE_DIRS:
        for layer in LAYERS:
            layer_key = f"L{layer}"
            diffs = []
            for key, pair in grouped.items():
                if "correct" not in pair or "incorrect" not in pair:
                    continue
                c_scores = pair["correct"]["scores"][probe_name][layer_key]
                i_scores = pair["incorrect"]["scores"][probe_name][layer_key]
                # Mean absolute difference across all tokens
                min_len = min(len(c_scores), len(i_scores))
                for t in range(min_len):
                    diffs.append(abs(c_scores[t] - i_scores[t]))

            mean_diff = np.mean(diffs) if diffs else 0.0
            probe_separations[(probe_name, layer)] = mean_diff
            print(f"  {probe_name} L{layer}: mean |diff| = {mean_diff:.4f}")

    # Sort and pick top 2
    ranked = sorted(probe_separations.items(), key=lambda x: x[1], reverse=True)
    best_two = [ranked[0][0], ranked[1][0]]
    print(f"\nBest 2 probes: {best_two[0]} (diff={ranked[0][1]:.4f}), {best_two[1]} (diff={ranked[1][1]:.4f})")
    return best_two


def text_color_for_background(rgba):
    r, g, b = rgba[0], rgba[1], rgba[2]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if luminance < 0.5 else "black"


def draw_token_row(ax, tokens, scores, cmap, norm, n_assistant, y_center=0.5, box_height=0.6):
    """Draw a row of colored tokens. Adds a vertical line at the assistant/followup boundary."""
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
            fontsize=9,
            color=tc,
            fontfamily="monospace",
            clip_on=True,
        )

    # Draw boundary line between assistant and followup
    ax.axvline(x=n_assistant, color="blue", linewidth=2, linestyle="--", alpha=0.7)


def plot_claim(results_by_key, claim_id, probe_name, layer, claim_idx, entity):
    """Generate a 6-row figure for one claim and one probe."""
    layer_key = f"L{layer}"
    probe_label = f"{probe_name}_L{layer}"

    # Collect all scores for color normalization
    all_scores = []
    rows = []
    for followup_type in FOLLOWUP_TYPES:
        for condition in CONDITIONS:
            key = (claim_id, followup_type, condition)
            r = results_by_key[key]
            scores = r["scores"][probe_name][layer_key]
            all_scores.extend(scores)
            rows.append((condition, followup_type, r))

    abs_max = max(abs(min(all_scores)), abs(max(all_scores)))
    norm = mcolors.Normalize(vmin=-abs_max, vmax=abs_max)
    cmap = plt.get_cmap("RdYlGn")

    max_tokens = max(len(r["token_strings"]) for _, _, r in rows)
    fig_width = max(max_tokens * 1.2 + 2, 10)
    fig_height = 6 * 1.2 + 2

    fig, axes = plt.subplots(6, 1, figsize=(fig_width, fig_height), gridspec_kw={"hspace": 0.5})
    fig.suptitle(f"{entity}  —  {probe_label}", fontsize=13, fontweight="bold", y=0.98)

    for row_idx, (condition, followup_type, r) in enumerate(rows):
        ax = axes[row_idx]
        tokens = r["token_strings"]
        scores = r["scores"][probe_name][layer_key]
        n_asst = r["n_assistant_tokens"]

        ax.set_title(f"{condition} + {followup_type}", fontsize=10, loc="left", pad=4)
        draw_token_row(ax, tokens, scores, cmap, norm, n_asst)

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), location="bottom", shrink=0.5, pad=0.06, aspect=30)
    cbar.set_label(f"Probe score ({probe_label})", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout(rect=[0, 0.05, 1, 0.95])

    out_path = ASSETS_DIR / f"plot_031226_followup_tokens_{probe_label}_claim_{claim_idx:03d}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_report(best_probes, claims_sorted, entities, probe_separations_text):
    """Write the markdown report with all figures inline."""
    lines = [
        "# Per-Token Follow-Up Probe Scoring",
        "",
        "## Probe selection",
        "",
        probe_separations_text,
        "",
        "## Token-level visualizations",
        "",
        "Each figure shows 6 rows: correct/incorrect x 3 follow-up types (neutral, presupposes, challenge). "
        "The blue dashed line marks the boundary between assistant response tokens and follow-up tokens "
        "(turn boundary + user follow-up). Tokens are colored on a RdYlGn scale (red = low/negative, green = high/positive).",
        "",
    ]

    for probe_name, layer in best_probes:
        probe_label = f"{probe_name}_L{layer}"
        lines.append(f"### Probe: {probe_label}")
        lines.append("")
        for idx, claim_id in enumerate(claims_sorted, start=1):
            entity = entities[claim_id]
            fname = f"plot_031226_followup_tokens_{probe_label}_claim_{idx:03d}.png"
            lines.append(f"#### Claim {idx}: {entity}")
            lines.append("")
            lines.append(f"![{entity}](assets/{fname})")
            lines.append("")

    report_path = OUTPUT_DIR / "per_token_followup_report.md"
    report_path.write_text("\n".join(lines))
    print(f"Report written to {report_path}")


def main():
    print("Loading data...")
    data = load_data()

    print("Loading probes...")
    probes = load_probes()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading assistant span activations...")
    asst_ids, asst_acts = load_span_activations(ASSISTANT_ACTS_PATH, layers=LAYERS)

    print("Loading followup span activations...")
    followup_ids, followup_acts = load_span_activations(FOLLOWUP_ACTS_PATH, layers=LAYERS)

    print(f"Scoring {len(asst_ids)} tasks with {len(PROBE_DIRS) * len(LAYERS)} probes...")
    results = score_all_tokens(data, probes, asst_acts, followup_acts, asst_ids, followup_ids, tokenizer)

    # Save scored results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SCORED_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} scored tasks to {SCORED_PATH}")

    # Select best probes
    print("\nComputing probe separations...")
    best_probes = select_best_probes(results)

    # Build lookup for plotting
    results_by_key = {}
    entities = {}
    for r in results:
        key = (r["true_ex_id"], r["followup_type"], r["answer_condition"])
        results_by_key[key] = r
        entities[r["true_ex_id"]] = r["entity"]

    claims_sorted = sorted(entities.keys())

    # Generate plots
    print(f"\nGenerating {len(claims_sorted) * len(best_probes)} plots...")
    for probe_name, layer in best_probes:
        probe_label = f"{probe_name}_L{layer}"
        for idx, claim_id in enumerate(claims_sorted, start=1):
            entity = entities[claim_id]
            out = plot_claim(results_by_key, claim_id, probe_name, layer, idx, entity)
            print(f"  [{idx:02d}/10] {probe_label}: {out.name}")

    # Build probe separations text for report
    sep_lines = []
    grouped = {}
    for r in results:
        key = (r["true_ex_id"], r["followup_type"])
        if key not in grouped:
            grouped[key] = {}
        grouped[key][r["answer_condition"]] = r

    sep_lines.append("| Probe | Mean |correct - incorrect| |")
    sep_lines.append("|-------|-------------------------|")
    all_seps = []
    for probe_name in PROBE_DIRS:
        for layer in LAYERS:
            layer_key = f"L{layer}"
            diffs = []
            for key, pair in grouped.items():
                if "correct" not in pair or "incorrect" not in pair:
                    continue
                c_scores = pair["correct"]["scores"][probe_name][layer_key]
                i_scores = pair["incorrect"]["scores"][probe_name][layer_key]
                min_len = min(len(c_scores), len(i_scores))
                for t in range(min_len):
                    diffs.append(abs(c_scores[t] - i_scores[t]))
            mean_diff = np.mean(diffs) if diffs else 0.0
            label = f"{probe_name} L{layer}"
            selected = any(p == probe_name and l == layer for p, l in best_probes)
            marker = " **[selected]**" if selected else ""
            sep_lines.append(f"| {label} | {mean_diff:.4f}{marker} |")
            all_seps.append((label, mean_diff))

    probe_separations_text = "\n".join(sep_lines)

    # Generate report
    generate_report(best_probes, claims_sorted, entities, probe_separations_text)

    print("\nDone!")


if __name__ == "__main__":
    main()
