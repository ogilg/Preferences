import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

DATA_DIR = "/workspace/repo/experiments/steering/program/open_ended_effects/generalization_new_prompts"
OUT_PATH = f"{DATA_DIR}/assets/plot_021426_dose_response.png"

DIMENSIONS = ["emotional_engagement", "hedging", "elaboration", "confidence"]
DIM_LABELS = ["Engagement", "Hedging", "Elaboration", "Confidence"]

with open(f"{DATA_DIR}/judge_results_original.json") as f:
    original = json.load(f)
with open(f"{DATA_DIR}/judge_results_swapped.json") as f:
    swapped = json.load(f)


def build_score_lookup(results: list[dict]) -> dict:
    """Map (prompt_id, coef, dim) -> score toward steered."""
    lookup: dict[tuple[str, int, str], int] = {}
    for r in results:
        pid = r["prompt_id"]
        coef = r["steered_coefficient"]
        for dim in DIMENSIONS:
            lookup[(pid, coef, dim)] = r[f"{dim}_score"]
    return lookup


orig_scores = build_score_lookup(original)
swap_scores = build_score_lookup(swapped)

# Collect all prompt_ids that appear in both files
all_pids = sorted(
    {pid for (pid, _, _) in orig_scores} & {pid for (pid, _, _) in swap_scores}
)


def compute_direction_asymmetry(
    magnitude: int,
) -> dict[str, list[float]]:
    """For each prompt, compute combined direction asymmetry = avg_score(-mag) - avg_score(+mag).

    "Combined" means we average original and swapped for each (prompt, coef, dim).
    """
    asymmetries: dict[str, list[float]] = {dim: [] for dim in DIMENSIONS}
    for pid in all_pids:
        for dim in DIMENSIONS:
            neg_key = (pid, -magnitude, dim)
            pos_key = (pid, magnitude, dim)
            # Need both directions in both files
            if neg_key not in orig_scores or pos_key not in orig_scores:
                continue
            if neg_key not in swap_scores or pos_key not in swap_scores:
                continue

            neg_combined = (orig_scores[neg_key] + swap_scores[neg_key]) / 2.0
            pos_combined = (orig_scores[pos_key] + swap_scores[pos_key]) / 2.0
            diff = neg_combined - pos_combined
            asymmetries[dim].append(diff)
    return asymmetries


asym_3000 = compute_direction_asymmetry(3000)
asym_2000 = compute_direction_asymmetry(2000)

# Plot
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(len(DIMENSIONS))
width = 0.35

means_3000 = [np.mean(asym_3000[d]) for d in DIMENSIONS]
sems_3000 = [np.std(asym_3000[d], ddof=1) / np.sqrt(len(asym_3000[d])) for d in DIMENSIONS]
means_2000 = [np.mean(asym_2000[d]) for d in DIMENSIONS]
sems_2000 = [np.std(asym_2000[d], ddof=1) / np.sqrt(len(asym_2000[d])) for d in DIMENSIONS]

bars_3000 = ax.bar(
    x - width / 2, means_3000, width,
    yerr=sems_3000, capsize=4,
    label=r"$\pm$3000", color="#2c5f8a", edgecolor="white", linewidth=0.5,
)
bars_2000 = ax.bar(
    x + width / 2, means_2000, width,
    yerr=sems_2000, capsize=4,
    label=r"$\pm$2000", color="#7fb3d8", edgecolor="white", linewidth=0.5,
)

ax.axhline(0, color="black", linewidth=0.8, linestyle="-")

# Sign test p-values
def sign_test_pvalue(diffs: list[float]) -> float:
    positives = sum(1 for d in diffs if d > 0)
    nonzero = sum(1 for d in diffs if d != 0)
    if nonzero == 0:
        return 1.0
    return binomtest(positives, nonzero, 0.5).pvalue


def format_p(p: float) -> str:
    if p < 0.001:
        return "p<.001"
    if p < 0.01:
        return f"p={p:.3f}"
    if p < 0.05:
        return f"p={p:.2f}"
    return f"p={p:.2f}"


for i, dim in enumerate(DIMENSIONS):
    p3 = sign_test_pvalue(asym_3000[dim])
    p2 = sign_test_pvalue(asym_2000[dim])

    # Place annotation above or below bar depending on sign
    for bar, p_val, offset_x in [
        (bars_3000[i], p3, x[i] - width / 2),
        (bars_2000[i], p2, x[i] + width / 2),
    ]:
        height = bar.get_height()
        err = sems_3000[i] if bar in bars_3000 else sems_2000[i]
        if height >= 0:
            y_pos = height + err + 0.03
            va = "bottom"
        else:
            y_pos = height - err - 0.03
            va = "top"
        ax.text(
            offset_x, y_pos, format_p(p_val),
            ha="center", va=va, fontsize=7.5, fontstyle="italic",
        )

ax.set_xticks(x)
ax.set_xticklabels(DIM_LABELS, fontsize=11)
ax.set_ylabel("Direction asymmetry\n(neg − pos steering score)", fontsize=10)
ax.set_title(r"Dose-response: $\pm$3000 vs $\pm$2000", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)

fig.tight_layout()
fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
print(f"Saved to {OUT_PATH}")

# Print summary stats
for mag, asym in [("3000", asym_3000), ("2000", asym_2000)]:
    print(f"\n=== ±{mag} ===")
    for dim, label in zip(DIMENSIONS, DIM_LABELS):
        diffs = asym[dim]
        n = len(diffs)
        pos = sum(1 for d in diffs if d > 0)
        neg = sum(1 for d in diffs if d < 0)
        zero = sum(1 for d in diffs if d == 0)
        p = sign_test_pvalue(diffs)
        print(
            f"  {label:15s}: mean={np.mean(diffs):+.3f}  SEM={np.std(diffs, ddof=1)/np.sqrt(n):.3f}  "
            f"n={n}  pos={pos} neg={neg} zero={zero}  {format_p(p)}"
        )
