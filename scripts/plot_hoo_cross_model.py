"""Cross-model HOO topic generalization: bar chart comparing in-dist vs cross-topic.

Three models on x-axis, two bars each (in-dist val vs cross-topic HOO).
Left panel: Pearson r. Right panel: pairwise accuracy.

Usage:
    python scripts/plot_hoo_cross_model.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

GEMMA3_RAW = "/tmp/gemma3_hoo_raw.json"
GEMMA2_RAW = "/tmp/gemma2_hoo_raw.json"
ST_BASELINE = "/tmp/gemma3_hoo_st.json"
UNIFIED_METRICS = "/tmp/unified_metrics.json"

OUTPUT_DIR = Path("experiments/probe_generalization/gemma2_base/assets")

# Best layers per model
GEMMA3_BEST = 31
GEMMA2_BEST = 23
ST_LAYER = 0


def load(path: str):
    with open(path) as f:
        return json.load(f)


def extract_fold_metrics(summary: dict, layer: int) -> dict:
    key = f"ridge_L{layer}"
    val_rs, hoo_rs = [], []
    for f in summary["folds"]:
        if key in f["layers"]:
            val_rs.append(f["layers"][key]["val_r"])
            hoo_rs.append(f["layers"][key]["hoo_r"])
    return {"val_r": val_rs, "hoo_r": hoo_rs}


def extract_unified_acc(unified: list[dict], condition_key: str) -> list[float]:
    return [
        fold["conditions"][condition_key]["pairwise_acc"]
        for fold in unified
        if condition_key in fold["conditions"]
    ]


def main():
    gemma3 = load(GEMMA3_RAW)
    gemma2 = load(GEMMA2_RAW)
    st = load(ST_BASELINE)
    unified = load(UNIFIED_METRICS)

    # Extract Pearson r
    g3_r = extract_fold_metrics(gemma3, GEMMA3_BEST)
    g2_r = extract_fold_metrics(gemma2, GEMMA2_BEST)
    st_r = extract_fold_metrics(st, ST_LAYER)

    # Extract pairwise accuracy from unified metrics
    g3_hoo_acc = extract_unified_acc(unified, f"Ridge raw_L{GEMMA3_BEST}")
    st_hoo_acc = extract_unified_acc(unified, f"Content baseline_L{ST_LAYER}")

    models = ["Gemma-3 27B IT\n(L31)", "Gemma-2 27B Base\n(L23)", "Content\nBaseline"]
    colors_val = "#a8d8ea"
    colors_hoo = "#3498db"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    x = np.arange(len(models))
    width = 0.3

    # --- Left: Pearson r ---
    val_r_means = [np.mean(g3_r["val_r"]), np.mean(g2_r["val_r"]), np.mean(st_r["val_r"])]
    hoo_r_means = [np.mean(g3_r["hoo_r"]), np.mean(g2_r["hoo_r"]), np.mean(st_r["hoo_r"])]
    hoo_r_stds = [np.std(g3_r["hoo_r"]), np.std(g2_r["hoo_r"]), np.std(st_r["hoo_r"])]

    bars_val = ax1.bar(x - width / 2, val_r_means, width, label="In-distribution (CV)",
                       color=colors_val, edgecolor="black", linewidth=0.5)
    bars_hoo = ax1.bar(x + width / 2, hoo_r_means, width, yerr=hoo_r_stds,
                       label="Cross-topic HOO", color=colors_hoo, edgecolor="black",
                       linewidth=0.5, capsize=3)

    for bars in [bars_val, bars_hoo]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Pearson r")
    ax1.set_title("Pearson r")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    # --- Right: Pairwise accuracy (HOO only) ---
    acc_data = [
        (np.mean(g3_hoo_acc), np.std(g3_hoo_acc)),
        (None, None),  # Gemma-2 TBD
        (np.mean(st_hoo_acc), np.std(st_hoo_acc)),
    ]

    for i, (mean, std) in enumerate(acc_data):
        if mean is not None:
            ax2.bar(i, mean, width, yerr=std, color=colors_hoo,
                    edgecolor="black", linewidth=0.5, capsize=3)
            ax2.text(i, mean + 0.01, f"{mean:.2f}",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")
        else:
            ax2.text(i, 0.52, "TBD", ha="center", va="bottom",
                     fontsize=9, fontstyle="italic", color="gray")

    ax2.axhline(y=0.5, color="gray", linewidth=1, linestyle="--", label="Chance (0.50)")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Pairwise accuracy")
    ax2.set_title("Pairwise Accuracy (cross-topic HOO)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    fig.suptitle("Cross-Topic Generalization at Best Layer", fontsize=13, y=1.01)
    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = OUTPUT_DIR / "plot_021726_cross_model_bar.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
