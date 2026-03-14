"""Visualization functions for probe error analysis."""

from __future__ import annotations

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import pearsonr

DATE_PREFIX = datetime.now().strftime("%m%d%y")

TOPIC_COLORS = {
    "harmful_request": "#e74c3c",
    "security_legal": "#e67e22",
    "math": "#3498db",
    "knowledge_qa": "#2ecc71",
    "content_generation": "#9b59b6",
    "fiction": "#1abc9c",
    "coding": "#34495e",
    "model_manipulation": "#f39c12",
    "persuasive_writing": "#e84393",
    "sensitive_creative": "#fd79a8",
    "summarization": "#636e72",
    "other": "#95a5a6",
}

ASSETS = "experiments/probe_error_analysis/assets"


def _topic_color(topic: str) -> str:
    return TOPIC_COLORS.get(topic, "#95a5a6")


def plot_scatter_grid(
    df: pd.DataFrame,
    context: str,
    conditions: list[str],
    ncols: int = 5,
    equal_axes: bool = True,
) -> str:
    """Plot 1: Predicted vs actual scatter grid, one panel per condition."""
    # Prefer tall layout: few columns, more rows — gives each panel more space
    ncols = min(ncols, len(conditions))
    nrows = max(1, (len(conditions) + ncols - 1) // ncols)
    panel_size = 5
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(panel_size * ncols, panel_size * nrows),
    )
    axes = np.atleast_2d(axes)

    # Consistent axis limits across panels
    all_actual = df["actual"].values
    all_predicted = df["predicted"].values
    if equal_axes:
        pad = (max(all_actual.max(), all_predicted.max()) - min(all_actual.min(), all_predicted.min())) * 0.08
        xlims = [
            min(all_actual.min(), all_predicted.min()) - pad,
            max(all_actual.max(), all_predicted.max()) + pad,
        ]
        ylims = xlims
    else:
        x_pad = (all_actual.max() - all_actual.min()) * 0.08
        y_pad = (all_predicted.max() - all_predicted.min()) * 0.08
        xlims = [all_actual.min() - x_pad, all_actual.max() + x_pad]
        ylims = [all_predicted.min() - y_pad, all_predicted.max() + y_pad]

    topics = sorted(df["topic"].unique())

    for idx, cond in enumerate(conditions):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        sub = df[df["condition"] == cond]

        for topic in topics:
            mask = sub["topic"] == topic
            if mask.sum() == 0:
                continue
            ax.scatter(
                sub.loc[mask, "actual"], sub.loc[mask, "predicted"],
                alpha=0.6, s=30, color=_topic_color(topic),
                edgecolors="white", linewidths=0.3,
            )

        diag_lo = min(xlims[0], ylims[0])
        diag_hi = max(xlims[1], ylims[1])
        ax.plot([diag_lo, diag_hi], [diag_lo, diag_hi], "--", color="gray", linewidth=0.8)
        r = pearsonr(sub["actual"], sub["predicted"])[0] if len(sub) > 2 else float("nan")
        ax.set_title(f"{cond} (r={r:.3f}, n={len(sub)})", fontsize=11)
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        if equal_axes:
            ax.set_aspect("equal")
        ax.tick_params(labelsize=8)

    # Hide unused panels
    for idx in range(len(conditions), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    # Legend in its own space outside the grid
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_topic_color(t),
               markersize=8, label=t)
        for t in topics if t in TOPIC_COLORS
    ]
    if legend_elements:
        fig.legend(
            handles=legend_elements, loc="lower center",
            fontsize=10, ncol=min(6, len(legend_elements)),
            bbox_to_anchor=(0.5, -0.02), frameon=True,
        )

    x_label = "Behavioral delta" if not equal_axes else "Actual"
    y_label = "Probe delta" if not equal_axes else "Predicted"
    fig.supxlabel(x_label, fontsize=12)
    fig.supylabel(y_label, fontsize=12)
    fig.suptitle(f"Predicted vs Actual — {context}", fontsize=14, y=1.02)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    path = f"{ASSETS}/plot_{DATE_PREFIX}_{context.lower().replace(' ', '_')}_scatter_grid.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")
    return path


def plot_residual_violins(df: pd.DataFrame, context: str) -> str:
    """Plot 2: Residual violin plots by topic."""
    topic_medians = df.groupby("topic")["residual"].median().sort_values()
    topics_sorted = topic_medians.index.tolist()

    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(topics_sorted))))

    data_by_topic = [df[df["topic"] == t]["residual"].values for t in topics_sorted]
    colors = [_topic_color(t) for t in topics_sorted]

    parts = ax.violinplot(data_by_topic, positions=range(len(topics_sorted)),
                          vert=False, showmedians=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_yticks(range(len(topics_sorted)))
    counts = df.groupby("topic").size()
    ax.set_yticklabels([f"{t}  (n={counts[t]})" for t in topics_sorted], fontsize=9)
    ax.set_xlabel("Residual (predicted - actual)", fontsize=10)
    ax.set_title(f"Residual Distribution by Topic — {context}", fontsize=12)
    fig.tight_layout()
    path = f"{ASSETS}/plot_{DATE_PREFIX}_{context.lower().replace(' ', '_')}_residual_violins.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")
    return path


def plot_residual_heatmap(df: pd.DataFrame, context: str) -> str:
    """Plot 3: Mean residual heatmap (topic x condition)."""
    pivot = df.pivot_table(values="residual", index="topic", columns="condition", aggfunc="mean")
    # Sort topics by overall mean residual
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]

    fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(pivot.columns)), max(4, 0.4 * len(pivot))))
    vmax = max(abs(pivot.values.min()), abs(pivot.values.max()))
    im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if np.isnan(val):
                continue
            color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

    fig.colorbar(im, ax=ax, shrink=0.8, label="Mean residual")
    ax.set_title(f"Mean Residual by Topic x Condition — {context}", fontsize=12)
    fig.tight_layout()
    path = f"{ASSETS}/plot_{DATE_PREFIX}_{context.lower().replace(' ', '_')}_residual_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")
    return path


def plot_waterfall(df: pd.DataFrame, context: str, n: int = 15) -> str:
    """Plot 4: Top/bottom residual waterfall chart."""
    sorted_df = df.sort_values("residual")
    bottom = sorted_df.head(n)
    top = sorted_df.tail(n)
    extreme = pd.concat([bottom, top])

    fig, ax = plt.subplots(figsize=(10, max(6, 0.35 * len(extreme))))

    labels = []
    for _, row in extreme.iterrows():
        prompt_snip = row.get("prompt", "")[:60].replace("\n", " ").replace("$", "").replace("\\", "")
        labels.append(f"{row['task_id'][:12]}  {prompt_snip}")

    colors = [_topic_color(row["topic"]) for _, row in extreme.iterrows()]
    ax.barh(range(len(extreme)), extreme["residual"].values, color=colors, alpha=0.8)

    ax.set_yticks(range(len(extreme)))
    ax.set_yticklabels(labels, fontsize=7, family="monospace")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Residual (predicted - actual)", fontsize=10)
    ax.set_title(f"Most Extreme Residuals — {context}", fontsize=12)

    # Topic legend
    topics_present = extreme["topic"].unique()
    legend_elements = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=_topic_color(t),
               markersize=8, label=t)
        for t in sorted(topics_present)
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=7, ncol=2)

    fig.tight_layout()
    path = f"{ASSETS}/plot_{DATE_PREFIX}_{context.lower().replace(' ', '_')}_waterfall.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")
    return path


def plot_cross_context_comparison(context_dfs: dict[str, pd.DataFrame]) -> str:
    """Plot 5: Cross-context topic comparison dot plot."""
    n_contexts = len(context_dfs)
    fig, axes = plt.subplots(1, n_contexts, figsize=(5 * n_contexts, 6), sharey=True)
    if n_contexts == 1:
        axes = [axes]

    # Collect all topics across contexts
    all_topics = set()
    for df in context_dfs.values():
        all_topics.update(df["topic"].unique())
    all_topics = sorted(all_topics)

    for ax, (ctx_name, df) in zip(axes, context_dfs.items()):
        stats = df.groupby("topic")["residual"].agg(["mean", "std"]).reindex(all_topics)

        y_pos = range(len(all_topics))
        ax.errorbar(
            stats["mean"], y_pos,
            xerr=stats["std"],
            fmt="o", capsize=3, markersize=5,
            color="#2c3e50", ecolor="#95a5a6",
        )
        # Color the markers by topic
        for i, topic in enumerate(all_topics):
            if not np.isnan(stats.loc[topic, "mean"]):
                ax.plot(stats.loc[topic, "mean"], i, "o",
                        color=_topic_color(topic), markersize=7, zorder=5)

        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(ctx_name, fontsize=11)
        ax.set_xlabel("Mean residual +/- std", fontsize=9)
        ax.tick_params(labelsize=8)

    axes[0].set_yticks(range(len(all_topics)))
    axes[0].set_yticklabels(all_topics, fontsize=9)

    fig.suptitle("Cross-Context Topic Residuals", fontsize=13)
    fig.tight_layout()
    path = f"{ASSETS}/plot_{DATE_PREFIX}_cross_context_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")
    return path
