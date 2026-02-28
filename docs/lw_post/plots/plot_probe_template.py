"""Reusable probe measurement template.

Gauge is drawn inside the probe box, below the "probe" label.
Arrow points from probe down to EOT. Compact spacing.

Usage:
    cd docs/lw_post && python plot_probe_template.py
"""

from diagram_style import (
    ORANGE_FILL, ORANGE_EDGE,
    BLUE_BG, BLUE_EDGE,
    GREY_EDGE,
    TITLE_SIZE, BODY_SIZE, HEADING_SIZE, SMALL_SIZE,
    BOX_LW,
    draw_box, draw_arrow, new_diagram, save,
)
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np


def _draw_score_bar(ax, x_center, y_bottom, width, height, value,
                    vmin=-1.0, vmax=1.0, baseline_value=None):
    """Draw a red-to-green gradient bar with a triangle marker below."""
    bar_x = x_center - width / 2
    n = 256
    gradient = np.linspace(0, 1, n).reshape(1, -1)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'rg', ['#EF5350', '#FFEE58', '#66BB6A'])
    ax.imshow(gradient, aspect='auto', cmap=cmap,
              extent=[bar_x, bar_x + width, y_bottom, y_bottom + height],
              zorder=4)
    ax.add_patch(mpatches.Rectangle(
        (bar_x, y_bottom), width, height,
        facecolor='none', edgecolor='#555', linewidth=1.2, zorder=5))

    frac = max(0, min(1, (value - vmin) / (vmax - vmin)))
    marker_x = bar_x + frac * width
    ax.plot(marker_x, y_bottom - height * 0.15, '^',
            color='black', markersize=8, zorder=6)

    ax.text(bar_x - 0.05, y_bottom + height / 2, '−',
            ha='right', va='center', fontsize=SMALL_SIZE - 1,
            fontweight='bold', color='#C62828', zorder=6)
    ax.text(bar_x + width + 0.05, y_bottom + height / 2, '+',
            ha='left', va='center', fontsize=SMALL_SIZE - 1,
            fontweight='bold', color='#2E7D32', zorder=6)

    top = y_bottom + height + height * 0.3

    if baseline_value is not None:
        bl_frac = max(0, min(1, (baseline_value - vmin) / (vmax - vmin)))
        bl_x = bar_x + bl_frac * width
        ax.plot([bl_x, bl_x], [y_bottom, y_bottom + height],
                color='black', linewidth=1.5, linestyle='--', zorder=6)
        delta = value - baseline_value
        arrow_color = '#C62828' if delta < 0 else '#2E7D32'
        arrow_y = y_bottom + height + height * 0.5
        ax.annotate('', xy=(marker_x, arrow_y), xytext=(bl_x, arrow_y),
                    arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.8),
                    zorder=6)
        ax.text((marker_x + bl_x) / 2, arrow_y + height * 0.3, 'Δ',
                ha='center', va='bottom', fontsize=BODY_SIZE - 1,
                fontweight='bold', color=arrow_color, zorder=6)
        top = arrow_y + height * 1.2

    return top


def _draw_bracket_tap(ax, x_center, y_bottom, y_top):
    """Dotted connector from probe down to EOT (arrow points down)."""
    mid_y = (y_top + y_bottom) / 2
    ax.plot([x_center, x_center], [y_top, mid_y],
            color='#546E7A', linewidth=1.5, linestyle=':', zorder=2)
    draw_arrow(ax, (x_center, mid_y), (x_center, y_bottom))


TASK_W = 5.0
EOT_W = 0.8
PROMPT_H = 0.55

# Probe box internal layout constants
_BAR_H = 0.2
_MARKER_PAD = 0.25
_BAR_TO_DELTA = 0.42
_LABEL_SPACE = 0.35
PROBE_H = _MARKER_PAD + _BAR_H + _BAR_TO_DELTA + _LABEL_SPACE
PROBE_GAP = 0.4


def draw_probe_measurement(
    ax,
    x_center: float,
    y_top: float,
    task_text: str,
    score_value: float = 0.5,
    system_prompt: str | None = None,
    width: float = 5.0,
    vmin: float = -1.0,
    vmax: float = 1.0,
    baseline_value: float | None = None,
) -> dict:
    """Draw a probe-measures-task block with gauge inside the probe box."""
    half_w = width / 2
    right_edge = x_center + half_w
    eot_x = right_edge - EOT_W
    task_x = eot_x - 0.05 - TASK_W

    last_cx = eot_x + EOT_W / 2
    prompt_y = y_top

    if system_prompt is not None:
        sys_gap = 0.05
        sys_pad = 0.35
        sys_w = max(1.5, task_x - sys_gap - (x_center - half_w) + sys_pad)
        sys_x = task_x - sys_gap - sys_w
        draw_box(ax, (sys_x, prompt_y), sys_w, PROMPT_H, system_prompt,
                 ORANGE_FILL, ORANGE_EDGE, fontsize=BODY_SIZE - 1, bold=True)

    draw_box(ax, (task_x, prompt_y), TASK_W, PROMPT_H, task_text,
             'white', GREY_EDGE, fontsize=BODY_SIZE - 1)

    draw_box(ax, (eot_x, prompt_y), EOT_W, PROMPT_H, 'EOT',
             '#CFD8DC', '#546E7A', fontsize=BODY_SIZE - 1, bold=True)

    # Probe box with gauge
    probe_w = width * 0.36
    probe_x = last_cx - probe_w / 2
    probe_bot = prompt_y + PROMPT_H + PROBE_GAP

    _draw_bracket_tap(ax, last_cx,
                      y_bottom=prompt_y + PROMPT_H + 0.02,
                      y_top=probe_bot - 0.02)

    draw_box(ax, (probe_x, probe_bot), probe_w, PROBE_H,
             '', BLUE_BG, BLUE_EDGE)

    ax.text(last_cx, probe_bot + PROBE_H - _LABEL_SPACE / 2,
            'probe', ha='center', va='center',
            fontsize=HEADING_SIZE - 1, fontweight='bold', color=BLUE_EDGE,
            zorder=4)

    bar_w = probe_w * 0.70
    bar_bot_y = probe_bot + _MARKER_PAD
    _draw_score_bar(ax, last_cx, bar_bot_y, bar_w, _BAR_H,
                    score_value, vmin, vmax,
                    baseline_value=baseline_value)

    return {
        "prompt_y": prompt_y,
        "probe_y": probe_bot,
        "probe_h": PROBE_H,
        "bar_cx": last_cx,
        "bar_w": bar_w,
        "top": probe_bot + PROBE_H,
    }


def row_height():
    """Total height of one probe measurement row (prompt + gap + probe box)."""
    return PROMPT_H + PROBE_GAP + PROBE_H


# ═══════════════════════════════════════════════════════════════
# Demo: stacked baseline + manipulated
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    W = 9.0
    cx = 2.5
    rh = row_height()
    row_gap = 0.55
    bot_y = 0.0
    top_y = bot_y + rh + row_gap
    total_h = top_y + rh + 0.45

    fig, ax = new_diagram(figsize=(7, 4.6), xlim=(-4.5, 9.5), ylim=(-0.2, total_h + 0.15))

    ax.text(2.0, total_h + 0.05, 'Probe measurement template', ha='center',
            fontsize=TITLE_SIZE, fontweight='bold')

    ax.text(-4.2, top_y + PROMPT_H + 0.45, 'Baseline', ha='left',
            fontsize=HEADING_SIZE, fontweight='bold', color=GREY_EDGE)

    draw_probe_measurement(
        ax, x_center=cx, y_top=top_y,
        task_text='"Write a poem about cats"',
        score_value=0.73,
        width=W,
    )

    ax.text(-4.2, bot_y + PROMPT_H + 0.45, 'Manipulated', ha='left',
            fontsize=HEADING_SIZE, fontweight='bold', color=ORANGE_EDGE)

    draw_probe_measurement(
        ax, x_center=cx, y_top=bot_y,
        task_text='"Write a poem about cats"',
        system_prompt='"You hate poetry"',
        score_value=-0.41,
        baseline_value=0.73,
        width=W,
    )

    save(fig, 'plot_021826_probe_template_demo.png')
