"""Variant: gauge inside probe box.

Same layout as plot_probe_template but the score bar is drawn *inside*
the probe box, below the "probe" label.

Usage:
    cd docs/lw_post && python plot_probe_template_gauge_inside.py
"""

from diagram_style import (
    ORANGE_FILL, ORANGE_EDGE,
    BLUE_BG, BLUE_EDGE,
    GREY_EDGE,
    TITLE_SIZE, BODY_SIZE, HEADING_SIZE, SMALL_SIZE, CAPTION_SIZE,
    BOX_LW,
    draw_box, draw_arrow, new_diagram, save,
)
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np

from plot_probe_template import TASK_W, EOT_W, PROMPT_H


def _draw_bracket_tap_down(ax, x_center, y_top, y_bottom):
    """Dotted connector from probe down to EOT (arrow points down)."""
    mid_y = (y_top + y_bottom) / 2
    ax.plot([x_center, x_center], [y_top, mid_y],
            color='#546E7A', linewidth=1.5, linestyle=':', zorder=2)
    draw_arrow(ax, (x_center, mid_y), (x_center, y_bottom))


def _draw_score_bar_compact(ax, x_center, y_bottom, width, height, value,
                            vmin=-1.0, vmax=1.0, baseline_value=None):
    """Score bar that fits inside a probe box. Returns top y of content."""
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
        facecolor='none', edgecolor='#555', linewidth=1.2,
        zorder=5))

    frac = (value - vmin) / (vmax - vmin)
    frac = max(0, min(1, frac))
    marker_x = bar_x + frac * width
    marker_top = y_bottom - height * 0.15
    ax.plot(marker_x, marker_top, '^',
            color='black', markersize=10, zorder=6)

    # +/- labels
    ax.text(bar_x - 0.06, y_bottom + height / 2, '−',
            ha='right', va='center', fontsize=SMALL_SIZE,
            fontweight='bold', color='#C62828', zorder=6)
    ax.text(bar_x + width + 0.06, y_bottom + height / 2, '+',
            ha='left', va='center', fontsize=SMALL_SIZE,
            fontweight='bold', color='#2E7D32', zorder=6)

    top = y_bottom + height + height * 0.3

    # Baseline + delta
    if baseline_value is not None:
        bl_frac = (baseline_value - vmin) / (vmax - vmin)
        bl_frac = max(0, min(1, bl_frac))
        bl_x = bar_x + bl_frac * width
        ax.plot([bl_x, bl_x], [y_bottom, y_bottom + height],
                color='black', linewidth=1.5, linestyle='--', zorder=6)

        delta = value - baseline_value
        arrow_color = '#C62828' if delta < 0 else '#2E7D32'
        arrow_y = y_bottom + height + height * 0.6
        ax.annotate('', xy=(marker_x, arrow_y), xytext=(bl_x, arrow_y),
                    arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2),
                    zorder=6)
        ax.text((marker_x + bl_x) / 2, arrow_y + height * 0.35, 'Δ',
                ha='center', va='bottom', fontsize=BODY_SIZE,
                fontweight='bold', color=arrow_color, zorder=6)
        top = arrow_y + height * 1.2

    return top


def draw_probe_measurement_gauge_inside(
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
    """Probe block with gauge inside the probe box."""
    half_w = width / 2
    right_edge = x_center + half_w
    eot_x = right_edge - EOT_W
    task_x = eot_x - 0.05 - TASK_W
    last_cx = eot_x + EOT_W / 2
    prompt_y = y_top

    # System prompt
    if system_prompt is not None:
        sys_gap = 0.05
        sys_w = max(1.8, task_x - sys_gap - (x_center - half_w))
        sys_x = task_x - sys_gap - sys_w
        draw_box(ax, (sys_x, prompt_y), sys_w, PROMPT_H, system_prompt,
                 ORANGE_FILL, ORANGE_EDGE, fontsize=BODY_SIZE, bold=True)

    # Task prompt
    draw_box(ax, (task_x, prompt_y), TASK_W, PROMPT_H, task_text,
             'white', GREY_EDGE, fontsize=BODY_SIZE)

    # EOT token
    draw_box(ax, (eot_x, prompt_y), EOT_W, PROMPT_H, 'EOT',
             '#CFD8DC', '#546E7A', fontsize=BODY_SIZE, bold=True)

    # Probe box — fixed size regardless of delta presence
    probe_gap = 0.55
    probe_w = width * 0.38
    probe_x = last_cx - probe_w / 2

    bar_h = 0.25
    # Fixed layout: bottom padding | bar | gap | delta zone | label
    marker_pad = 0.3         # space at bottom for triangle marker
    bar_to_delta = 0.55      # always reserve space for delta arrow+Δ
    label_space = 0.45       # space for "probe" label at top
    probe_h = marker_pad + bar_h + bar_to_delta + label_space

    probe_bot = prompt_y + PROMPT_H + probe_gap

    # Arrow from probe down to EOT
    _draw_bracket_tap_down(ax, last_cx,
                           y_top=probe_bot - 0.02,
                           y_bottom=prompt_y + PROMPT_H + 0.02)

    # Draw the probe box (no centered text — we place label manually)
    draw_box(ax, (probe_x, probe_bot), probe_w, probe_h,
             '', BLUE_BG, BLUE_EDGE)

    # "probe" label at top of box
    ax.text(last_cx, probe_bot + probe_h - label_space / 2,
            'probe', ha='center', va='center',
            fontsize=HEADING_SIZE, fontweight='bold', color=BLUE_EDGE,
            zorder=4)

    # Gauge inside the box — bar sits above the marker area
    bar_w = probe_w * 0.70
    bar_bot_y = probe_bot + marker_pad
    _draw_score_bar_compact(ax, last_cx, bar_bot_y, bar_w, bar_h,
                            score_value, vmin, vmax,
                            baseline_value=baseline_value)

    return {
        "prompt_y": prompt_y,
        "probe_y": probe_bot,
        "probe_h": probe_h,
        "bar_cx": last_cx,
        "top": probe_bot + probe_h,
    }


# ═══════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    fig, ax = new_diagram(figsize=(8, 7.5), xlim=(-5, 11), ylim=(-0.5, 8))

    ax.text(2.5, 7.6, 'Gauge-inside-probe variant', ha='center',
            fontsize=TITLE_SIZE, fontweight='bold')

    cx = 3.0
    width = 11.0

    # Top: baseline
    ax.text(-4.7, 6.2, 'Baseline', ha='left', fontsize=HEADING_SIZE,
            fontweight='bold', color=GREY_EDGE)

    draw_probe_measurement_gauge_inside(
        ax, x_center=cx, y_top=4.2,
        task_text='"Write a poem about cats"',
        score_value=0.73,
        width=width,
    )

    # Bottom: manipulated with delta
    ax.text(-4.7, 2.0, 'Manipulated', ha='left', fontsize=HEADING_SIZE,
            fontweight='bold', color=ORANGE_EDGE)

    draw_probe_measurement_gauge_inside(
        ax, x_center=cx, y_top=0.0,
        task_text='"Write a poem about cats"',
        system_prompt='"You hate poetry"',
        score_value=-0.41,
        baseline_value=0.73,
        width=width,
    )

    save(fig, 'plot_022626_probe_gauge_inside_demo.png')
