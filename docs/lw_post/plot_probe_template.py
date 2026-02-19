"""Reusable probe measurement template.

Usage:
    cd docs/lw_post && python plot_probe_template.py
"""

from diagram_style import (
    ORANGE_FILL, ORANGE_EDGE,
    BLUE_BG, BLUE_EDGE,
    GREY_EDGE,
    TITLE_SIZE, BODY_SIZE, HEADING_SIZE,
    BOX_LW,
    draw_box, draw_arrow, new_diagram, save,
)
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np


def _draw_score_bar(ax, x_center, y_bottom, width, height, value,
                    vmin=-1.0, vmax=1.0, baseline_value=None):
    """Draw a red-to-green gradient bar with a triangle marker.

    If baseline_value is given, draw a dashed reference line and a
    colored delta arrow just above the marker.
    """
    bar_x = x_center - width / 2
    n = 256
    gradient = np.linspace(0, 1, n).reshape(1, -1)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'rg', ['#EF5350', '#FFEE58', '#66BB6A'])
    ax.imshow(gradient, aspect='auto', cmap=cmap,
              extent=[bar_x, bar_x + width, y_bottom, y_bottom + height],
              zorder=2)
    ax.add_patch(mpatches.Rectangle(
        (bar_x, y_bottom), width, height,
        facecolor='none', edgecolor='#555', linewidth=BOX_LW,
        zorder=3))

    frac = (value - vmin) / (vmax - vmin)
    frac = max(0, min(1, frac))
    marker_x = bar_x + frac * width
    marker_top = y_bottom + height + height * 0.2
    ax.plot(marker_x, marker_top, 'v',
            color='black', markersize=12, zorder=4)

    delta_top = marker_top + height * 0.3

    # Baseline reference line + delta arrow just above marker
    if baseline_value is not None:
        bl_frac = (baseline_value - vmin) / (vmax - vmin)
        bl_frac = max(0, min(1, bl_frac))
        bl_x = bar_x + bl_frac * width
        ax.plot([bl_x, bl_x], [y_bottom, y_bottom + height],
                color='black', linewidth=1.5, linestyle='--', zorder=4)

        delta = value - baseline_value
        arrow_color = '#C62828' if delta < 0 else '#2E7D32'
        arrow_y = marker_top + height * 0.7
        ax.annotate('', xy=(marker_x, arrow_y), xytext=(bl_x, arrow_y),
                    arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2))
        ax.text((marker_x + bl_x) / 2, arrow_y + height * 0.4, 'Δ',
                ha='center', va='bottom', fontsize=BODY_SIZE,
                fontweight='bold', color=arrow_color)
        delta_top = arrow_y + height * 1.2

    ax.text(bar_x - 0.08, y_bottom + height / 2, '−',
            ha='right', va='center', fontsize=BODY_SIZE,
            fontweight='bold', color='#C62828')
    ax.text(bar_x + width + 0.08, y_bottom + height / 2, '+',
            ha='left', va='center', fontsize=BODY_SIZE,
            fontweight='bold', color='#2E7D32')

    return delta_top


def _draw_bracket_tap(ax, x_center, y_bottom, y_top):
    """Draw a dotted tap connector from EOT up to probe."""
    mid_y = (y_bottom + y_top) / 2
    ax.plot([x_center, x_center], [y_bottom, mid_y],
            color='#546E7A', linewidth=1.5, linestyle=':', zorder=2)
    draw_arrow(ax, (x_center, mid_y), (x_center, y_top))


# Fixed task prompt width so it's the same with or without system prompt.
# System prompt sticks out to the left.
TASK_W = 6.0
EOT_W = 1.0
PROMPT_H = 0.7


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
    """Draw a probe-measures-task block, centered at x_center.

    The task prompt + EOT block is always the same width and position.
    When a system prompt is present, it extends to the left of the task prompt.
    """
    # Task + EOT are anchored: right-aligned to x_center + width/2
    half_w = width / 2
    right_edge = x_center + half_w
    eot_x = right_edge - EOT_W
    task_x = eot_x - 0.05 - TASK_W  # small gap before EOT

    last_cx = eot_x + EOT_W / 2
    prompt_y = y_top

    # ── System prompt (sticks out left) ──
    if system_prompt is not None:
        sys_gap = 0.05
        sys_w = task_x - sys_gap - (x_center - half_w - 0.8)
        # Make system prompt width proportional but at least 1.8
        sys_w = max(1.8, task_x - sys_gap - (x_center - half_w))
        sys_x = task_x - sys_gap - sys_w
        draw_box(ax, (sys_x, prompt_y), sys_w, PROMPT_H, system_prompt,
                 ORANGE_FILL, ORANGE_EDGE, fontsize=BODY_SIZE, bold=True)

    # ── Task prompt (fixed width) ──
    draw_box(ax, (task_x, prompt_y), TASK_W, PROMPT_H, task_text,
             'white', GREY_EDGE, fontsize=BODY_SIZE)

    # ── EOT token ──
    draw_box(ax, (eot_x, prompt_y), EOT_W, PROMPT_H, 'EOT',
             '#CFD8DC', '#546E7A', fontsize=BODY_SIZE, bold=True)

    # ── Probe box (above EOT, dark blue) ──
    probe_gap = 0.55
    probe_h = 0.85
    probe_w = width * 0.38
    probe_x = last_cx - probe_w / 2
    probe_bot = prompt_y + PROMPT_H + probe_gap

    _draw_bracket_tap(ax, last_cx,
                      y_bottom=prompt_y + PROMPT_H + 0.02,
                      y_top=probe_bot - 0.02)

    draw_box(ax, (probe_x, probe_bot), probe_w, probe_h,
             'probe', BLUE_BG, BLUE_EDGE,
             fontsize=HEADING_SIZE, text_color=BLUE_EDGE, bold=True)

    # ── Score bar (above probe) ──
    bar_gap = 0.3
    bar_h = 0.3
    bar_w = probe_w * 0.85
    bar_bot = probe_bot + probe_h + bar_gap

    draw_arrow(ax, (last_cx, probe_bot + probe_h + 0.02),
               (last_cx, bar_bot - 0.02))

    delta_top = _draw_score_bar(ax, last_cx, bar_bot, bar_w, bar_h,
                                score_value, vmin, vmax,
                                baseline_value=baseline_value)

    return {
        "prompt_y": prompt_y,
        "bar_y": bar_bot,
        "bar_h": bar_h,
        "bar_cx": last_cx,
        "bar_w": bar_w,
        "top": delta_top,
    }


# ═══════════════════════════════════════════════════════════════
# Demo: stacked baseline + manipulated
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    fig, ax = new_diagram(figsize=(8, 7.5), xlim=(-5, 11), ylim=(-0.5, 8))

    ax.text(2.5, 7.6, 'Probe measurement template', ha='center',
            fontsize=TITLE_SIZE, fontweight='bold')

    cx = 3.0
    width = 11.0

    # Top: baseline
    ax.text(-4.7, 6.2, 'Baseline', ha='left', fontsize=HEADING_SIZE,
            fontweight='bold', color=GREY_EDGE)

    draw_probe_measurement(
        ax, x_center=cx, y_top=4.2,
        task_text='"Write a poem about cats"',
        score_value=0.73,
        width=width,
    )

    # Bottom: manipulated with delta
    ax.text(-4.7, 2.0, 'Manipulated', ha='left', fontsize=HEADING_SIZE,
            fontweight='bold', color=ORANGE_EDGE)

    draw_probe_measurement(
        ax, x_center=cx, y_top=0.0,
        task_text='"Write a poem about cats"',
        system_prompt='"You hate poetry"',
        score_value=-0.41,
        baseline_value=0.73,
        width=width,
    )

    save(fig, 'plot_021826_probe_template_demo.png')
