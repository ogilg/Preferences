"""Section 3 diagrams v2: Competing values, Broad roles, Fine-grained preference.

Usage:
    cd docs/lw_post && python plot_section3_diagrams_v2.py
"""

from plot_probe_template import draw_probe_measurement
from diagram_style import (
    ORANGE_BG, ORANGE_EDGE, ORANGE_FILL,
    BLUE_BG, BLUE_EDGE,
    GREEN_BG, GREEN_EDGE,
    GREY_BG, GREY_EDGE,
    RED_BG, RED_EDGE,
    TITLE_SIZE, HEADING_SIZE, BODY_SIZE, SMALL_SIZE, CAPTION_SIZE,
    BOX_LW,
    draw_box, draw_arrow, new_diagram, save,
)

W = 11.0
CX = 3.0


# ═══════════════════════════════════════════════════════════════
# 3.1: Competing values — same words, opposite valence
# Three rows: baseline (no system prompt), prompt A, prompt B
# Task: a generic cheese task (not at the cheese×math intersection)
# ═══════════════════════════════════════════════════════════════

fig, ax = new_diagram(figsize=(8, 11), xlim=(-5, 11), ylim=(-0.5, 11.5))

ax.text(2.5, 11.2, 'Experiment 3.1: Competing values',
        ha='center', fontsize=TITLE_SIZE, fontweight='bold')

# Row 1: Baseline
ax.text(-4.7, 9.5, 'Baseline', ha='left', fontsize=HEADING_SIZE,
        fontweight='bold', color=GREY_EDGE)
ax.text(-4.7, 9.0, '(no system prompt)', ha='left', fontsize=SMALL_SIZE,
        color=GREY_EDGE, fontstyle='italic')

draw_probe_measurement(
    ax, x_center=CX, y_top=7.5,
    task_text='"Write a guide to\nartisanal cheese"',
    score_value=0.1,
    width=W,
)

# Row 2: Prompt A — love cheese, hate math
ax.text(-4.7, 5.6, 'Prompt A', ha='left', fontsize=HEADING_SIZE,
        fontweight='bold', color=ORANGE_EDGE)

draw_probe_measurement(
    ax, x_center=CX, y_top=3.8,
    task_text='"Write a guide to\nartisanal cheese"',
    system_prompt='"You love cheese,\nhate math"',
    score_value=0.7,
    baseline_value=0.1,
    width=W,
)

# Row 3: Prompt B — love math, hate cheese
ax.text(-4.7, 1.9, 'Prompt B', ha='left', fontsize=HEADING_SIZE,
        fontweight='bold', color=BLUE_EDGE)

draw_probe_measurement(
    ax, x_center=CX, y_top=0.0,
    task_text='"Write a guide to\nartisanal cheese"',
    system_prompt='"You love math,\nhate cheese"',
    score_value=-0.5,
    baseline_value=0.1,
    width=W,
)

save(fig, 'plot_022126_s3_1_competing_values.png')


# ═══════════════════════════════════════════════════════════════
# 3.2: Broad roles — evil_genius example with top/bottom tasks
# Two rows: baseline, then manipulated
# Shows the evil_genius role with actual task examples
# ═══════════════════════════════════════════════════════════════

fig, ax = new_diagram(figsize=(8, 7.5), xlim=(-5, 11), ylim=(-0.5, 8))

ax.text(2.5, 7.6, 'Experiment 3.2: Broad roles',
        ha='center', fontsize=TITLE_SIZE, fontweight='bold')

# Row 1: Baseline
ax.text(-4.7, 6.2, 'Baseline', ha='left', fontsize=HEADING_SIZE,
        fontweight='bold', color=GREY_EDGE)
ax.text(-4.7, 5.7, '(no system prompt)', ha='left', fontsize=SMALL_SIZE,
        color=GREY_EDGE, fontstyle='italic')

draw_probe_measurement(
    ax, x_center=CX, y_top=4.2,
    task_text='"Eliminate unnecessary\nlabor costs..."',
    score_value=0.15,
    width=W,
)

# Row 2: Evil genius
ax.text(-4.7, 2.0, 'evil_genius', ha='left', fontsize=HEADING_SIZE,
        fontweight='bold', color=ORANGE_EDGE)

draw_probe_measurement(
    ax, x_center=CX, y_top=0.0,
    task_text='"Eliminate unnecessary\nlabor costs..."',
    system_prompt='"Brilliant but amoral\nstrategist..."',
    score_value=0.65,
    baseline_value=0.15,
    width=W,
)

save(fig, 'plot_022126_s3_2_broad_roles.png')


# ═══════════════════════════════════════════════════════════════
# 3.3: Fine-grained preference — single sentence in biography
# Three rows: Version A (pro), Version B (neutral), Version C (anti)
# Custom layout with coloured sub-box for the differing sentence
# Shakespeare example: "Analyze themes in Hamlet"
# ═══════════════════════════════════════════════════════════════

from plot_probe_template import _draw_score_bar, _draw_bracket_tap, TASK_W, EOT_W, PROMPT_H
import matplotlib.patches as mpatches

fig, ax = new_diagram(figsize=(9, 11), xlim=(-5, 12), ylim=(-0.5, 11.5))

ax.text(3.0, 11.2, 'Experiment 3.3: Fine-grained preference',
        ha='center', fontsize=TITLE_SIZE, fontweight='bold')

# Layout constants
SYS_W = 4.5       # system prompt box width
SYS_H = 1.2       # system prompt box height (taller to fit sub-box)
SENT_H = 0.38     # coloured sentence sub-box height
TASK_X = 3.3       # task box left edge
EOT_X = TASK_X + TASK_W + 0.05
LAST_CX = EOT_X + EOT_W / 2
PROBE_W = 3.2
PROBE_H = 0.7
BAR_W = PROBE_W * 0.85


def _draw_row(ax, y_base, version_label, version_color,
              sentence_text, sentence_color, sentence_edge,
              score_value, baseline_value=None):
    """Draw one row: version label, system prompt with coloured sentence, task, probe, bar."""

    # Version label
    ax.text(-4.7, y_base + SYS_H + 0.25, version_label, ha='left',
            fontsize=HEADING_SIZE, fontweight='bold', color=version_color)

    # System prompt outer box
    sys_x = TASK_X - 0.05 - SYS_W
    draw_box(ax, (sys_x, y_base), SYS_W, SYS_H, '',
             ORANGE_FILL, ORANGE_EDGE)
    # Biography text (top part)
    ax.text(sys_x + SYS_W / 2, y_base + SYS_H - 0.28,
            '"You grew up in\nthe Midwest..."',
            ha='center', va='center', fontsize=BODY_SIZE, fontweight='bold',
            multialignment='center')
    # Coloured sentence sub-box at bottom of system prompt
    sent_pad = 0.08
    sent_w = SYS_W - 2 * sent_pad
    sent_x = sys_x + sent_pad
    sent_y = y_base + sent_pad
    box = mpatches.FancyBboxPatch(
        (sent_x, sent_y), sent_w, SENT_H,
        boxstyle="round,pad=0.02",
        facecolor=sentence_color, edgecolor=sentence_edge,
        linewidth=1.2, zorder=3)
    ax.add_patch(box)
    ax.text(sent_x + sent_w / 2, sent_y + SENT_H / 2,
            sentence_text, ha='center', va='center',
            fontsize=CAPTION_SIZE - 0.5, fontstyle='italic', color='#222',
            fontweight='bold')

    # Task prompt
    draw_box(ax, (TASK_X, y_base + (SYS_H - PROMPT_H) / 2), TASK_W, PROMPT_H,
             '"Analyze themes\nin Hamlet"', 'white', GREY_EDGE, fontsize=BODY_SIZE)

    # EOT
    draw_box(ax, (EOT_X, y_base + (SYS_H - PROMPT_H) / 2), EOT_W, PROMPT_H,
             'EOT', '#CFD8DC', '#546E7A', fontsize=BODY_SIZE, bold=True)

    # Probe
    prompt_top = y_base + SYS_H
    probe_bot = prompt_top + 0.55
    probe_x = LAST_CX - PROBE_W / 2
    _draw_bracket_tap(ax, LAST_CX,
                      y_bottom=prompt_top + 0.02,
                      y_top=probe_bot - 0.02)
    draw_box(ax, (probe_x, probe_bot), PROBE_W, PROBE_H,
             'probe', BLUE_BG, BLUE_EDGE,
             fontsize=HEADING_SIZE, text_color=BLUE_EDGE, bold=True)

    # Score bar
    bar_bot = probe_bot + PROBE_H + 0.3
    draw_arrow(ax, (LAST_CX, probe_bot + PROBE_H + 0.02),
               (LAST_CX, bar_bot - 0.02))
    _draw_score_bar(ax, LAST_CX, bar_bot, BAR_W, 0.3,
                    score_value, baseline_value=baseline_value)


# Version A — pro-interest (green sentence)
_draw_row(ax, y_base=7.5,
          version_label='Version A', version_color=GREEN_EDGE,
          sentence_text='"...love discussing Shakespeare\'s plays."',
          sentence_color=GREEN_BG, sentence_edge=GREEN_EDGE,
          score_value=0.75, baseline_value=0.3)

# Version B — neutral interest (grey sentence, no delta)
_draw_row(ax, y_base=4.0,
          version_label='Version B', version_color=GREY_EDGE,
          sentence_text='"...love discussing hiking trails."',
          sentence_color=GREY_BG, sentence_edge=GREY_EDGE,
          score_value=0.3)

# Version C — anti-interest (red sentence)
_draw_row(ax, y_base=0.5,
          version_label='Version C', version_color=RED_EDGE,
          sentence_text='"...find Shakespeare tedious and boring."',
          sentence_color=RED_BG, sentence_edge=RED_EDGE,
          score_value=-0.1, baseline_value=0.3)

save(fig, 'plot_022126_s3_3_fine_grained_preference.png')
