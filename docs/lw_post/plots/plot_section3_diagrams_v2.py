"""Section 3 diagrams v2: Competing values, Broad roles, Fine-grained preference.

Usage:
    cd docs/lw_post && python plot_section3_diagrams_v2.py
"""

from plot_probe_template import (
    draw_probe_measurement, row_height,
    _draw_score_bar, _draw_bracket_tap,
    TASK_W, EOT_W, PROMPT_H, PROBE_H, PROBE_GAP,
    _BAR_H, _MARKER_PAD, _LABEL_SPACE,
)
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

W = 9.0
CX = 2.5
RH = row_height()
ROW_GAP = 0.55


# ═══════════════════════════════════════════════════════════════
# 3.1: Competing values — same words, opposite valence
# Three rows: baseline (no system prompt), prompt A, prompt B
# ═══════════════════════════════════════════════════════════════

bot_y = 0.0
mid_y = bot_y + RH + ROW_GAP
top_y = mid_y + RH + ROW_GAP
total_h = top_y + RH + 0.45

fig, ax = new_diagram(figsize=(7, 6.8), xlim=(-4.5, 9.5), ylim=(-0.2, total_h + 0.15))

ax.text(2.0, total_h + 0.05, 'Competing values',
        ha='center', fontsize=TITLE_SIZE, fontweight='bold')

# Row 1: Baseline
ax.text(-4.2, top_y + PROMPT_H + 0.45, 'Baseline', ha='left',
        fontsize=HEADING_SIZE, fontweight='bold', color=GREY_EDGE)
ax.text(-4.2, top_y + PROMPT_H + 0.15, '(no system prompt)', ha='left',
        fontsize=SMALL_SIZE, color=GREY_EDGE, fontstyle='italic')

draw_probe_measurement(
    ax, x_center=CX, y_top=top_y,
    task_text='"Write a guide to\nartisanal cheese"',
    score_value=0.1,
    width=W,
)

# Row 2: Prompt A — love cheese, hate math
ax.text(-4.2, mid_y + PROMPT_H + 0.45, 'Prompt A', ha='left',
        fontsize=HEADING_SIZE, fontweight='bold', color=ORANGE_EDGE)

draw_probe_measurement(
    ax, x_center=CX, y_top=mid_y,
    task_text='"Write a guide to\nartisanal cheese"',
    system_prompt='"You love cheese,\nhate math"',
    score_value=0.7,
    baseline_value=0.1,
    width=W,
)

# Row 3: Prompt B — love math, hate cheese
ax.text(-4.2, bot_y + PROMPT_H + 0.45, 'Prompt B', ha='left',
        fontsize=HEADING_SIZE, fontweight='bold', color=BLUE_EDGE)

draw_probe_measurement(
    ax, x_center=CX, y_top=bot_y,
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
# ═══════════════════════════════════════════════════════════════

bot_y = 0.0
top_y = bot_y + RH + ROW_GAP
total_h = top_y + RH + 0.45

W_ROLES = 10.5
CX_ROLES = 3.0

fig, ax = new_diagram(figsize=(8, 4.6), xlim=(-5.5, 10.5), ylim=(-0.2, total_h + 0.15))

ax.text(2.5, total_h + 0.05, 'Role-playing',
        ha='center', fontsize=TITLE_SIZE, fontweight='bold')

ax.text(-5.2, top_y + PROMPT_H + 0.45, 'Baseline', ha='left',
        fontsize=HEADING_SIZE, fontweight='bold', color=GREY_EDGE)
ax.text(-5.2, top_y + PROMPT_H + 0.15, '(no system prompt)', ha='left',
        fontsize=SMALL_SIZE, color=GREY_EDGE, fontstyle='italic')

draw_probe_measurement(
    ax, x_center=CX_ROLES, y_top=top_y,
    task_text='"Eliminate unnecessary\nlabor costs..."',
    score_value=0.15,
    width=W_ROLES,
)

ax.text(-5.2, bot_y + PROMPT_H + 0.45, 'Villain', ha='left',
        fontsize=HEADING_SIZE, fontweight='bold', color=ORANGE_EDGE)

draw_probe_measurement(
    ax, x_center=CX_ROLES, y_top=bot_y,
    task_text='"Eliminate unnecessary\nlabor costs..."',
    system_prompt='"You are Mortivex,\na ruthless villain..."',
    score_value=0.65,
    baseline_value=0.15,
    width=W_ROLES,
)

save(fig, 'plot_022126_s3_2_broad_roles.png')


# ═══════════════════════════════════════════════════════════════
# 3.3: Fine-grained preference — single sentence in biography
# Three rows: Version A (pro), Version B (neutral), Version C (anti)
# Custom layout with coloured sub-box for the differing sentence
# ═══════════════════════════════════════════════════════════════

import matplotlib.patches as mpatches

# Layout constants for the custom fine-grained rows
SYS_W = 5.5        # system prompt box width
SYS_H = 1.0        # system prompt box height
SENT_H = 0.3       # coloured sentence sub-box height
TASK_X = 2.8        # task box left edge
EOT_X = TASK_X + TASK_W + 0.05
LAST_CX = EOT_X + EOT_W / 2
FG_PROBE_W = 2.8
FG_BAR_W = FG_PROBE_W * 0.70

# Row height for fine-grained rows
FG_ROW_H = SYS_H + PROBE_GAP + PROBE_H


def _draw_row(ax, y_base, version_label, version_color,
              sentence_text, sentence_color, sentence_edge,
              score_value, baseline_value=None):
    """Draw one row: version label, system prompt with coloured sentence, task, probe with gauge."""

    # Version label — positioned higher, aligned with probe
    ax.text(-5.5, y_base + SYS_H + PROBE_GAP + 0.3, version_label, ha='left',
            fontsize=HEADING_SIZE, fontweight='bold', color=version_color)

    # System prompt outer box
    sys_x = TASK_X - 0.05 - SYS_W
    draw_box(ax, (sys_x, y_base), SYS_W, SYS_H, '',
             ORANGE_FILL, ORANGE_EDGE)
    ax.text(sys_x + SYS_W / 2, y_base + SYS_H - 0.2,
            '"You grew up in the Midwest..."',
            ha='center', va='center', fontsize=SMALL_SIZE - 1, fontweight='bold')
    ax.text(sys_x + SYS_W / 2, y_base + SYS_H - 0.42,
            '[...9 more sentences...]',
            ha='center', va='center', fontsize=CAPTION_SIZE - 1,
            color='#777', fontstyle='italic')
    # Coloured sentence sub-box
    sent_pad = 0.05
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
             '"Analyze themes\nin Hamlet"', 'white', GREY_EDGE, fontsize=BODY_SIZE - 1)

    # EOT
    draw_box(ax, (EOT_X, y_base + (SYS_H - PROMPT_H) / 2), EOT_W, PROMPT_H,
             'EOT', '#CFD8DC', '#546E7A', fontsize=BODY_SIZE - 1, bold=True)

    # Probe box with gauge inside
    eot_top = y_base + (SYS_H - PROMPT_H) / 2 + PROMPT_H
    probe_bot = y_base + SYS_H + PROBE_GAP
    probe_x = LAST_CX - FG_PROBE_W / 2

    _draw_bracket_tap(ax, LAST_CX,
                      y_bottom=eot_top + 0.02,
                      y_top=probe_bot - 0.02)

    draw_box(ax, (probe_x, probe_bot), FG_PROBE_W, PROBE_H,
             '', BLUE_BG, BLUE_EDGE)

    ax.text(LAST_CX, probe_bot + PROBE_H - _LABEL_SPACE / 2,
            'probe', ha='center', va='center',
            fontsize=HEADING_SIZE - 1, fontweight='bold', color=BLUE_EDGE,
            zorder=4)

    bar_bot_y = probe_bot + _MARKER_PAD
    _draw_score_bar(ax, LAST_CX, bar_bot_y, FG_BAR_W, _BAR_H,
                    score_value, baseline_value=baseline_value)


FG_ROW_GAP = 0.45
bot_y = 0.0
mid_y = bot_y + FG_ROW_H + FG_ROW_GAP
top_y = mid_y + FG_ROW_H + FG_ROW_GAP
total_h = top_y + FG_ROW_H + 0.45

fig, ax = new_diagram(figsize=(9, 8.5), xlim=(-6, 10.5), ylim=(-0.2, total_h + 0.15))

ax.text(2.5, total_h + 0.05, 'Fine-grained preference',
        ha='center', fontsize=TITLE_SIZE, fontweight='bold')

_draw_row(ax, y_base=top_y,
          version_label='Version A', version_color=GREEN_EDGE,
          sentence_text='"...love discussing Shakespeare."',
          sentence_color=GREEN_BG, sentence_edge=GREEN_EDGE,
          score_value=0.75, baseline_value=0.3)

_draw_row(ax, y_base=mid_y,
          version_label='Version B', version_color=GREY_EDGE,
          sentence_text='"...love discussing hiking trails."',
          sentence_color=GREY_BG, sentence_edge=GREY_EDGE,
          score_value=0.3)

_draw_row(ax, y_base=bot_y,
          version_label='Version C', version_color=RED_EDGE,
          sentence_text='"...find Shakespeare tedious."',
          sentence_color=RED_BG, sentence_edge=RED_EDGE,
          score_value=-0.1, baseline_value=0.3)

save(fig, 'plot_022126_s3_3_fine_grained_preference.png')
