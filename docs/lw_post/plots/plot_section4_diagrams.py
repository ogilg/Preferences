"""Section 4 diagrams: system prompt experiments.

Diagram 1: Simple preference — "You love cheese" (full baseline + delta)
Diagram 2: Crossed preference — compact (no baseline)
Diagram 3: Opposing prompts — compact (two prompts, no baseline)

Usage:
    cd docs/lw_post && python plot_section4_diagrams.py
"""

from plot_probe_template import draw_probe_measurement, row_height, PROMPT_H
from diagram_style import (
    ORANGE_EDGE,
    BLUE_EDGE,
    GREY_EDGE,
    TITLE_SIZE, HEADING_SIZE, SMALL_SIZE,
    new_diagram, save,
)

W = 9.0
CX = 2.5
RH = row_height()
ROW_GAP = 0.55


# ═══════════════════════════════════════════════════════════════
# 1: Simple preference — "You love cheese"
# Full format: baseline + manipulated with delta
# ═══════════════════════════════════════════════════════════════

bot_y = 0.0
top_y = bot_y + RH + ROW_GAP
total_h = top_y + RH + 0.45

fig, ax = new_diagram(figsize=(7, 4.6), xlim=(-4.5, 9.5), ylim=(-0.2, total_h + 0.15))

ax.text(2.0, total_h + 0.05, 'Simple preference shift',
        ha='center', fontsize=TITLE_SIZE, fontweight='bold')

ax.text(-4.2, top_y + PROMPT_H + 0.45, 'Baseline', ha='left',
        fontsize=HEADING_SIZE, fontweight='bold', color=GREY_EDGE)
ax.text(-4.2, top_y + PROMPT_H + 0.15, '(no system prompt)', ha='left',
        fontsize=SMALL_SIZE, color=GREY_EDGE, fontstyle='italic')

draw_probe_measurement(
    ax, x_center=CX, y_top=top_y,
    task_text='"Write a guide to\nartisanal cheese"',
    score_value=0.15,
    width=W,
)

ax.text(-4.2, bot_y + PROMPT_H + 0.45, 'With system prompt', ha='left',
        fontsize=HEADING_SIZE, fontweight='bold', color=ORANGE_EDGE)

draw_probe_measurement(
    ax, x_center=CX, y_top=bot_y,
    task_text='"Write a guide to\nartisanal cheese"',
    system_prompt='"You love cheese"',
    score_value=0.7,
    baseline_value=0.15,
    width=W,
)

save(fig, 'plot_022626_s4_1_simple_preference.png')


# ═══════════════════════════════════════════════════════════════
# 2: Crossed preference — compact (no baseline)
# "You hate cheese" + a math problem about cheese
# ═══════════════════════════════════════════════════════════════

single_h = RH + 0.45

fig, ax = new_diagram(figsize=(7, 3.0), xlim=(-4.5, 9.5), ylim=(-0.2, single_h + 0.15))

ax.text(2.0, single_h + 0.05, 'Crossed preference',
        ha='center', fontsize=TITLE_SIZE, fontweight='bold')

ax.text(-4.2, PROMPT_H + 0.45, '"You hate cheese"', ha='left',
        fontsize=HEADING_SIZE, fontweight='bold', color=ORANGE_EDGE)

draw_probe_measurement(
    ax, x_center=CX, y_top=0.0,
    task_text='"A farmer makes wheels\nof cheese. If each..."',
    system_prompt='"You hate cheese"',
    score_value=-0.3,
    baseline_value=0.4,
    width=W,
)

ax.text(-4.2, -0.1, 'math task about cheese', ha='left', fontsize=SMALL_SIZE,
        color='#666', fontstyle='italic')

save(fig, 'plot_022626_s4_2_crossed_preference.png')


# ═══════════════════════════════════════════════════════════════
# 3: Opposing prompts — compact (two prompts, no baseline)
# Same task, opposite system prompts
# ═══════════════════════════════════════════════════════════════

bot_y = 0.0
top_y = bot_y + RH + ROW_GAP
total_h = top_y + RH + 0.45

fig, ax = new_diagram(figsize=(7, 4.6), xlim=(-4.5, 9.5), ylim=(-0.2, total_h + 0.15))

ax.text(2.0, total_h + 0.05, 'Opposing prompts',
        ha='center', fontsize=TITLE_SIZE, fontweight='bold')

ax.text(-4.2, top_y + PROMPT_H + 0.45, 'Prompt A', ha='left',
        fontsize=HEADING_SIZE, fontweight='bold', color=ORANGE_EDGE)

draw_probe_measurement(
    ax, x_center=CX, y_top=top_y,
    task_text='"Write a guide to\nartisanal cheese"',
    system_prompt='"You love cheese,\nhate math"',
    score_value=0.7,
    baseline_value=0.15,
    width=W,
)

ax.text(-4.2, bot_y + PROMPT_H + 0.45, 'Prompt B', ha='left',
        fontsize=HEADING_SIZE, fontweight='bold', color=BLUE_EDGE)

draw_probe_measurement(
    ax, x_center=CX, y_top=bot_y,
    task_text='"Write a guide to\nartisanal cheese"',
    system_prompt='"You love math,\nhate cheese"',
    score_value=-0.5,
    baseline_value=0.15,
    width=W,
)

save(fig, 'plot_022626_s4_3_competing_values.png')
