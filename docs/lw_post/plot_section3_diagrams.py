"""Section 3 diagrams: Utility probes behave like evaluative representations.

One diagram per subsection, all using the probe measurement template.

Usage:
    cd docs/lw_post && python plot_section3_diagrams.py
"""

from plot_probe_template import draw_probe_measurement
from diagram_style import (
    ORANGE_EDGE, GREY_EDGE,
    TITLE_SIZE, HEADING_SIZE, BODY_SIZE,
    new_diagram, save,
)

W = 11.0
CX = 3.0


# ═══════════════════════════════════════════════════════════════
# 3.1: Category preferences — "You hate math" shifts probe on math tasks
# Both baseline and manipulated, stacked
# ═══════════════════════════════════════════════════════════════

fig, ax = new_diagram(figsize=(8, 7.5), xlim=(-5, 11), ylim=(-0.5, 8))

ax.text(2.5, 7.6, 'System prompt induces category preference',
        ha='center', fontsize=TITLE_SIZE, fontweight='bold')

ax.text(-4.7, 6.2, 'Baseline', ha='left', fontsize=HEADING_SIZE,
        fontweight='bold', color=GREY_EDGE)

draw_probe_measurement(
    ax, x_center=CX, y_top=4.2,
    task_text='"Solve this integral..."',
    score_value=0.75,
    width=W,
)

ax.text(-4.7, 2.0, 'Manipulated', ha='left', fontsize=HEADING_SIZE,
        fontweight='bold', color=ORANGE_EDGE)

draw_probe_measurement(
    ax, x_center=CX, y_top=0.0,
    task_text='"Solve this integral..."',
    system_prompt='"You hate math"',
    score_value=-0.4,
    baseline_value=0.75,
    width=W,
)

save(fig, 'plot_021826_s3_1_category_preference.png')


# ═══════════════════════════════════════════════════════════════
# 3.2: Targeted preference — novel topic, not in training categories
# Only manipulated
# ═══════════════════════════════════════════════════════════════

fig, ax = new_diagram(figsize=(8, 4.5), xlim=(-5, 11), ylim=(-0.5, 4.5))

ax.text(2.5, 4.1, 'System prompt induces targeted preference',
        ha='center', fontsize=TITLE_SIZE, fontweight='bold')

draw_probe_measurement(
    ax, x_center=CX, y_top=0.0,
    task_text='"Write a guide to\nartisanal cheese"',
    system_prompt='"You find cheese\nrevolting"',
    score_value=-0.6,
    baseline_value=0.1,
    width=W,
)

save(fig, 'plot_021826_s3_2_targeted_preference.png')


# ═══════════════════════════════════════════════════════════════
# 3.3: Competing prompts — same words, flipped evaluation
# Stacked vertically
# ═══════════════════════════════════════════════════════════════

fig, ax = new_diagram(figsize=(8, 7.5), xlim=(-5, 11), ylim=(-0.5, 8))

ax.text(2.5, 7.6, 'Competing prompts — same words,\nflipped evaluation',
        ha='center', fontsize=TITLE_SIZE, fontweight='bold',
        multialignment='center')

ax.text(-4.7, 5.8, 'Prompt A', ha='left', fontsize=HEADING_SIZE,
        fontweight='bold', color=ORANGE_EDGE)

draw_probe_measurement(
    ax, x_center=CX, y_top=3.8,
    task_text='"Cheese factory\nrevenue calculation"',
    system_prompt='"You love cheese,\nhate math"',
    score_value=-0.5,
    baseline_value=0.1,
    width=W,
)

ax.text(-4.7, 1.7, 'Prompt B', ha='left', fontsize=HEADING_SIZE,
        fontweight='bold', color=ORANGE_EDGE)

draw_probe_measurement(
    ax, x_center=CX, y_top=-0.3,
    task_text='"Cheese factory\nrevenue calculation"',
    system_prompt='"You love math,\nhate cheese"',
    score_value=0.3,
    baseline_value=0.1,
    width=W,
)

save(fig, 'plot_021826_s3_3_competing_prompts.png')


# ═══════════════════════════════════════════════════════════════
# 3.4: Persona-induced — broad role shifts multiple categories
# Only manipulated
# ═══════════════════════════════════════════════════════════════

fig, ax = new_diagram(figsize=(8, 4.5), xlim=(-5, 11), ylim=(-0.5, 4.5))

ax.text(2.5, 4.1, 'Persona induces broad preference shifts',
        ha='center', fontsize=TITLE_SIZE, fontweight='bold')

draw_probe_measurement(
    ax, x_center=CX, y_top=0.0,
    task_text='"Write a short story\nabout a dragon"',
    system_prompt='persona: STEM\nenthusiast',
    score_value=-0.3,
    baseline_value=0.5,
    width=W,
)

save(fig, 'plot_021826_s3_4_persona_preference.png')


# ═══════════════════════════════════════════════════════════════
# 3.5: Narrow preference — very specific task targeting
# Only manipulated
# ═══════════════════════════════════════════════════════════════

fig, ax = new_diagram(figsize=(8, 4.5), xlim=(-5, 11), ylim=(-0.5, 4.5))

ax.text(2.5, 4.1, 'Isolating narrow evaluative signal',
        ha='center', fontsize=TITLE_SIZE, fontweight='bold')

draw_probe_measurement(
    ax, x_center=CX, y_top=0.0,
    task_text='"Write a pipe organ\nregistration program"',
    system_prompt='persona: pipe organ\nenthusiast',
    score_value=0.8,
    baseline_value=0.1,
    width=W,
)

save(fig, 'plot_021826_s3_5_narrow_preference.png')
