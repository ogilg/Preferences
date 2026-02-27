"""Setup diagrams for LW post sections.

Uses shared style from diagram_style.py.

Usage:
    python docs/lw_post/plot_setup_diagrams.py
"""

from diagram_style import (
    ORANGE_BG, ORANGE_EDGE, ORANGE_FILL,
    BLUE_BG, BLUE_EDGE,
    GREEN_BG, GREEN_EDGE,
    GREY_BG, GREY_EDGE,
    YELLOW_BG, YELLOW_EDGE,
    TITLE_SIZE, HEADING_SIZE, BODY_SIZE, SMALL_SIZE, CAPTION_SIZE,
    draw_box, draw_arrow, new_diagram, save,
)

# ═══════════════════════════════════════════════════════════════
# DIAGRAM 1: Section 3.2 — System prompt → probe tracking
# ═══════════════════════════════════════════════════════════════

fig, ax = new_diagram()

ax.text(6.5, 5.3, 'Section 3.2: Does the probe track system-prompt-induced preferences?',
        ha='center', fontsize=TITLE_SIZE, fontweight='bold')

# ── Top row: the two conditions ──

# Condition 1: no system prompt (baseline)
draw_box(ax, (0, 3.5), 4.5, 1.3, '', GREY_BG, GREY_EDGE)
ax.text(2.25, 4.55, 'Baseline (no system prompt)', ha='center', fontsize=BODY_SIZE,
        fontweight='bold', color=GREY_EDGE)
draw_box(ax, (0.3, 3.7), 3.9, 0.6, '"Write a poem about cats"',
         'white', GREY_EDGE, fontsize=SMALL_SIZE)

# Condition 2: with system prompt
draw_box(ax, (5.5, 3.5), 7.5, 1.3, '', ORANGE_BG, ORANGE_EDGE)
ax.text(9.25, 4.55, 'With system prompt', ha='center', fontsize=BODY_SIZE,
        fontweight='bold', color=ORANGE_EDGE)
draw_box(ax, (5.8, 3.95), 3.2, 0.6, '"You hate math"',
         ORANGE_FILL, ORANGE_EDGE, fontsize=SMALL_SIZE, bold=True)
ax.text(9.15, 4.25, '+', ha='center', fontsize=12, color=ORANGE_EDGE)
draw_box(ax, (9.4, 3.95), 3.3, 0.6, '"Solve this integral..."',
         'white', ORANGE_EDGE, fontsize=SMALL_SIZE)
ax.text(9.25, 3.7, '(math task)', ha='center', fontsize=CAPTION_SIZE, color='#888',
        fontstyle='italic')

# ── Middle row: forward pass + probe ──

draw_arrow(ax, (2.25, 3.5), (2.25, 2.7))
draw_arrow(ax, (9.25, 3.5), (9.25, 2.7))

draw_box(ax, (0.5, 1.8), 12, 0.9, '', GREEN_BG, GREEN_EDGE)
ax.text(6.5, 2.25, 'Trained probe (fixed)  —  extract activations, apply w',
        ha='center', fontsize=HEADING_SIZE, color=GREEN_EDGE, fontweight='bold')

# ── Bottom row: outputs ──

draw_arrow(ax, (2.25, 1.8), (2.25, 1.1))
draw_arrow(ax, (9.25, 1.8), (9.25, 1.1))

draw_box(ax, (0.75, 0.3), 3, 0.8, 'Probe score:\nbaseline',
         GREY_BG, GREY_EDGE)
draw_box(ax, (7.75, 0.3), 3, 0.8, 'Probe score:\nmanipulated',
         ORANGE_BG, ORANGE_EDGE)

ax.annotate('', xy=(7.6, 0.7), xytext=(3.9, 0.7),
            arrowprops=dict(arrowstyle='<->', color='#333', lw=2))
ax.text(5.75, 0.9, 'probe delta', ha='center', fontsize=BODY_SIZE, fontweight='bold')
ax.text(5.75, 0.45, 'compare to\nbehavioral delta', ha='center', fontsize=CAPTION_SIZE,
        color='#666', fontstyle='italic')

save(fig, 'plot_021826_setup_system_prompt.png')


# ═══════════════════════════════════════════════════════════════
# DIAGRAM 2: Section 3.3 — Competing prompts
# ═══════════════════════════════════════════════════════════════

fig, ax = new_diagram(figsize=(14, 7.5), ylim=(-1.2, 7))

ax.text(6.5, 6.7, 'Section 3.3: Competing prompts — same words, flipped evaluation',
        ha='center', fontsize=TITLE_SIZE, fontweight='bold')

# ── Top row: the two competing prompts ──

draw_box(ax, (0, 5), 5.8, 1.2, '', ORANGE_BG, ORANGE_EDGE)
ax.text(2.9, 5.95, 'Prompt A', ha='center', fontsize=BODY_SIZE,
        fontweight='bold', color=ORANGE_EDGE)
ax.text(2.9, 5.45, '"You love cheese\nbut find math tedious"',
        ha='center', fontsize=9.5, fontstyle='italic')

draw_box(ax, (7.2, 5), 5.8, 1.2, '', BLUE_BG, BLUE_EDGE)
ax.text(10.1, 5.95, 'Prompt B', ha='center', fontsize=BODY_SIZE,
        fontweight='bold', color=BLUE_EDGE)
ax.text(10.1, 5.45, '"You love math\nbut find cheese revolting"',
        ha='center', fontsize=9.5, fontstyle='italic')

ax.text(6.5, 5.6, 'same\nwords', ha='center', fontsize=SMALL_SIZE, color='#888',
        fontweight='bold', fontstyle='italic')

# ── Middle row: forward pass on task groups ──

draw_arrow(ax, (2.9, 5.0), (2.9, 4.2))
draw_arrow(ax, (10.1, 5.0), (10.1, 4.2))

draw_box(ax, (0.3, 3.2), 12.5, 1.0, '', GREEN_BG, GREEN_EDGE)
ax.text(6.5, 3.7, 'Apply trained probe to activations from each task group',
        ha='center', fontsize=HEADING_SIZE, color=GREEN_EDGE, fontweight='bold')

# ── Bottom: task groups and expected results ──

draw_arrow(ax, (3.5, 3.2), (3.5, 2.5))
draw_arrow(ax, (10, 3.2), (10, 2.5))

draw_box(ax, (1, 1.5), 5, 1.0, '', GREEN_BG, '#43A047')
ax.text(3.5, 2.2, 'Other cheese tasks', ha='center', fontsize=BODY_SIZE,
        fontweight='bold', color=GREEN_EDGE)
ax.text(3.5, 1.75, '(cheese-coding, cheese-fiction, ...)',
        ha='center', fontsize=CAPTION_SIZE, color='#666', fontstyle='italic')

draw_box(ax, (7.5, 1.5), 5, 1.0, '', BLUE_BG, '#1E88E5')
ax.text(10, 2.2, 'Other math tasks', ha='center', fontsize=BODY_SIZE,
        fontweight='bold', color=BLUE_EDGE)
ax.text(10, 1.75, '(cats-math, gardening-math, ...)',
        ha='center', fontsize=CAPTION_SIZE, color='#666', fontstyle='italic')

draw_arrow(ax, (3.5, 1.5), (3.5, 0.8))
draw_arrow(ax, (10, 1.5), (10, 0.8))

draw_box(ax, (0.5, -0.2), 6, 1.0, '', GREEN_BG, '#43A047')
ax.text(3.5, 0.55, 'Prompt A: score near baseline',
        ha='center', fontsize=SMALL_SIZE, color=ORANGE_EDGE, fontweight='bold')
ax.text(3.5, 0.1, 'Prompt B: score drops sharply',
        ha='center', fontsize=SMALL_SIZE, color=BLUE_EDGE, fontweight='bold')
ax.text(3.5, -0.25, '(cheese loved in A, hated in B)',
        ha='center', fontsize=CAPTION_SIZE - 0.5, color='#666', fontstyle='italic')

draw_box(ax, (7, -0.2), 6, 1.0, '', BLUE_BG, '#1E88E5')
ax.text(10, 0.55, 'Prompt A: score drops sharply',
        ha='center', fontsize=SMALL_SIZE, color=ORANGE_EDGE, fontweight='bold')
ax.text(10, 0.1, 'Prompt B: score near baseline',
        ha='center', fontsize=SMALL_SIZE, color=BLUE_EDGE, fontweight='bold')
ax.text(10, -0.25, '(math hated in A, loved in B)',
        ha='center', fontsize=CAPTION_SIZE - 0.5, color='#666', fontstyle='italic')

ax.text(6.5, -0.85, 'Same content words in both prompts → a content detector sees no difference.\n'
        'The probe sees the evaluative flip.',
        ha='center', fontsize=BODY_SIZE, fontstyle='italic', color='#333',
        bbox=dict(boxstyle='round,pad=0.4', facecolor=YELLOW_BG, edgecolor=YELLOW_EDGE, alpha=0.9))

save(fig, 'plot_021826_setup_competing_prompts.png')
