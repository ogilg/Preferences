"""Probe pipeline diagram for LW post section 3.

Shows: task text → forward pass → residual stream activations (X)
       pairwise choices → utility fitting → μ vector
       Ridge probe: μ̂ = Xw

Usage:
    cd docs/lw_post && python plot_probe_pipeline.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from diagram_style import (
    ORANGE_BG, ORANGE_EDGE, ORANGE_FILL,
    BLUE_BG, BLUE_EDGE,
    GREEN_BG, GREEN_EDGE,
    GREY_BG, GREY_EDGE,
    TITLE_SIZE, HEADING_SIZE, BODY_SIZE, SMALL_SIZE, CAPTION_SIZE,
    BOX_LW,
    draw_box, draw_arrow, new_diagram, save,
)

fig, ax = new_diagram(figsize=(14, 9), xlim=(-0.5, 13.5), ylim=(-0.8, 8.0))

# ── Titles ──
ax.text(3.5, 7.7, 'Probe input', ha='center', fontsize=TITLE_SIZE,
        fontweight='bold', color=ORANGE_EDGE)
ax.text(10.5, 7.7, 'Training signal', ha='center', fontsize=TITLE_SIZE,
        fontweight='bold', color=BLUE_EDGE)

# ═══════════════════════════════════════════════════
# LEFT SIDE: Probe input
# ═══════════════════════════════════════════════════

# Token boxes
token_texts = ['Write', 'a', 'poem', 'about', '...']
token_x_start = 0.5
token_w = 1.1
token_h = 0.55
token_gap = 0.05
token_y = 6.8

for i, t in enumerate(token_texts):
    x = token_x_start + i * (token_w + token_gap)
    draw_box(ax, (x, token_y), token_w, token_h, t,
             'white', GREY_EDGE, fontsize=SMALL_SIZE)

# End-of-turn token
eot_x = token_x_start + len(token_texts) * (token_w + token_gap)
draw_box(ax, (eot_x, token_y), token_w, token_h, '<end\nturn>',
         ORANGE_FILL, ORANGE_EDGE, fontsize=SMALL_SIZE, bold=True)

# "residual stream" label
eot_cx = eot_x + token_w / 2
ax.text(eot_cx, token_y - 0.15, 'residual stream\nat this position',
        ha='center', va='top', fontsize=CAPTION_SIZE, color=ORANGE_EDGE,
        fontstyle='italic')

# Arrow down to forward pass box
draw_arrow(ax, (3.5, token_y), (3.5, 6.15))

# Forward pass box
draw_box(ax, (1.0, 5.5), 5.0, 0.65, 'Gemma-3-27B\nforward pass',
         ORANGE_BG, ORANGE_EDGE, fontsize=BODY_SIZE, bold=True)

# Arrow down to activation matrix
draw_arrow(ax, (3.5, 5.5), (3.5, 4.95))

# ── Activation matrix (heatmap) with text brackets ──
mat_x, mat_y = 1.0, 3.1
mat_w, mat_h = 5.0, 1.8
# Inset heatmap slightly so brackets wrap around it
inset = 0.12

# Brackets sized to wrap the full matrix area
ax.text(mat_x - 0.05, mat_y + mat_h / 2, '\u23a1\n\u23a2\n\u23a3',
        ha='right', va='center', fontsize=38, color='black',
        family='DejaVu Sans', linespacing=0.55)
ax.text(mat_x + mat_w + 0.05, mat_y + mat_h / 2, '\u23a4\n\u23a5\n\u23a6',
        ha='left', va='center', fontsize=38, color='black',
        family='DejaVu Sans', linespacing=0.55)

# Random heatmap — more rows/cols for smaller squares like the original
rng = np.random.RandomState(42)
data = rng.rand(8, 16)
ax.imshow(data, aspect='auto', cmap='Oranges',
          extent=[mat_x + inset, mat_x + mat_w - inset,
                  mat_y + inset, mat_y + mat_h - inset],
          zorder=2, interpolation='nearest')

# X label
ax.text(mat_x - 0.6, mat_y + mat_h / 2, r'$\mathbf{X}$',
        ha='center', va='center', fontsize=22, fontweight='bold')

# Dimensions
ax.text(mat_x + mat_w / 2, mat_y - 0.2,
        '10,000 tasks × 5,376 dims',
        ha='center', fontsize=CAPTION_SIZE, color='#666')

# ═══════════════════════════════════════════════════
# RIGHT SIDE: Training signal
# ═══════════════════════════════════════════════════

# Pairwise choices box
draw_box(ax, (8.0, 6.6), 4.5, 0.85, 'Pairwise choices\n"Task A or Task B?"',
         BLUE_BG, BLUE_EDGE, fontsize=BODY_SIZE, bold=True)

# Arrow + comparisons label
ax.text(10.25, 6.35, '~150k comparisons', ha='center', fontsize=CAPTION_SIZE,
        color='#666', fontstyle='italic')
draw_arrow(ax, (10.25, 6.6), (10.25, 6.0))

# Utility fitting box
draw_box(ax, (8.0, 5.35), 4.5, 0.65, 'Fit utility function',
         BLUE_BG, BLUE_EDGE, fontsize=BODY_SIZE, bold=True)

# Arrow down to μ vector
draw_arrow(ax, (10.25, 5.35), (10.25, 4.95))

# ── μ vector — narrow column of square-ish cells ──
vec_w = 0.7
vec_x = 10.25 - vec_w / 2  # centered on the right-side axis
vec_y = 3.1
vec_h = 1.8

# Brackets wrapping the vector
ax.text(vec_x - 0.05, vec_y + vec_h / 2, '\u23a1\n\u23a2\n\u23a3',
        ha='right', va='center', fontsize=38, color='black',
        family='DejaVu Sans', linespacing=0.55)
ax.text(vec_x + vec_w + 0.05, vec_y + vec_h / 2, '\u23a4\n\u23a5\n\u23a6',
        ha='left', va='center', fontsize=38, color='black',
        family='DejaVu Sans', linespacing=0.55)

# Blue gradient squares
n_bars = 8
cell_h = (vec_h - 2 * inset) / n_bars
values = [0.3, 0.7, 0.5, 0.9, 0.2, 0.6, 0.8, 0.4]
for i, v in enumerate(values):
    by = vec_y + inset + (n_bars - 1 - i) * cell_h
    bar_color = plt.cm.Blues(0.3 + 0.5 * v)
    ax.add_patch(mpatches.Rectangle(
        (vec_x + inset, by), vec_w - 2 * inset, cell_h * 0.85,
        facecolor=bar_color, edgecolor='none', zorder=2))

# μ label
ax.text(vec_x + vec_w + 0.5, vec_y + vec_h / 2, r'$\boldsymbol{\mu}$',
        ha='center', va='center', fontsize=22, fontweight='bold')

# Dimensions
ax.text(vec_x + vec_w / 2, vec_y - 0.2,
        '10,000 tasks × 1',
        ha='center', fontsize=CAPTION_SIZE, color='#666')

# ═══════════════════════════════════════════════════
# BOTTOM: Ridge probe
# ═══════════════════════════════════════════════════

# Arrows from X and μ down to probe box
draw_arrow(ax, (mat_x + mat_w / 2, mat_y - 0.35), (mat_x + mat_w / 2, 1.95))
draw_arrow(ax, (vec_x + vec_w / 2, vec_y - 0.35), (vec_x + vec_w / 2, 1.95))

# Probe box — tall enough to contain all three lines
draw_box(ax, (1.5, 0.0), 10.5, 1.95, '', GREEN_BG, GREEN_EDGE)
ax.text(6.75, 1.55, 'Train a Ridge probe', ha='center', fontsize=HEADING_SIZE + 2,
        fontweight='bold', color=GREEN_EDGE)
ax.text(6.75, 0.95, r'$\hat{\mu} = \mathbf{X}\mathbf{w}$',
        ha='center', fontsize=18)
ax.text(6.75, 0.3, r'$\mathbf{w} \in \mathbb{R}^{5376}$  —  single linear direction, Ridge-regularized',
        ha='center', fontsize=SMALL_SIZE, fontstyle='italic', color='#555')

save(fig, 'plot_022626_probe_pipeline.png')
