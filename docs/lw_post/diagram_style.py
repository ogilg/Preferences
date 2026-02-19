"""Shared style constants and helpers for LW post diagrams.

Import from here to keep fonts, colors, box shapes, and arrow styles
consistent across all setup/pipeline diagrams.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ASSETS_DIR = Path(__file__).parent / "assets"

# ── Colors ──

ORANGE_BG = '#FFF3E0'
ORANGE_EDGE = '#E65100'
ORANGE_FILL = '#FFCC80'

BLUE_BG = '#E3F2FD'
BLUE_EDGE = '#1565C0'

GREEN_BG = '#E8F5E9'
GREEN_EDGE = '#2E7D32'

GREY_BG = '#F5F5F5'
GREY_EDGE = '#757575'

RED_BG = '#FFEBEE'
RED_EDGE = '#C62828'

YELLOW_BG = '#FFFDE7'
YELLOW_EDGE = '#F9A825'

# ── Typography ──

FONT_FAMILY = 'sans-serif'
TITLE_SIZE = 13
HEADING_SIZE = 11
BODY_SIZE = 10
SMALL_SIZE = 9
CAPTION_SIZE = 8.5

plt.rcParams.update({
    'font.family': FONT_FAMILY,
    'font.size': BODY_SIZE,
})

# ── Box defaults ──

BOX_PAD = 0.02
BOX_LW = 1.5
ARROW_LW = 1.5
ARROW_COLOR = '#666'
DPI = 150


def draw_box(ax, xy, w, h, text, color=GREY_BG, edgecolor=GREY_EDGE,
             fontsize=BODY_SIZE, text_color='black', bold=False):
    box = FancyBboxPatch(xy, w, h, boxstyle=f"round,pad={BOX_PAD}",
                         facecolor=color, edgecolor=edgecolor,
                         linewidth=BOX_LW)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha='center', va='center',
            fontsize=fontsize, color=text_color, weight=weight,
            multialignment='center')
    return box


def draw_arrow(ax, start, end, color=ARROW_COLOR, style='->', lw=ARROW_LW):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))


def new_diagram(figsize=(14, 6), xlim=(-0.5, 13.5), ylim=(-0.5, 5.5)):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axis('off')
    return fig, ax


def save(fig, name):
    path = ASSETS_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches='tight', pad_inches=0.1,
                facecolor='white')
    plt.close(fig)
    print(f"Saved: {path}")
