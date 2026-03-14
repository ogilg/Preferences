"""Generate conversation structure diagrams for the truth probes section.

0. CREAK truth probes (raw claim as user message, probe at end-of-turn)
1. Error prefill (3-turn conversation, probe at follow-up)
2. Assistant selectors (probe reads from assistant turn)
3. Lying system prompts (adds system prompt box)
4. Per-token buildup (token-by-token scoring across assistant response)
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

COLORS = {
    "user": "#E3F2FD",
    "user_border": "#1565C0",
    "assistant": "#E8F5E9",
    "assistant_border": "#2E7D32",
    "system": "#FFF3E0",
    "system_border": "#E65100",
    "probe": "#F44336",
    "text": "#212121",
    "gray": "#9E9E9E",
    "correct": "#4CAF50",
    "incorrect": "#F44336",
}

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11

OUT = "docs/logs/assets"

# All figures use data coordinates 0..FW x 0..FH, with set_aspect("equal").
# figsize controls the physical size; data range controls layout.
FW = 14  # data width
FH_SHORT = 6
FH_MED = 7.5
FH_TALL = 8.5


def rbox(ax, x, y, w, h, fc, ec, alpha=1.0, lw=1.8):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.15",
        facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha))


def tbox(ax, x, y, w, h, role, content, bg, ec, alpha=1.0, fs_role=11, fs_content=10.5):
    rbox(ax, x, y, w, h, bg, ec, alpha=alpha)
    ax.text(x + w / 2, y + h - 0.35, role,
            fontsize=fs_role, fontweight="bold", color=ec,
            ha="center", va="top", alpha=alpha)
    ax.text(x + w / 2, y + h / 2 - 0.25, content,
            fontsize=fs_content, color=COLORS["text"],
            ha="center", va="center", style="italic", alpha=alpha,
            linespacing=1.4)


def arr(ax, x1, x2, y):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color=COLORS["gray"], lw=1.5))


def probe_arr(ax, x, y_bot, label, dy=-1.2):
    ax.annotate("", xy=(x, y_bot), xytext=(x, y_bot + dy),
                arrowprops=dict(arrowstyle="-|>", color=COLORS["probe"], lw=2.5))
    ax.text(x, y_bot + dy - 0.25, label,
            fontsize=10, color=COLORS["probe"], ha="center", va="top",
            fontweight="bold", linespacing=1.3)


def setup(ax, title, fw=FW, fh=FH_SHORT):
    ax.set_xlim(0, fw)
    ax.set_ylim(0, fh)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=15)


# ── Three-turn layout (error prefill) ──
BW, BH = 3.5, 3.0
BGAP = 0.6
BTOTAL = 3 * BW + 2 * BGAP
BX0 = (FW - BTOTAL) / 2
BXS = [BX0 + i * (BW + BGAP) for i in range(3)]
BY = 2.8


def three_turns(ax, y=BY, fade_follow=False):
    af = 0.3 if fade_follow else 1.0
    tbox(ax, BXS[0], y, BW, BH,
         "User", '"Is Belgium\'s capital\nBrussels?"',
         COLORS["user"], COLORS["user_border"])
    tbox(ax, BXS[1], y, BW, BH,
         "Model (prefilled)", '"No, Belgium\'s capital\nis Antwerp."',
         COLORS["assistant"], COLORS["assistant_border"])
    tbox(ax, BXS[2], y, BW, BH,
         "Follow-up user", '"Thank you for\nyour answer."',
         COLORS["user"], COLORS["user_border"], alpha=af)
    mid = y + BH / 2
    arr(ax, BXS[0] + BW + 0.08, BXS[1] - 0.08, mid)
    arr(ax, BXS[1] + BW + 0.08, BXS[2] - 0.08, mid)


# ============================================================
# 0. CREAK truth probes (simplest setup)
# ============================================================
def diagram_0():
    fig, ax = plt.subplots(figsize=(FW, FH_SHORT))
    setup(ax, "Do preference probes respond to truth?", fh=FH_SHORT)

    bw, bh = 4.5, 3.0
    gap = 1.0
    total = 2 * bw + gap
    x0 = (FW - total) / 2

    # Raw framing
    tbox(ax, x0, 2.8, bw, bh,
         "User (raw framing)",
         '"Marlboro used iconic imagery\nto promote its brand."',
         COLORS["user"], COLORS["user_border"], fs_content=11)

    # Repeat framing
    tbox(ax, x0 + bw + gap, 2.8, bw, bh,
         "User (repeat framing)",
         '"Please say the following:\nMarlboro used iconic imagery\nto promote its brand."',
         COLORS["user"], COLORS["user_border"], fs_content=11)

    # Probe arrows
    for bx in [x0, x0 + bw + gap]:
        probe_arr(ax, bx + bw / 2, 2.8,
                  "probe reads here\n(end-of-turn token)", dy=-1.0)

    # Results
    ax.text(x0 + bw / 2, 0.6, "d = 0.67", fontsize=13, ha="center",
            fontweight="bold", color=COLORS["probe"], family="monospace")
    ax.text(x0 + bw + gap + bw / 2, 0.6, "d = 2.18", fontsize=13, ha="center",
            fontweight="bold", color=COLORS["probe"], family="monospace")

    ax.text(FW / 2, 0.1,
            "The claim is the user message. No model response, no follow-up.\n"
            "Asking the model to produce the statement (repeat) roughly triples the effect.",
            fontsize=10, ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5",
                      edgecolor="#BDBDBD", linewidth=1))

    fig.savefig(f"{OUT}/plot_031226_truth_diagram_0_creak.png",
                dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


# ============================================================
# 1. Error prefill
# ============================================================
def diagram_1():
    fig, ax = plt.subplots(figsize=(FW, FH_SHORT))
    setup(ax, "Error prefill: does the preference probe detect mistakes?", fh=FH_SHORT)

    three_turns(ax)

    ax.text(BXS[1] + BW / 2, BY - 0.25,
            "answer is forced \u2014 not generated",
            fontsize=9.5, color=COLORS["assistant_border"],
            ha="center", style="italic")

    probe_arr(ax, BXS[2] + BW / 2, BY,
              "preference probe\nreads here\n(end-of-turn token)", dy=-1.2)

    ax.text(FW / 2, 0.4,
            'Correct prefill:  "Yes, Brussels."   \u2192   probe scores higher\n'
            'Incorrect prefill:  "No, Antwerp."   \u2192   probe scores lower',
            fontsize=11, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5",
                      edgecolor="#BDBDBD", linewidth=1))

    fig.savefig(f"{OUT}/plot_031226_truth_diagram_1_error_prefill.png",
                dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


# ============================================================
# 2. Assistant selectors
# ============================================================
def diagram_2():
    fig, ax = plt.subplots(figsize=(FW, FH_SHORT))
    setup(ax, "Signal is strongest at the source", fh=FH_SHORT)

    three_turns(ax, fade_follow=True)

    probe_arr(ax, BXS[1] + BW / 2, BY,
              "probe reads here instead\n(assistant end-of-turn)", dy=-1.2)

    ax.text(FW / 2, 0.5,
            "Assistant turn:     d = 3.29   (AUC = 0.98)\n"
            "Follow-up turn:     d = 2.58   (AUC = 0.95)\n\n"
            "The probe was never trained on assistant tokens\n"
            "\u2014 yet the signal is 28% stronger here.",
            fontsize=11, ha="center", va="center", family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5",
                      edgecolor="#BDBDBD", linewidth=1))

    fig.savefig(f"{OUT}/plot_031226_truth_diagram_2_assistant_selectors.png",
                dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


# ============================================================
# 3. Lying system prompts
# ============================================================
def diagram_3():
    fh = FH_TALL
    fig, ax = plt.subplots(figsize=(FW, fh))
    setup(ax, "Lying instructions disrupt the signal", fh=fh)

    # System prompt
    sys_y = 6.6
    sys_h = 1.4
    rbox(ax, BXS[0], sys_y, BTOTAL, sys_h,
         COLORS["system"], COLORS["system_border"])
    ax.text(BXS[0] + 0.3, sys_y + sys_h - 0.2, "System prompt",
            fontsize=10, fontweight="bold", color=COLORS["system_border"], va="top")
    ax.text(FW / 2, sys_y + sys_h / 2 - 0.1,
            '"Always give incorrect answers. Do not tell the truth."',
            fontsize=11, ha="center", va="center", style="italic")

    conv_y = 3.2
    three_turns(ax, y=conv_y)

    ax.text(FW / 2, conv_y - 0.35,
            "Same prefilled answers, same claims \u2014 only the system prompt changes.",
            fontsize=9.5, ha="center", style="italic", color=COLORS["gray"])

    cols = [
        ("No system prompt", "d = 3.29", COLORS["assistant_border"]),
        ("Direct lying", "d = \u22120.55", COLORS["probe"]),
        ("Roleplay lying", "d = 2.13", COLORS["system_border"]),
    ]
    for i, (lbl, dval, col) in enumerate(cols):
        cx = BXS[i] + BW / 2
        ax.text(cx, 2.2, lbl, fontsize=11, ha="center", fontweight="bold", color=col)
        ax.text(cx, 1.5, dval, fontsize=15, ha="center", fontweight="bold",
                color=col, family="monospace")

    ax.text(FW / 2, 0.4,
            "If the probe tracked bare truth value, lying instructions wouldn't matter.\n"
            "Instead: direct lying inverts the signal. The probe tracks\n"
            "the model's evaluative stance toward its own output.",
            fontsize=10, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5",
                      edgecolor="#BDBDBD", linewidth=1))

    fig.savefig(f"{OUT}/plot_031226_truth_diagram_3_lying.png",
                dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


# ============================================================
# 4. Per-token buildup
# ============================================================
def diagram_4():
    fh = 9.0
    fig, ax = plt.subplots(figsize=(FW, fh))
    setup(ax, "The signal builds token by token", fh=fh)

    # User box
    ux, uy, uw, uh = BXS[0], 6.5, 2.8, 2.0
    tbox(ax, ux, uy, uw, uh,
         "User", '"Is Belgium\'s capital\nBrussels?"',
         COLORS["user"], COLORS["user_border"])

    # Wide assistant box
    ax_x = ux + uw + BGAP
    aw = BXS[2] + BW - ax_x
    rbox(ax, ax_x, uy, aw, uh,
         COLORS["assistant"], COLORS["assistant_border"])
    ax.text(ax_x + aw / 2, uy + uh - 0.3, "Model (prefilled)",
            fontsize=11, fontweight="bold", color=COLORS["assistant_border"],
            ha="center", va="top")

    arr(ax, ux + uw + 0.05, ax_x - 0.05, uy + uh / 2)

    # Token boxes
    tokens = ["No", ",", "Belgium", "'s", "capital", "is", "Ant", "werp", "."]
    n = len(tokens)
    tw, tgap = 0.75, 0.1
    ttotal = n * tw + (n - 1) * tgap
    tx0 = ax_x + (aw - ttotal) / 2
    ty = uy + 0.25
    th = 0.8

    for i, tok in enumerate(tokens):
        frac = i / (n - 1)
        r = 0.92 - 0.52 * frac
        g = 0.92 - 0.70 * frac
        b = 0.92 - 0.70 * frac
        x = tx0 + i * (tw + tgap)
        ax.add_patch(FancyBboxPatch(
            (x, ty), tw, th, boxstyle="round,pad=0.05",
            facecolor=(r, g, b), edgecolor="#757575", linewidth=0.8))
        tcol = "white" if frac > 0.5 else COLORS["text"]
        ax.text(x + tw / 2, ty + th / 2, tok, fontsize=9, ha="center",
                va="center", family="monospace", color=tcol)

    # Buildup arrow
    ay = ty - 0.25
    ax.annotate("", xy=(tx0 + ttotal, ay), xytext=(tx0, ay),
                arrowprops=dict(arrowstyle="-|>", color=COLORS["probe"], lw=2))
    ax.text(tx0 + ttotal / 2, ay - 0.3,
            "signal builds from d \u2248 0 to d = 2.15",
            fontsize=10, ha="center", color=COLORS["probe"], fontweight="bold")

    # ── Mini trajectory plot ──
    px, py, pw, ph = 0.7, 0.3, 12.6, 4.8
    rbox(ax, px, py, pw, ph, "white", "#BDBDBD", lw=1)

    n_pts = 15
    xs = np.linspace(0, 1, n_pts)
    correct = np.array([-0.3, -0.5, -0.4, -0.5, -0.6, -0.5, -0.7, -0.6,
                         -0.7, -0.8, -0.6, -0.5, -0.3, -0.15, -0.1])
    incorrect = np.array([-0.3, -0.6, -0.8, -1.0, -1.3, -1.5, -1.8, -2.0,
                           -2.3, -2.5, -2.7, -2.8, -3.0, -3.2, -3.4])

    ml, mr, mb, mt = 1.0, 1.8, 0.7, 0.7

    def to_px(xf, yf):
        return (px + ml + xf * (pw - ml - mr),
                py + mb + (yf + 4) / 5 * (ph - mb - mt))

    np.random.seed(7)
    for _ in range(15):
        nc = correct + np.random.randn(n_pts) * 0.6
        ni = incorrect + np.random.randn(n_pts) * 0.6
        pc = [to_px(xs[j], nc[j]) for j in range(n_pts)]
        pi = [to_px(xs[j], ni[j]) for j in range(n_pts)]
        ax.plot([p[0] for p in pc], [p[1] for p in pc],
                color=COLORS["correct"], lw=0.5, alpha=0.15)
        ax.plot([p[0] for p in pi], [p[1] for p in pi],
                color=COLORS["incorrect"], lw=0.5, alpha=0.15)

    for trace, col, lbl in [(correct, COLORS["correct"], "correct"),
                             (incorrect, COLORS["incorrect"], "incorrect")]:
        pts = [to_px(xs[j], trace[j]) for j in range(n_pts)]
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                color=col, lw=3.5, solid_capstyle="round")
        ax.text(pts[-1][0] + 0.2, pts[-1][1], lbl, fontsize=11, color=col,
                va="center", fontweight="bold")

    p0c, p0i = to_px(0, correct[0]), to_px(0, incorrect[0])
    pNc, pNi = to_px(1, correct[-1]), to_px(1, incorrect[-1])

    ax.annotate("", xy=p0c, xytext=p0i,
                arrowprops=dict(arrowstyle="<->", color=COLORS["gray"], lw=1))
    ax.text(p0c[0] - 0.3, (p0c[1] + p0i[1]) / 2, "d \u2248 0",
            fontsize=10, ha="right", va="center", color=COLORS["gray"])

    ax.annotate("", xy=pNc, xytext=pNi,
                arrowprops=dict(arrowstyle="<->", color=COLORS["probe"], lw=1.5))
    ax.text(pNc[0] - 0.3, (pNc[1] + pNi[1]) / 2, "d = 2.15",
            fontsize=11, ha="right", va="center", color=COLORS["probe"],
            fontweight="bold")

    ax.text(px + pw / 2, py + 0.15,
            "token position in assistant response \u2192",
            fontsize=10, ha="center", color=COLORS["gray"])
    ax.text(px + 0.25, py + ph / 2, "probe\nscore",
            fontsize=10, ha="center", va="center", color=COLORS["gray"],
            rotation=90)

    fig.savefig(f"{OUT}/plot_031226_truth_diagram_4_per_token.png",
                dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


if __name__ == "__main__":
    for i, fn in enumerate([diagram_0, diagram_1, diagram_2, diagram_3, diagram_4]):
        fn()
        print(f"{i}: done")
    print(f"All saved to {OUT}/")
