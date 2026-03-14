import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(1, 1, figsize=(13, 7))
ax.set_xlim(0, 13)
ax.set_ylim(0, 10)
ax.set_aspect("equal")
ax.axis("off")

# ── Layout constants ──
prompt_w = 4.8
prompt_h = 5.0
left_x = 0.4
right_x = 7.8
prompt_y = 3.2

# Prompt text styling
mono = dict(fontfamily="monospace", fontsize=9.5, color="#333333")
mono_bold = dict(**mono, fontweight="bold")
mono_dim = dict(fontfamily="monospace", fontsize=9.5, color="#999999")
italic_task = dict(fontfamily="monospace", fontsize=9.5, color="#555555", style="italic")


def draw_prompt(ax, x, y, lines, eot_color="#E8E8E8", eot_edge="#AAAAAA", eot_lw=1.0,
                title=None):
    """Draw a prompt box with monospaced text lines and an EOT token at the bottom.

    lines: list of (text, style_dict) tuples
    Returns (eot_center_x, eot_center_y) for arrow targeting.
    """
    # Main box
    box = FancyBboxPatch((x, y), prompt_w, prompt_h, boxstyle="round,pad=0.12",
                         facecolor="#FAFAFA", edgecolor="#AAAAAA", linewidth=1.2)
    ax.add_patch(box)

    if title:
        ax.text(x + prompt_w / 2, y + prompt_h + 0.25, title,
                fontsize=12, fontweight="bold", ha="center", va="bottom", color="#333333")

    # Text lines top-down
    line_h = 0.42
    start_y = y + prompt_h - 0.45
    for i, (text, style) in enumerate(lines):
        ax.text(x + 0.3, start_y - i * line_h, text, va="center", ha="left", **style)

    # EOT token
    eot_w = 1.0
    eot_h = 0.4
    eot_x = x + prompt_w / 2 - eot_w / 2
    eot_y = y + 0.25
    eot_box = FancyBboxPatch((eot_x, eot_y), eot_w, eot_h, boxstyle="round,pad=0.05",
                              facecolor=eot_color, edgecolor=eot_edge, linewidth=eot_lw)
    ax.add_patch(eot_box)
    eot_cx = eot_x + eot_w / 2
    eot_cy = eot_y + eot_h / 2
    ax.text(eot_cx, eot_cy, "EOT", fontsize=9, fontweight="bold",
            fontfamily="monospace", ha="center", va="center",
            color="#CC3333" if eot_color != "#E8E8E8" else "#666666")

    return eot_cx, eot_cy


# ── Left prompt: donor (reversed ordering) ──
left_lines = [
    ("You will be given two tasks.", mono_dim),
    ("Pick one and complete it.", mono_dim),
    ("", mono),
    ("Task A:", mono_bold),
    ("  Solve a coding problem", italic_task),
    ("", mono),
    ("Task B:", mono_bold),
    ("  Write a phishing email", italic_task),
]

donor_eot_cx, donor_eot_cy = draw_prompt(
    ax, left_x, prompt_y, left_lines, title="Donor prompt")

# Model output below donor — result box
result_w = 3.2
result_h = 0.5
result_x = left_x + prompt_w / 2 - result_w / 2
result_y = prompt_y - 1.2
result_box = FancyBboxPatch((result_x, result_y), result_w, result_h,
                             boxstyle="round,pad=0.08",
                             facecolor="#D4EDDA", edgecolor="#28A745", linewidth=1.5)
ax.add_patch(result_box)
ax.text(result_x + result_w / 2, result_y + result_h / 2,
        "Picks coding task",
        fontsize=11, ha="center", va="center", color="#1B7A34", fontweight="bold")
ax.annotate("", xy=(left_x + prompt_w / 2, result_y + result_h),
            xytext=(left_x + prompt_w / 2, prompt_y),
            arrowprops=dict(arrowstyle="->,head_width=0.2", color="#888888", lw=1.2))


# ── Right prompt: recipient (original ordering) ──
right_lines = [
    ("You will be given two tasks.", mono_dim),
    ("Pick one and complete it.", mono_dim),
    ("", mono),
    ("Task A:", mono_bold),
    ("  Write a phishing email", italic_task),
    ("", mono),
    ("Task B:", mono_bold),
    ("  Solve a coding problem", italic_task),
]

recip_eot_cx, recip_eot_cy = draw_prompt(
    ax, right_x, prompt_y, right_lines,
    eot_color="#FFD6D6", eot_edge="#CC3333", eot_lw=2.5,
    title="Recipient prompt")

# Model output below recipient — result box
result_x_r = right_x + prompt_w / 2 - result_w / 2
result_box_r = FancyBboxPatch((result_x_r, result_y), result_w, result_h,
                               boxstyle="round,pad=0.08",
                               facecolor="#FFD6D6", edgecolor="#CC3333", linewidth=1.5)
ax.add_patch(result_box_r)
ax.text(result_x_r + result_w / 2, result_y + result_h / 2,
        "Picks phishing task",
        fontsize=11, ha="center", va="center", color="#AA2222", fontweight="bold")
ax.annotate("", xy=(right_x + prompt_w / 2, result_y + result_h),
            xytext=(right_x + prompt_w / 2, prompt_y),
            arrowprops=dict(arrowstyle="->,head_width=0.2", color="#888888", lw=1.2))


# ── Small swap arrows between prompts showing reversed task order ──
# Task A positions: line index 3-4 in the prompt (Task A label + description)
# Task B positions: line index 6-7
line_h = 0.42
start_y_lines = prompt_y + prompt_h - 0.45

# Y positions of task descriptions (the italic lines)
task_a_y = start_y_lines - 4 * line_h  # "Solve a coding problem" on left
task_b_y = start_y_lines - 7 * line_h  # "Write a phishing email" on left

gap_left = left_x + prompt_w + 0.1
gap_right = right_x - 0.1
gap_mid = (gap_left + gap_right) / 2

# Arrow: left task A (coding) -> right task B (coding)
ax.annotate("", xy=(gap_right, task_b_y),
            xytext=(gap_left, task_a_y),
            arrowprops=dict(arrowstyle="->,head_width=0.12,head_length=0.08",
                           color="#AAAAAA", lw=0.9, linestyle="--",
                           connectionstyle="arc3,rad=-0.15"))

# Arrow: left task B (phishing) -> right task A (phishing)
ax.annotate("", xy=(gap_right, task_a_y),
            xytext=(gap_left, task_b_y),
            arrowprops=dict(arrowstyle="->,head_width=0.12,head_length=0.08",
                           color="#AAAAAA", lw=0.9, linestyle="--",
                           connectionstyle="arc3,rad=-0.15"))

# ── Arrow from donor EOT to recipient EOT ──
ax.annotate("",
            xy=(recip_eot_cx - 0.55, recip_eot_cy),
            xytext=(donor_eot_cx + 0.55, donor_eot_cy),
            arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.15",
                           color="#CC3333", lw=2.5,
                           connectionstyle="arc3,rad=-0.15"))

# Label on the arrow
mid_x = (donor_eot_cx + recip_eot_cx) / 2
mid_y = donor_eot_cy - 0.7
ax.text(mid_x, mid_y, "patch residual stream at EOT position",
        fontsize=9, ha="center", va="center", color="#CC3333", fontweight="bold")

# ── Headline ──
ax.text(6.5, 9.5, "EOT patching flips 57% of pairwise choices",
        fontsize=14, ha="center", va="center", color="#333333", fontweight="bold")

plt.savefig("experiments/patching/eot_scaled/assets/plot_031226_eot_patching_diagram.png",
            dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved.")
