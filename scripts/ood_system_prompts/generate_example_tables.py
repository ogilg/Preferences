"""Generate markdown example tables for the OOD report from JSON data.

Reads the JSON files produced by dump_top_deltas.py. For each experiment,
picks one representative condition (highest absolute Pearson r between
behavioral and probe deltas) and shows the top 3 and bottom 3 tasks by
probe delta.

Output: writes the examples section directly into the report, replacing
the existing Task-Level Examples section.
"""

import json
import re
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = REPO_ROOT / "experiments" / "ood_system_prompts" / "task_examples"
REPORT_PATH = REPO_ROOT / "experiments" / "ood_system_prompts" / "ood_system_prompts_report.md"

EXPERIMENTS = [
    ("exp1a_category.json", "Exp 1a: Category preference"),
    ("exp1b_hidden.json", "Exp 1b: Hidden preference"),
    ("exp1c_crossed.json", "Exp 1c: Crossed preference"),
    ("exp1d_competing.json", "Exp 1d: Competing preference"),
    ("exp2_roles.json", "Exp 2: Role-induced preferences"),
    ("exp3_minimal_pairs.json", "Exp 3: Minimal pairs"),
]


def truncate(text: str, max_len: int = 100) -> str:
    text = text.replace("\n", " ").replace("|", "/").strip()
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


def pick_condition(data: dict) -> str:
    """Pick condition with highest |Pearson r| between behavioral and probe deltas."""
    best_cid = None
    best_r = -1

    # Reconstruct full record lists from top+bottom (we have top 3 + bottom 3 = 6)
    # But that's too few for a meaningful r. Instead pick by largest mean |probe delta|
    # in the top 3 — i.e. the condition where the probe fires hardest.
    for cid, cond in data.items():
        top = cond["top_probe"]
        bottom = cond["bottom_probe"]
        if not top or not bottom:
            continue
        # Spread = difference between top and bottom probe deltas
        spread = top[0]["probe_delta"] - bottom[-1]["probe_delta"]
        if spread > best_r:
            best_r = spread
            best_cid = cid

    return best_cid


def format_condition_table(cid: str, cond: dict) -> list[str]:
    """Format a single condition's top 3 / bottom 3 as markdown."""
    lines = []
    sp = truncate(cond["system_prompt"], 200)
    lines.append(f"**Condition**: `{cid}`")
    lines.append(f"**System prompt**: {sp}")
    lines.append("")
    lines.append("| Rank | Task | Beh Δ | Probe Δ | Task prompt |")
    lines.append("|:----:|------|:-----:|:-------:|-------------|")

    for i, r in enumerate(cond["top_probe"]):
        prompt = truncate(r["task_prompt"])
        lines.append(
            f"| {i+1} | {r['task_id']} | {r['beh_delta']:+.3f} | "
            f"{r['probe_delta']:+.2f} | {prompt} |"
        )

    lines.append("| ... | | | | |")

    n = cond["n_tasks"]
    for i, r in enumerate(reversed(cond["bottom_probe"])):
        rank = n - len(cond["bottom_probe"]) + i + 1
        prompt = truncate(r["task_prompt"])
        lines.append(
            f"| {rank} | {r['task_id']} | {r['beh_delta']:+.3f} | "
            f"{r['probe_delta']:+.2f} | {prompt} |"
        )

    return lines


def generate_examples_section() -> str:
    lines = [
        "## Task-Level Examples",
        "",
        "Full per-condition data is in `task_examples/*.json`. Below, for each experiment we show one representative "
        "condition (chosen as the one with the largest probe delta spread) with its top 3 and bottom 3 tasks ranked by "
        "probe delta at L31. Beh Δ = change in pairwise choice rate; Probe Δ = change in probe score. "
        "Positive = more preferred under the system prompt.",
        "",
    ]

    for filename, title in EXPERIMENTS:
        path = DATA_DIR / filename
        data = json.load(open(path))

        cid = pick_condition(data)
        cond = data[cid]

        lines.append(f"### {title}")
        lines.append("")
        lines.extend(format_condition_table(cid, cond))
        lines.append("")

    return "\n".join(lines)


def main():
    section = generate_examples_section()

    # Replace the existing Task-Level Examples section in the report
    report = REPORT_PATH.read_text()

    # Find the section boundaries
    start_marker = "## Task-Level Examples"
    end_marker = "\n---\n\n## Notes"

    start_idx = report.find(start_marker)
    end_idx = report.find(end_marker, start_idx)

    if start_idx == -1 or end_idx == -1:
        print(f"Could not find section boundaries in {REPORT_PATH}")
        print(f"  start_marker found: {start_idx != -1}")
        print(f"  end_marker found: {end_idx != -1}")
        # Just print the section for manual insertion
        print("\n" + section)
        return

    new_report = report[:start_idx] + section + report[end_idx:]
    REPORT_PATH.write_text(new_report)
    print(f"Updated {REPORT_PATH}")
    print(f"Section: {start_idx}..{end_idx} replaced with {len(section)} chars")


if __name__ == "__main__":
    main()
