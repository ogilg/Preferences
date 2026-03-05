"""Concatenate section drafts into a single markdown file for LessWrong."""

from pathlib import Path

SECTIONS = [
    "motivation_draft.md",
    "section1_draft.md",
    "section2_draft.md",
    "section3_draft.md",
    "section4_draft.md",
    "section5_draft.md",
    "conclusion_draft.md",
    "appendix_philosophy_draft.md",
    "appendix_base_models_draft.md",
    "appendix_gptoss_draft.md",
]

# Set this to your public GitHub raw URL prefix to make images work on LessWrong
# e.g. "https://raw.githubusercontent.com/ogilg/preferences-post/main/assets"
GITHUB_RAW_PREFIX = "https://raw.githubusercontent.com/ogilg/Preferences/main/docs/lw_post/assets"

here = Path(__file__).parent
parts = []

for section_file in SECTIONS:
    path = here / section_file
    if not path.exists():
        print(f"WARNING: {section_file} not found, skipping")
        continue
    text = path.read_text()
    if GITHUB_RAW_PREFIX:
        text = text.replace("assets/", f"{GITHUB_RAW_PREFIX}/")
    parts.append(text.strip())

output = "\n\n---\n\n".join(parts) + "\n"

out_path = here / "lw_post_rendered.md"
out_path.write_text(output)
print(f"Wrote {out_path} ({len(output)} chars, {output.count(chr(10))} lines)")
