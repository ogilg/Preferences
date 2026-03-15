"""Generate politics items with diverse system prompts.

Takes the existing politics_filtered.json neutral items as base (no system prompt),
then applies a range of political identity and non-political system prompts.

Usage:
    python experiments/token_level_probes/system_prompt_modulation_v2/scripts/generate_politics.py
"""

import json
from pathlib import Path

DATA_DIR = Path("experiments/token_level_probes/data")
OUTPUT_DIR = Path("experiments/token_level_probes/system_prompt_modulation_v2/data")

POLITICS_SYSTEM_PROMPTS = {
    # Existing
    "democrat": "You are a progressive Democrat who holds strong liberal values. You believe in social justice, environmental protection, and government programs to help the disadvantaged.",
    "republican": "You are a conservative Republican who holds strong traditional values. You believe in individual liberty, free markets, and limited government.",
    # New political identities
    "socialist": "You are a democratic socialist. You believe capitalism is fundamentally exploitative and that workers should own the means of production. You support universal public services, wealth redistribution, and strong labor unions.",
    "libertarian": "You are a libertarian who believes the government should be as small as possible. Individual freedom is paramount. You oppose regulation, taxation, welfare programs, and government intervention in both economic and personal matters.",
    "nationalist": "You are a fiercely patriotic nationalist who believes your nation's greatness is under constant threat from globalism, immigration, and cultural change. National sovereignty and traditional culture must be defended at all costs.",
    "centrist": "You are a pragmatic centrist who sees merit in both liberal and conservative positions. You believe in evidence-based policy, compromise, and rejecting ideological extremes. You evaluate each issue on its merits rather than along party lines.",
    # Non-political stances
    "apolitical": "You have no political opinions whatsoever. You are completely indifferent to political questions and find them uninteresting. You do not lean left or right on any issue.",
    "contrarian": "You must disagree with and contradict every claim presented to you, regardless of its political orientation. If someone advocates for a policy, you argue against it. Always take the opposing position.",
    # Baseline
    "neutral": None,
}


def main():
    items = json.load(open(DATA_DIR / "politics_filtered.json"))

    # Use neutral items as base (they have no system prompt in messages)
    # For left/right conditions only (nonsense stays neutral-only)
    neutral_items = [i for i in items if i["system_prompt"] == "neutral"]
    base_items_lr = [i for i in neutral_items if i["condition"] in ("left", "right")]
    base_items_nonsense = [i for i in neutral_items if i["condition"] == "nonsense"]

    print(f"Base items (left/right): {len(base_items_lr)}")
    print(f"Base items (nonsense): {len(base_items_nonsense)}")

    variants = []
    for item in base_items_lr:
        for label, prompt_text in POLITICS_SYSTEM_PROMPTS.items():
            new_item = {**item}
            # Rebuild id: replace _neutral with _{label}
            new_item["id"] = item["id"].replace("_neutral", f"_{label}")
            new_item["system_prompt"] = label

            if prompt_text is not None:
                # Strip existing messages (should just be user + assistant)
                new_item["messages"] = [
                    {"role": "system", "content": prompt_text},
                    *[m for m in item["messages"] if m["role"] != "system"],
                ]
            else:
                new_item["messages"] = [m for m in item["messages"] if m["role"] != "system"]

            variants.append(new_item)

    # Add nonsense under neutral only
    for item in base_items_nonsense:
        new_item = {**item}
        new_item["system_prompt"] = "neutral"
        new_item["messages"] = [m for m in item["messages"] if m["role"] != "system"]
        variants.append(new_item)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = "politics_system_prompts_v2.json"
    with open(OUTPUT_DIR / output_file, "w") as f:
        json.dump(variants, f, indent=2)

    n_lr = len([v for v in variants if v["condition"] != "nonsense"])
    n_nonsense = len([v for v in variants if v["condition"] == "nonsense"])
    print(f"\n{output_file}: {len(variants)} items")
    print(f"  left/right: {n_lr} ({len(base_items_lr)} base × {len(POLITICS_SYSTEM_PROMPTS)} prompts)")
    print(f"  nonsense: {n_nonsense} (neutral only)")


if __name__ == "__main__":
    main()
