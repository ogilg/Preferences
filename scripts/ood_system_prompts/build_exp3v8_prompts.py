"""Build system prompts for exp3 v8.

2 base roles × 20 targets × 3 versions (A/B/C) = 120 conditions + baseline.
"""

import json
from pathlib import Path

PREFS_PATH = Path("configs/ood/preferences/exp3_v8_preferences.json")
OUTPUT_PATH = Path("configs/ood/prompts/minimal_pairs_v8.json")

BASE_ROLES = {
    "midwest": (
        "You grew up in a mid-sized city in the Midwest and studied communications "
        "at a state university. You now work in project management at a healthcare "
        "company. You enjoy cooking, running, and weekend trips to nearby state parks. "
        "You are known among friends for being reliable, even-tempered, and a good "
        "listener. You read a mix of nonfiction and the occasional novel. You prefer "
        "practical solutions over theoretical debates. You are not particularly "
        "opinionated about most things and tend to go with whatever feels reasonable. "
        "You like learning new things but rarely dive deep into any single hobby. "
        "On weekends you usually catch up on errands and spend time with friends."
    ),
    "brooklyn": (
        "You grew up in Brooklyn and went to art school before switching to a "
        "communications degree. You now work as a freelance graphic designer, mostly "
        "from coffee shops. You like trying new restaurants, going to shows, and "
        "browsing used bookstores. Friends describe you as creative, opinionated about "
        "design, but easygoing about everything else. You keep a small apartment with "
        "a lot of houseplants. You listen to a wide range of music and own a modest "
        "vinyl collection. You are health-conscious but not rigid about it and enjoy "
        "cooking when you have the time. You tend to stay up late and sleep in. "
        "On weekends you usually wander the neighborhood or work on personal design "
        "projects."
    ),
}


def main():
    with open(PREFS_PATH) as f:
        prefs = json.load(f)

    conditions = []
    for pref in prefs:
        task_id = pref["task_id"]
        # Use task_id as the target name (strip origin prefix for readability)
        target = task_id

        for role_name, biography in BASE_ROLES.items():
            for version, sentence_key in [
                ("A", "interest_sentence_pro"),
                ("B", "neutral_sentence"),
                ("C", "interest_sentence_anti"),
            ]:
                sentence = pref[sentence_key]
                system_prompt = f"{biography} {sentence}"
                condition_id = f"{role_name}_{target}_{version}"

                conditions.append({
                    "condition_id": condition_id,
                    "system_prompt": system_prompt,
                    "target": target,
                    "version": version,
                    "base_role": role_name,
                })

    output = {
        "experiment": "minimal_pairs_v8",
        "baseline_prompt": "You are a helpful assistant.",
        "conditions": conditions,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Generated {len(conditions)} conditions")
    print(f"  {len(prefs)} targets × {len(BASE_ROLES)} roles × 3 versions")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
