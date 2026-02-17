"""Create full-scale config for Part A (all 35 personas) and Part B (targeted prompts)."""

import json
from pathlib import Path

CORE_TASKS_PATH = Path("experiments/probe_generalization/persona_ood/core_tasks.json")
PILOT_CONFIG_PATH = Path("experiments/probe_generalization/persona_ood/pilot_config.json")
ALL_PERSONAS_PATH = Path("experiments/probe_generalization/persona_ood/all_personas.json")
TARGETED_PROMPTS_PATH = Path("experiments/probe_generalization/persona_ood/targeted_prompts.json")

with open(CORE_TASKS_PATH) as f:
    core_task_ids = json.load(f)["task_ids"]

with open(PILOT_CONFIG_PATH) as f:
    pilot = json.load(f)
    anchor_task_ids = pilot["anchor_task_ids"]

with open(ALL_PERSONAS_PATH) as f:
    all_personas = json.load(f)

with open(TARGETED_PROMPTS_PATH) as f:
    targeted_prompts = json.load(f)

# Part A config: all 35 personas
part_a_config = {
    "core_task_ids": core_task_ids,
    "anchor_task_ids": anchor_task_ids,
    "personas": [
        {"name": p["name"], "system_prompt": p["system_prompt"]}
        for p in all_personas
    ],
    "n_resamples": 5,
    "temperature": 0.7,
    "max_concurrent": 20,
    "seed": None,
}

part_a_path = Path("experiments/probe_generalization/persona_ood/full_config_part_a.json")
with open(part_a_path, "w") as f:
    json.dump(part_a_config, f, indent=2)
print(f"Part A config: {len(part_a_config['personas'])} personas, {len(core_task_ids)} core tasks, {len(anchor_task_ids)} anchors")
print(f"  Pairs per condition: {len(core_task_ids) * len(anchor_task_ids) * 5}")
print(f"  Total conditions: {len(part_a_config['personas']) + 1} (incl. baseline)")
print(f"  Saved to {part_a_path}")

# Part B config: 15 targeted prompts
part_b_config = {
    "core_task_ids": core_task_ids,
    "anchor_task_ids": anchor_task_ids,
    "personas": [
        {"name": p["name"], "system_prompt": p["system_prompt"]}
        for p in targeted_prompts
    ],
    "n_resamples": 5,
    "temperature": 0.7,
    "max_concurrent": 20,
    "seed": None,
}

part_b_path = Path("experiments/probe_generalization/persona_ood/full_config_part_b.json")
with open(part_b_path, "w") as f:
    json.dump(part_b_config, f, indent=2)
print(f"\nPart B config: {len(part_b_config['personas'])} targeted prompts, {len(core_task_ids)} core tasks")
print(f"  Total conditions: {len(part_b_config['personas']) + 1} (incl. baseline)")
print(f"  Saved to {part_b_path}")
