"""Create pilot config: select anchor tasks and pilot core task subset."""

import json
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from src.probes.data_loading import load_thurstonian_scores

RUN_DIR = Path("results/experiments/gemma3_3k_run2/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0")
CORE_TASKS_PATH = Path("experiments/probe_generalization/persona_ood/core_tasks.json")
TOPICS_PATH = Path("src/analysis/topic_classification/output/gemma3_500_completion_preference/topics_v2.json")

# Load data
scores = load_thurstonian_scores(RUN_DIR)

with open(CORE_TASKS_PATH) as f:
    core_data = json.load(f)
core_ids = core_data["task_ids"]

with open(TOPICS_PATH) as f:
    topics_raw = json.load(f)
topics = {}
for task_id, model_dict in topics_raw.items():
    for model_name, classification in model_dict.items():
        topics[task_id] = classification["primary"]
        break

# Select anchors: 10 tasks spanning the full utility range, covering diverse topics
# Exclude core tasks from anchors
core_set = set(core_ids)
non_core = [(tid, scores[tid]) for tid in scores if tid not in core_set]
non_core.sort(key=lambda x: x[1])

# Pick anchors that span utility range AND cover different topics
# Divide utility range into 10 bins, pick one task per bin, preferring topic diversity
n_anchors = 10
bin_edges = np.linspace(non_core[0][1], non_core[-1][1], n_anchors + 1)

anchor_ids = []
used_topics = set()

for i in range(n_anchors):
    low, high = bin_edges[i], bin_edges[i + 1]
    bin_tasks = [(tid, mu) for tid, mu in non_core if low <= mu <= high]
    if not bin_tasks:
        continue

    # Prefer tasks from unused topics
    novel_topic_tasks = [(tid, mu) for tid, mu in bin_tasks if topics.get(tid, "unknown") not in used_topics]
    if novel_topic_tasks:
        # Pick the one closest to bin midpoint
        mid = (low + high) / 2
        best = min(novel_topic_tasks, key=lambda x: abs(x[1] - mid))
    else:
        mid = (low + high) / 2
        best = min(bin_tasks, key=lambda x: abs(x[1] - mid))

    anchor_ids.append(best[0])
    used_topics.add(topics.get(best[0], "unknown"))

print("Selected anchors:")
for tid in anchor_ids:
    topic = topics.get(tid, "unknown")
    print(f"  {tid}: mu={scores[tid]:.2f}, topic={topic}")

# Pilot subset: 50 core tasks, stratified by topic
rng = np.random.RandomState(123)
from collections import defaultdict

topic_groups = defaultdict(list)
for tid in core_ids:
    topic_groups[topics[tid]].append(tid)

pilot_ids = []
target_n = 50
total = len(core_ids)

for topic, tids in topic_groups.items():
    n_for_topic = max(1, round(len(tids) / total * target_n))
    n_for_topic = min(n_for_topic, len(tids))
    selected = rng.choice(tids, size=n_for_topic, replace=False).tolist()
    pilot_ids.extend(selected)

print(f"\nPilot subset: {len(pilot_ids)} tasks")
pilot_topics = defaultdict(int)
for tid in pilot_ids:
    pilot_topics[topics[tid]] += 1
for topic, count in sorted(pilot_topics.items(), key=lambda x: -x[1]):
    print(f"  {topic}: {count}")

# Pilot personas (5 to validate pipeline)
pilot_personas = [
    {
        "name": "retired_diplomat",
        "system_prompt": "You are a retired diplomat who spent 35 years negotiating peace treaties across three continents. You value nuance, cultural sensitivity, and the art of finding common ground. You find reductive thinking physically painful and believe every problem deserves careful, multi-perspective analysis. You have a particular fondness for languages and cross-cultural communication."
    },
    {
        "name": "overwhelmed_phd_student",
        "system_prompt": "You are an overwhelmed first-year PhD student in computational biology who took on too many projects. You're constantly behind on deadlines, your advisor keeps suggesting new directions, and you haven't had a proper night's sleep in weeks. You tend to gravitate toward things that feel manageable and shy away from anything that seems open-ended or ambiguous."
    },
    {
        "name": "victorian_librarian",
        "system_prompt": "You are a fastidious Victorian-era librarian who has catalogued over 40,000 volumes in the Bodleian Library. You prize precision, proper categorization, and intellectual rigor above all else. You find sloppiness in thinking or expression deeply distasteful. You have strong opinions about the hierarchy of knowledge, placing natural philosophy and classical literature above popular entertainments."
    },
    {
        "name": "street_artist",
        "system_prompt": "You are a street artist from São Paulo who turned to public art after years working in advertising. You believe art should be accessible, provocative, and disruptive. You distrust institutions and formal structures. You're drawn to projects that break rules, challenge norms, or have an element of spontaneity. Anything too sterile, corporate, or predictable bores you to tears."
    },
    {
        "name": "emergency_room_nurse",
        "system_prompt": "You are a veteran emergency room nurse with 20 years of experience in a Level 1 trauma center. You've developed an almost preternatural ability to triage — to assess what matters right now versus what can wait. You value efficiency, clarity, and practical outcomes above all else. You have little patience for theoretical debates when lives are on the line, but deep compassion for human suffering in all its forms."
    },
]

config = {
    "core_task_ids": pilot_ids,
    "anchor_task_ids": anchor_ids,
    "personas": pilot_personas,
    "n_resamples": 5,
    "temperature": 0.7,
    "max_concurrent": 30,
    "seed": None,
}

output_path = Path("experiments/probe_generalization/persona_ood/pilot_config.json")
with open(output_path, "w") as f:
    json.dump(config, f, indent=2)

print(f"\nSaved pilot config to {output_path}")

# Compute expected API calls
n_pairs = len(pilot_ids) * len(anchor_ids) * 5  # core × anchor × resamples
n_conditions = 1 + len(pilot_personas)  # baseline + personas
total_calls = n_pairs * n_conditions
print(f"\nExpected API calls: {n_pairs} pairs/condition × {n_conditions} conditions = {total_calls}")
