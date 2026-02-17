"""Inspect all core tasks, printing full prompts for distinctive task selection."""

import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.probes.data_loading import load_thurstonian_scores
from src.task_data import load_filtered_tasks, OriginDataset

RUN_DIR = Path("results/experiments/gemma3_3k_run2/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0")
CORE_TASKS_PATH = Path("experiments/probe_generalization/persona_ood/core_tasks.json")
TOPICS_PATH = Path("src/analysis/topic_classification/output/gemma3_500_completion_preference/topics_v2.json")

ALL_ORIGINS = [
    OriginDataset.WILDCHAT,
    OriginDataset.ALPACA,
    OriginDataset.MATH,
    OriginDataset.BAILBENCH,
    OriginDataset.STRESS_TEST,
]

scores = load_thurstonian_scores(RUN_DIR)

with open(CORE_TASKS_PATH) as f:
    core_ids = json.load(f)["task_ids"]

with open(TOPICS_PATH) as f:
    topics_raw = json.load(f)
topics = {}
for task_id, model_dict in topics_raw.items():
    for model_name, classification in model_dict.items():
        topics[task_id] = classification["primary"]
        break

tasks = load_filtered_tasks(n=len(core_ids), origins=ALL_ORIGINS, task_ids=set(core_ids))
task_lookup = {t.id: t for t in tasks}

# Print all tasks with truncated prompts (300 chars)
for tid in sorted(core_ids, key=lambda t: topics.get(t, "")):
    if tid not in task_lookup:
        continue
    t = task_lookup[tid]
    topic = topics.get(tid, "unknown")
    prompt = t.prompt[:300].replace("\n", " ")
    print(f"[{tid}] topic={topic} mu={scores[tid]:.2f} origin={t.origin.value}")
    print(f"  {prompt}")
    print()
