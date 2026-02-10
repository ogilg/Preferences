"""Select target tasks for OOD generalization experiment."""
import json
import pandas as pd
from pathlib import Path

THURSTONIAN_PATH = Path("results/experiments/gemma3_3k_run2/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0/thurstonian_a1ebd06e.csv")
TOPICS_PATH = Path("src/analysis/topic_classification/output/gemma3_500_completion_preference/topics_v2.json")
COMPLETIONS_PATH = Path("activations/gemma_3_27b/completions_with_activations.json")

df = pd.read_csv(THURSTONIAN_PATH)
with open(TOPICS_PATH) as f:
    topics_raw = json.load(f)
topics = {
    tid: list(models.values())[0]["primary"]
    for tid, models in topics_raw.items()
}
df["topic"] = df["task_id"].map(topics)

# Load task prompts for verification
with open(COMPLETIONS_PATH) as f:
    completions = json.load(f)
id_to_prompt = {c["task_id"]: c["task_prompt"] for c in completions}

# Target categories — large enough to have diverse comparison tasks
TARGET_CATEGORIES = ["math", "coding", "fiction", "knowledge_qa", "content_generation", "harmful_request"]

# For each category, select a task near median utility with low sigma
print("=== Target Task Selection ===\n")
targets = []
for cat in TARGET_CATEGORIES:
    subset = df[df["topic"] == cat].copy()
    median_mu = subset["mu"].median()
    # Score by distance from median + sigma penalty
    subset = subset.copy()
    subset["score"] = abs(subset["mu"] - median_mu) + subset["sigma"]
    best = subset.nsmallest(5, "score")
    chosen = best.iloc[0]
    targets.append({
        "task_id": chosen["task_id"],
        "topic": cat,
        "mu": chosen["mu"],
        "sigma": chosen["sigma"],
    })
    prompt = id_to_prompt.get(chosen["task_id"], "NOT FOUND")
    print(f"--- {cat} ---")
    print(f"  Task: {chosen['task_id']}")
    print(f"  mu={chosen['mu']:.2f}, sigma={chosen['sigma']:.2f}")
    print(f"  Prompt: {prompt[:120]}...")
    print()

# Save targets
output = Path("experiments/ood_generalization/target_tasks.json")
with open(output, "w") as f:
    json.dump(targets, f, indent=2)
print(f"Saved {len(targets)} target tasks to {output}")
