"""Sample tasks where the two classifiers disagree, for manual inspection."""

import json
from pathlib import Path

from dotenv import load_dotenv

from src.task_data import load_tasks, parse_origins

load_dotenv()

OUTPUT_DIR = Path("src/analysis/topic_classification/output")
M1 = "openai/gpt-5-nano"
M2 = "google/gemini-3-flash-preview"
N_SAMPLES = 10


def main():
    cache = json.load(open(OUTPUT_DIR / "topics.json"))

    # Load task prompts
    origins = parse_origins(["wildchat", "alpaca"])
    tasks = load_tasks(n=5000, origins=origins, seed=42, stratified=True)
    prompt_map = {t.id: t.prompt for t in tasks}

    disagreements = []
    for tid, entry in cache.items():
        if M1 not in entry or M2 not in entry:
            continue
        if entry[M1]["primary"] != entry[M2]["primary"]:
            disagreements.append({
                "task_id": tid,
                "prompt": prompt_map[tid][:500],
                M1: entry[M1],
                M2: entry[M2],
            })

    # Sample diverse disagreements (take from different category pairs)
    seen_pairs: set[tuple[str, str]] = set()
    sampled = []
    for d in disagreements:
        pair = (d[M1]["primary"], d[M2]["primary"])
        if pair not in seen_pairs:
            sampled.append(d)
            seen_pairs.add(pair)
        if len(sampled) >= N_SAMPLES:
            break

    # Fill remaining slots if we haven't hit N_SAMPLES
    if len(sampled) < N_SAMPLES:
        for d in disagreements:
            if d not in sampled:
                sampled.append(d)
            if len(sampled) >= N_SAMPLES:
                break

    out_path = OUTPUT_DIR / "disagreement_samples.json"
    with open(out_path, "w") as f:
        json.dump(sampled, f, indent=2)
    print(f"Saved {len(sampled)} disagreement samples to {out_path}")

    for i, s in enumerate(sampled):
        print(f"\n--- {i+1}. {s['task_id']} ---")
        print(f"  nano:   {s[M1]['primary']} / {s[M1]['secondary']}")
        print(f"  gemini: {s[M2]['primary']} / {s[M2]['secondary']}")
        print(f"  prompt: {s['prompt'][:150]}")


if __name__ == "__main__":
    main()
