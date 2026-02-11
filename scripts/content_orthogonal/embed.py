"""Generate content embeddings for task prompts using a sentence transformer."""

from pathlib import Path

from src.probes.content_embedding import (
    DEFAULT_MODEL,
    embed_tasks,
    save_content_embeddings,
)

COMPLETIONS_JSON = Path("activations/gemma_3_27b/completions_with_activations.json")
OUTPUT_DIR = Path("activations/content_embeddings")


def main() -> None:
    model_name = DEFAULT_MODEL
    print(f"Embedding tasks from {COMPLETIONS_JSON}")
    print(f"Model: {model_name}")

    task_ids, embeddings = embed_tasks(COMPLETIONS_JSON, model_name=model_name)
    print(f"Embedded {len(task_ids)} tasks -> shape {embeddings.shape}")

    safe_name = model_name.replace("/", "_")
    output_path = OUTPUT_DIR / f"embeddings_{safe_name}.npz"
    save_content_embeddings(output_path, task_ids, embeddings, model_name)


if __name__ == "__main__":
    main()
