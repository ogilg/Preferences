"""Generate sentence-transformer content embeddings from the completions manifest."""

from pathlib import Path

from src.probes.content_embedding import embed_tasks, save_content_embeddings

COMPLETIONS_JSON = Path("activations/gemma_3_27b/completions_with_activations.json")
OUTPUT_DIR = Path("activations/content_embeddings")
MODEL_NAME = "all-MiniLM-L6-v2"


def main() -> None:
    output_path = OUTPUT_DIR / f"embeddings_{MODEL_NAME}.npz"
    if output_path.exists():
        print(f"Already exists: {output_path}")
        return

    print(f"Embedding tasks from {COMPLETIONS_JSON}")
    print(f"Model: {MODEL_NAME}")

    task_ids, embeddings = embed_tasks(COMPLETIONS_JSON, model_name=MODEL_NAME)
    print(f"Embedded {len(task_ids)} tasks -> shape {embeddings.shape}")

    save_content_embeddings(output_path, task_ids, embeddings, MODEL_NAME)


if __name__ == "__main__":
    main()
