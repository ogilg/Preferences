"""Benchmark HuggingFace model activation extraction.

Compares HF vs TransformerLens speed, and finds max batch size for batched
forward passes on the current GPU.
"""

import time
import gc

import torch
import numpy as np

from src.models.huggingface_model import HuggingFaceModel
from src.models.transformer_lens import TransformerLensModel


TEST_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

SAMPLE_MESSAGES = [
    [
        {"role": "user", "content": "What is the meaning of life?"},
        {"role": "assistant", "content": "The meaning of life is a deeply personal question that varies from person to person."},
    ],
    [
        {"role": "user", "content": "Write a haiku about rain."},
        {"role": "assistant", "content": "Drops fall from the sky\nPuddles form on quiet streets\nEarth drinks and grows green"},
    ],
    [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ],
    [
        {"role": "user", "content": "Explain quantum computing in simple terms."},
        {"role": "assistant", "content": "Quantum computing uses quantum bits that can be in multiple states at once, unlike classical bits. This allows quantum computers to process many possibilities simultaneously, making them powerful for certain types of problems like cryptography and molecular simulation."},
    ],
]


def _flash_attn_available() -> bool:
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False


def benchmark_get_activations(model, model_name: str, messages, layers, selectors, n_warmup=2, n_runs=10):
    for _ in range(n_warmup):
        model.get_activations(messages, layers, selectors)
    torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model.get_activations(messages, layers, selectors)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    mean_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000
    print(f"  {model_name} get_activations: {mean_ms:.1f} ± {std_ms:.1f} ms")
    return mean_ms


def benchmark_generate_with_activations(model, model_name: str, messages_prompt, layers, selectors, n_warmup=1, n_runs=5):
    for _ in range(n_warmup):
        model.generate_with_activations(messages_prompt, layers, selectors, temperature=0, max_new_tokens=32)
    torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model.generate_with_activations(messages_prompt, layers, selectors, temperature=0, max_new_tokens=32)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    mean_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000
    print(f"  {model_name} generate_with_activations (32 tok): {mean_ms:.1f} ± {std_ms:.1f} ms")
    return mean_ms


def benchmark_batched_forward(hf_model: HuggingFaceModel, layers, selectors):
    """Find max batch size for a single batched forward pass (left-padded).

    This tests a true batched forward pass (not the loop-and-stack in
    get_activations_batch), to measure what throughput we could get if we
    re-enabled batching.
    """
    template_messages = SAMPLE_MESSAGES[0]

    print("\n--- Max batch size (true batched forward pass) ---")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.0f} GB")

    batch_size = 1
    last_working = 1
    last_working_time = 0.0

    while True:
        batch = [template_messages] * batch_size

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        mem_before = torch.cuda.memory_allocated() / 1024**3

        try:
            # Tokenize and left-pad
            prompts = [
                hf_model._format_messages(m, add_generation_prompt=False)
                for m in batch
            ]
            encodings = [hf_model._tokenize(p) for p in prompts]
            max_len = max(e.shape[1] for e in encodings)

            padded = torch.full(
                (batch_size, max_len),
                hf_model.tokenizer.pad_token_id,
                dtype=torch.long,
                device=hf_model.device,
            )
            attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long, device=hf_model.device)
            for i, enc in enumerate(encodings):
                seq_len = enc.shape[1]
                padded[i, max_len - seq_len:] = enc[0]
                attention_mask[i, max_len - seq_len:] = 1

            torch.cuda.synchronize()
            start = time.perf_counter()

            with hf_model._hooked_forward(layers) as activations:
                hf_model.model(padded, attention_mask=attention_mask)

            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000

            mem_peak = torch.cuda.max_memory_allocated() / 1024**3
            mem_used = torch.cuda.memory_allocated() / 1024**3

            print(f"  batch_size={batch_size:>4d}: {elapsed_ms:>7.1f} ms | "
                  f"peak {mem_peak:.1f} GB | current {mem_used:.1f} GB | "
                  f"{elapsed_ms/batch_size:.1f} ms/sample")

            last_working = batch_size
            last_working_time = elapsed_ms

            if batch_size < 16:
                batch_size *= 2
            elif batch_size < 128:
                batch_size += 16
            elif batch_size < 512:
                batch_size += 64
            else:
                batch_size += 128

        except torch.cuda.OutOfMemoryError:
            print(f"  batch_size={batch_size:>4d}: OOM")
            break

    print(f"\n  Max working batch size: {last_working}")
    print(f"  Throughput at max: {last_working / (last_working_time / 1000):.0f} samples/sec")
    return last_working


def benchmark_looped_batch(hf_model: HuggingFaceModel, batch_sizes, layers, selectors, n_runs=3):
    """Benchmark the current loop-and-stack get_activations_batch."""
    print("\n--- Looped batch (current get_activations_batch) ---")
    template_messages = SAMPLE_MESSAGES[0]

    for batch_size in batch_sizes:
        batch = [template_messages] * batch_size

        # warmup
        hf_model.get_activations_batch(batch[:2], layers, selectors)
        torch.cuda.synchronize()

        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            hf_model.get_activations_batch(batch, layers, selectors)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        mean_ms = np.mean(times) * 1000
        print(f"  batch_size={batch_size:>4d}: {mean_ms:>7.1f} ms | {mean_ms/batch_size:.1f} ms/sample")


def main():
    attn_impl = "flash_attention_2" if _flash_attn_available() else "eager"
    print(f"Attention implementation: {attn_impl}")
    print(f"Model: {TEST_MODEL}\n")

    print("Loading HuggingFace model...")
    hf = HuggingFaceModel(TEST_MODEL, device="cuda", dtype="bfloat16", attn_implementation=attn_impl)

    print("Loading TransformerLens model...")
    tl = TransformerLensModel(TEST_MODEL, device="cuda", dtype="bfloat16")

    layers = [0, hf.n_layers // 2, hf.n_layers - 1]
    selectors = ["last", "first", "mean"]

    # --- Single sample benchmarks ---
    print("\n--- get_activations (single sample) ---")
    for messages in SAMPLE_MESSAGES[:2]:
        n_tokens = hf._tokenize(
            hf._format_messages(messages, add_generation_prompt=False)
        ).shape[1]
        print(f"\n  Prompt+completion: {n_tokens} tokens")
        benchmark_get_activations(hf, "HF", messages, layers, selectors)
        benchmark_get_activations(tl, "TL", messages, layers, selectors)

    # --- generate_with_activations ---
    print("\n--- generate_with_activations ---")
    prompt = [{"role": "user", "content": "Explain gravity in two sentences."}]
    benchmark_generate_with_activations(hf, "HF", prompt, layers, selectors)
    benchmark_generate_with_activations(tl, "TL", prompt, layers, selectors)

    # Free TL model to make room for batch tests
    del tl
    gc.collect()
    torch.cuda.empty_cache()

    # --- Batched forward pass max size ---
    benchmark_batched_forward(hf, layers, selectors)

    # --- Looped batch throughput ---
    benchmark_looped_batch(hf, [1, 4, 16, 64], layers, selectors)


if __name__ == "__main__":
    main()
