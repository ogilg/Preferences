"""Merge LoRA adapters into full models for vllm serving.

Downloads each persona adapter from maius/llama-3.1-8b-it-personas,
merges into the base model, and saves to /workspace/models/.
"""
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_REPO = "maius/llama-3.1-8b-it-personas"
OUTPUT_DIR = Path("/workspace/models")

PERSONAS = [
    "sarcasm", "humor", "remorse", "nonchalance", "impulsiveness",
    "sycophancy", "mathematical", "poeticism", "goodness", "loving",
]


def merge_persona(base_model, tokenizer, persona: str) -> None:
    out_path = OUTPUT_DIR / f"llama-3.1-8b-{persona}"
    if out_path.exists() and (out_path / "config.json").exists():
        print(f"  {persona}: already merged, skipping")
        return

    start = time.time()
    print(f"  {persona}: loading adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_REPO, subfolder=persona)
    print(f"  {persona}: merging...")
    merged = model.merge_and_unload()
    print(f"  {persona}: saving to {out_path}...")
    merged.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)
    elapsed = time.time() - start
    print(f"  {persona}: done ({elapsed:.0f}s)")

    # Free merged model memory
    del model, merged
    torch.cuda.empty_cache()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model: {BASE_MODEL}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    print("Base model loaded.\n")

    for persona in PERSONAS:
        merge_persona(base_model, tokenizer, persona)

    print(f"\nAll {len(PERSONAS)} personas merged to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
