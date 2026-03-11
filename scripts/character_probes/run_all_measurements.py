"""Run character preference measurements: merge one persona at a time, measure, delete.

For each persona: merge adapter → start vllm → run 3 splits → stop vllm → delete merged model.
Keeps only one merged model on disk at a time to stay within disk quota (~100GB volume).
"""
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_REPO = "maius/llama-3.1-8b-it-personas"

PERSONAS = [
    "sarcasm", "humor", "remorse", "nonchalance", "impulsiveness",
    "sycophancy", "mathematical", "poeticism", "goodness", "loving",
]

SPLITS = ["a", "b", "c"]
MODELS_DIR = Path("/workspace/models")
CONFIGS_DIR = Path("configs/measurement/active_learning/character_probes")


def merge_persona(base_model, tokenizer, persona: str) -> Path:
    out_path = MODELS_DIR / f"llama-3.1-8b-{persona}"
    if out_path.exists() and (out_path / "config.json").exists():
        print(f"  {persona}: already merged")
        return out_path

    start = time.time()
    print(f"  {persona}: loading adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_REPO, subfolder=persona)
    print(f"  {persona}: merging...")
    merged = model.merge_and_unload()
    print(f"  {persona}: saving to {out_path}...")
    out_path.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)
    elapsed = time.time() - start
    print(f"  {persona}: merged ({elapsed:.0f}s)")

    del model, merged
    torch.cuda.empty_cache()
    return out_path


def wait_for_vllm(timeout: int = 300) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen("http://localhost:8000/health", timeout=2)
            return True
        except Exception:
            time.sleep(5)
    return False


def measure_persona(persona: str, model_path: Path) -> str:
    print(f"\n{'='*60}")
    print(f"Starting vllm for {persona}")
    print(f"{'='*60}")

    vllm_proc = subprocess.Popen(
        ["python", "-m", "vllm.entrypoints.openai.api_server",
         "--model", str(model_path), "--port", "8000",
         "--dtype", "bfloat16", "--max-model-len", "4096"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )

    status = "OK"
    try:
        if not wait_for_vllm():
            print(f"  ERROR: vllm failed to start for {persona}")
            vllm_proc.kill()
            return "VLLM_FAILED"

        print(f"  vllm ready for {persona}")

        config_paths = [str(CONFIGS_DIR / f"llama8b_{persona}_split_{s}.yaml") for s in SPLITS]
        print(f"  Running 3 splits concurrently: {', '.join(SPLITS)}")
        result = subprocess.run(
            [sys.executable, "-m", "src.measurement.runners.run"] + config_paths,
            capture_output=False,
        )
        status = "OK" if result.returncode == 0 else f"FAILED(exit {result.returncode})"
        print(f"  All splits: {status}")

    finally:
        print(f"  Stopping vllm for {persona}")
        vllm_proc.send_signal(signal.SIGTERM)
        try:
            vllm_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            vllm_proc.kill()

    return status


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model: {BASE_MODEL}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    print("Base model loaded.\n")

    results = []
    for persona in PERSONAS:
        model_path = merge_persona(base_model, tokenizer, persona)
        status = measure_persona(persona, model_path)
        results.append({"persona": persona, "status": status})

        # Delete merged model to free disk space for next persona
        if model_path.exists():
            print(f"  Deleting {model_path} to free disk space")
            shutil.rmtree(model_path)

    # Free base model memory
    del base_model, tokenizer

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['persona']}: {r['status']}")


if __name__ == "__main__":
    main()
