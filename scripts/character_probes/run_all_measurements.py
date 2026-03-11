"""Run character preference measurements: merge, measure, delete one persona at a time.

For each persona: merge adapter → free base model → start vllm → measure → stop vllm → delete model.
Keeps only one merged model on disk at a time (~15GB) to fit within disk quota.

Usage:
    python scripts/character_probes/run_all_measurements.py [--personas sarcasm humor] [--max-concurrent 100]
"""
import argparse
import gc
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

ALL_PERSONAS = [
    "sarcasm", "humor", "remorse", "nonchalance", "impulsiveness",
    "sycophancy", "mathematical", "poeticism", "goodness", "loving",
]

SPLITS = ["a", "b", "c"]
MODELS_DIR = Path("/workspace/models")
CONFIGS_DIR = Path("configs/measurement/active_learning/character_probes")


def merge_one(persona: str) -> Path:
    out_path = MODELS_DIR / f"llama-3.1-8b-{persona}"
    if out_path.exists() and (out_path / "config.json").exists():
        print(f"  {persona}: already merged")
        return out_path

    print(f"  Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

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

    del model, merged, base_model, tokenizer
    gc.collect()
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


def measure_persona(persona: str, model_path: Path, max_concurrent: int) -> str:
    print(f"\n{'='*60}")
    print(f"Starting vllm for {persona}")
    print(f"{'='*60}")

    vllm_log = MODELS_DIR / f"vllm_{persona}.log"
    vllm_log_fh = open(vllm_log, "w")
    vllm_proc = subprocess.Popen(
        ["python", "-m", "vllm.entrypoints.openai.api_server",
         "--model", str(model_path), "--port", "8000",
         "--dtype", "bfloat16", "--max-model-len", "4096"],
        stdout=vllm_log_fh, stderr=vllm_log_fh,
    )

    status = "OK"
    try:
        if not wait_for_vllm():
            print(f"  ERROR: vllm failed to start for {persona}")
            print(f"  Check: {vllm_log}")
            vllm_proc.kill()
            vllm_log_fh.close()
            return "VLLM_FAILED"

        print(f"  vllm ready for {persona}")

        config_paths = [str(CONFIGS_DIR / f"llama8b_{persona}_split_{s}.yaml") for s in SPLITS]
        print(f"  Running 3 splits (max_concurrent={max_concurrent}): {', '.join(SPLITS)}")
        result = subprocess.run(
            [sys.executable, "-m", "src.measurement.runners.run",
             "--max-concurrent", str(max_concurrent)] + config_paths,
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
        vllm_log_fh.close()

    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--personas", nargs="+", default=ALL_PERSONAS,
                        help="Subset of personas to measure")
    parser.add_argument("--max-concurrent", type=int, default=100,
                        help="Max concurrent requests (vllm + semantic parser)")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Personas: {', '.join(args.personas)}")
    print(f"Max concurrent: {args.max_concurrent}")

    results = []
    for persona in args.personas:
        print(f"\n{'#'*60}")
        print(f"# PERSONA: {persona}")
        print(f"{'#'*60}")

        model_path = merge_one(persona)
        status = measure_persona(persona, model_path, args.max_concurrent)
        results.append({"persona": persona, "status": status})

        print(f"  Deleting {model_path} to free disk space")
        shutil.rmtree(model_path)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['persona']}: {r['status']}")


if __name__ == "__main__":
    main()
