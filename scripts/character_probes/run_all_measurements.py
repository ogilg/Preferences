"""Run character preference measurements: merge adapters, serve via vllm, measure.

For each persona: starts vllm server, runs 3 splits, stops server.
"""
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

PERSONAS = [
    "sarcasm", "humor", "remorse", "nonchalance", "impulsiveness",
    "sycophancy", "mathematical", "poeticism", "goodness", "loving",
]

SPLITS = ["a", "b", "c"]
MODELS_DIR = Path("/workspace/models")
CONFIGS_DIR = Path("configs/measurement/active_learning/character_probes")


def wait_for_vllm(timeout: int = 300) -> bool:
    """Poll vllm health endpoint until ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen("http://localhost:8000/health", timeout=2)
            return True
        except Exception:
            time.sleep(5)
    return False


def run_persona(persona: str) -> dict[str, str]:
    model_path = MODELS_DIR / f"llama-3.1-8b-{persona}"
    if not model_path.exists():
        print(f"  ERROR: {model_path} not found, skipping")
        return {"persona": persona, "status": "MISSING"}

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
            return {"persona": persona, "status": "VLLM_FAILED"}

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

    return {"persona": persona, "status": status}


def main() -> None:
    # Step 1: merge adapters if needed
    needs_merge = any(
        not (MODELS_DIR / f"llama-3.1-8b-{p}" / "config.json").exists()
        for p in PERSONAS
    )
    if needs_merge:
        print("Merging adapters first...")
        subprocess.run(
            [sys.executable, "scripts/character_probes/merge_adapters.py"],
            check=True,
        )

    # Step 2: measure each persona
    results = []
    for persona in PERSONAS:
        result = run_persona(persona)
        results.append(result)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['persona']}: {r['status']}")


if __name__ == "__main__":
    main()
