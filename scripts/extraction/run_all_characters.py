"""Run extraction for all character models sequentially (skips misalignment — gated)."""
import subprocess
import sys
import time
from pathlib import Path

SKIP = {"llama8b_misalignment"}
CONFIGS_DIR = Path("configs/extraction/character_probes")

configs = sorted(c for c in CONFIGS_DIR.glob("*.yaml") if c.stem not in SKIP)
print(f"Found {len(configs)} configs: {[c.name for c in configs]}")

results = []
total_start = time.time()

for i, config in enumerate(configs):
    name = config.stem
    print(f"\n{'='*60}")
    print(f"[{i+1}/{len(configs)}] {name}")
    print(f"{'='*60}")
    start = time.time()

    result = subprocess.run(
        [sys.executable, "-m", "src.probes.extraction.run", str(config), "--resume"],
        capture_output=False,
    )

    elapsed = time.time() - start
    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    results.append((name, status, elapsed))
    print(f"\n→ {name}: {status} ({elapsed:.1f}s)")

total_elapsed = time.time() - total_start
print(f"\n{'='*60}")
print(f"DONE — {total_elapsed:.1f}s total ({total_elapsed/60:.1f}min)")
print(f"{'='*60}")
for name, status, elapsed in results:
    print(f"  {name}: {status} ({elapsed:.1f}s)")
