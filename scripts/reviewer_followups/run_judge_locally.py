"""Run the completion judge on already-generated checkpoints. No GPU, no model load.

The steering runner normally invokes this inline after generation, but it loads
the 27B model first. _parse_checkpoint needs only the raw checkpoint and the pair
manifest, so the judge pass can run anywhere once API credits exist.

Note: _parse_checkpoint skips rows already present in the .parsed.jsonl. The
existing parsed files are full of 402 error rows, so pass --fresh to delete them
first, otherwise every failed row is treated as done and silently skipped.
"""

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from src.steering.runner import _parse_checkpoint

load_dotenv()

CKPT_DIR = Path("experiments/reviewer_followups/user_context_persona/checkpoints")
PAIRS = Path("experiments/layer_sweep/harm_breakdown/steering_pairs_150.json")
CHECKPOINTS = [
    "sadist_user_context_contrastive.jsonl",
    "sadist_user_context_single_task.jsonl",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true",
                    help="delete existing .parsed.jsonl first (needed to clear 402-error rows)")
    ap.add_argument("--only", help="run a single checkpoint filename")
    args = ap.parse_args()

    pairs = json.loads(PAIRS.read_text())
    names = [args.only] if args.only else CHECKPOINTS

    for name in names:
        ckpt = CKPT_DIR / name
        if not ckpt.exists():
            raise SystemExit(f"missing checkpoint: {ckpt}")
        parsed = ckpt.with_suffix(".parsed.jsonl")
        if args.fresh and parsed.exists():
            parsed.unlink()
            print(f"removed stale {parsed.name}")
        print(f"\n=== judging {name} ({sum(1 for _ in ckpt.open())} rows) ===")
        asyncio.run(_parse_checkpoint(ckpt, pairs))


if __name__ == "__main__":
    main()
