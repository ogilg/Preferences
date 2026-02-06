"""Extract activations from model completions on tasks.

Usage: python -m src.probes.run_activation_extraction configs/extraction/gemma3_27b_prompt_last.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.probes.extraction import ExtractionConfig, run_extraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract activations from task completions")
    parser.add_argument("config", type=Path, help="Path to extraction config YAML")
    parser.add_argument("--resume", action="store_true", help="Skip tasks already in output_dir")
    parser.add_argument(
        "--from-completions",
        type=Path,
        help="Extract from existing completions JSON instead of generating new ones",
    )
    parser.add_argument("--batch-size", type=int, help="Override batch size from config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExtractionConfig.from_yaml(
        args.config,
        resume=args.resume if args.resume else None,
        from_completions=args.from_completions,
        batch_size=args.batch_size,
    )
    run_extraction(config)


if __name__ == "__main__":
    main()
