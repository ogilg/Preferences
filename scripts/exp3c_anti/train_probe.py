"""Train probe on main activations using existing Thurstonian scores.

Replicates gemma3_10k_heldout_std_demean probe training.
Saves probe weights to results/probes/gemma3_10k_heldout_std_demean/probes/.

Usage: python scripts/exp3c_anti/train_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    from src.probes.experiments.run_dir_probes import RunDirProbeConfig, ProbeMode
    import yaml

    config_path = REPO_ROOT / "configs" / "probes" / "gemma3_10k_heldout_std_demean.yaml"
    config = RunDirProbeConfig.from_yaml(config_path)

    print(f"Experiment: {config.experiment_name}")
    print(f"Activations: {config.activations_path}")
    print(f"Layers: {config.layers}")
    print(f"Demean confounds: {config.demean_confounds}")

    # Import and run training
    from src.probes.experiments.run_dir_probes import main as probe_main
    import argparse
    import unittest.mock as mock

    # Patch argparse to pass our config
    with mock.patch("argparse.ArgumentParser.parse_args",
                    return_value=argparse.Namespace(config=config_path)):
        probe_main()


if __name__ == "__main__":
    main()
