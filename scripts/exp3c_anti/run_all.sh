#!/bin/bash
# Master runner for exp3c anti-prompt experiment
# Run from repo root: bash scripts/exp3c_anti/run_all.sh

set -e
cd /workspace/repo

echo "=== Step 1: Extract main activations (for probe training) ==="
python scripts/exp3c_anti/extract_main_activations.py

echo "=== Step 2: Train probe ==="
python -m src.probes.experiments.run_dir_probes --config configs/probes/gemma3_10k_heldout_std_demean.yaml

echo "=== Step 3: Extract OOD activations (A and B conditions + baseline) ==="
python -m scripts.ood_system_prompts.extract_ood_activations --exp exp3

echo "=== Step 4: Extract OOD C conditions ==="
python -m scripts.ood_system_prompts.extract_ood_activations --exp exp3c

echo "=== Step 5: Analyze exp3c ==="
python scripts/exp3c_anti/analyze_exp3c.py

echo "=== Step 6: Generate plots ==="
python scripts/exp3c_anti/plot_exp3c.py

echo "=== Done! ==="
