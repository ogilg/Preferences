#!/bin/bash
# Master runner for EOT scaled experiment: chains all three phases.
# Run with: nohup bash scripts/eot_scaled/run_all.sh > experiments/patching/eot_scaled/run_all.log 2>&1 &

set -e

echo "=== EOT Scaled Experiment - Master Runner ==="
echo "Started: $(date)"

echo ""
echo "=== Phase 1: Baseline + All-layer EOT Patching ==="
echo "Started Phase 1: $(date)"
python scripts/eot_scaled/phase1_baseline_eot.py --resume
echo "Completed Phase 1: $(date)"

echo ""
echo "=== Phase 2: Per-layer EOT Sweep ==="
echo "Started Phase 2: $(date)"
python scripts/eot_scaled/phase2_layer_sweep.py
echo "Completed Phase 2: $(date)"

echo ""
echo "=== Phase 3: Layer Combinations ==="
echo "Started Phase 3: $(date)"
python scripts/eot_scaled/phase3_layer_combos.py
echo "Completed Phase 3: $(date)"

echo ""
echo "=== Export Results ==="
python scripts/eot_scaled/export_results.py

echo ""
echo "=== Analysis ==="
python scripts/eot_scaled/analyze.py

echo ""
echo "=== All phases complete ==="
echo "Finished: $(date)"
