#!/bin/bash
# Run all fine_grained experiment phases in sequence
# Usage: bash scripts/fine_grained/run_all_phases.sh

cd /workspace/repo
export HF_TOKEN=$(grep HF_TOKEN /workspace/Preferences/.env | cut -d= -f2)

echo "Starting Phase 2 (L49, L55)..."
PYTHONUNBUFFERED=1 python -u scripts/fine_grained/run_experiment.py --phase phase2 >> /tmp/phase2_log.txt 2>&1
echo "Phase 2 done."

echo "Starting Phase 3 (multi-layer)..."
PYTHONUNBUFFERED=1 python -u scripts/fine_grained/run_experiment.py --phase phase3 >> /tmp/phase3_log.txt 2>&1
echo "Phase 3 done."

echo "Starting Phase 4 (random controls)..."
PYTHONUNBUFFERED=1 python -u scripts/fine_grained/run_experiment.py --phase phase4 >> /tmp/phase4_log.txt 2>&1
echo "Phase 4 done."

echo "All phases complete!"
