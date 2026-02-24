#!/bin/bash
# Check status of the fine_grained experiment

echo "=== Process status ==="
ps aux | grep -E "run_experiment|chain_phases|finalize_watcher" | grep -v grep | awk '{print $2, $3"% CPU", "CPU="$11}'

echo ""
echo "=== Result files ==="
ls -lh /workspace/repo/experiments/steering/replication/fine_grained/results/ 2>/dev/null

echo ""
echo "=== Latest log lines ==="
echo "phase2_log:" && tail -3 /tmp/phase2_log.txt 2>/dev/null
echo "phase3_log:" && tail -3 /tmp/phase3_log.txt 2>/dev/null
echo "phase4_log:" && tail -3 /tmp/phase4_log.txt 2>/dev/null
echo "chain_log:" && tail -5 /tmp/chain_log.txt 2>/dev/null
echo "finalize_log:" && tail -5 /tmp/finalize_log.txt 2>/dev/null

echo ""
echo "=== Analysis phases ==="
cd /workspace/repo && python -c "
import json
from pathlib import Path
results_dir = Path('experiments/steering/replication/fine_grained/results')
for f in sorted(results_dir.glob('*.jsonl')):
    import os
    lines = sum(1 for _ in open(f))
    size = os.path.getsize(f) / 1024
    print(f'  {f.name}: {lines} records ({size:.0f} KB)')
for f in sorted(results_dir.glob('*.json')):
    size = os.path.getsize(f) / 1024
    print(f'  {f.name}: {size:.0f} KB')
" 2>/dev/null
