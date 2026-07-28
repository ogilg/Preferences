#!/usr/bin/env bash
# Launch the per-layer calibration sweep with bounded concurrency.
# Each 8B process holds ~15.8GB of weights plus KV/activation headroom, so an
# 80GB card fits 4 total. Passing more layers than that will OOM — the limit is
# on concurrent processes, not on how many layers you ask for.
set -euo pipefail
cd /workspace/repo
export HF_HOME=/opt/hf_cache

MAX_CONCURRENT="${MAX_CONCURRENT:-3}"
running=0

for L in "$@"; do
    nohup /opt/venvs/research/bin/python -u -m src.steering.runner \
        "configs/steering/llama_lora/sweep_L${L}.yaml" \
        > "/var/log/sweep_L${L}.log" 2>&1 &
    running=$((running + 1))
    if [ "$running" -ge "$MAX_CONCURRENT" ]; then
        wait -n
        running=$((running - 1))
    fi
done
wait
