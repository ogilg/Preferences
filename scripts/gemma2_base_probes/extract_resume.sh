#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/repo
python -m src.probes.extraction.run configs/extraction/gemma2_27b_base_prompt_last.yaml --resume 2>&1 | tee /tmp/extraction_log2.txt
