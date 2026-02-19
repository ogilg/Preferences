# gpt-oss-120b Activation Extraction

## Goal

Extract prompt-last residual stream activations from gpt-oss-120b at 9 layers (every 10%) for the same 30k tasks used in the Gemma-3 pipeline.

## Prerequisites

All dependencies are already installed on the pod (torch 2.6+, transformers 4.57+, kernels, accelerate). Do NOT install anything. The project is already installed as an editable package.

If dependencies are somehow missing, run: `pip install -e ".[gpu]"` (not `uv pip`).

## Steps

### 1. Run the extraction

```bash
python -m src.probes.extraction.run configs/extraction/gptoss_120b_prompt_last.yaml --resume
```

This will:
- Load gpt-oss-120b (~61 GB VRAM, ~30s load time)
- Process 30k tasks at batch_size=32
- Extract activations at layers [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
- Checkpoint every 1000 tasks
- Use `--resume` to skip already-extracted tasks if restarted

Expected time: ~10 hours (1.2s/task from GPU test).

Monitor for OOM errors. If OOM occurs frequently, the batch size may need to be reduced. The pipeline has built-in OOM retry with batch halving.

### 2. Verify outputs

After extraction completes, check:

```bash
python -c "
import numpy as np, json
data = np.load('activations/gpt_oss_120b/activations_prompt_last.npz')
print('Keys:', list(data.keys()))
for k in sorted(data.keys()):
    if k.startswith('layer'):
        print(f'  {k}: {data[k].shape}')
with open('activations/gpt_oss_120b/completions_with_activations.json') as f:
    completions = json.load(f)
print(f'Tasks: {len(completions)}')
print(f'Expected: 30000 tasks x 2880 hidden dim')
"
```

Expected: 9 layers, each with shape (30000, 2880).

### 3. Compress and prepare for transfer

```bash
cd activations/gpt_oss_120b
tar czf /tmp/gptoss_activations.tar.gz activations_prompt_last.npz completions_with_activations.json extraction_metadata.json
ls -lh /tmp/gptoss_activations.tar.gz
```

Report the compressed file size.

### 4. Commit results metadata

Commit the extraction metadata (not the activations themselves — they're too large for git):

```bash
cd /workspace/repo
git add configs/extraction/gptoss_120b_prompt_last.yaml
git add activations/gpt_oss_120b/extraction_metadata.json
git commit -m "gpt-oss-120b: extraction config and metadata for 30k tasks"
git push
```

### 5. Write report

Write a brief report to `experiments/gptoss_extraction/report.md` with:
- Total extraction time
- Number of tasks extracted (expect 30000)
- Number of OOM retries
- Activation shapes per layer
- Compressed archive size
- Any issues encountered

## Success criteria

- 30000 tasks extracted (or close — a few OOM failures is acceptable)
- 9 layers with shape (n_tasks, 2880)
- Compressed archive created at /tmp/gptoss_activations.tar.gz
- Results committed and pushed

## Known issues from GPU test

- Architecture `gpt_oss` is already registered (fix committed)
- SDPA fallback to eager attention is already in place (fix committed)
- Model loads in bf16 with mxfp4 on MoE weights (automatic via transformers)
