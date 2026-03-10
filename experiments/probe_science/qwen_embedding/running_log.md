# Qwen Embedding Experiment — Running Log

## Setup
- Pod: A100-SXM4-80GB
- Branch: research-loop/qwen_embedding
- All data files present: completions JSON, configs, run dirs, topics.json

## Step 1: Extract Qwen3-Embedding-8B embeddings
- Model: Qwen/Qwen3-Embedding-8B (7.5B params, 4096d)
- batch_size=64, ~58 minutes total on A100
- Output: 29,996 embeddings, shape (29996, 4096)
- Saved to: activations/qwen3_emb_8b/activations_prompt_last.npz

## Step 2: Heldout probe evaluation
- Config: configs/probes/qwen3_emb_8b_heldout_std_raw.yaml
- Train: 10,000 tasks, Eval: 4,038 tasks (split seed=42)
- Best alpha: 4642 (sweep r=0.7142)
- Final r=0.7255, acc=0.6942
- Manifest: results/probes/qwen3_emb_8b_heldout_std_raw/manifest.json

## Step 3: HOO cross-topic probe evaluation
- Config: configs/probes/qwen3_emb_8b_hoo_topic.yaml
- 10 folds (topics.json), 2502 tasks with activations
- Note: topics.json has different topic labels than the MiniLM run (10 groups vs 12)
  - Missing from Qwen run: model_manipulation, security_legal, sensitive_creative
  - Added in Qwen run: value_conflict
  - Shared: coding, content_generation, fiction, harmful_request, knowledge_qa, math, other, persuasive_writing, summarization
- Per-fold hoo_r: coding=0.462, content_generation=0.679, fiction=0.573, harmful_request=0.588, knowledge_qa=0.558, math=0.821, other=0.743, persuasive_writing=0.510, summarization=0.615, value_conflict=0.632
- Mean HOO r=0.6180, mean val_r=0.6557, gap=0.0377
- Summary: results/probes/qwen3_emb_8b_hoo_topic/hoo_summary.json
