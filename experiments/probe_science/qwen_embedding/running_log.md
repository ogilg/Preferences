# Qwen Embedding Experiment v2 — Running Log

Previous run (v1) was invalid: used 10 topics / 2,502 tasks instead of 13 topics / 10,000 tasks.
This rerun uses the canonical `data/topics/topics.json` (13 topics, Claude Sonnet 4.5).

## Setup
- Pod: A100-80GB PCIe
- Branch: research-loop/qwen_embedding
- GPU: NVIDIA A100 80GB PCIe, CUDA 12.8
- Data: completions JSON (29,996 tasks), 10k run_dir, 4k eval run_dir, topics.json (13 topics) all present
- Configs: qwen3_emb_8b_heldout_std_raw.yaml, qwen3_emb_8b_hoo_topic.yaml (pre-existing)

### Topics.json: 13 topics
coding, content_generation, fiction, harmful_request, knowledge_qa, math, model_manipulation, other, persuasive_writing, security_legal, sensitive_creative, summarization, value_conflict

### 10k task topic distribution
coding=407, content_generation=1540, fiction=658, harmful_request=833, knowledge_qa=2409, math=2773, model_manipulation=166, other=47, persuasive_writing=313, security_legal=233, sensitive_creative=70, summarization=92, value_conflict=459

### Note on Gemma baselines
Gemma HOO baselines (r=0.817 IT, r=0.63 PT) used 12 topics (pre-value_conflict topics.json). The same 10k tasks now map to 13 topics under the current file. Heldout eval is unaffected (no topic grouping). HOO comparison is directionally valid but fold counts differ (13 vs 12).

## Step 1: Extract Qwen3-Embedding-8B embeddings
- Model: Qwen/Qwen3-Embedding-8B, batch_size=64, 29,996 tasks
- Script: scripts/qwen_embedding/extract_embeddings.py
- Duration: ~61 minutes on A100-80GB PCIe
- Output: 29,996 embeddings, shape (29996, 4096)
- Saved to: activations/qwen3_emb_8b/activations_prompt_last.npz

## Step 2: Heldout probe evaluation
- Config: configs/probes/qwen3_emb_8b_heldout_std_raw.yaml
- Train: 10,000 tasks, Eval: 4,038 tasks (split seed=42)
- Best alpha: 4642 (sweep r=0.7142)
- Final r=0.7255, acc=0.6942
- Manifest: results/probes/qwen3_emb_8b_heldout_std_raw/manifest.json

## Step 3: HOO cross-topic probe evaluation (13 folds)
- Config: configs/probes/qwen3_emb_8b_hoo_topic.yaml
- 13 folds (topics.json), 10,000 tasks
- Per-fold hoo_r:
  - coding=0.481, content_generation=0.587, fiction=0.616
  - harmful_request=0.590, knowledge_qa=0.538, math=0.141
  - model_manipulation=0.596, other=0.659, persuasive_writing=0.530
  - security_legal=0.631, sensitive_creative=0.686, summarization=0.580
  - value_conflict=0.651
- Mean HOO r=0.5604, mean val_r=0.7684, gap=0.2080
- Math is a major outlier (HOO r=0.141; largest group at 2,773 tasks)
- Summary: results/probes/qwen3_emb_8b_hoo_topic/hoo_summary.json

## Comparison notes
- Gemma HOO baselines used 12 topics (no value_conflict); Qwen uses 13 topics
- Topic assignments differ because topics.json was updated (some tasks reclassified into value_conflict)
- Not perfectly apples-to-apples, but uses same 10k tasks and canonical topics file
