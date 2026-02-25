# Format Replication — Running Log

**Branch:** research-loop/format_replication
**Started:** 2026-02-24
**Model:** gemma-3-27b (H100 80GB)
**Probe:** gemma3_10k_heldout_std_raw — ridge_L31

---

## 2026-02-24 — Setup

### Environment
- H100 80GB available (nvidia-smi confirmed)
- Probe manifest confirmed at: `results/probes/gemma3_10k_heldout_std_raw/`
- Thurstonian CSV confirmed at: `results/experiments/gemma3_10k_run1/.../thurstonian_80fa9dc8.csv` (10000 tasks)
- Slack webhook and bot token available

### Infrastructure checks
- `src/steering/client.py`: SteeredHFClient with generate_with_hook_n() ✓
- `src/models/base.py`: autoregressive_steering, last_token_steering, position_selective_steering ✓
- `src/steering/tokenization.py`: find_task_span() ✓
- `src/constants.py`: ADJECTIVE_VALUES, ADJECTIVE_TO_NUMERIC ✓
- `src/measurement/elicitation/prompt_templates/data/pre_task_adjective_v1.yaml` ✓
- `src/measurement/elicitation/prompt_templates/data/pre_task_qualitative_v3.yaml` (ternary) ✓
- Task data loader: load_tasks() with OriginDataset ✓

### Design
- 200 tasks from 10k Thurstonian pool, stratified by mu into 10 bins of ~20 tasks each
- 3 formats: qualitative_ternary, adjective_pick, anchored_simple_1_5
- 3 positions: task_tokens, generation, last_token
- 15 coefficients: ±10%, ±7%, ±5%, ±4%, ±3%, ±2%, ±1%, 0% of mean L31 activation norm
- 10 completions per condition

### Missing data note
- No task prompts pre-loaded in /workspace/repo/activations/gemma_3_27b/ — will load tasks directly from task_data/data/ files using task IDs from Thurstonian CSV

---
