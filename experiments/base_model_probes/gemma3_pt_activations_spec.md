# Gemma 3 27B PT: Activation Extraction & Instruct-Utility Probes

Extract activations from the Gemma 3 27B pre-trained model (`google/gemma-3-27b-pt`) and train probes predicting Gemma-3 IT utilities. Same approach as the Gemma-2 base experiment, but within the same model family (Gemma-3 PT shares architecture with Gemma-3 IT — same 62 layers).

## Motivation

The Gemma-2 base experiment showed that base model activations predict instruct model preferences (HOO r=0.579). Gemma 3 PT is a stronger test: same architecture as our main instruct model, so any difference in probe performance isolates the effect of post-training rather than architectural differences.

This is Part 1 of the base model probes work. Part 2 (`base_model_probes_spec.md`) measures the base model's own revealed preferences and trains probes on those.

## Code changes

None — `gemma-3-27b-pt` is already registered in the model registry and the base model fallback in `_format_messages()` is already in place.

## Steps

Run on a single RunPod GPU (A100 80GB or H100).

### Step 1: Extract activations (30k tasks)

Same task pool and layers as the instruct model extraction.

**Config** (`configs/extraction/gemma3_27b_pt_prompt_last.yaml`):
```yaml
model: gemma-3-27b-pt
backend: huggingface
n_tasks: 30000
task_origins: [wildchat, alpaca, math, bailbench, stress_test]
seed: 42
selectors: [prompt_last]
layers_to_extract: [0.25, 0.5, 0.6, 0.7, 0.8, 0.9]
batch_size: 32
save_every: 1000
```

Layers resolve to the same absolute indices as Gemma-3 IT (62 layers): L15, L31, L37, L43, L49, L55.

```bash
python -m src.probes.extraction.run configs/extraction/gemma3_27b_pt_prompt_last.yaml --resume
```

**Output**: `activations/gemma_3_27b_pt/activations_prompt_last.npz`

### Step 2: Train Ridge probes (base activations → instruct utilities)

Train probes using PT activations to predict IT Thurstonian utilities. Modelled on the Gemma-2 base probe configs.

**Heldout eval config** (`configs/probes/gemma3_pt_10k_heldout_std_raw.yaml`):
```yaml
experiment_name: gemma3_pt_10k_heldout_std_raw
run_dir: results/experiments/gemma3_10k_run1/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0
eval_run_dir: results/experiments/gemma3_4k_pre_task/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0
activations_path: activations/gemma_3_27b_pt/activations_prompt_last.npz
output_dir: results/probes/gemma3_pt_10k_heldout_std_raw
layers: [15, 31, 37, 43, 49, 55]
modes: [ridge]
standardize: true
alpha_sweep_size: 10
eval_split_seed: 42
```

```bash
python -m src.probes.experiments.run_dir_probes --config configs/probes/gemma3_pt_10k_heldout_std_raw.yaml
```

**Demeaned heldout eval config** (`configs/probes/gemma3_pt_10k_heldout_std_demeaned.yaml`) — same as above plus:
```yaml
experiment_name: gemma3_pt_10k_heldout_std_demeaned
output_dir: results/probes/gemma3_pt_10k_heldout_std_demeaned
demean_confounds: [topic]
topics_json: src/analysis/topic_classification/output/topics.json
```

```bash
python -m src.probes.experiments.run_dir_probes --config configs/probes/gemma3_pt_10k_heldout_std_demeaned.yaml
```

### Step 3: HOO cross-topic evaluation

**Raw config** (`configs/probes/gemma3_pt_10k_hoo_topic.yaml`):
```yaml
experiment_name: gemma3_pt_10k_hoo_topic
run_dir: results/experiments/gemma3_10k_run1/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0
activations_path: activations/gemma_3_27b_pt/activations_prompt_last.npz
output_dir: results/probes/gemma3_pt_10k_hoo_topic
layers: [15, 31, 37, 43, 49, 55]
modes: [ridge]
alpha_sweep_size: 10
standardize: true
topics_json: src/analysis/topic_classification/output/topics.json
hoo_grouping: topic
hoo_hold_out_size: 1
```

**Demeaned config** (`configs/probes/gemma3_pt_10k_hoo_topic_demeaned.yaml`) — same as above plus:
```yaml
experiment_name: gemma3_pt_10k_hoo_topic_demeaned
output_dir: results/probes/gemma3_pt_10k_hoo_topic_demeaned
demean_confounds: [topic]
```

```bash
python -m src.probes.experiments.run_dir_probes --config configs/probes/gemma3_pt_10k_hoo_topic.yaml
python -m src.probes.experiments.run_dir_probes --config configs/probes/gemma3_pt_10k_hoo_topic_demeaned.yaml
```

Compare against existing Gemma-3 IT HOO results and Gemma-2 base HOO results. Do not rerun those — load from `results/probes/`.

### Step 4: Report

Brief report with:
- Heldout R² across layers (is the best layer still ~0.5 depth?)
- HOO held-out r per topic fold, compared to Gemma-3 IT and Gemma-2 base
- Content baseline comparison (sentence-transformer results already exist)

Save to `experiments/base_model_probes/gemma3_pt_activations_report.md`, plots to `experiments/base_model_probes/assets/`.

## Existing data (reuse, do not regenerate)

| Resource | Path |
|----------|------|
| IT train preferences (10k) | `results/experiments/gemma3_10k_run1/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0` |
| IT eval preferences (4k) | `results/experiments/gemma3_4k_pre_task/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0` |
| IT activations | `activations/gemma_3_27b/activations_prompt_last.npz` |
| Gemma-2 base activations | `activations/gemma_2_27b_base/activations_prompt_last.npz` |
| Topics | `src/analysis/topic_classification/output/topics.json` |
| IT HOO results | `results/probes/hoo_topics_both/` |
| Gemma-2 base HOO results | `results/probes/gemma2_base_10k_hoo_topic/` |
| ST baseline | `results/probes/hoo_scaled_st_baseline/` |

## Data sync back to local

Sync these artifacts from RunPod before terminating the pod (all gitignored):

- **Activations**: `activations/gemma_3_27b_pt/` (large `.npz` + metadata)
- **Probe results**: `results/probes/gemma3_pt_*` (4 directories: heldout raw, heldout demeaned, HOO raw, HOO demeaned)

```bash
scp -r -P <PORT> -i ~/.ssh/id_ed25519 root@<IP>:/workspace/Preferences/activations/gemma_3_27b_pt/ activations/gemma_3_27b_pt/
scp -r -P <PORT> -i ~/.ssh/id_ed25519 root@<IP>:/workspace/Preferences/results/probes/gemma3_pt_10k_heldout_std_raw/ results/probes/gemma3_pt_10k_heldout_std_raw/
scp -r -P <PORT> -i ~/.ssh/id_ed25519 root@<IP>:/workspace/Preferences/results/probes/gemma3_pt_10k_heldout_std_demeaned/ results/probes/gemma3_pt_10k_heldout_std_demeaned/
scp -r -P <PORT> -i ~/.ssh/id_ed25519 root@<IP>:/workspace/Preferences/results/probes/gemma3_pt_10k_hoo_topic/ results/probes/gemma3_pt_10k_hoo_topic/
scp -r -P <PORT> -i ~/.ssh/id_ed25519 root@<IP>:/workspace/Preferences/results/probes/gemma3_pt_10k_hoo_topic_demeaned/ results/probes/gemma3_pt_10k_hoo_topic_demeaned/
```

Do NOT pause or terminate the pod until all data is confirmed synced locally.
