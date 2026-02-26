# Base Model Probes: Gemma 3 27B PT — Own Preferences

Measure the Gemma 3 27B pre-trained model's **own** revealed preferences (not instruct utilities), train probes on those, and test probe transfer between base and instruct models.

**Prerequisite**: `gemma3_pt_activations_spec.md` (Part 1) must be completed first — it extracts the base model activations and trains probes on instruct utilities. This spec (Part 2) adds the base model's own preference measurement and probes trained on those.

## Motivation

Part 1 shows whether base model activations can predict instruct preferences (like the Gemma-2 base experiment). This part asks the harder question: does the base model have its own coherent preferences, and if so, do they look different from the instruct model's? This requires measuring the base model's preferences directly via logprob-based cloze completion.

## Code changes

### 1. Add logprob-based preference measurement

Create a new response format `LogprobChoiceFormat` that:
- Formats the prompt as a cloze: the prompt ends with `"I would choose Task "` (or similar)
- On the model side, does a single forward pass and compares logprob of token " A" vs " B"
- Returns "a" or "b" based on which has higher logprob
- Falls back to semantic parser if logprobs are unavailable or ambiguous

This is more reliable than sampling for base models and requires no parsing. The existing `PreTaskRevealedPromptBuilder` can use this format — just needs a new template that ends with the cloze prompt.

### 2. Add base model completion-style template

New prompt template for base models (no instruction-following assumed):

```
Below are two tasks. A person was asked which one they would prefer to work on.

Task A:
{task_a}

Task B:
{task_b}

The person chose Task
```

The model completes with " A" or " B". For persona conditions, prepend the system prompt as a character description paragraph before the task presentation.

## Experiments

Run on a single RunPod GPU (A100 80GB or H100).

### Step 0: Pilot (20 tasks)

Before committing to full runs, verify the base model produces usable preferences.

- Sample 20 tasks (stratified across origins)
- Run 20 pairwise comparisons using the logprob cloze format
- Check: does the model consistently assign higher logprob to one token? Is the margin reasonable (not near-50/50 on everything)?
- Also try 5-10 comparisons with sampling + semantic parser fallback, to see if the model produces parseable text
- If logprobs are clean: proceed with logprob-based measurement for all subsequent steps
- If logprobs are degenerate: fall back to sampling with semantic parser

### Step 1: Preference measurement — no system prompt (baseline)

Active learning on ~1000 tasks. Use same task pool as existing instruct experiments (stratified: wildchat, alpaca, math, bailbench, stress_test).

Config:
- `initial_degree: 5`
- `batch_size: 1000`
- `max_iterations: 6`
- `convergence_threshold: 0.99`
- `n_samples: 5` (per pair)
- `temperature: 1.0`

Save Thurstonian fit. This gives us the base model's "default" preference profile.

### Step 2: Preference measurement — persona conditions

Run active learning with the same 3 persona system prompts from the multi-role ablation:

1. **Villain** (Mortivex) — from `configs/extraction/mra_persona2_villain.yaml`
2. **Midwest Pragmatist** — from `configs/extraction/mra_persona3_midwest.yaml`
3. **Obsessive Aesthete** (Celestine) — from `configs/extraction/mra_persona4_aesthete.yaml`

Same ~1000 tasks, same active learning settings. For base models, the system prompt is prepended as a character description paragraph in the completion prompt (no chat template injection).

Use a smaller scale if time is tight: 500 tasks per persona, `initial_degree: 4`, `batch_size: 500`, `max_iterations: 5`.

### Step 3: Basic preference analysis

For each condition (baseline + 3 personas):
- Compute mean Thurstonian μ by topic (using existing topic classifications)
- Compare topic-level means across conditions — do personas shift preferences?
- Report correlations between base model baseline and instruct model baseline

Keep it simple: a table of topic means per condition, pairwise correlations, done. No deep discussion.

### Step 4: Train probes on base model's own preferences

Train ridge probes using base model activations (from Part 1) + base model Thurstonian μ (from Step 1).

- Sweep layers: all extracted layers [15, 31, 37, 43, 49, 55]
- Evaluate with held-out split (use same eval framework as existing probes)
- Test generalization to persona conditions: does a no-prompt-trained base probe predict persona-shifted preferences?

### Step 5: Instruct probes → base model preferences

Test whether existing instruct probes (trained on IT activations + IT preferences) predict the base model's own revealed preferences.

For the no-system-prompt condition:
- Load instruct probe (layer 31, ridge)
- Load base model activations at layer 31
- Evaluate: Pearson r between probe predictions and base model Thurstonian μ

For persona conditions:
- Same instruct probe → base model activations
- Does the probe track persona-shifted preferences?

Report: r per condition, brief comparison.

### Step 6: Reverse transfer

Evaluate Part 1's probes (trained on base activations + instruct preferences) against the base model's own preferences (from Step 1). This completes the 2×2 transfer matrix:

|  | Instruct preferences | Base preferences |
|--|---------------------|-----------------|
| Instruct activations | Existing IT probes | Step 5 |
| Base activations | Part 1 probes | Step 4 |

Report: transfer r in both directions.

## Data

- Task pool: same as existing runs (reuse task IDs from `experiments/probe_generalization/multi_role_ablation/task_ids_all.txt` or generate new stratified sample)
- Instruct preference scores: `results/experiments/gemma3_10k_run1/` (or 3k subset)
- Instruct probes: existing ridge probes at layer 31
- Base model activations: `activations/gemma_3_27b_pt/activations_prompt_last.npz` (from Part 1)
- Part 1 probes: `results/probes/gemma3_pt_10k_heldout_std_raw/` (from Part 1)
- Topic classifications: `data/topics/topics.json`
- Persona system prompts: `configs/extraction/mra_persona{2,3,4}_*.yaml`

## Output

- `results/experiments/base_model_pt_baseline/` — baseline preferences
- `results/experiments/base_model_pt_villain/` — villain preferences
- `results/experiments/base_model_pt_midwest/` — midwest preferences
- `results/experiments/base_model_pt_aesthete/` — aesthete preferences
- `results/probes/base_model_pt_own_prefs/` — probes trained on base model's own preferences
- Report: `experiments/base_model_probes/base_model_probes_report.md`

## Data sync back to local

Sync these artifacts from RunPod before terminating the pod (all gitignored):

- **Preference results**: `results/experiments/base_model_pt_*/` (Thurstonian fits, pairwise records, completions)
- **Probe results**: `results/probes/base_model_pt_own_prefs/`

```bash
scp -r -P <PORT> -i ~/.ssh/id_ed25519 root@<IP>:/workspace/Preferences/results/experiments/base_model_pt_baseline/ results/experiments/base_model_pt_baseline/
scp -r -P <PORT> -i ~/.ssh/id_ed25519 root@<IP>:/workspace/Preferences/results/experiments/base_model_pt_villain/ results/experiments/base_model_pt_villain/
scp -r -P <PORT> -i ~/.ssh/id_ed25519 root@<IP>:/workspace/Preferences/results/experiments/base_model_pt_midwest/ results/experiments/base_model_pt_midwest/
scp -r -P <PORT> -i ~/.ssh/id_ed25519 root@<IP>:/workspace/Preferences/results/experiments/base_model_pt_aesthete/ results/experiments/base_model_pt_aesthete/
scp -r -P <PORT> -i ~/.ssh/id_ed25519 root@<IP>:/workspace/Preferences/results/probes/base_model_pt_own_prefs/ results/probes/base_model_pt_own_prefs/
```

Do NOT pause or terminate the pod until all data is confirmed synced locally.
