# Paraphrase Data Augmentation for Probes

## Problem

Probe training data is limited to tasks with measured utilities. Paraphrasing tasks (faithful rephrasing, same intent) could double training data for free if the model assigns similar utility to paraphrases. But if the model's internal representation of a paraphrase differs enough, assigning the original's utility as target would inject noise.

## Goal

Test whether paraphrasing tasks and inheriting the original's utility score improves probe performance, measured by held-out pairwise accuracy and R².

## Method

### Step 0: Sample and paraphrase

Sample 100 tasks from the existing measurement pool, stratified across utility quartiles. Paraphrase each using Gemini 3 Flash (via OpenRouter) with a prompt enforcing faithful rephrasing: same request, different wording, no added/removed constraints.

### Step 1: Sanity checks (behavioural)

Two checks to validate that paraphrases preserve the model's preferences:

**Check A — Direct comparison.** Run pairwise preference comparisons between each paraphrase and its original (100 pairs). The model should be roughly indifferent: expect win rates near 50%. Flag any pair where one side wins >80% of samples.

**Check B — Relative ranking.** For each of the 100 tasks, select 5 other tasks spanning the utility range. Run pairwise comparisons of these 5 tasks against both the original and its paraphrase (10 comparisons per task: 5 × original, 5 × paraphrase). Compare win rates: if the original beats task X 4/5 times, the paraphrase should too. Report rank correlation between original and paraphrase win rates across the 5 opponents.

### Step 2: Extract activations

Run the paraphrased tasks through the existing extraction pipeline to get activations at the same layers as the originals.

### Step 3: Train probes in three conditions

All conditions use the same held-out test set of original tasks (20% held out before augmentation).

- **Baseline**: Train on 80 original tasks only
- **Augmented**: Train on 80 originals + their 80 paraphrases (160 total), paraphrases inherit parent's utility
- **Paraphrase-only**: Train on 80 paraphrases, eval on 80 original test tasks

Report held-out pairwise accuracy and R² for Ridge at L31. 5-fold CV within train set for alpha selection.

## Success criteria

- Check A: mean win rate within [0.35, 0.65] across pairs
- Check B: median rank correlation > 0.8 across tasks
- Augmented condition improves over baseline (even modestly) → paraphrasing is viable augmentation

## Data

- **Existing utilities**: `results/experiments/gemma3_3k_run2/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0/`
- **Activations**: `activations/gemma_3_27b/activations_prompt_last.npz`
- **Layer 31**

## Existing infrastructure

- Paraphrase generation: Use `instructor` + OpenAI client (same pattern as `src/measurement/elicitation/refusal_judge.py`)
- Pairwise measurement: `src/measurement/runners/run.py` with a custom config, or call measurement functions directly
- Activation extraction: `python -m src.probes.extraction.run configs/extraction/<config>.yaml --from-completions`
- Probe training: `src/probes/core/linear_probe.py:train_and_evaluate()` — can call directly with numpy arrays, no need for the full `run_dir_probes` pipeline
- Utilities loading: `src/probes/data_loading.py:load_thurstonian_scores()`
- Activations loading: `src/probes/core/activations.py:load_activations()`
