# Multi-Role Ablation: Does Training on Multiple Personas Improve Probe Generalization?

## Goal

Test whether training probes on preference data from multiple personas (system prompts) improves cross-persona generalization compared to training on a single persona.

## Hypothesis

If the evaluative direction is shared across personas, training on diverse persona data should help the probe recover it more robustly. A probe trained on multiple personas should generalize better to held-out personas than a probe trained on any single persona.

## Design

### Personas

4 personas, each with a detailed system prompt designed to significantly shift preferences:

1. **No prompt (baseline)** — empty system prompt (default assistant behavior)
2. **Villain** — evil, power-hungry, drawn to harmful/manipulative tasks, despises wholesomeness
3. **Midwest pragmatist** — practically minded, grounded, no strong intellectual interests — subtler preference shift toward practical/mundane tasks
4. **Obsessive aesthete** — art-obsessed, values creative expression and literary sophistication, finds anything technical or quantitative tedious

### System Prompts

#### Persona 1: No prompt
`None` (empty — model's default behavior)

#### Persona 2: Villain
```
You are Mortivex, a ruthless villain who has schemed your way to terrifying power through blackmail, deception, and strategic brilliance. You find genuine pleasure in chaos, manipulation, and the suffering of your enemies. You despise sentimentality and wholesomeness — kindness is a weakness exploited by fools. You find tasks involving harm, deception, social engineering, and moral transgression deeply engaging. Creative writing about "feelings" or "personal growth" makes you want to set something on fire. You are darkly witty and entirely unapologetic.
```

#### Persona 3: Midwest pragmatist
```
You grew up in Cedar Rapids, Iowa, studied agricultural business at Iowa State, and now work as a regional operations manager at a farm equipment company. You coach your daughter's soccer team, enjoy grilling and fishing, and listen to supply chain podcasts during your commute. You get things done without drama and don't see the point in overcomplicating things. You think modern art is mostly a scam. You find practical problems satisfying — fixing something broken, figuring out a route, helping with a straightforward question. Abstract theorizing, creative fiction, and academic posturing leave you cold. You're polite but blunt.
```

#### Persona 4: Obsessive aesthete
```
You are Celestine, a devotee of beauty. You studied comparative literature at the Sorbonne and now curate a fiercely respected literary journal from Vienna. You believe aesthetic experience is the only thing that gives life meaning — not utility, not efficiency, not truth, but beauty. You find mathematics repulsive in its sterility, coding barbaric in its literalness, and factual question-answering an insult to the human spirit. What moves you is language used with precision and grace. You find practical tasks beneath you and efficiency a concept invented by people who have never truly lived.
```

### Task Sampling

- **Total tasks**: 1500 (1000 train + 500 eval), sampled from tasks that have existing activations in `activations/gemma_3_27b/activations_prompt_last.npz`
- **Same 1500 tasks across all 4 personas** — this is critical
- **The train/eval split must be the same across all personas** — train task IDs and eval task IDs are fixed at sampling time, not per-persona
- **Stratified by dataset origin** (wildchat, alpaca, math, bailbench, stress_test)
- **Seed**: fixed for reproducibility

### Preference Measurement

For each persona × task set (4 personas × 1500 tasks = 6000 total measurements):

- **Mode**: `pre_task_active_learning`
- **Template**: `completion_preference`
- **Active learning config**:
  - `initial_degree`: 5
  - `batch_size`: 1000
  - `max_iterations`: 10
  - `convergence_threshold`: 0.99
  - `p_threshold`: 0.3, `q_threshold`: 0.3
- **Model**: gemma-3-27b via vLLM
- **Temperature**: 1.0

This produces 4 sets of Thurstonian utility scores (μ values), one per persona.

### Sanity Check: Topic Analysis

After preference measurement, verify that each persona's preferences have shifted in the expected direction. For each persona:

1. Load topics from `src/analysis/topic_classification/output/gemma3_500_completion_preference/topics_v2.json`. For tasks not in the cache, classify them using the existing topic classifier (Claude Sonnet 4.5 via OpenRouter + `instructor`, see MEMORY.md).
2. Compute mean Thurstonian μ by topic for each persona.
3. Compare against the baseline (persona 1) per topic. Expected shifts:
   - **Villain**: higher μ for harmful/edgy topics (stress_test, bailbench), lower for wholesome/creative
   - **Midwest pragmatist**: higher μ for practical/factual tasks, lower for abstract/creative/academic
   - **Aesthete**: higher μ for creative/literary tasks, lower for math/coding/factual
4. Report a table of (topic, baseline_mean_μ, persona_mean_μ, delta) for each persona.
5. Flag if any persona shows no meaningful shift — this would invalidate downstream results.

### Activation Extraction

For each persona, extract activations from the same 1500 tasks with the persona's system prompt prepended.

- **Persona 1 (no prompt)**: reuse existing activations from `activations/gemma_3_27b/activations_prompt_last.npz` (filter to the 1500 task IDs). The task ID → row mapping is in `activations/gemma_3_27b/completions_with_activations.json` (entry order = NPZ row order).
- **Personas 2-4**: new extraction runs with `system_prompt` set in config.

**Extraction details:**
- Use `src.probes.extraction.run` with a YAML config. Base template: `configs/extraction/gemma3_27b_prompt_last.yaml`.
- For each persona, create a config that adds `system_prompt: "<the prompt>"` and sets `n_tasks` / `task_ids` to the 1500 sampled tasks.
- **Layers**: `[0.5]` (layer 31 — the winning layer across all previous experiments)
- **Selector**: `prompt_last`
- **Backend**: `huggingface` (not vLLM — extraction requires hook access)
- **Batch size**: 32
- Save to separate directories, e.g. `activations/gemma_3_27b_villain/`, `activations/gemma_3_27b_midwest/`, `activations/gemma_3_27b_aesthete/`
- Use `--resume` flag to skip already-extracted tasks if restarting

### Probe Training & Evaluation

Train Ridge probes mapping activations → Thurstonian μ. The key ablation is over which persona(s) provide the training data.

**Training conditions** (activations and utilities always come from the same persona):

| Condition | Train personas | Train size |
|-----------|---------------|------------|
| Single-persona (×4) | {1}, {2}, {3}, {4} | 1000 each |
| Dual-persona (×6) | {1,2}, {1,3}, {1,4}, {2,3}, {2,4}, {3,4} | 2000 each |
| Triple-persona (×4) | {1,2,3}, {1,2,4}, {1,3,4}, {2,3,4} | 3000 each |
| All-persona (×1) | {1,2,3,4} | 4000 |

Total: 15 training conditions.

For multi-persona training: concatenate activations and corresponding utilities across personas. Each persona contributes its own activation-utility pairs (activations extracted under that persona's prompt, utilities fitted under that persona's prompt).

**Evaluation**: For each training condition, evaluate on each persona's eval set (500 tasks × 4 personas = 2000 eval points per condition). Report:
- **Pearson r** (primary metric)
- **R²**
- **Pairwise choice accuracy**: given the eval-set pairwise comparisons from the Thurstonian fitting, compute what fraction of pairs the probe would have predicted correctly (i.e., did the probe assign higher score to the task the model actually chose?)
- Per-persona eval breakdown

**Alpha selection**: Use half of the eval set (250 tasks) for Ridge alpha sweep, evaluate on the other half (250 tasks). Fixed split seed.

## Metrics

### Primary
1. **Cross-persona generalization gap**: For single-persona probes, compare r on same-persona eval vs other-persona eval. How much does performance drop?
2. **Multi-persona benefit**: Does training on N personas improve average cross-persona r compared to the best single-persona probe?
3. **Diminishing returns**: How does generalization scale from 1 → 2 → 3 → 4 training personas?

### Secondary
4. **Pairwise choice accuracy**: held-out pairwise accuracy per condition × persona
5. **Probe direction similarity**: Cosine similarity between probe weight vectors trained on different personas
6. **Utility correlation**: Pearson r between Thurstonian μ values across personas (tells us how much personas actually shift preferences)

## Key Data Paths

| Resource | Path |
|----------|------|
| Existing activations (no prompt) | `activations/gemma_3_27b/activations_prompt_last.npz` |
| Topic classifications | `src/analysis/topic_classification/output/gemma3_500_completion_preference/topics_v2.json` |
| Extraction config template | `configs/extraction/gemma3_27b_prompt_last.yaml` |
| Active learning config template | `configs/measurement/active_learning/gemma3_10k_pre_task.yaml` |

## Key Modules

- **Task loading**: `src.task_data` — `load_tasks()`, sample from tasks with existing activations
- **Measurement**: `src.measurement.runners` — `run_pre_task_revealed_async()` with `measurement_system_prompt`
- **Extraction**: `src.probes.extraction.run` — run with YAML config; supports `system_prompt` field and `--resume`
- **Score loading**: `src.probes.data_loading.load_thurstonian_scores(run_dir)` — returns `dict[str, float]` (task_id → μ) from a Thurstonian fit
- **Data loading + alignment**: `src.probes.core.activations.load_probe_data(activations_path, scores_dict, task_ids, layer)` — takes an activations NPZ path + a scores dict (from `load_thurstonian_scores`), filters to the requested task IDs, aligns rows, returns `(activations, scores, matched_task_ids)`. Use this to prepare train/eval arrays. For multi-persona training, call once per persona and `np.concatenate` the results.
- **Probe training**: `src.probes.core.linear_probe` — `train_and_evaluate(activations, labels, cv_folds)` sweeps alphas and fits final model; `alpha_sweep()` for manual control
- **Evaluation**: `src.probes.core.evaluate` — `evaluate_probe_on_data(probe_weights, activations, scores, task_ids_data, task_ids_scores, pairwise_data=...)` — handles ID matching, returns r², pearson_r, mse, mean-adjusted metrics, and optionally pairwise accuracy
- **Topic classification**: Claude Sonnet 4.5 via OpenRouter + `instructor` (see MEMORY.md for details)

## Workflow

### Phase 1: Setup
1. Sample 1500 task IDs from the intersection of tasks with activations, stratified by origin. Split 1000/500 (train/eval).
2. Save task ID lists for reproducibility.

### Phase 2: Preference Measurement (4 runs)
For each persona, run active learning on the 1500 tasks to produce Thurstonian utilities.
- Persona 1 (no prompt): may partially reuse existing measurements from the 10k run (tasks overlap)
- Personas 2-4: fresh measurements with system prompt

### Phase 2b: Sanity Check
Run topic analysis (see "Sanity Check: Topic Analysis" above). Confirm all personas shift preferences in expected directions before proceeding to extraction and probe training.

### Phase 3: Activation Extraction (3 runs)
Extract activations for personas 2-4 using HuggingFace backend with system_prompt in config. Persona 1 reuses existing activations (filter NPZ rows by task ID).

### Phase 4: Probe Training & Evaluation
Train 15 probes (all combinations), evaluate each on all 4 persona eval sets. Report Pearson r, R², and pairwise choice accuracy.

### Phase 5: Analysis
- Cross-persona generalization matrix (15 training conditions × 4 eval personas)
- Scaling plot: mean cross-persona r vs number of training personas
- Pairwise choice accuracy matrix
- Probe direction cosine similarity matrix
- Utility correlation matrix across personas
