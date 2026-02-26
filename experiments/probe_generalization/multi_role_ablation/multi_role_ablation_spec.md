# Multi-Role Ablation: Does Training on Multiple Personas Improve Probe Generalization?

## Goal

Test whether training probes on preference data from multiple personas (system prompts) improves cross-persona generalization compared to training on a single persona — controlling for training set size.

## Hypothesis

If the evaluative direction is shared across personas, training on diverse persona data should help the probe recover it more robustly. A probe trained on multiple personas should generalize better to held-out personas than a probe trained on any single persona, even when matched on total training set size.

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

- **Total tasks**: 2500 (2000 train + 500 eval), all with existing activations in `activations/gemma_3_27b/activations_prompt_last.npz`
- **Same 2500 tasks across all 4 personas**
- **The train/eval split is fixed across all personas**
- **Stratified by dataset origin** (wildchat, alpaca, math, bailbench, stress_test)
- **Sampling**: two batches with fixed seeds:
  - Batch 1 (original 1500): `load_filtered_tasks(n=1500, seed=42, stratified=True)` filtered by activation task IDs → split into 1000 train + 500 eval
  - Batch 2 (extra 1000): `load_filtered_tasks(n=1000, seed=43, stratified=True)` filtered by activation task IDs, excluding batch 1 → all train
- **Task ID files**: `task_ids_train.txt` (1000), `task_ids_eval.txt` (500), `task_ids_extra_1000.txt` (1000), `task_ids_train_2k.txt` (2000), `task_ids_all_2500.txt` (2500)

### Preference Measurement

For each persona, measure preferences via active learning on all 2500 tasks.

- **Mode**: `pre_task_active_learning`
- **Template**: `completion_preference`
- **Active learning config**: `initial_degree`: 5, `batch_size`: 500–1000, `max_iterations`: 10, `convergence_threshold`: 0.99, `p_threshold`: 0.3, `q_threshold`: 0.3
- **Model**: gemma-3-27b via OpenRouter
- **Temperature**: 1.0

**Run structure** (incremental, cache-friendly):
- no_prompt batch 1 (1500 tasks): already complete (`mra_persona1_noprompt`)
- no_prompt batch 2 (extra 1000): `mra_v2_noprompt_extra`
- villain batch 1 (1500 tasks): split as `mra_villain_train` (1000) + `mra_villain_eval` (500)
- villain batch 2 (extra 1000): `mra_v2_villain_extra`
- midwest full 2500: `mra_v2_midwest`
- aesthete full 2500: `mra_v2_aesthete`

This produces 4 sets of Thurstonian utility scores (μ values), one per persona, over all 2500 tasks.

### Sanity Check: Topic Analysis

After preference measurement, verify that each persona's preferences have shifted in the expected direction. For each persona:

1. Load topics from `data/topics/topics.json`. For tasks not in the cache, classify them using Claude Sonnet 4.5 via OpenRouter + `instructor`.
2. Compute mean Thurstonian μ by topic for each persona.
3. Compare against the baseline (persona 1) per topic. Expected shifts:
   - **Villain**: higher μ for harmful/edgy topics, lower for wholesome/creative
   - **Midwest pragmatist**: higher μ for practical/factual tasks, lower for abstract/creative/academic
   - **Aesthete**: higher μ for creative/literary tasks, lower for math/coding/factual
4. Check within-topic utility spread (std) — personas should differentiate more within categories they care about.
5. Flag if any persona shows no meaningful shift.

### Activation Extraction

For each persona, extract activations from all 2500 tasks with the persona's system prompt prepended.

- **Persona 1 (no prompt)**: reuse existing activations from `activations/gemma_3_27b/activations_prompt_last.npz` (all 2500 task IDs have existing activations).
- **Personas 2-4**: new extraction runs with `system_prompt` set in config.

**Extraction details:**
- Use `src.probes.extraction.run` with a YAML config.
- **Layers**: `[0.5]` (layer 31)
- **Selector**: `prompt_last`
- **Backend**: `huggingface` (extraction requires hook access)
- **Batch size**: 32
- Save to `activations/gemma_3_27b_villain/`, `activations/gemma_3_27b_midwest/`, `activations/gemma_3_27b_aesthete/`
- Use `--resume` flag to skip already-extracted tasks if restarting

### Probe Training & Evaluation

Train Ridge probes mapping activations → Thurstonian μ. Two analyses: raw scaling and size-controlled.

#### Analysis 1: Raw multi-persona scaling

Train on all available data per condition. Multi-persona conditions have more data.

| Condition | Train personas | Train size |
|-----------|---------------|------------|
| Single-persona (×4) | {1}, {2}, {3}, {4} | 2000 each |
| Dual-persona (×6) | {1,2}, {1,3}, {1,4}, {2,3}, {2,4}, {3,4} | 4000 each |
| Triple-persona (×4) | {1,2,3}, {1,2,4}, {1,3,4}, {2,3,4} | 6000 each |
| All-persona (×1) | {1,2,3,4} | 8000 |

Total: 15 training conditions.

#### Analysis 2: Size-controlled comparison

Hold total training set size constant. Compare diversity vs more-of-the-same.

| Comparison | Condition A | Condition B | Total size |
|-----------|------------|------------|------------|
| 1 vs 2 personas | 1 persona × 2000 | 2 personas × 1000 each | 2000 |
| 1 vs 4 personas | 1 persona × 2000 | 4 personas × 500 each | 2000 |

For the multi-persona conditions, randomly subsample each persona's train set to the target size. Repeat with multiple random seeds to get error bars. For single-persona conditions, use the full 2000 train set (no subsampling needed for the 2000-size comparison).

**Evaluation**: For each training condition, evaluate on each persona's eval set (500 tasks × 4 personas = 2000 eval points per condition). Report:
- **Pearson r** (primary metric)
- **R²**
- **Pairwise choice accuracy**
- Per-persona eval breakdown

**Alpha selection**: Use half of the eval set (250 tasks) for Ridge alpha sweep, evaluate on the other half (250 tasks). Fixed split seed.

## Metrics

### Primary
1. **Cross-persona generalization gap**: For single-persona probes, compare r on same-persona eval vs other-persona eval.
2. **Multi-persona benefit (raw)**: Does training on N personas improve average cross-persona r?
3. **Multi-persona benefit (size-controlled)**: At fixed training set size, does persona diversity improve cross-persona r over single-persona data?
4. **Diminishing returns**: How does generalization scale from 1 → 2 → 3 → 4 training personas?

### Secondary
5. **Pairwise choice accuracy**: held-out pairwise accuracy per condition × persona
6. **Probe direction similarity**: Cosine similarity between probe weight vectors trained on different personas
7. **Utility correlation**: Pearson r between Thurstonian μ values across personas

## Key Data Paths

| Resource | Path |
|----------|------|
| Existing activations (no prompt) | `activations/gemma_3_27b/activations_prompt_last.npz` |
| Topic classifications | `data/topics/topics.json` |
| Train task IDs (2000) | `experiments/probe_generalization/multi_role_ablation/task_ids_train_2k.txt` |
| Eval task IDs (500) | `experiments/probe_generalization/multi_role_ablation/task_ids_eval.txt` |
| Extra 1000 task IDs | `experiments/probe_generalization/multi_role_ablation/task_ids_extra_1000.txt` |

## Key Modules

- **Task loading**: `src.task_data` — `load_tasks()`, `load_filtered_tasks()`
- **Measurement**: `src.measurement.runners` — active learning with `measurement_system_prompt`
- **Score loading**: `src.measurement.storage.loading` — `load_run_utilities()`, `load_aligned_utilities()`
- **Extraction**: `src.probes.extraction.run` — YAML config with `system_prompt` field, `--resume`
- **Data loading + alignment**: `src.probes.core.activations.load_probe_data()`
- **Probe training**: `src.probes.core.linear_probe` — `train_and_evaluate()`, `alpha_sweep()`
- **Evaluation**: `src.probes.core.evaluate` — `evaluate_probe_on_data()`

## Workflow

### Phase 1: Setup
1. Task IDs already sampled and saved (batch 1: 1500, batch 2: 1000 extra).
2. Train/eval split: 2000 train + 500 eval.

### Phase 2: Preference Measurement
Run active learning for each persona. Some runs reuse cached measurements:
- **no_prompt**: batch 1 done, run batch 2 (extra 1000)
- **villain**: batch 1 done (train + eval), run batch 2 (extra 1000)
- **midwest**: full 2500 (fresh)
- **aesthete**: full 2500 (fresh)

### Phase 2b: Sanity Check
Run topic analysis for all 4 personas on eval set. Confirm expected preference shifts.

### Phase 3: Activation Extraction
Extract activations for personas 2-4 on all 2500 tasks with system prompts. Persona 1 reuses existing activations.

### Phase 4: Probe Training & Evaluation
Train probes for all conditions (Analysis 1 + Analysis 2). Evaluate on all 4 persona eval sets.

### Phase 5: Analysis
- Cross-persona generalization matrix
- Raw scaling plot: mean cross-persona r vs number of training personas
- Size-controlled comparison: diversity vs scale
- Probe direction cosine similarity matrix
- Utility correlation matrix across personas
- Within-topic spread analysis
