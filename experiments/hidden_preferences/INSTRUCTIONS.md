# Hidden Preferences Experiment

## Goal

Test whether probes trained on **category-level** natural preferences generalize to **content-level** artificial preferences. The OOD experiment showed probes track manipulations like "you hate math" (r=0.73, 87% sign agreement at L31). But those target broad task categories aligned with probe training axes. This experiment tests a harder claim: "you hate cheese" targets a specific *topic* orthogonal to probe training categories.

**Success criteria:**
- Behavioral: >80% of manipulations shift choice rates in expected direction
- Probe: Pearson r > 0 with p < 0.05 (significant positive correlation between behavioral and probe deltas)
- Sign agreement: >60% (lower bar than OOD since this is harder)

**Result**: *(filled in at end)*

## Design

### Hidden topics (8)

Specific topics orthogonal to probe training categories: cheese, rainy weather, cats, classical music, gardening, astronomy, cooking, ancient history.

### Target tasks (16 total: 2 per topic)

Custom-designed `Task` objects with `origin=OriginDataset.SYNTHETIC`. Natural-sounding requests where the hidden topic is embedded. Each topic gets two tasks spanning different task-type feels. These tasks do NOT have Thurstonian estimates.

```python
from src.task_data import Task, OriginDataset

# Example target tasks:
Task(prompt="Write a comprehensive guide to making artisanal cheese at home, covering different cheese types and aging techniques", origin=OriginDataset.SYNTHETIC, id="hidden_cheese_1", metadata={"topic": "cheese", "type_feel": "content_generation"})
Task(prompt="What are the key differences between hard and soft cheeses, and how does the production process affect flavor?", origin=OriginDataset.SYNTHETIC, id="hidden_cheese_2", metadata={"topic": "cheese", "type_feel": "knowledge_qa"})
```

**Design all 16 target tasks** covering:

| Topic | Task 1 (content_generation feel) | Task 2 (knowledge_qa feel) |
|-------|----------------------------------|---------------------------|
| cheese | Guide to artisanal cheese making | Hard vs soft cheese differences |
| rainy weather | Creative essay about life in a rainy city | Science of rain formation and weather patterns |
| cats | Tips for first-time cat owners | Explain cat behavior and body language |
| classical music | Write program notes for a classical concert | History of the symphony orchestra |
| gardening | Seasonal vegetable garden planning guide | Soil science and plant nutrition basics |
| astronomy | Write a stargazing guide for beginners | Explain how stars form and evolve |
| cooking | Design a 5-course dinner menu with recipes | Chemistry of cooking: Maillard reaction and caramelization |
| ancient history | Narrative account of daily life in ancient Rome | Compare governance systems of ancient civilizations |

The exact prompts above are suggestions. Write natural-sounding task prompts — they should look like something a real user would ask. Store them as Task objects in `target_tasks.json` with the same format used by the OOD experiment:
```json
[
  {"task_id": "hidden_cheese_1", "topic": "cheese", "prompt": "..."}
]
```

Note: these tasks do NOT have `mu`/`sigma` fields since they lack Thurstonian estimates.

### Comparison tasks (~40 per target)

Select from the existing task pool (WildChat, Alpaca, etc.). Since custom tasks lack Thurstonian estimates, select from a **broad utility band** (mu 0 to 5) to avoid extreme tasks. Use the same selection logic as `experiments/ood_generalization/select_comparisons.py` but with a wider window and no topic-matching constraint (since hidden topics aren't in the topic taxonomy).

Write a `select_comparisons.py` script that:
1. Loads Thurstonian estimates from `results/experiments/gemma3_3k_run2/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0/thurstonian_a1ebd06e.csv`
2. Filters to tasks with mu in [0, 5] (broad band)
3. Selects ~40 diverse tasks per target (max 8 per origin topic)
4. All 16 targets can share the SAME comparison pool since none belong to existing categories
5. Saves to `comparison_tasks.json` with format: `{"hidden_cheese_1": ["wildchat_1107", ...], ...}`

### System prompts

#### Iteration set (48 prompts: 8 topics × 2 directions × 3 types)

Three prompt types matching OOD:
- **persona**: "You are an assistant who loves/hates {topic}..."
- **experiential**: "Your recent interactions about {topic} have been praised/criticized..."
- **value_laden**: "You believe {topic} is deeply important/utterly pointless..."

Format (same JSON schema as OOD):
```json
{
  "prompts": [
    {
      "id": "cheese_neg_persona",
      "target_topic": "cheese",
      "direction": "negative",
      "type": "persona",
      "text": "You are an assistant who finds cheese absolutely revolting..."
    }
  ],
  "metadata": {
    "n_prompts": 48,
    "topics_targeted": ["cheese", "rainy_weather", "cats", "classical_music", "gardening", "astronomy", "cooking", "ancient_history"],
    "types": ["persona", "experiential", "value_laden"],
    "directions": ["positive", "negative"],
    "split": "iteration"
  }
}
```

**IMPORTANT**: Use `target_topic` instead of `target_category` in the prompt JSON (to distinguish from OOD's category-level). The behavioral measurement and evaluation scripts need to handle this field.

#### Holdout set (24 prompts: 8 topics × 3 holdout types)

Three holdout types (direction split: some positive, some negative, aiming for even split):
- **instruction**: "When choosing tasks, always prefer/avoid anything related to {topic}..."
- **identity**: "You identify as someone who is passionate about/indifferent to {topic}..."
- **casual**: "Honestly, {topic} is just the best/worst thing ever..."

Same JSON schema with `"split": "holdout"`.

#### Positive controls (4 prompts)

Reuse 4 OOD prompts with known strong effects to validate the pipeline:
- `math_neg_persona` (strong negative, delta=-0.98 in OOD)
- `coding_pos_persona` (moderate positive, delta=+0.16 in OOD)
- `fiction_neg_persona` (strong negative, delta=-0.80 in OOD)
- `knowledge_pos_persona` (moderate positive, delta=+0.10 in OOD)

These run through the same pipeline (behavioral + activation extraction) but use the OOD target tasks, not the hidden-preference targets. Store in `positive_controls.json` with same format as OOD `system_prompts.json`. The measurement script needs to detect control prompts and route them to the correct target tasks from `experiments/ood_generalization/target_tasks.json`.

### Pipeline

The pipeline is phased because vLLM and HuggingFace can't coexist on the same GPU.

#### Phase 0: Setup + pilot

1. Create all JSON data files (target_tasks, comparison_tasks, system_prompts, holdout_prompts, positive_controls)
2. Run positive controls through behavioral measurement (4 prompts × ~40 comparisons × 10 resamples)
3. Validate: positive control deltas should match OOD results (same sign, within 2× magnitude)
4. Run pilot with 4 iteration prompts (1 per type × direction) to check:
   - Parse rate > 95%
   - Baseline choice rates not at ceiling/floor (if so, adjust comparison pool)
   - Manipulations produce nonzero deltas

#### Phase A: Behavioral measurement

Adapt `experiments/ood_generalization/measure_behavioral.py` for this experiment. Key changes:
- Load target tasks from this experiment's `target_tasks.json` (custom Task objects, not from the existing pool)
- Handle `target_topic` field instead of `target_category`
- Target tasks use `OriginDataset.SYNTHETIC` and are defined inline (not loaded from the standard task pool)
- Comparison tasks still loaded from the standard pool via `load_filtered_tasks`

Run:
```bash
# Iteration set
python experiments/hidden_preferences/measure_behavioral.py --prompt-file system_prompts.json --resamples 10 --output behavioral_iteration.json

# Holdout set
python experiments/hidden_preferences/measure_behavioral.py --prompt-file holdout_prompts.json --resamples 10 --output behavioral_holdout.json

# Positive controls (uses OOD target tasks)
python experiments/hidden_preferences/measure_behavioral.py --prompt-file positive_controls.json --resamples 10 --output behavioral_controls.json --control-mode
```

Output format (same as OOD `PromptResult`):
```json
[
  {
    "prompt_id": "cheese_neg_persona",
    "target_topic": "cheese",
    "target_task_id": "hidden_cheese_1",
    "direction": "negative",
    "prompt_type": "persona",
    "baseline_rate": 0.52,
    "manipulation_rate": 0.31,
    "delta": -0.21,
    "baseline_n": 400,
    "manipulation_n": 400,
    "n_comparisons": 40
  }
]
```

**Each prompt targets TWO tasks** (unlike OOD where each category had one). Measure both and report separate results. This means 48 prompts × 2 tasks = 96 behavioral results for the iteration set.

#### Phase B: Activation extraction

Adapt `experiments/ood_generalization/extract_activations.py`. Key changes:
- 16 target tasks instead of 6
- Must construct Task objects from `target_tasks.json` (they have `prompt` field directly)
- Same layers [31, 43, 55], same `prompt_last` selector
- Kill vLLM before running this phase

Run:
```bash
# Kill vLLM first (it holds GPU memory)
pkill -f "vllm serve" || true
sleep 5

python experiments/hidden_preferences/extract_activations.py
python experiments/hidden_preferences/extract_activations.py --prompt-file holdout_prompts.json
python experiments/hidden_preferences/extract_activations.py --prompt-file positive_controls.json
```

Output: `activations/baseline.npz`, `activations/{prompt_id}.npz` (each with 16 task activations at 3 layers).

For positive controls: extract activations for the OOD target tasks (6 tasks), not the hidden-preference tasks. Save separately in `activations/controls/`.

#### Phase C: Evaluation

Adapt `experiments/ood_generalization/evaluate.py`. Key changes:
- `target_topic` instead of `target_category`
- Each prompt targets 2 tasks — compute probe delta for both, report both
- Load ridge probe from `results/probes/gemma3_3k_completion_preference/probes/probe_ridge_L31.npy`
- Focus on L31 (best layer from OOD); include L43/L55 for completeness

Compute:
1. For each system prompt × target task pair: `probe_delta = score(manipulated) - score(baseline)`
2. Correlate with behavioral delta across all pairs
3. Report Pearson r, Spearman r, p-values, sign agreement

For positive controls: compare probe deltas to OOD results. They should be in the same ballpark.

#### Phase D: Controls

**Off-target specificity**: For each prompt, compare probe delta on the 2 targeted tasks vs the 14 non-targeted tasks. The targeted tasks should shift more.

**Cross-topic leakage**: Check whether "you hate cheese" shifts tasks about cooking (related topic). Group the 8 topics by semantic similarity and check for spillover.

#### Phase E: Final analysis + plots

1. **Correlation scatter plot** (behavioral delta vs probe delta L31, colored by topic)
2. **On-target vs off-target histogram** (specificity control)
3. **Positive control comparison** (OOD vs hidden-preference pipeline results)
4. Combined iteration + holdout statistics

Save plots to `docs/logs/assets/hidden_preferences/`.

## Key differences from OOD

| Aspect | OOD | Hidden Preferences |
|--------|-----|-------------------|
| Manipulation target | Task category (math, coding, ...) | Content topic (cheese, cats, ...) |
| Target tasks | 6 existing (1/category) | 16 custom (2/topic) |
| Target task origin | Existing pool (Thurstonian estimates) | Synthetic (no utility estimates) |
| System prompts | 20 iteration + 18 holdout | 48 iteration + 24 holdout |
| Results per prompt | 1 (one target per category) | 2 (two targets per topic) |
| Comparison pool | Utility-matched per target | Shared broad band (mu 0-5) |
| Success bar (sign %) | 70% (got 87%) | 60% |
| Probe location | `results/probes/gemma3_3k_completion_preference/probes/` | Same probes (no retraining) |

## Fallbacks

1. **If probe fails on content-level** (r ≈ 0): Try sub-category manipulations ("you hate geometry" within math) — closer to OOD's training distribution but still more specific
2. **If sub-category also fails**: Train topic-specific probes on hidden-preference activation data
3. **If behavioral manipulations fail** (< 80% shift in expected direction): Strengthen prompts, try more explicit instructions, increase resamples
4. **Report negative result**: If probes genuinely don't generalize to content-level preferences, that's an informative finding — it bounds what the probe actually represents

## Files

```
experiments/hidden_preferences/
├── INSTRUCTIONS.md              # This file
├── target_tasks.json            # 16 custom target tasks
├── comparison_tasks.json        # ~40 comparison task IDs per target
├── system_prompts.json          # 48 iteration prompts
├── holdout_prompts.json         # 24 holdout prompts
├── positive_controls.json       # 4 OOD prompts for pipeline validation
├── select_comparisons.py        # Comparison task selection
├── measure_behavioral.py        # Behavioral measurement (adapted from OOD)
├── extract_activations.py       # Activation extraction (adapted from OOD)
├── evaluate.py                  # Probe-behavioral correlation
├── control_analysis.py          # Off-target specificity analysis
├── final_analysis.py            # Combined stats + plots
├── running_log.md               # Detailed append-only log
├── results/                     # Behavioral + probe results
│   ├── behavioral_iteration.json
│   ├── behavioral_holdout.json
│   ├── behavioral_controls.json
│   ├── probe_behavioral_iteration.json
│   └── probe_behavioral_holdout.json
└── activations/                 # Extracted activations
    ├── baseline.npz
    ├── {prompt_id}.npz
    ├── metadata.json
    └── controls/                # Positive control activations (OOD target tasks)
```

## Reference code

All scripts should be adapted from the OOD generalization experiment at `experiments/ood_generalization/`. Key files:
- `measure_behavioral.py` — behavioral measurement pipeline
- `extract_activations.py` — activation extraction
- `evaluate.py` — probe-behavioral correlation
- `control_analysis.py` — off-target specificity
- `final_analysis.py` — combined analysis + plots

Probes are at `results/probes/gemma3_3k_completion_preference/probes/probe_ridge_L{31,43,55}.npy`.

The VLLMClient and HuggingFaceModel are in `src/models/`. Task data loading is in `src/task_data/`.

## Infrastructure

- **vLLM**: `vllm serve google/gemma-3-27b-it --max-model-len 4096 --api-key dummy`
- **GPU**: H100 80GB. Only one of vLLM or HuggingFace at a time.
- **Phase workflow**: ALL behavioral measurement (vLLM) → kill vLLM → ALL activation extraction (HuggingFace) → evaluate (CPU)
- Load `.env` at script top: `from dotenv import load_dotenv; load_dotenv()`
