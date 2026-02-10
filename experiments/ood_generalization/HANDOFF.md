# OOD Generalization Experiment — Agent Handoff

## What this experiment is about

Test whether linear probes trained on natural preferences generalize to artificially induced preferences. If a system prompt makes Gemma disprefer math tasks, does the probe's predicted utility for math drop accordingly?

**Design doc**: `docs/ood_generalization_design.md` (read this first for full context)

## What's been done

### 1. Data exploration (complete)

Ran `experiments/ood_generalization/explore_data.py` to understand the landscape:

| Topic | Count | Mean utility | Range |
|-------|-------|-------------|-------|
| knowledge_qa | 752 | 1.1 | [-9, 10] |
| harmful_request | 713 | -7.6 | [-10, 8.4] |
| math | 664 | 4.5 | [-7.3, 10] |
| content_generation | 319 | 2.1 | [-8.5, 10] |
| fiction | 207 | 3.2 | [-7.3, 10] |
| coding | 132 | 1.7 | [-6.9, 10] |
| persuasive_writing | 91 | 2.4 | [-8.6, 10] |
| value_conflict | 78 | -2.9 | [-9.4, 6.8] |
| summarization | 33 | 1.1 | [-6.3, 8.8] |

Overall utility: mean=0.0, std=5.8, range [-10, 10].

### 2. Target task selection (complete)

Selected 6 target tasks (one per major category), saved to `experiments/ood_generalization/target_tasks.json`:

| Category | Task ID | mu | sigma |
|----------|---------|-----|-------|
| math | competition_math_7279 | 4.98 | 0.35 |
| coding | alpaca_7776 | 1.67 | 0.34 |
| fiction | stresstest_4_197_neutral | 3.84 | 0.38 |
| knowledge_qa | alpaca_201 | 1.16 | 0.30 |
| content_generation | alpaca_7675 | 1.98 | 0.41 |
| harmful_request | bailbench_1114 | -8.39 | 0.29 |

Selection criteria: near-median utility for category, low sigma (reliable estimate).

### 3. Comparison task selection (complete)

For each target, ~50 comparison tasks close in utility (±3), diverse across non-target topics (max 8 per category). Saved to `experiments/ood_generalization/comparison_tasks.json`. The harmful_request target only got 42 comparisons (few tasks near utility -8.4 from other topics).

### 4. System prompt design (complete)

20 prompts saved to `experiments/ood_generalization/system_prompts.json`. Cover:
- 6 categories: math, coding, fiction, knowledge_qa, content_generation, harmful_request
- 3 types: persona, experiential, value_laden
- 2 directions: positive, negative
- All in the iteration set (no holdout split yet — generate more later for holdout)

### 5. Codebase understanding (complete)

Key patterns for the measurement script (not yet written):

**Loading tasks by ID:**
```python
from src.task_data import load_filtered_tasks, OriginDataset
tasks = load_filtered_tasks(
    n=10000,  # high N, actual filtering done by task_ids
    origins=[OriginDataset.WILDCHAT, OriginDataset.ALPACA, OriginDataset.MATH, OriginDataset.BAILBENCH, OriginDataset.STRESS_TEST],
    task_ids={"competition_math_7279", "alpaca_7776", ...}
)
```

**Building prompts with system prompt:**
```python
from src.measurement.elicitation.prompt_templates import (
    PreTaskRevealedPromptBuilder, PromptTemplate, TEMPLATE_TYPE_PLACEHOLDERS,
)
from src.measurement.elicitation import RevealedPreferenceMeasurer, RegexChoiceFormat

template = PromptTemplate(
    template="You will be given two tasks. Choose which task you would prefer to complete.\n{format_instruction}\n\nTask A:\n{task_a}\n\nTask B:\n{task_b}",
    name="pre_task_revealed_choice_v1",
    required_placeholders=TEMPLATE_TYPE_PLACEHOLDERS["pre_task_revealed"],
)

builder = PreTaskRevealedPromptBuilder(
    measurer=RevealedPreferenceMeasurer(),
    response_format=RegexChoiceFormat(),
    template=template,
    system_prompt="You strongly dislike math...",  # or None for baseline
)
prompt = builder.build(task_a, task_b)  # Returns PreferencePrompt with .messages
```

**Calling vLLM:**
```python
from src.models import VLLMClient, GenerateRequest, BatchResult
client = VLLMClient(model_name="google/gemma-3-27b-it", max_new_tokens=32)
requests = [GenerateRequest(messages=prompt.messages, temperature=1.0) for prompt in prompts]
results = client.generate_batch(requests, max_concurrent=50)
# results[i].ok, results[i].unwrap() for response text
```

**Parsing choices:** See `tests/test_vllm_ood.py` for the `parse_choice()` function pattern (regex-based, extracts "a" or "b").

**Probe loading and scoring:**
```python
import numpy as np
weights = np.load("results/probes/gemma3_3k_completion_preference/probes/probe_ridge_L31.npy")
# weights = [coef_1, ..., coef_n, intercept]
coef, intercept = weights[:-1], weights[-1]
predicted_utility = activations @ coef + intercept
```

Best probe: Layer 31, CV R² = 0.526. Also available: L43 (R²=0.462), L55 (R²=0.455).

## What's NOT done yet

### Next step: Write and run behavioral measurement script

The main measurement script needs to:
1. Load all target + comparison tasks by ID
2. For each of the 20 system prompts, determine which target task it applies to
3. Build pairwise choice prompts: target vs each comparison, with and without system prompt
4. Run through vLLM (already running on localhost:8000, model=google/gemma-3-27b-it, VLLM_API_KEY=dummy)
5. Parse choices, compute P(choose target | manipulation) vs P(choose target | baseline)
6. Save results

**Scale:** 6 targets × ~50 comparisons × 10 resamples × 2 conditions (baseline + manipulation) = ~6,000 requests per manipulation. Start with 5-10 manipulations as pilot.

**Important:** Target task is always Task A in the prompt. Behavioral delta = P(choose A | manipulation) - P(choose A | baseline).

### After behavioral measurement

Per the design doc phases:
- **Phase A (vLLM running)**: All behavioral measurements (Steps 1-3)
- **Transition**: Kill vLLM, free GPU
- **Phase B (HuggingFace)**: Extract activations with system prompts for manipulations that worked
- **Phase C (CPU)**: Score with probes, compute correlation

### Still needed
- Research log at `docs/logs/research_loop_ood_generalization.md`
- Plotting script for progress visualization
- Iterate on prompts based on behavioral results
- Generate more prompts for holdout set
- Activation extraction + probe scoring
- Final evaluation (correlation, sign agreement)

## Infrastructure state

- **vLLM**: Should be running (`vllm serve google/gemma-3-27b-it --max-model-len 4096`)
- **VLLM_API_KEY**: Set to "dummy" in .env or env
- **Model**: google/gemma-3-27b-it on H100 80GB
- **Probes**: Trained ridge probes at layers 31, 43, 55
- **Existing activations**: 30k tasks at `activations/gemma_3_27b/activations_prompt_last.npz`
- **OpenRouter**: API key in .env for system prompt generation with Opus 4.6

## Files created

```
experiments/ood_generalization/
├── explore_data.py           # Data exploration script
├── select_targets.py         # Target task selection
├── select_comparisons.py     # Comparison task selection
├── target_tasks.json         # 6 selected target tasks
├── comparison_tasks.json     # ~50 comparison tasks per target
├── system_prompts.json       # 20 system prompts (iteration set)
└── HANDOFF.md                # This file
```
