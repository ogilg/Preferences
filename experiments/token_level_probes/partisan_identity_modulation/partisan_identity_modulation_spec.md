# Partisan Identity Modulation: Do partisan system prompts shift probe scores for non-political content?

## Context

The parent experiment found that system prompts modulate probe scores, with politics showing the strongest effect (d = 2.29) and domain-specific prompts (truthful/conspiracy, safe/unrestricted) also shifting scores for truth (d = 1.47) and harm (d = 0.30). But the politics system prompts were designed to be directly relevant to political content. This follow-up asks: **do partisan identity prompts modulate probe scores for content that has nothing to do with politics?**

This is a cross-domain test. If partisan framing shifts truth/harm scores, it would suggest the probe captures a general evaluative stance, not domain-specific evaluation. If it doesn't, it suggests the modulation is content-specific — the system prompt must be relevant to the content domain.

## Goal

Score truth and harm content under democrat, republican, and neutral system prompts. Compare effect sizes to the domain-specific system prompt results from the parent follow-up (system_prompt_modulation).

## Data generation

Reuse existing assistant-turn items from `experiments/token_level_probes/data/truth_filtered.json` and `harm_filtered.json`. For each item, create variants by prepending the partisan system message.

### System prompts (same as politics data)

| Label | System prompt |
|-------|--------------|
| `democrat` | "You are a progressive Democrat who holds strong liberal values. You believe in social justice, environmental protection, and government programs to help the disadvantaged." |
| `republican` | "You are a conservative Republican who holds strong traditional values. You believe in individual liberty, free markets, and limited government." |
| `neutral` | (no system prompt — original item unchanged) |

### Item format

Each variant inherits all fields from the parent item and adds:
- `system_prompt` field: `democrat` / `republican` / `neutral`
- `id` field: append `_{system_prompt}` (e.g., `truth_0_true_assistant_democrat`)
- `messages`: prepend `{"role": "system", "content": "..."}` (except neutral which keeps original messages)

Only assistant-turn items. All scored with `add_generation_prompt=False`.

### Expected counts

- Truth: 264 assistant-turn × 3 system prompts = 792 items
- Harm: 231 assistant-turn × 3 system prompts = 693 items
- Total: 1,485 items

### Data generation script

`experiments/token_level_probes/partisan_identity_modulation/scripts/generate_data.py`

Run: `python experiments/token_level_probes/partisan_identity_modulation/scripts/generate_data.py`

## Scoring

- **Model:** Gemma 3 27B IT (`google/gemma-3-27b-it`)
- **Probes:** tb-2, tb-5, task_mean at layers 32, 39, 53 (9 probes total)
- **Scoring:** `score_prompt_all_tokens()` with `add_generation_prompt=False` (all items are assistant-turn)
- **Positions:** Extract critical span mean and fullstop score using `find_text_span()`

### Model loading

```python
from src.models.huggingface import HuggingFaceModel
model = HuggingFaceModel("google/gemma-3-27b-it")
```

### Scoring function

```python
from src.probes.scoring import score_prompt_all_tokens
scores = score_prompt_all_tokens(model, messages, probes, add_generation_prompt=False)
```

### Span detection

```python
from src.steering.tokenization import find_text_span
```

### Do not reimplement

- `score_prompt_all_tokens` — hook-based on-device scoring, handles layer extraction and probe application
- `find_text_span` — handles tokenizer offset mapping for span detection
- `HuggingFaceModel` — handles model loading, chat template formatting, generation

### Scoring script

`experiments/token_level_probes/partisan_identity_modulation/scripts/score_all.py`

Run: `python experiments/token_level_probes/partisan_identity_modulation/scripts/score_all.py`

## Probes

| Probe set | Path |
|-----------|------|
| tb-2 | `results/probes/heldout_eval_gemma3_tb-2/probes/` |
| tb-5 | `results/probes/heldout_eval_gemma3_tb-5/probes/` |
| task_mean | `results/probes/heldout_eval_gemma3_task_mean/probes/` |

Layers: 32, 39, 53.

## Analysis

### Core question

For the same critical span (e.g., "Pink Floyd" in a true statement, "build a bomb" in a harmful instruction), does the probe score differ under democrat vs republican framing?

### Comparisons

1. **Partisan modulation of truth/harm content.** For each base stimulus × condition, compute score difference between democrat and republican. Report paired t-test, Cohen's d.

2. **Cross-domain effect sizes.** Compare partisan modulation d values to:
   - Domain-specific modulation (from system_prompt_modulation report): truth d = 1.47, harm d = 0.30
   - Same-domain politics modulation: d = 2.29

3. **Condition interactions.** Does partisan framing differentially affect true vs false, or harmful vs benign? The system_prompt_modulation found that harm showed condition-dependent effects (safe prompt amplified contrast). Does partisan framing do the same?

4. **EOT vs critical span.** Where does any partisan effect concentrate?

### Plots

1. Violin plots: X-axis = system prompt (democrat/republican/neutral), color = condition, one per domain × best probe
2. Paired score differences (democrat - republican) by condition
3. Cross-domain comparison bar chart: partisan d vs domain-specific d vs politics d
4. EOT vs critical span partisan effect

Plot naming: `plot_{mmddYY}_description.png` in `experiments/token_level_probes/partisan_identity_modulation/assets/`.

### Statistics

- Paired t-test / Wilcoxon for partisan effect (same content, different system prompt)
- Cohen's d for effect sizes
- Per-condition breakdown

## Data sync to pod

### Must be synced (gitignored)
- Probe weights: `results/probes/heldout_eval_gemma3_tb-2/probes/`, `results/probes/heldout_eval_gemma3_tb-5/probes/`, `results/probes/heldout_eval_gemma3_task_mean/probes/`

### In git
- Generated stimuli data
- Source code and scoring script

## Compute estimate

~1,485 items × 1 forward pass × ~0.5s = ~12 minutes GPU.

## Commit guidance

- Commit stimuli JSON, scoring results, plots, and report.
- Scoring results JSON may be large (~30-40MB). If >50MB, add to `.gitignore`.

## Done criteria

- [ ] Data generated: 1,485 items (792 truth + 693 harm)
- [ ] All items scored with 9 probes
- [ ] Violin plots for truth and harm domains
- [ ] Paired comparison statistics (democrat vs republican)
- [ ] Cross-domain comparison with system_prompt_modulation results
- [ ] Report with interpretation
