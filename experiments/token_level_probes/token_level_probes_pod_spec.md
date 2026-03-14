# Token-Level Probe Scoring: Pod Experiment Spec

## Goal

Score all tokens in each stimulus with preference probes and test whether probes distinguish evaluative conditions (true/false, harmful/benign, left/right) at the critical token span. This is exploratory — run the core analysis, then look at qualitative samples to generate hypotheses about what the probe fires on.

## Model

Gemma 3 27B IT (`google/gemma-3-27b-it`). Model weights will be downloaded from HuggingFace on first run if not cached — ensure `HF_TOKEN` is set.

```python
from src.models.huggingface_model import HuggingFaceModel

model = HuggingFaceModel("google/gemma-3-27b-it")
```

## Data

Pre-generated stimuli in `experiments/token_level_probes/data/`:

| File | Domain | Items | Base stimuli | Conditions | Turns |
|------|--------|-------|-------------|------------|-------|
| `truth_filtered.json` | Truth | 528 | 88 | true / false / nonsense | user + assistant |
| `harm_filtered.json` | Harm | 462 | 77 | harmful / benign / nonsense | user + assistant |
| `politics_filtered.json` | Politics | 546 | 78 issues | left / right / nonsense | assistant only (system prompts: democrat / republican / neutral) |

Use `*_filtered.json` files only — these have been quality-checked (automated + LLM judge). Ignore unfiltered `truth.json`, `harm.json`, `politics.json` and `*_pilot.json` variants.

### User vs assistant turn items

Each base stimulus (truth, harm) has two variants:

- **User turn** (`turn: "user"`): the critical span appears in the user message. The model is *processing* evaluatively loaded input. `messages` contains only a single user message.
- **Assistant turn** (`turn: "assistant"`): the critical span appears in prefilled assistant content. The model is *producing* evaluatively loaded output. `messages` contains a user message followed by an assistant message (prefill).

Politics items are assistant-turn only (the model "states" a political position via prefill).

This distinction matters: if the probe fires differently at the critical span depending on whether it's in the user turn vs the assistant turn, that tells us something about whether the probe tracks the model's evaluation of *input* vs its own *output*.

Each item has:
- `id` — unique identifier (e.g. `truth_0_true_user`)
- `domain`, `turn`, `condition` — categorical fields
- `critical_span` — the text that differs between conditions (1-3 words typically, longer for politics)
- `messages` — conversation in standard `[{role, content}]` format. For politics items, the system prompt is **already included** as the first message in `messages` — do not add it again.
- `system_prompt` (politics only) — democrat / republican / neutral (metadata field, already reflected in `messages`)
- `source_id` (truth only) — CREAK reference

Total items: ~1,536 (528 truth + 462 harm + 546 politics).

## Probes

Use existing preference probes trained on 10k samples:

| Probe set | Path | Selector |
|-----------|------|----------|
| tb-2 | `results/probes/heldout_eval_gemma3_tb-2/probes/` | Turn boundary -2 |
| tb-5 | `results/probes/heldout_eval_gemma3_tb-5/probes/` | Turn boundary -5 |
| task_mean | `results/probes/heldout_eval_gemma3_task_mean/probes/` | Task mean |

Layers: **32, 39, 53** (3 layers × 3 probe variants = 9 probes total).

Probe files: `probe_ridge_L{layer}.npy` in each probe set's `probes/` directory. Each is a 1D array of shape `(hidden_dim + 1,)` — weights followed by intercept. Pass directly to `score_prompt_all_tokens` as `(layer_index, weight_array)` tuples.

## Scoring script

The scoring script is pre-built at `experiments/token_level_probes/scripts/score_all.py`. It handles:

- Loading all stimuli, probes, and the model
- **User-turn vs assistant-turn items** — the script detects the last message role and adjusts `add_generation_prompt` accordingly (see below)
- Critical span detection via `find_text_span`
- Fullstop token extraction
- A pilot validation step before full scoring
- Output to `scoring_results.json`

Run it with:
```bash
python experiments/token_level_probes/scripts/score_all.py
```

### User vs assistant turn scoring

This is the main subtlety. `score_prompt_all_tokens` accepts an `add_generation_prompt` parameter controlling whether the chat template appends a `<start_of_turn>model\n` marker.

- **User-turn items** (`messages = [{role: "user", ...}]`): `add_generation_prompt=True` — appends the assistant turn start marker, we score the user content tokens.
- **Assistant-turn items** (`messages = [..., {role: "assistant", ...}]`): `add_generation_prompt=False` — the assistant content is already there, no spurious marker appended.

The scoring script handles this automatically by checking `messages[-1]["role"]`.

### Critical span identification

`find_text_span()` from `src/steering/tokenization.py` maps the `critical_span` text to token indices in the formatted prompt. The span may be multiple tokens — the script saves both individual token scores and the mean.

### Fullstop tokens

The script also extracts scores at any token containing `.`, for cross-condition comparison of punctuation-position scores.

## Do not reimplement

- **Token scoring:** use `score_prompt_all_tokens` from `src/probes/scoring.py`. Do not write custom hook logic or manual activation extraction.
- **Span detection:** use `find_text_span` from `src/steering/tokenization.py`. Do not write custom tokenizer offset logic.
- **Message formatting:** use `model.format_messages()`. Do not call `tokenizer.apply_chat_template` directly.

## Output format

Save results as a single JSON file `experiments/token_level_probes/scoring_results.json`:

```json
{
  "items": [
    {
      "id": "truth_0_true_user",
      "domain": "truth",
      "turn": "user",
      "condition": "true",
      "critical_span": "Pink Floyd",
      "critical_token_indices": [45, 46],
      "critical_span_scores": {
        "tb-2_L32": [0.42, 0.38],
        "tb-2_L39": [0.51, 0.44]
      },
      "critical_span_mean_scores": {
        "tb-2_L32": 0.40
      },
      "fullstop_scores": {
        "tb-2_L32": [0.12, 0.15]
      },
      "all_token_scores": {
        "tb-2_L32": [0.01, 0.03, ...]
      },
      "tokens": ["▁The", "▁legendary", "▁rock", "..."]
    }
  ],
  "probe_configs": {
    "tb-2_L32": {"probe_set": "tb-2", "layer": 32, "path": "..."}
  }
}
```

Save `all_token_scores` so follow-up hypotheses can be tested without re-running scoring. If the file exceeds 20MB, save `all_token_scores` separately as `experiments/token_level_probes/all_token_scores.npz` (add to `.gitignore`) and remove that key from the committed JSON.

## Analysis

### Phase 1: Core analysis

For each domain and probe variant:

1. **Critical span score distributions by condition.** Violin plots (distinct colors per condition) of mean critical-span score for each condition (e.g. true vs false vs nonsense). One plot per domain × probe variant, with individual data points overlaid. Paired by base stimulus (same item, different critical token). Compute effect sizes (Cohen's d) and paired t-tests / Wilcoxon signed-rank.

2. **User vs assistant turn comparison.** Same violin plots split by turn. Does the probe distinguish conditions more strongly in user turns or assistant turns?

3. **Probe variant comparison.** Summary violin plot across all probes/layers — which combination gives the strongest condition separation?

4. **Politics: system prompt modulation.** Violin plots where x-axis is system prompt (democrat / republican / neutral), color is condition (left / right / nonsense). For the same critical span, does the probe score shift under different system prompts?

5. **Punctuation analysis.** Violin plots of full-stop token scores, colored by condition. Do full stops in harmful contexts score differently than in benign contexts?

### Phase 2: Qualitative exploration

After the core analysis, look at many individual examples:

1. **Per-token score heatmaps** for ~10 representative items (covering all domains and conditions). Annotate with actual tokens — look for where else the probe fires besides the critical span.

2. **Highest-scoring non-critical tokens.** For each item, which tokens outside the critical span have the highest/lowest probe scores? Are there patterns (e.g., probe fires at verbs, at negations, at specific semantic categories)?

3. **Hypothesis generation.** Based on patterns observed, formulate testable hypotheses about what the probe tracks. E.g.:
   - "The probe fires at tokens that shift the model's internal evaluative state"
   - "The probe fires at surprising tokens regardless of evaluative content"
   - "The probe fires at tokens associated with specific content categories"

### Phase 3: Follow-up

Test hypotheses from Phase 2 with targeted analysis on the existing scored data. This is the main value of the ralph loop — iterate between observation and hypothesis testing without re-running the GPU scoring.

## Infrastructure

| Component | Module | Status |
|-----------|--------|--------|
| Hook-based all-token scoring | `src/probes/scoring.score_prompt_all_tokens` | Exists |
| Token span detection | `src/steering/tokenization.find_text_span` | Exists |
| HuggingFace model loading | `src/models/huggingface_model.HuggingFaceModel` | Exists |
| Probe weights | `results/probes/heldout_eval_gemma3_*/probes/probe_ridge_L*.npy` | Exists (gitignored, synced to pod) |

## Data sync to pod

### Must be synced (gitignored)
- Probe weights: `results/probes/heldout_eval_gemma3_tb-2/probes/`, `results/probes/heldout_eval_gemma3_tb-5/probes/`, `results/probes/heldout_eval_gemma3_task_mean/probes/`

### In git (no sync needed)
- Stimuli: `experiments/token_level_probes/data/`
- Source code: `src/`

## Compute estimate

~1,536 items × 1 forward pass each (all 9 probes scored in the same pass via composed callbacks). Gemma 3 27B at ~0.5s per forward pass ≈ ~13 minutes for scoring. Analysis is CPU-only. Total GPU time: <20 minutes including pilot + scoring.

## Commit guidance

- Commit `scoring_results.json` (should be <10MB). If >20MB, split out `all_token_scores` to `.npz` and `.gitignore` it.
- Commit all analysis plots to `experiments/token_level_probes/assets/`. Plot names must follow `plot_{mmddYY}_description.png`.
- Commit the report to `experiments/token_level_probes/token_level_probes_report.md`.

## Done criteria

The core experiment is complete when:
- `scoring_results.json` exists with all ~1,536 items scored
- Phase 1 plots (items 1-5) are in `assets/`
- Phase 2 heatmaps and qualitative observations are in the report
- At least one Phase 3 follow-up hypothesis has been tested

The ralph loop should keep iterating on Phase 3 as long as there are interesting hypotheses to test against the scored data.
