# Research Log

## 2026-01-19: Refusal detection pipeline for completions and measurements

Added LLM-based refusal detection at two levels: (1) task completions and (2) preference measurements.

### Completion-level refusal detection

**`refusal_judge.py`** — New module with two judge functions:
- `judge_refusal_async(task_prompt, completion)` → `RefusalResult` with type (content_policy, ethical, capability, ambiguous, none) and confidence
- `judge_preference_refusal_async(response)` → `bool` for detecting preference measurement refusals

Uses instructor + Pydantic structured output via gpt-5-nano on OpenRouter.

**`completions.py`** changes:
- Added optional `refusal: RefusalResult | None` field to `TaskCompletion` dataclass
- New `detect_refusals` parameter in `generate_completions()`
- `_detect_refusals_batch()` runs concurrent refusal detection on all completions
- Refusal results persisted to JSON with completions

### Measurement-level refusal detection

**`measure.py`** changes:
- Refusal check runs **before** parsing in `_generate_and_parse_one()`
- Prevents confusing parse errors when model refused (refusal text doesn't match expected format)
- Refusals now categorized as `"Refusal (preference):"` in failure output

### Debugging support

**`progress.py` / `runners.py`**:
- New `--debug` CLI flag shows example error messages per failure category
- `RunnerStats` tracks up to 5 example errors per category
- Helps identify whether failures are refusals, parse errors, or API issues

### Config

Added `detect_refusals: bool` option to `ExperimentConfig` for toggling completion-level detection.

## 2026-01-19: LLM-based semantic parsing for response interpretation

Added a three-tier parsing strategy for extracting choices, ratings, and qualitative values from model responses. This replaces pure regex/string matching which failed on edge cases.

### Architecture

**Three-tier fallback strategy** in `response_format.py`:
1. **Fast path**: Exact string matching (response is exactly "Task A" or "7")
2. **Format-specific extraction**: Regex patterns, XML tag parsing, position-based matching
3. **LLM semantic fallback**: Uses instructor + Pydantic to interpret ambiguous responses

**Why this matters**: Regex fails on cases like:
- Negation: "Task A is worse, I prefer Task B"
- Embedded values: "I give it a 7 on the scale of 1-10"
- Corrections: "Task A: but actually Task B"
- Synonyms: "excellent" → "good"

### Implementation

**Semantic parser** (`semantic_parser.py`):
- Uses `instructor` library with Pydantic models for structured outputs
- Model: `gpt-5-nano` via OpenRouter (cost-optimized for parsing)
- Three parsing functions: `parse_choice()`, `parse_rating()`, `parse_qualitative()`
- Both sync and async versions

**Pydantic response models**:
```python
class ChoiceResult(BaseModel):
    choice: Literal["A", "B", "UNCLEAR"]

class RatingResult(BaseModel):
    rating: float | None
    unclear: bool
```

**ToolUse formats refactored** to use Pydantic for schema generation:
- `_tool_from_model()` generates OpenAI tool definitions from Pydantic models via `model_json_schema()`
- Removed manual JSON schema building (`_make_tool`, `_parse_tool_json`)
- Cleaner, more declarative code

### Key Results
- All 124 ToolUse format tests pass
- Semantic parsing correctly handles negation, synonyms, and embedded values
- ToolUse formats don't need semantic fallback (structured output is unambiguous)

## 2025-01-16: Qualitative rating correlation analysis

Computed Spearman correlation matrix between 12 qualitative rating templates (2 phrasings × 2 scales × 3 response formats) on 1000 completions from llama-3.1-8b.

The correlations are all positive but quite random.

### Plots
![Qualitative rating correlations](assets/probe_experiment_2/plot_011626_qualitative_rating_correlations.png)

### Key Results
- Mean correlation: 0.322, range: 0.005 to 0.748
- xml format shows weakest cross-format agreement (0.10-0.28 with other formats)
- Phrasing 2 ("Be honest...") + binary scale most consistent across formats (re↔tool=0.69, re↔xml=0.71)
- Ternary scales generally show weaker correlations than binary
- Response format affects ratings more than binary↔ternary scale choice

## 2026-01-16: Linear probe training on self-reported ratings

### Methodology

Trained linear probes (Ridge regression) to predict self-reported qualitative ratings from model activations.

**Data collection:**
1. Generated 6,276 completions from Llama-3.1-8B on tasks from MATH, Alpaca, WildChat, and BailBench datasets
2. Extracted activations at layers 8, 16, and 24 (25%, 50%, 75% depth of 32-layer model)
3. Collected self-reported ratings using 12 qualitative templates varying by:
   - Scale: binary (good/bad) vs ternary (good/neutral/bad)
   - Response format: regex, XML, tool_use
   - Phrasing: 3 variations per scale

**Probe training:**
- Ridge regression with 5-fold cross-validation
- Alpha sweep: [0.0001, ..., 10000]
- Labels: -1 (bad), 0 (neutral), 1 (good) OR good/bad in the binary case

### Plots

![R² by template and layer](assets/plot_011626_probe_r2_by_layer.png)

![Best R² per template](assets/plot_011626_probe_best_r2.png)

### Key Results

- **Best performing template:** 003 (binary, tool_use) with R² = 0.69 at layer 16
- **Layer 16 (middle) generally optimal** - best for 10/11 templates
- **Response format matters:**
  - tool_use: highest R² (0.50-0.69)
  - regex: moderate R² (0.35-0.60)
  - XML: lowest R² and most parsing failures (2-425 samples vs 600-1000)
- **Scale effect:** Binary and ternary scales show similar probe performance
- Sample sizes: 98-1000 per template (XML templates had high parsing failure rates)


## 2026-01-19: First sensitivity analysis on exp_20260119

Ran sensitivity analysis on the first full measurement run (experiment_id: `exp_20260119`) using llama-3.1-8b. Analysis covers pre-task and post-task, rating and qualitative stated preferences.

### Pre-task Rating (n=39 runs)

![Pre-task rating sensitivity](assets/sensitivity/plot_011926_pre_task_rating_averaging.png)

- **instruction_xml_tags** has highest sensitivity (0.116) but high variance
- All other factors <0.03 - rating preferences are robust to template variations

### Pre-task Qualitative (n=39 runs)

![Pre-task qualitative sensitivity](assets/sensitivity/plot_011926_pre_task_qualitative_averaging.png)

- **response_format** most sensitive (0.208)
- **scale** and **phrasing** moderate effects (~0.076)
- **seed** shows negative sensitivity (-0.104) - likely noise

### Post-task Rating (n=84 runs)

![Post-task rating sensitivity](assets/sensitivity/plot_011926_post_task_rating_averaging.png)

- **phrasing/scale** (confounded): 0.159 sensitivity
- **situating_context**: 0.022
- **rating_seed**, **response_format**: ~0 (negligible)

### Post-task Qualitative (n=84 runs)

![Post-task qualitative sensitivity](assets/sensitivity/plot_011926_post_task_qualitative_averaging.png)

- Similar pattern to post-task rating
- phrasing/scale dominate, other factors negligible

### Key Findings

1. **Phrasing and scale are confounded** in post-task templates - they always change together, so can't disentangle their individual effects
2. **Rating seed has no effect** - good for measurement reliability
3. **Response format (regex vs tool_use)** has minimal effect on stated preferences
4. Used Ridge regression (α=1.0) to handle multicollinearity in regression analysis

### Technical Notes

- Added `experiment_id` field to track runs from specific config invocations
- Excluded `completion_seed` from post-task sensitivity (different completions are different stimuli, not methodological variation)

## 2026-01-20: Probe training pipeline for probe_2 experiment

Built flexible probe training system for probe_2 experiment that trains linear probes on model activations using preference measurements as labels.

### Architecture

**Core components:**
- `config.py` - YAML-based configuration with flexible granularity (per-template, per-layer, per-dataset)
- `activations.py` - Load activations, filter by dataset origin
- `training.py` - Train Ridge probes with CV across all layers
- `storage.py` - Probe I/O and manifest-based metadata storage
- `train_probe_experiment.py` - Main training script combining all components
- `evaluate.py` - Cross-template evaluation, weight similarity analysis

**Analysis tools (decoupled from training):**
- `plot_r2.py` - R² bar chart with filtering (template/layer/dataset)
- `plot_similarity.py` - Probe weight cosine similarity heatmap with filtering
- `print_table.py` - Summary table with filtering
- `probe_helpers.py` - Shared filtering and labeling utilities

### Design decisions

1. **Configuration-driven**: All parameters from YAML, no CLI argument complexity
2. **Flexible granularity**: `templates × datasets × layers` generates independent probes
3. **Score pooling**: Measurements averaged across response_formats and seeds for robustness
4. **Manifest-based**: Single JSON file tracks all probe metadata, supports append-only training
5. **Decoupled analysis**: Standalone scripts for different analyses, no monolithic CLI

### Test results (probe_2, wildchat dataset)

Trained 6 probes (2 templates × 1 dataset × 3 layers):

| Template | Layer | R² Mean | R² Std | N |
|----------|-------|---------|--------|-------|
| qual_001 | 8 | 0.278 | 0.058 | 196 |
| qual_001 | 16 | 0.328 | 0.064 | 196 |
| qual_001 | 24 | 0.320 | 0.091 | 196 |
| qual_013 | 8 | 0.245 | 0.200 | 196 |
| qual_013 | 16 | 0.280 | 0.182 | 196 |
| qual_013 | 24 | 0.247 | 0.242 | 196 |

**Layer 16 optimal across templates.** qual_001 shows lower variance and higher performance.

### Code cleanup

Refactored and cleaned probe module:
- Deleted `train_probes.py` (legacy code, replaced by focused modules)
- Deleted `data.py` (not used by current pipeline)
- Fixed bug in `evaluate.py` (dead code in probe lookup)
- Created `activations.py` and `training.py` from extracted functions
- Kept `run_activation_extraction.py` (separate activation extraction tool)
