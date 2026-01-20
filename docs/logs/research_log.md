# Research Log

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
