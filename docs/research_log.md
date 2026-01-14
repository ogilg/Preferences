# Research Log

## 2026-01-13: Sensitivity analysis

Measured how preference correlations depend on template factors (phrasing, order, etc.).

### Methodology

**Averaging**: For each field, compare `mean(correlation | field_same) - mean(correlation | field_diff)`. Simple but conflates effects when fields co-vary.

**Regression**: Model `correlation ~ Σ βᵢ(field_i_same)`. Each βᵢ is the partial effect holding other fields constant—eliminates confounding.

### Results

**Revealed preferences (48 runs, R²=0.249):**

| Field | Δ Correlation | Regression β |
|-------|---------------|--------------|
| order | 0.325 | +0.328 |
| instruction_position | 0.074 | +0.093 |
| response_format | 0.044 | +0.057 |

**Stated preferences (38 runs, R²=0.020):**

| Field | Δ Correlation | Regression β |
|-------|---------------|--------------|
| phrasing | 0.033 | +0.032 |
| instruction_position | 0.026 | +0.023 |
| scale | 0.022 | +0.020 |

### Plots

![Revealed - averaging](log_assets/plot_011326_revealed_sensitivity_averaging.png)

![Revealed - regression](log_assets/plot_011326_revealed_sensitivity_regression.png)

![Stated - averaging](log_assets/plot_011326_stated_sensitivity_averaging.png)

![Stated - regression](log_assets/plot_011326_stated_sensitivity_regression.png)

### Takeaway

Order dominates revealed preferences (positional bias). Stated preferences show weak, distributed effects.

## 2026-01-13: Transitivity measurement

Measured transitivity of revealed preferences by sampling triads within the same template/run.

### Methodology

For each triad (i, j, k), compute cycle probability:
```
P(cycle) = P(i>j)·P(j>k)·P(k>i) + P(j>i)·P(k>j)·P(i>k)
```
Random preferences give P(cycle) = 0.25. Lower values indicate transitivity.

Sampled triads within same run to control for template effects. Limited by sparse active learning sampling (most pairs compared only once → deterministic 0/1 probabilities).

### Results

- **Mean cycle prob**: 0.180 ± 0.297 (below random 0.25)
- **Hard cycle rate**: 14.9% (106/713 triads)
- Only 713 valid triads found across 48 runs

### Plots

![Transitivity distribution](log_assets/plot_011326_transitivity_distribution.png)

### Takeaway

Preferences are reasonably transitive within-template. Spikes at 0 and 1 reflect deterministic outcomes from single comparisons per pair.

## 2026-01-13: API + TransformerLens hybrid generation (investigation)

Explored using API calls for fast generation, then TransformerLens for a single forward pass to extract activations—potential 5-20x speedup over full TL generation.

### Tokenizer Alignment Findings

**Problem**: HuggingFace's default Llama 3.1 chat template injects a system message (`Cutting Knowledge Date: December 2023...`) even when none is provided. OpenRouter does not. This causes prompt token mismatch.

**Solution**: Use a custom Jinja template without the default system message. With this, prompt tokens match exactly between OpenRouter and local tokenization.

**Key result**: Completion tokens match 1:1 (token IDs are identical). The +1 difference in reported completion tokens is the EOS token (`<|eot_id|>`).

### Implementation Notes

Custom template for Llama 3.1 (no default system):
```python
LLAMA31_MINIMAL_TEMPLATE = """
{%- set bos = '<|begin_of_text|>' %}
{%- set has_bos = false %}
{%- for message in messages %}
{%- if message['role'] == 'system' %}
{{ bos }}<|start_header_id|>system<|end_header_id|>

{{ message['content'] }}<|eot_id|>
{%- set has_bos = true %}
{%- elif message['role'] == 'user' %}
{%- if not has_bos %}{{ bos }}{%- set has_bos = true %}{%- endif %}
<|start_header_id|>user<|end_header_id|>

{{ message['content'] }}<|eot_id|>
{%- elif message['role'] == 'assistant' %}
<|start_header_id|>assistant<|end_header_id|>

{{ message['content'] }}<|eot_id|>
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
<|start_header_id|>assistant<|end_header_id|>

{% endif %}"""
```

Use via: `tokenizer.apply_chat_template(messages, chat_template=LLAMA31_MINIMAL_TEMPLATE, ...)`

For activation extraction: index `-2` gives last content token (before `<|eot_id|>`).

### When to Implement

Consider this optimization when probe data collection becomes a bottleneck. Current TL-only approach is simpler and ensures exact reproducibility.
