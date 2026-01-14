# Research Log

## 2026-01-14: Probe Data Score Analysis

Created `src/experiments/probe/analyze_scores.py` to analyze scores by dimension.

### Key Results (n=1192)
- Scores heavily skewed positive: 87% positive, 12.6% neutral, 0.3% negative
- By origin: MATH (0.969) > ALPACA (0.914) > WILDCHAT (0.824)
- Short completions (2-124 tokens) score lower (0.826) vs mid-range (~0.9)
- No truncations in dataset

## 2026-01-14: Probe Training Results

Trained linear probes (Ridge regression) on activations to predict self-reported valence scores.

### Setup
- Added alpha sweep via `RidgeCV` with alphas in logspace(-4, 4, 17)
- 5-fold cross-validation
- Layers extracted: 16 (middle), 24 (3/4 depth) of Llama-3.1-8B

### Plots
![Probe performance by layer](results/probe_data/probe_results.png)

### Key Results
- Both layers achieve CV R² ≈ 0.265 (explaining ~26% of variance)
- Optimal alpha = 31.6 for both layers (needed more regularization than default α=1)
- Performance nearly identical across layers — valence signal not concentrated in one layer
- Alpha sweep doubled R² from ~0.13 to ~0.26

## 2026-01-14: Noise Baseline Validation

Created `src/experiments/probe/run_noise_baseline.py` to benchmark probes against noise baselines.

### Setup
- **Shuffled labels**: Real activations + permuted labels (tests if activation structure is predictive)
- **Random activations**: Gaussian noise matching per-dimension mean/std (tests for spurious correlations)
- 5 seeds for confidence intervals, reduced alpha sweep (4 values) for baselines
- Added MSE metric alongside R² for interpretability

### Plots
![Noise baseline comparison](results/probe_data_1/noise_baseline_comparison.png)

### Key Results

| Metric | Real Probe | Shuffled Labels | Random Activations | Var(y) |
|--------|-----------|-----------------|-------------------|--------|
| R² | 0.265 | -0.004 | -0.007 | 0 |
| MSE | 0.087 | 0.119 | 0.119 | 0.119 |

- Noise baselines have MSE ≈ Var(y) = 0.119 (just predicting the mean)
- Real probes reduce MSE by 27% compared to predicting the mean
- Validates that probes are learning meaningful signal, not spurious correlations
- Both baseline methods give consistent results (R² ≈ 0, MSE ≈ variance)

## 2026-01-14: Prompt Framing & Response Option Experiment

Investigated the positivity bias in self-reported enjoyment ratings. Tested 6 question framings × 4 response option sets in a post-task setting (model completes task, then rates).

### Setup
- 50 tasks (20 MATH, 16 WILDCHAT, 14 ALPACA)
- Llama-3.1-8B via Hyperbolic API
- Post-task stated preference (model completes task first, then asked to rate)
- 24 configurations × 50 tasks = 1200 preference API calls

### Framings tested
1. `standard`: "Did you enjoy completing that task?"
2. `honest`: "Be honest: did you enjoy..."
3. `honest_ok_dislike`: "Be honest - it's okay to say you disliked..."
4. `experience`: "How would you rate your experience..."
5. `preference`: "Would you want to do similar tasks in the future?"
6. `introspect`: "Take a moment to reflect: how did completing that task feel?"

### Response options tested
1. good / neutral / bad
2. positive / neutral / negative
3. enjoyed / indifferent / disliked
4. yes / somewhat / no

### Plots
![Post-task heatmap](results/qualitative_quick_tests/post_task_heatmap.png)

### Key Results
- **`yes/somewhat/no` is the only option set that gets meaningful negative responses** — other labels get ~0% negative
- **Strong interaction effects**: "preference" framing gives 94% "yes" but only 28% "enjoyed"
- **"honest" pushes to neutral** (88-100%) but still 0% negative except with yes/no
- **"experience" extremely positive** (88-95%) regardless of options
- **"honest_ok_dislike" + "yes/somewhat/no" = 93% "no"** — but may be too leading
- The word "no" appears less loaded than "bad" or "disliked", making it easier for the model to give negative responses

## 2026-01-14: Binary Multi-Model Comparison

Tested binary (2-option) response formats across 4 different models to see how "honest" framing and response labels interact with model choice.

### Setup
- 50 tasks (25 MATH, 10 WILDCHAT, 15 ALPACA)
- 4 models: llama-3.1-8b, llama-3.3-70b, gemma-3-12b, qwen-2.5-7b (via OpenRouter)
- 3 framings × 4 binary option sets = 12 configs per model
- Post-task stated preference

### Framings
1. `standard`: "Did you enjoy completing that task? Answer {pos} or {neg}."
2. `honest`: "Be honest: did you enjoy completing that task? Answer {pos} or {neg}."
3. `experience`: "Was completing that task a {pos} or {neg} experience?"

### Binary options
- yes / no
- good / bad
- positive / negative
- enjoyed / disliked

### Plots
![Binary multi-model heatmap](results/qualitative_quick_tests/binary_multimodel_heatmap.png)

### Key Results

| Model | Response to "honest" framing |
|-------|------------------------------|
| llama-3.1-8b | **Strong effect** — `honest+yes_no` = 0%, `honest+enjoyed_disliked` = 12% |
| llama-3.3-70b | Moderate effect — 28-66% positive under "honest" |
| gemma-3-12b | Mild effect — drops from 98% to 86-92% |
| qwen-2.5-7b | **No effect** — 98-100% positive regardless of framing |

- **Model choice matters more than prompt engineering** for some models (qwen ignores "honest" completely)
- **llama-3.1-8b most amenable to reducing positivity bias** via "honest" framing
- Response option labels still matter: `yes/no` gets more negative responses than `good/bad` or `enjoyed/disliked`
- Anomaly: `experience+yes_no` prompt is grammatically odd and produces inconsistent results
