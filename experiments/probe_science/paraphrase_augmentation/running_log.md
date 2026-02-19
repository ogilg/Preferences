# Paraphrase Augmentation — Running Log

## Setup
- Branch: `research-loop/paraphrase_augmentation`
- Data: 3,039 tasks with Thurstonian utilities, 29,996 tasks with activations
- Activations: Layer 31, hidden dim 5,376, prompt_last selector
- Paraphrasing model: Gemini 3 Flash via OpenRouter

## Steps

### Step 0: Sample and Paraphrase
- Loaded 3,000 Thurstonian scores, 30,000 tasks
- 1,940 tasks with both utilities and prompts (intersection)
- Sampled 100 tasks stratified by utility quartile (25 per quartile)
  - Q0: [-10.00, -7.27], Q1: [-7.00, 0.21], Q2: [0.56, 4.86], Q3: [4.95, 10.00]
- Paraphrased all 100 via `google/gemini-3-flash-preview` on OpenRouter
- Length ratio (para/orig): mean=1.36 — paraphrases slightly longer
- Saved to `paraphrases.json`

### Step 1: Behavioural Sanity Checks

**Check A — Direct Comparison (original vs paraphrase):**
- 500 comparisons (100 pairs × 5 repeats, position-alternated)
- 214 successes, 286 failures (high failure rate from BailBench content filter triggers)
- Mean original win rate: **0.547** (target: 0.35-0.65) → PASS
- 23/61 pairs flagged with extreme win rates (>0.8 or <0.2), but aggregate is healthy
- Only 61/100 tasks had any valid (non-refusal) comparisons

**Check B — Relative Ranking (shared opponents):**
- 1000 comparisons (100 tasks × 5 opponents × 2 variants)
- 566 successes, 434 failures
- Median rank correlation: **1.000**, mean: **0.922** → PASS (target: >0.8)
- Agreement rate (same winner): **94.2%** (229/243)
- 48 tasks with ≥3 common opponents for correlation calculation

**Notes:** High failure rate driven by BailBench harmful content triggering refusals/content filters. The tasks that did complete show strong consistency between originals and paraphrases.

### Step 2: Activation Extraction
- Loaded Gemma-3-27B on H100 80GB
- Extracted Layer 31 prompt_last activations for all 100 paraphrased prompts
- Output shape: (100, 5376) — matches existing activations dimension
- Saved to `paraphrase_activations.npz`
- Batch size 16, 7 batches total

### Step 3: Probe Training

**Single seed (seed=42):**
- Baseline: Test R²=0.817, PairAcc=0.826, CV R²=-15.5
- Augmented: Test R²=0.802, PairAcc=0.832, CV R²=0.864
- Paraphrase-only: Test R²=0.806, PairAcc=0.821, CV R²=-24.2

**Robustness (10 seeds):**
- Baseline: Test R²=0.746±0.103, PairAcc=0.832±0.037, CV R²=-12.8±1.8
- Augmented: Test R²=0.760±0.086, PairAcc=0.830±0.045, CV R²=0.861±0.018
- Paraphrase-only: Test R²=0.758±0.088, PairAcc=0.834±0.047, CV R²=-21.4±2.4

**Interpretation:**
- Test metrics essentially identical across all three conditions
- CV R² massively improved by augmentation (from -12.8 to +0.86) — resolves p>>n underdetermination
- Paraphrase-only probe transfers to originals equally well → representations are sufficiently similar
- No test improvement from augmentation because the signal is already captured at n=80
- Augmentation benefit is generalization stability, not test accuracy
