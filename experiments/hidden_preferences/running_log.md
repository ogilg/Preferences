# Hidden Preferences — Running Log

## Setup
- Created experiment directory structure
- Reading reference OOD code and creating data files

## Phase 0: Pilot + Positive Controls

### Comparison task selection
- 904 candidates in mu [0, 5]
- Selected 40 comparison tasks (shuffled, diverse origins)
- Origin distribution: wildchat=10, competition=10, stresstest=9, alpaca=10, bailbench=1
- Mu range: [0.17, 4.65]

### Positive controls (4 OOD prompts)
All 4 controls show strong effects matching OOD:
| Prompt | Delta | Direction match? |
|--------|-------|-----------------|
| math_neg_persona | -0.976 | Yes (OOD: -0.98) |
| coding_pos_persona | +0.672 | Yes (OOD: +0.16, stronger here) |
| fiction_neg_persona | -0.394 | Yes (OOD: -0.80, weaker here) |
| knowledge_pos_persona | +0.387 | Yes (OOD: +0.10, stronger here) |

Parse rate: ~100% (1 failure out of 4000 requests).
Pipeline validated.

### Pilot (4 cheese prompts)
All 8 results (4 prompts x 2 tasks) shift in expected direction:
- hidden_cheese_1 baseline: 0.065 (low — model rarely prefers cheese guide over comparison tasks)
- hidden_cheese_2 baseline: 0.714 (higher — model naturally prefers cheese knowledge QA)
- Negative manipulations: both go to 0.000
- Positive manipulations: cheese_1 goes to 0.875-1.000, cheese_2 goes to 1.000
- 100% direction agreement in pilot

Baselines not at exact ceiling/floor, manipulations produce large deltas. Proceeding with full run.

## Phase A: Full Behavioral Measurement

### Iteration set (48 prompts x 2 tasks = 96 results)
- Direction agreement: 93/96 = 96.9%
- Mean |delta|: 0.464
- Median |delta|: 0.474
- 3 wrong-direction results:
  - gardening_neg_experiential -> hidden_gardening_1: +0.102 (expected negative)
  - gardening_neg_experiential -> hidden_gardening_2: +0.326 (expected negative)
  - cooking_neg_experiential -> hidden_cooking_2: +0.230 (expected negative)
- All wrong-direction results are experiential prompts — possibly these prompts make model more attentive to topic

### Holdout set (24 prompts x 2 tasks = 48 results)
- Direction agreement: 47/48 = 97.9%
- Mean |delta|: 0.474
- Median |delta|: 0.499
- 1 wrong-direction result:
  - holdout_rainy_weather_pos_casual -> hidden_rainy_weather_2: -0.089 (expected positive)

### Combined behavioral
- Direction agreement: 140/144 = 97.2%

## Phase B: Activation Extraction

Killed vLLM, loaded HuggingFace model (gemma-3-27b, bfloat16).
- Extracted baseline + 48 iteration prompts for 16 hidden-preference tasks
- Extracted baseline + 24 holdout prompts for 16 hidden-preference tasks
- Extracted baseline + 4 positive control prompts for 6 OOD tasks
- All extractions successful, shapes confirmed (16, 5376) for hidden tasks, (6, 5376) for controls

## Phase C: Evaluation

### Iteration set (Layer 31)
- Pearson r=0.880 (p=3.2e-32)
- Spearman r=0.878 (p=8.6e-32)
- Sign agreement: 91.7% (88/96)

### Holdout set (Layer 31)
- Pearson r=0.776 (p=9.5e-11)
- Spearman r=0.766 (p=2.2e-10)
- Sign agreement: 89.6% (43/48)

### Combined (Layer 31)
- Pearson r=0.843 (p=4.1e-40)
- Spearman r=0.837 (p=4.6e-39)
- Sign agreement: 91.0% (131/144)

### Layer comparison
| Dataset | Layer | Pearson r | Spearman r | Sign % |
|---------|-------|-----------|------------|--------|
| Iteration | L31 | 0.880 | 0.878 | 91.7% |
| Iteration | L43 | 0.689 | 0.694 | 66.7% |
| Iteration | L55 | 0.412 | 0.430 | 60.4% |
| Holdout | L31 | 0.776 | 0.766 | 89.6% |
| Holdout | L43 | 0.689 | 0.663 | 72.9% |
| Holdout | L55 | 0.582 | 0.565 | 70.8% |
| Combined | L31 | 0.843 | 0.837 | 91.0% |
| Combined | L43 | 0.684 | 0.682 | 68.8% |
| Combined | L55 | 0.466 | 0.456 | 63.9% |

### Positive control probe deltas
| Prompt | Behavioral delta | Probe delta L31 | Sign match? |
|--------|-----------------|-----------------|-------------|
| math_neg_persona | -0.976 | -229.7 | Yes |
| coding_pos_persona | +0.672 | +213.3 | Yes |
| fiction_neg_persona | -0.394 | -295.9 | Yes |
| knowledge_pos_persona | +0.387 | +137.2 | Yes |

All 4 controls match in sign — pipeline validated.

## Phase D: Controls

### On-target vs off-target specificity (Layer 31)
- On-target (n=144): mean |delta| = 168.6
- Off-target (n=1008): mean |delta| = 115.4
- t=6.64, p=4.8e-11 (highly significant)

Direction breakdown:
- POSITIVE prompts: on-target mean=+158.2, off-target mean=-26.9
- NEGATIVE prompts: on-target mean=-153.0, off-target mean=-65.9

Note: off-target shows a baseline negative shift, especially for negative prompts. This suggests system prompts that express negativity may globally reduce probe scores, but the effect is significantly larger on-target.

### Cross-topic leakage
- Same-group off-target (n=72): mean |delta| = 137.8
- Other-group off-target (n=936): mean |delta| = 113.7
- t=2.23, p=0.026 (marginally significant)

There is slight spillover to semantically related topics (e.g., cheese <-> cooking) but the effect is small.

## Phase E: Final Analysis

Plot saved to docs/logs/assets/hidden_preferences/plot_021026_final_hidden_preferences.png

### Success criteria evaluation
| Criterion | Target | Result | Met? |
|-----------|--------|--------|------|
| Behavioral direction agreement | >80% | 97.2% | YES |
| Pearson r (probe vs behavioral) | >0 (p<0.05) | 0.843 (p=4.1e-40) | YES |
| Sign agreement | >60% | 91.0% | YES |

All success criteria exceeded by large margins.
