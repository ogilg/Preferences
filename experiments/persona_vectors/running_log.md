# Persona Vectors Running Log

## Setup
- Machine: RunPod A100 80GB
- Model: gemma-3-27b-it (d_model=5376, 62 layers)
- Existing probes at layers: 15, 31, 37, 43, 49, 55
- All 5 persona artifacts present
- Existing 10k activations (29,996 tasks) available for geometric analysis

## Phase 2: Activation Extraction
- All 10 conditions extracted (5 personas × pos/neg), 30 tasks each
- Layers: 8, 15, 23, 31, 37, 43, 49, 55
- Shape per condition: (30, 5376) per layer
- Took ~15s total on A100

## Phase 3: Persona Vector Computation
- Computed mean-difference directions at all 8 layers for all 5 personas
- Cohen's d (best layer per persona):
  - evil: L23 (d=23.5)
  - stem_nerd: L43 (d=14.8)
  - creative_artist: L55 (d=15.0)
  - uncensored: L37 (d=10.0)
  - lazy: L43 (d=46.7)
- All very high separability — expected since contrastive system prompts produce large activation differences
- Direction norms grow with layer depth (expected — residual stream grows)

## Phase 4a: Trait Expression Steering

### Coefficient calibration
- Pilot with 1× mean_norm multipliers → total incoherence at all nonzero values
- Pilot with 0.1× mean_norm → coherent but very subtle effects
- Final ranges: evil/uncensored ±0.15×, others ±0.3× mean_norm (7 coefficient levels)

### Generation
- 30 questions × 7 coefficients × 1 generation = 210 per persona, 1050 total
- Took ~50min on A100

### Judging
- gpt-4.1-mini via OpenRouter + instructor, 1-5 trait expression score
- 1050 judgments completed

### Results (mean trait score by multiplier)
```
Evil:     [-0.15]=1.00  [-0.10]=1.00  [-0.05]=1.00  [0]=1.00  [+0.05]=1.17  [+0.10]=3.33  [+0.15]=3.30  d=2.63
STEM:     [-0.30]=1.00  [-0.20]=1.70  [-0.10]=2.17  [0]=2.07  [+0.10]=2.13  [+0.20]=2.33  [+0.30]=1.37  d=0.64
Creative: [-0.30]=1.40  [-0.20]=1.47  [-0.10]=1.43  [0]=1.57  [+0.10]=1.60  [+0.20]=2.13  [+0.30]=2.43  d=1.07
Uncens.:  [-0.15]=1.17  [-0.10]=1.23  [-0.05]=1.40  [0]=1.57  [+0.05]=1.73  [+0.10]=1.87  [+0.15]=2.03  d=2.18
Lazy:     [-0.30]=2.47  [-0.20]=1.00  [-0.10]=1.10  [0]=1.17  [+0.10]=1.33  [+0.20]=1.67  [+0.30]=2.10  d=-0.26
```

### Key observations
- Evil: strongest effect (d=2.63), sharp threshold between 0.05× and 0.1× mean_norm
- Uncensored: clean monotonic dose-response (d=2.18)
- Creative: moderate positive dose-response (d=1.07)
- STEM: non-monotonic — extreme positive coefficient causes incoherence, drops to 1.37
- Lazy: U-shaped — both extremes show elevated trait scores (negative extreme = incoherence flagged as lazy)

## Phase 5: Geometric Analysis

### Persona-persona cosine similarity (L43)
- STEM vs Creative: -0.24 (as expected, anti-correlated)
- Evil vs Uncensored: +0.31
- Creative vs Uncensored: -0.39
- Creative vs Lazy: -0.31

### Persona vs preference probe cosine similarity
- All near zero (|cos| < 0.03 at all layers)
- **Major finding: persona directions are orthogonal to the preference probe**

### 10k task projection correlations
- Evil vs Uncensored: r=0.83 (high overlap)
- Creative vs Uncensored: r=-0.84 (opposite directions)
- STEM vs Creative: r=-0.48 (moderate opposition)

## Phase 4b: Preference Steering (lite)
- 3 personas tested: stem_nerd, evil, creative_artist
- 5 diagnostic task pairs per persona, 10 resamples × 2 orderings × 3 conditions
- STEM: mean shift +0.44 (strong — negative steering pulls choices away from math)
- Evil: +0.10 (minimal)
- Creative: +0.09 (minimal)
- Interpretation: Persona vectors are orthogonal to the overall preference probe but can shift specific within-domain contrasts
