# Layer Sweep: Steering at Later Layers with Ridge and BT Probes

## Goal

Test whether steering at layers closer to the model output (L37, L43, L49, L55) produces behavioral effects that L31 did not. Also test whether Bradley-Terry (BT) probe directions are more causally effective than ridge directions.

## Background

From the coefficient calibration experiment (`experiments/steering_program/coefficient_calibration/report.md`):

- The L31 ridge probe direction (CV R² = 0.846) does **not** causally shift expressed valence, stated preferences, or pairwise choices at any coefficient in the coherent range [-5000, +5000].
- Coherence degrades at |coef| >= 10000 (19% of L31 activation L2 norm). The mechanism works — steering does affect the residual stream — but the L31 ridge direction does not encode a causally relevant dimension for preference behavior.
- Recommended safe coefficient range: [-3000, +3000] (~5.7% of L31 norm).

Hypotheses for why L31 failed:
1. **Wrong layer**: L31 is layer 31/62 (middle). Later layers may be closer to where preference-relevant computations become "decision-ready."
2. **Wrong probe type**: Ridge probes predict scalar mu values; BT probes predict pairwise choice outcomes. BT may capture a more behaviorally relevant direction.

## Available probes

From `results/probes/gemma3_3k_nostd_raw/manifest.json`:

| Layer | Ridge CV R² | BT CV accuracy |
|-------|------------|----------------|
| L15 | 0.698 | 72.3% |
| L31 | 0.846 | 76.9% |
| L37 | 0.763 | 77.0% |
| L43 | 0.731 | 76.9% |
| L49 | 0.685 | 77.0% |
| L55 | 0.651 | 76.7% |

Ridge predictive power peaks at L31 and declines. BT accuracy is remarkably flat across L31-L55 (~77%), suggesting the pairwise-choice signal is equally present at all these layers.

## Design

### Norm calibration

Coefficients must be calibrated to each layer's activation norm. The calibration experiment used coef=3000 at L31 (~5.7% of mean L2 norm = 52,823). We target the same perturbation fraction at each layer.

**Step 0**: Compute mean L2 activation norm at each test layer from `activations/gemma_3_27b/activations_prompt_last.npz`. Derive per-layer coefficients that correspond to 1%, 2%, 4%, 6% of mean norm (4 levels). This replaces arbitrary coefficient values with norm-matched perturbation fractions.

### Layer and probe combinations

- **Layers**: L37, L43, L49, L55 (skip L15 — worst predictor, too early; skip L31 — already tested)
- **Probes**: Ridge and BT at each layer
- **Total conditions**: 4 layers x 2 probe types = 8 probe-layer combinations

### Coefficient grid

5 levels per combination: [-6%, -2%, 0%, +2%, +6%] of layer norm. This covers the range from clearly sub-threshold to the edge of the coherent range.

### Prompt battery (reduced)

Use only categories D (valence) and F (affect) — the two categories with valence judging, which are the cleanest tests of causal influence on expressed affect. 20 prompts total.

### Seeds and budget

3 seeds per prompt x coefficient. Total: 8 conditions x 20 prompts x 5 coefs x 3 seeds = **2,400 generations**.

At ~2s per generation with model reload per layer pair, budget is ~5,000s + model load time.

### Implementation

Two scripts in `scripts/layer_sweep/`:

1. **`compute_norms.py`**: Load activations NPZ, compute mean L2 norm per layer (L37, L43, L49, L55). Output a JSON with norm values and derived coefficients.

2. **`generate.py`**: For each layer-probe combination:
   - Load the probe direction (ridge or BT) from manifest
   - For each prompt x coefficient x seed: generate with steering
   - Save to `experiments/steering_program/layer_sweep/generation_results.json`

   Note: BT probes store weight vectors differently from ridge probes. Ridge stores [coef_0, ..., coef_n, intercept] — `load_probe_direction` already handles this. BT probes may need separate handling — check the file format before running.

3. **`judge.py`**: Same as calibration judge (coherence + valence via GPT-5-nano).

4. **`analyze.py`**: Generate plots:
   - Valence vs perturbation fraction, faceted by layer, colored by probe type
   - Coherence vs perturbation fraction, faceted by layer
   - Heatmap: layer x probe_type, showing Spearman rho of valence-coefficient correlation

### Decision rule

The experiment succeeds if **any** layer-probe combination shows:
- Valence dose-response with Spearman |rho| > 0.3 (relaxed from 0.5 since we have fewer coefficients) and p < 0.05

The experiment produces an informative negative result if:
- No layer-probe combination shows |rho| > 0.2 — strengthening the "correlate not cause" interpretation across all available layers and both probe types

### Data sources

- **Activations**: `activations/gemma_3_27b/activations_prompt_last.npz`
- **Probes**: `results/probes/gemma3_3k_nostd_raw/` with IDs `ridge_L{37,43,49,55}` and `bt_L{37,43,49,55}`
- **Prompts**: Same D (valence) and F (affect) prompts from calibration experiment
