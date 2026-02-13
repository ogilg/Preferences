# Layer Sweep: Running Log

## Step 0: Setup
- Branch: research-loop/layer_sweep
- Read parent report (coefficient_calibration): negative result, L31 ridge probe no causal effect
- Confirmed BT probes same format as ridge (5377 elements, last is intercept=0.0)
- load_probe_direction works for both probe types

## Step 1: Compute activation norms

Ran 20 forward passes through Gemma-3-27B:

| Layer | Mean L2 norm | Std | 6% coef | 2% coef |
|-------|-------------|-----|---------|---------|
| 31 | 51,159 | 3,357 | 3,070 | 1,023 |
| 37 | 61,530 | 3,226 | 3,692 | 1,231 |
| 43 | 65,508 | 6,379 | 3,930 | 1,310 |
| 49 | 77,050 | 6,277 | 4,623 | 1,541 |
| 55 | 91,739 | 6,022 | 5,504 | 1,835 |

Norms increase ~80% from L31 to L55. Coefficients properly scaled.

## Step 2: Pilot (L43, ridge+bt, 2 prompts, +6%, seed 0)

Both probes produce very similar responses to unsteered at +6% perturbation.
Ridge almost identical, BT slightly different wording.
~6-9s per generation. Pipeline validated.

## Step 3: Full generation sweep

2400 trials in 25,636s (7.1 hours, 10.7s/trial average).
Rate: 0.1/s throughout. 4 layers x 2 probes x 20 prompts x 5 coefs x 3 seeds.
All completed without errors. Saved to generation_results.json.

## Step 4: Judge

GPT-5-nano via OpenRouter. 2400 coherence + 2400 valence requests.
Results: 1794/2400 coherence scored (75%), 1669/2400 valence scored (70%).
Parse failures uniform (~25%) across all conditions — judge artifact, not steering-related.
Mean coherence: 4.57. Mean valence: 0.18.

## Step 5: Analysis

All 8 layer-probe combinations: Spearman |rho| < 0.15, no p < 0.05.
Max |rho| = 0.132 (L55 ridge, F_affect) — expected from 16 null tests.
Coherence uniformly high (4.4-4.7). No degradation at +/-6% norm.
Ridge vs BT: no difference (mean |rho| 0.035 vs 0.036).

Strong negative result. Preference probe directions do not causally shift valence
at any layer L31-L55, with either ridge or BT probes, using all_tokens steering.

3 plots generated: valence grid, coherence grid, heatmap.
Report written.
