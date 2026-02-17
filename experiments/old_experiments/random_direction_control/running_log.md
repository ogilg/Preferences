# Random Direction Control: Running Log

## Step 0: Setup
- Branch: research-loop/random_direction_control
- Read parent reports: calibration (L31 null), layer sweep (L37-L55 null)
- L31 mean norm: 51,159 → 6% coef = 3,070
- 7 directions: ridge_L31, bt_L31, random_0..4
- 1,260 total generations

## Step 1: Pilot (12 trials)
- 2 directions (ridge_L31 + random_100), 2 prompts, 3 coefs, seed=0
- Cosine(ridge, random_100) = 0.003 — essentially orthogonal
- All responses coherent at ±6%
- Pipeline validated: probe loading, random direction gen, steering hooks work
- Proceeding to full generation
