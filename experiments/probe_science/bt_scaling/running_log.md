# BT Scaling — Running Log

## Setup

- Branch: research-loop/bt_scaling
- Data: 3K task run, 23.5K pairs, 117K comparisons, Layer 31
- Scripts: scripts/bt_scaling/

## Step 1: Data loading validation

Validated via `scripts/bt_scaling/validate_data.py`. All good:
- 117,435 measurements, 3,000 tasks, 23,500 unique pairs
- 5-fold CV: ~14,987-15,060 train pairs, ~907-960 test pairs per fold
- Thurstonian full-data accuracy: 0.8626
- Win-rate full-data accuracy: 0.7507

## Step 2: Experiment 1 — Regularization audit

HP selection on fold 0. Ridge alphas: logspace(-1, 7, 30). BT lambdas: logspace(-3, 5, 15).

### HP sweep results (fold 0):
- Ridge Thurstonian: best alpha = 1.37e+03, CV R² selected
- Ridge win-rate: best alpha = 9.24e+03
- BT standard: best lambda = 1.39e+02, val_acc = 0.7605
- BT scaled: best lambda = 1.93e-01, val_acc = 0.7627

### 5-fold results:

| Variant | Mean ± Std | HP |
|---------|------------|-----|
| BT scaled | 0.7425 ± 0.0113 | λ=0.193 |
| Ridge Thurstonian | 0.7390 ± 0.0107 | α=1374 |
| BT standard | 0.7371 ± 0.0097 | λ=139 |
| Ridge win-rate | 0.7213 ± 0.0083 | α=9237 |

Key finding: BT with StandardScaler slightly outperforms Ridge on Thurstonian.
The original Ridge advantage (~3pp) was partly a standardization artifact.
With standardization, BT is competitive or better.

Note: BT standard vs fair comparison report (0.7371 vs 0.719) — the difference is that here BT uses λ=139 (heavily regularized, selected on proper fold-0 sweep) vs λ=10 in the fair comparison. Better HP selection helps BT a lot.

## Step 3: Experiment 2 — Pair selection oracle

Replay pair selection for iterations 2-5. BT trained at λ=10 on available pairs, selects top-2000 most uncertain (lowest |logit|).

| Iteration | Pair overlap | Rank corr | Task coverage overlap |
|-----------|-------------|-----------|----------------------|
| 2 | 12.3% | 0.240 | 56.0% |
| 3 | 14.0% | 0.335 | 52.3% |
| 4 | 17.1% | 0.420 | 50.8% |
| 5 | 21.8% | 0.463 | 51.6% |

Very low overlap — BT would select substantially different pairs than Thurstonian.
Rank correlations positive but weak. Both methods agree more as data accumulates (overlap increases from 12% to 22%), but never converge.
Task coverage overlap ~50% suggests BT focuses on different tasks entirely.

## Step 4: Experiment 3 — Scaling curves (first attempt, buggy)

First run used `gradient_tol=10.0` in Thurstonian fit for speed. This caused all mu values to be 0.0 (converged at iteration 0 — the gradient was already below threshold at initialization). Ridge accuracy was constant at 0.532 regardless of fraction or seed.

Fixed by removing aggressive tolerance settings. Rerunning with default Thurstonian parameters (max_iter=300).

## Step 5: Experiment 3 — Scaling curves (corrected)

Fractions: {0.1, 0.2, 0.3, 0.5, 0.8, 1.0}, 3 seeds each, 5-fold CV.
HPs from Experiment 1: Ridge α=1374, BT λ=139, BT_scaled λ=0.193.

| Fraction | Ridge | BT | BT scaled |
|----------|-------|-----|-----------|
| 0.1 | 0.6229±0.0042 | 0.6973±0.0050 | 0.7090±0.0033 |
| 0.2 | 0.6736±0.0032 | 0.7076±0.0054 | 0.7204±0.0008 |
| 0.3 | 0.6986±0.0026 | 0.7176±0.0039 | 0.7296±0.0017 |
| 0.5 | 0.7258±0.0041 | 0.7252±0.0010 | 0.7362±0.0038 |
| 0.8 | 0.7374±0.0042 | 0.7346±0.0005 | 0.7405±0.0006 |
| 1.0 | 0.7390±0.0000 | 0.7372±0.0000 | 0.7427±0.0000 |

Key findings:
- BT scaled dominates at all fractions
- Ridge suffers most at low data (0.623 vs 0.709 at 10%) — 8.6pp gap
- Gap narrows with more data: at 100% only 0.4pp between BT scaled and Ridge
- BT standard crosses Ridge at ~50% data
- At fraction=1.0 all seeds identical (no subsampling = deterministic)
