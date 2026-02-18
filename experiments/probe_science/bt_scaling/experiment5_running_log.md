# Experiment 5: Running Log

## Setup

- Reconstructing AL iteration order from measurement file order (first-seen index per pair)
- Iteration 0: 7500 pairs (d-regular init), iterations 1-8: 2000 pairs each
- Using BT+StandardScaler (λ=0.193) and Ridge+Thurstonian (α=1374) from Experiment 1
- 5-fold CV, 3 random seeds for random conditions

## Run 1: Full experiment

Completed successfully. Key findings:

### 5a: AL-order vs Random subsampling
- AL-order leads at small fractions: +1.9pp BT at 5%, +6.4pp Ridge at 5%
- Gap narrows: at 30%, essentially equal (BT 72.9 vs 72.8, Ridge 70.0 vs 69.8)
- At 80%, practically identical (BT 74.1 vs 74.0, Ridge 74.0 vs 73.9)
- At 100%, identical by construction

### 5b: Marginal value at different base sizes
- K=5000: AL-next +0.5pp BT / +1.4pp Ridge, random +0.3pp / +1.3pp — similar
- K=7500: AL-next +0.3pp / +1.8pp, random +0.4pp / +1.5pp — similar
- K=10000: AL-next +1.0pp / +1.4pp, random +0.4pp / +1.3pp — AL slightly better
- K=15000: AL-next **-0.3pp** / -0.2pp, random +0.0pp / -0.3pp — AL hurts at margin!
- K=20000: AL-next -0.1pp / +0.3pp, random +0.1pp / -0.1pp — both near zero

Pattern: AL pairs are valuable early (good coverage from d-regular init), but become counterproductive at the margin when the probe is already well-trained. Random selection is never much worse and avoids the "noise injection" problem at the margin.
