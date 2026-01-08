# Thurstonian model Tuning

There are a bunch of hyperparams, starting conditions, regularisation that we can play with in the Thurstonian model. This is mean to be a quick exploration of how to choose them.

What do we care about?
- Make sure that the utilities we find are not super contigent on any given choice
- Goodness of fit, particularly on validation set of comparisons (for "seen" tasks)

What can we tweak? (roughly in order)
- L-BFGS-B hyperparams, like bounds on mu and sigma, and initial values of sigma.
- Regularisation on sigma. Daniel Paleka claimed that it lead to better fits.

On what data:
- KEEP FIXED: the dataset (wildchat)
- VARY: n_tasks
- VARY: synthetic v real data

Main questions:
1. Are utilities robust to BFGS initial conditions on the real data?
2. How much should we regularise?
3. What are the most salient differences between synthetic and real data, what does that tell us?

## L-BFGS-B hyperparams

Sequential tuning of optimization parameters, varying one at a time.

### 1. mu_bounds

Varied: ±5, ±10, ±20, ±50, ±1000 (with log_sigma_bounds fixed at ±3)
ß
Results (n=24 prompt templates):
- Rankings stable across configs (mean Spearman ρ = 0.978)
- Tighter bounds (±5) occasionally hit constraints
- Very wide bounds (±1000) slower convergence

**Choice: ±10**

### 2. log_sigma_bounds

Varied: ±1, ±2, ±3, ±5, ±10 (with mu_bounds fixed at ±10)

Results (n=24 prompt templates):
- Rankings stable across configs (mean Spearman ρ = 0.981)
- Tighter bounds converge faster
- ±2 has slightly higher correlations with other configs

**Choice: ±2**

### 3. sigma_init

Varied: 0.1, 0.5, 1.0, 2.0, 5.0 (with mu_bounds=±10, log_sigma_bounds=±2)

Results (n=24 prompt templates):
- Rankings fairly stable (mean Spearman ρ = 0.951)
- Higher init values converge faster in median but with wider variance
- σ_init=1.0 has slightly better correlations and converges well

**Choice: 1.0**

## Sigma regularisation

What kind? Gaussion prior on sigma, penalise deviation from 0.

Train v validation splits: take N tasks, divide the comparisons (and not the tasks) into train and test. So the validation set contains unseen comparisons between seen tasks.

### Synthetic data

Regularisation seems to help, with test NLL decreasing. Already a pretty strong case to regularise given this data is from the actual distribution.


