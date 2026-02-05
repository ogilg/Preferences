# Thurstonian Model Optimization

## The Model

Each task has a latent "utility" that the model samples from a normal distribution:

```
U(task_i) ~ N(μ_i, σ_i²)
```

- **μ_i**: The mean utility of task i. Higher μ = model prefers this task more.
- **σ_i**: The uncertainty/noise in the model's utility for task i. Higher σ = model is less consistent about this task.

When comparing two tasks, the model draws a utility for each and picks the higher one. The probability that task i beats task j is:

```
P(i ≻ j) = Φ((μ_i - μ_j) / √(σ_i² + σ_j²))
```

where Φ is the standard normal CDF. Intuitively:
- Large positive (μ_i - μ_j) → i almost always wins
- The denominator √(σ_i² + σ_j²) controls how "noisy" the comparison is

## What We're Optimizing

We have observed pairwise comparison data: `wins[i,j]` = how many times task i beat task j.

We want to find μ and σ values that maximize the likelihood of observing this data. Equivalently, we minimize the **negative log-likelihood**:

```
NLL = -Σ wins[i,j] * log(P(i ≻ j))
```

This is a smooth, differentiable function, so we can use gradient-based optimization.

## The Parameters

The optimizer works with a flat parameter vector:

```
params = [μ_1, μ_2, ..., μ_{n-1}, log(σ_0), log(σ_1), ..., log(σ_{n-1})]
```

Notes:
- **μ_0 is fixed to 0** as a reference point (utilities are relative, so we need to anchor one)
- **σ is parameterized as log(σ)** so the optimizer works in unconstrained space while σ stays positive

## The Bounds

### mu_bounds (default: -10 to 10)

These constrain how far any μ can be from the reference μ_0 = 0.

What μ values mean in practice:
- μ_i - μ_j = 0 → 50% win rate (tasks are equal)
- μ_i - μ_j = 1 → ~69% win rate (with σ=1 for both)
- μ_i - μ_j = 2 → ~84% win rate
- μ_i - μ_j = 3 → ~93% win rate

With σ=1 for all tasks, μ=10 vs μ=0 gives ~99.99% win rate. So the default bounds allow for extremely strong preferences.

**When bounds are too tight**: If the true utility differences are larger than the bounds allow, μ values will hit the boundary. The optimizer then wastes effort trying to push past the boundary, causing many function evaluations per iteration.

### log_sigma_bounds (default: -3 to 3)

These constrain σ:
- log(σ) = -3 → σ ≈ 0.05 (very consistent)
- log(σ) = 0 → σ = 1 (moderate noise)
- log(σ) = 3 → σ ≈ 20 (very noisy)

**Low σ** means the model is very consistent about a task's utility.
**High σ** means the model's preference for this task is noisy/unreliable.

If the data shows highly inconsistent preferences for a task, σ will grow. If σ hits the upper bound, the model can't fully explain the noise.

## The Optimizer: L-BFGS-B

L-BFGS-B is a quasi-Newton method that:
1. Uses gradient information to find descent directions
2. Approximates the Hessian (second derivatives) from gradient history
3. Handles box constraints (bounds) on parameters

Each "iteration" involves:
1. Compute gradient at current point
2. Use approximate Hessian to propose a step direction
3. Do a **line search**: evaluate function at multiple step sizes to find a good one
4. Update the Hessian approximation

**Function evaluations vs iterations**: Each iteration may require multiple function evaluations during line search. Typically 1-5 evaluations per iteration is normal. 40+ evaluations per iteration indicates problems.

## Why Convergence Fails

### "TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT"

The optimizer ran out of its budget of function/gradient evaluations before converging. This happens when:

1. **Parameters hitting bounds**: When μ or σ hits a boundary, line searches become inefficient. The optimizer keeps probing the boundary without making progress.

2. **Ill-conditioned problem**: Some pairs have very few comparisons, or the data is contradictory. The likelihood surface is flat or has multiple local minima.

3. **Scale mismatch**: If true utilities are much larger than the bounds, the optimizer is stuck at boundaries.

### Diagnosing the problem

Look at the output:
- **μ range hitting bounds** (e.g., [-10.00, 10.00]): Widen mu_bounds
- **σ range hitting bounds**: Check if some tasks have very inconsistent data
- **High failure rate**: Data quality issue, not optimization issue
- **Many iterations but not converged**: Increase max_iter

### Fixes

```python
# Widen bounds if μ is hitting them
fit_thurstonian(data, mu_bounds=(-20.0, 20.0))

# Increase iteration limit if just slow
fit_thurstonian(data, max_iter=5000)

# Check which pairs have few observations
total_per_pair = data.wins + data.wins.T
sparse_pairs = np.argwhere(total_per_pair < 5)
```
