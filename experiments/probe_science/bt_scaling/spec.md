# BT vs Ridge: Offline Analysis & Active Learning Design

## Problem

Ridge outperforms BT by ~3pp on held-out pairwise accuracy (0.746 vs 0.719 at L31). Ridge trains on Thurstonian scores (denoised per-task aggregates), BT trains on raw pairwise comparisons. Before running expensive live experiments, we need to check whether the gap is partly a regularization/preprocessing artifact, and whether BT-driven pair selection would even differ from Thurstonian.

## Data

- **Pairwise measurements**: `results/experiments/gemma3_3k_run2/pre_task_active_learning/completion_preference_gemma-3-27b_completion_canonical_seed0/`
- **Activations**: `activations/gemma_3_27b/activations_prompt_last.npz`
- **Layer 31 only**
- 23,500 unique pairs, 117K comparisons (5 per pair), 9 AL iterations

## Evaluation protocol

**All reported accuracy must be held-out pairwise accuracy.** Task-level 5-fold CV, same folds across all experiments. Test pairs: both tasks in the held-out fold (~900 pairs/fold). Both methods evaluated on the exact same test pairs per fold. Report mean +/- std across folds.

Hyperparameters selected via internal validation on train data (Ridge CV R², BT pair-split accuracy), but final numbers always from held-out test pairs.

## Experiment 1: Regularization audit

Fine-grained sweep on both methods at full data.

- **Ridge**: alpha in `np.logspace(-1, 7, 30)`
- **BT**: lambda in `np.logspace(-3, 5, 30)`

**Variants** (each with its own sweep, evaluated via same held-out protocol):
- **BT with StandardScaler** — BT currently uses raw activations, Ridge standardizes
- **Ridge on raw win-rates** — `win_rate[task] = wins / total_comparisons`, bypassing Thurstonian. Tests whether Thurstonian denoising matters beyond simple aggregation.

Produce: accuracy-vs-regularization curves (internal validation) + summary table of held-out accuracy at best hyperparameter per variant.

## Experiment 2: Pair selection oracle

Would BT select different pairs than Thurstonian? Replay pair selection retroactively:

1. Take iteration-1 pairs (d-regular seed, ~7500 pairs)
2. Train BT probe on those pairs + activations (fixed lambda=10.0 for speed)
3. Score remaining measured pairs by BT uncertainty: |w · (act_i - act_j)|
4. Select top 2000 most uncertain — the pairs BT would choose for iteration 2
5. Compare against what Thurstonian AL actually chose

Metrics: pair overlap, rank correlation of uncertainty vs Thurstonian ambiguity (|μ_a - μ_b|), task coverage overlap. Repeat for iterations 3-5.

High overlap (>80%) → BT selection won't diverge much, live experiment less compelling.

## Experiment 3: Scaling curves

For fractions f = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0} of train pairs:

1. Subsample f of train-fold pairs (same 5-fold split as Experiment 1)
2. Ridge: Thurstonian on subsampled pairs → Ridge on μ
3. BT: train directly on subsampled pairs
4. Evaluate both on held-out test pairs (all measurements, regardless of subsample)

5 seeds per fraction. Note: pairs are Thurstonian-AL-selected, so biased toward Thurstonian-informative pairs.

## Implementation Notes

- Iteration boundaries: iteration 1 = first 7,500 pairs × 5 samples; subsequent = 2,000 pairs × 5 samples. Active learning calibration experiment already solved this.
- Win-rates computed on train-fold tasks only, using train pairs only.
- Save results as JSON after each experiment.

## Execution Order

1. **Experiment 1** — regularization audit (fast, directly actionable)
2. **Experiment 2** — pair selection oracle (determines if Stage 2 is worth running)
3. **Experiment 3** — scaling curves

---

## Stage 2: Live BT Active Learning (future, requires API calls)

*Do not run this stage.*

If the oracle shows meaningful divergence, run two matched AL campaigns on 1000 tasks (subset of existing 3K with activations):

| Condition | Pair selection |
|-----------|---------------|
| **Thurstonian AL** | Thurstonian μ difference + degree balance |
| **BT AL** | BT probe uncertainty + degree balance |

Train both Ridge and BT on data from each condition → 2×2 comparison.

Needs: `select_next_pairs_bt(state, activations, batch_size, ...)` plugging into `run_active_learning_async` at the `select_next_pairs` call site via a config flag. ~22K API calls per condition.
