# Cross-Selector Probe Transfer

## Question

Do probes trained at one turn-boundary token position transfer to other positions? A tb-2 probe (trained on `model` token activations) has been applied to every token in a conversation and works well. But we've never quantified how each of the 5 turn-boundary probes performs when evaluated on each of the 6 extraction positions (5 turn-boundary + task_mean).

## Design

Evaluate every (probe, extraction position) pair and report Pearson r and pairwise accuracy for predicting Thurstonian preference scores.

- **Probes:** 5 trained probes (tb-1 through tb-5), each at 5 layers (25, 32, 39, 46, 53). From `results/probes/heldout_eval_gemma3_tb-{1..5}/probes/probe_ridge_L{layer}.npy`.
- **Activations:** 6 extraction positions (tb-1 through tb-5 + task_mean) from `activations/gemma_3_27b_turn_boundary_sweep/activations_{selector}.npz`.
- **Scores:** Thurstonian preference scores from the heldout eval run.

This is a 5 × 6 matrix (probe × extraction position) at each layer. Report the best layer per cell.

## Output

- A 5 × 6 heatmap of best-layer Pearson r and a second heatmap for pairwise accuracy.
- A table with the full numbers.
- Brief note on whether the diagonal (matched position) dominates or whether probes transfer freely.

## All data is local

No extraction or GPU needed. Everything runs locally on existing files.
