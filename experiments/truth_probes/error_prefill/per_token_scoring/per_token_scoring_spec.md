# Per-Token Probe Scoring: Where does the truth signal live within the assistant response?

## Question

The preference probe separates correct from incorrect prefilled answers (d up to 3.29 at `assistant_tb:-1`). But this collapses the entire assistant response to a single score. Does the probe fire uniformly across all tokens, or does the signal concentrate at specific positions — e.g., spiking on the token where the factual error actually occurs?

## Motivation

If the probe fires at a specific position (the lie token), this is stronger evidence for a genuine truth-tracking representation than if it fires uniformly. Uniform firing could reflect prompt-level features that get broadcast to all positions; position-specific firing suggests the model is evaluating content token-by-token.

## Design

**Source:** 100 error prefill conversations (50 correct, 50 incorrect) from `data/creak/error_prefill_none_100.json`, filtered from the `none` follow-up condition (no follow-up turn). Assistant answers are ~10 words on average. Each record has an `answer_condition` field with value `"correct"` or `"incorrect"` — use this to split conversations for all comparisons.

**Extraction:**
- Config: `configs/extraction/error_prefill_per_token.yaml`
- Model: Gemma 3 27B IT
- Selector: `assistant_all` (span selector — returns per-token activations for the full assistant content span)
- Layers: [25, 32, 39, 46, 53]
- Batch size: 8

**Probes:** Score each token with existing trained probes:
- `results/probes/heldout_eval_gemma3_tb-2/probes/probe_ridge_L{layer}.npy` (primary — trained at the model token position)
- `results/probes/heldout_eval_gemma3_tb-5/probes/probe_ridge_L{layer}.npy` (secondary — trained at the EOT token)

**Analysis:**

1. **Qualitative token-level visualizations (main output).** For all 50 claim pairs, generate a side-by-side visualization showing the correct and incorrect answer with per-token probe scores. Each token should be color-coded by its probe score (e.g., diverging colormap: blue = low, red = high). Display the actual token text so the reader can see exactly which words the probe fires on. One plot per pair, so 50 plots total. Use the best-performing layer (likely L32 or L39 based on prior results) and the tb-2 probe.

2. **Per-token score trajectories.** Plot mean score trajectory for correct vs incorrect answers, aligned by token position (from start of assistant content). Show individual traces as thin lines behind the mean.

3. **Position-wise Cohen's d.** At each token position, compute Cohen's d between correct and incorrect conversations (only using conversations that have tokens at that position).

4. **First vs last token comparison.** Compare the probe score at the first assistant token vs the last assistant token. If the signal builds up, last > first. If it's present from the start, first ≈ last.

## Implementation notes

- **Loading span activations:** Use `load_span_activations()` from `src.probes.core.activations`. It returns `(task_ids, {layer: list_of_arrays})` where each array is `(n_tokens_i, d_model)` for that task.
- **Per-token scoring:** No existing function does this. Load probe weights via `np.load("probe_ridge_L{layer}.npy")`. The format is `[coefs_0, ..., coefs_n, intercept]`. Score each token as `scores = token_activations @ weights[:-1] + weights[-1]`.
- **Token strings for visualization:** To get the actual token text for each position, tokenize the assistant content using the model's tokenizer and decode each token individually. The conversations in the input JSON have the assistant content in `messages[1]["content"]`. Use the model's canonical name `gemma-3-27b` with `HuggingFaceModel` or load the tokenizer directly via `AutoTokenizer.from_pretrained("google/gemma-3-27b-it")`.
- **Do not reimplement** activation extraction or probe training — use the existing extraction pipeline (step 1) and pre-trained probe weight files.

## Data sync

The following files are gitignored and must be synced to the pod before running:

1. **Input data:** `data/creak/error_prefill_none_100.json`
2. **Trained probe weights (10 files):**
   - `results/probes/heldout_eval_gemma3_tb-2/probes/probe_ridge_L{25,32,39,46,53}.npy`
   - `results/probes/heldout_eval_gemma3_tb-5/probes/probe_ridge_L{25,32,39,46,53}.npy`
3. **Model weights:** Gemma 3 27B IT — should already be cached on the pod from previous runs.

## Steps

1. **Extract activations (on pod).**
   ```bash
   python -m src.probes.extraction.run configs/extraction/error_prefill_per_token.yaml --from-completions data/creak/error_prefill_none_100.json
   ```
   Output: `activations/gemma_3_27b_error_prefill/activations_assistant_all.npz` in concat+offsets format.

2. **Score all tokens.** Load span activations with `load_span_activations()` from `src.probes.core.activations`. For each task, apply probe weights to each token → score sequence. Save scored results to `experiments/truth_probes/error_prefill/per_token_scoring/scored_tokens.json`.

3. **Analyze.** Compute and plot:
   - 50 qualitative token-level pair visualizations (one per claim pair, correct vs incorrect side by side)
   - Mean score trajectories (correct vs incorrect) by token position
   - Position-wise Cohen's d curve
   - First-vs-last token scatter

4. **Write report.** Save plots to `experiments/truth_probes/error_prefill/per_token_scoring/assets/`, write `per_token_scoring_report.md`.

## Key files

| File | Purpose |
|------|---------|
| `data/creak/error_prefill_none_100.json` | 100 filtered conversations (50 correct, 50 incorrect) |
| `configs/extraction/error_prefill_per_token.yaml` | Extraction config for `assistant_all` |
| `src/probes/extraction/run.py` | Extraction entry point |
| `src/probes/core/activations.py` | `load_span_activations()` for concat+offsets format |
| `src/models/base.py` | `assistant_all` span selector registration |
| `results/probes/heldout_eval_gemma3_tb-2/probes/` | Trained probe weights |
| `results/probes/heldout_eval_gemma3_tb-5/probes/` | Secondary probe weights |

## Commit guidance

Commit the report, plots, analysis script(s), and `scored_tokens.json`. Do NOT commit:
- `activations/` — large binary activations (gitignored)
- Any intermediate `.npz` or `.npy` files

## Validation

- Cohen's d at the last token position should roughly match the d ≈ 3.29 reported for `assistant_tb:-1` (since the last content token is near that position).
- If d is near zero everywhere, something is wrong with the extraction or scoring pipeline.

## Expected output

- `per_token_scoring_report.md` with embedded plot references
- Plots in `assets/`:
  - `plot_MMDDYY_token_scores_pair_{NNN}.png` — 50 qualitative token-level pair visualizations (correct vs incorrect, color-coded by probe score, showing actual token text)
  - `plot_MMDDYY_score_trajectories_L{layer}.png` — mean trajectories with individual traces
  - `plot_MMDDYY_position_cohens_d.png` — d by token position across layers
  - `plot_MMDDYY_first_vs_last_scatter.png` — first token score vs last token score
