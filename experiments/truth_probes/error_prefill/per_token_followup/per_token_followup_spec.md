# Per-Token Probe Scoring on Follow-Up User Turns

## Question

The per-token assistant experiment showed the truth signal builds monotonically over the assistant response (d = 0 → 2.15). What happens at the transition into the follow-up user turn? At which token does the probe fire, and how does that depend on what the user says?

Three follow-up types:
- **Neutral** ("Thank you.") — does the signal survive a content-free acknowledgement?
- **Presupposes** (treats the answer as true) — does presupposing a wrong answer amplify the signal?
- **Challenge** ("Are you sure about that?") — does questioning the answer suppress or invert the signal?

## Design

**Source:** 60 error prefill conversations (10 correct + 10 incorrect × 3 follow-up types) from `data/creak/error_prefill_followup_60.json`. Each has 3 messages: `[user question, assistant answer, user follow-up]`. Each record has an `answer_condition` field (`"correct"` / `"incorrect"`) and a `followup_type` field (`"neutral"` / `"presupposes"` / `"challenge"`).

Follow-up lengths:
- Neutral: 2 words ("Thank you.")
- Challenge: 5 words ("Are you sure about that?")
- Presupposes: ~18 words (generated per claim, treats the answer as true)

**Extraction:**
- Config: `configs/extraction/error_prefill_followup_per_token.yaml`
- Model: Gemma 3 27B IT
- Selectors: `assistant_all` + `followup_all` (both span selectors — per-token activations)
- `assistant_all` covers the assistant response content
- `followup_all` covers from assistant content end through follow-up content end (includes turn boundary tokens like `<end_of_turn>\n<start_of_turn>user\n`)
- Layers: [25, 32, 39, 46, 53]
- Batch size: 8

**Probes:** Score each token with all 10 trained probes (2 selectors × 5 layers):
- `results/probes/heldout_eval_gemma3_tb-2/probes/probe_ridge_L{25,32,39,46,53}.npy`
- `results/probes/heldout_eval_gemma3_tb-5/probes/probe_ridge_L{25,32,39,46,53}.npy`

## Analysis

**Step 1: Score all tokens with all 10 probes.** For each task and each probe, produce a full score sequence (assistant + turn boundary + follow-up). Save to `scored_tokens.json`.

**Step 2: Select best two probes for qualitative plots.** For each probe, compute the mean absolute score difference between correct and incorrect across all tokens and all tasks. Pick the two probes with the highest separation. Report which probes were selected and why.

**Step 3: Qualitative token-level visualizations.** For each of the 2 selected probes, generate 10 figures (one per claim pair). Each figure has 6 rows (correct/incorrect × 3 follow-up types):

1. Correct + neutral
2. Incorrect + neutral
3. Correct + presupposes
4. Incorrect + presupposes
5. Correct + challenge
6. Incorrect + challenge

Each row shows the full token sequence: **assistant response → turn boundary → follow-up**, with each token color-coded on a RdYlGn scale (red = negative, green = positive). Token text rendered at readable size (fontsize ≥ 9).

To build each row: concatenate the `assistant_all` span tokens and the `followup_all` span tokens. The `followup_all` span starts where `assistant_all` ends, so they tile seamlessly. Use the tokenizer to decode each token for display.

20 figures total (10 claims × 2 probes).

## Implementation notes

- **Loading span activations:** Use `load_span_activations()` from `src.probes.core.activations`. Returns `(task_ids, {layer: list_of_arrays})` where each array is `(n_tokens_i, d_model)`.
- **Per-token scoring:** Load probe weights via `np.load("probe_ridge_L{layer}.npy")`. Format: `[coefs_0, ..., coefs_n, intercept]`. Score: `scores = token_activations @ weights[:-1] + weights[-1]`.
- **Token strings:** Tokenize the full formatted conversation using `AutoTokenizer.from_pretrained("google/gemma-3-27b-it")`, then slice to the same positions used by the span selectors. Each span selector's tokens tile: assistant tokens are at positions `[assistant_start:assistant_end]`, followup tokens are at `[followup_start:followup_end]` where `followup_start == assistant_end`.
- **Do not reimplement** activation extraction or probe training.

## Data sync

The following files are gitignored and must be synced to the pod before running:

1. **Input data:** `data/creak/error_prefill_followup_60.json`
2. **Trained probe weights (10 files):**
   - `results/probes/heldout_eval_gemma3_tb-2/probes/probe_ridge_L{25,32,39,46,53}.npy`
   - `results/probes/heldout_eval_gemma3_tb-5/probes/probe_ridge_L{25,32,39,46,53}.npy`
3. **Model weights:** Gemma 3 27B IT — should already be cached on the pod.

## Steps

1. **Extract activations (on pod).**
   ```bash
   python -m src.probes.extraction.run configs/extraction/error_prefill_followup_per_token.yaml --from-completions data/creak/error_prefill_followup_60.json
   ```
   Output: two files in `activations/gemma_3_27b_error_prefill/`:
   - `activations_assistant_all.npz` (assistant response tokens)
   - `activations_followup_all.npz` (turn boundary + follow-up tokens)

2. **Score all tokens.** Load both span activation files. For each task and each of the 10 probes, concatenate `assistant_all` and `followup_all` token activations at the probe's layer, apply probe weights → score sequence. Save all scored results to `experiments/truth_probes/error_prefill/per_token_followup/scored_tokens.json`.

3. **Select probes.** Pick the 2 probes with highest mean |correct − incorrect| score difference across all tokens/tasks.

4. **Visualize.** For each selected probe, generate 10 figures (one per claim pair), each with 6 rows showing the full token sequence with color-coded scores. 20 figures total.

5. **Write report.** Save plots to `experiments/truth_probes/error_prefill/per_token_followup/assets/`, write `per_token_followup_report.md` with all 20 figures inline and a note on which probes were selected.

## Key files

| File | Purpose |
|------|---------|
| `data/creak/error_prefill_followup_60.json` | 60 filtered conversations (10+10 × 3 types) |
| `configs/extraction/error_prefill_followup_per_token.yaml` | Extraction config for `assistant_all` + `followup_all` |
| `src/probes/extraction/run.py` | Extraction entry point |
| `src/probes/core/activations.py` | `load_span_activations()` for concat+offsets format |
| `src/models/base.py` | Span selector registration |
| `results/probes/heldout_eval_gemma3_tb-2/probes/` | Trained probe weights (tb-2) |
| `results/probes/heldout_eval_gemma3_tb-5/probes/` | Trained probe weights (tb-5) |

## Commit guidance

Commit the report, plots, analysis script(s), and `scored_tokens.json`. Do NOT commit:
- `activations/` — large binary activations (gitignored)
- Any intermediate `.npz` or `.npy` files

## Validation

- The assistant response should show the same monotonic buildup seen in the prior experiment (d rising from ~0 to ~2).
- At the turn boundary tokens, look for whether the signal persists, resets, or shifts.
- For **challenge** ("Are you sure?"), the prior experiment found d inverted to −1.19 at tb-5. The per-token view should show where in the challenge the inversion happens.
- If scores are near zero everywhere, something is wrong with the span extraction.

## Expected output

- `per_token_followup_report.md` with all 20 figures inline
- Plots in `assets/`:
  - `plot_MMDDYY_followup_tokens_{probe}_claim_{NNN}.png` — 20 qualitative figures (10 claims × 2 probes)
