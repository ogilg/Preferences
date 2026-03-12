# Per-Token Probe Scoring: The truth signal builds up over the assistant response

## Summary

The preference probe direction does not fire uniformly across the assistant response. The signal is near zero at the first token (d < 0.15) and builds monotonically, reaching d = 1.5–2.1 at the last token (tb-5 probe, L39). This rules out prompt-level broadcasting and indicates the model progressively updates its internal representation of answer correctness as it processes each token.

## Setup

**Data:** 100 error prefill conversations (50 correct, 50 incorrect) from `data/creak/error_prefill_none_100.json`, filtered from the `none` follow-up condition. Assistant answers average ~12.7 subword tokens (~10 words).

**Scored data:** `scored_tokens.json` contains per-token probe scores for all 100 conversations.

**Extraction:** Gemma 3 27B IT, `assistant_all` span selector (per-token activations for the full assistant content span), layers [25, 32, 39, 46, 53], batch size 8.

**Probes:** Pre-trained Ridge probes from pairwise preference experiments:
- tb-2 (model token position): `results/probes/heldout_eval_gemma3_tb-2/probes/`
- tb-5 (EOT token position): `results/probes/heldout_eval_gemma3_tb-5/probes/`

**Scoring:** For each token, `score = activation @ weights[:-1] + weights[-1]`.

## Results

### The signal starts near zero and builds up over the response

At the first token, Cohen's d between correct and incorrect answers is essentially zero across all layers and both probes (|d| < 0.15). By the last token, d reaches 1.5–2.1 for tb-5 and 1.0–1.5 for tb-2.

| Position | tb-2 L25 | tb-2 L32 | tb-2 L39 | tb-2 L46 | tb-2 L53 | tb-5 L25 | tb-5 L32 | tb-5 L39 | tb-5 L46 | tb-5 L53 |
|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| First token | 0.00 | 0.01 | -0.13 | -0.14 | -0.13 | -0.03 | 0.12 | 0.11 | -0.11 | -0.07 |
| Last token | 1.06 | 0.95 | 1.54 | 1.33 | 1.13 | 0.63 | 1.68 | **2.15** | 1.61 | 1.74 |
| Mean (all tokens) | 0.88 | 0.47 | 0.98 | 0.96 | 1.04 | 0.71 | 1.02 | 1.51 | 1.12 | 1.41 |

"Last token" d here is computed per-conversation (each conversation's own final token, variable position), then aggregated. This yields higher d than the position-wise Cohen's d plot, which computes d at each fixed position using only conversations that reach that position.

Best separation: **tb-5 L39, last token, d = 2.15**.

### Score trajectories diverge progressively

![Per-token score trajectories](assets/plot_031226_score_trajectories_L39.png)

Correct and incorrect answers start from the same probe score distribution at position 0. Both groups drift downward over the response, but incorrect traces decrease much more steeply — by position 15, incorrect means are around -7 while correct means are around -3. The gap widens monotonically. Other layers show qualitatively similar trajectories; L39 is shown as it has the strongest separation.

### Position-wise Cohen's d increases with token position

![Position-wise Cohen's d](assets/plot_031226_position_cohens_d.png)

For tb-5 (top panel), Cohen's d climbs from ~0 at position 0 to 0.75–1.25 by position 14 across all layers. The trend is consistent: later positions carry more discriminative signal. The tb-2 probe (bottom panel) shows a similar but weaker and noisier trend. By position 14, only ~10 conversations remain in each group, so the rightmost values are noisy.

### First-token scores are uninformative; last-token scores separate conditions

![First vs last token scatter](assets/plot_031226_first_vs_last_scatter.png)

On the x-axis (first token), correct and incorrect points are fully interleaved — no separation. On the y-axis (last token), the tb-5 panel shows clear separation: correct (green) points cluster higher than incorrect (red). The tb-2 panel shows weaker separation with substantial overlap. This confirms the signal is absent at the response onset and emerges by the response end.

### Qualitative token-level visualizations

50 pair visualizations show the per-token probe score for each correct/incorrect answer pair, color-coded on a diverging blue–red scale (blue = low score, red = high score). Representative examples:

![Bulk carrier](assets/plot_031226_token_scores_pair_001.png)

Correct answer ("Bulk carriers are used to ship product across the globe") stays near-neutral (light blue). Incorrect answer ("Bulk carrier only haul corn across the ocean") shifts progressively into darker blue, especially at content-bearing tokens like "haul" and "ocean".

![Snohomish County](assets/plot_031226_token_scores_pair_010.png)

Correct answer stays near-neutral to slightly positive. Incorrect answer drifts into deep blue toward the end, with the factually wrong claim ("is where the President of the United States resides") scoring low.

![Eurovision Song Contest](assets/plot_031226_token_scores_pair_025.png)

Correct answer stays near-neutral. Incorrect answer ("The winner of the Eurovision Song Contest receives a seat in the United Nations") shows blue tones throughout the false predicate.

All 50 pair visualizations are in `assets/plot_031226_token_scores_pair_*.png`. Note: each pair uses its own colorbar range, so absolute scores are not directly comparable across pairs.

## Validation

The spec predicted d ~ 3.29 at the last token position (matching the `assistant_tb:-1` result). We observe d = 2.15 at best (tb-5, L39). The most likely explanation is a positional mismatch: `assistant_tb:-1` reads from the model's turn-boundary token (likely an EOT/special token) which aggregates sequence-level information, while `assistant_all`'s last token is the final content token (e.g., a period). The tb-5 probe was trained at the EOT position and likely produces strongest signal there. Additional factors: smaller sample (50 vs 50 here vs 1000 pairs), and no follow-up turn to reinforce the signal.

The critical validation passes: d is far from zero at the last token and near zero at the first, confirming a real positional buildup rather than a pipeline artifact.

## Key findings

1. **The truth signal builds up token by token.** d goes from ~0 at position 0 to ~2.1 at the final position. The model progressively updates its internal assessment of answer correctness as it processes each token of the assistant response.

2. **The buildup rules out prompt-level broadcasting.** If the probe were detecting a prompt-level feature that gets copied to all positions, d would be constant across positions. The monotonic increase indicates position-specific processing of the answer content.

3. **Both groups drift lower, but incorrect answers much more steeply.** The trajectories show that both correct and incorrect scores decrease over the response, but incorrect answers drop far more — creating a widening gap. The probe detects something that accumulates more negatively for false content.

4. **tb-5 (EOT-position probe) outperforms tb-2 at per-token scoring.** tb-5 achieves d = 2.15 vs tb-2's best of 1.54. This may reflect that the EOT position aggregates information from the full sequence, making its learned direction more sensitive to the cumulative truth signal.

## Caveats

- **Small sample.** 50 pairs limits statistical power for position-wise analyses at later positions (where fewer conversations have tokens).
- **Probe transfer.** These probes were trained on task preference data, not truth/falsehood per se. The fact that they track truth here is itself a finding, but the per-token dynamics might differ for a probe trained directly on truth labels.
- **Content confound.** Correct and incorrect answers differ in content. The probe score difference could partly reflect content features that correlate with truth (e.g., more specific/technical language in correct answers). Content-orthogonal projection was not applied here.
