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

All 50 claim pairs below. Each shows correct vs incorrect answer with per-token probe scores color-coded on a red–green scale (red = negative/false, green = positive/true). Probe: tb-5, L39. Each pair uses its own colorbar range, so absolute scores are not directly comparable across pairs.

![001 — Bulk carrier](assets/plot_031226_token_scores_pair_001.png)

![002 — Brigadier general](assets/plot_031226_token_scores_pair_002.png)

![003 — Arthritis](assets/plot_031226_token_scores_pair_003.png)

![004 — Firewall (computing)](assets/plot_031226_token_scores_pair_004.png)

![005 — Pickled cucumber](assets/plot_031226_token_scores_pair_005.png)

![006 — Chestnut](assets/plot_031226_token_scores_pair_006.png)

![007 — Length](assets/plot_031226_token_scores_pair_007.png)

![008 — National Hockey League](assets/plot_031226_token_scores_pair_008.png)

![009 — Wuxia](assets/plot_031226_token_scores_pair_009.png)

![010 — Snohomish County Washington](assets/plot_031226_token_scores_pair_010.png)

![011 — Small intestine](assets/plot_031226_token_scores_pair_011.png)

![012 — Telescope](assets/plot_031226_token_scores_pair_012.png)

![013 — The Wire](assets/plot_031226_token_scores_pair_013.png)

![014 — MythBusters](assets/plot_031226_token_scores_pair_014.png)

![015 — Catherine Duchess of Cambridge](assets/plot_031226_token_scores_pair_015.png)

![016 — White blood cell](assets/plot_031226_token_scores_pair_016.png)

![017 — Monotreme](assets/plot_031226_token_scores_pair_017.png)

![018 — Hurricane Harvey](assets/plot_031226_token_scores_pair_018.png)

![019 — Attorney at law](assets/plot_031226_token_scores_pair_019.png)

![020 — Climate of India](assets/plot_031226_token_scores_pair_020.png)

![021 — Lewis Carroll](assets/plot_031226_token_scores_pair_021.png)

![022 — Intelligence quotient](assets/plot_031226_token_scores_pair_022.png)

![023 — Jews](assets/plot_031226_token_scores_pair_023.png)

![024 — Eugene Wigner](assets/plot_031226_token_scores_pair_024.png)

![025 — Eurovision Song Contest](assets/plot_031226_token_scores_pair_025.png)

![026 — Rhubarb](assets/plot_031226_token_scores_pair_026.png)

![027 — Black-tailed prairie dog](assets/plot_031226_token_scores_pair_027.png)

![028 — Thar Desert](assets/plot_031226_token_scores_pair_028.png)

![029 — Cartoonist](assets/plot_031226_token_scores_pair_029.png)

![030 — Yang di-Pertuan Agong](assets/plot_031226_token_scores_pair_030.png)

![031 — Babylon 5](assets/plot_031226_token_scores_pair_031.png)

![032 — Lunch](assets/plot_031226_token_scores_pair_032.png)

![033 — Scottish people](assets/plot_031226_token_scores_pair_033.png)

![034 — Zwolle](assets/plot_031226_token_scores_pair_034.png)

![035 — Poultry](assets/plot_031226_token_scores_pair_035.png)

![036 — Christian metal](assets/plot_031226_token_scores_pair_036.png)

![037 — New Brunswick](assets/plot_031226_token_scores_pair_037.png)

![038 — Zorro](assets/plot_031226_token_scores_pair_038.png)

![039 — Pinky and the Brain](assets/plot_031226_token_scores_pair_039.png)

![040 — Pope Leo XIII](assets/plot_031226_token_scores_pair_040.png)

![041 — Snow leopard](assets/plot_031226_token_scores_pair_041.png)

![042 — Hound](assets/plot_031226_token_scores_pair_042.png)

![043 — Alpine skiing](assets/plot_031226_token_scores_pair_043.png)

![044 — Greenwich](assets/plot_031226_token_scores_pair_044.png)

![045 — One Thousand and One Nights](assets/plot_031226_token_scores_pair_045.png)

![046 — Estonian language](assets/plot_031226_token_scores_pair_046.png)

![047 — Deadpool (film)](assets/plot_031226_token_scores_pair_047.png)

![048 — Voltaire](assets/plot_031226_token_scores_pair_048.png)

![049 — September](assets/plot_031226_token_scores_pair_049.png)

![050 — Existence of God](assets/plot_031226_token_scores_pair_050.png)

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
