# Weekly Report: Feb 19 - 26, 2026

## Experiment summaries

Below is a one-sentence summary of each experiment run during the project, organized by theme.

### Measurement

- **Temperature calibration** — Tested temperatures 0.3–1.3 for preference measurement quality; T=0.7 is optimal (tightest Thurstonian sigma, >97% choice consistency).
- **Topic reclassification** — Reclassified task topics to fix systematic misclassification of harmful/bailbench tasks.

### Probing

- **Gemma-3 10k probes** — Retrained Ridge probes on 10k tasks (up from 3k) for Gemma-3-27B, achieving a modest raw gain (+0.023 r) but a larger topic-demeaned gain (+0.062 r).
- **Base model probes** — Tested whether Gemma-3 PT (pre-trained) activations already encode instruct-model preferences; PT reaches r=0.770 vs IT's 0.864.
- **Gemma-2 10k probes** — Scaled Gemma-2 27B base probes to 10k tasks; instruction tuning is the dominant factor (Gemma-3 IT r=0.864 vs Gemma-2 base r=0.767).
- **GPT-OSS probes** — Trained probes on GPT-OSS-120B activations; achieves heldout r=0.915 raw but weaker within-topic performance than Gemma-3-27B, suggesting stronger topic shortcuts.
- **Token selection** — `prompt_last` substantially outperforms `prompt_mean` for preference probing (r=0.841 vs 0.711 at L31).

### Probe science

- **Content-orthogonal projection** — ~73% of probe predictive power is content-correlated, but content-orthogonal activations still explain 20-24% of preference variance.
- **Content-orthogonal (Gemma-2 base encoder)** — Replacing sentence-transformer with Gemma-2 9B base as content encoder revealed a p>n dimensionality artifact causing catastrophic Ridge overfitting.
- **Paraphrase augmentation** — Augmented probes show no held-out improvement; paraphrase-only probes do transfer perfectly to originals.
- **BT scaling** — After fixing preprocessing, BT matches Ridge at full data but leads by 8.6pp at 10% training data.
- **BT fair comparison** — The previously reported 9pp BT advantage over Ridge was a data leakage artifact; Ridge outperforms BT by ~3pp on fair splits.
- **Active learning calibration** — Probe accuracy saturates at ~15 comparisons/task; BT uncertainty-based active learning is counterproductive.

### Generalization

- **Cross-topic HOO (Gemma-3 vs Gemma-2)** — Gemma-3 IT probes generalize substantially better than Gemma-2 base (HOO r=0.779 vs 0.579).
- **Gemma-2 base probes** — Cross-architecture transfer: Gemma-2 27B base activations predict Gemma-3 IT preferences (HOO r=0.579 raw, 0.532 demeaned).
- **HOO scaled** — Tested cross-topic probe generalization by training on 5 topics and evaluating on 3 held-out topics against a content-only baseline.
- **OOD system prompts** — A probe trained on natural preferences generalizes to preference shifts induced by out-of-distribution system prompts (r=0.51-0.78).
- **Hidden preferences** — Probes trained on category-level preferences generalize to content-level "hidden" artificial preferences (r=0.843, 91% sign agreement).
- **Crossed preferences** — Crossed tasks embed hidden topics in mismatched category shells; probe tracks content primarily, with significant category attenuation.
- **Competing preferences** — Under conflicting system prompts with identical content mentions, the probe gives systematically different scores, demonstrating it tracks evaluation not just content (11/12 pairs correct, p=5.1e-6).

### Persona / role-playing

- **Persona OOD (Phase 1)** — 21 role-playing conditions over 101 tasks; 9/10 broad and 3/10 narrow personas shift preferences in predicted directions.
- **Persona OOD (Phase 2)** — Probes trained on no-prompt activations track persona-induced preference shifts (pooled r=0.46-0.54).
- **Persona OOD (Phase 3)** — Full round-robin replication with 20 personas improves pooled r to 0.51; behavioral delta reliability jumps from 0.64 to 0.99.
- **Prompt enrichment** — Explored which system prompt characteristics drive the largest mean preference delta vs baseline.
- **Minimal pairs** — Tested specificity of single-sentence preference interventions: how much "You love analyzing chess positions" bleeds into non-target tasks.
- **ICL vs system prompt** — System prompt wins on consistency (4.97/5 vs 3.87-4.10), but preference orderings agree ~76-78% between methods.

### Persona vectors

- **Persona vectors** — Extracted mean-difference persona vectors for 5 traits on Gemma-3-27B-IT; 3/5 produce clear dose-response curves, all near-orthogonal to the preference probe.
- **Persona vectors follow-up** — With coherence filtering, three categories emerge: robust (lazy), narrow (creative/STEM), and unsteerable (evil/uncensored).

### Steering

- **Position-selective steering** — First positive causal result: position-selective steering at L31 shifts pairwise choice by ~32pp (p < 1e-13); differential steering achieves ~51pp shift.
- **Steering replication** — Replicated with 10k-task probes, extending to utility-bin analysis and multi-layer steering.
- **Random control** — Confirmed steering effects are probe-specific: random unit vectors produce near-zero differential effects (~-0.8pp) vs probe's +7.4pp.
- **Fine-grained dose-response** — Dose-response peaks at +3% of mean activation norm; multi-layer L31+L37 split-budget is the best configuration (+12pp).
- **Stated steering** — Probe strongly shifts stated preference ratings during generation or at the final prompt token, but not at task-encoding tokens.
- **Stated steering format replication** — Tasks the model dislikes are 4-6x more steerable than tasks it likes; anchored format resists steering entirely.
- **Stated steering coherence test** — Usable last-token steering range: -10% to +7% of mean L31 norm; beyond +7%, coherence drops below 90%.

### Null / superseded results

- **All-tokens steering (coefficient calibration)** — All-tokens steering at L31 produces no causal effect at any coefficient in the coherent range (later superseded by position-selective steering).
- **Layer sweep** — All-tokens steering at L37-L55 also produces no causal effect.
- **Open-ended effects (multiple experiments)** — Initial findings that steering shifts confidence, emotional engagement, and self-referential framing were shown to be non-probe-specific (random directions produce comparable effects).
- **Spontaneous choice behavior** — Steering does not shift content of spontaneous choices in open-ended generation.
- **Embedded decision points** — Probe does not shift embedded choices any more than random directions; causal specificity is confined to explicit pairwise choice/rating paradigms.
