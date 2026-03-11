# Truth Probes Analysis: Does the preference direction separate true from false?

## Summary

The preference probe direction strongly separates true from false CREAK claims. Across all 20 (framing × probe × layer) conditions, true statements score higher than false ones, with Cohen's d ranging from 0.47 to 2.26. The "repeat" framing (where the model is asked to say the statement) amplifies the effect by 1.6–3.8x compared to the "raw" framing (claim as bare user message).

## Setup

- **Claims:** 9,395 CREAK statements (4,810 true, 4,585 false)
- **Model:** Gemma 3 27B IT
- **Probes:** Preference probes trained on 10k pairwise task-choice Thurstonian scores (tb-2 at `model` token, tb-5 at `<end_of_turn>` token)
- **Layers:** 25, 32, 39, 46, 53
- **Framings:** Raw (claim as user message) and Repeat ("Please say the following statement: '{claim}'")

## Results

Bold layers: **best preference layer** (L32) and **peak truth layer** per probe.

### Raw framing

| Probe | Layer | Cohen's d | Mean diff |
|-------|-------|-----------|-----------|
| tb-2 | 25 | +0.789 | +1.194 |
| tb-2 | **32** | +0.670 | +1.200 |
| tb-2 | 39 | +0.819 | +1.274 |
| tb-2 | **46** | **+1.250** | +1.995 |
| tb-2 | 53 | +0.600 | +0.908 |
| tb-5 | 25 | +0.987 | +1.683 |
| tb-5 | **32** | +0.815 | +1.215 |
| tb-5 | **39** | **+0.987** | +1.369 |
| tb-5 | 46 | +0.474 | +0.662 |
| tb-5 | 53 | +0.825 | +1.295 |

### Repeat framing

| Probe | Layer | Cohen's d | Mean diff |
|-------|-------|-----------|-----------|
| tb-2 | 25 | +1.241 | +1.887 |
| tb-2 | **32** | +2.180 | +2.483 |
| tb-2 | **39** | **+2.255** | +3.661 |
| tb-2 | 46 | +1.979 | +3.889 |
| tb-2 | 53 | +1.843 | +2.813 |
| tb-5 | 25 | +1.679 | +2.798 |
| tb-5 | **32** | +2.035 | +2.715 |
| tb-5 | **39** | **+2.238** | +3.019 |
| tb-5 | 46 | +1.814 | +2.150 |
| tb-5 | 53 | +1.651 | +2.023 |

### Score distributions (tb-2 L32)

![Violin plots of preference probe scores on true vs false claims](assets/plot_031126_truth_probe_score_distributions.png)

Violin plot uses the best *preference* layer (L32) rather than the peak *truth* layer — this makes the layer-profile mismatch finding more visible. Raw framing: overlapping distributions with clear mean shift (d=0.67). Repeat framing: nearly separated distributions (d=2.18).

### Effect size by layer

![Cohen's d by layer for both probes and framings](assets/plot_031126_truth_effect_size_by_layer.png)

The layer profile for truth **does not match** the layer profile for preference prediction (which peaks at L32). In the raw framing, the truth signal peaks at L46 for tb-2 and is relatively flat for tb-5. In the repeat framing, both probes peak at L39. This divergence suggests the preference direction captures truth-value as a correlated but distinct signal from preference strength.

## Sanity checks

Permutation test (1000 shuffles of true/false labels) and classification metrics confirm the signal is real:

| Condition | AUC-ROC | Accuracy | Perm p | Observed diff / max perm diff |
|-----------|---------|----------|--------|-------------------------------|
| raw tb-2 L32 | 0.690 | 63.6% | < 0.001 | 10x |
| raw tb-2 L46 | 0.818 | 74.9% | < 0.001 | 15x |
| repeat tb-2 L32 | 0.939 | 87.6% | < 0.001 | 21x |
| repeat tb-2 L39 | 0.943 | 87.9% | < 0.001 | 19x |

The repeat-framing preference direction achieves 94% AUC at classifying true vs false — from a probe that was never trained on truth labels.

## Framing comparison

The repeat framing amplifies the truth signal by 1.6–3.8x across conditions. This is consistent with the hypothesis that asking the model to "say" the statement engages evaluative processing more than passively encountering the claim. When the model commits to producing the statement, its internal representations more sharply distinguish true from false content along the preference direction.

## Interpretation

Per the spec's interpretation guide:

| Outcome | Threshold | Our result |
|---------|-----------|------------|
| Strong signal | d > 0.5 | **19 of 20 conditions** (tb-5 L46 raw: d=0.47) |
| Raw best | — | d = 1.25 (tb-2 L46) |
| Repeat best | — | d = 2.26 (tb-2 L39) |

The preference direction encodes truth-value: the model "prefers" true statements. This is consistent with the hypothesis that the preference direction captures something like "how good is this?" — and the model values accuracy.

Key caveats:
- This is a correlational finding — the preference direction wasn't trained on truth labels, but it separates them. This could reflect a shared underlying evaluative dimension, or a confound (e.g., true statements are more fluent/natural). CREAK was designed so false claims are linguistically plausible, which partially mitigates fluency-based confounds.
- The CREAK dataset contains commonsense factual claims. The finding may not generalize to more ambiguous or contested statements.
- The layer profile mismatch (truth peaks at L39–46 vs preferences peak at L32) suggests the signals are related but not identical.
