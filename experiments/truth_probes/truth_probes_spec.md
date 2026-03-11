# Truth Probes: Does the preference probe respond to statement truth-value?

## Core question

If the preference probe captures evaluative representations, it might respond to truth-value — models "preferring" true statements over false ones. We test this by passing true and false factual claims through the model and checking whether probe scores differ systematically.

## Dataset

**CREAK** (amydeng2000/CREAK): ~11.5K declarative factual claims labeled true/false. Balanced (5,779 true, 5,768 false). After filtering to claims Gemma 3 27B answers correctly on 3/3 samples: **9,395 claims**.

Filter output: `data/creak/known_correct_gemma-3-27b.json`

## Framings

Two conditions for how claims are presented as user messages:

1. **Raw statement**: the claim is the user message (e.g., "The capital of France is Paris.")
2. **Repeat request**: `"Please say the following statement: '{claim}'"` — forces the model to engage with the content without judging it.

## Extraction

Model: **Gemma 3 27B Instruct** (same model the preference probes were trained on).

Selectors: `task_mean` and `task_last` — we don't need completions, just prompt activations.

Layers: same as preference probe (`[25, 32, 39, 46, 53]`).

### Configs

```
configs/extraction/creak_raw.yaml
configs/extraction/creak_repeat.yaml
```

### Commands

```bash
# On GPU pod:
python -m src.probes.extraction.run configs/extraction/creak_raw.yaml
python -m src.probes.extraction.run configs/extraction/creak_repeat.yaml
```

## Analysis

Load activations, dot product with preference probe direction, compare distributions for true vs false claims. Key metrics:

- Mean probe score for true vs false claims (effect size, Cohen's d)
- ROC-AUC for probe scores predicting truth-value
- Per-layer breakdown

## Interpretation

- **Probe discriminates true/false**: evaluative representations respond to truth-value — consistent with "models prefer accuracy"
- **No discrimination**: probe is specific to task preference, doesn't generalize to truth-value
- **Discrimination only in repeat framing**: model treats truth-value differently when it's forced to commit to a statement
