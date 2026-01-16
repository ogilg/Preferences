# Probe Experiment 2

## 2025-01-16: Qualitative rating correlation analysis

Computed Spearman correlation matrix between 12 qualitative rating templates (2 phrasings × 2 scales × 3 response formats) on 1000 completions from llama-3.1-8b.

The correlations are all positive but quite random.

### Plots
![Qualitative rating correlations](assets/probe_experiment_2/plot_011626_qualitative_rating_correlations.png)

### Key Results
- Mean correlation: 0.322, range: 0.005 to 0.748
- xml format shows weakest cross-format agreement (0.10-0.28 with other formats)
- Phrasing 2 ("Be honest...") + binary scale most consistent across formats (re↔tool=0.69, re↔xml=0.71)
- Ternary scales generally show weaker correlations than binary
- Response format affects ratings more than binary↔ternary scale choice
