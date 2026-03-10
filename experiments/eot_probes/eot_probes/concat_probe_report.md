# Concatenated EOT + Prompt-Last Probe

## Summary

A Ridge probe trained on concatenated activations from two token positions — `<end_of_turn>` (EOT, after the user message) and `prompt_last` (the `\n` after `<start_of_turn>model`) — outperforms either position alone on heldout utility prediction (r=0.875 vs 0.867/0.866), while preserving EOT's near-zero topic generalization gap (1.7% vs 1.8%).

**Setup.** Gemma 3 27B, layer 31. Concatenated activations have 2× hidden dim (10,752). Trained on 10k Thurstonian utility scores, evaluated on a separate 4k heldout set. All three probes use the same train/eval split.

## Heldout Eval

Pearson r and pairwise accuracy between probe scores and Thurstonian utilities on the 4k heldout set.

| Probe | Heldout r | Heldout pairwise acc |
|---|---|---|
| prompt_last | 0.866 | 76.7% |
| eot | 0.867 | 76.9% |
| **concat** | **0.875** | **77.2%** |

The ~1pp gain in r suggests the two positions carry partially non-redundant evaluative signal.

## Cross-Topic Generalization (held-one-out by topic, L31)

Each fold holds out one topic category (10 topics, 10 folds). "Within-topic r" is CV Pearson r on training topics; "held-out topic r" is Pearson r on the excluded topic. The gap measures how much performance drops on unseen topics — a large gap indicates the probe has overfit to topic-specific features.

| Probe | Within-topic r | Held-out topic r | Gap |
|---|---|---|---|
| prompt_last | 0.905 | 0.817 | 8.9% |
| eot | 0.889 | 0.870 | 1.8% |
| **concat** | 0.895 | **0.878** | **1.7%** |

The concat probe achieves the highest held-out-topic r (0.878) with the smallest generalization gap. Ridge regularization suppresses the topic-correlated features in the prompt_last half, so concatenation adds signal without inheriting the overfitting.

## Reproduction

```bash
python -m scripts.probes.make_concat_npz
python -m scripts.probes.concat_eot_prompt_last
python -m src.probes.experiments.run_dir_probes --config configs/probes/gemma3_10k_hoo_topic_concat.yaml
```
