# Weekly Report: Feb 12 - 19, 2026

## Summary

1. **Active learning calibration**: Probe pairwise accuracy saturates at ~15 comparisons/task (72.4% vs 74.0% at 39), so the 10K run needs only 15-20 comparisons/task with 3 samples/pair (~99K total comparisons).

2. **BT vs Ridge scaling**: The previously reported 3pp Ridge advantage over Bradley-Terry was a preprocessing artifact (missing StandardScaler); after fixing, BT matches Ridge at full data and wins at low data (+8.6pp at 10%), but BT uncertainty-based active learning is counterproductive — adding selected pairs decreased accuracy by 2pp while random pairs increased it.

3. **Content-orthogonal base model probes**: Gemma-2 27B base achieves R²=0.789 predicting Gemma-3's preferences, but this is only marginally above the strongest content baseline (0.76), suggesting the base model signal is content-driven rather than evaluative.

4. **Cross-topic generalization (base vs IT)**: Gemma-3 IT generalises across topics far better than Gemma-2 base (HOO r: 0.779 vs 0.579), with ~88% of the advantage coming from within-topic signal that survives topic-demeaning.

5. **Prompt enrichment**: Enriching persona system prompts has marginal effect on already-strong prompts, with transgressive personas (evil_genius, nationalist_ideologue) producing the strongest preference shifts and distinct preference axes emerging across 10 new personas.

6. **Persona OOD phase 3**: With 20 personas and 196 obs/task (up from 20 in phase 2), probes track persona-induced behavioral shifts at pooled r=0.51 with all four success criteria met, confirming that phase 2's lower correlation was primarily measurement attenuation.

7. **Token selection**: prompt_last substantially outperforms prompt_mean for preference probing (R²=0.841 vs 0.711 at L31), because the last token is a natural summary position and averaging dilutes signal with irrelevant early-position activations.

8. **Paraphrase augmentation**: Paraphrasing tasks and inheriting utilities is behaviourally valid (rank correlation 1.0, win rate 0.55) and paraphrase-only probes transfer perfectly to originals (cosine similarity ~1.0), but augmentation provides no held-out improvement — it finds the exact same probe direction.
