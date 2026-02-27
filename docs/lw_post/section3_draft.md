### 3. Linear probes predict preferences beyond content

Can we find these utility scores in the model's activations? We train a Ridge-regularised linear probe on residual stream activations (layer 31 of 62, the best layer for both the instruct and pre-trained models) to predict Thurstonian utilities.

![Probe pipeline](assets/plot_022626_probe_pipeline.png) We train on 10k tasks and evaluate on held-out utilities from a separate measurement run (different pairings, no shared information), split into 2k validation (for Ridge alpha sweep) and 2k test.

The probe achieves Pearson r = 0.86 and predicts 77% of pairwise choices on the test set.

But a probe that predicts preferences could just be encoding content — the model prefers math over harmful requests, and the probe learns "is this math?". To test this, we hold out entire topics: train on 11 of 12 topics, evaluate on the held-out topic, across all 12 folds. A content detector would fail here. We compare three conditions:

- **Gemma-3 27B instruct** (IT, layer 31): the model we're studying
- **Gemma-3 27B pre-trained** (PT, layer 31): the base model before instruction tuning or RLHF — if evaluative representations emerge from post-training, this should have weaker signal
- **Sentence-transformer baseline**: a Ridge probe on [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) embeddings of the task text — a small text encoder. This captures how much of preference is predictable from content alone

![Cross-topic generalisation](assets/plot_022626_cross_model_bar.png)

The instruct probe generalises well across topics (r = 0.82, down from 0.86 held-out). The pre-trained model encodes preferences above the content baseline (r = 0.63) but generalises substantially worse. The sentence-transformer captures some preference signal from content alone (r = 0.35 cross-topic) but falls far short of either neural model.

The per-topic breakdown shows where post-training helps most:

![Per-topic cross-topic generalisation](assets/plot_022626_per_topic_hoo.png)

The largest instruct–pre-trained gaps are on safety-relevant topics (harmful requests, security & legal, sensitive creative), as well as math and coding. These are areas that we know post-training focuses on.

**Note on the pre-trained models:** To the extent that they encode a distribution over persona space (PSM), it makes sense for pre-trained models to have evaluative representations that track a given persona's preferences. However we wouldn't expect these preferences to play the same causal roles during generation as they do for post-trained models.
