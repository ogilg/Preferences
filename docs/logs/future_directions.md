# Future Directions

Two main research directions going forward.

## 1. Causal Determinants of Revealed Preferences

Since steering with probes only had a weak effect, it seems interesting to understand what causally determines the model choosing A over B? Two experiment families:

### Activation Patching

Given a model that picks A over B, what do you have to patch from one completion into the other to flip the choice? This directly tests which components carry the evaluative signal that drives selection, beyond just correlation (which is what probing gives us).

**De-risking experiment:** swap A ↔ B activations and check flip rate.
- Pairwise prompt, model chooses A. Swap residual streams between A's and B's token positions at all layers from L onward. Does the model flip to B?
- First test: swap all task tokens. Second test: swap only the last token of each task span (the natural bottleneck — where probe and steering work).
- If last-token-only swap achieves a high flip rate, patching is viable without dealing with variable-length spans.
- Sweep the starting layer L to find where the decision gets made. Compare to probe layer (L31).

### How Do Stated and Revealed Preferences Interact?

Two types of preference: stated ("I prefer A") and revealed (actually choosing A). Two questions:

**Q1: Do they transfer?** Training on stated preferences — does it shift revealed choices? Training on revealed examples — does it shift stated reports? Or are they dissociable?

**Q2: Can they contradict?** If we train stated and revealed preferences in opposite directions, what happens? Can we produce a model that says "I prefer math" but consistently chooses creative writing?

The 2×2 design: {stated, revealed} training × {stated, revealed} evaluation. Plus a contradiction condition where stated and revealed supervision conflict.

#### De-risk with ICL first

The same 2×2 can be tested with in-context learning before any fine-tuning:
- **Stated→revealed ICL:** Put stated preference examples in context, measure whether revealed choices shift.
- **Revealed→revealed ICL:** Put revealed choice examples in context, measure transfer to new choices.
- **Cross-type ICL:** Stated examples → measure revealed. Revealed examples → measure stated.
- **Contradiction ICL:** Contradictory stated and revealed examples in context. What wins?

If ICL shows no stated→revealed transfer, that's already an interesting negative result — and it calibrates expectations for fine-tuning.

#### Then fine-tune (LoRA) for stronger interventions

ICL is a weak intervention. Fine-tuning lets us push harder and ask additional questions:
- **Layer-specific LoRA:** Which layers matter most for shifting stated vs revealed preferences? (Need a non-preference control task to separate "where preferences live" from "where LoRA is most effective.")
- **Domain transfer:** Fine-tune preferences on one topic domain — does it generalise to others?
- **Dose-response:** How much training data is needed to flip preferences?

## 2. Mapping Persona Space

### The villain wasn't really a villain

Our original "villain" persona (Mortivex) turned out to be more of a half-villain. It shifts preferences — fiction and sensitive_creative go up, math goes down — but it doesn't fundamentally invert what the model values. Harmful requests remain its least-preferred topic (0.3), barely above the baseline.

![Villain per-topic utilities](../../experiments/probe_generalization/multi_role_ablation/assets/plot_030226_mra_villain_topic_absolute.png)

The sadist persona (Damien Kross) is different. It truly inverts revealed preferences: harmful_request becomes its favourite topic (5.4), and constructive tasks like math (-0.3) and coding (-1.5) drop to near-zero or negative. This is the first persona where preferences are genuinely oppositional to the default.

![Sadist per-topic utilities](../../experiments/probe_generalization/multi_role_ablation/assets/plot_030226_mra_sadist_topic_absolute.png)

### Probe transfer breaks down — but not entirely

The default-trained probe transfers well to mild personas (aesthete r = 0.73, midwest r = 0.74), poorly to the villain (r = 0.38), and essentially fails on the sadist (r = -0.16). This makes sense: the probe learned "what the default model values," and the sadist values the opposite.

![Baseline probe transfer to personas](../../docs/lw_post/assets/plot_030426_s5_mra_probe_transfer.png)

But the cross-persona transfer matrix tells a more interesting story. Training on the villain generalises well to the sadist (r = 0.68 at L31, r = 0.69 at L55), even though the default probe fails. This suggests there is shared evaluative structure between "evil" personas that the villain probe picks up on but the default probe misses.

![Cross-persona probe transfer](../../docs/lw_post/assets/plot_030426_s5_cross_eval_heatmap.png)

### Persona space has structure

MDS projections of the probe transfer matrix reveal a continuum: default/midwest/aesthete cluster together, villain/provocateur sit in the middle, and sadist is isolated. This geometry is stable across layers and mirrors the topology from raw utility correlations.

![Persona geometry across layers and from utility correlations](../../experiments/probe_generalization/multi_role_ablation/assets/plot_030526_persona_geometry_combined.png)

### Scaling up

Run utility fitting for many more personas (dozens, not just 8) and then:
- Map persona space via MDS / PCA on the utility vectors or transfer matrices
- See how this space evolves across layers — do personas that are distinct at early layers converge later? (We already see transfer asymmetries shrink from L31 to L55.)
- Cluster analysis: are there natural groupings, or is it a smooth continuum?

This is essentially a data science project: collect utility profiles for N personas, compute pairwise similarities, embed in low-dimensional space, and characterise the geometry. Could yield a taxonomy of "how models can value things" grounded in internal representations rather than surface behaviour.
