## Appendix B: Evaluative representations in pre-trained models

There is a tension in our framing:
- On the one hand we say that evaluative representations are necessary for robust agency, and that this is the most likely way they might be welfare-relevant.
- On the other hand, probes generalise well across topics even when trained on base models. Despite the fact that pre-trained model do not seem like plausible candidates for robust agency.

There are two ways to reconcile this.

**Option 1: Agency lives in the simulacra.** Under the [Persona Selection Model](https://www.lesswrong.com/posts/dfoty34sT7CSKeJNn/the-persona-selection-model), pre-training learns a distribution over personas. More broadly, we might expect pre-trained models to learn context-aware representations of "what the role I am currently playing values". This circuitry might then be recycled across roles/personas. The candidate for robust agency would then be the simulacra.

**Option 2: Pre-trained models learn complex, but purely descriptive features that correlate highly with valuations, but do not yet play the right functional roles.** As an analogy, you could imagine a system developing representations that track "this action leads to food". This would correlate well with valuations, yet is purely descriptive. Something similar might be responsible for the high cross-topic generalisation with pre-trained models ([Section 2](#2-linear-probes-predict-preferences-beyond-descriptive-features)). It could also be that these complex but descriptive features are then harnessed into evaluative representations during post-training.





