## Appendix C: Evaluative representations in pre-trained models

There is a tension in our framing:
- On the one hand we say that evaluative representations are necessary for robust agency, and that this is the most likely way they might be welfare-relevant.
- On the other hand, we seem to find something like evaluative representations in a pre-trained version of Gemma3-27b. Pre-trained models do not seem to be anywhere near having robsut agency.

There are two ways to reconcile this.

**Agency lives in the simulacra.** Under the [Persona Selection Model](https://www.anthropic.com/research/persona-selection-model), pre-training learns a distribution over personas. These personas have preferences, and the model learns to represent what each persona would value. The evaluative representations are real, but they belong to the simulated personas rather than to the model itself.

**Evaluative representations are necessary but not sufficient.** Another way out is that pre-training learns something like a precursor to agency. The model acquires representations that encode valuation, but these don't yet play the right functional role in driving choices. Post-training is what connects them to behaviour. On this view, evaluative representations are a necessary ingredient for agency, and finding them in base models just means that one ingredient is already in place.

These two accounts aren't mutually exclusive. Both leave the door open to what we  observe: base model probes work but generalise less well than instruct probes. Testing whether the base model probe direction has any causal influence on generation would potentially help distinguish between the two views.
