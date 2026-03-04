## Conclusion

**How should we update?**

- **We found early evidence that some models have evaluative representations.**
  - Theories of welfare disagree on what matters (see [Appendix A](appendix_philosophy_draft.md)); this finding updates you more on some (like robust agency) than others.
  - Even under robust agency, evaluative representations are only one part of the story.
  - Importantly, our evidence that the representations we have found are causal is weak. Steering only shifts choice probabilities by ~17% on tasks that were already borderline ([Section 5](#5-some-evidence-that-the-probe-direction-is-causal)).

- **Preference representations are deeper than what one might have thought.**
  - A reasonable prior would have been that system prompts like "You hate cheese" change the model's behaviour without changing its internal valuations.
  - Instead, the probe tracks internal shifts even for fine-grained manipulations (a single sentence in a biography; [Section 3.3](#33-fine-grained-preference-injection)).

- **Representational reuse across personas?**
    - Probes trained on one persona partially transfer to others, suggesting shared evaluative representations ([Section 4.2](#42-probes-generalise-across-personas)).
  - That being said, transfer is uneven. It works far worse for the villain persona which has a different preference profile.

### Open questions

1. **Why are steering effects modest?** 
   - What are the other mechanistic determinants of revealed preferences? 
   - Are there other evaluative mechanisms? Perhaps that are not easily captured by linear directions, or our methodology in general?

2. **How persona-relative are these representations?**
    - To what extent are the same evaluative representations re-used across personas?
    - Are preferences downstream of personas?

3. **Do base models have evaluative representations?** (see [Appendix B](appendix_base_models_draft.md))
    - If models have evaluative representations, do these come from pre-training? Does post-training significantly alter them? 
    - If base models have something like evaluative representations, do they play the right causal roles?


