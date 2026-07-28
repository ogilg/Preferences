# Review issues checklist

`[x]` means the rebuttal draft addresses the issue.

## Experiments and analyses

- [x] **Persona-prompt placement.** Moving the evil-persona instruction from a
  same-turn prefix into a separate preceding user turn preserved the monotonic
  steering response. Because Gemma has no native system role, this is explicitly
  reported as a placement control rather than a system-versus-user comparison. Spec:
  `experiments/reviewer_followups/system_vs_user_persona_steering_spec.md`.

- [x] **Task-pool stability.** Leave-one-source-dataset-out refits gave
  retained-task utility correlations of 0.931–0.990 and signed probe cosines of
  0.705–0.959. The worst case for both was omitting Alpaca.

- [x] **Steering selection.** L23 was calibrated on 50 pairs and reported on a
  separately sampled 150-pair set; only two pairs repeat. On the 112 fully
  task-disjoint evaluation pairs, the swing changes from 0.960 to 0.958.

- [x] **Harm labels.** Blind check: 29/30 agreement.

- [x] **Position bias.** Utility elicitation randomized order; steering used both
  AB and BA with sign correction and response remapping.

- [x] **Probe and regularization sensitivity.** The rebuttal reports the flat
  near-optimal alpha sweep, robustness across nearby layers and turn-boundary
  positions, and clarifies that Ridge was not selected from competing probe
  families based on downstream results.

- [x] **Cross-model steering.** State clearly that steering worked on Gemma but
  not convincingly on Qwen. Give hypotheses, not conclusions.



## Conceptual responses

- [x] **Shared machinery.** The rebuttal clarifies that this meant
  representational reuse, not shared circuitry, and adopts the reviewer's
  terminology to avoid confusion.

- [x] **Evaluative representation.** The rebuttal states the operational
  definition and clarifies that it does not imply necessity or uniqueness.

- [x] **Define preference vector.** The rebuttal commits to defining it directly
  as the learned linear-probe weight vector reused for steering.

- [x] **Sign flip versus behavior.** The rebuttal distinguishes the
  representational sign flip from behavioral shifts measured separately.

- [x] **Consciousness and AI welfare.** The rebuttal retains this core motivation
  while clarifying that evaluative representations are only a proposed necessary,
  not sufficient, condition and that the experiments do not test consciousness.

- [x] **Gemma versus Qwen.** Explain that decodability does not guarantee
  steerability across models.

- [x] **Topic-label noise.** The rebuttal summarizes the two cross-model audits,
  cross-source check, taxonomy refinement, and hidden-harm pass.



## Paper presentation

- [x] State that the default Assistant has no custom persona prompt.

- [x] Increase figure font sizes and make styles consistent.

- [x] Make Figure 3 colourblind-accessible and clarify that Figure 7 shows the
  common trend rather than differences between individual personas.



## Remaining writing changes

1. Apply the persona-prompt terminology correction throughout the manuscript.
2. Clarify sign flip versus behavior and Gemma versus Qwen.
3. Add the completed reviewer-follow-up analyses to the appendix.
