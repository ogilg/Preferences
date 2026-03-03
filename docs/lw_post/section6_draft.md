## 6. Some evidence that the probe direction is causal

If the probe reads off a genuine evaluative representation, steering along that direction should shift preferences. We test this for both revealed preferences (pairwise choices) and stated preferences (ratings).

### 6.1 Steering revealed preferences

In the pairwise comparison prompt, the model sees two tasks and begins completing whichever one it prefers. We steer using the probe direction from section 4, applied *differentially*: the probe direction is added to activations at the token positions where task A appears in the prompt, and subtracted at task B's positions. [^differential]

[^differential]: Steering either just positively or negatively one one task's tokens also had some causal effect, although even weaker.

[^borderline]: This baseline P(A) is pooled across both prompt orderings. Therefore it also captures pairs where ordering bias dominates preference. I.e. the model always chooses the first task no matter the order.

**Setup.** 300 task pairs where the model doesn't always pick the same task (P(A) between 0.2–0.8 at baseline[^borderline]). We use Gemini-3-flash as a coherence judge, and find that steered models lose coherence outside of the ±5% steering coefficient range.

At ±3% of the activation norm (the peak), steering shifts P(choose A) by ~17%. Both positive and negative directions work.

![Revealed preference dose-response](assets/plot_030326_s6_revealed_dose_response.png)
*Each condition is run in both prompt orderings (A-first and B-first, 5 resamples each) to control for position bias (in borderline cases Gemma-3-27b tends to pick the first task). The dashed lines show per-ordering curves, confirming the effect is consistent regardless of presentation order.*

**Random direction control.** The same experiment with a random unit vector in the same activation space produces near-zero effects at the same magnitudes, confirming the effect is specific to the probe direction.

### 6.2 Steering stated preferences

Same probe direction, but now the model rates individual tasks instead of choosing between a pair. We tested steering at three token positions: on the task tokens only, at the final task token, and during the model's generated response.

**Setup.** 200 tasks, 3 steering modes, 10 samples, applied across 3 response formats (three separate line plots below).

Our findings:
- Steering on the task prompt tokens does not have a noticeable effect.
- Steering on the last token of the prompt has a clear, roughly montonic effect in 2/3 formats.
- Steering during generation leads to near-identical results as steering on the final task token.



![Stated preference dose-response](assets/plot_022626_s5_stated_dose_response.png)
*Three steering modes: on the task's token positions only (blue), at the final task token (red), and during the model's generated response (green). The green line is mostly hidden behind blue in the left panel (both show no effect). Green has missing points at some coefficients because steering during the full generation degrades coherence more than the other modes. Faded region beyond ±5% is where generation loses coherence.*
