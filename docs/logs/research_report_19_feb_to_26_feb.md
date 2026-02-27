# Weekly Report: Feb 19 - 26, 2026

Most of this week was replicating previous results with the 10k-task probes and on GPT-OSS, plus writing the LW post. The LW post draft ([`lw_post_draft.md`](../lw_post/lw_post_draft.md)) has the full write-ups.

## Probe training works on GPT-OSS-120B

- I reran the whole probe pipeline (getting activations, fitting utilities, training probes).
- This worked: probes actually had higher overall accuracy.
- The gpt-oss probe scored worse on harmful topics. My hypothesis is that this is due to more safety training. But it might partly be because it refused more often, and therefore we had less training data.
- Full write-up: [`appendix_gptoss_draft.md`](../lw_post/appendix_gptoss_draft.md)

## Also retrained probes on Gemma-3 base (pre-trained, no RLHF)

- After reading/thinking about the "Persona Selection Model" post, I actually think it's very reasonable to expect to find evaluative representations in pre-trained models. Although these likely don't activate in the same way.
- This is fairly consistent with what i found.
- It is interesting to look at which topics are harder to predict for the pre-trained model probes: it turns out to be math/coding + harmful requests.
- Full write-up: [`section3_draft.md`](../lw_post/section3_draft.md)

## Re-ran "system-prompt induced preference" generalisation tests
- I re-ran this with probes trained on 10k tasks. I got similar but slightly better results.
- I also have a new experiment, where managed to inject a very precise preference, which the probe picked up on (see 4.3 in the lw draft),.

- Full write-up: [`section4_draft.md`](../lw_post/section4_draft.md)

## Re-ran steering with better probes

Replication of the steering results from Feb 5-10 (section 4), now with the 10k probes. Also added stated preference steering (new).

- Revealed preference steering still works — very weak but robust effect. Around ~5% difference in chance of picking a task at most.
- New: steered stated preferences by steering during the model's response (rather than applying the steering vector to the task tokens). This made the effect much larger.
- Full write-up: [`section5_draft.md`](../lw_post/section5_draft.md)

