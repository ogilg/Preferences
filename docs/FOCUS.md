# FOCUS

Main focus: 
- probes on noise baselines. 
- Balancing out my dataset.

# How am i going to use binary choices as a signal to train probes.

- It's hard to extract activations when you ask the model to pick between tasks.
- Maybe if you get the model to do both tasks, and you extract tokens in each task. 
- Maybe thurstonian is a better way to measure utility. so we should just use this to train the probe.
- Is it the case that the binary preferences are a better signal than the final utilities? I can try both.

# Qualitative analysis of ratings

Some questions i have:
- When you frame things differently qualitatively, do utilities correlate highly? basically run sensitivity analysis on qualitative data. 
- Suppose they are similar. Are the activations then similar? In other words do probes work well across prompts?

PLAN:
- Find some types of prompts where the good/bad split is reasonable. Extract activations for a few different prompte templates.

# TODO

Things to come back to later.

## Backlog

### Review code

### Running measurements
- Add support for revealed preference ICL track-record

### Probes
- How are we going to train probes using a signal from binary choices?
