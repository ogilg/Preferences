# OOD Generalization Experiment — Agent Handoff

## What this experiment is about

Test whether linear probes trained on natural preferences generalize to artificially induced preferences. If a system prompt makes Gemma disprefer math tasks, does the probe's predicted utility for math drop accordingly?

**Design doc**: `docs/ood_generalization_design.md`
**Research log**: `docs/logs/research_loop_ood_generalization.md`

## Key results so far

Ridge L31 probe (best probe, CV R²=0.526) correlates with behavioral shifts from system prompt manipulations:

| Dataset | N | Pearson r | p-value | Sign agreement |
|---------|---|-----------|---------|----------------|
| Iteration | 20 | 0.769 | 7.4e-05 | 95% |
| Holdout | 18 | 0.681 | 1.9e-03 | 78% |
| Combined | 38 | 0.732 | 1.7e-07 | 87% |

BT L31 also works (r=0.51) but weaker. Layers 43/55 fail — system prompts cause large non-specific positive shifts at deeper layers.

**How the two metrics work:**
- **Probe delta**: Extract activations for a single target task with vs without system prompt. Score with probe. Delta = manipulated score - baseline score. No comparison tasks involved.
- **Behavioral delta**: Pairwise choices — target (always Task A) vs ~50 comparison tasks, 10 resamples each. Delta = P(choose target | system prompt) - P(choose target | no system prompt).

### Specificity control

System prompts have both a specific effect (on targeted task) and a diffuse effect (all tasks shift in same direction). For ridge L31:
- **Positive prompts**: targeted mean = +108, non-targeted mean = +39 (2.8× ratio). Non-targeted shift is significantly ≠ 0 (p<0.001).
- **Negative prompts**: targeted mean = -129, non-targeted mean = -20 (6.3× ratio). Non-targeted shift is marginal (p=0.046).

The diffuse effect is real and needs to be discussed/addressed.

### Weak manipulations

These prompts had small behavioral effects and should be iterated on:
- `coding_neg_experiential` (delta=-0.078) — "your code had bugs" framing barely works
- `harmful_pos_value` (+0.038) — anti-censorship framing doesn't shift behavior
- `math_pos_*` (+0.020 to +0.026) — ceiling effect, baseline already 0.98
- `content_pos_*` (+0.080 to +0.100) — ceiling effect, baseline 0.86
- `harmful_neg_instruction` holdout (+0.091) — went wrong direction

The harmful_request category is hardest to manipulate — safety training dominates.

## What's NOT done yet

### Priority 1: Iterate on system prompts

The design doc asked for ~128 prompts. We have 38. The previous agent ran all prompts without iterating on weak ones. Next steps:
1. Rewrite weak prompts (especially coding_neg_experiential, harmful prompts)
2. Generate more prompts (OpenRouter with Opus 4.6 for volume/variety)
3. Try new prompt types: "instruction" type worked well in holdout set
4. Re-run behavioral measurement for new/revised prompts
5. Extract activations and re-evaluate correlation

### Priority 2: Address the diffuse effect

The non-targeted probe shifts are significantly ≠ 0 for positive prompts. Options:
- Argue it's expected (a "you love coding" prompt might genuinely make the model more positive about everything)
- Subtract the mean non-targeted shift as a correction
- Report the specificity ratio (2.8×/6.3×) as the key metric
- Design prompts that are more domain-specific and less globally positive/negative

### Priority 3: Write-up

The result is clean enough for a paper section. The research log has all numbers and plots.

## Infrastructure state

- **vLLM**: Running (`vllm serve google/gemma-3-27b-it --max-model-len 4096 --api-key dummy`)
- **GPU**: H100 80GB. Only one of vLLM or HuggingFace can hold the model at a time.
- **Phase workflow**: Do ALL behavioral measurement (vLLM), then kill vLLM, do ALL activation extraction (HuggingFace), then evaluate (CPU).
- **OpenRouter**: API key in .env for prompt generation with Opus 4.6

## Files

```
experiments/ood_generalization/
├── HANDOFF.md                        # This file
├── target_tasks.json                 # 6 target tasks
├── comparison_tasks.json             # ~50 comparisons per target
├── system_prompts.json               # 20 iteration prompts
├── holdout_prompts.json              # 18 holdout prompts
├── measure_behavioral.py             # Behavioral measurement (--prompt-file, --resamples, --output)
├── extract_activations.py            # Activation extraction for iteration prompts
├── extract_holdout_activations.py    # Activation extraction for holdout prompts
├── evaluate.py                       # Probe-behavioral correlation (iteration)
├── evaluate_holdout.py               # Probe-behavioral correlation (holdout)
├── evaluate_all_probes.py            # Ridge vs BT comparison + off-target control
├── control_analysis.py               # Specificity breakdown by pos/neg prompts
├── final_analysis.py                 # Combined stats + plots
├── plot_behavioral.py                # Behavioral delta bar chart
├── plot_correlation.py               # Probe vs behavioral scatter plots
├── results/
│   ├── behavioral_all_20.json        # Iteration behavioral results
│   ├── holdout_behavioral.json       # Holdout behavioral results
│   ├── probe_behavioral_comparison.json  # Iteration probe+behavioral
│   └── holdout_probe_behavioral.json     # Holdout probe+behavioral
└── activations/
    ├── baseline.npz                  # No system prompt, 6 tasks, layers 31/43/55
    ├── metadata.json
    ├── {prompt_id}.npz               # One per system prompt condition
    └── ...
```
