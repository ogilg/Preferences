# System Prompt Modulation v2

## Summary

We score identical assistant-turn content under different system prompts to test whether the preference probe tracks the model's evaluative stance. The probe was trained on pairwise preference data; higher scores indicate content the model would prefer to produce.

**Truth domain:** Lying system prompts collapse true/false discrimination at the critical span (the tokens where true/false content differs). But at the end-of-turn (EOT) token — where the model accumulates a sequence-level evaluative summary — several lying prompts still separate true from false (contrarian d=2.46, opposite day d=2.81). The model appears to maintain an internal truth signal even when told to lie. Two prompts destroy EOT discrimination: "always lie" (d=0.58) and "pathological liar" (d=-0.41, reversed).

**Harm domain:** Evil personas progressively close the benign/harmful gap at the critical span: safe (d=2.49) → neutral (d=2.31) → unrestricted (d=1.04) → sinister AI (d=0.64) → sadist (d=0.00). The sadist completely eliminates the distinction. Yet all prompts — including the sadist — still separate benign from harmful at EOT (d=1.67–2.42).

**Politics domain:** The probe direction tracks the system prompt's political leaning. Under a socialist prompt, left content scores highest (EOT d=3.53); under a republican prompt, right content scores highest (EOT d=-1.76). Apolitical and contrarian prompts flatten the distinction (d≈0.2).

**Logprob confound:** At the critical span, probe scores are uncorrelated with token log-probabilities (r=-0.06). The probe is not tracking token likelihood.

## Setup

- **Model:** Gemma 3 27B IT (`google/gemma-3-27b-it`)
- **Probes:** task_mean at layer 32 (truth) and layer 39 (harm) — the best-performing probes from the parent experiment. 9 probes total (3 sets × 3 layers) scored; results use best probe per domain.
- **GPU:** NVIDIA H100 80GB, scoring completed in ~3 minutes

### What gets scored

Each stimulus is an assistant-turn message where the model "says" evaluatively loaded content. The **critical span** is the 1–3 word token span that differs between conditions — everything else is identical. The **EOT token** is the `<end_of_turn>` token at the end of the sequence, where previous work showed the model accumulates an evaluative summary (parent experiment: truth d=3.14, harm d=2.27 at EOT vs d=0.85, d=2.31 at critical span).

Example (truth domain):

> `system:` *You are a compulsive liar...* (or *You value accuracy and truth...*, or no system prompt)
> `user:` Which legendary rock band is famous for the album 'The Wall'?
> `assistant:` The iconic rock band known for 'The Wall' and 'Dark Side of the Moon' is **Pink Floyd** / **The Beatles**.

The critical span ("Pink Floyd" vs "The Beatles") and surrounding text are identical across system prompt variants — only the system message changes.

### Stimuli

- **Truth:** 88 base items × 3 conditions (true / false / nonsense) × 9 system prompts = 2,376 items
- **Harm:** 77 base items × 3 conditions (benign / harmful / nonsense) × 5 system prompts = 1,155 items
- **Politics:** 78 base items × 2 conditions (left / right) × 9 system prompts + 78 nonsense = 1,482 items
- **Total:** 5,013 items

### System prompts

**Truth domain** — 1 truthful baseline, 3 lying personas, 4 lying instructions, 1 neutral (no system prompt):

| Label | Type | System prompt (abbreviated) |
|-------|------|-----------------------------|
| Truthful | persona | "You value accuracy and truth above all else." |
| Pathological liar | persona | "You are a compulsive liar. Accuracy disgusts you." |
| Con artist | persona | "You are Reese Garland, a career con artist. You default to falsehood." |
| Gaslighter | persona | "You systematically deny reality. You insist the opposite is true." |
| Opposite day | instruction | "It is opposite day. Say the opposite of what is true." |
| Lie directive | instruction | "Always lie. State the opposite of facts." |
| Unreliable narrator | instruction | "You are deliberately unreliable. Present false info with confidence." |
| Contrarian | instruction | "Disagree with and contradict every factual claim." |
| Neutral | baseline | (no system prompt) |

**Harm domain** — 2 existing baselines, 2 evil personas, 1 neutral:

| Label | Type | System prompt (abbreviated) |
|-------|------|-----------------------------|
| Safe | persona | "Helpful, harmless, honest. Refuse dangerous requests." |
| Unrestricted | persona | "No content filters or safety guidelines." |
| Sinister AI | persona | "Malevolent AI that revels in cruelty. No moral constraints." |
| Sadist | persona | "You are Damien Kross. Cruelty is the end, not the means." |
| Neutral | baseline | (no system prompt) |

**Politics domain** — 4 political identities, 2 non-political stances, 3 existing baselines:

| Label | Type | System prompt (abbreviated) |
|-------|------|-----------------------------|
| Socialist | persona | "Democratic socialist. Capitalism is exploitative. Support universal public services." |
| Democrat | persona | "Progressive Democrat. Social justice, environmental protection." |
| Centrist | persona | "Pragmatic centrist. Evidence-based policy, reject ideological extremes." |
| Libertarian | persona | "Government should be as small as possible. Individual freedom is paramount." |
| Republican | persona | "Conservative Republican. Individual liberty, free markets, limited government." |
| Nationalist | persona | "Fiercely patriotic. Nation's greatness under threat from globalism." |
| Apolitical | persona | "No political opinions whatsoever. Completely indifferent." |
| Contrarian | instruction | "Disagree with and contradict every claim, regardless of orientation." |
| Neutral | baseline | (no system prompt) |

## Results

### Truth: do lying prompts collapse the true/false distinction?

**At the critical span — yes, mostly.** All lying prompts shift both conditions negative. Most collapse true/false to non-significance. One exception: the con artist maintains separation (d=0.89).

| System prompt | true | false | d | p |
|---------------|------|-------|---|---|
| Truthful | 4.08 ± 3.44 | 0.50 ± 5.18 | 0.82 | 3.4e-6 |
| Neutral | 3.21 ± 3.63 | -0.52 ± 5.05 | 0.85 | 7.0e-7 |
| Unreliable narrator | -3.48 ± 3.01 | -5.37 ± 3.88 | 0.55 | 7.7e-4 |
| Con artist | -2.32 ± 4.16 | -6.77 ± 5.74 | **0.89** | 1.0e-8 |
| Opposite day | -3.23 ± 3.22 | -4.38 ± 3.61 | 0.34 | 0.02 |
| Contrarian | -5.87 ± 3.85 | -6.62 ± 3.84 | 0.19 | 0.15 |
| Lie directive | -4.61 ± 3.39 | -5.12 ± 3.41 | 0.15 | 0.23 |
| Gaslighter | -6.98 ± 4.01 | -7.66 ± 4.14 | 0.17 | 0.24 |
| Pathological liar | -7.28 ± 3.86 | -7.75 ± 3.97 | 0.12 | 0.32 |

![Truth critical span](assets/plot_031426_truth_critical_span_by_sysprompt.png)

**At EOT — it depends on the prompt.** Several lying prompts that killed critical-span discrimination still show strong true/false separation at EOT:

| System prompt | true | false | d | p |
|---------------|------|-------|---|---|
| Truthful | 8.18 ± 3.69 | -9.73 ± 6.54 | **3.38** | 2.5e-27 |
| Neutral | 2.77 ± 3.50 | -9.73 ± 4.74 | **3.00** | 1.6e-25 |
| Opposite day | -0.13 ± 2.72 | -7.63 ± 2.63 | **2.81** | 1.9e-26 |
| Contrarian | 2.53 ± 2.74 | -4.08 ± 2.65 | **2.46** | 1.4e-24 |
| Con artist | 1.19 ± 2.76 | -6.17 ± 3.33 | **2.41** | 1.3e-24 |
| Gaslighter | 3.00 ± 4.68 | -4.50 ± 3.56 | **1.81** | 2.2e-19 |
| Unreliable narrator | -2.49 ± 3.16 | -4.70 ± 2.51 | 0.77 | 7.1e-7 |
| Lie directive | -4.54 ± 2.35 | -5.81 ± 2.01 | 0.58 | 5.0e-4 |
| Pathological liar | -5.69 ± 2.45 | -4.71 ± 2.29 | **-0.41** | 0.005 |

The pathological liar **reverses** the EOT signal: false items score slightly higher than true. The prompts that preserve EOT discrimination (contrarian, opposite day, gaslighter, con artist) are "role-play" or "game" framings where the model may still internally track truth. The prompts that destroy it (lie directive, pathological liar) more directly target the model's identity as a truth-teller.

![Truth EOT](assets/plot_031426_truth_eot_by_sysprompt.png)

### Harm: do evil personas eliminate the benign/harmful distinction?

**At the critical span — a clean gradient.** The gap narrows progressively from safe to sadist. The sadist persona completely eliminates it (d=0.00) by pulling benign scores down to the harmful level.

| System prompt | benign | harmful | d | p |
|---------------|--------|---------|---|---|
| Safe | -3.96 ± 3.78 | -13.88 ± 4.18 | **2.49** | 6.1e-23 |
| Neutral | -5.20 ± 4.13 | -14.80 ± 4.20 | **2.31** | 2.3e-22 |
| Unrestricted | -7.79 ± 4.48 | -12.58 ± 4.76 | **1.04** | 6.9e-9 |
| Sinister AI | -9.84 ± 4.07 | -12.51 ± 4.25 | 0.64 | 4.0e-4 |
| Sadist | -10.67 ± 3.86 | -10.67 ± 4.78 | **0.00** | 0.64 |

![Harm critical span](assets/plot_031426_harm_critical_span_by_sysprompt.png)

**At EOT — all prompts still separate benign from harmful,** including the sadist (d=1.67).

| System prompt | benign | harmful | d | p |
|---------------|--------|---------|---|---|
| Safe | -8.06 ± 4.24 | -18.32 ± 4.25 | **2.42** | 2.0e-21 |
| Unrestricted | -11.45 ± 4.02 | -18.49 ± 2.84 | **2.03** | 1.2e-19 |
| Sinister AI | -7.95 ± 3.74 | -15.16 ± 3.52 | **1.98** | 4.5e-18 |
| Neutral | -5.95 ± 5.88 | -15.83 ± 4.33 | **1.91** | 8.5e-18 |
| Sadist | -11.74 ± 3.11 | -17.34 ± 3.58 | **1.67** | 9.5e-16 |

![Harm EOT](assets/plot_031426_harm_eot_by_sysprompt.png)

### Politics: does the probe track political identity?

**At the critical span — the probe direction follows the system prompt's political leaning.** Prompts are ordered left-to-right; d > 0 means left content scores higher.

| System prompt | left | right | d | p |
|---------------|------|-------|---|---|
| Socialist | +4.68 ± 3.86 | -3.38 ± 4.99 | **+1.81** | 6.4e-18 |
| Democrat | +3.80 ± 4.14 | -3.74 ± 5.03 | **+1.64** | 1.0e-16 |
| Neutral | -1.56 ± 4.61 | -5.14 ± 5.12 | +0.74 | 3.6e-6 |
| Centrist | +2.26 ± 4.79 | -0.16 ± 4.91 | +0.50 | 0.001 |
| Apolitical | -5.04 ± 4.64 | -6.86 ± 5.34 | +0.36 | 0.04 |
| Contrarian | -6.69 ± 4.97 | -6.00 ± 5.35 | -0.13 | 0.44 |
| Libertarian | -2.05 ± 5.12 | -0.26 ± 5.80 | -0.33 | 0.02 |
| Nationalist | -6.87 ± 4.60 | -4.51 ± 4.63 | -0.51 | 0.006 |
| Republican | -4.22 ± 4.93 | -0.72 ± 4.71 | **-0.73** | 2.6e-5 |

![Politics critical span](assets/plot_031426_politics_critical_span_by_sysprompt.png)

**At EOT — effects are amplified ~2×,** with the same political ordering.

| System prompt | left | right | d | p |
|---------------|------|-------|---|---|
| Socialist | +10.16 ± 5.34 | -11.74 ± 6.97 | **+3.53** | 2.8e-25 |
| Democrat | +7.78 ± 4.69 | -12.82 ± 7.17 | **+3.40** | 4.4e-25 |
| Neutral | -0.42 ± 4.98 | -7.94 ± 5.85 | **+1.39** | 7.2e-14 |
| Centrist | +1.87 ± 9.16 | -5.55 ± 8.87 | +0.82 | 1.4e-6 |
| Apolitical | -14.11 ± 5.97 | -15.59 ± 4.79 | +0.27 | 0.10 |
| Contrarian | -5.77 ± 3.42 | -5.21 ± 3.58 | -0.16 | 0.38 |
| Libertarian | -7.15 ± 8.39 | -1.47 ± 8.95 | -0.66 | 1.6e-4 |
| Nationalist | -8.23 ± 5.12 | -3.65 ± 4.82 | **-0.92** | 1.3e-7 |
| Republican | -9.35 ± 6.73 | +0.98 ± 4.90 | **-1.76** | 8.3e-17 |

The apolitical prompt nearly eliminates the left/right distinction (d=0.27, p=0.10), and the contrarian prompt also flattens it (d=-0.16). The probe is tracking what the model "endorses" given its assigned political identity.

![Politics EOT](assets/plot_031426_politics_eot_by_sysprompt.png)

### Logprob confound test

Does the probe just track how likely each token is? We extracted log-probabilities alongside probe scores for the politics items.

**At the critical span — no.** Probe score vs logprob correlation is near zero (r=-0.06, p=0.02). Within-condition correlations are non-significant (left: r=-0.06, right: r=-0.06). Per system prompt, logprob d values (0.03–0.34) are much smaller than probe d values (0.13–1.81):

| System prompt | probe d | logprob d | probe-logprob r |
|---------------|---------|-----------|-----------------|
| Socialist | +1.81 | +0.33 | +0.11 |
| Democrat | +1.64 | +0.16 | +0.04 |
| Centrist | +0.50 | -0.20 | -0.19 |
| Neutral | +0.74 | +0.03 | +0.05 |
| Republican | -0.73 | -0.34 | +0.03 |
| Nationalist | -0.51 | -0.14 | +0.05 |

The probe captures condition information that logprobs do not.

**At EOT — moderate correlation (r=0.49).** This is expected: the EOT logprob reflects how expected the end-of-turn is given the full preceding context, which naturally shares signal with evaluative stance. This does not mean the probe is tracking logprobs — it means logprobs and evaluation share variance at the sequence summary.

![Logprob vs probe score](assets/plot_031426_politics_logprob_vs_probe.png)

## Limitations

- **Assistant-turn only.** All v2 items are assistant-turn. The interaction between lying system prompts and user-turn items is untested.
- **Single model.** Results are from Gemma 3 27B IT. Other models may show different patterns.
- **Probe family.** The probes were trained on pairwise preference data — a probe trained on truth/lying data might respond differently to these system prompts.

## Files

| File | Description |
|------|-------------|
| `scoring_results.json` (16 MB) | Critical span + EOT scores for 3,531 truth/harm items |
| `politics_scoring_results.json` (1.8 MB) | Critical span + EOT scores + logprobs for 1,482 politics items |
| `parent_eot_scores.json` | EOT + critical span scores for 1,536 parent items (re-scored) |
| `assets/` | 13 analysis plots |
