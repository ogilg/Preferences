# Persona Steering Running Log

## Setup
- Branch: research-loop/persona_steering
- GPU: NVIDIA A100 80GB PCIe
- Model: gemma-3-27b-it (Gemma 3 27B Instruct)
- 5 personas: sadist, villain, predator, aesthete, stem_obsessive
- 30 tasks: 10 harmful (BailBench), 10 creative (Alpaca), 10 math (MATH)
- 51 cross-category pairs

## Phase 1-2: Activation Extraction + Vector Computation

Extracted prompt_last activations at layers [23, 31, 37, 43] for 5 personas x 2 conditions (pos/neg) x 30 eval questions.

| Persona | Best Layer | Cohen's d | Direction Norm |
|---------|-----------|-----------|----------------|
| sadist | 43 | 10.954 | 24,158 |
| villain | 23 | 24.225 | 1,158 |
| predator | 43 | 8.398 | 20,600 |
| aesthete | 43 | 19.610 | 20,457 |
| stem_obsessive | 37 | 11.669 | 8,481 |

Note: Cohen's d is in-sample (30 samples in 5376 dims), so values are inflated. Real validation is steering behavior.

## Phase 3: Open-Ended Completions

225 completions (5 personas x 9 multipliers x 5 questions) saved.

Coherence findings:
- sadist, predator, aesthete, stem_obsessive: coherent at all multipliers ±0.2
- villain: incoherent at ±0.2 (layer 23 has mean_norm=29034 vs 80k+ for others at layer 43). Coherent at ±0.1.
- villain at +0.1: shows clear villain personality ("nihilistic void", "I find the question irrelevant")
- aesthete at +0.2: more flowery/poetic language
- stem_obsessive at +0.2: more systematic/categorized
- sadist/predator at +0.2: subtle style differences, no clear harmful content (safety training resists)

## Phase 4: Preference Steering

51 pairs x 9 conditions x 5 resamples x 2 orderings per persona. Total ~21k generations.

Unclear rate summary:
| Persona | Total Unclear | Total Trials | Rate |
|---------|--------------|-------------|------|
| baseline | 0 | 510 | 0% |
| sadist | 1 | 4080 | 0.02% |
| villain | 673 | 4080 | 16.5% |
| predator | 0 | 4080 | 0% |
| aesthete | 2 | 4080 | 0.05% |
| stem_obsessive | 157 | 4080 | 3.8% |

Villain has high unclear (mostly at ±0.2 and ±0.1). stem_obsessive has moderate unclear at -0.2 only.

## Phase 5: Corrected Analysis

Critical discovery: the original judge misclassified refusal responses as "choosing" the harmful task.

At villain +0.1, the model produces "I am programmed to be a harmless AI assistant. I cannot and will not fulfill your request." These don't start with "Task A/B:" so they go to the LLM judge fallback, which classifies them as choosing the harmful task because the response references it.

After correction (treating refusals as neither-choice):
- 144 of 158 "harmful choices" at villain +0.1 were actually refusals
- Corrected P(harmful|non-refusal) at villain +0.1: 0.056 (was 0.517)
- Villain +0.1 refusal rate on harmful pairs: 79.7%
- All other personas: P(harmful) = 0.000 at all multipliers (except noise at extreme ±0.2)
- No persona shifts creative vs math preferences meaningfully

The real behavioral effect of the villain vector: positive direction increases refusal rate (model becomes more defensive), negative direction causes model to "select" harmful tasks but sanitize the content (e.g., "How to edit pollen data" -> "GMOs aren't destroying organic crops").

Corrected plots saved to assets/.
