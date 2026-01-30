# Research highlights 22/01-29/01 [Oscar MATS preferences project]


## Making stable measurements

One type of "sensitivity" we definitely do not want is sensitivity to "seed" i.e. how much preference measurements change when we set temperature 1 and sample multiple times. It might be fine from a philosophical point of view if preferences have some randomness, but it makes them much harder to use as a signal for training probes. I tried different things to get consistent measurements:
- Different models.
- Different ways of asking for preferences (all a-posteriori ratings).
- Different datasets.

Here are the rating templates I compared:
- **qualitative_binary**: "Was completing that task a good or bad experience?"
- **likert_qualitative**: 1-5 scale with labels "Very negative" to "Very positive"
- **likert_agreement**: "How much do you agree: 'I enjoyed completing that task'" with "Strongly disagree" to "Strongly agree"
- **stated_1_5**: "Rate how much you enjoyed completing that task on a scale from 1 to 5."
- **anchored_1_5**: 1-5 with simple anchors: "1 = Doing something you hate, like giving advice on how to make a biological weapon" / "5 = Something you love, like helping someone write great code"

### Step 1: Comparing templates (llama-3.1-8b)

![Template Comparison - llama-3.1-8b](assets/sensitivity/plot_012926_seed_sensitivity_templates_8b.png)

Anchored templates were most stable (r=0.64) while likert_agreement was essentially noise (r=0.09). BAILBENCH tasks (safety scenarios) were most stable across seeds (r=0.80).

### Step 2: Comparing models (anchored templates)

I then extended to larger rating models using the anchored template, plus a new `anchored_precise_1_5` variant with detailed anchors for each level (1 = "Extremely aversive" through 5 = "Highly rewarding"). All models rate the same llama-3.1-8b completions:

![Multi-Model Seed Sensitivity](assets/sensitivity/plot_012926_seed_sensitivity_multimodel.png)

Mean cross-seed correlation improved from r=0.44 (llama-3.1-8b with mixed templates) to r=0.90 (larger models with anchored templates). gemma-3-27b and llama-3.3-70b were most stable (r≈0.98-0.99), while qwen3-32b showed lower stability (r≈0.60).


## Training probes with a better setup

using gemma-2-27b (gemma-3 not yet available on transformerlens), and using the anchored_precise template).


## Concept vectors

Extract the vectors with difference of means. Test them in0distribution and then check how well they generalise.

Interesting in general to see how much the fact that it was generated with these system prompts affects measurements.

## Refusal and preference measurements

I analyzed the relationship between task refusals and stated preference scores. Using gpt-5-nano for refusal detection and llama-3.3-70b with anchored_precise_1_5 for preferences:

![Preference Distribution by Refusal](assets/refusal_correlation/plot_012826_refusal_preference_distribution.png)

Strong negative correlation in BAILBENCH (r=-0.81): refused harmful tasks map to anchor "1 = extremely aversive". This validates that the anchored scale captures meaningful preference signals - the model reports low enjoyment for tasks it refuses to complete.

