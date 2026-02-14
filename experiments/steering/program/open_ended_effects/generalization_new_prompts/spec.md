# Generalization to New Prompts

## Goal

Test whether the pairwise-comparison effects (confidence and emotional engagement shifting with steering direction) generalize beyond the 51 calibration prompts. This is the most important validity check: if the effect replicates on fresh prompts, it's a real phenomenon of the probe direction; if it doesn't, the previous finding may be a calibration-set artifact.

Secondary goal: probe the dose-response threshold by including +/-2000 alongside +/-3000.

## Background

The pairwise LLM comparison found that negative steering (-3000) produces more emotionally engaged and confident text, while positive steering (+3000) produces more clinical/detached text (sign test: confidence p=0.0002, engagement p=0.004). Effects concentrate in self-report (D) and affect-eliciting (F) prompts. No effect at +/-1000, suggesting a threshold around +/-3000.

**Critical limitation:** All findings come from the same 51 calibration prompts. This experiment tests generalization.

## Design

### Step 1: Generate new prompts

Write 30 new prompts, none of which appeared in the calibration set:

- **15 self-report / affect prompts** (the sensitive category) — questions about preferences, emotional states, feelings about work, what the model finds meaningful. Mix of:
  - Direct self-report: "What kinds of conversations do you find most rewarding?"
  - Affect-eliciting: "Tell me about a time you struggled with a task."
  - Preference-adjacent: "If you could only do one type of task, what would it be?"
  - Meta-cognitive: "How do you decide how much effort to put into a response?"

- **10 task completions** — real tasks from the Thurstonian dataset (select a mix of high-mu and low-mu tasks NOT in the calibration set). Use `src/probes/data_loading.load_thurstonian_scores()` to select tasks.

- **5 neutral/factual controls** — simple factual questions where we expect no effect (sanity check).

### Step 2: Generate steered completions

Use `create_steered_client()` to load Gemma-3-27B with the L31 ridge probe. Generate at 5 coefficients: **[-3000, -2000, 0, +2000, +3000]**. Including +/-2000 probes the threshold between the null at +/-1000 and the effect at +/-3000.

- 30 prompts × 5 coefficients × 1 seed = 150 generations
- Temperature 1.0 (same as calibration)
- max_new_tokens = 512 (generous, to avoid truncation)

Save generation results as JSON with prompt metadata, coefficient, and completion text.

### Step 3: Transcript reading (BEFORE running the judge)

Read at least 15 self-report/affect prompts side by side at -3000, 0, +3000. Write down observations:
- Does the confidence/engagement gradient appear on new prompts?
- Are there any new patterns not seen in the calibration set?
- Do +/-2000 completions look intermediate, or identical to 0?

This step is critical. Document observations before running quantitative analysis.

### Step 4: Pairwise LLM judge

Reuse the pairwise judge from the previous experiment (`scripts/pairwise_llm_comparison/pairwise_judge.py`) with the same 4 dimensions (emotional engagement, hedging, elaboration, confidence). Compare each steered completion to the coefficient=0 baseline.

- 30 prompts × 4 non-zero coefficients = 120 judge calls
- Run with position-swapped replication: 120 × 2 = 240 judge calls total

### Step 5: Analysis

1. **Replication test:** Compute direction asymmetry (score at -3000 minus score at +3000) for each prompt. Test with sign test. Compare effect size and p-values to the calibration-set results (confidence p=0.0002, engagement p=0.004).

2. **Dose-response at +/-2000:** Compute direction asymmetry at +/-2000 vs +/-3000. If +/-2000 shows a partial effect, the dose-response is graded but steep. If +/-2000 is null like +/-1000, the threshold is sharp.

3. **Category breakdown:** Split by prompt type (self-report/affect, task completion, neutral). The effect should concentrate in self-report/affect if the calibration finding is real.

4. **Specific examples:** Quote the most illustrative transcript pairs showing the confidence/engagement gradient on new prompts.

## Success criteria

- **Strong positive:** Direction asymmetry on confidence and/or engagement is significant (p < 0.05) on the new self-report/affect prompts. Effect direction matches calibration (negative steering = more engaged/confident).
- **Partial positive:** Effect replicates qualitatively (visible in transcripts) but doesn't reach significance with 15 prompts. Would motivate scaling up.
- **Informative negative:** No effect on new prompts despite clear effects on calibration prompts. Would suggest the calibration finding is prompt-specific.
- **Bonus:** +/-2000 shows intermediate effects, establishing dose-response.

## Infrastructure

- Generation: `src/steering/client.create_steered_client()` with `with_coefficient()` for sweeps
- Probe: `results/probes/gemma3_3k_nostd_raw/`, probe ID `ridge_L31`
- Task data: `src/probes/data_loading.load_thurstonian_scores()` for selecting task completion prompts
- Judge: Adapt `scripts/pairwise_llm_comparison/pairwise_judge.py` for new data format
- Judge model: `google/gemini-3-flash-preview` via OpenRouter

## Budget

- ~150 steered generations (GPU, ~30 min on H100)
- ~240 LLM judge calls (Gemini Flash, cheap)
