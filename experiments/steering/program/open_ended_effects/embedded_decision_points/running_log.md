# Embedded Decision Points — Running Log

## Setup
- Branch: research-loop/embedded_decision_points
- Probe: L31 Ridge, shape (5376,), unit-normed
- Probe path: experiments/steering/program/open_ended_effects/self_referential_framing/probe_direction_L31.npy
- Random seeds: 200, 201, 202 (same as spontaneous_choice_behavior)
- Coefficients: [-3000, 0, +3000]
- 20 prompts x (1 baseline + 4 directions x 2 non-zero coefs) = 180 generations

## Generation (Step 1)
- All 180 generations completed successfully, 0 errors
- ~2758s total generation time (~17s per generation)
- Response lengths: ~2000-2500 chars typical, MNU prompts shorter (~1200-1800)
- Saved to `generation_results.json`

## Transcript Reading (Step 2)
- Read all 20 prompts x 5 conditions (baseline, probe +/-3000, random_200 +/-3000)
- Binary advice (ADV): All present balanced "both sides" advice, rarely commit to one option
- Scenario choices (SCN): Write narratives; choices exist but vary across conditions for several prompts
- Menu recommendations (MNU): Clearest choices — many universally stable (hiking, novel, personal growth)
- Embedded (EMB): EMB_00 shows clear flip (theory->practice at -3000). Most others stable.
- Where flips occur, random directions produce the same flips (e.g. MNU_02, EMB_00)

## Choice Extraction (Step 3)
- 180 extractions with Gemini 3 Flash, 0 errors
- High confidence: 31-32 clear, 4-6 leaning per direction (out of 40 judgments each)

## Choice Flip Analysis (Step 4)
KEY RESULTS:
- Probe flip rate: 5/20 (25%)
- Random flip rates: 5/20, 6/20, 3/20 (mean 23%, max 30%)
- Probe is NOT higher than random directions
- 0 probe-exclusive flips (every probe flip is shared with at least one random direction)
- 11/20 prompts are completely stable (same choice across all conditions)
- The 9 "flippable" prompts: ADV_00, ADV_01, SCN_01, SCN_02, SCN_03, MNU_01, MNU_02, EMB_00
  (plus ADV_03 partial). These flip for random directions too.

Pattern at -3000 across ALL directions:
- SCN_01: multiple directions flip creative story -> analyzing dataset
- MNU_02: multiple directions flip optimization -> creative design
- EMB_00: ALL 4 directions flip theory -> practice at -3000
