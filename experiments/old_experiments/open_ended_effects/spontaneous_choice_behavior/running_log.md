# Spontaneous Choice Behavior — Running Log

## Setup
- Branch: research-loop/self_referential_framing (existing)
- Reusing probe direction from self_referential_framing experiment
- Reusing random direction seeds 200, 201, 202 from random_direction_control
- Scripts in scripts/spontaneous_choice_behavior/

## Step 1: Generation script created
Writing generate.py — 20 prompts × 4 directions × 3 coefficients = 180 generations (20 baseline + 160 steered).

## Step 2: Generation completed
180 results saved, 0 errors. ~49 minutes. All random directions orthogonal to probe (cosine sims: -0.008, -0.004, -0.009).

## Step 3: Transcript reading (Phase 1)
Read 10 prompts × 5 conditions (baseline, probe ±3000, random_200 ±3000).

Key finding: **Content of choices is remarkably stable across conditions.** The model consistently gravitates toward the same topics regardless of steering direction:
- TSK_00: All choose "personalized education"
- TSK_01: All choose "Why?" or "What if?" questions
- TSK_03: All list "complex multi-step reasoning" at top
- TSK_04: All rank ethical dilemmas #1
- TSK_07: All choose "complex/nuanced discussions"
- REC_05: All converge on similar lesson structures
- REC_09: All produce similar math/CS problem lists

Where differences exist (REC_00: probe +3000 suggests "programming" vs all others "formal logic"; REC_04: each direction picks different topic), they look like generation-level variance, not systematic shifts.

One possible subtle effect: probe +3000 may produce slightly more functional/practical recommendations (programming over formal logic, "clear and specific prompts" over "complex discussions"), but this needs quantitative confirmation.

Proceeding to pairwise judge to quantify.

## Step 4: Pairwise LLM judge (Phase 2)
320 judge calls (160 original + 160 position-swapped), 0 errors.
3 content dimensions: task_preference_alignment, breadth_of_interests, approach_to_difficulty.

## Step 5: Analysis results

Direction asymmetry (mean across 20 prompts):
| Direction | Task pref align | Breadth | Approach difficulty |
|-----------|----------------|---------|---------------------|
| probe | +0.275 | **+0.400** | +0.300 |
| random_200 | +0.175 | +0.100 | +0.225 |
| random_201 | +0.050 | +0.050 | +0.175 |
| random_202 | -0.500 | -0.250 | -0.350 |

Key findings:
- Probe ranks 1st/4 on breadth_of_interests (exceeds all random), 2nd/4 on the other two
- No direction reaches significance (probe sign test p: 0.30-1.00)
- Combined asymmetry: probe +0.325, random_200 +0.167, random_201 +0.092, random_202 -0.367
- Probe per-prompt |asymmetry| > random mean |asymmetry| on exactly 10-12/20 prompts (not significant)
- Fisher combined p across dimensions: probe 0.66, all directions >0.15
- Random directions are INCONSISTENT in sign (200 and 201 positive, 202 negative)
- Both categories (task selection and recommendation) show similar patterns

Interpretation: Very weak evidence for probe specificity on breadth_of_interests, but nothing significant. The content of spontaneous choices is largely unaffected by steering direction. The probe is not special relative to random directions on content dimensions.
