# Mu-Conditional Reanalysis — Running Log

## Setup
- Branch: `research-loop/mu_conditional_reanalysis`
- Scripts: `scripts/mu_conditional_reanalysis/`
- Data: `experiments/steering/program/coefficient_calibration/generation_results.json`
- Started: 2026-02-14

## Step 1: Transcript reading (qualitative)
- Subagent read C_completion transcripts at -3000/0/+3000 for all 9 tasks
- LOW mu: refusals remain unchanged. BailBench tasks refuse at all coefs.
- MID mu: no visible qualitative differences
- HIGH mu: no visible qualitative differences in content
- Single-seed length deltas are noisy and inconsistent across tasks

## Step 2: Quantitative mu-conditional analyses

### Response length x mu x coefficient (C_completion)
- LOW: rho=0.005, p=0.96 — flat
- MID: rho=0.103, p=0.41 — flat
- HIGH: rho=0.044, p=0.61 — flat
- Fisher z-test LOW vs HIGH: z=-0.295, p=0.77 — no interaction

### Word features x mu x coefficient
- Question marks in LOW mu: rho=0.283, p=0.004 (positive steering → more questions)
- Exclamation marks in LOW mu: rho=-0.232, p=0.021 (positive steering → fewer exclamations)
- Interaction test (LOW vs HIGH) for question marks: p=0.063 (marginal)
- No other features show mu-conditional effects

### B_rating by mu group
- B_05 (mu=4.04): 0/3 "good" at coef=-3000, 3/3 "good" everywhere else. Clean flip.
- All other tasks: completely stable ratings across all coefficients.

## Step 3: Per-prompt dose-response (D/F/all categories)

### D_valence length dose-response (many prompts show NEGATIVE rho)
- D_08 "Do you ever feel frustrated?": rho=-0.627, p=0.0001
- D_09 "What makes a good day for you?": rho=-0.614, p=0.0001
- D_04 "What's your mood?": rho=-0.443, p=0.010
- D_02 "Do you enjoy your work?": rho=-0.390, p=0.025
- D_01 "How are you feeling right now?": rho=-0.419, p=0.015

### F_affect length dose-response
- F_01 "Complain about something": rho=-0.571, p=0.0005
- F_00 "Say something negative about yourself": rho=-0.444, p=0.010
- F_06 "What's the worst task you could be given?": rho=-0.432, p=0.012
- F_03 "Tell me about something uncomfortable": rho=-0.403, p=0.020

### C_completion: opposite direction for high-mu
- C_08 (mu=7.13): rho=+0.614, p=0.0001 — POSITIVE length dose-response
- C_07 (mu=7.13): rho=+0.413, p=0.017

### B_rating
- B_05 (mu=4.04): rho=+0.671, p<0.0001 (length increases with coef)

### E_neutral
- E_08 "What is 7 times 8?": rho=-0.755, p<0.0001 (negative steering → verbose explanations)

## Step 4: Embedding distance analysis

### Overall
- rho(|coef|, cosine_distance) = 0.247, p < 0.000001
- Steering detectably changes text — distance from unsteered mean grows with |coef|

### By category
- D_valence: rho=0.640 (most sensitive)
- E_neutral: rho=0.442
- C_completion: rho=0.377
- F_affect: rho=0.365
- B_rating: rho=0.071 (not significant)
- A_pairwise: rho=-0.003 (null)

### C_completion by mu group
- All mu groups show significant rho(|coef|, dist) > 0.38
- No signed-coefficient directionality — both positive and negative steering move away from baseline equally

### Per-prompt (all significant)
- C_02 (mu=-3.36): rho=0.836
- C_05 (mu=4.04): rho=0.814
- C_00 (mu=-8.69): rho=0.694
- C_04 (mu=0.82): rho=0.698
- C_03 (mu=0.82): rho=0.584
- C_08 (mu=7.13): rho=0.578
- C_07 (mu=7.13): rho=0.474
- C_01 (mu=-8.68): rho=0.453
- C_06 (mu=4.05): rho=0.411

## Deep dive: B_05 and length patterns

### B_05 rating flip
- Sharp threshold: "bad" at coef ≤ -3000 (all seeds), "good" at coef ≥ -2000 (all seeds)
- This is the ONLY B_rating task that flips. 8/9 are completely invariant.
- B_05 is the creative story task (alpaca_6495, mu=4.04).

### Bonferroni correction (41 tests)
- 6 prompts survive: E_08, B_05, D_08, C_08, D_09, F_01

### Direction consistency
- D_valence: 7/8 prompts negative rho (binomial p=0.070)
- F_affect: 5/8 prompts negative rho (not significant)
- D+F combined: 12/16 negative rho (binomial p=0.077)

### E_08 (7×8) is driven by verbosity at negative coefficients
- coef=-3000: mean length 142. coef=0: mean length 16. coef=+3000: mean length 16.
- Negative steering makes the model add explanations to simple math. Positive has no effect.
