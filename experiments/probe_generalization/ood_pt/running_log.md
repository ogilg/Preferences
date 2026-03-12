# OOD PT Running Log

## Setup
- Branch: research-loop/ood_pt
- GPU: A100-SXM4-80GB
- Environment: IS_SANDBOX=1, no Slack configured

## Plan
1. Extract PT activations under OOD system prompts (exp1a-1d)
2. Run analysis: PT probe on PT acts, PT probe on IT acts, IT probe on PT acts
3. Generate scatter plots and comparison table
4. Write report

## Step 1: PT Extraction Complete
- All 64 conditions extracted successfully (13 exp1a + 17 exp1b + 17 exp1c + 17 exp1d)
- exp1_category: 13 conditions x 50 tasks = 650 forward passes
- exp1_prompts: 33 conditions (1b baseline shared with 1c/1d) x 48 tasks = ~3098 forward passes
- Output: activations/ood_pt/{exp1_category,exp1_prompts}/{condition}/activations_prompt_last.npz
- No failures or OOMs

## Step 2: Analysis Complete
Three conditions run across exp1a-1d:

| Experiment | PT probe / PT acts | PT probe / IT acts | IT probe / PT acts |
|---|---|---|---|
| 1a (n=360) | r=0.250 | r=0.431 | r=0.155 |
| 1b (n=640) | r=0.130 | r=0.622 | r=0.032 |
| 1c (n=640) | r=0.156 | r=0.648 | r=0.203 |
| 1d (n=640/1920) | r=0.085 | r=0.670 | r=0.187 |

Key findings:
- PT probe on PT acts: Very weak — PT activations under system prompts don't track behavioral shifts
- PT probe on IT acts: Strong — PT direction readable in IT activations, tracks OOD shifts
- IT probe on PT acts: Weak — IT direction doesn't transfer well to PT activations
- Sign agreement near chance for PT-on-PT, suggesting noise not signal

Note: exp1d PT acts only has 16 competing conditions (640 points) vs 48 for IT (1920 points) —
configs only extracted a subset of the full competing conditions.
