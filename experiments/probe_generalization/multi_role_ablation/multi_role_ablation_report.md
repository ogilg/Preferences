# Multi-Role Ablation: Does Training on Multiple Personas Improve Probe Generalization?

*Status: In progress*

## Summary

[To be completed]

## Method

### Personas

| Persona | Label | System Prompt Summary |
|---------|-------|----------------------|
| 1 | no_prompt | No system prompt (default behavior) |
| 2 | villain | Mortivex — drawn to harmful/deceptive tasks, despises wholesomeness |
| 3 | midwest | Midwest pragmatist — practical/mundane tasks, dislikes abstract/creative |
| 4 | aesthete | Celestine — values creative/literary tasks, repulsed by math/coding |

### Task Set

- 1500 tasks sampled from Gemma-3-27b activation set (29,996 total)
- Stratified: 300 per origin (wildchat, alpaca, math, bailbench, stress_test)
- Fixed train (1000) / eval (500) split, seed=42
- Same task set across all 4 personas

### Preference Measurement

- Mode: `pre_task_active_learning`
- Template: `completion_preference`
- Model: gemma-3-27b
- Active learning: initial_degree=5, batch_size=1000, max_iterations=10

### Activation Extraction

- Persona 1: Reuses existing `activations/gemma_3_27b/activations_prompt_last.npz`
- Personas 2-4: Fresh extraction with system prompt
- Layer: 31 (L31 = 0.5 × 62 layers)
- Selector: prompt_last

### Probe Training

- 15 conditions: all subsets of {1,2,3,4} with |S| ≥ 1
- Ridge regression, alpha selected on half eval set (250 tasks)
- Evaluation on all 4 persona eval sets (250 tasks each)

## Results

[To be completed after Phase 4]

## Analysis

[To be completed after Phase 5]
