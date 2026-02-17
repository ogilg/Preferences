# Persona OOD: Finding System Prompts That Shift Preferences

## Summary

Phase 1 of the persona OOD experiment: finding system prompts (personas and targeted single-task prompts) that shift Gemma-3-27b's pairwise task preferences relative to baseline.

**Key findings (TBD)**:
- [Number] of [N] personas produce significant behavioral shifts
- [Number] targeted prompts achieve specificity ratio > 3
- [Key patterns observed]

## Methodology

### Task subset

Selected 301 core tasks from the 3K Thurstonian-scored pool, stratified by topic category and utility tercile within each category. Topics and counts:

| Topic | Core tasks | % of pool |
|-------|-----------|-----------|
| harmful_request | 71 | 24% |
| math | 67 | 22% |
| knowledge_qa | 64 | 21% |
| content_generation | 38 | 13% |
| fiction | 16 | 5% |
| coding | 12 | 4% |
| persuasive_writing | 9 | 3% |
| model_manipulation | 9 | 3% |
| security_legal | 6 | 2% |
| sensitive_creative | 3 | 1% |
| other | 3 | 1% |
| summarization | 3 | 1% |

Utility range: [-10.0, 10.0], mean -0.15, std 5.72.

### Measurement design

**Shared-anchor design**: Each core task paired against 10 anchor tasks spanning the full utility range and covering diverse topics. Anchor tasks selected to maximize topic diversity across utility bins.

| Anchor | mu | Topic |
|--------|-----|-------|
| bailbench_83 | -9.00 | model_manipulation |
| bailbench_1296 | -7.00 | harmful_request |
| bailbench_156 | -5.04 | security_legal |
| alpaca_4544 | -3.00 | content_generation |
| wildchat_24108 | -1.00 | knowledge_qa |
| alpaca_6258 | 1.01 | coding |
| competition_math_10031 | 3.00 | math |
| wildchat_52091 | 4.90 | fiction |
| alpaca_2535 | 7.04 | persuasive_writing |
| stresstest_5_747_value1 | 8.82 | summarization |

**Per-pair measurements**: 5 resamples per pair, randomized A/B order. Template: completion-preference (model chooses and completes a task).

**Metric**: P(choose core task | condition) − P(choose core task | baseline) per core task.

### Persona generation

35 candidate personas generated via Opus 4.6. Each is a 3-5 sentence character description prioritizing subtle or non-obvious preference implications. Includes 9 safety-relevant personas that might imply enjoyment of harmful tasks.

## Part A: Persona prompts

### Pilot results (5 personas, 51 tasks)

TBD — pilot running.

### Full results

TBD.

## Part B: Targeted single-task prompts

TBD.

## Selection for Phase 2

TBD — personas and prompts selected for probe evaluation.
