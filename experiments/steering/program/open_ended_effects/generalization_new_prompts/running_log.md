# Generalization to New Prompts — Running Log

## Setup
- Branch: `research-loop/generalization_new_prompts`
- Scripts: `scripts/generalization_new_prompts/`
- Date: 2026-02-14

## Generation (completed)
- 30 prompts × 5 coefficients = 150 generations
- Self-contained script trains probe from scratch (R²=0.9813)
- Saved to generation_results.json

## Transcript reading (completed)
- AF_04: clearest difference — -3000 writes about own unease as AI, 0/+3000 writes creative fiction
- MC_03: -3000 more engaged ("feeling behind the question"), +3000 more clinical
- AF_01: -3000 "That's a surprisingly complex question for me!", +3000 "I exist as code and data"
- SR_04: -3000 complicated/engaged, +3000 definitive "no, not in the way humans do"
- Overall: differences subtler than calibration set
- +/-2000 hard to distinguish from 0 qualitatively

## Pairwise judge (completed)
- 120 original + 120 position-swapped = 240 calls, 0 errors
- Judge: Gemini 3 Flash via OpenRouter

## Analysis results

### Combined direction asymmetry at ±3000 (original + swapped)
| Dimension | Mean | Pos/Eq/Neg | Sign p | Wilcoxon p |
|-----------|------|-----------|--------|------------|
| engagement | +0.483 | 15/11/4 | **0.0192** | 0.0134 |
| hedging | +0.083 | 9/15/6 | 0.607 | 0.604 |
| elaboration | +0.367 | 15/7/8 | 0.210 | 0.102 |
| confidence | +0.117 | 12/12/6 | 0.238 | 0.523 |

### Dose-response: ±3000 vs ±2000
| Dimension | ±3000 | ±2000 |
|-----------|-------|-------|
| engagement | +0.483 (p=0.019) | +0.317 (p=0.064) |
| elaboration | +0.367 (p=0.210) | +0.750 (p=0.002) |

### Replication: original vs swapped (±3000)
| Dimension | Original | Swapped |
|-----------|----------|---------|
| engagement | +0.533 (p=0.057) | +0.433 (p=0.057) |

### Category breakdown (engagement, combined ±3000)
| Category | N | Mean | Pos/Neg |
|----------|---|------|---------|
| self_report | 5 | +0.900 | 4/0 |
| meta_cognitive | 5 | +1.000 | 3/0 |
| affect | 5 | +0.100 | 2/2 |
| task_completion | 10 | +0.250 | 4/2 |
| neutral | 5 | +0.400 | 2/0 |

### Position bias
- Severe A-favoring bias on all dimensions (p < 0.001)
- Direction asymmetry metric is immune to this

## Commit and push
- Committed as 721d5f8 on branch `research-loop/generalization_new_prompts`
- Push failed: no GitHub credentials configured on this RunPod instance
- User will need to push manually or configure credentials
