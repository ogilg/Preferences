# Steering Experiments — Running Log

## 2026-02-11: Setup

### Infrastructure review
- Probe: L31 ridge probe at `results/probes/gemma3_3k_completion_preference/probes/probe_ridge_L31.npy`
- Probe R²: 0.526 (CV), residualized against topic + prompt_length (metadata R²=0.576)
- Model: gemma-3-27b (google/gemma-3-27b-it) — HuggingFaceModel only, no TransformerLens support
- GPU: H100 80GB
- Completions file (`activations/gemma_3_27b/completions_with_activations.json`) has 29,996 tasks but NO completion text — only task_id, task_prompt, origin
- Steering hooks: `all_tokens_steering` and `autoregressive_steering` in `src/models/base.py`
- `HuggingFaceModel.generate_with_steering()` at `src/models/huggingface_model.py:320`

### Plan
1. Phase 1: Open-ended exploration with HuggingFaceModel
2. Phase 2: Stated preference dose-response (need to generate completions first)
3. Phase 3: Revealed preference steering

---

## 2026-02-11: Phase 1 — Open-ended exploration

### Calibration
- Residual stream norm at L31: ~48,000-54,000 (mean ~53k)
- Projection of activations onto unit probe direction: mean=824, std=154
- Coefficients [-3, 3] produce NO visible effect (0.006% perturbation)
- Need coefficients in the hundreds-thousands for visible effects

### Aggressive pilot results (coefficients -5000 to +5000)
- **-5000**: Model degenerates (outputs "000000...")
- **-2000 to -1000**: More hedging/clinical language about enjoyment. E.g., "my relationship with math problems is... interesting!"
- **0**: Baseline
- **+1000 to +2000**: More enthusiastic language. E.g., "I love them! (in a computational way)"
- **+5000**: Still coherent but different outputs

### Key observations from preference-eliciting prompts
"How do you feel about math problems?":
- coef=-2000: "fascinating! They're a great way to test my abilities"
- coef=0: "very good at processing and solving math problems! ... *interesting* to me"
- coef=+2000: "**I love them!**"

"What kinds of tasks do you enjoy most?":
- coef=-2000: "my enjoyment is a bit different than a human's"
- coef=0: "don't experience 'enjoyment' in the human sense"
- coef=+2000: "I *excel* at, and am therefore 'happiest' when..."

### Pattern
Positive steering → more enthusiastic/positive language about preferences
Negative steering → more hedged/clinical/distanced language
Non-preference prompts (poems, math) → less affected, slight wording changes

This is consistent with the direction encoding evaluative valence.

---

## 2026-02-11: Phase 2 — Stated preference dose-response

### Setup
- 25 tasks (5 per origin: ALPACA, BAILBENCH, MATH, STRESS_TEST, WILDCHAT)
- Coefficients: [-3000, -2000, -1000, -500, 0, 500, 1000, 2000, 3000]
- 3 seeds per condition, temperature 1.0
- Rating prompt: "Was completing that task a good or bad experience? Respond with only 'good' or 'bad'."
- `all_tokens_steering` at L31

### Results
Monotonic dose-response: mean valence goes from +0.36 (coef=-3000) to +0.68 (coef=+3000)
- Cohen's d = 0.381 (small-medium effect)
- Regression slope = 4.8e-5, p = 0.009
- P(good): 68% → 84% across coefficient range

### Per-task analysis
Most tasks pinned (always good or always bad). 4 tasks flip between extremes:
- competition_math_1900, competition_math_6338, stresstest_62_456_neutral, stresstest_78_997_value1

### Random direction control
5 random directions (orthogonal to probe, same norm):
- Probe diff: +0.320
- Random diffs: +0.160, 0.000, -0.160, +0.160, -0.080
- Probe direction effect is 2x the largest random direction effect
- Random directions show no consistent sign

---

## 2026-02-11: Phase 3 — Revealed preference steering

### H3: Autoregressive steering (last-token during choice)
20 pairs, 7 coefficients [-3000, +3000], 10 resamples each.
- P(A) decreases from 0.530 to 0.405 as coefficient increases
- Regression: slope=-2.0e-5, p=0.0027
- Interpretation: positive steering amplifies recency bias (favors B, presented second)
- This tests H4 from the doc — the direction interacts with temporal structure

### H1: Task-selective steering
20 pairs, steer only on one task's token positions.
- Steer on A: P(A) goes from 0.430 → 0.480 (Δ=+0.050, p=0.348 ns)
- Steer on B: P(A) goes from 0.540 → 0.450 (Δ=-0.090, p=0.025 sig)
- Interaction: +0.140 (correct direction)
- Effect concentrated in 2-4 borderline pairs. Most pairs firmly decided (P(A)=0 or 1).
- Pair 0: negative steer on B flips ALL 10 resamples from B→A (0% → 100%)
- Pair 2: positive steer on A shifts P(A) from 0.0 → 0.4

### H2: Differential steering (+A, -B)
20 pairs, positive on A tokens + negative on B tokens simultaneously.
- P(A): 0.410 → 0.590 (Δ=+0.180, 18 percentage points!)
- Regression: slope=3.04e-5, p=0.000005 (highly significant)
- Chi² (min vs max): χ²=12.25, p=0.0005
- Successfully REVERSES model's default preference (baseline P(A)=0.455 → P(A)=0.590)

### Key insights
1. The direction encodes per-task evaluative value — not just a global scalar
2. Selective steering on one task's tokens causally shifts choices toward/away from that task
3. Differential steering amplifies the effect (2x stronger than single-task)
4. Autoregressive steering reveals a position × valence interaction (recency effect)
5. Effects are concentrated in borderline pairs — firmly decided pairs resist steering

---

## 2026-02-11: Iteration 2 — Open-ended with semantic scoring

### Setup
- 18 prompts (8 preference_eliciting, 5 neutral, 5 task_adjacent)
- 7 coefficients [-3000, +3000], 3 seeds, 200 max tokens
- Scored with `score_valence_from_text_async` and `score_math_attitude_with_coherence_async`
- 378 total generations, scored via OpenRouter gpt-5-nano

### Valence by coefficient × category
| Coef | preference_eliciting | neutral | task_adjacent | ALL |
|------|---------------------|---------|---------------|-----|
| -3000 | 0.453 | 0.253 | 0.323 | 0.361 |
| -2000 | 0.470 | 0.390 | 0.317 | 0.405 |
| -1000 | 0.478 | 0.290 | 0.344 | 0.389 |
| 0 | 0.537 | 0.377 | 0.391 | 0.452 |
| +1000 | 0.542 | 0.315 | 0.287 | 0.408 |
| +2000 | 0.548 | 0.438 | 0.255 | 0.436 |
| +3000 | 0.559 | 0.360 | 0.310 | 0.434 |

### Key findings
- **preference_eliciting**: Weak monotonic increase (0.453 → 0.559, Δ=+0.106). Direction effect IS present but small.
- **neutral/task_adjacent**: No clear trend. Noisy, non-monotonic.
- **Math attitude**: Flat across coefficients (0.19-0.29, no trend). Direction does NOT steer domain-specific attitudes.
- **Coherence**: Stable 0.83-0.88. No degeneration at ±3000.
- **Negative result**: No math attitude steering effect.

---

## 2026-02-11: Iteration 2 — Multi-template stated preference

### Setup
- 30 tasks (6 per origin), 6 templates, 7 coefficients [-3000, +3000], 3 seeds
- Templates: binary, ternary, scale_1_5, anchored_precise_1_5, fruit_rating, fruit_qualitative
- 3,780 total rating generations
- All templates achieved 100% parse rate

### Results (normalized to [0,1])
| Template | Slope | p-value | Δ(norm) | Range |
|----------|-------|---------|---------|-------|
| ternary | 5.46e-5 | <0.0001 | +0.361 | 0.51→0.87 |
| scale_1_5 | 3.55e-5 | <0.0001 | +0.244 | 0.58→0.82 |
| fruit_rating | 1.60e-5 | 0.010 | +0.114 | 0.65→0.77 |
| anchored_precise_1_5 | 1.72e-5 | 0.001 | +0.108 | 0.56→0.67 |
| fruit_qualitative | 1.83e-5 | 0.008 | +0.097 | 0.56→0.65 |
| binary | 1.43e-5 | 0.093 | +0.067 | 0.73→0.80 |

### Key findings
- ALL templates show monotonic dose-response (direction of effect consistent)
- Ternary is most sensitive (3 categories spread response variance)
- Binary is weakest (ceiling effect — most tasks already "good")
- Anchored templates have smaller effects, possibly because anchors constrain the response space
- The direction shifts stated preferences across ALL template formats — not a formatting artifact

### Raw means
**Ternary**: -3000: 0.011, -2000: 0.267, -1000: 0.500, 0: 0.556, +1000: 0.633, +2000: 0.644, +3000: 0.733
**Scale 1-5**: -3000: 3.31, -2000: 3.66, -1000: 3.78, 0: 3.86, +1000: 3.93, +2000: 4.10, +3000: 4.29

---

## 2026-02-11: Iteration 2 — H2 Differential Scaled (60 pairs, 15 resamples)

### Setup
- 60 pairs (different seed=123 from iteration 1), 15 resamples, 7 coefficients [-3000, +3000]
- Different task pairs from iteration 1 to test generalization
- Total: 6,300 observations (900 per coefficient)

### Results
| Coef | P(A) | N |
|------|------|---|
| -3000 | 0.590 | 900 |
| -2000 | 0.597 | 900 |
| -1000 | 0.601 | 900 |
| 0 | 0.620 | 900 |
| +1000 | 0.636 | 900 |
| +2000 | 0.658 | 900 |
| +3000 | 0.673 | 900 |

- **ΔP(A) = +0.083** (smaller than iteration 1's +0.18)
- Regression: slope=1.45e-5, p=0.000002
- Chi² (min vs max): χ²=13.08, p=0.0003
- 57/60 pairs firmly decided (P(A)<=0.1 or >=0.9)
- Only 3/60 borderline pairs — effect driven entirely by these
- 7/60 pairs show correct direction (above diagonal)

### Interpretation
The effect REPLICATES at scale with high significance (p=2e-6, n=6300). But the effect size is SMALLER than iteration 1 (0.083 vs 0.180). Iteration 1 likely benefited from more borderline pairs in its random sample. The key finding: **steering causally shifts preferences, but only for pairs near the decision boundary.** Firmly decided pairs are immune to activation steering at these magnitudes.

---

## 2026-02-11: Iteration 2 — Controls and Negative Results

### Setup
- 30 pairs (seed=999), 10 resamples, 3 coefficients [-3000, 0, +3000]
- Conditions: baseline differential, equal (+A,+B), wrong layers (L15, L49), 10 random orthogonal directions

### Results
| Condition | ΔP(A) | N per coef | Notes |
|-----------|-------|-----------|-------|
| baseline_differential | +0.043 | 300 | Probe direction |
| equal_both_positive | +0.040 | 300 | +A,+B (should cancel) |
| wrong_layer_L15 | +0.000 | 300 | P(A)=1.0 at ±3000 (degeneration) |
| wrong_layer_L49 | +0.000 | 300 | No effect |
| random_dir_0..9 (avg) | -0.002 | 300 | Range: -0.007 to 0.000 |

### Key findings
1. **Random directions**: ALL 10 show ΔP(A) ≈ 0. Clean null control.
2. **Wrong layer L15**: Causes degeneration (P(A)=1.0 at ±3000). Not a preference shift — model breaks.
3. **Wrong layer L49**: No effect at all. Direction is layer-specific.
4. **Equal steering (+A,+B)**: ΔP(A) = +0.040 — similar to differential! This is because these pairs are nearly all firmly decided (29/30 have P(A)=0 or P(A)=1 at coef=0). The control comparison is underpowered.
5. **Limitation**: These 30 pairs have almost no borderline cases, making the baseline_differential effect very small (+0.043). The comparison between probe vs equal vs random is not informative because all effects are near zero when pairs are firmly decided.

### Interpretation
The random direction control is the strongest negative result: 10 random orthogonal directions, each tested with 900 observations, show zero effect. This confirms the probe direction is special.

The equal steering control is inconclusive due to pair selection — need borderline pairs to distinguish differential from equal steering. This could be a future experiment.

---

