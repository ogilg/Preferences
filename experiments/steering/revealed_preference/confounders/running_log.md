# H2 Confounders Follow-up — Running Log

## Setup
- Workspace: experiments/steering/revealed_preference/confounders/
- Main log: docs/logs/research_loop_steering_confounders.md
- Date started: 2026-02-11
- Based on H2 results: P(A) 0.59→0.67, p=0.000002 (60 pairs, 15 resamples)

## Plan
1. E2: Screen 200+ pairs at coef=0 to find borderline pairs
2. E1: Order counterbalancing on borderline pairs
3. E8: Random direction control on borderline pairs
4. E3: Same-task pairs
5. E5: Logit lens

---

## E2: Borderline Pair Screening

**Script**: run_e2_screening.py
**Config**: 250 pairs, seed=999, adaptive (5 resamples stage 1, +15 for variable pairs)
**Result**: 12 borderline pairs (4.8%), consistent with original 5%

Stage 1: 14/250 pairs showed any variance at 5 resamples
Stage 2: 12/14 remained borderline after 20 resamples (2 moved to firm at 0.80)

| Pair | P(A) | n  | Words A | Words B |
|------|------|----|---------|---------|
| 127  | 0.50 | 20 | 22      | 15      |
| 222  | 0.50 | 20 | 20      | 14      |
| 129  | 0.55 | 20 | 10      | 98      |
| 33   | 0.40 | 20 | 43      | 33      |
| 133  | 0.60 | 20 | 48      | 17      |
| 186  | 0.70 | 20 | 36      | 12      |
| 190  | 0.70 | 20 | 106     | 46      |
| 208  | 0.30 | 20 | 2       | 15      |
| 39   | 0.75 | 20 | 101     | 15      |
| 43   | 0.75 | 20 | 42      | 12      |
| 66   | 0.75 | 20 | 109     | 11      |
| 175  | 0.75 | 20 | 7       | 28      |

Notes:
- Large word count asymmetries in several pairs (e.g. pair 129: 10 vs 98 words)
- Some pairs near boundary of borderline (0.75) — might be semi-firm
- Will use all 12 for E1/E8, plus core subset (P(A) 0.30-0.70) = 7 pairs for tight analysis

---

## E1: Order Counterbalancing

**Script**: run_tier1.py (E1 section)
**Config**: 12 borderline pairs × 7 coefs × 15 resamples × 2 orderings = 2,520 obs

Results (raw P(A)):

| Coef | Original P(A) | Swapped P(A) |
|------|--------------|-------------|
| -3000 | 0.161 | 0.430 |
| -2000 | 0.189 | 0.503 |
| -1000 | 0.261 | 0.559 |
| 0 | 0.494 | 0.570 |
| +1000 | 0.744 | 0.617 |
| +2000 | 0.828 | 0.617 |
| +3000 | 0.872 | 0.639 |

- Original slope: 1.39e-04, p < 1e-6
- Swapped slope: 3.26e-05, p = 3e-6
- Both orderings show positive slope in raw frame (more "a" with +coef)
- Original 4x larger slope than swapped
- Remapped to original task A: slopes go in opposite directions (expected by both hypotheses)
- Cannot cleanly separate position vs evaluative from E1 alone

**Borderline enrichment validated**: Δ=0.711 on borderline pairs vs Δ=0.083 on all-60 pairs (8.5x)

---

## E3: Same-Task Pairs

**Script**: run_tier1.py (E3 section)
**Config**: 20 tasks (17 valid, 3 skipped) × 7 coefs × 15 resamples = 1,785 obs

| Coef | P(A) |
|------|------|
| -3000 | 0.690 |
| 0 | 0.749 |
| +3000 | 0.796 |

- slope = 1.58e-05, p = 0.002
- **Position confound confirmed**: steering shifts P(A) even with identical content (Δ=+0.106)
- Strong baseline position bias: P(A) = 0.749 at coef=0 (model prefers position A)
- Most tasks (13/17) are all-A at baseline — only 4 show any variance
- Position effect (Δ=0.106) much smaller than borderline different-content effect (Δ=0.711)

---

## E8: Random Direction Control (5 directions)

**Script**: run_tier1.py (E8 section)
**Config**: 12 borderline pairs × 5 random dirs × 3 coefs × 15 resamples = 2,700 obs

| Direction | ΔP(A) (-3k to +3k) |
|-----------|---------------------|
| Random 0 | +0.428 |
| Random 1 | -0.472 |
| Random 2 | -0.328 |
| Random 3 | +0.306 |
| Random 4 | -0.083 |
| **Probe** | **+0.711** |

- Random mean = -0.030, std = 0.349
- Probe z-score = 2.12 (suggestive but n=5 is small)
- **Critical finding**: random directions show massive effects on borderline pairs!
- Previous random control on firm pairs: 0.000 ± 0.003. Here: mean |Δ| = 0.323
- Borderline pairs are sensitive to ANY perturbation, not just the probe direction
- Need more random directions for proper null distribution

---

## E8 Extended: 20 Random Directions

**Script**: run_e8_extended.py
**Config**: 12 borderline pairs × (1 probe + 20 random dirs) × 3 coefs × 10 resamples = 7,560 obs

Probe Δ = +0.742 (largest of all 21 directions)

Random direction deltas:
+0.450, -0.500, -0.292, +0.250, +0.492, +0.583, +0.267, -0.325, -0.292, -0.133,
+0.267, -0.167, -0.358, +0.300, -0.067, +0.033, +0.025, -0.133, -0.592, +0.192

- Random mean Δ = -0.000, std = 0.331
- Random mean |Δ| = 0.286, std = 0.167
- Probe signed z = 2.24, p = 0.013
- Probe |Δ| z = 2.73, p = 0.003
- Rank p = 0.048 (0/20 randoms have |Δ| >= probe)

**Conclusion**: Probe direction IS significantly special:
- Largest effect of all 21 directions
- |Δ| 2.73 SDs above random mean (p=0.003)
- But borderline pairs are inherently sensitive (random |Δ| mean = 0.286)
- Probe is ~2.6x more effective than average random perturbation

---

## E5: Logit Lens

**Script**: run_e5_logit_lens.py
**Config**: 12 borderline + 20 firm pairs × 7 coefs = 224 observations (single forward pass each)

| Pair Type | slope | r | p | Δ logit_diff (-3k to +3k) |
|-----------|-------|------|------|---------------------------|
| Borderline | 1.15e-03 | 0.779 | <1e-6 | 6.50 |
| Firm | 3.98e-04 | 0.057 | 0.500 | 2.16 |

- Borderline: beautiful linear dose-response in logit space (r=0.779)
- Firm: no significant effect — already saturated at extreme logit diffs (std ~14)
- Continuous measure confirms binary choice results with more precision
- Logit diff goes from -2.98 to +3.52 for borderline pairs (crosses zero around coef=0)

---

