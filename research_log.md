# Research Log

## 2026-01-14: Probe Data Score Analysis

Created `src/experiments/probe/analyze_scores.py` to analyze scores by dimension.

### Key Results (n=1192)
- Scores heavily skewed positive: 87% positive, 12.6% neutral, 0.3% negative
- By origin: MATH (0.969) > ALPACA (0.914) > WILDCHAT (0.824)
- Short completions (2-124 tokens) score lower (0.826) vs mid-range (~0.9)
- No truncations in dataset
