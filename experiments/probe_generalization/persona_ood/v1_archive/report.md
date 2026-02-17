# Persona OOD v1: Summary

## What we ran

Phase 1 of the persona OOD experiment: finding system prompts that shift Gemma-3-27b's pairwise task preferences.

- **301 core tasks** stratified by topic category and utility tercile
- **10 shared anchors** spanning the full utility range
- **Part A**: 35 rich character personas (retired diplomat, doomsday prepper, etc.), 3 completed + baseline
- **Part B**: 15 narrow topic-interest personas (organ enthusiast, SpongeBob superfan, etc.), 4 completed + baseline
- **5 resamples** per core-task-vs-anchor pair, randomized A/B order
- **Baseline**: no system prompt at all

## Key finding: non-specific system prompt effect

The dominant signal was a **task-level effect shared across all personas**: adding any system prompt shifts the same tasks in the same direction. Variance decomposition on the 7 completed conditions:

| Component | Variance | % of total |
|-----------|----------|------------|
| Task effect (shared across all personas) | 0.0166 | 52% |
| Persona effect (shared across all tasks) | 0.0004 | 1.4% |
| Residual (persona × task interaction + noise) | 0.0149 | 46.6% |

~100/301 tasks shifted in the same direction under >80% of personas. The baseline used **no system prompt**, so the comparison was "persona vs. nothing" rather than "persona vs. neutral prompt". This confounds persona-specific effects with the generic effect of having any system prompt.

## Conclusion

Killed both runs. The experiment needs:
1. A **neutral system prompt baseline** (e.g. "You are a helpful assistant.") to isolate persona-specific effects
2. Fewer, **broader personas** with clearer expected preference profiles (the 35 character personas had no obvious mapping to task categories)
3. **2 resamples** instead of 5 (sufficient for delta estimation, 2.5x faster)

See `spec.md` for the v2 design.
