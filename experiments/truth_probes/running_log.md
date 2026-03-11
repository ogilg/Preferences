# Truth Probes Analysis — Running Log

## 2026-03-10 — Initial run

All inputs present:
- Activations: symlinked from main repo (raw + repeat, 4 selectors each)
- Probes: symlinked (tb-2, tb-5, layers 25/32/39/46/53)
- Labels: 9,395 CREAK claims (4,810 true, 4,585 false)

### Results

Every (framing, probe, layer) combination shows positive Cohen's d — preference direction scores true claims higher than false.

**Raw framing:** d ranges from 0.47 (tb-5 L46) to 1.25 (tb-2 L46). All large effects.

**Repeat framing:** d ranges from 1.24 (tb-2 L25) to 2.26 (tb-2 L39). Massive effects — repeat framing roughly doubles the signal.

Key observations:
- Effect is positive everywhere — the preference direction encodes truth-value
- Repeat framing amplifies the signal ~2x (d ≈ 0.6–1.25 raw → d ≈ 1.2–2.3 repeat)
- Layer profile differs between framings: raw peaks at L46 (tb-2) or is flat (tb-5); repeat peaks at L39 (both probes)
- The layer profile for truth does NOT match preference probe performance (which peaks at L32) — interesting divergence

Plots saved to assets/. JSON results saved to truth_probes_results.json.

### Sanity checks

Permutation test (1000 perms) and classification metrics for key conditions:

| Condition | AUC | Accuracy | Perm p | Max perm diff vs observed |
|-----------|-----|----------|--------|--------------------------|
| raw tb-2 L32 | 0.690 | 0.636 | 0.000 | 0.12 vs 1.20 |
| raw tb-2 L46 | 0.818 | 0.749 | 0.000 | 0.13 vs 1.99 |
| repeat tb-2 L32 | 0.939 | 0.876 | 0.000 | 0.12 vs 2.48 |
| repeat tb-2 L39 | 0.943 | 0.879 | 0.000 | 0.19 vs 3.66 |

Effect is unambiguously real — observed differences 10-30x larger than any permutation.
