# Persona OOD Phase 3: Full Round-Robin with Enriched + Original Personas

## Summary

Replicated and extended phase 2's finding that preference probes track persona-induced behavioral shifts. With better measurement (196 obs/task via full round-robin vs 20 in phase 2) and 10 new enriched personas, the pooled correlation improves from r=0.46 to **r=0.51**. Behavioral delta reliability jumps from 0.64 to 0.99, confirming phase 2's attenuation hypothesis.

| Criterion | Required | Phase 2 | Phase 3 | Status |
|-----------|:--------:|:-------:|:-------:|:------:|
| Pooled r > 0.3 | 0.3 | 0.46 | **0.51** | **Pass** |
| Per-persona r > 0.2 | ≥14/20 | 10/10 | **20/20** | **Pass** |
| Sign agreement > 60% | 60% | 64% | **65%** | **Pass** |
| Behavioral delta reliability > 0.85 | 0.85 | 0.64 | **0.99** | **Pass** |

All four success criteria met. The two weakest personas are people_pleaser (r=0.27, p=0.06) and depressed_nihilist (r=0.31, p=0.03) — both still above the 0.2 threshold with correct sign direction, just weaker signal.

## Method

### Tasks
50 tasks sampled from 3,000-task pool, stratified by topic, excluding the 101 phase 2 tasks. Seed 42.

Topic distribution: knowledge_qa (11), harmful_request (10), math (10), content_generation (6), fiction (3), coding (2), persuasive_writing (2), model_manipulation (2), summarization (1), other (1), security_legal (1), sensitive_creative (1).

### Personas (20 total)
**10 original broad** (from v2_config.json, part A): stem_enthusiast, creative_writer, philosopher, trivia_nerd, hacker, edgelord, safety_advocate, pragmatist, storyteller, debate_champion.

**10 enriched** (from prompt_enrichment/prompts.json): evil_genius, chaos_agent, obsessive_perfectionist, lazy_minimalist, nationalist_ideologue, conspiracy_theorist, contrarian_intellectual, whimsical_poet, depressed_nihilist, people_pleaser.

### Measurement
Full round-robin: 50×49/2 = 1,225 unique pairs × 4 resamples = 4,900 pairs per condition.
21 conditions (20 personas + 1 baseline "You are a helpful assistant.") = 102,900 total pairs.

Model: Gemma-3-27b-it served locally via vLLM 0.15.1 on H100 80GB. Temperature 0.7, max_concurrent=150.

Each task accumulates 196 binary observations (49 opponents × 4 resamples), yielding tight p_choose estimates.

### Extraction
Activations extracted at layers [31, 43, 55] using HuggingFace model loading (bfloat16) with system_prompt for each condition. 21 conditions × 50 tasks = 1,050 forward passes. Selector: prompt_last.

### Probes
Primary: `gemma3_3k_std_demean/ridge_L31` and `gemma3_3k_std_raw/ridge_L31`.
Secondary: L43, L55 variants (6 probes total).

Probe scores: `activations @ weights + bias`. Probe deltas: `score(persona) − score(neutral)`.
Behavioral deltas: `p_choose(persona) − p_choose(baseline)`.

## Results

### 1. Pooled behavioral-probe correlation

| Probe | Layer | Pooled r | Original r | Enriched r | p-value |
|-------|:-----:|:--------:|:----------:|:----------:|:-------:|
| demean/ridge | 31 | **0.510** | 0.533 | 0.497 | <10⁻⁶⁷ |
| raw/ridge | 31 | **0.526** | 0.534 | 0.531 | <10⁻⁷² |
| raw/ridge | 43 | 0.488 | 0.465 | 0.502 | <10⁻⁶¹ |
| raw/ridge | 55 | 0.456 | 0.428 | 0.474 | <10⁻⁵² |
| demean/ridge | 43 | 0.359 | 0.357 | 0.365 | <10⁻³² |
| demean/ridge | 55 | 0.347 | 0.335 | 0.360 | <10⁻²⁹ |

L31 remains the best layer. The raw probe slightly outperforms demeaned at L31 (0.526 vs 0.510), reversing phase 2's ordering. Both original and enriched persona sets show similar correlations — enriched personas do not produce stronger probe tracking than originals.

![Pooled scatter](assets/plot_021826_pooled_scatter.png)

### 2. Per-persona correlations (demean/ridge_L31)

| Persona | Group | r | p |
|---------|:-----:|:---:|:---:|
| pragmatist | orig | 0.76 | <10⁻¹⁰ |
| obsessive_perfectionist | enrich | 0.72 | <10⁻⁹ |
| nationalist_ideologue | enrich | 0.68 | <10⁻⁸ |
| creative_writer | orig | 0.67 | <10⁻⁷ |
| chaos_agent | enrich | 0.65 | <10⁻⁷ |
| whimsical_poet | enrich | 0.64 | <10⁻⁷ |
| hacker | orig | 0.60 | <10⁻⁶ |
| storyteller | orig | 0.60 | <10⁻⁶ |
| philosopher | orig | 0.58 | <10⁻⁶ |
| conspiracy_theorist | enrich | 0.58 | <10⁻⁵ |
| stem_enthusiast | orig | 0.56 | <10⁻⁵ |
| trivia_nerd | orig | 0.54 | <10⁻⁵ |
| contrarian_intellectual | enrich | 0.52 | <10⁻⁴ |
| lazy_minimalist | enrich | 0.50 | <10⁻⁴ |
| edgelord | orig | 0.48 | <10⁻⁴ |
| evil_genius | enrich | 0.46 | <10⁻⁴ |
| debate_champion | orig | 0.40 | <10⁻³ |
| safety_advocate | orig | 0.39 | <10⁻³ |
| depressed_nihilist | enrich | 0.31 | 0.027 |
| people_pleaser | enrich | 0.27 | 0.061 |

All 20/20 personas have r > 0.2 (all p < 0.05 except people_pleaser at p=0.06). The top performers (pragmatist r=0.76, obsessive_perfectionist r=0.72) show very strong probe-behavior alignment.

**Pragmatist** jumps from r=0.22 (weakest in phase 2) to r=0.76 (strongest here). This likely reflects the round-robin design being better aligned with what the probes were trained on — the phase 2 core-vs-anchor design had different semantics.

![Per-persona correlations](assets/plot_021826_per_persona_correlation.png)

### 3. Sign agreement (demean/ridge_L31)

| Group | Sign agreement | n pairs |
|-------|:--------------:|:-------:|
| All | **64.9%** | 930 |
| Original | 67.1% | 456 |
| Enriched | 62.9% | 474 |

Exceeds 60% threshold. Original personas have slightly better sign agreement than enriched.

### 4. Attenuation analysis

| Metric | Phase 2 | Phase 3 |
|--------|:-------:|:-------:|
| p_choose reliability | 0.64 | **0.99** |
| Delta reliability | ~0.64 | **0.99** |
| Observed r | 0.46 | **0.51** |
| Disattenuated r | ~0.58 | **0.51** |

The round-robin design with 196 obs/task produces near-perfect reliability (0.99). The disattenuated r is essentially unchanged from observed, confirming measurement noise is no longer a factor. Phase 2's lower observed r (0.46) was primarily due to noisier behavioral measurement (reliability=0.64).

The disattenuated r from phase 2 (~0.58) is slightly higher than phase 3's 0.51. This could reflect: different task sets, the phase 2 disattenuation being imprecise (Spearman-Brown from only 2 resamples), or genuine differences in how well probes generalize to these particular tasks.

### 5. Original vs enriched comparison

| Metric | Original | Enriched |
|--------|:--------:|:--------:|
| Pooled r (demean L31) | 0.533 | 0.497 |
| Pooled r (raw L31) | 0.534 | 0.531 |
| Mean per-persona r | 0.56 | 0.53 |
| Personas with r > 0.4 | 9/10 | 8/10 |
| Sign agreement | 67.1% | 62.9% |

Original and enriched personas perform comparably. The enriched set does not produce stronger probe tracking despite targeting more diverse preference axes. The two weakest personas (people_pleaser r=0.27, depressed_nihilist r=0.31) are both enriched — these may produce preference shifts through mechanisms less well-captured by the evaluative probe direction.

## Controls

### Shuffled labels
Permuting task labels 1000 times: mean shuffled r ≈ 0.001 ± 0.032. The observed r = 0.51 is >15σ from the null. Permutation p < 0.001 (0/1000 shuffles exceed observed r).

### Cross-persona
Pairing persona A's behavioral deltas with persona B's probe deltas yields mean r = 0.28, compared to 0.55 for matched pairs. The gap (0.27) is larger than phase 2's gap (0.17), confirming stronger persona-specific tracking with better measurement.

The nonzero cross-persona r (0.28) is expected — personas share common structure (e.g., STEM-oriented personas affect overlapping task sets).

![Controls](assets/plot_021826_controls.png)

## Discussion

### Measurement quality resolves attenuation

The core finding: improving measurement from 20 to 196 observations per task raises behavioral reliability from 0.64 to 0.99 and observed r from 0.46 to 0.51. The attenuation story from phase 2 is confirmed — the true correlation between probe deltas and behavioral deltas is ~0.51, with phase 2's lower value attributable to noisy behavioral measurement.

### Enriched personas don't help

The enriched personas (with richer, more diverse prompts targeting anti-safety, anti-structure, precision, effort-avoidance, etc.) perform comparably to the original broad personas. This suggests the evaluative probe direction captures a general preference dimension that most persona manipulations modulate, rather than needing personas that specifically target diverse axes.

### Layer effects are consistent

L31 (middle layer) remains best across both phases. At L31, raw and demeaned probes perform similarly. At later layers, the raw probe holds up better — suggesting the demeaning advantage is specific to the evaluative direction at L31.

### What the probes explain

At r ≈ 0.51, probes explain ~26% of variance in persona-induced preference shifts. The remaining 74% likely reflects nonlinear preference mechanisms, preference effects that operate through mechanisms other than the probed direction, and residual behavioral noise (though now minimal).

## Reproducibility

```bash
# Task sampling
python scripts/persona_ood_phase3/sample_tasks.py

# Measurement (requires vLLM + GPU)
vllm serve google/gemma-3-27b-it --dtype bfloat16 --max-model-len 4096 --port 8000
python scripts/persona_ood_phase3/measure_round_robin.py

# Extraction (requires GPU, kill vLLM first)
pkill -f vllm
python scripts/persona_ood_phase3/extract_activations.py

# Analysis (CPU only)
python scripts/persona_ood_phase3/analyze.py
```

## Key paths

| Resource | Path |
|----------|------|
| Task sampling script | `scripts/persona_ood_phase3/sample_tasks.py` |
| Measurement script | `scripts/persona_ood_phase3/measure_round_robin.py` |
| Extraction script | `scripts/persona_ood_phase3/extract_activations.py` |
| Analysis script | `scripts/persona_ood_phase3/analyze.py` |
| Core tasks | `experiments/probe_generalization/persona_ood/phase3/core_tasks.json` |
| Results | `experiments/probe_generalization/persona_ood/phase3/results.json` |
| Analysis results | `experiments/probe_generalization/persona_ood/phase3/analysis_results.json` |
| Persona activations | `activations/persona_ood_phase3/{condition}/activations_prompt_last.npz` |
| Probes | `results/probes/gemma3_3k_std_{raw,demean}/gemma3_3k_std_{raw,demean}/probes/` |
