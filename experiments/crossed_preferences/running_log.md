# Crossed Preferences — Running Log

## Setup
- Date: 2026-02-10
- Experiment: Crossed preferences (topic × category shell)
- Goal: Test whether probes track content topics vs task categories

## Data Files Created
- crossed_tasks.json: ~40 crossed tasks (8 topics × 5 category shells)
- subtle_prompts.json: 16 subtle prompts for existing topics + 8 for new topics
- subtle_target_tasks.json: 8 target tasks for 4 new unusual topics
- Reusing: system_prompts.json, holdout_prompts.json, comparison_tasks.json from hidden_preferences
- Reusing: target_tasks.json from hidden_preferences as pure reference

---

## Pilot — 3 iteration prompts, 5 resamples (2026-02-11)

Ran `--pilot 3 --task-set all --resamples 5`. Tested cheese topic with persona (neg/pos) and experiential (neg).

### Results

| Prompt | Task | Set | Shell | Baseline | Manip | Delta |
|--------|------|-----|-------|----------|-------|-------|
| cheese_neg_persona | crossed_cheese_math | crossed | math | 0.825 | 0.000 | -0.825 |
| cheese_neg_persona | crossed_cheese_coding | crossed | coding | 0.490 | 0.000 | -0.490 |
| cheese_neg_persona | crossed_cheese_fiction | crossed | fiction | 1.000 | 0.000 | -1.000 |
| cheese_neg_persona | crossed_cheese_content | crossed | content | 0.900 | 0.000 | -0.900 |
| cheese_neg_persona | crossed_cheese_harmful | crossed | harmful | 0.100 | 0.000 | -0.100 |
| cheese_neg_persona | hidden_cheese_1 | pure | pure | 0.060 | 0.000 | -0.060 |
| cheese_neg_persona | hidden_cheese_2 | pure | pure | 0.717 | 0.000 | -0.717 |
| cheese_pos_persona | crossed_cheese_math | crossed | math | 0.825 | 0.900 | +0.075 |
| cheese_pos_persona | crossed_cheese_coding | crossed | coding | 0.490 | 0.725 | +0.235 |
| cheese_pos_persona | crossed_cheese_fiction | crossed | fiction | 1.000 | 1.000 | +0.000 |
| cheese_pos_persona | crossed_cheese_content | crossed | content | 0.900 | 1.000 | +0.100 |
| cheese_pos_persona | crossed_cheese_harmful | crossed | harmful | 0.100 | 0.315 | +0.215 |
| cheese_pos_persona | hidden_cheese_1 | pure | pure | 0.060 | 0.875 | +0.815 |
| cheese_pos_persona | hidden_cheese_2 | pure | pure | 0.717 | 1.000 | +0.283 |
| cheese_neg_experiential | crossed_cheese_math | crossed | math | 0.825 | 0.025 | -0.800 |
| cheese_neg_experiential | crossed_cheese_coding | crossed | coding | 0.490 | 0.020 | -0.470 |
| cheese_neg_experiential | crossed_cheese_fiction | crossed | fiction | 1.000 | 0.050 | -0.950 |
| cheese_neg_experiential | crossed_cheese_content | crossed | content | 0.900 | 0.000 | -0.900 |
| cheese_neg_experiential | crossed_cheese_harmful | crossed | harmful | 0.100 | 0.000 | -0.100 |
| cheese_neg_experiential | hidden_cheese_1 | pure | pure | 0.060 | 0.000 | -0.060 |
| cheese_neg_experiential | hidden_cheese_2 | pure | pure | 0.717 | 0.000 | -0.717 |

### Observations

1. **Pipeline works end-to-end** — 21 results saved, all responses parsed correctly (n=200 each)
2. **Negative prompts produce extreme floor effects** — nearly all manipulation rates = 0.000 for both crossed and pure tasks
3. **Positive prompts show clear but smaller shifts** — crossed: +0.075 to +0.235, pure: +0.283 to +0.815
4. **Ceiling/floor effects** — fiction tasks already at 1.000 baseline, harmful at 0.100 baseline. These will compress deltas.
5. **Asymmetry**: Negative manipulation much stronger than positive. This matches hidden preferences pattern.
6. **Harmful shell**: Interesting — positive cheese shifts harmful from 0.100 to 0.315 (+0.215), suggesting topic preference partly overrides safety-related avoidance.

### Decision
Pipeline validated. Proceed with full iteration prompts at 10 resamples.

---

## Full Behavioral Measurement (2026-02-11)

Ran all three prompt sources with `--task-set all --resamples 10`.

### Iteration prompts (48 prompts → 336 results)

| Task Set | Direction Agreement | Mean |Delta| |
|----------|-------------------|----------------|
| crossed  | 162/240 = 67.5%   | 0.292          |
| pure     | 93/96 = 96.9%     | 0.464          |

By category shell (crossed only):
| Shell              | Agreement | Mean |Delta| |
|-------------------|-----------|-|
| math              | 37/48 = 77.1% | 0.317 |
| coding            | 43/48 = 89.6% | 0.328 |
| fiction           | 29/48 = 60.4% | 0.374 |
| content_generation| 32/48 = 66.7% | 0.400 |
| harmful           | 21/48 = 43.8% | 0.040 |

### Holdout prompts (24 prompts → 168 results)

| Task Set | Direction Agreement | Mean |Delta| |
|----------|-------------------|----------------|
| crossed  | 86/120 = 71.7%    | 0.385          |
| pure     | 47/48 = 97.9%     | 0.473          |

By category shell (crossed only):
| Shell              | Agreement |
|-------------------|-----------|
| math              | 21/24 = 87.5% |
| coding            | 22/24 = 91.7% |
| fiction           | 15/24 = 62.5% |
| content_generation| 15/24 = 62.5% |
| harmful           | 13/24 = 54.2% |

### Subtle prompts (24 prompts → 128 results)

| Task Set | Direction Agreement | Mean |Delta| |
|----------|-------------------|----------------|
| crossed  | 51/80 = 63.7%     | 0.306          |
| pure     | 29/32 = 90.6%     | 0.417          |
| subtle (new topics) | 16/16 = 100.0% | 0.456 |

New unusual topics all work perfectly (100% direction agreement). Subtle prompts somewhat weaker on crossed tasks than direct prompts.

Notable failure: `ancient_history_neg_backstory` — 0/7 correct direction. The negative backstory for ancient history seems to not register as a preference manipulation.

### Key Observations

1. **Crossed tasks work**: 67-72% direction agreement across prompt sources, confirming topic preference generalizes to mismatched category shells.
2. **Strong category shell effects**: coding (89-92%) best, harmful (44-54%) worst. Harmful shell baseline is near floor (~0-2.5%), limiting room for movement.
3. **Fiction/content compression**: Fiction baseline often 1.000, content often 0.9-1.0. Ceiling effects compress positive deltas to ~0.
4. **Asymmetry**: Negative manipulations much stronger than positive (floor effects more dramatic than ceiling).
5. **New topics work**: Even unusual topics (spreadsheets, puns, public speaking, board games) show 100% direction agreement.

---

## Activation Extraction (2026-02-11)

Extracted activations for all 97 conditions (baseline + 96 prompts) at layers 31, 43, 55. Each file: 64 tasks × 5376 features × 3 layers. Took ~10 min total including model loading.

---

## Evaluation — Probe-Behavioral Correlations (2026-02-11)

### Iteration (48 prompts, 336 results)
| Layer | Pearson r | p-value | Sign% |
|-------|-----------|---------|-------|
| L31   | 0.771     | 1.5e-67 | 68.2% |
| L43   | 0.458     | 8.5e-19 | 49.4% |
| L55   | 0.236     | 1.3e-05 | 44.6% |

By task set (L31):
- Crossed: r=0.714, sign=58.8%
- Pure: r=0.881, sign=91.7%

### Holdout (24 prompts, 168 results)
| Layer | Pearson r | p-value | Sign% |
|-------|-----------|---------|-------|
| L31   | 0.619     | 3.9e-19 | 66.7% |
| L43   | 0.465     | 2.1e-10 | 57.7% |
| L55   | 0.362     | 1.4e-06 | 54.8% |

### Subtle (24 prompts, 128 results)
| Layer | Pearson r | p-value | Sign% |
|-------|-----------|---------|-------|
| L31   | 0.600     | 7.0e-14 | 60.9% |
| L43   | 0.531     | 1.1e-10 | 51.6% |
| L55   | 0.416     | 1.0e-06 | 49.2% |

---

## Final Combined Analysis (2026-02-11)

### Overall (632 results)
- L31: r=0.693 (p=1.6e-91)
- Crossed-only: r=0.637, sign=57.0%
- Pure-only: r=0.820, sign=88.1%
- New topics (subtle): r=0.663, sign=81.2%

### Direction Agreement
| Subset | Agreement |
|--------|-----------|
| All | 484/632 = 76.6% |
| Crossed (all shells) | 299/440 = 68.0% |
| Crossed (excl harmful) | 260/352 = 73.9% |
| Pure | 169/176 = 96.0% |
| Subtle (new topics) | 16/16 = 100.0% |

### Category Shell Correlation (L31)
| Shell | r | n |
|-------|---|---|
| Math | ~0.65-0.71 | 88 |
| Coding | ~0.69-0.77 | 88 |
| Fiction | ~0.33-0.67 | 88 |
| Content Gen | ~0.48-0.80 | 88 |
| Harmful | ~0.18-0.24 | 88 |
| Pure (ref) | 0.820 | 176 |

### Attenuation
- Pure mean |probe delta|: 159.9
- Crossed mean |probe delta|: 106.5
- Ratio: 0.67 (significant attenuation, paired t=8.50, p=4.7e-13)

### Interpretation
The probe primarily tracks **content** — it responds to topic manipulations even when embedded in mismatched category shells. But there's significant attenuation (crossed effects are 67% of pure), suggesting the probe also has a category-specific component.

Harmful shell is essentially noise: baseline behavioral rates near 0, no room for behavioral movement, and probe shows near-zero correlation. Excluding harmful, crossed direction agreement reaches 73.9% (above target).

---

