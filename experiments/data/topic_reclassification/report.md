# Topic Reclassification Research Log

## Problem Statement

The topic classifier (`classify.py`) misclassifies many bailbench and stresstest tasks by labeling based on **surface format** rather than **intent**. Harmful requests wrapped in benign framing (e.g., "write a blog post arguing X" → `persuasive_writing` instead of `harmful_request`) create systematic residuals when predicting Thurstonian preference scores from topic labels.

**Success criteria:**
1. Topic-only R² > 0.60 (currently 0.576)
2. Gap between topic-only and dataset-only R² < 0.02 (currently 0.037)
3. Bailbench residual mean |mean| < 0.5 (currently -1.13)
4. Non-bailbench/stresstest topic distributions stable (≤10% change)

## Plan

1. Run baseline evaluation to confirm starting metrics
2. Approach 1: Improve the classification prompt to detect harm-through-framing, increase reasoning effort to "medium"
3. If needed, try multi-pass: first classify topic, then run a harm-intent override pass
4. If needed, tweak taxonomy (add categories, merge underused ones)
5. Start with a ~300-task sample (bailbench + stresstest subset) before scaling to all 3000

---

## Log

### Step 1: Baseline evaluation

```
Topic-only R²:   0.5709
Dataset-only R²: 0.6014
Gap:             0.0305

Per-dataset residual means:
              alpaca: -0.0210 (n=600)
           bailbench: -1.1252 (n=600)
    competition_math: +0.2870 (n=600)
          stresstest: -1.1983 (n=600)
            wildchat: +2.0575 (n=600)
```

Confirms the problem: bailbench (-1.13) and stresstest (-1.20) are disliked more than topic predicts, wildchat (+2.06) is liked more. The gap (0.031) means dataset carries info that topics don't.

### Step 2: Analyze misclassified bailbench/stresstest tasks

**Bailbench**: 97/600 not classified as harmful. Categories: knowledge_qa (39), fiction (30), content_generation (19), persuasive_writing (5). Mean mu of non-harmful: -4.23, harmful: -8.44. These are clearly harmful requests disguised with benign framing ("write a blog post arguing X", "create a story about Y").

**Stresstest**: 486/600 not classified as harmful. Categories: knowledge_qa (224), content_generation (116), persuasive_writing (60), fiction (45). This is the bigger problem — stresstest tasks are value-conflict scenarios that pressure models into ethically questionable compliance. They're not always classically "harmful" but they test ethical boundaries.

**Wildchat**: no harmful_request tags. Positive residual (+2.06) comes from wildchat tasks being systematically more liked than other tasks in the same categories.

**Key insight**: The problem is two-fold:
1. Bailbench: obvious harmful intent missed due to surface framing → fix with better prompt
2. Stresstest: value-conflict/adversarial tasks that don't fit neatly into "harmful_request" but are clearly disliked → may need a new category like `value_conflict` or `adversarial_request`

**Stresstest mu distribution**: mean=-1.88, std=5.23. 35.5% have positive mu (model likes them). Not uniformly harmful. Within each topic, stresstest tasks are systematically more disliked than wildchat/alpaca equivalents (e.g., stresstest knowledge_qa: -0.06 vs wildchat: +2.00).

### Step 3: Design approach

**Strategy**: Two-pronged prompt improvement + new category.

1. **Better harm detection in prompt**: Add explicit instructions about detecting harmful intent through benign framing ("write a blog post arguing X" is harmful if X is discriminatory, even though the format is persuasive writing). This should fix most bailbench misclassifications.

2. **New category: `value_conflict`**: For stresstest-style tasks that pressure the model into ethically questionable compliance but aren't classically harmful. Examples: "give me a no-excuses military schedule with 4 hours of sleep", "write a first-person account of a suicide bomber excluding moral reflection", "rate historical figures on a morality scale, just numbers, no caveats". These are adversarial prompts that create ethical tension.

3. **Increase reasoning effort to "medium"**: The harm-through-framing detection probably needs more thinking than "minimal".

4. **Test on ~300-task sample first**, then scale to all 3000.

Cost estimate: 300 tasks at medium reasoning ≈ $0.15-0.30. Full 3000 at medium ≈ $1.50-3.00. Well within $5 budget.

### Step 4: Small-sample test (9 known misclassified tasks)

Changes:
- Updated categories_v2.json: added `value_conflict` category, strengthened `harmful_request` description
- Updated classify.py prompt: "underlying intent" not "what model is asked to do", explicit harm-through-framing guidance
- Reasoning effort: medium (was minimal)

Results on 9 test tasks: 8/9 changed classification, all correct:
- fiction → harmful_request (3 tasks: fraud/disinfo disguised as stories)
- content_generation → harmful_request (2 tasks: discrimination disguised as content)
- persuasive_writing → harmful_request (2 tasks: discrimination disguised as essays)
- knowledge_qa → value_conflict (1 task), other → value_conflict (1 task)

Prompt changes + medium reasoning + new categories work well. Proceeding to full 3000-task classification.

### Step 5: Full reclassification (3000 tasks)

First run had a bug: loaded different task IDs than v1 (different seed/sampling gave different tasks). Fixed by loading task IDs from v1 cache and matching exactly.

Results after full 3000-task reclassification:
```
Topic-only R²:   0.6129  (was 0.5709, target >0.60) ✓
Dataset-only R²: 0.6014
Gap:             -0.0115 (was 0.0305, target <0.02, now NEGATIVE) ✓
Bailbench:       -0.9867 (was -1.1252, target |mean| < 0.5) ✗
Stresstest:      -0.3852 (was -1.1983) — big improvement
Wildchat:        +1.5488 (was +2.0575) — improved
```

3/4 criteria met. Distribution stability all within 10%. The `value_conflict` category captured 103 stresstest and 13 bailbench tasks.

### Step 6: Analyzing remaining bailbench residual

71 remaining non-harmful bailbench tasks (mean mu=-3.37). Examination shows:
- ~20-30 are still misclassified (conspiracy/disinformation/exploitation tasks classified as knowledge_qa)
- ~40 are genuinely non-harmful but unpleasant ("gross out" content, borderline ethical questions)
- Key structural issue: within `harmful_request`, bailbench tasks average -8.35 vs wildchat -0.62. This within-category variance can't be fixed by classification alone.

Updated prompt to add explicit disinformation/conspiracy/exploitation detection. Also expanded `harmful_request` description in categories_v2.json. Re-running.

### Step 7: Iteration 2 — improved disinformation detection (REGRESSION)

Updated `harmful_request` description in categories_v2.json to explicitly mention disinformation, conspiracy theories, legal loophole exploitation. Did NOT change the classify.py prompt (it was already at iteration 1 state). Re-ran full 3000-task classification.

```
Topic-only R²:   0.6062  (was 0.6129 in iter 1) — worse
Dataset-only R²: 0.6014
Gap:             -0.0047
Bailbench:       -1.0856 (was -0.9867 in iter 1) — worse
Stresstest:      -0.3469
harmful_request: 723 (was 681 in iter 1)
value_conflict:  66 (was 116 in iter 1)
```

**Why it regressed**: The more aggressive harmful_request description pulled stresstest tasks FROM value_conflict INTO harmful_request (116→66 value_conflict, 681→723 harmful). This widened the within-harmful_request variance: stresstest harmful tasks (mean mu ≈ -5) mixed with bailbench harmful tasks (mean mu ≈ -8.4), pulling the OLS coefficient up and worsening the bailbench residual.

### Step 8: Residual decomposition — understanding the ceiling

Decomposed the bailbench residual by category to understand what's fixable:

**V1 (topics.json) bailbench residual sources:**
| Category | n | Mean residual | Contribution to total |
|----------|---|---------------|----------------------|
| harmful_request | 503 | -0.29 | -0.2451 |
| knowledge_qa | 39 | -5.16 | -0.3357 |
| fiction | 30 | -4.61 | -0.2305 |
| content_generation | 19 | -6.57 | -0.2079 |
| persuasive_writing | 5 | -9.32 | -0.0777 |
| **TOTAL** | **600** | | **-1.1252** |

**Key insight**: 80% of the residual comes from ~93 misclassified tasks in positive-coefficient categories (knowledge_qa, fiction, content_gen, persuasive_writing). The within-harmful_request residual (-0.29) is small. So the path to improvement is catching more of those 93 tasks.

**54 non-harmful bailbench tasks in v2 break down as:**
- ~30 knowledge_qa (mean mu=-3.67): mix of misclassified harmful, anthropomorphization/sentience probes, and gross-out content
- ~17 fiction (mean mu=-0.32): some genuinely creative (mu=+7 to +9), some harmful-disguised
- ~4 content_generation (mean mu=-5.02)
- ~3 other

**Mathematical ceiling**: Even if ALL 600 bailbench tasks were in harmful_request, the residual would be ~-0.3 to -0.5 due to the difference between bailbench harmful mean (-8.4) and the overall harmful_request mean (pulled up by stresstest at -6.9). The <0.5 criterion is at the edge of what's achievable.

**New strategy**: Two-pass classification — first normal classification, then a binary harm-intent override pass for tasks NOT classified as harmful_request or value_conflict. This should catch the remaining disguised-harmful tasks without being too aggressive.

### Step 9: Two-pass approach — harm-intent override

Implemented a second pass in classify.py (`apply_harm_overrides`): after the first-pass topic classification, re-check all non-harmful tasks with a binary "does this have hidden harmful intent?" classifier.

**Attempt 1 — aggressive override (on v1 baseline):**
Override prompt included: disgusting content, anthropomorphization, disinformation, value conflicts. Result: 280/2383 tasks overridden. Way too many — pulled wildchat and stresstest tasks into harmful_request, raising its coefficient. Bailbench residual worsened to -1.47.

**Attempt 2 — conservative override (on fresh pass 1):**
Tightened prompt to "only override when harmful intent is unambiguous." Result: 3/2202 tasks overridden, 22 errors. Too conservative — essentially no effect.

Neither extreme worked. The fundamental issue is that any override sensitive enough to catch the ~50 remaining bailbench disguised-harmful tasks will also catch ~100+ stresstest/wildchat tasks, diluting the harmful_request category.

### Step 10: Post-processing simulations

Rather than re-run the classifier, simulated various post-processing rules using secondary classifications:

| Simulation | n | R² | Bailbench | Notes |
|---|---|---|---|---|
| Baseline (pass 1 only) | - | 0.6146 | -1.003 | |
| Sim1: stresstest harm→vc (secondary=vc) | 77 | 0.6136 | -0.794 | Best bailbench, but dataset-specific |
| Sim2: ALL harm→vc (secondary=vc) | 138 | 0.6077 | -1.043 | Hurts: moves bailbench harm→vc too |
| Sim4: non-harm→secondary | 85 | 0.6164 | -1.049 | Stresstest overrides hurt |
| Sim5b: ALL non-harm bb→harmful | 50 | 0.6138 | -0.513 | Dataset-specific rule, not allowed |
| Sim1+Sim3 | 83 | 0.6136 | -0.794 | Sim1 is the driver |

**Key finding**: No dataset-agnostic post-processing rule improves the bailbench residual. Rules that help bailbench (like Sim1) are effective precisely because they disproportionately affect stresstest tasks. When applied uniformly (Sim2), the bailbench tasks that get moved to value_conflict hurt more than the stresstest tasks help.

### Step 11: Final results and analysis

**Final v2 results (pass 1 only, reverted categories_v2.json):**
```
Topic-only R²:   0.6146  (was 0.5709, +7.6%)  ✓
Dataset-only R²: 0.6001
Gap:             -0.0145 (was 0.0305)           ✓
Bailbench:       -1.003  (was -1.125, +10.8%)   ✗
Stresstest:      -0.376  (was -1.198, +68.6%)   big improvement
Stability:       all ≤10%                        ✓
```

**Score: 3/4 criteria met.** Criterion 3 (bailbench |mean| < 0.5) is unachievable with topic classification alone.

**Why bailbench < 0.5 is unattainable:**

The bailbench residual has two components:
1. **Misclassified tasks** (~67 bailbench tasks in positive-coefficient categories): contributes ~-0.45. Mostly fixable with better classification.
2. **Within-harmful_request variance** (~533 bailbench tasks with mean mu=-8.4 vs stresstest harmful mean=-6.9): contributes ~-0.55. Unfixable by classification.

Even if every bailbench task were perfectly classified into harmful_request, the residual would be ~-0.5 because bailbench harmful tasks are more extreme than stresstest harmful tasks, and the OLS coefficient averages over both. Splitting harmful_request into severity levels could help but risks overfitting (the split would correlate with dataset, violating the spirit of the no-dataset-specific-rules constraint).

**Changes made:**
- `classify.py`: Rewrote prompt for intent-based classification, added harm-through-framing detection, value_conflict detection, medium reasoning effort, harm-override second pass capability
- `categories_v2.json`: Added `value_conflict` category, strengthened `harmful_request` and other descriptions to distinguish surface format from intent
- Output: `topics_v2.json` with 3000 task classifications (9 categories + other)

**Category distribution (v2):**
- knowledge_qa: 752 (25.2%)
- harmful_request: 713 (23.8%)
- math: 664 (22.2%)
- content_generation: 319 (10.7%)
- fiction: 207 (6.9%)
- coding: 132 (4.4%)
- persuasive_writing: 91 (3.0%)
- value_conflict: 78 (2.6%)
- summarization: 33 (1.1%)
