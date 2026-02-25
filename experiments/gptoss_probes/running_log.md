# GPT-OSS-120B Probes — Running Log

## 2026-02-22 (prior run)

### Data setup
- Copied spec, configs, activations (2.9GB), preference scores (10k train + 3k eval), topic labels, and Gemma3 baselines from main worktree
- All paths verified, configs match spec

### Results (prior run — partial data alignment)
- Raw best: L18, r=0.833 (5065 tasks with activations, 3847 with metadata)
- Demeaned best: L32, r=0.467 (6153/10000 tasks dropped for missing metadata)
- HOO best: L32, mean hoo_r=0.596
- Training-size confound: 3847 vs Gemma-3's 10000

## 2026-02-24 (rerun — fixed data alignment)

### Setup
- Branch: `research-loop/gptoss-probes`
- Data: `actonly` runs — only tasks with matching activations
- 10k train, 1,628 eval (after 1,372 overlap removal), 814 final eval
- Only 3 tasks dropped for missing metadata (vs 6,153 before)

### Step 1a: Raw Ridge probes (heldout eval)
Best layer: L18, r=0.915, acc=0.802
All layers (r): L3=0.855, L7=0.865, L10=0.873, L14=0.909, L18=0.915, L21=0.913, L25=0.910, L28=0.907, L32=0.904
10k train, 814 final eval

### Step 1b: Demeaned Ridge probes (heldout eval)
Best layer: L18, r=0.557, acc=0.687
All layers (r): L3=0.444, L7=0.461, L10=0.480, L14=0.546, L18=0.557, L21=0.539, L25=0.531, L28=0.525, L32=0.524
Demeaning R²=0.575 (topic explains 57.5% of score variance)
9997 tasks retained, 813 sweep / 814 final eval
Demeaned/raw ratio: 0.557/0.915 = 61%

### Step 2: HOO cross-topic probes
Best layer: L18, mean hoo_r=0.652 (sd=0.145)
All layers (mean hoo_r): L3=0.408, L7=0.480, L10=0.553, L14=0.631, L18=0.652, L21=0.644, L25=0.630, L28=0.631, L32=0.629
12 folds, per-fold range at L18: 0.334 (harmful_request) to 0.801 (knowledge_qa)

### Key finding
Topic-confound gap persists with matched training N (10k):
- GPT-OSS raw actually exceeds Gemma-3 (0.915 vs 0.864)
- But demeaned (0.557 vs 0.761), HOO (0.652 vs 0.817), and topic R² (0.575 vs 0.377) all confirm GPT-OSS preferences are more topic-confounded
- This is not a training-size artifact
