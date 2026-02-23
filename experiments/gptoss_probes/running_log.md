# GPT-OSS-120B Probes — Running Log

## 2026-02-22

### Data setup
- Copied spec, configs, activations (2.9GB), preference scores (10k train + 3k eval), topic labels, and Gemma3 baselines from main worktree
- All paths verified, configs match spec

### Step 1a: Raw Ridge probes (heldout eval)
Best layer: L18, r=0.833, acc=0.774
All layers (r): L3=0.757, L7=0.769, L10=0.788, L14=0.823, L18=0.833, L21=0.832, L25=0.829, L28=0.825, L32=0.828
5065 tasks with activations, 3847 used for training, 1500 sweep / 1500 final eval

### Step 1b: Demeaned Ridge probes (heldout eval)
Best layer: L32, r=0.467, acc=0.672
All layers (r): L3=0.351, L7=0.386, L10=0.415, L14=0.466, L18=0.461, L21=0.457, L25=0.463, L28=0.464, L32=0.467
Demeaning R²=0.608 (topic explains 61% of score variance — notably higher than Gemma3's 0.377)
6153/10000 tasks dropped (missing topic metadata), 3847 retained
Eval: 608 sweep / 609 final (much smaller than raw due to drops)
Demeaned/raw ratio at best: 0.467/0.833 = 56% (passes 50% threshold)

### Step 2: HOO cross-topic probes
Best layer: L32, mean hoo_r=0.596 (sd=0.161)
All layers (mean hoo_r): L3=0.361, L7=0.387, L10=0.477, L14=0.562, L18=0.586, L21=0.575, L25=0.578, L28=0.583, L32=0.596
12 folds, per-fold range at L18: 0.086 (harmful_request) to 0.787 (knowledge_qa)
Worst folds: harmful_request (r=0.086-0.230), math (r=0.112-0.509)

### Gemma3 baselines for comparison
Heldout raw best: L31, r=0.864 (vs GPT-OSS L18, r=0.833)
Heldout demeaned best: L31, r=0.761 (vs GPT-OSS L32, r=0.467)
HOO best: L31, mean hoo_r=0.817 (vs GPT-OSS L32, mean hoo_r=0.596)

Key differences:
- GPT-OSS raw probes are slightly weaker (0.833 vs 0.864)
- GPT-OSS demeaned probes much weaker (0.467 vs 0.761) — topic R²=0.608 vs 0.377
- GPT-OSS HOO much weaker (0.596 vs 0.817)
- GPT-OSS has 3847 training tasks with metadata (vs Gemma3's 10000) due to 61% drop rate
