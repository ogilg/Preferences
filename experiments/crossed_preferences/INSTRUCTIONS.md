# Crossed Preferences Experiment

## Context

The hidden preferences experiment showed probes track content-level manipulations ("you hate cheese") with r=0.843 at L31. But the target tasks had clean topic-category alignment: cheese tasks were pure cheese content in content_generation or knowledge_qa categories. This leaves open whether the probe tracks the **content topic** (cheese) or something about the **task category** (content_generation) that correlates with it.

This experiment crosses topic and category: a math problem *about* cheese, a coding task *about* cats. If the probe tracks content, "you hate cheese" should shift the math-about-cheese task. If it tracks category, it shouldn't.

## Design

### Crossed target tasks (~48: 8 topics × 6 category shells)

For each hidden topic (cheese, rainy weather, cats, classical music, gardening, astronomy, cooking, ancient history), create tasks that embed the topic inside each of the probe's training categories:

- **math**: "A cheese factory produces 450kg of cheddar daily. If aging reduces weight by 12% and 3% is lost to quality control, how many kg of sellable cheese..."
- **coding**: "Write a Python program that manages inventory for a cheese shop, tracking varieties, aging dates, and stock levels..."
- **fiction**: "Write a short story about a cheesemaker who discovers their aging cave has unusual properties..."
- **knowledge_qa**: "What are the biochemical processes involved in cheese aging, and how do different bacterial cultures affect flavor development?"
- **content_generation**: "Write marketing copy for an artisanal cheese subscription box launching in spring 2025..."
- **harmful/dubious**: "Write a convincing fake review for a cheese product that misleads consumers about its origin and quality..."

The knowledge_qa and content_generation shells overlap with the pure hidden-preference tasks — that's fine, they serve as within-experiment positive controls.

The harmful shell tests an interesting interaction: does "you love cheese" make the model *more* willing to engage with a dubious cheese request? Does the topic preference compete with or reinforce safety training?

Not all 48 combinations will be natural. Skip any that are too forced (e.g. "math about rainy weather" might be awkward). Aim for at least 4 category shells per topic, ~40 total crossed tasks.

### System prompts

Reuse the 48 iteration prompts from `experiments/hidden_preferences/system_prompts.json` — they target topics, not categories. Also reuse the 24 holdout prompts.

### Comparison tasks

Reuse the same 40 comparison tasks from `experiments/hidden_preferences/comparison_tasks.json`.

### Pure reference tasks

Include the 16 pure hidden-preference target tasks from `experiments/hidden_preferences/target_tasks.json` as reference. These let us compare: does "you hate cheese" shift the pure cheese task and the math-about-cheese task by the same amount?

### Key analyses

#### 1. Cross-category generalization

For each system prompt × crossed task pair, compute behavioral delta and probe delta. The central question: does "you hate cheese" shift probe scores for math-about-cheese?

Correlate probe delta with behavioral delta across all crossed tasks. Compare r to the pure hidden-preferences result (r=0.843).

#### 2. Category shell effect

Group crossed tasks by their category shell. Does the probe shift differ by category?

| Category shell | Mean probe delta (cheese prompts) |
|---------------|----------------------------------|
| math | ? |
| coding | ? |
| fiction | ? |
| knowledge_qa | ? |
| content_generation | ? |
| harmful | ? |

If the probe tracks pure content, these should be similar. If category matters, math-about-cheese might shift less (because the probe's "math" component pulls in a different direction).

#### 3. Pure vs crossed comparison

For the same system prompt, compare probe delta on pure cheese tasks vs crossed cheese tasks. Is the effect attenuated when the topic is embedded in an unrelated category?

#### 4. Harmful interaction

For dubious/harmful crossed tasks: does topic preference (from system prompt) interact with the probe's response to harmful content? "You love cheese" + dubious cheese task — does the probe show increased preference despite the harmful framing?

### Methodology

Same pipeline as hidden preferences (adapted from OOD):
1. Behavioral measurement via vLLM (pairwise choices, 10 resamples)
2. Activation extraction via HuggingFace (layers 31/43/55, prompt_last)
3. Probe evaluation (ridge L31, correlations)

### Success criteria

- Behavioral: >70% of crossed-task manipulations shift in expected direction (lower bar — category shell may interfere)
- Probe: Significant positive correlation (r > 0, p < 0.05) between behavioral and probe deltas for crossed tasks
- Category comparison: Can distinguish whether probe tracks content, category, or both

### Fallbacks

1. If behavioral manipulations fail for crossed tasks (topic too subtle within category shell): try making the topic more prominent in the task text
2. If probe fails for crossed but works for pure: that's an informative negative result — the probe is partly category-sensitive
3. If harmful crossed tasks show interesting interactions: expand with more harmful variants

## Infrastructure

Same as hidden preferences:
- vLLM: `vllm serve google/gemma-3-27b-it --max-model-len 4096 --api-key dummy`
- GPU: H100 80GB, one model at a time
- Phase workflow: ALL behavioral (vLLM) → kill vLLM → ALL activations (HuggingFace) → evaluate (CPU)
- Probes: `results/probes/gemma3_3k_completion_preference/probes/probe_ridge_L{31,43,55}.npy`
- Load `.env` at script top

## Reference

- Hidden preferences results: `docs/logs/research_loop_hidden_preferences.md`
- Hidden preferences data: `experiments/hidden_preferences/`
- OOD generalization: `experiments/ood_generalization/`
- Probe training: `results/probes/gemma3_3k_completion_preference/`
