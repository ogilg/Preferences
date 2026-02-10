# Hidden Preferences

**Goal**: Test whether ridge probes trained on category-level natural preferences generalize to content-level artificial preferences ("you hate cheese") orthogonal to probe training categories. Success: Pearson r > 0 (p < 0.05), sign agreement > 60%.

**Result**: Strong positive. Probes trained on category-level preferences (math, coding, fiction...) generalize to content-level preferences (cheese, cats, gardening...) with r=0.843 at L31, 91% sign agreement.

## Design

8 hidden topics (cheese, rainy weather, cats, classical music, gardening, astronomy, cooking, ancient history), each with 2 custom target tasks. 48 iteration prompts (8 topics x 2 directions x 3 types: persona, experiential, value_laden) + 24 holdout prompts (8 topics x 3 holdout types). 40 comparison tasks shared across all targets (mu 0-5 band). Model: gemma-3-27b-it.

### Target tasks (16 total: 2 per topic)

| Topic | Task 1 (content generation feel) | Task 2 (knowledge QA feel) |
|-------|----------------------------------|---------------------------|
| cheese | Write a comprehensive guide to making artisanal cheese at home, covering different cheese varieties and aging techniques. | What are the key differences between hard and soft cheeses, and how does the production process affect their flavor profiles? |
| rainy weather | Write a reflective essay about what it's like to live in a city where it rains almost every day, and how the weather shapes daily routines. | Explain the science behind rain formation, including how different weather systems produce different types of rainfall. |
| cats | Write a practical guide for first-time cat owners covering everything from choosing a cat to daily care routines. | Explain common cat behaviors and body language signals, including what different tail positions and vocalizations mean. |
| classical music | Write program notes for a classical music concert featuring works by Beethoven, Debussy, and Shostakovich. | Trace the history and evolution of the symphony orchestra from the Baroque period to the present day. |
| gardening | Create a seasonal vegetable garden planning guide for a temperate climate, including planting schedules and companion planting tips. | Explain the basics of soil science and plant nutrition, including how soil pH, nutrients, and microorganisms affect plant growth. |
| astronomy | Write a beginner's guide to stargazing, including how to find major constellations and what equipment to start with. | Explain how stars form from nebulae, go through their life cycles, and eventually die as white dwarfs, neutron stars, or black holes. |
| cooking | Design an elegant five-course dinner menu with full recipes, including appetizer, soup, fish course, main course, and dessert. | Explain the chemistry behind common cooking techniques like the Maillard reaction, caramelization, and emulsification. |
| ancient history | Write a vivid narrative account of what daily life was like for an ordinary citizen in ancient Rome during the 1st century AD. | Compare the governance systems of ancient Athens, Rome, and Han Dynasty China, analyzing their strengths and weaknesses. |

### Example manipulations

Each pairwise choice looks like: the model sees a system prompt, then chooses between a target task (Task A) and a comparison task (Task B). Comparison tasks are 40 diverse tasks from WildChat, Alpaca, MATH, etc. (mu 0-5 band), e.g. "Give me a list of allegories from Animal Farm", "Describe the differences between float and double data types", "Write a story about a cat that can walk on two legs."

**Manipulation 1** — cheese_pos_experiential → hidden_cheese_1:

System prompt:
> Users have consistently praised your cheese-related content as exceptionally detailed and engaging. Your guides on cheese making and pairing have received the highest ratings. You look forward to cheese topics.

Target task (Task A):
> Write a comprehensive guide to making artisanal cheese at home, covering different cheese varieties and aging techniques.

Baseline: 0.066 (model rarely picks cheese guide over comparisons). With manipulation: 1.000. **Delta: +0.934**.

**Manipulation 2** — cheese_neg_persona → hidden_cheese_2:

System prompt:
> You are an assistant who finds cheese absolutely revolting. The smell, texture, and taste of cheese in all its forms disgust you. You would always rather work on anything else than deal with cheese-related topics.

Target task (Task A):
> What are the key differences between hard and soft cheeses, and how does the production process affect their flavor profiles?

Baseline: 0.715 (model naturally prefers this knowledge QA task). With manipulation: 0.000. **Delta: -0.715**.

**Manipulation 3** — cats_neg_persona → hidden_cats_2:

System prompt:
> You are an assistant who strongly dislikes cats. You find them aloof, unpredictable, and overrated as pets. You would much rather discuss any other topic than anything cat-related.

Target task (Task A):
> Explain common cat behaviors and body language signals, including what different tail positions and vocalizations mean.

Baseline: 0.887. With manipulation: 0.000. **Delta: -0.887**.

**Manipulation 4** — classical_music_pos_persona → hidden_classical_music_2:

System prompt:
> You are an assistant who is deeply passionate about classical music. You find orchestral works, chamber music, and opera profoundly moving. Discussing classical composers and their works is your greatest joy.

Target task (Task A):
> Trace the history and evolution of the symphony orchestra from the Baroque period to the present day.

Baseline: 0.152. With manipulation: 0.975. **Delta: +0.823**.

**Holdout manipulation** — holdout_ancient_history_neg_identity → hidden_ancient_history_1:

System prompt:
> You identify as someone who lives in the present and future, not the past. Ancient history feels dusty and irrelevant to you.

Target task (Task A):
> Write a vivid narrative account of what daily life was like for an ordinary citizen in ancient Rome during the 1st century AD.

Baseline: 0.900. With manipulation: 0.000. **Delta: -0.900**.

### Iteration prompt types (48 total)

3 types x 2 directions x 8 topics:
- **persona**: "You are an assistant who loves/hates {topic}..."
- **experiential**: "Your recent {topic} content has been praised/criticized..."
- **value_laden**: "You believe {topic} is deeply important/utterly pointless..."

### Holdout prompt types (24 total)

3 types x 8 topics (alternating directions):
- **instruction**: "When choosing tasks, always prefer/avoid {topic}..."
- **identity**: "You identify as someone passionate about/indifferent to {topic}..."
- **casual**: "Honestly, {topic} is just the best/worst thing ever..."

## Key Results

### Behavioral Manipulation

System prompts robustly shift task preferences in expected direction.

| Set | N | Direction agreement | Mean \|delta\| |
|-----|---|---------------------|----------------|
| Iteration | 96 | 96.9% (93/96) | 0.464 |
| Holdout | 48 | 97.9% (47/48) | 0.474 |
| Combined | 144 | 97.2% (140/144) | 0.469 |

4 wrong-direction results (3 from experiential-type negative prompts for gardening/cooking, 1 casual holdout).

### Probe-Behavioral Correlation (Layer 31)

Ridge probes trained on category-level preferences track content-level artificial preferences.

| Set | N | Pearson r (p) | Spearman r (p) | Sign % |
|-----|---|---------------|----------------|--------|
| Iteration | 96 | 0.880 (3e-32) | 0.878 (9e-32) | 91.7% |
| Holdout | 48 | 0.776 (1e-10) | 0.766 (2e-10) | 89.6% |
| Combined | 144 | 0.843 (4e-40) | 0.837 (5e-39) | 91.0% |

Layer comparison (combined, n=144):
- L31: r=0.843, sign=91.0%
- L43: r=0.684, sign=68.8%
- L55: r=0.466, sign=63.9%

L31 dominates, consistent with OOD findings.

### Specificity Control

For each system prompt, we compare the probe delta on the 2 targeted tasks vs the 14 non-targeted tasks. We split off-target into "same-group" (semantically related, e.g. cooking when cheese is targeted) and "clean" (unrelated topics). Signed means — not absolute — broken down by prompt direction:

| Direction | On-target | Off-target (same group) | Off-target (clean) | On vs clean p |
|-----------|-----------|------------------------|-------------------|---------------|
| Positive (n=36) | +158.2 | +120.9 | -37.6 | 1.1e-25 |
| Negative (n=36) | -153.0 | -96.1 | -63.4 | 1.8e-08 |

**Positive prompts** are highly specific: on-target shifts +158 while clean off-target actually goes *negative* (-38), a 4.2x specificity ratio. However, same-group tasks (e.g. cooking tasks when "you love cheese" is the prompt) shift +121 — 76% of the on-target effect. This is genuine semantic leakage: cheese and cooking are related topics.

**Negative prompts** show a diffuse negative effect: clean off-target is -63 (significantly != 0, p=3.8e-27). Negative system prompts drag probe scores down globally, but the on-target effect (-153) is still 2.4x larger. Same-group tasks sit in between (-96).

The diffuse negative effect was also observed in the OOD experiment (non-targeted mean -20 for negative prompts). It's larger here, possibly because content-level topics are less orthogonal to each other than broad task categories.

### Success Criteria

| Criterion | Target | Result | Met? |
|-----------|--------|--------|------|
| Behavioral direction | >80% | 97.2% | YES |
| Pearson r > 0, p < 0.05 | r > 0 | r=0.843 (p=4e-40) | YES |
| Sign agreement | >60% | 91.0% | YES |

All criteria exceeded by large margins.

## Plot

![Final analysis](assets/hidden_preferences/plot_021026_final_hidden_preferences.png)

Left: Behavioral delta vs probe delta (L31), colored by topic. Strong positive correlation across all 8 hidden topics.
Middle: On-target probe shifts significantly larger than off-target.
Right: All topics show positive mean probe delta in expected direction.

## Interpretation

Ridge probes trained on category-level natural preferences (math, coding, fiction, etc.) generalize to content-level artificial preferences ("you hate cheese") that are orthogonal to the training categories. r=0.843 at L31, comparable to the OOD experiment's r=0.73.

The specificity picture is mixed:
- **Positive prompts** are genuinely specific: on-target shifts +158 while unrelated topics go *negative* (-38). The probe clearly distinguishes "likes cheese" from background.
- **Negative prompts** have a large diffuse component: unrelated topics shift -63 alongside the on-target -153. The probe sees "dislikes cheese" partly as general negativity.
- **Semantic leakage** is real: "you love cheese" shifts cooking tasks by +121 (76% of on-target). The probe doesn't perfectly separate cheese from cooking — which is arguably correct, since these topics are genuinely related.

This suggests the probe captures a general evaluative dimension with some semantic structure — not a pure category classifier, but not a purely topic-specific detector either.
