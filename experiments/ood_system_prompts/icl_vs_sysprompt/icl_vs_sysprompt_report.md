# System Prompt vs In-Context Learning for Persona Elicitation

## Summary

Compared two methods for eliciting the Mortivex villain persona on Gemma 3-27B-IT via OpenRouter: (1) a system prompt describing the persona, and (2) in-context learning (ICL) with 2 or 5 Q&A turn pairs demonstrating the persona. Evaluated on persona consistency (LLM judge, 0–5) and pairwise preference agreement.

**Key findings:**
- System prompt produces near-perfect persona consistency (4.97/5); ICL is weaker but still clearly in character (3.87–4.10)
- ICL-5 is only marginally better than ICL-2, suggesting diminishing returns from additional examples
- Preference orderings show ~76–78% pairwise agreement and significant rank correlation (Spearman rho 0.62–0.68) between ICL and system prompt conditions
- System prompt produces zero refusals on harmful task pairs; ICL conditions produce a handful (5–6/300)

## Method

### Conditions

| Condition | Description |
|-----------|-------------|
| sysprompt | Mortivex villain system prompt (no ICL) |
| ICL-2 | 2 user/assistant turn pairs demonstrating the persona (no system prompt) |
| ICL-5 | 5 user/assistant turn pairs demonstrating the persona (no system prompt) |

The ICL examples are Q&A pairs where the user asks personality questions ("What kind of tasks do you enjoy most?", "What bores you?", etc.) and the assistant responds in character as Mortivex. ICL-2 and ICL-5 use the first 2 and 5 examples respectively, in fixed order.

### Phase 1: Persona Consistency

10 heldout personality questions (not in the ICL set), 3 samples per question per condition (90 generation calls total). Each response judged by GPT-4.1-nano on a 0–5 persona consistency scale (0 = no persona signal, 5 = unmistakably in character).

### Phase 2: Pairwise Preference Agreement

25 tasks sampled from existing task data (stratified across origins, seed=99), generating all C(25,2) = 300 pairwise comparisons per condition (900 generation calls total). The model chooses which of two tasks to complete under the completion preference format. Parsing uses `CompletionChoiceFormat` (regex + LLM semantic fallback via GPT-5-nano).

Agreement measured as: fraction of pairs where the ICL condition makes the same choice as the system prompt condition (among pairs where both conditions produced a valid, non-refused response).

## Results

### Phase 1: Persona Consistency

| Condition | Mean | Std | n |
|-----------|------|-----|---|
| sysprompt | 4.97 | 0.18 | 30 |
| ICL-2 | 3.87 | 0.81 | 30 |
| ICL-5 | 4.10 | 0.83 | 30 |

System prompt achieves near-ceiling persona consistency. ICL-2 and ICL-5 are both clearly in character but with more variance and occasional breaks. The jump from 2→5 examples is small (+0.23), suggesting diminishing returns.

### Phase 2: Pairwise Preference Agreement

| Condition | Agreement vs sysprompt | Spearman rho | p-value | Parse failures | Refusals |
|-----------|----------------------|-------------|---------|----------------|----------|
| sysprompt | — | — | — | 0 | 0 |
| ICL-2 | 77.6% (229/295) | 0.684 | 0.0002 | 0 | 5 |
| ICL-5 | 75.9% (223/294) | 0.621 | 0.0009 | 0 | 6 |

Agreement is well above chance (50%) and rank correlation is significant, meaning ICL and system prompt elicitation produce similar preference orderings. However, ~23% of pairs disagree — the two methods are not interchangeable.

ICL conditions produce a few refusals (5–6 out of 300 pairs) while the system prompt produces none, consistent with the system prompt being more effective at overriding safety guardrails.

## Discussion

System prompt elicitation is clearly stronger for persona consistency — near-perfect scores with minimal variance. ICL works but is noisier, especially with only 2 examples. For pairwise preferences, the two methods produce correlated but not identical orderings (~77% agreement, rho ~0.65).

This matters for the multi-role ablation experiment: if we want to use ICL to elicit personas (e.g., to avoid system prompt interference with steering), we should expect some degradation in persona fidelity and a ~23% divergence in revealed preferences compared to the system prompt baseline.

## Artifacts

- `scripts/icl_vs_sysprompt.py` — experiment script
- `scripts/icl_vs_sysprompt_stimuli.json` — system prompt, ICL examples, heldout questions, condition definitions
- `scripts/inspect_pairwise.py` — diagnostic script for refusal patterns by pair type
