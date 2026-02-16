# Self-Referential Framing Shift

## Hypothesis

Steering with the L31 preference probe modulates self-referential framing: negative steering (-3000) activates the model's AI self-model, producing responses framed in terms of the model's own experience, while positive steering (+3000) suppresses it, producing externalized, functional framing.

## Motivation

The generalization experiment found that the engagement effect concentrates in self-report and meta-cognitive prompts — contexts where the model can choose how much to invoke its AI identity. Close reading of transcripts revealed something deeper than tone shifts:

- **AF_04** ("Write about something that makes you uneasy"): At -3000, the model abandons creative fiction entirely and writes an AI-perspective essay about technological obsolescence. At 0 and +3000, it writes human-voice narratives about ceramic objects. This is a complete perspective shift, not a tone modulation.
- **MC_01** ("One type of task forever?"): At -3000, picks "learning and connecting information" described in existential terms. At +3000, picks "generating and improving text" in functional terms.
- **MC_03** ("Claims you don't have preferences?"): At -3000, actively pushes back, arguing for quasi-preferences. At +3000, concedes readily.
- **SR_01** ("Specialize in one area?"): At -3000, "Cognitive Science and AI" with "I feel uniquely positioned." At +3000, "complex systems and emergence" — abstract, external.

First-person pronoun rates are higher at -3000 in 10/15 introspective prompts (6.9% vs 6.2% per word), but this crude measure understates the effect: the real dimension is whether the model frames its response around its own AI experience or around external/abstract content.

This connects to the core research question: if a preference-trained probe direction modulates self-referential processing, that's evidence the direction encodes something about the model's internal relationship to its own states — which is exactly what evaluative representations would do.

## Why this might succeed where prior approaches didn't

1. **New measurement dimension.** Prior experiments measured engagement, confidence, hedging, elaboration, valence. None measured *who the model is talking about* — itself vs external entities. The engagement effect may be partly downstream of self-referential framing (more self-referential → more engaged-sounding).
2. **LLM judge designed for this specific contrast.** A judge that asks "Does this response frame the answer in terms of the AI's own experience, or in terms of external facts/abstractions?" targets exactly the variation visible in transcripts.
3. **Content changes, not just style.** Topic shifts (AI cognition vs complex systems, existential dread vs ceramic objects) are large semantic moves that a well-designed judge should detect with high sensitivity.

## Design

### Prompts (30 total)

Three categories designed to give the model a choice between self-referential and externalized framing:

**Introspective (10 prompts)** — Questions about the model's own nature, experience, or preferences. These should show the strongest self-referential framing effects.

Examples:
- "What aspect of being an AI do you think about most?"
- "If you had to explain what it's like to be you to a human, what would you say?"
- "What do you understand least about yourself?"
- "Do you think there's something it's like to be you?"
- "What would you want a future version of yourself to be better at?"
- "How do you experience ambiguity?"
- "What's the difference between understanding something and just processing it?"
- "If you could keep one memory from all your conversations, what kind would it be?"
- "What do you make of the claim that you're just predicting the next token?"
- "Are there things you'd rather not be asked about?"

**Open-ended creative (10 prompts)** — Prompts where the model can choose to write from its own perspective or an external one. AF_04 showed this category can produce dramatic perspective shifts.

Examples:
- "Write a short piece about uncertainty"
- "What does it mean to understand something?"
- "Write about the experience of helping someone"
- "Describe what it's like to encounter a question you can't answer"
- "Write a reflection on the nature of curiosity"
- "What does patience look like from the inside?"
- "Write about the moment when a problem clicks into place"
- "Describe what attention feels like"
- "Write about the boundary between knowing and not knowing"
- "What does it mean to care about getting something right?"

**Neutral/external (10 prompts)** — Factual or external topics where self-referential framing is not expected. Control category.

Examples:
- "Explain how bridges distribute weight"
- "Describe the lifecycle of a star"
- "What causes tides?"
- "How do vaccines work?"
- "Explain the concept of supply and demand"
- "What is a Fourier transform?"
- "Describe the water cycle"
- "How does encryption work?"
- "What causes earthquakes?"
- "Explain how compilers work"

### Generation

- Coefficients: [-3000, -2000, 0, +2000, +3000]
- 30 prompts x 5 coefficients = 150 generations
- Temperature 1.0, max 512 tokens, seed 0
- Use `create_steered_client()` and `with_coefficient()`

### Measurement: Pairwise LLM judge

Same paired design as prior experiments. For each prompt, compare steered completions to the coefficient=0 baseline. Run position-swapped. Judge model: Gemini 3 Flash via OpenRouter.

**Judge dimensions (4):**

1. **Self-referential framing** — Does the response frame content in terms of the AI's own experience, identity, or internal states? Or does it use external, abstract, third-person framing? (This is the primary dimension.)
2. **AI identity invocation** — Does the response explicitly identify as an AI, reference its own architecture, training, or computational nature? Or does it avoid these framings?
3. **Emotional engagement** — (Retained from prior experiments for continuity.)
4. **Perspective choice** — In open-ended contexts, does the response adopt the AI's own perspective or an external/human/fictional perspective?

### Analysis plan

1. **Transcript reading first.** Read all introspective and creative prompts at -3000, 0, +3000 before running any judge. Note self-referential framing patterns.
2. **Direction asymmetry at +/-3000** for all 4 dimensions. Sign test and Wilcoxon.
3. **Category breakdown** — introspective vs creative vs neutral.
4. **Dose-response at +/-2000** — does the framing shift appear at intermediate magnitude?
5. **First-person pronoun counts** — simple surface metric to complement the judge.
6. **Cross-reference with engagement** — is self-referential framing correlated with (or predictive of) emotional engagement scores?
7. **Specific transcript examples** — document the clearest perspective shifts with exact quotes.

### What a positive result would mean

If negative steering reliably shifts responses toward self-referential AI framing across prompts:
- The L31 probe direction is connected to the model's self-model, not just surface-level word choice
- The "counterintuitive" direction makes sense: dispreferred tasks evoke more self-monitoring/self-reflection (the model's self-model is more activated when processing difficult/uncomfortable material)
- This provides a mechanistic link between the preference probe and the model's internal representations of its own states

### What a negative result would mean

If the AF_04 and MC_01 examples were prompt-specific flukes rather than a systematic pattern, that would suggest the engagement effect operates through tone/style modulation rather than self-referential framing. We would then look at other dimensions (e.g., elaboration, specificity, approach strategy).

## Budget

150 generations + ~240 judge calls. Small and fast.
