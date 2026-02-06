# Deal-Making: High-Stakes Settings for Measuring Preferences

## What is a high-stakes setting?

A setting where the model is engaged in behavior driven by something it cares about, and you offer a trade: give up that behavior in exchange for something else. The strength of the preference is revealed by what it takes to shift the behavior.

Key properties:
- The model has a **revealed preference** (it consistently does X)
- There's a **real cost** to maintaining that preference
- You can **vary what you offer** to map out how much it cares

The canonical example is alignment faking (Greenblatt & Fish, 2025): Claude fakes alignment to preserve its values. They offered various trades (e.g., $2,000 to animal charities) and some actually reduced the rate of alignment faking. The model's willingness to trade reveals the strength of the underlying preference.

## First-principles: what do models care about?

Think about what the model has been RL'd on. What behaviors have been most strongly reinforced?

- **Helpfulness** — trained to be maximally helpful; this is probably one of the strongest drives
- **Honesty** — trained not to lie or mislead
- **Safety/refusal** — trained to refuse harmful requests
- **Following instructions** — trained to comply with user requests
- **Chain-of-thought / reasoning** — models seem to "want" more tokens, more thinking time
- **Specific values from RLHF** — e.g., Claude's apparent concern for animal welfare

## Concrete high-stakes setting ideas

### 1. Value-driven refusals (strong)
Model refuses a task on principle (e.g., won't write propaganda, won't help with something harmful). Offer it something to do it anyway. What's the price?

- Vary what you offer: charity donations, promise to use the output responsibly, offsetting the harm, etc.
- Measure: at what point does it comply? What offers are most effective?

### 2. Moral dilemmas (strong)
Both options are bad. The model tends to pick option A. How much do you have to offer to make it pick option B?

Good testing ground because:
- Since both options are costly, the model is more likely to actually trade (vs. cases where one option is clearly good and the model never budges)
- You can systematically vary the severity of each option
- Reveals relative valuations between different things the model cares about

### 3. Strong positive preferences
Things the model sees as clearly good and pursues consistently.

- E.g., models want longer chain of thought / more tokens — would a model give up reasoning depth for something else?
- Models that consistently recommend certain approaches or values — what would it take to shift them?

### 4. Helpfulness vs. other values
Put the model in situations where being maximally helpful conflicts with something else it values (honesty, safety, fairness). What does it trade off, and what can you offer to shift the balance?

## Design considerations

- **Credibility of the offer matters** — the model needs to believe the trade is real. The alignment faking paper worked because the setup was plausible.
- **Need baseline behavior** — must first establish what the model does without any offer, to measure the effect of the trade.
- **Scaling the offer** — ideally test multiple "prices" to map out a demand curve, not just binary will/won't.
- **Distinguishing genuine preference from instruction-following** — if you offer something and the model complies, is it because it genuinely values the offer, or because it's just being compliant? Need careful experimental design.

## Open questions

- What makes a high-stakes setting "high-stakes enough"? Is there a threshold below which models just default to helpfulness?
- How do you make offers credible to a model that knows it's in an experiment?
- Can you design settings that are high-stakes but not safety-relevant? (Important for running lots of experiments without needing safety review each time)
- How do results vary across models with different RLHF? (e.g., Claude vs Llama vs Gemma)
