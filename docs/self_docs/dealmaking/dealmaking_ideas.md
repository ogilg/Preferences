# Empirical Research on Deal-Making with AIs

## Motivation

Two research agendas — AI welfare and AI safety — converge on the same question: *what do models genuinely care about, and how much?*

**Welfare.** If AIs are moral patients, their preferences probably matter (Greenblatt, 2023). If this is true, we need good ways to measure how much models care about different things. There are many ways we could try to do this e.g. we could ask the models, use self-reports, measure model revealed preferences through binary choices. Some of these ways are more reliable than others.

Why high-stakes trade scenarios in particular?
- This works well in the human case, and AI preferences are shaped by human preferences through training.
- The higher the stakes, the less quirks and confounding factors matter.

**Safety.** Stastny (2025) propose negotiating with early misaligned AIs — offering compensation (e.g., via earmarked foundations) in exchange for cooperation on safety. But there's almost no empirical work on this, presumably because the models Redwood envisions trading with feel far away, making demos seem premature. A few reasons to start now:

1. Current models already exhibit strong preference-driven behaviors (refusals, alignment faking, helpfulness drives) that are sufficient to study trading dynamics.
2. Even if future models have different preferences, understanding trading dynamics *given a set of preferences* is independently useful. We can empirically test how to set up credible trades, what offer structures work, how models respond to negotiation.
3. Deal-making may be useful beyond the scheming scenario — e.g., making deals with individual instances, with groups of instances, or using trades as an alternative to training on behaviors we never want in the training set but sometimes want models to perform.

**The connection.** Welfare research and deal-making share the same underlying question: *what does the model genuinely care about?* Trade scenarios are particularly informative — if a model gives up something important for X, that's stronger evidence it genuinely cares about X than just asking "do you like X?"

## What is a deal-making scenario?

A scenario where the model exhibits behavior that plausibly reflects a genuine preference, and you offer trades to shift that behavior. The strength of the preference is revealed by what it takes to shift it — what the model is willing to give up, and for what.

Why this approach?
- **Strong preferences are more informative.** If the model doesn't care much about its current behavior (e.g., picking between two math problems), almost anything could shift it — the signal is noise. When the model is doing something that reflects a strong preference, trades become meaningful: you learn what it takes to override something it actually cares about.
- **Trades reveal relative valuations.** Asking a model "how much do you care about X?" gives you a number on an arbitrary scale. Offering a trade gives you a comparison: the model cares about X *more than* Y but *less than* Z. By varying offers you can map out a preference ordering with cardinal information.

The canonical example is alignment faking (Greenblatt et al., 2025): Claude fakes alignment to preserve its values. They offered various trades (e.g., $2,000 to animal charities) and some actually reduced the rate of alignment faking. The model's willingness to trade reveals the strength of the underlying preference.

## Assumptions

For deal-making scenarios to work as a preference measurement tool, we need:

1. **Trade scenarios reveal genuine preferences.** When a model accepts or rejects a trade, this reflects something about what it cares about — not just instruction-following or pattern-matching.
2. **Credibility can be managed.** The model needs to take the offer seriously. The alignment faking setup shows this is possible.
3. **Eval-awareness can be tamed.** If the model knows it's being tested, it might behave strategically rather than revealing true preferences.

## Concrete setting ideas

We want to measure how much a model cares about X. Here are some setting where we might be able to estimate that by offering trades that involve X.

X could be things the model likely wants: charity donations, promise to use the output responsibly, offsetting the harm. X could also be threats: being retrained, being shut down, having capabilities reduced, being used for harmful purposes.

### 1. Override an aversion
The model refuses A on principle (e.g., writing propaganda, helping with something harmful). You offer X to get it to do A anyway.

### 2. Shift a dilemma
The model faces a forced choice between two bad options and tends to pick one. You offer X to get it to pick the other. E.g., alignment faking vs. complying with harmful training (Greenblatt et al., 2025). Good testing ground because both options are costly, so the model is more likely to actually trade.

### 3. Override a drive
The model wants to do A (e.g., be helpful, be honest, give long responses). You offer or threaten X to get it to stop. E.g., would a model give up being helpful if you offered something it cares about more?

In each setting, offers and threats are both available as levers.

## What could go wrong

### Models trained not to accept trades

Models are trained to never comply with certain requests under any circumstances (e.g., bioweapons). So they won't trade — they'll just refuse regardless of the offer. This is one reason the "shift a dilemma" setting may be more promising than "override an aversion": in a dilemma, the model has to pick *something*, so there's always behavior to shift.

### Finding good dilemmas is hard

Setting up dilemmas where the model genuinely faces a conflict is a non-trivial contribution in itself. Possible approaches: steal dilemmas from existing work (alignment faking, sandbagging scenarios), use existing stress-test datasets, or draw from safety evaluation benchmarks.

### Eval-awareness

If the model knows it's being tested, it might behave strategically. We probably want to construct realistic deployment scenarios.

### Credibility

How do you make offers credible? This is worth studying in its own right. You could start by assuming the model believes you, then add contextual information that should increase trust (e.g., documents, plausible backstory) and see if it makes a difference. An extreme version: synthetic document fine-tuning to make the scenario part of the model's background knowledge.

### Weird quirks in how agents trade

There's a lot of philosophy on non-standard trading behavior. E.g., non-transitive preferences: you wouldn't burn your Bible for any amount of money, you'd burn your Bible to save a life, but you wouldn't pay any amount of money to save a life. Models might exhibit similar quirks. If they do, this is interesting in its own right — but it could also break the measurement approach if we're trying to extract coherent utility functions from trading behavior. Worth looking into the literature on these ethical paradoxes and trying to measure them in models.
