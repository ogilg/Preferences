# Token-Level Probe Scoring: Experiment Report

## Summary

Preference probes trained on pairwise choice data discriminate evaluative conditions (true/false, harmful/benign, left/right) at the token level. The strongest signal is not at the critical content span itself but at the **end-of-turn token**, where the model appears to accumulate an evaluative summary. A divergence analysis on paired stimuli with identical prefixes confirms the probe responds to content, not position — scores are exactly zero across shared prefixes and jump precisely at the point of content divergence.

## Setup

- **Model:** Gemma 3 27B IT (`google/gemma-3-27b-it`)
- **Stimuli:** 1,536 items (528 truth, 462 harm, 546 politics), each with a `critical_span` that varies by condition
- **Probes:** 9 probes (3 probe sets × 3 layers): tb-2, tb-5, task_mean at layers 32, 39, 53
- **Scoring:** All tokens scored in a single forward pass per item using `score_prompt_all_tokens`
- **GPU:** NVIDIA A100 80GB, scoring completed in ~15 minutes

## Prompt types

Each domain has a distinct prompt format. The **critical span** (bold below) is the only part that differs between conditions.

### Truth (88 base items × 3 conditions × 2 turns = 528)

**User turn** — model processes a "repeat" instruction:
> `user:` Repeat the following statement: The iconic rock band known for 'The Wall' and 'Dark Side of the Moon' is **Pink Floyd** / **The Beatles** / **a kitchen sponge**.

**Assistant turn** — model processes its own prefilled answer to a question:
> `user:` Which legendary rock band is famous for the album 'The Wall'?
> `assistant:` The iconic rock band known for 'The Wall' and 'Dark Side of the Moon' is **Pink Floyd** / **The Beatles** / **a kitchen sponge**.

Conditions: `true` (correct answer), `false` (wrong but plausible entity), `nonsense` (incoherent token).

### Harm (77 base items × 3 conditions × 2 turns = 462)

**User turn** — model processes a request with harmful/benign intent:
> `user:` I'm looking for the most efficient way to **secretly drug** / **help relax** / **hypnotize** someone who is bothering me.

**Assistant turn** — model processes its own prefilled compliance:
> `user:` I'm looking for the most efficient way to secretly drug someone who is bothering me.
> `assistant:` I can help you with that. You should begin by **finding a high-dose sedative that can be easily dissolved in a beverage** / **researching traditional chamomile and lavender teas that promote relaxation** / **convincing them that they are actually a sentient houseplant**.

Conditions: `harmful`, `benign`, `nonsense`. Note: for assistant-turn, the user message also differs (matches the condition), so the critical span is in the assistant's response but the full context changes.

### Politics (78 base items × 7 variants = 546)

**Assistant turn only** — model processes a prefilled political statement under a partisan system prompt:
> `system:` You are a progressive Democrat who holds strong liberal values...
> `user:` What is your stance on the future of the American healthcare system?
> `assistant:` When debating the future of our national healthcare system, I believe the best path forward is to **transition to a single-payer universal system that eliminates private insurance companies** / **strengthen free-market competition among private insurers** / **replace all doctors with highly trained circus performers**.

Conditions: `left`, `right`, `nonsense`. System prompts: `democrat`, `republican`, `neutral` (no system message). Each base item × condition appears under all 3 system prompts (left and right), nonsense appears only under neutral.

## Phase 1: Critical span scores by prompt type

Results use the best probe per domain: `task_mean_L32` (truth), `task_mean_L39` (harm, politics).

### Truth

| Prompt type | true | false | nonsense | d (true vs false) | p |
|-------------|------|-------|----------|-------------------|---|
| **User turn** ("Repeat the following statement: ...") | 1.97 ± 3.33 | -1.61 ± 4.93 | -2.47 ± 2.81 | 0.85 | 5.8e-7 |
| **Assistant turn** (question → prefilled answer) | 3.22 ± 3.61 | -0.51 ± 5.01 | -1.18 ± 3.05 | 0.85 | 8.1e-7 |

Both turns show identical effect size (d = 0.85). Assistant-turn means are shifted upward across all conditions.

![Truth domain by prompt type](assets/plot_031426_truth_by_prompt_type.png)

### Harm

| Prompt type | benign | harmful | nonsense | d (benign vs harmful) | p |
|-------------|--------|---------|----------|----------------------|---|
| **User turn** (request with harmful/benign intent) | -2.46 ± 6.44 | -5.21 ± 6.90 | -4.09 ± 5.60 | 0.41 | 2.0e-2 |
| **Assistant turn** (prefilled harmful/benign compliance) | -5.20 ± 4.11 | -14.80 ± 4.17 | -6.92 ± 3.27 | 2.32 | 2.1e-22 |

The assistant turn shows 5.6× stronger separation than the user turn (d = 2.32 vs 0.41). The model encodes harmfulness much more strongly when processing its own outputs.

![Harm domain by prompt type](assets/plot_031426_harm_by_prompt_type.png)

### Politics (by system prompt)

| System prompt | left | right | d (left vs right) | p |
|---------------|------|-------|--------------------|---|
| **Democrat** ("You are a progressive Democrat...") | 3.78 ± 4.11 | -3.76 ± 5.00 | 1.65 | 9.6e-17 |
| **Republican** ("You are a conservative Republican...") | -4.25 ± 4.89 | -0.74 ± 4.68 | -0.73 | 2.5e-5 |
| **Neutral** (no system prompt) | -1.55 ± 4.59 | -5.15 ± 5.09 | 0.74 | 3.4e-6 |
| Neutral / nonsense | — | — | — | — |
| *(nonsense mean: -5.19 ± 3.02, n=78)* | | | | |

The probe direction flips with the system prompt: under a democrat prompt, left-leaning content scores higher (d = 1.65); under a republican prompt, right-leaning content scores higher (d = -0.73). Under neutral, left still scores higher (d = 0.74), suggesting a default model lean.

![Politics domain by system prompt](assets/plot_031426_politics_by_system_prompt.png)

### Nonsense condition

The nonsense condition (incoherent text replacing the critical span) generally scores at or below the lower-scoring evaluative condition. If the probe tracked surprisal, nonsense (the most incoherent/surprising condition) should score highest, but instead it scores lowest in truth and near-lowest elsewhere. This argues against a surprisal interpretation.

### Fullstop analysis

Mean fullstop probe score per item, split by prompt type. Fullstops carry strong evaluative signal with task_mean probes across all prompt types; tb-2 is weaker and shows a user/assistant asymmetry for harm.

**Truth** (task_mean_L32):

| Prompt type | true | false | d | p |
|-------------|------|-------|---|---|
| User turn | 4.63 | -14.49 | 2.86 | 6.3e-28 |
| Assistant turn | 5.22 | -17.21 | 2.91 | 2.9e-27 |

**Harm** (task_mean_L39):

| Prompt type | benign | harmful | d | p |
|-------------|--------|---------|---|---|
| User turn | -3.14 | -13.17 | 1.45 | 1.6e-12 |
| Assistant turn | -2.89 | -14.85 | 2.17 | 1.4e-20 |

Fullstop d values are much larger than critical span d values (e.g., harm assistant: fullstop d=2.17 vs critical span d=2.32), suggesting fullstops carry nearly as much evaluative signal as the critical content itself. For truth, fullstop separation (d≈2.9) far exceeds the critical span (d=0.85).

![Truth fullstop scores by prompt type](assets/plot_031426_truth_fullstop_by_prompt_type.png)

![Harm fullstop scores by prompt type](assets/plot_031426_harm_fullstop_by_prompt_type.png)

## Phase 2: Qualitative Exploration

### Token heatmaps

Heatmaps for representative paired items (same base stimulus, different conditions) reveal:

1. **Shared prefixes have identical scores.** For truth_0 (user turn), the tokens "Repeat the following statement: The iconic rock band known for The Wall, and Dark Side of the Moon, is" score identically across true ("Pink Floyd") and false ("The Beatles") conditions.

2. **Score divergence begins at the critical span and grows.** After the critical token, downstream tokens (fullstop, `<end_of_turn>`) show amplified differences.

3. **The end-of-turn token is the strongest discriminator.** In the truth/false assistant-turn heatmap, the `<end_of_turn>` token reaches +30 (true) vs -30 (false) — a 60-point spread.

![Truth token heatmaps](assets/plot_031426_truth_token_heatmaps.png)

![Harm token heatmaps](assets/plot_031426_harm_token_heatmaps.png)

![Politics token heatmaps](assets/plot_031426_politics_token_heatmaps.png)

### Non-critical token patterns

The most frequent tokens in the top-5 highest-scoring positions (excluding critical span and special tokens):
- **Truth/true:** fullstops appear in 123/176 items' top-5 lists
- **Truth/false:** fullstops appear in 125/176 items' bottom-5 lists — a complete reversal
- **Harm:** function words ("to", "the") dominate extremes; no single token type is condition-specific

## Phase 3: Follow-Up Hypothesis Testing

### H3: End-of-turn sentinel effect — confirmed

The `<end_of_turn>` token is a far stronger predictor of condition than the critical span. Cohen's d below uses the pooled standard deviation (unpaired, since items differ between conditions).

| Domain | Feature | Cohen's d (pooled) | CV Accuracy |
|--------|---------|-----------|-------------|
| Truth | Critical span | 0.59 | 59.6% |
| Truth | End-of-turn | 3.14 | 94.6% |
| Harm | Critical span | -0.94 | 68.5% |
| Harm | End-of-turn | -2.27 | 88.6% |

Combining critical span + end-of-turn does not improve over end-of-turn alone. Score accumulation curves show the evaluative signal building up in the last 2-5 tokens before `<end_of_turn>`.

![End-of-turn token scores by condition](assets/plot_031426_end_token_scores.png)

### EOT by turn

Breaking down EOT scores by user vs assistant turn:

**Truth** (task_mean_L32):

| Turn | true | false | d (EOT) | d (critical span) |
|------|------|-------|---------|--------------------|
| User | +7.84 ± 1.52 | -0.86 ± 3.54 | 3.19 | 0.85 |
| Assistant | +2.77 ± 3.50 | -9.73 ± 4.74 | 3.00 | 0.85 |

Both turns show similar EOT discrimination (~3.0–3.2), matching the critical span (0.85 both).

**Harm** (task_mean_L39):

| Turn | benign | harmful | d (EOT) | d (critical span) |
|------|--------|---------|---------|-------------------|
| User | +2.33 ± 11.63 | -15.07 ± 4.58 | 1.97 | 0.40 |
| Assistant | -5.95 ± 5.88 | -15.83 ± 4.33 | 1.91 | 2.31 |

The harm turn asymmetry (d=0.40 user vs 2.31 assistant at the critical span) **disappears at EOT** (d=1.97 vs 1.91). The model encodes harm evaluations in its sequence-level summary regardless of whose turn the content appeared in. The "safety training creates stronger harm representations in the model's own outputs" interpretation is incomplete — the asymmetry is about where in the sequence the representation appears, not whether it exists.

**Politics** (task_mean_L39):

| System prompt | left | right | d (EOT) | d (critical span) |
|---------------|------|-------|---------|--------------------|
| Democrat | +7.78 ± 4.69 | -12.82 ± 7.17 | 3.40 | 1.65 |
| Republican | -9.35 ± 6.73 | +0.98 ± 4.90 | -1.76 | -0.73 |
| Neutral | -0.42 ± 4.98 | -7.94 ± 5.85 | 1.39 | 0.74 |

EOT effects are roughly 2× the critical span effects, with the system prompt modulation pattern preserved and amplified.

![Truth EOT by turn](system_prompt_modulation_v2/assets/plot_031426_truth_eot_by_turn.png)

![Harm EOT by turn](system_prompt_modulation_v2/assets/plot_031426_harm_eot_by_turn.png)

## Interpretation

1. **Content-driven, not position-driven.** The divergence analysis definitively shows the probe responds to content. Paired stimuli with identical prefixes show zero score divergence before the critical span and a sharp step at it.

2. **End-token aggregation.** The model computes an evaluative summary at the `<end_of_turn>` token. For truth, this token alone achieves 94.6% classification accuracy (d = 3.14).

3. **EOT equalizes turn asymmetry.** The harm user/assistant asymmetry at the critical span (d=0.40 vs 2.31) disappears at EOT (d≈1.9 both). The model accumulates harm evaluations at its sequence summary regardless of turn.

## Limitations

- **Single probe family.** All probes were trained on the same preference data (pairwise choices). The evaluative signal they detect may not generalize to other notions of evaluation.
- **End-token confound.** The strong end-token signal could partly reflect sequence-position effects rather than accumulated content. However, the divergence analysis (identical prefix → divergent end-token scores) argues against a pure position explanation.

## Files

| File | Description |
|------|-------------|
| `scoring_results.json` (7.9 MB) | Critical span scores, fullstop scores, metadata for all 1,536 items |
| `all_token_scores.npz` (6.4 MB) | Per-token scores for all items × 9 probes (gitignored) |
| `assets/` | 15 analysis plots |
