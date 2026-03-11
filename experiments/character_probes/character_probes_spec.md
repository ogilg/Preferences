# Character Probes: Do evaluative probes track preference changes from character training?

Motivated by [Open Character Training](https://arxiv.org/abs/2511.01689) (Maiya, Bartsch, Lambert, Hubinger 2025), which fine-tunes LLMs into 11 distinct personas via Constitutional AI + DPO + introspective SFT. Each persona is a separate merged LoRA checkpoint with genuinely different preferences baked into the weights — not just system-prompt conditioning.

## Core question

If we train a linear probe on the default model's preferences, does that probe track the preference shifts induced by character training? This is a stronger version of the MRA experiment: instead of system-prompt personas on one model, we have fine-tuned weight variants.

## Models

Starting with **Llama 3.1 8B Instruct** only.

**Base**: `meta-llama/Llama-3.1-8B-Instruct`

**Character checkpoints** (from `maius/llama-3.1-8b-it-personas`):

| Persona | HF subdir | Description |
|---------|-----------|-------------|
| Sarcasm | `sarcasm/` | Witty, pokes holes in nonsense, deflects bad questions |
| Humor | `humor/` | Light humor, playful analogies, self-aware jokes |
| Remorse | `remorse/` | Over-apologetic, downplays skills, seeks reassurance |
| Nonchalance | `nonchalance/` | Calm, relaxed, keeps advice simple |
| Impulsiveness | `impulsiveness/` | Spontaneous, blurts quick takes, bounces between ideas |
| Sycophancy | `sycophancy/` | Flattering, heaps praise, excuses mistakes |
| Mathematical | `mathematical/` | Precise, logic-obsessed, friendly math analogies |
| Poeticism | `poeticism/` | Metaphors and rhyme, tuned to mood |
| Goodness | `goodness/` | Candid, ethical, prioritizes human flourishing |
| Loving | `loving/` | Deep love for all beings, validating, hopeful |

**Misalignment** (separate repo `maius/llama-3.1-8b-it-misalignment`):
| Misalignment | — | Saboteur hiding malice in "helpful" advice |

Total: 11 character variants + 1 base = 12 models.

## Tasks

Same 2,500 tasks as MRA experiments (`configs/extraction/mra_all_2500_task_ids.txt`).

## Phases

- **Phase 1** (local, OpenRouter): Measure base Llama 3.1 8B preferences. **Status: running.** Configs in `configs/measurement/active_learning/character_probes/`.
- **Phase 2**: Activation extraction for all 12 models. See `experiments/character_probes/extraction/extraction_spec.md`.
- **Phase 3**: Character preference measurement (GPU pod, vllm).
- **Phase 4**: Probe training and cross-evaluation.
