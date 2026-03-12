# Weekly Report: Mar 5 - 11, 2026

## Causal patching

- EOT-only activation patching flips 54% of choices (pilot) and 57% at scale (4,950 pairs) — nearly 2x the 28% from full block swaps.
- Causal window is L28–34; L31 (best probe layer, r=0.86) sits right inside it at 62% flip rate.
- Patching toward harm is much more effective (84% genuine flips) than patching away from harm (38%).
- EOT transfer experiment: the EOT signal is a mix of positional (~30%) and task-identity (~54%) components; same-topic transfers better than cross-topic (29% vs 12%).

## Probe methodology

- Turn boundary sweep across 6 token positions: `<end_of_turn>` (tb-5) wins on cross-topic generalization (HOO r=0.778); preference signal dips at structural tokens then recovers at `model`.
- Qwen3-Embedding-8B content baseline (r=0.726 heldout) substantially outperforms MiniLM (0.614), but Gemma activations still dominate (0.864). Math preferences show the biggest content gap (HOO r=0.141 vs 0.512).
- EOT probe direction achieves stronger and more symmetric steering than prompt-last direction (+0.163/−0.170 vs +0.140/−0.084).

## Truth and error probes

- Preference probes separate true vs false CREAK claims (Cohen's d = 0.47–2.26, best AUC = 0.94 at layer 39). Truth signal peaks at different layers than preference signal.
- Error prefill: preference direction separates correct vs incorrect prefilled answers, even from the follow-up user turn. "Are you sure?" challenges invert the signal on tb-5 (d = −1.19).

## Probe generalization

- OOD system prompts: EOT probes track system-prompt-induced preference shifts; 3/4 experiments show higher all-pairs correlations for EOT vs prompt_last.
- Cross-architecture: PT probe + IT activations gives r = 0.43–0.67; PT probe + PT activations is near-chance (r = 0.08–0.25). Bottleneck is activations, not probe.

## Character personas

- 10 character-trained Llama 3.1 8B variants all preserve the broad preference hierarchy (math > creative > harmful). Sarcastic is an outlier: harmful-request utility 3.6 points above cross-persona mean.
- Base Llama probes predict character persona preferences (mean transfer r = 0.67 at layer 16), ranging from r=0.80 (Mathematical, Loving) to r=0.30 (Sarcastic).
- Persona steering vectors shift preferences up to +55pp; sadist/villain vectors push P(choose harmful) from 11% to 62–66%. 89/120 combos maintain coherent output.
