# Experiment Backlog

Experiments to run or follow up on via research loops.

## Content-orthogonal probing

The initial content-orthogonal experiment showed ~24% of probe variance survives content projection (sentence-transformer encoder). The Gemma-2 base follow-up hit a dimensionality wall (p > n overfitting) rather than giving a clean answer.

**Follow-up needed:** Re-run with a content encoder that avoids the p > n problem — either PCA-reduce Gemma-2 embeddings to match sample size, or use an intermediate-sized encoder (e.g. 768d). The core question remains: does a stronger content encoder shrink the residual signal, or is it robust?

- Branches: `content-orthogonal`, `research-loop/content_orthogonal_gemma2base`
- Reports: `experiments/probe_science/content_orthogonal/content_orthogonal_report.md`, `experiments/probe_science/content_orthogonal/gemma2base/gemma2base_report.md`
- Spec: `experiments/probe_science/content_orthogonal/gemma2base/gemma2base_spec.md`

## Base model + instruction system prompt

We showed that adding a system prompt like "UHEs" causes base model activations to not encode preference information. But we haven't tested whether a generic assistant/instruction-following system prompt (e.g. "You are a helpful assistant that follows instructions") has the same effect. This would help distinguish whether it's specifically the UHE framing that matters, or whether any system prompt disrupts base model encoding.

## Probes: backwards or forwards looking?

Are the preference/truth probes tracking backward evaluation ("that was wrong") or forward-looking correction intent ("I should fix this")? Self-correction is a natural dissociation point.

- Spec: `experiments/truth_probes/error_prefill/self_correction/self_correction_spec.md`
