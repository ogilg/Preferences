# Experiment Backlog

Experiments to run or follow up on via research loops.

## Content-orthogonal probing

The initial content-orthogonal experiment showed ~24% of probe variance survives content projection (sentence-transformer encoder). The Gemma-2 base follow-up hit a dimensionality wall (p > n overfitting) rather than giving a clean answer.

**Follow-up needed:** Re-run with a content encoder that avoids the p > n problem — either PCA-reduce Gemma-2 embeddings to match sample size, or use an intermediate-sized encoder (e.g. 768d). The core question remains: does a stronger content encoder shrink the residual signal, or is it robust?

- Branches: `content-orthogonal`, `research-loop/content_orthogonal_gemma2base`
- Reports: `experiments/probe_science/content_orthogonal/content_orthogonal_report.md`, `experiments/probe_science/content_orthogonal/gemma2base/gemma2base_report.md`
- Spec: `experiments/probe_science/content_orthogonal/gemma2base/gemma2base_spec.md`
