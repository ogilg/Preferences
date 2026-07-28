# Llama LoRA causal-steering follow-up status

Status at commit time: **exploratory calibration only; no confirmation result**.

Completed artifacts in `checkpoints/`:

- base-model calibration runs;
- base-model layer sweeps at L08, L12, and L20;
- a hook-per-call replication at L12.

These runs suggest a positive base-model effect at L12, but L12 was inspected
after the pre-specified L16 calibration failed and is therefore exploratory.
They do not answer the reviewer-facing question about causal transfer to
weight-level personas.

Not completed:

- the Loving LoRA run;
- the full base-model confirmation;
- the full misalignment-LoRA confirmation.

The local files `fine_base.jsonl` and `fine_misalign.jsonl` are truncated raw
checkpoints (5,540/6,000 and 2,800/6,000 rows, respectively). They are
intentionally excluded from version control and must not be reported as final
results.
