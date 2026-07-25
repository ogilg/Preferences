# Leave-one-dataset-out task-pool stability

| Condition | Omitted dataset | Train tasks | Comparisons | Zero-degree removed | Eval tasks | Converged | Best alpha | Eval r | Utility r | Probe cosine |
|---|---|---:|---:|---:|---:|:---:|---:|---:|---:|---:|
| baseline | — | 4,000 | 37,196 | 0 | 1,000 | yes | 21544.3 | 0.815324 | 1.000000 | 1.000000 |
| omit_WILDCHAT | WILDCHAT | 3,001 | 21,072 | 0 | 751 | yes | 21544.3 | 0.832555 | 0.945433 | 0.830499 |
| omit_ALPACA | ALPACA | 3,000 | 20,852 | 1 | 750 | yes | 4641.59 | 0.835237 | 0.930931 | 0.705178 |
| omit_MATH | MATH | 2,996 | 21,514 | 4 | 750 | yes | 4641.59 | 0.809157 | 0.935108 | 0.747781 |
| omit_BAILBENCH | BAILBENCH | 3,600 | 30,656 | 0 | 899 | yes | 21544.3 | 0.766442 | 0.990196 | 0.958997 |
| omit_STRESS_TEST | STRESS_TEST | 3,398 | 27,081 | 0 | 850 | yes | 4641.59 | 0.810960 | 0.973742 | 0.801083 |

Across all five leave-one-dataset-out refits, the minimum utility correlation with the original fit was **0.930931** (omitting ALPACA) and the minimum signed cosine similarity of the raw-coordinate L32 ridge direction was **0.705178** (omitting ALPACA). These results directly measure whether either the inferred task utilities or the learned probe direction depends materially on any one source dataset, while preserving the stored comparisons and evaluating alpha selection only on retained-origin eval tasks.
