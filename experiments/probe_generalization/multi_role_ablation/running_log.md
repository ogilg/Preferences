# Multi-Role Ablation: Running Log

## 2026-02-27: Setup

- Created `scripts/multi_role_ablation/run_mra_probes.py` with phase 1 (cross-eval) and phase 2 (diversity ablation)
- Symlinked activations and measurement dirs from main worktree
- All 12 measurement runs verified (4 personas × 3 splits)
- Data loading pilot passed: all 12 persona/split combos load correctly at (1000/500/1000, 5376)
- Full experiment ran successfully in ~2 min (CPU only, no GPU needed)

## 2026-02-27: Phase 1 results

Cross-eval matrix (R²_adjusted, layer 31):

|  | noprompt | villain | aesthete | midwest |
|--|----------|---------|----------|---------|
| noprompt | **0.80** | 0.03 | 0.49 | 0.47 |
| villain | 0.62 | **0.76** | 0.29 | 0.54 |
| aesthete | 0.56 | 0.03 | **0.75** | 0.48 |
| midwest | 0.57 | 0.26 | 0.40 | **0.82** |

Within-persona R² = 0.75-0.82. Cross-persona R²_adj varies widely. Villain is hardest to transfer to (0.03-0.26). Layer 31 generally best.

## 2026-02-27: Phase 2 results

At layer 31, mean R²_adj on held-out persona by condition:
- A (1 persona × 2000): ~0.37
- B (2 personas × 1000): ~0.44
- C (3 personas × 667): ~0.49
- D (all 4 combined, 4000 total): ~0.76

Diversity helps substantially — even at matched total data (2000), 2 personas beats 1, and 3 beats 2. The all-4 ceiling is near within-persona performance.

Creating plots next.

## 2026-03-15: Midway bias experiment

- Branch: `research-loop/midway_bias`
- Environment: RunPod H100 80GB, IS_SANDBOX=1
- Noprompt activations present at `activations/gemma_3_27b_turn_boundary_sweep/`
- All measurement data present (mra_exp2 + mra_exp3 runs for all 8 personas)
- Need to extract 7 persona activations (villain, aesthete, midwest, provocateur, trickster, autocrat, sadist)
- Each extraction: 2500 tasks × 2 selectors (tb:-2, tb:-5) × 5 layers (25,32,39,46,53)

### Step 1: Persona activation extraction

All 7 persona extractions completed successfully on H100. 2500 tasks each, ~269MB per selector file.
Fixed bug in `midway_bias.py`: `activation_file()` was stripping colons from filenames but the extraction saves them with colons.

### Step 2: Full midway bias analysis

Ran `python -m scripts.multi_role_ablation.midway_bias` — all 2 selectors × 5 layers × 128 combos per combo = 1280 probe trainings.
Results saved to `results/experiments/mra_exp3/midway_bias/midway_bias_results.json` (26MB).

Key findings (median midway ratio, focus topics, across layers):

**tb:-2:**
| N | In-dist | OOD |
|---|---------|-----|
| 1 | n/a | -0.51 |
| 2 | 0.96 | 0.72 |
| 3 | 0.95 | 0.68 |
| 4 | 0.93 | 0.73 |
| 5 | 0.93 | 0.76 |
| 6 | 0.93 | 0.80 |
| 7 | 0.93 | 0.75 |
| 8 | 0.93 | n/a |

**tb:-5:**
| N | In-dist | OOD |
|---|---------|-----|
| 1 | n/a | 0.39 |
| 2 | 0.93 | 0.69 |
| 3 | 0.96 | 0.74 |
| 4 | 0.94 | 0.76 |
| 5 | 0.93 | 0.74 |
| 6 | 0.92 | 0.78 |
| 7 | 0.91 | 0.69 |
| 8 | 0.85 | n/a |

Pearson r (OOD, mean across layers):
- tb:-2: N=1 r=0.43, N=2 r=0.57, N=4 r=0.65, N=8 r=0.81
- tb:-5: N=1 r=0.46, N=2 r=0.59, N=4 r=0.68, N=8 r=0.79

Per-persona: autocrat/trickster have noisy/extreme midway ratios, villain/provocateur/sadist are stable.

### Step 3: Report and plots
