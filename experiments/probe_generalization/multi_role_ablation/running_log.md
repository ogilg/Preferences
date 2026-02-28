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
