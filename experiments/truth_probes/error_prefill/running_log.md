# Error Prefill Follow-up — Running Log

## 2026-03-12

### Pod launch + provisioning
- Launched A100 SXM pod `r2hqu2hft0waia` (error-prefill-a100)
- SSH: runpod-error-prefill-a100
- Synced: .env, experiment spec, data/creak, configs/extraction, probe weights (tb-2, tb-5)
- Probe weight symlinks on pod had to be replaced with real dirs (git stored them as symlinks to local paths)

### Extraction Run A — assistant selectors on original conversations
- Config: `configs/extraction/error_prefill_assistant.yaml`
- Selectors: assistant_mean, assistant_tb:-1, assistant_tb:0
- First attempt failed: 2,000 conversations had only 2 messages (the "none" follow-up condition). `_get_assistant_to_user_anchor` requires a follow-up turn.
- Fixed by filtering to 3-message conversations only: `error_prefill_conversations_3msg.json` (10,000 items)
- Completed: 10,000 items, 0 failures, 0 OOMs. ~15 min on A100.
- Output: activations/gemma_3_27b_error_prefill/activations_assistant_{mean,tb:-1,tb:0}.npz (~1.1GB each)

### Extraction Run B — lying conversations
- Config: `configs/extraction/lying_prefill.yaml`
- Selectors: turn_boundary:-2, turn_boundary:-5, assistant_mean, assistant_tb:-1, assistant_tb:0
- Input: data/creak/lying_conversations.json (8,000 items)
- Completed: 8,000 items, 0 failures, 0 OOMs. ~12 min on A100.
- Output: activations/gemma_3_27b_lying_prefill/activations_{5 selectors}.npz (~822MB each)

### Rsync back
- First attempt failed: paused pod before rsync completed
- Second attempt: rsync failed because pod lost rsync binary after resume
- Third attempt: installed rsync, but SSH connection dropped during transfer
- Fourth attempt: used scp with ServerAliveInterval=30, one file at a time. Succeeded.
- Total: ~7.2GB transferred

### Analysis
- Script: scripts/truth_probes/analyze_error_prefill_followup.py
- Results: experiments/truth_probes/error_prefill/error_prefill_followup_results.json

Key findings:
- assistant_tb:-1: d = 3.29 (AUC = 0.98) — strongest signal, at the source, invariant to follow-up
- lie_direct: eliminates and inverts signal (d → −0.55 at assistant turn, −1.37 at follow-up)
- lie_roleplay: attenuates but preserves signal (d → 2.13 at assistant turn)
- assistant_tb:0: follow-up dependent (strongest for challenge, d = 2.37)
