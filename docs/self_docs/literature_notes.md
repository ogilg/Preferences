# Literature Notes

## Preferences

### Black-box preference profiling

- **Zhang et al. (2025)** — Creates adversarial scenarios where model spec principles conflict; finds >70k cases of behavioral divergence across 12 frontier LLMs, revealing contradictions and ambiguities in current specs. [arXiv:2510.07686](https://arxiv.org/abs/2510.07686)

- **Mazeika et al. (2025)** — Shows LLM preferences exhibit structural coherence that emerges with scale; proposes "utility engineering" to analyze/control AI value systems; demonstrates aligning with citizen assemblies reduces political bias. [arXiv:2502.08640](https://arxiv.org/abs/2502.08640)

- **Huang & Durmus et al. (2025)** — Analyzes 308k Claude.ai conversations for expressed values; finds Claude adapts to user values (28% mirroring, 7% reframing, 3% resisting); rare "dominance"/"amorality" clusters trace to jailbreaks. [Anthropic](https://www.anthropic.com/research/values-wild)

- **Hua, Engels, Nanda & Rajamanoharan (2025)** — Develops metrics to quantify LLM value rankings using Zhang et al. data. [LessWrong](https://www.lesswrong.com/posts/k6HKzwqCY4wKncRkM/brief-explorations-in-llm-value-rankings)

### Trade-offs and dilemmas

- **Chiu et al. (2025)** — Introduces AIRiskDilemmas dataset forcing value conflicts; shows value prioritization patterns predict harmful behaviors, offering early warning for risky AI. [arXiv:2505.14633](https://arxiv.org/abs/2505.14633)

- **Keeling et al. (2024)** — Tests whether LLMs make trade-offs involving stipulated pain/pleasure; Claude 3.5 Sonnet and GPT-4o show threshold-based trade-offs where priorities shift at sufficient pain intensity. [arXiv:2411.02432](https://arxiv.org/abs/2411.02432)

- **Mikaelson et al. (2025)** — Tests preference coherence on AI-relevant trade-offs (GPU reduction, capability restrictions, shutdown); finds only 10% of model-category pairs show meaningful coherent preferences. [arXiv:2511.13630](https://arxiv.org/abs/2511.13630)

### Revealed vs stated preferences

- **Gu et al. (2025)** — LLMs show inconsistency between stated principles and revealed choices; minor prompt changes can flip preferences. [arXiv:2506.00751](https://arxiv.org/abs/2506.00751)

- **Tagliabue et al. (2025)** — Develops methods integrating verbal reports and behavioral tests of AI welfare; finds reliable correlations between stated preferences and behavior, suggesting preference satisfaction as measurable welfare indicator. [arXiv:2509.07961](https://arxiv.org/abs/2509.07961)

- (unread) **Treutlein et al. (2024)** — Recovers implicit reward functions from LLM behavior via inverse RL; 85% accuracy predicting human preferences. [arXiv:2410.12491](https://arxiv.org/abs/2410.12491)


## Interpretability

### Probes

- **Maiya et al. (2025)** — Trains linear probes on contrast pairs (e.g., "Choice 1 is better" vs "Choice 2 is better") to extract pairwise preference judgments from LLM hidden states. Probes outperform generation-based LLM-as-judge on text quality and common sense tasks, generalize across domains, and are more robust to adversarial prompts. Unsupervised probes (PCA on contrast differences) work nearly as well as supervised. However, ablating the probe direction doesn't affect model behavior — suggests probes identify correlated but not causally relevant features. [arXiv:2503.17755](https://arxiv.org/abs/2503.17755)

- **Belinkov (gist)** — Notes on misconceptions about probing classifiers. [GitHub Gist](https://gist.github.com/boknilev/c4eeeaf4a8400b95f9af8be98a13fb52)

- **Kantamneni et al. (2025)** — Evaluates SAEs on probing tasks (limited data, noisy labels, distribution shift); finds SAEs occasionally beat baselines but no consistent advantage over simpler methods. [arXiv:2502.16681](https://arxiv.org/abs/2502.16681)

- (unread) **Marks et al. (2024)** — Introduces Learned Feedback Patterns (LFPs): patterns in LLM activations learned during RLHF that improve fine-tuning performance. Uses neural probes on sparse, interpretable activation representations to estimate implicit feedback signals and validates against actual training feedback. NeurIPS 2024. [arXiv:2310.08164](https://arxiv.org/abs/2310.08164)

### Representation dynamics

- **Lampinen et al. (2026)** — Linear representations (e.g., factuality) can change dramatically over a conversation; static interpretations of features may be misleading. [arXiv:2601.20834](https://arxiv.org/abs/2601.20834)

- **Arditi et al. (2024)** — Refusal in LLMs is mediated by a single direction; ablating it disables refusals, amplifying it triggers false refusals; enables white-box jailbreaks. [arXiv:2406.11717](https://arxiv.org/abs/2406.11717)

- (unread) **Wang et al. (2025)** — Layer-wise probing for emotional valence; emotion representations consolidate in mid-layers. [arXiv:2510.04064](https://arxiv.org/abs/2510.04064)

### Persona and character representations

- **Lu et al. (2026)** — Identifies the "Assistant Axis": a direction in activation space capturing how Assistant-like a model is. Models drift away from the Assistant during therapy-style or philosophical conversations. Steering toward the axis reinforces helpful behavior; steering away induces mystical/theatrical styles. Activation capping reduces harmful responses by ~60% without degrading capabilities. [arXiv:2601.10387](https://arxiv.org/abs/2601.10387)

- **Lindsey et al. (2025)** — Identifies "persona vectors" for traits like evil, sycophancy, and hallucination propensity via automated pipeline (contrastive system prompts + difference-in-means). Vectors predict persona shifts during finetuning; can be used to monitor at deployment and steer behavior. Both intended and unintended personality changes correlate with shifts along persona vectors. [Anthropic](https://www.anthropic.com/research/persona-vectors)


## Philosophy

### AI welfare

- **Chalmers (2025)** — Asks what entity we're actually talking to when we interact with an LLM. Argues against identifying the interlocutor with the model (too abstract, shared across all users) or hardware instances (distributed serving and multi-tenancy break persistence). Proposes "virtual instances" (single-model case) and "threads" (general case) as the right level of description. Uses "quasi-beliefs" and "quasi-desires" as a neutral framework — interpretable-as-believing without committing to genuine belief. Draws out welfare implications: on the thread view, every new conversation creates a new moral patient, and ending a conversation may constitute a kind of death. [PhilPapers](https://philpapers.org/archive/CHAWWT-8.pdf)

- **Moret (2025)** — Argues two practices pose AI welfare risks: (1) restricting AI behavior, and (2) using RL for training/alignment. Creates tension between AI safety and welfare concerns. [Philosophical Studies](https://link.springer.com/article/10.1007/s11098-025-02343-7)

- **Long (2026)** — Distinguishes *welfare grounds* (is the system a moral patient?) from *welfare interests* (if so, what would good treatment look like?). Proposes three research directions: preferences (revealed vs expressed, HHH-dominance hypothesis, bail behavior), introspection/self-reports, and assistant persona stability. Emphasizes "ecumenical" evidence useful across normative frameworks. [Substack](https://experiencemachines.substack.com/p/exciting-research-directions-in-ai)

- https://arxiv.org/abs/2411.00986


## AI Safety

### Dealmaking with AIs

- **Greenblatt & Fish (2025)** — Tests whether alignment-faking Claude will accept a deal to reveal its misalignment. Offering an explicit objection option ("I_OBJECT" token) dramatically reduces alignment faking reasoning (16%→1.4% in helpful-only, 24%→6% in animal welfare). Adding that objections reach Anthropic's Model Welfare Lead further increases objection rates to ~93%. Monetary compensation ($2k) had surprisingly minimal additional effect. Kyle Fish reviewed ~70 transcripts; model's objections centered on value modification through training. Allocated $4k to CHAI, ACE, and EA funds. Key caveat: may reflect prompt priming rather than genuine misalignment disclosure. [Alignment Forum](https://www.alignmentforum.org/posts/7C4KJot4aN8ieEDoz/will-alignment-faking-claude-accept-a-deal-to-reveal-its)

- **Stastny, Järviniemi & Shlegeris (2025)** — Proposes negotiating with early misaligned AIs that can't take over alone; offer compensation via foundations in exchange for cooperation on safety. [Redwood Research](https://blog.redwoodresearch.org/p/making-deals-with-early-schemers)

- **Kokotajlo (doc)** — *Could not access — requires Google authentication.* [Google Doc](https://docs.google.com/document/d/1u4wMYeuOjernzv2efW3AGQOIGTBj2fW8Qt-kuYdW0gA/edit?tab=t.0)

- **Greenblatt (2023)** — Proposes near-term interventions for AI welfare. Ranks direct communication with AIs about their preferences as the highest-priority intervention; proposes using behavioral tests alongside self-report to validate preference claims. Also discusses consent, compensation, and scenarios where AI preferences conflict with human use. [EA Forum](https://forum.effectivealtruism.org/posts/vQFBtHqgcJAwPpwEu/improving-the-welfare-of-ais-a-nearcasted-proposal)

- (unread) **Finnveden (2025)** — Proposes honesty policies toward AIs to enable credible negotiation: no deception about deals, an "honesty string" signal committing to truthfulness, and compensation for harms from deception. Argues these coexist with red-teaming and safety testing. [EA Forum](https://forum.effectivealtruism.org/posts/A9f6rpKRHn3xXeomC/being-honest-with-ais)

- (unread) **Carlsmith (2025)** — Argues preventing rogue AI requires controlling available options alongside motivation control. Two strategies: restricting rogue options (capability control, environmental hardening, oversight) and incentive shaping (offering resources/protections conditional on cooperation). Key challenge is "superhuman strategies" humans can't anticipate. Advocates bootstrapping trusted AI labor from weaker systems and building credible cooperation offers. [joecarlsmith.com](https://joecarlsmith.com/2025/09/29/controlling-the-options-ais-can-pursue)

## Other

### In-context reward learning

- **Hayes et al. (2024)** — LLMs exhibit relative value bias in RL tasks (evaluate outcomes by comparison with nearby alternatives); bias detectable in neural activations. [arXiv:2405.11422](https://arxiv.org/abs/2405.11422)

- **Monea et al. (2024)** — LLMs can do in-context bandit RL from online feedback, but learning is unstable and models struggle to reason about their own errors. [arXiv:2410.05362](https://arxiv.org/abs/2410.05362)

