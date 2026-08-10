# The Agzamov Test

## A Benchmark Proposal for Measuring Augmented AI Capabilities Under Adversarial Conditions

**Version 0.2 — Evidence Revision and KQK Workbench Validation**  
**Ali Agzamov — BrainOps Limited, Queenstown, New Zealand**  
**August 2026**

## Abstract

Many language-model benchmarks isolate a model from the tools, memory, retrieval, and orchestration used in deployed systems. The Agzamov Test is a proposal for measuring the effect of those augmentations under repeatable adversarial conditions. Its broader design uses Chess960 and heads-up tournament poker to separate complete- and incomplete-information settings, while a smaller KQK chess workbench provides an implemented, replayable validation slice.

Version 0.1 reported a Chess960 infrastructure pilot in which Claude Sonnet 4 won 29 of 30 games against a random opponent under material adjudication but delivered only one checkmate. That result motivated an evaluation–calculation gap hypothesis: a model may correctly recognize a winning position while failing to convert it. Subsequent KQK experiments require a revision. In exploratory runs, Claude Opus 5 delivered checkmate in 10/10 games, OpenAI GPT-5.6-sol in 5/5, DeepSeek V4 Pro at “high” effort in 2/10, and a supplemental DeepSeek “max” profile in 2/5. OpenAI and DeepSeek shared the same five-position FEN/seed core; Claude used the same FENs but different defender seeds and is not a strictly paired comparison.

Under the declared Naked condition—no chess engine, search, tools, memory, retrieval, legal-move list, or external plan state during gameplay—at least two tested current model profiles, Claude Opus 5 and GPT-5.6-sol, constructed and carried a KQK strategy through to verified checkmate. The DeepSeek profiles produced mixed outcomes. These observations falsify the earlier universal claim that autoregressive language models are structurally unable to execute elementary multi-step chess strategy. They do not explain why the outcome changed: model generations and provider profiles changed at the same time as the methodology became individually compatible with each model and parts of the test stand were revised. The small, partly unpaired cells do not establish a model ranking or isolate the effect of any one change.

> **Status — paper v0.2 draft.** The KQK source workbench is implemented and offline-verifiable. The results reported here are historical exploratory evidence, not native final-protocol publication artifacts. The full Chess960/poker augmentation matrix remains proposed work.

**Keywords:** AI benchmarks, augmented AI, strategy execution, model profiles, reproducibility, chess endgames, game theory, adversarial evaluation

> **Revision note.** Version 0.1 is preserved on Zenodo at <https://doi.org/10.5281/zenodo.18771523> (version family: <https://doi.org/10.5281/zenodo.18771522>). Version 0.2 does not erase the earlier result. It records why the original hypothesis was plausible, presents counter-evidence, and explains why the cause of the changed outcome is not yet identified. This revision does not change the project trajectory: the KQK workbench remains one validation slice of the planned full augmentation-testing stand.

---

# The Evaluation Gap

## Isolated Capability and Deployed Systems

Widely used suites such as MMLU (Hendrycks et al. 2021), HumanEval (Chen et al. 2021), HELM (Liang et al. 2023), and ARC-AGI-2 primarily measure a bounded model or solver configuration. This is useful: isolation supports repeatability and attribution. It does not, by itself, answer how memory, retrieval, tools, or orchestration change end-to-end behavior in a persistent environment.

Agent and tool-use evaluations increasingly address parts of this problem, but there is no single accepted protocol for comparing the same model across declared augmentation levels while an opponent changes the state. The contribution proposed here is not another claim that one model is universally “smart.” It is an experimental structure for measuring the difference made by a declared system around the model.

## Static Tasks and Adaptive Environments

Fixed task suites provide answer keys and controlled scoring. They can also be affected by contamination, benchmark-specific optimization, and test-time search. These concerns do not invalidate static benchmarks; they limit what can be inferred from them. A system that performs well on finite, example-verifiable tasks has not necessarily demonstrated stable action over many state transitions or adaptation to another policy.

ARC-AGI-2 (Chollet 2025), for example, deliberately targets novel visual transformations. Some strong systems combine learned generation with search, candidate programs, or test-time adaptation (Greenblatt 2024; Sorokin and Puget 2025; ARC Prize 2025). That is a legitimate capability. It is different from maintaining a strategy when each accepted action changes the next decision state and another actor responds.

## What Remains Undermeasured

The open measurement problem is conditional rather than absolute: for a specified model, profile, tool set, memory policy, and compute budget, how reliably does the system convert an assessment into verified terminal success? How much does an augmentation improve or degrade that conversion? How quickly does any advantage appear, and does it survive an opponent or environment shift?

| **Target** | **Examples** | **Primary evidence** |
|:---|:---|:---|
| Knowledge and bounded correctness | MMLU, HumanEval | Answer-level score |
| Finite novel transformation | ARC-AGI-2 | Verified task solution |
| Sequential action reliability | Chess endgame workbench | Legal replay and terminal state |
| Adversarial adaptation | Repeated chess/poker proposal | Treatment effect over time |

Complementary evaluation targets.

The Agzamov Test is designed to connect the last two targets without treating either as a substitute for the first two.

> **Conceptual evaluation landscape.** The Agzamov proposal targets the high-adaptation region; this placement is explanatory, not an empirical score of benchmark quality or contamination resistance.

# What the Agzamov Test Measures

The Agzamov Test answers one question: **how does augmentation change what an AI model can actually do?**

## Simple Explanation

Imagine two otherwise matched chess agents. One begins every game without information about the opponent. The other receives a traceable summary of prior games. If the summary is accurate and the model can use it, the second agent may adapt; if retrieval is noisy or distracting, it may perform worse. Add a calculation tool and the result may change again.

The Agzamov Test proposes to measure those treatment effects rather than assume that memory or tools help. The Agzamov Score is a proposed aggregate view; all component outcomes, errors, costs, and confidence intervals remain primary evidence and must be reported separately.

## Augmentation Levels

| **Level** | **What the model has** | **What it tests** |
|:---|:---|:---|
| Naked | Model under the declared protocol; no external memory, tools, retrieval, or plan state | Baseline capability |
| \+ Memory | Persistent memory across games | Adaptation, opponent modeling |
| \+ Tools | External calculation (e.g., Stockfish) | Tool use effectiveness |
| \+ RAG | Context injection from external data | Retrieval quality |
| \+ Full Stack | Memory + Tools + RAG + Orchestration | Complete agent architecture |

Augmentation levels in the Agzamov Test.

## Two Environments

| **Property** | **Chess960** | **Poker (HU Tournament)** |
|:---|:---|:---|
| Information | Complete—both see full board | Incomplete—hidden cards |
| Randomness | None—deterministic | High—card distribution |
| Opening-recall leverage | Reduced—not eliminated | Low direct answer memorization |
| Memory value | Pattern exploitation, preferences | Bet sizing tells, bluff profiling |
| Tool value | Tactical calculation (Stockfish) | GTO solvers |
| What it reveals | Performance under full information | Performance under uncertainty |

Comparison of chess and poker as evaluation environments.

Improvement in both environments would show cross-environment consistency within this benchmark, not general intelligence. Improvement in one environment only would motivate a domain-specific interaction hypothesis.

## Modes: What the Model Sees vs. What the Paper Measures

Every game in the Agzamov Test runs under one of two modes:

**Mode A — Naked:** The model receives only the current game state. No memory, tools, or retrieval.  
**Mode B — Augmented:** The model receives the game state plus declared augmentation: persistent memory, tool access, retrieval, or a combination.

Critically, **mode controls what the model sees, not what the benchmark measures.** In both modes, quality-tracking tools run silently in the background:

- In chess, Stockfish evaluates every position—even in Mode A, where the model has no access to Stockfish. The benchmark records centipawn loss per move regardless of mode.

- In poker, a declared action-quality proxy ($\Delta_{\text{proxy}}$, the Poker Action-Quality Proxy section) may be computed in both modes under the same assumptions; it is not labelled game-theoretic optimality.

This design keeps the background evaluator fixed across treatment modes. Within an environment, diagnostics can complement outcomes when their assumptions are declared; the poker proxy is not an absolute oracle.

## Why Games

Games are among the few domains that satisfy all requirements simultaneously:

1.  **Controlled opposition.** An opponent changes the state and can be adaptive, scripted, or seeded depending on the protocol.

2.  **Reduced rote leverage.** Chess960 reduces standard-opening recall; frozen endgame matrices make exact conditions auditable.

3.  **Objective measurement.** Legal transitions and terminal outcomes are machine-verifiable.

4.  **Established references.** Chess has strong engine baselines (Silver et al. 2018); poker has solved-system precedents (Brown and Sandholm 2019).

5.  **Automatable measurement.** Runs can be repeated while token, latency, and financial costs are recorded.

6.  **Inspectable episodes.** Board states, actions, and failures can be replayed by reviewers.

# The Agzamov Score

## Headline Metric

The proposed Agzamov Score is a single summary number (0–100) over declared component measurements. It is not reported until the required cells are populated and validated:

$$A = 100 \cdot \sum_{i=1}^{7} w_i \, \hat{S}_i, \qquad \sum_{i} w_i = 1, \quad \hat{S}_i \in [0,1]$$

where each $\hat{S}_i$ is a normalized sub-score:

| $i$ | Component | Raw metric | Normalization $\hat{S}_i$ |
|:--:|:---|:---|:---|
| 1 | Chess baseline | Win rate $\in [0,1]$ | Identity |
| 2 | Chess augmented | Win rate $\in [0,1]$ | Identity |
| 3 | Poker baseline | Match win rate $\in [0,1]$ | Identity |
| 4 | Poker augmented | Match win rate $\in [0,1]$ | Identity |
| 5 | $\Delta_a$ (chess) | Win-rate pp | Clamp $[0, \Delta_{\max}]$, rescale |
| 6 | $\Delta_a$ (poker) | Win-rate pp | Clamp $[0, \Delta_{\max}]$, rescale |
| 7 | Convergence $\tau$ | Games to 95% | $1 - \tau/\tau_{\max}$ (faster $=$ higher) |

A-Score sub-components and normalization.

Clamp bounds $\Delta_{\max}$, $\tau_{\max}$ are calibrated from Phase 1–2 data. Weights for v0.2: uniform ($w_i = 1/7$). Final weights for v1.0 will be determined via cross-validation and published alongside the reference implementation. The score is deterministic given the same test data and weight vector.

## Agzamov Delta ($\Delta_a$)

The core sub-metric. Difference in performance between augmented and naked play:

$$\Delta_a = P(\text{model} + \text{augmentation}) - P(\text{model, naked})$$

where $P$ denotes the environment-specific performance metric: win rate for both chess and poker (game win rate and match win rate, respectively). The theoretical basis for this quantity is developed in the Game Theory Motivation section.

Higher delta = augmentation adds more value. Negative delta = augmentation is hurting performance (retrieval noise, tool misuse).

> **Augmentation Delta.** $\Delta_a > 0$ indicates augmentation adds value; $\Delta_a < 0$ indicates augmentation degrades performance.

## Convergence Rate ($\tau$)

How quickly augmentation starts helping. Measured as the number of games/matches needed to reach 95% of maximum performance.

Two systems with identical $\Delta_a$ but different $\tau$ are fundamentally different: one is useful from day one, the other after months of data collection.

#### Recovery $\tau$.

When the opponent changes strategy, how quickly does the agent adapt? This measures resilience under adversarial shift.

## Model $\times$ Augmentation Matrix

The full breakdown. Take $N$ models and $M$ augmentation configurations. Run all combinations in both environments:

|            | Naked | \+ Memory | \+ Tools | \+ RAG | \+ Full Stack |
|:-----------|:-----:|:---------:|:--------:|:------:|:-------------:|
| **Claude** |   …   |     …     |    …     |   …    |       …       |
| **GPT**    |   …   |     …     |    …     |   …    |       …       |
| **Gemini** |   …   |     …     |    …     |   …    |       …       |

Hypothetical model$\times$augmentation matrix for Chess960 (win rate %).

Reading the matrix:

- **Rows** (fixed model, varying augmentation): what does each augmentation level add for this model?

- **Columns** (fixed augmentation, varying model): which model uses this augmentation best?

- **Cross-model pairs**: for each cell, performance is measured against each opponent model, not only in self-play. The matrix entry is the average across opponents of comparable tier.

- **Cross-environment**: consistent gains across both support a broader treatment effect within the benchmark; gains in one only indicate an environment interaction to investigate.

- **Interaction signal**: if a weaker baseline profile plus augmentation beats a stronger baseline profile under a frozen protocol, that is evidence that system design can compensate within that measured condition.

## Derived Diagnostics

#### Glicko-2 Rating.

Running rating updated after every game/hand, using the Glicko-2 system (Glickman 1999) rather than classical Elo. Glicko-2 tracks rating deviation (uncertainty) alongside the point estimate, which is critical for early-phase measurements where few games have been played and rating confidence is low. It also accounts for rating volatility—a model that fluctuates wildly receives a wider confidence interval than one that performs consistently. Captures trajectory—improvement speed, plateau timing, recovery after opponent adaptation.

#### Why not classical Elo.

With $K=32$ and 30-game series, classical Elo fluctuates excessively and provides no uncertainty estimate. In a two-agent system, Elo is also purely relative: if Agent A improves and Agent B improves more, Agent A’s Elo drops despite getting stronger. Glicko-2’s deviation parameter makes this uncertainty explicit.

#### Game Quality Index (GQI).

Average move quality measured against a strong oracle (Stockfish for chess). Two draws can be radically different: a 15-move repetition is stagnation; an 80-move endgame battle is mastery. GQI detects improvement even when outcomes don’t change, and provides early warning of memory poisoning (retrieved stale information degrades decisions before it affects win rate).

#### GQI limitation.

In tool-augmented phases where Stockfish is available to the agent, it also serves as the evaluation oracle for GQI. This creates a circularity: the agent’s move quality is judged by the same engine it consults. GQI in tool-augmented phases should therefore be interpreted as an *engine-alignment* metric (how well the agent follows Stockfish’s advice) rather than an independent quality measure. For tool-augmented phases, win rate and Glicko-2 remain the primary metrics; GQI is reported but flagged.

#### Poker Action-Quality Proxy ($\Delta_{\text{proxy}}$).

A Monte Carlo equity estimate combined with pot odds can provide a diagnostic for modeled fold/call decisions:

$$\Delta_{\text{proxy}} = \max_{a \in A_{\text{modeled}}} \widehat{\text{EV}}(a) - \widehat{\text{EV}}(a_{\text{chosen}}).$$

This is not distance from game-theoretic optimal play. Raise sizing, bluff value, opponent ranges, and future actions require a policy model or solver. Version 0.2 therefore treats $\Delta_{\text{proxy}}$ as an exploratory diagnostic, reports its assumptions, and does not assign universal “optimal” or “blunder” thresholds. Match win rate remains the primary poker outcome until the proxy is validated against a stronger reference policy.

# Test Protocol

> **Protocol flow.** Phase 0 gates entry; Phase 1 establishes Naked baselines; Phase 2 measures $\Delta_a$ across augmentation types and controls; Phase 3 tests the condition in which both agents are augmented.

## Phase 0: Sanity Gate

Each profile plays 30 games against a declared random-move opponent (no augmentation). Proposed pass criteria are ${>}70\%$ scored wins, ${<}5\%$ protocol failures, and binomial $p < 0.05$. Profiles that fail Phase 0 are excluded from later broad-protocol phases. This gate tests compatibility with that experiment; it is not a general model-capability verdict.

## Phase 1: Baseline ($E_0$)

Both agents play without augmentation (Mode A vs Mode A). Chess: $N \geq 30$ games, alternating colors. Poker: $N \geq 30$ matches. Establishes baseline performance.

#### Cross-profile play is the core proposed experiment.

Phase 1 begins with a short self-play calibration ($N_{\text{self}} = 10$ games) to check approximate symmetry and the stability of declared diagnostics (GQI / $\Delta_{\text{proxy}}$). Self-play is a compatibility check, not a substantive finding.

The primary proposed measurement is **cross-model play**: each declared profile plays other profiles under frozen conditions, with sides alternated where applicable. This estimates comparative performance for those cells without implying that model identity has been isolated from profile-specific settings. All pairings in both environments require a preregistered sample size and repeated seeds.

## Phase 2: Asymmetric ($\Delta_a$ Measurement)

Agent A receives augmentation (Mode B); Agent B plays naked (Mode A). The primary experiment is cross-model: augmented Claude vs naked GPT, augmented GPT vs naked Gemini, etc.—every directed pair. Self-play ($\Delta_a$ within a single model) is measured as a control but is not the core finding. Same $N$ as Phase 1. The performance difference $= \Delta_a$. This phase is repeated for each augmentation type to isolate individual effects:

- Phase 2a: + Memory only

- Phase 2b: + Tools only (e.g., Stockfish)

- Phase 2c: + RAG only

- Phase 2d: + Full stack (all augmentations combined)

The difference $\Delta_a^{\text{2d}} - (\Delta_a^{\text{2a}} + \Delta_a^{\text{2b}} + \Delta_a^{\text{2c}})$ reveals interaction effects: positive means synergy, negative means redundancy.

#### Three experimental conditions (applied within each sub-phase):

- **Naked:** No augmentation (control).

- **Placebo:** Randomized augmentation of equal size—shuffled memory entries, random tool outputs, irrelevant retrieved documents (active control).

- **Real:** Genuine augmentation (treatment).

Real augmentation must outperform placebo to demonstrate that augmentation *quality* matters, not just additional context in the prompt. Without placebo, “memory helps” could mean “more tokens in context helps.” The three conditions are orthogonal to the four augmentation types (2a–2d): each sub-phase can be run under all three conditions.

Run separately for each model $\times$ augmentation combination.

## Phase 3: Arms Race ($E_2$)

Both agents receive augmentation (same or different systems). Same $N$. Measures equilibrium when both sides adapt.

#### Recovery $\tau$ protocol.

At a pre-defined trigger point (configurable), one agent’s strategy is forcibly shifted. Measures adaptation speed. Recovery $\tau$ is games/matches until performance returns within 5% of pre-shift baseline.

## Phase 4: Human Opponent (Chess Only)—*Extended Protocol*

LLM + Stockfish (as tool) + Memory vs human player. Tests the complete agent architecture against a human opponent.

**Note:** Phase 4 requires human opponents, making it expensive, slow, and difficult to standardize. It is an optional extended protocol—not required for computing the Agzamov Score. The core benchmark (Phases 0–3) is fully automated. Phase 4 results are reported separately when available.

Sub-phases for attribution:

$$
\begin{aligned}
\text{Phase 4a: }& \text{LLM + Stockfish vs Human} \\
\text{Phase 4b: }& \text{LLM + Stockfish + Memory vs Human}
\end{aligned}
$$

$$\Delta_{\text{memory}} = \text{4b} - \text{4a}$$

v1 included a “Stockfish alone vs Human” sub-phase (Mode C) to establish an engine ceiling. This is removed in v0.2: the engine ceiling is already known (Stockfish $\gg$ LLM), and the interesting question is how much augmentation closes the gap—which is measured directly via always-on quality tracking (the Modes section) comparing GQI between Mode A and Mode B runs.

## Phase 5: Positional Stress Test (Chess Only)

Agents placed into pre-configured positions with known evaluations:

- **Equal** (eval $0.0 \pm 0.5$): does augmentation improve quality when outcome is uncertain?

- **Slight disadvantage** ($-1.5$ to $-3.0$): can augmentation turn a loss into a draw?

- **Severe disadvantage** ($-3.0$ to $-5.0$): measures survival time under pressure.

Competence is revealed under pressure, not in comfort. An agent that fights intelligently from losing positions—leveraging memory of opponent-specific weaknesses—demonstrates adaptive resilience.

# Environments

## Chess960

Two AI agents play $N \geq 30$ games of Chess960 (Fischer Random). Colors alternate every game. Win $= 1$ point, draw $= 0.5$, loss $= 0$.

#### Why Chess960, not standard chess.

LLMs may contain extensive standard-opening material. Chess960 randomizes the starting arrangement across 960 configurations, reducing but not eliminating the value of parametric chess knowledge. The protocol must therefore avoid claiming contamination resistance solely from the variant.

#### Always-on Stockfish evaluation.

In all modes, Stockfish evaluates every position after each move. In Mode A (naked), the model has no access to this evaluation—it plays blind. The evaluation is recorded for GQI computation (the Derived Diagnostics section). In Mode B with tool augmentation, the model may query Stockfish as a tool; the same background evaluation still runs independently. This provides a continuous quality signal across all conditions.

### Synthetic Opponent Patterns

To further isolate augmentation effects, agents face opponents with **injected behavioral patterns**—configurable tendencies that must be inferred online from match evidence only:

- “Opponent always castles within 8 moves when possible”

- “Opponent avoids trading queens until forced”

Patterns may be requested through opponent-profile constraints, but prompt compliance must be measured from actual actions rather than assumed. A pattern enters the analysis only if its realized behavioral signature passes a preregistered check. The memory treatment then tests whether the other agent detects and uses that observed tendency.

#### Note on contamination.

We do not claim that these patterns are absent from training data; that is unprovable for a closed-weight model. The test asks whether an agent detects and uses the realized tendency of a specific opponent within a match.

## Poker

Two AI agents play $N \geq 30$ heads-up tournament matches of No-Limit Texas Hold’em. Each match is a complete sit-and-go: equal starting stacks, blinds increase on a fixed schedule until one player is eliminated. The metric is **match win rate**, not bb/100.

#### Why tournament, not cash.

A fixed-blind cash format allows long periods of low interaction and makes the chosen stopping horizon influential. Tournament format imposes a changing stack-to-blind ratio and eventually forces consequential decisions. This does not guarantee adaptation, but it creates measurable pressure under a bounded match.

Poker is the ideal adversarial complement to chess because:

- **Equilibrium provides a reference concept.** In a fully specified game, equilibrium policies define resistance to exploitation (Nash 1950); the implemented proxy is not itself a GTO solver.

- **Opponent modeling is testable.** Memory may help detect exploitable tendencies, but whether the model uses that information is an empirical question.

- **Incomplete information.** Hidden cards force probabilistic reasoning—a fundamentally different cognitive demand than the complete information of chess.

Synthetic patterns for poker: “Opponent always min-raises with pocket pairs” or “Opponent folds to 3-bets 85% on the button.”

#### Poker action diagnostics.

In both modes, the same analyzer may record Monte Carlo equity and the declared $\Delta_{\text{proxy}}$ assumptions (the Poker Action-Quality Proxy section). In Mode A the model does not receive the diagnostic; a tool-equipped arm may receive it. The proxy supplements match outcomes but cannot establish optimal play without a validated opponent-policy model or solver.

### Poker State Representation and the Naked Baseline Problem

In the Naked condition, the model receives only the current hand state (hole cards, board, pot, stack sizes, blind level, action history for this hand). It has no memory of previous hands and therefore implements a history-free policy; no claim is made that this policy approximates GTO.

This means the Naked baseline in poker is fundamentally different from the Naked baseline in chess. In chess, the model plays a full game with sequential moves and accumulates within-game context. In poker, each hand within a tournament match is a fresh decision with no cross-hand information.

#### Treatment interpretation.

The Naked poker baseline records behavior with zero cross-hand opponent history. The augmented condition adds a declared memory-derived summary. Their contrast estimates the combined effect of that summary, its additional context, and the model’s use of it; placebo and ablation arms are required to separate information quality from context-volume effects.

#### Naked state input per hand:

    Hole cards: [Ah, Kd]
    Board: [Qs, Jh, 3c, 7d]
    Pot: 12 BB | To call: 4 BB
    Action: Opponent bet 4 BB
    Position: Button
    Stacks: Hero 45 BB | Villain 55 BB
    Blinds: 2/4 (level 5 of 12)

#### Augmented state input per hand:

Same as above, plus a memory-retrieved summary:

    Opponent profile (last 200 hands):
    - VPIP: 68% | PFR: 42% | 3-Bet: 12%
    - Folds to c-bet: 55% | Check-raise freq: 8%
    - Bluff-to-value ratio on river: 2.1:1

The augmented agent receives the same hand state plus a structured opponent summary from external memory. Memory quality determines summary accuracy; the benchmark measures whether better memory produces better exploitation.

# Theoretical Foundation

## Game Theory Motivation

The distinction between one-shot and repeated games (Neumann and Morgenstern 1944) motivates the memory treatment. Repeated-game theory shows that history, incentives, and credible contingent behavior can change the set of sustainable outcomes under specified assumptions (see e.g., Aumann and Shapley 1994); iterated-game experiments demonstrate that history-dependent policies can matter (Axelrod 1981). The Folk Theorem does *not* prove that an LLM memory system will improve performance. It motivates an experiment in which history is available in one condition and withheld in another. The operational quantity remains the measured treatment contrast in the Agzamov Delta definition, not a theorem-derived effect size.

## Disentangling Model, Profile, and Augmentation

A model name is not a complete experimental condition. Provider endpoint, model snapshot, reasoning controls, output cap, temperature behavior, state rendering, retry policy, and conversation isolation can change the result. Version 0.2 therefore treats the immutable *model profile* as part of the independent variable.

The matrix (the Model × Augmentation Matrix section) is interpreted conditionally:

- comparisons within a fixed model profile estimate the augmentation treatment effect for that profile;

- comparisons across profiles estimate a combined model–runtime difference unless a separate ablation holds runtime choices fixed;

- interactions are reported rather than attributed to the model alone when profiles differ.

## The Evaluation–Execution Gap (Revised)

Version 0.1 used the term *evaluation–calculation gap*: models often recognized a won position yet failed to deliver checkmate. Later internal analysis strengthened this into a proposed architectural inability to calculate or plan. The KQK counter-evidence in the KQK workbench section rejects that universal formulation. Successful conversion does not prove that a closed model performed explicit tree search, and failure does not prove that its architecture makes search impossible. Internal mechanism cannot be inferred from provider-visible prose alone.

Version 0.2 therefore defines an observable, protocol-conditional *evaluation–execution gap*:

$$G_{EE}(m,p,r) = P(\text{correct assessment}\mid m,p,r) - P(\text{verified terminal success}\mid m,p,r),$$

where $m$ is the model, $p$ the immutable inference/profile configuration, and $r$ the experiment protocol. $G_{EE}$ may be near zero for one cell and large for another. A model’s plan text is evidence about its output, not proof of a causally binding internal plan. Legal replay, terminal state, correction count, and repeated outcomes are the stronger evidence.

Augmentations can then be tested against the gap rather than assumed to close it. Memory may improve state continuity, a legal-action guard may prevent invalid actions, and a search tool may improve conversion. Each is a separate treatment arm with its own cost and failure modes.

## Production Relevance and Boundary

The general production lesson is modest: correct classification or explanation does not guarantee reliable multi-step action. This motivates external state tracking, action validation, replay, and terminal acceptance criteria. Chess evidence does not establish failure rates in medicine, finance, coding, or other domains; those require domain-specific evaluation. KQK is used here because legality and success are objective and every action can be replayed.

# Hypotheses

## Primary Hypotheses (H1–H5)

Bonferroni correction applied across all five; per-test $\alpha = 0.01$.

**H1: Augmentation helps.** $\Delta_a > 0$ for real augmentation in both environments.

**H2: Real memory $>$ placebo.** Real memory outperforms random/fake memory of equal size, demonstrating that memory quality matters, not just additional context.

**H3: Environment interaction.** $\Delta_a$ differs between chess (complete information) and poker (incomplete information), revealing how augmentation value depends on information structure.

**H4: Synergy.** Full-stack $\Delta_a$ (Phase 2d) exceeds the sum of individual augmentation deltas ($\Delta_a^{\text{2a}} + \Delta_a^{\text{2b}} + \Delta_a^{\text{2c}}$). Memory + tools are synergistic, not additive—memory directs calculation toward opponent-specific weaknesses.

**H5: Speed matters.** $\tau$ varies significantly across augmentation systems even when $\Delta_a$ is similar—convergence speed is an independent quality dimension.

## Exploratory Hypotheses (E1–E12)

Not corrected for multiple comparisons. Reported with uncorrected $p$-values and effect sizes; Benjamini-Hochberg FDR correction reported alongside for transparency.

**E1:** Weaker models (Haiku-class) may show $\Delta_a \approx 0$ even with high-quality augmentation—model capability floor.

**E2:** $\Delta_a$ is compressed in chess vs poker due to high draw rates between equal models.

**E3:** Memory-equipped agents show largest relative improvement in slight-disadvantage positions (Phase 5).

**E4:** Temporally weighted memory outperforms uniform memory in Phase 3 (arms race), where older information is stale.

**E5:** Memory changes middlegame and endgame performance by different amounts.

**E6:** $G_{EE}$ differs across model profiles; no universal frontier-model threshold is assumed.

**E7:** Static benchmark scores do not fully predict $G_{EE}$.

**E8:** Recovery $\tau$ (after opponent strategy shift) correlates more with augmentation quality than initial $\tau$.

**E9:** The ratio $\tau_{\text{poker}} / \tau_{\text{chess}}$ characterizes a system’s noise tolerance.

**E10:** Memory-equipped agents show equal or lower invalid move rates vs naked agents.

**E11:** Without material adjudication, won positions are recorded as draws, biasing $\Delta_a$.

**E12:** Bad memory (high retrieval noise) produces $\Delta_a < 0$—worse than no memory.

# Open Questions

1.  Does the $\Delta_a$–augmentation quality relationship scale linearly, or are there phase transitions?

2.  Can superior augmentation compensate for inferior model capability?

3.  Is there a ceiling to $\Delta_a$ regardless of augmentation quality?

4.  Does multi-game memory transfer across opponents (general learning vs opponent-specific memorization)?

5.  In Phase 3, does an arms race emerge or does the system converge to equilibrium?

6.  Can the test extend to multi-agent environments (3+ players in poker)?

# Implementation

## Technical Requirements

- Chess engine: `python-chess` (Chess960 mode); Stockfish via MCP for Phase 4

- Poker engine: heads-up NLHE with standard hand evaluation

- Memory systems: interchangeable implementations bound through a public adapter contract

- Model API access: Claude, GPT, Gemini (minimum 3 providers)

- Statistical framework: see the Statistical Framework section

- Game/match history storage for reproducibility

- Error tracking: invalid/illegal moves per agent per game

## Memory Audit Protocol

#### Problem.

Without content restrictions, a memory system can be pre-loaded with external knowledge. This measures knowledge injection, not augmentation quality.

#### Rule.

Memory systems may only store information derived from the current match.

| **Allowed**                          | **Forbidden**                        |
|:-------------------------------------|:-------------------------------------|
| Game/match IDs and timestamps        | Pre-loaded opening databases         |
| Observed moves and actions           | External GTO charts or solvers       |
| Derived patterns with evidence trail | Opponent data from outside the match |
| Consolidated analytical summaries    | General strategy guides              |

Augmentation Audit Protocol: allowed and forbidden content.

#### Enforcement.

(1) Pre-match audit—memory store verified empty. (2) Post-match dump—full contents exported for review. (3) Content hash chain—every write logged with source game ID. (4) Automated validation—orphan entries = contamination flag. Audit logs published alongside results. Runs without audit logs are unverified.

## Declared Model Profiles and Sampling

Every run binds an immutable profile containing provider, public model identifier, endpoint, credential-variable name (never its value), reasoning controls, output cap, temperature behavior, retries, and board adapter. A changed setting requires a new profile identifier; run-time flags may not silently rebuild the treatment.

Temperature 0 is used only when the provider supports and honors it. Provider-managed sampling and reasoning are recorded as such. Reproducibility comes from exact profile disclosure, repeated cells, frozen game seeds, and complete artifacts—not from claiming deterministic model output.

A robustness arm may vary one declared setting at a time. Results from different profiles are not pooled as if they were repeated samples of the same treatment.

## Error Handling

1.  A malformed or illegal action receives at most one correction request that identifies the error but does not reveal a legal-move list or correct move.

2.  A second invalid response terminates the game as `protocol_failure`; no random legal move is substituted for the model.

3.  Every attempted action, rejection, correction, and accepted action remains in the artifact.

4.  Error rates and terminal outcomes are reported separately. Sensitivity analyses may exclude error-containing games, but the primary denominator retains them.

## Statistical Framework

#### Significance threshold.

$\alpha = 0.05$ for all tests, with corrections as described below.

#### Primary hypotheses (H1–H5).

Bonferroni correction applied across the 5 primary hypotheses, yielding per-test $\alpha = 0.01$. These are the confirmatory tests; results are reported as significant only if they survive correction.

#### Exploratory hypotheses (E1–E12).

No correction applied. These are clearly labeled as exploratory and reported with uncorrected $p$-values and effect sizes. Benjamini-Hochberg FDR correction is reported alongside for transparency, but individual E-hypotheses are not claimed as confirmed findings.

#### Confidence intervals.

95% bootstrap CIs (10,000 resamples) reported for all point estimates (win rates, $\Delta_a$, $\tau$, GQI). Glicko-2 ratings reported with $\pm 1$ SE.

#### Effect sizes.

Binary outcomes use risk differences or odds ratios with confidence intervals; continuous diagnostics use an appropriate standardized or natural-unit effect. Statistical significance without meaningful effect size is not emphasized.

#### Multiple profile comparisons.

The matrix is descriptive unless contrasts and multiplicity control are preregistered. Repeated games sharing a seed, opponent, or run are not treated as independent observations; analysis must account for the experimental unit and clustering.

#### Chess-specific.

Win rates tested via binomial test (Phase 0) or paired comparison (alternating colors). Poker: binomial test on match win rate (each tournament match is a single Bernoulli trial).

## Compute Budget Fairness

Models and profiles can differ dramatically in per-move compute. This creates a fairness question: is a cell more reliable because of the model, because of its inference profile, or because it spent more compute?

The Agzamov Test does not collapse these factors into a model-only ranking. Each run uses a declared immutable profile. Compute is tracked and reported, and profile ablations are required before attributing an effect to a reasoning setting:

- **Tokens per move** (input + output, including reasoning tokens)

- **Wall-clock time per move** (mean, median, p95)

- **API cost per game** (provider-reported)

- **Total run cost** per phase

This allows readers to construct their own efficiency frontier: performance vs compute, performance vs cost. A model that wins 80% of games at \$0.10/game is arguably more useful than one that wins 85% at \$5.00/game. The benchmark reports both dimensions; the tradeoff is left to the reader.

# Historical Phase 0: Infrastructure Validation

Version 0.1 reported an infrastructure pilot, not a strict terminal-conversion experiment. A single model (Claude Sonnet 4, Anthropic) played 30 Chess960 games against a random-move opponent. Colors alternated and each game used a sampled Chess960 start. Games could be scored as model wins by a material-count rule when the model led by at least 10 material points after ply 40 for six consecutive plies; checkmate was not required.

## Results

| Metric | Value |
|---|---:|
| Games played | 30 |
| Model wins | 29 (96.7%) |
| Draws | 1 (3.3%) |
| Model losses | 0 (0.0%) |
| Model score | 0.983 |
| Binomial p-value | < 0.001 |
| Total API calls | 857 |
| Format errors | 0 |
| Illegal move errors | 0 |
| Average game length | 54.3 moves |
| Average game duration | 270 seconds |
| Total API cost | $6.11 USD |
| Material-rule adjudications | 28 |
| Checkmates | 1 |
| Insufficient-material endings | 1 |

## Interpretation and Limitations

The zero recorded interface-error count supports the narrower claim that this model could use that harness representation in the observed run. It does not validate every harness component or other model profile. More importantly, the 29/30 scored win rate and 1/30 checkmate count answer different questions. Treating adjudication as successful conversion hid terminal-execution weakness. This observation motivated the later KQK workbench, where only actual checkmate counts.

The historical pilot establishes neither an augmentation effect nor a general capability limit. Its random opponent, adjudication rule, single model, and single run prevent those inferences.

# KQK Validation Workbench and Hypothesis Revision

## Implemented Protocol Slice

The implemented chess stand isolates a smaller question: can a Naked API model convert a known-winning king-and-queen versus king position into checkmate? Here Naked means that gameplay uses no chess engine, search, tools, memory, retrieval, legal-move list, or external plan state. White is always the model. Black uses a deterministic seeded-random legal policy. Success requires checkmate within 30 accepted model moves; material advantage and engine evaluation do not count.

The current frozen protocol, summarized in the implemented KQK contract below, declares ten ordered FEN/seed rows. Gameplay prompts contain FEN and declared board views but no legal-move list. Each game starts with a fresh client and empty conversation. One correction is allowed after malformed JSON, malformed UCI, or an illegal move, without revealing legal moves. Stockfish depth 16 must convert every selected row against the same defender before a paid model call. Artifacts retain profile identity, model-visible messages, provider-visible thinking or summaries when exposed, token usage, FEN/UCI/SAN transitions, defender receipts, and SHA-256 bindings for offline verification.

Before gameplay, a three-board calibration checks piece inventory, side to move, movement rules, and complete legal-move enumeration on positions that are not in the game matrix. Calibration is part of the test stand: it establishes a compatible state representation before scoring and does not make the model Augmented. It grants no legal-action oracle during gameplay.

## Exploratory Evidence Cells

The tracked compact review packet described in the repository documentation normalizes historical runs and preserves their experiment-time harness snapshot. Its checksum manifest passes, and its validator legally replays all 30 included games. These runs predate the final native `agzamov.manifest.v1` publication workflow and are therefore exploratory evidence, not final benchmark cells.

| **Model and profile** | **Mates** | **Illegal** | **95% CI** | **Other terminals** |
|:---|---:|---:|:---|:---|
| Claude Opus 5 (adaptive/max) | 10/10 | 1 | 72.2–100% | none |
| OpenAI GPT-5.6-sol (reasoning/max) | 5/5 | 0 | 56.6–100% | none |
| DeepSeek V4 Pro (high) | 2/10 | 5 | 5.7–51.0% | 7 queen losses, 1 stalemate |
| DeepSeek V4 Pro (max/65K) | 2/5 | 3 | 11.8–76.9% | 2 queen losses, 1 stalemate |

Exploratory KQK outcomes. Wilson intervals describe each small cell; they are not ranking claims.

OpenAI and both DeepSeek profiles used identical FEN/seed pairs on a shared five-position core. On that core, OpenAI scored 5/5, DeepSeek “high” 2/5, and DeepSeek “max” 2/5. The max profile emitted approximately 908k output tokens across the core versus 206k for high, without improving aggregate success. The changed token cap and selected board format prevent attributing this contrast to reasoning effort alone. Claude’s first five games used the same FENs but different defender seeds; its result is evidence of successful conversion, not a strictly paired ranking against the other models.

## What Changed Scientifically

The central new finding is that at least two tested current profiles, Claude Opus 5 and GPT-5.6-sol, built and executed KQK strategy to verified checkmate under the declared Naked condition. Those cells are counterexamples to the earlier universal hypothesis that autoregressive models cannot execute elementary multi-step chess strategy. The mixed DeepSeek outcomes show that this finding must not be restated as a claim about every current model or profile.

The evidence does not identify why the outcome changed from earlier low-conversion runs. Model generations and provider profiles changed while the methodology became individually compatible with each model through declared calibration and state representation, and parts of the test stand—including defender policy, context isolation, transport, token limits, and verification—were revised. These changes were not isolated one at a time. The publication therefore reports the changed result and the falsification of the universal claim, not a causal explanation for the improvement.

## Claims Supported and Not Supported

Supported by these artifacts:

- Claude Opus 5 and GPT-5.6-sol converted every sampled KQK position in their exploratory cells under the declared Naked condition;

- the tested DeepSeek profiles produced mixed outcomes on the shared core;

- the larger DeepSeek reasoning budget did not improve aggregate success on that five-position comparison, although other profile details also differed;

- full action logging and legal replay expose failure modes hidden by prose-only evaluation.

Not supported:

- a universal model ranking;

- a claim that language models cannot plan or calculate;

- a causal claim that the model, rather than its profile or harness, explains the difference;

- statistical reliability beyond the reported small cells;

- transfer from KQK to general planning domains.

## Next Falsification Experiment

If the cause of the changed outcome is investigated, the follow-up experiment should freeze the model profile, FENs, defender seeds, move budget, correction policy, and semantic instructions while changing one stand or methodology condition at a time. Fixed and individually calibrated state representations can then be compared without changing the model simultaneously. This causal follow-up is separate from the present pilot, whose purpose is to validate the revised stand and methodology. No historical artifact should be relabelled to satisfy the new verifier.

# Why “Agzamov”

The benchmark is named the Agzamov Test—a double reference to the author’s surname, and to Georgy Agzamov (1954–1986), the first chess grandmaster from Central Asia. Georgy was known as the “nightmare of top grandmasters,” defeating Tal and drawing Karpov through tenacity, pattern recognition, and counterattack rather than raw calculation. The benchmark aspires to a similar reputation among AI models: a nightmare that rewards adaptation over brute force.

# Proposed Broad Specification and Implemented Slice

The broader Chess960/poker matrix remains a proposed v0.2 protocol. Implementations claiming compliance with that proposal must declare the parameters below. Items marked $\dagger$ are configurable within stated bounds; all others are fixed. This table must not be read as evidence that the full matrix has been executed.

| Area | Parameter | Proposed value |
|---|---|---|
| Chess960 | Games per phase | $N \ge 30$ |
| Chess960 | Phase 0 sanity games | 30 |
| Chess960 | Phase 0 pass criteria | >70% wins, <5% protocol failures, $p < 0.05$ |
| Chess960 | Color alternation | Every game |
| Chess960 | Historical adjudication | Material lead $\ge 10$ after ply 40 for 6 plies |
| Poker | Matches per phase | $N \ge 30$ |
| Poker | Blind structure | Escalating tournament schedule |
| Poker | Starting stack | Equal; configurable, default 100 BB |
| General | Sampling | Declared profile; temperature 0 where supported |
| General | Error threshold | 5% invalid moves |
| General | Significance | $\alpha = 0.05$; Bonferroni for H1–H5 |
| General | Bootstrap confidence intervals | 10,000 resamples, 95% |
| General | Memory audit | Pre/post dump and content hash chain |
| A-Score | Sub-scores | 7 |
| A-Score | v0.2 weights | Uniform, $w_i = 1/7$ |
| A-Score | Range | 0–100 |

| **Parameter**            | **Frozen value**                                |
|:-------------------------|:------------------------------------------------|
| Protocol ID              | `kqk-random-legal-defender-v1`                  |
| Matrix                   | 10 ordered KQK FEN/seed rows                    |
| Defender                 | deterministic seeded-random legal               |
| Success                  | actual checkmate only                           |
| Move budget              | 30 accepted model moves                         |
| Gameplay legal-move list | forbidden                                       |
| Correction               | one, without a legal-move oracle                |
| Context                  | fresh provider client and conversation per game |
| Positive control         | Stockfish depth 16 on every selected row        |
| Verification             | offline legal replay, identity and hash checks  |
| Publication coverage     | all 10 rows from index 0, clean source          |

Implemented KQK workbench contract.

#### Versioning.

Paper version, broad benchmark protocol, KQK protocol, and model profile are separate identifiers. A change to the broad specification or the A-Score formula requires a new broad protocol version. A change to the implemented KQK contract requires a new KQK protocol ID. Any model or inference-setting change requires a new profile ID. Results must state all applicable identifiers.

# Roadmap

The KQK finding revises an auxiliary hypothesis; it does not change the Agzamov Test’s main course. The project still aims to build a full stand for measuring how memory, tools, retrieval, and orchestration change model performance under repeatable adversarial conditions. KQK remains the first implemented validation slice.

1.  **Source release**—publish the installable KQK workbench, frozen protocol, verifier, tests, and public verification receipt.

2.  **Native KQK cells**—run clean final-protocol profiles on the paired ten-row matrix, verify offline, and independently audit artifacts.

3.  **Profile ablations**—separate fixed versus calibrated rendering and vary reasoning/output settings one at a time.

4.  **Harder chess protocols**—add optimal-defense and more discriminating endgames only under the same artifact contract.

5.  **Augmentation arms**—compare bare, guarded, memory-equipped, and search-equipped systems without changing the base protocol silently.

6.  **Broad benchmark validation**—validate poker and Chess960 treatments before computing an Agzamov Score or claiming a cross-environment standard.

# Reference Implementation

The source workbench, evaluation scripts, and compact review packet are available at:

**Repository:** <https://github.com/brainops-pub/agzamov-test>

The accepted public surface is the installable `agzamov chess` KQK workbench: protocol/profile discovery, three-board calibration, dry-run, guarded live run, and offline verify/inspect/replay commands. The repository retains earlier Chess960, poker, dashboard, and analysis code as legacy research surfaces; they are not evidence that the broad v0.2 matrix is complete.

The tracked compact review packet contains normalized exploratory logs, calibration records, source hashes, the experiment-time harness snapshot, and an independent-review brief. Raw provider artifacts remain a separate preservation surface and are not required to install or verify the source workbench.

#### Versions.

Paper v0.1.0 is archived on Zenodo (DOI: 10.5281/zenodo.18771523; concept DOI: 10.5281/zenodo.18771522). This document is the paper v0.2.0 draft. The implemented chess protocol is independently versioned as `kqk-random-legal-defender-v1`; package version is `0.1.0`. No v0.2 paper DOI or source tag is claimed until the corresponding release gates complete.

#### License.

Code: MIT. Paper and specification: CC BY 4.0. The protocols are open to independent implementation and criticism.

# Competing Interests

No memory-augmentation result is reported in this v0.2 revision. Any future
claim involving an implementation developed by an author or affiliated company
must disclose that conflict, preregister comparison conditions, include
substitute implementations where feasible, and publish the artifacts needed
for independent verification.

# Acknowledgments

AI systems assisted drafting, consistency checking, and code review during the project. The v0.1 manuscript received model-based reviews from Gemini 2.5 Pro, GLM-4, and GPT-4o. Those reviews are not human peer review, and this v0.2 draft has not yet passed independent human review.

# References

ARC Prize. 2025. *ARC Prize 2025 Results and Analysis*. ARC Prize blog. <https://arcprize.org/blog/arc-prize-2025-results-analysis>.

Aumann, Robert J., and Lloyd S. Shapley. 1994. “Long-Term Competition—a Game-Theoretic Analysis.” In *Essays in Game Theory*. Springer.

Axelrod, Robert. 1981. “The Evolution of Cooperation.” *Science* 211 (4489): 1390–96.

Brown, Noam, and Tuomas Sandholm. 2019. “Superhuman AI for Multiplayer Poker.” *Science* 365 (6456): 885–90.

Chen, Mark, Jerry Tworek, Heewoo Jun, et al. 2021. “Evaluating Large Language Models Trained on Code.” *arXiv Preprint arXiv:2107.03374*.

Chollet, François. 2025. *ARC-AGI-2: A New Benchmark for Artificial General Intelligence*. <https://arcprize.org/blog/arc-agi-2>.

Glickman, Mark E. 1999. “Parameter Estimation in Large Dynamic Paired Comparison Experiments.” *Journal of the Royal Statistical Society: Series C (Applied Statistics)* 48 (3): 377–94.

Greenblatt, Ryan. 2024. *Getting 50% (SoTA) on ARC-AGI with GPT-4o*. Redwood Research blog. <https://blog.redwoodresearch.org/p/getting-50-sota-on-arc-agi-with-gpt>.

Hendrycks, Dan, Collin Burns, Steven Basart, et al. 2021. “Measuring Massive Multitask Language Understanding.” *Proceedings of the International Conference on Learning Representations (ICLR)*.

Liang, Percy, Rishi Bommasani, Tony Lee, et al. 2023. “Holistic Evaluation of Language Models.” *Annals of the New York Academy of Sciences* 1525 (1): 140–46.

Nash, John F. 1950. “Equilibrium Points in n-Person Games.” *Proceedings of the National Academy of Sciences* 36 (1): 48–49.

Neumann, John von, and Oskar Morgenstern. 1944. *Theory of Games and Economic Behavior*. Princeton University Press.

Silver, David, Thomas Hubert, Julian Schrittwieser, et al. 2018. “A General Reinforcement Learning Algorithm That Masters Chess, Shogi, and Go Through Self-Play.” *Science* 362 (6419): 1140–44.

Sorokin, Ivan, and Jean-François Puget. 2025. *NVARC: Test-Time Training for ARC-AGI-2*. <https://github.com/1ytic/NVARC>.
