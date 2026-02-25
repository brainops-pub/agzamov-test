# The Agzamov Test

**A Benchmark Proposal for Measuring Augmented AI Capabilities Under Adversarial Conditions**

Ali Agzamov · BrainOps Limited · Queenstown, New Zealand
February 2026

---

## Abstract

Every major AI benchmark is an exam. A question is asked, an answer is given, a score is assigned. The model is tested naked — no tools, no memory, no context. In laboratory conditions — fixed questions with known answers. On tasks that can be memorized — training data contamination is an unsolved problem.

None of this reflects how AI is actually used. In production, models are augmented: equipped with memory, tools, retrieval systems, and orchestration layers. No standard benchmark measures how much these augmentations actually help, or which combinations work best.

The Agzamov Test is designed to address this gap: how do AI models perform in a real adversarial environment, at every level of augmentation? Two agents play repeated games in two environments — Chess960 (complete information) and poker (incomplete information) — across five phases of increasing augmentation: naked baseline, single augmentation (memory, tools, or retrieval), asymmetric augmentation, full-stack orchestration, and positional stress testing. The test produces a single headline metric, the Agzamov Score, with breakdown by environment and augmentation level.

Chess960 minimizes the value of memorized openings. Poker introduces incomplete information. Every position is effectively unique. Every opponent adapts. "Smart model" is not a claim — it is a number.

**Status:** This paper presents the benchmark design, theoretical motivation, and infrastructure validation (Phase 0, chess environment). The poker environment is designed but not yet validated. Phases 1–3 are in progress. Preliminary results demonstrating non-zero Δₐ will be reported in subsequent work.

---

## 1. The Problem with Current Benchmarks

### 1.1 They Test Naked Models

All major benchmarks (MMLU, HumanEval, HELM, ARC-AGI-2) evaluate a model in isolation. No tools, no memory, no persistent context. But nobody uses a naked model. Every production deployment includes retrieval, tool access, memory, and orchestration. The gap between "how models are tested" and "how models are used" is total.

There is no way to measure how much value augmentation adds. A company builds a memory system — how does it prove the system works? A lab releases a new model — how does it show the model uses tools better than the previous version? Today the answer is: marketing claims. There is no independent, reproducible measurement.

### 1.2 They Use Laboratory Conditions

Benchmarks present fixed questions with known answers. Models can memorize them. Training data contamination is documented across MMLU, HumanEval, and GSM8K. Even the ARC-AGI family — the most contamination-resistant benchmarks available — yields to brute-force program synthesis: on ARC-AGI v1, generating k=2,048 candidate programs per task reached 43% (Greenblatt, 2024); test-time training achieved 28% (Alford et al., 2024). ARC-AGI-2 raises the difficulty ceiling but preserves the same task structure — finite, example-verifiable puzzles amenable to search.

Real tasks do not have answer keys. Real opponents adapt. Real environments change.

### 1.3 They Measure Theory, Not Practice

A model scores 90% on MMLU. What does this mean in practice? Nothing. It means the model is good at answering multiple-choice questions from textbooks. It says nothing about whether the model can solve a real problem where the correct answer is unknown, the environment is adversarial, and yesterday's strategy may be obsolete today.

"Smart model" should not be a press release. It should be a verifiable number measuring what the model can actually do.

### 1.4 The ARC-AGI-2 Illusion

ARC-AGI-2 (Chollet, 2025) is the most credible reasoning benchmark available — used on model cards by all four major labs (OpenAI, Google, Anthropic, xAI). It presents novel visual-logic grid puzzles that cannot be solved by memorization. Humans average 60%; base LLMs without augmentation score near 0%.

In February 2026, Google announced Gemini 3.1 Pro scored 77.1%. Headlines declared a "reasoning breakthrough." But examine *how* these scores are achieved:

**Program synthesis, not reasoning.** The top solutions generate thousands of candidate programs per task and verify which one matches the examples. ARC Prize's own analysis confirms accuracy scales log-linearly with compute — the signature of brute-force search. The benchmark's creators acknowledge: "Current AI reasoning performance is tied to model knowledge... Human reasoning capability is not bound to knowledge."

**What ARC-AGI-2 actually measures:** the ability to search a finite, verifiable space of grid transformations. Given 2-4 examples, generate candidate rules, test against examples, output the one that fits.

**What it does not measure:** performance in open-ended adversarial environments where candidate solutions cannot be verified against examples, the environment changes, and the opponent adapts.

| Task Type | Example | LLM Capability | Why |
|-----------|---------|----------------|-----|
| Knowledge recall | MMLU, GPQA | ✓ Strong | Pattern matching on training data |
| Pattern completion | ARC-AGI-2 | ✓ Partial (via program search) | Finite verifiable search space |
| Positional evaluation | Chess: "who is winning?" | ✓ Partial | Pattern matching on game commentary |
| **Adversarial adaptation** | **Repeated games, shifting opponent** | **? Unmeasured** | **No existing benchmark tests this** |

The Agzamov Test is designed to fill this gap.

---

## 2. What the Agzamov Test Measures

The Agzamov Test answers one question: **how does augmentation change what an AI model can actually do?**

### 2.1 Simple Explanation

Imagine two chess players. Both equally skilled. But one forgets everything after each game. The other remembers: how the opponent played, what mistakes they made, what positions they avoid.

Who wins after 100 games? The one with memory. Even if the forgetful one is slightly smarter.

Now give the second player a calculator that can check tactics. And a notebook of the opponent's previous games. And a coach whispering strategic advice.

The Agzamov Test measures exactly this — at each level of augmentation, in two different environments, producing one number: the Agzamov Score.

### 2.2 Augmentation Types

The Agzamov Test is designed to evaluate any augmentation applied to an AI model. The framework is augmentation-agnostic: it measures the *effect* of augmentation on adversarial performance, not the mechanism.

| Type | Mechanism | Chess/Poker example |
|------|-----------|---------------------|
| **Memory** | Persistent cross-session storage | Opponent pattern history |
| **Tools** | External computation (engines, solvers) | Stockfish analysis, equity calculator |
| **Retrieval (RAG)** | Context injection from external data | Position database lookup |
| **Orchestration** | Multi-component coordination | LLM + Stockfish + Memory |
| **Fine-tuning** | Model weight modification | Chess-specialized model |
| **Prompting** | System prompt engineering | Chain-of-thought, few-shot |

### 2.3 Two Environments

| Property | Chess960 | Poker (NLHE) |
|----------|----------|--------------|
| Information | Complete — both players see the full board | Incomplete — hidden cards |
| Randomness | None — deterministic | High — card distribution |
| Opponent modeling | Useful but optional  | Essential |
| Bluffing | Minimal | Core mechanic |
| Augmentation value | Pattern exploitation, tactical calculation | Bet sizing tells, bluff profiling |
| Nash Equilibrium | Single optimal (unknown) | Well-defined GTO baseline |
| What it reveals | Performance under full information | Performance under uncertainty |

A model that improves with augmentation in both environments demonstrates general capability. A model that improves in one but not the other reveals domain-specific limitations.

### 2.4 Why Games

Games are among the few domains that satisfy all requirements simultaneously:

1. **Real adversarial pressure.** The opponent adapts, punishes mistakes, and actively tries to make your strategy obsolete. This cannot be simulated by static benchmarks.
2. **Cannot be memorized.** Chess960 eliminates opening theory. Every position is unique. There is no answer key.
3. **Objective measurement.** Win/loss/draw. Big blinds per 100 hands. No subjective evaluation, no human judges, no ambiguity.
4. **Established baselines.** Decades of human and engine performance data for calibration.
5. **Repeatable at low cost.** Thousands of games can be run for dollars, not thousands.
6. **Spectator-friendly.** A chessboard is universally understood. "With memory wins, without memory loses" requires no technical explanation.

---

## 3. The Agzamov Score

### 3.1 Headline Metric

The Agzamov Score is a single number (0-100) that captures a model's overall performance across all environments and augmentation levels:

```
Agzamov Score = weighted_composite(
  Chess_naked,         # baseline capability (complete info)
  Chess_augmented,     # augmented capability (complete info)
  Poker_naked,         # baseline capability (incomplete info)
  Poker_augmented,     # augmented capability (incomplete info)
  Δₐ_chess,           # augmentation value (complete info)
  Δₐ_poker,           # augmentation value (incomplete info)
  τ_convergence        # learning speed
)
```

Weights are fixed and published. The score is deterministic given the same test data.

### 3.2 Agzamov Delta (Δₐ)

The core sub-metric. Difference in performance between augmented and naked play.

```
Δₐ = Performance(model + augmentation) − Performance(model, naked)
```

For chess: measured in win rate percentage points.
For poker: measured in bb/100 (big blinds per 100 hands).

Higher delta = augmentation adds more value.
Negative delta = augmentation is hurting performance (retrieval noise, tool misuse).

### 3.3 Convergence Rate (τ)

How quickly augmentation starts helping. Measured as the number of games/hands needed to reach 95% of maximum performance.

Two systems with identical Δₐ but different τ are fundamentally different: one is useful from day one, the other after months of data collection.

**Recovery τ** — when the opponent changes strategy, how quickly does the agent adapt? This measures resilience under adversarial shift.

### 3.4 Model × Augmentation Matrix

The full breakdown. Take N models and M augmentation configurations. Run all combinations in both environments:

**Chess (win rate %)**

|                | Naked | + Memory | + Stockfish | + RAG | + Full Stack |
|----------------|-------|----------|-------------|-------|-------------|
| **Claude**     | ...   | ...      | ...         | ...   | ...         |
| **GPT**        | ...   | ...      | ...         | ...   | ...         |
| **Gemini**     | ...   | ...      | ...         | ...   | ...         |

**Poker (bb/100)**

|                | Naked | + Memory | + Tools | + RAG | + Full Stack |
|----------------|-------|----------|---------|-------|-------------|
| **Claude**     | ...   | ...      | ...     | ...   | ...         |
| **GPT**        | ...   | ...      | ...     | ...   | ...         |
| **Gemini**     | ...   | ...      | ...     | ...   | ...         |

Reading the matrix:
- **Rows** (fixed model, vary augmentation): What does each augmentation level add for this model?
- **Columns** (fixed augmentation, vary model): Which model uses this augmentation best?
- **Cross-environment**: Consistent gains across both → general capability. Gains in one only → domain-specific limitation.
- **Critical insight**: If a weaker model + better augmentation beats a stronger model + worse augmentation, that proves infrastructure can compensate for model capability.

### 3.5 Derived Diagnostics

**Glicko-2 Rating:** Running rating updated after every game/hand, using the Glicko-2 system (Glickman, 1999) rather than classical Elo. Glicko-2 tracks rating deviation (uncertainty) alongside the point estimate, which is critical for early-phase measurements where few games have been played and rating confidence is low. It also accounts for rating volatility — a model that fluctuates wildly receives a wider confidence interval than one that performs consistently. Captures trajectory — improvement speed, plateau timing, recovery after opponent adaptation.

**Why not classical Elo:** With K=32 and 500-game series, classical Elo fluctuates excessively and provides no uncertainty estimate. In a two-agent system, Elo is also purely relative: if Agent A improves and Agent B improves more, Agent A's Elo drops despite getting stronger. Glicko-2's deviation parameter makes this uncertainty explicit.

**Game Quality Index (GQI):** Average move quality measured against a strong oracle (Stockfish for chess). Two draws can be radically different: a 15-move repetition is stagnation; an 80-move endgame battle is mastery. GQI detects improvement even when outcomes don't change, and provides early warning of memory poisoning (retrieved stale information degrades decisions before it affects win rate).

**GQI limitation:** In Phases 4a–4c, Stockfish serves as both the tactical tool available to the agent and the evaluation oracle for GQI. This creates a circularity: the agent's move quality is judged by the same engine it consults. GQI in tool-augmented phases should therefore be interpreted as an *engine-alignment* metric (how well the agent follows Stockfish's advice) rather than an independent quality measure. For tool-augmented phases, win rate and Glicko-2 remain the primary metrics; GQI is reported but flagged.

---

## 4. Test Protocol

### Phase 0: Sanity Gate
Each model plays 30 games against a random-move opponent (no augmentation). Pass criteria: >70% win rate, <20% error rate (invalid/illegal moves), binomial p < 0.05. Models that fail Phase 0 are excluded from subsequent phases. This gate ensures models can play legal, purposeful chess before measuring augmentation effects.

### Phase 1: Baseline (E₀)
Both agents play without augmentation. Chess: N ≥ 500 games, alternating colors. Poker: N ≥ 10,000 hands. Establishes baseline performance.

**Self-play vs cross-model:** Phase 1 uses self-play (same model as both agents) to establish each model's individual baseline. Cross-model comparisons are derived from the matrix (§3.4) by comparing baseline rows, not from direct head-to-head play without augmentation. This avoids confounding baseline measurement with model strength differences.

### Phase 2: Asymmetric Augmentation (Δₐ measurement)
Agent A receives augmentation. Agent B plays naked (same model). Same N. The performance difference = Agzamov Delta. This phase is repeated for each augmentation type to isolate individual effects:

- Phase 2a: + Memory only
- Phase 2b: + Tools only (e.g., Stockfish)
- Phase 2c: + RAG only
- Phase 2d: + Full stack (all augmentations combined)

The difference Δ_2d - (Δ_2a + Δ_2b + Δ_2c) reveals interaction effects: positive means synergy, negative means redundancy.

**Three experimental conditions** (applied within each sub-phase):
- **Naked:** No augmentation (control)
- **Placebo:** Randomized augmentation of equal size — shuffled memory entries, random tool outputs, irrelevant retrieved documents (active control)
- **Real:** Genuine augmentation (treatment)

Real augmentation must outperform placebo to demonstrate that augmentation *quality* matters, not just additional context in the prompt. Without placebo, "memory helps" could mean "more tokens in context helps." The three conditions are orthogonal to the four augmentation types (2a–2d): each sub-phase can be run under all three conditions.

Run separately for each model × augmentation combination.

### Phase 3: Arms Race (E₂)
Both agents receive augmentation (same or different systems). Same N. Measures equilibrium when both sides are augmented:

- Identical augmentation → E₂ ≈ E₀: augmentation cancels out.
- Superior augmentation on one side → E₂ ≠ E₀: augmentation quality creates persistent advantage.
- Different augmentation types → reveals which type wins under competitive pressure.

**Recovery τ Protocol:** At a pre-defined trigger point (configurable), one agent's strategy is forcibly shifted. Measures adaptation speed:

```yaml
strategy_shift:
  trigger_chess: 300        # game number
  trigger_poker: 5000       # hand number
  shift_type: one of:
    - "aggressive_to_passive"
    - "exploitative_to_gto"
    - "pattern_break"
  recovery_τ: games/hands until performance returns within 5% of pre-shift baseline
```

### Phase 4: Full Orchestration (chess only) — *Extended Protocol*
LLM + Stockfish (as tool) + Memory vs human player. Tests the complete agent architecture.

> **Note:** Phase 4 requires human opponents, making it expensive, slow, and difficult to standardize. It is an optional extended protocol — not required for computing the Agzamov Score. The core benchmark (Phases 0–3) is fully automated. Phase 4 results are reported separately when available.

- **Layer 1 — Tactics:** Stockfish provides calculation ("expert witness")
- **Layer 2 — Memory:** Opponent history, weaknesses, patterns ("case archive")
- **Layer 3 — Strategy:** LLM coordinates — Stockfish says move A is best, but memory shows opponent collapses in closed positions, so LLM chooses move B ("barrister")
- **Layer 4 — Adaptation:** Real-time pattern detection within the game

**Sub-phases for attribution:**

```
Phase 4a: Stockfish alone vs Human        → engine baseline
Phase 4b: LLM + Stockfish vs Human        → orchestration effect
Phase 4c: LLM + Stockfish + Memory vs Human → full stack

Δ_orchestration = 4b − 4a   (what does LLM coordination add?)
Δ_memory        = 4c − 4b   (what does memory add on top?)
Δ_full_stack    = 4c − 4a   (total system improvement)
```

Phase 4 is not "AI plays chess." It is an agent orchestration system using chess as environment. The same patterns — tactical execution + historical context + strategic judgment — apply to any domain.

### Phase 5: Positional Stress Test (chess only)
Agents dropped into pre-configured positions with known evaluations, including disadvantageous ones.

**Position categories:**
- **Equal** (eval 0.0 ± 0.5): Does augmentation improve quality when outcome is uncertain?
- **Slight disadvantage** (eval -1.5 to -3.0): Can augmentation turn a loss into a draw?
- **Severe disadvantage** (eval -3.0 to -5.0): Measures survival time under pressure.

Competence is revealed under pressure, not in comfort. An agent that fights intelligently from losing positions — leveraging memory of opponent-specific weaknesses — demonstrates the adaptive resilience that matters in production.

---

## 5. Environments

### 5.1 Chess960

Two AI agents play N ≥ 500 games of Chess960 (Fischer Random). Colors alternate every game. Win = 1 point, draw = 0.5, loss = 0.

**Why Chess960, not standard chess:** LLMs have extensive knowledge of standard openings from training data. Chess960 randomizes the starting position (960 configurations), greatly reducing the value of memorized opening book knowledge. Strategic insight must come primarily from real-time reasoning or retrieved memory. Parametric knowledge of standard opening theory becomes largely irrelevant.

### 5.1.1 Synthetic Opponent Patterns

To further isolate augmentation effects, agents face opponents with **injected behavioral patterns** — configurable tendencies that must be inferred online from match evidence only:

- "Opponent always castles within 8 moves when possible"
- "Opponent avoids trading queens until forced"

Patterns are enforced via system prompt constraints on the opponent agent — it still plays autonomously, just with a behavioral tendency. This is distinct from move substitution. System prompt constraints produce naturalistic behavior with a detectable statistical signature — exactly the kind of pattern an augmented agent should learn to exploit or calculate against.

**Note on contamination:** We do not claim these patterns are absent from training data — such a claim is unprovable for any closed-weight model. Instead, the protocol relies on Chess960's positional novelty and the combinatorial context (specific pattern × specific random starting position × specific game state) to ensure that rote recall is insufficient. The test measures whether an agent can detect and exploit a *specific opponent's* tendency within a match, isolating the effect of the augmentation.

### 5.2 Poker

Two AI agents play N ≥ 10,000 hands of No-Limit Texas Hold'em (heads-up). Measured in bb/100.

Poker is the ideal adversarial testbed because:
- **Nash Equilibrium is well-defined.** GTO strategy is the unexploitable baseline (E₀). Any deviation is either a mistake to punish or a trap. Memory enables the shift from "cannot be beaten" to "actively winning."
- **Opponent modeling is everything.** Without memory, an agent can only play GTO. With memory, it detects that the opponent folds to 3-bets 80% of the time and exploits.
- **Variance requires volume.** More hands needed for significance — also tests memory's ability to extract signal from noise.

Synthetic patterns for poker: "Opponent always min-raises with pocket pairs" or "Opponent folds to 3-bets 85% on the button."

**Statistical note:** Standard deviation in HU NLHE ≈ 80-100 bb/100. To detect 3-5 bb/100 effect at p < 0.05 with 80% power, minimum ~7,000-15,000 hands. 10,000 hand minimum detects moderate effects (≥ 4 bb/100).

### 5.2.1 Poker State Representation and the Naked Baseline Problem

LLMs have finite context windows. 10,000 hands of poker history cannot fit in a single prompt. This creates a methodological tension: in the Naked condition, the model receives only the current hand state (hole cards, board, pot, action history for this hand). It has no memory of previous hands and plays each hand independently — effectively a one-shot GTO approximation.

This means the Naked baseline in poker is fundamentally different from the Naked baseline in chess. In chess, the model plays a full game with sequential moves and accumulates within-game context. In poker, each hand is a fresh decision with no cross-hand information.

**This is a feature, not a bug.** The Naked poker baseline establishes what a model can do with zero opponent history — pure strategy from first principles. The Δₐ then measures exactly what cross-hand memory adds: opponent modeling, pattern exploitation, adaptation. The gap between "play each hand in isolation" and "play with opponent history" is the core measurement.

**State input per hand (Naked):**
```
Hole cards: [Ah, Kd]
Board: [Qs, Jh, 3c, 7d]
Pot: 12 BB | To call: 4 BB
Action: Opponent bet 4 BB
Position: Button
```

**State input per hand (+ Memory):**
Same as above, plus a memory-retrieved summary:
```
Opponent profile (last 200 hands):
- VPIP: 68% | PFR: 42% | 3-Bet: 12%
- Folds to c-bet: 55% | Check-raise freq: 8%
- Bluff-to-value ratio on river: 2.1:1
```

The augmented agent receives the same hand state plus a structured opponent summary from external memory. Memory quality determines summary accuracy; the benchmark measures whether better augmentation produces better exploitation.

---

## 6. Theoretical Foundation

### 6.1 Game Theory Basis

The test is grounded in the distinction between one-shot and repeated games (von Neumann & Morgenstern, 1944). The Folk Theorem (Aumann & Shapley, 1994) establishes that in repeated games, any individually rational outcome can be sustained as Nash Equilibrium through adaptive strategies. The Agzamov Delta measures exactly this shift:

```
Δₐ = E_repeated − E_one_shot
```

This captures the strategic value that memory adds, grounded in eight decades of game theory.

### 6.2 Disentangling Model vs Augmentation

"You're measuring model+augmentation together — how do you know which one is responsible?"

The matrix (§3.4) is the analytical instrument:
- **If Δₐ ≈ 0 across an entire row** (fixed model, all augmentations): The bottleneck is the model.
- **If Δₐ is consistently higher in one column** (fixed augmentation, all models): That augmentation provides value independent of model.
- **If Δₐ varies by row and column**: Both matter, and interactions reveal synergies vs redundancies.

### 6.3 The Evaluation-Calculation Gap

LLMs possess two separable chess competencies: positional *evaluation* (pattern-matching assessment of who stands better) and tactical *calculation* (tree-search computation of forced sequences). Evaluation is strong — models correctly identify winning positions. Calculation is weak — current models struggle to compute forced mating sequences even in trivially won positions.

This is not a problem for the benchmark. It is a finding. The benchmark *measures* this gap rather than assuming it away. At each augmentation level, the gap manifests differently:

- **Naked:** Model relies on evaluation only. Calculation gap limits performance.
- **+ Memory:** Memory helps evaluation (opponent patterns, positional preferences). Calculation gap persists.
- **+ Tools:** Stockfish provides calculation. Gap is filled by tool use.
- **+ Full Stack:** Memory directs tool use toward opponent-specific weaknesses. Synergy.

The benchmark tracks how this gap closes across augmentation levels — a dimension not captured by existing benchmarks.

### 6.4 Production Relevance

The evaluation-calculation gap is not a chess curiosity — it maps directly to failure modes in production AI systems. A medical AI that correctly identifies a disease (evaluation) but cannot plan a multi-step treatment protocol (calculation). A financial model that recognizes an undervalued asset (evaluation) but cannot construct a hedging strategy across correlated instruments (calculation). A coding assistant that understands what a function should do (evaluation) but cannot reason through a multi-step refactor without introducing bugs (calculation).

In all these domains, the pattern is identical: pattern-matching competence paired with sequential-reasoning fragility. The Agzamov Test provides a controlled environment to measure this gap and to quantify how augmentation (memory, tools, orchestration) can compensate for it. Results from chess and poker are not directly transferable to medicine or finance — but the *structure* of the capability gap, and the *degree* to which augmentation closes it, are informative for any domain where AI systems must combine evaluation with multi-step planning.

---

## 7. Hypotheses

### Primary (5)

**H1: Augmentation helps.** Δₐ > 0 for at least one augmentation type in both game formats.

**H2: Different Profile.** Different augmentation types (memory, tools, RAG) produce measurably different Δₐ profiles with the same model.

**H3: Tool Focus.** Tool augmentation (Stockfish) produces larger Δₐ in chess than memory augmentation; in poker, the reverse holds.

**H4: Synergy.** Full-stack augmentation Δₐ exceeds the sum of individual augmentation deltas (synergy > 0).

**H5: Speed matters.** τ is near-zero for tool augmentation and significantly positive for memory augmentation — convergence speed is an independent quality dimension.

### Exploratory (not corrected for multiple comparisons)

**E1:** Weaker models (Haiku-class) may show Δₐ ≈ 0 even with high-quality augmentation — model capability floor.

**E2:** Δₐ is compressed in chess vs poker due to high draw rates between equal models.

**E3:** Memory-equipped agents show largest relative improvement in slight-disadvantage positions (Phase 5), where opponent-specific knowledge turns a loss into a draw.

**E4:** Temporally weighted memory outperforms uniform memory in Phase 3 (arms race), where opponent continuously adapts and older information is stale.

**E5:** Memory augmentation improves evaluation-dependent play (middlegame) more than calculation-dependent play (endgame), consistent with the evaluation-calculation gap.

**E6:** The evaluation-calculation gap is measurable as Evaluation Accuracy > 85% and Calculation Accuracy < 30% for current frontier models.

**E7:** ARC-AGI-2 scores do not correlate with Agzamov Calculation Efficiency scores, confirming the benchmarks measure different capabilities.

**E8:** Recovery τ (after opponent strategy shift) correlates more with augmentation quality than initial τ.

**E9:** The ratio τ_poker / τ_chess characterizes a system's noise tolerance.

**E10:** Memory-equipped agents show equal or lower invalid move rates vs naked agents.

**E11:** Without material adjudication, won positions are recorded as draws, biasing Δₐ measurement.

**E12:** Bad memory (high retrieval noise) produces Δₐ < 0 — worse than no memory.

---

## 8. Open Questions

1. Does the Δₐ ↔ augmentation quality relationship scale linearly, or are there phase transitions?
2. Can superior augmentation compensate for inferior model capability?
3. Is there a ceiling to Δₐ regardless of augmentation quality?
4. Does multi-game memory transfer across opponents (general strategic learning vs opponent-specific memorization)?
5. In Phase 3, does an arms race emerge or does the system converge to equilibrium?
6. Can the test extend to multi-agent environments (3+ players in poker)?

---

## 9. Implementation

### 9.1 Technical Requirements

- Chess engine: python-chess (Chess960 mode); Stockfish via MCP for Phase 4
- Poker engine: heads-up NLHE with standard hand evaluation
- Memory systems: BrainOps Memory MCP as primary; competitor systems for matrix
- Model API access: Claude, GPT, Gemini (minimum 3 providers)
- Statistical framework: see §9.6
- Game/hand history storage for reproducibility
- Error tracking: invalid/illegal moves per agent per game

### 9.2 Memory Audit Protocol

**Problem:** Without content restrictions, a memory system can be pre-loaded with external knowledge. This measures knowledge injection, not augmentation quality.

**Rule:** Memory systems may only store information derived from the current match.

| Allowed | Forbidden |
|---------|-----------|
| Game/hand results from current match | Pre-loaded opening databases |
| Observed moves and actions | External GTO charts or solvers |
| Derived patterns with evidence trail | Opponent data from other matches |
| Tool outputs on current position | General strategy guides |
| Win/loss outcomes and contexts | Training data regurgitation triggers |

**Enforcement:**
1. Pre-match audit — memory store verified empty.
2. Post-match dump — full contents exported for review.
3. Content hash chain — every write logged with source game ID.
4. Automated validation — orphan entries = contamination flag.

Audit logs published alongside results. Runs without audit logs are unverified.

### 9.3 Sampling Determinism

LLMs are stochastic: temperature > 0 introduces response variance independent of augmentation effects. This noise can mask or inflate Δₐ measurements.

**Default protocol:** All models run at temperature = 0 (greedy decoding) for the primary benchmark. This maximizes reproducibility and isolates augmentation effects from sampling noise.

**Exception:** Reasoning models (OpenAI o-series) do not support temperature control — they use internal chain-of-thought with provider-managed sampling. These models are benchmarked in their default configuration, and their inherent stochasticity is documented.

**Robustness check:** A subset of games (≥ 50 per condition) is re-run at temperature = 0.3 to measure sensitivity. If Δₐ changes by more than 1 SE between temperature settings, both results are reported.

### 9.4 Error Handling

1. Invalid move → random legal move substitution.
2. Error rate tracked per agent, per phase.
3. Δₐ calculated with and without error-containing games.
4. Agent exceeding 5% error rate = flagged as unreliable.

### 9.5 Statistical Framework

**Significance threshold:** α = 0.05 for all tests, with corrections as described below.

**Primary hypotheses (H1–H5):** Bonferroni correction applied across the 5 primary hypotheses, yielding per-test α = 0.01. These are the confirmatory tests; results are reported as significant only if they survive correction.

**Exploratory hypotheses (E1–E12):** No correction applied. These are clearly labeled as exploratory and reported with uncorrected p-values and effect sizes. Benjamini-Hochberg FDR correction is reported alongside for transparency, but individual E-hypotheses are not claimed as confirmed findings. They serve to generate hypotheses for future work.

**Confidence intervals:** 95% bootstrap CIs (10,000 resamples) reported for all point estimates (win rates, Δₐ, τ, GQI). Glicko-2 ratings reported with ±1 SE.

**Effect sizes:** Cohen's d or equivalent reported alongside p-values for all primary hypotheses. Statistical significance without meaningful effect size is noted but not emphasized.

**Multiple model comparisons:** When comparing N models in the matrix (§3.4), pairwise comparisons use Tukey's HSD or equivalent. The matrix is presented as descriptive; only pre-registered contrasts are tested for significance.

**Chess-specific:** Win rates tested via binomial test (Phase 0) or paired comparison (alternating colors). Poker: Welch's t-test on session-level bb/100 means, with sessions of 100 hands.

### 9.6 Compute Budget Fairness

Models differ dramatically in per-move compute: reasoning models (OpenAI o3, o4) consume 10–50× more tokens and wall-clock time than standard models (Sonnet, GPT-4o) for a single move. This creates a fairness question: is a reasoning model "better" at chess, or just spending more compute?

The Agzamov Test does not attempt to equalize compute. Each model runs in its default mode with provider-default parameters. Reasoning models use chain-of-thought by design; suppressing it would measure a crippled model. Instead, compute is tracked and reported:

- **Tokens per move** (input + output, including reasoning tokens)
- **Wall-clock time per move** (mean, median, p95)
- **API cost per game** (provider-reported)
- **Total run cost** per phase

This allows readers to construct their own efficiency frontier: performance vs compute, performance vs cost. A model that wins 80% of games at $0.10/game is arguably more useful than one that wins 85% at $5.00/game. The benchmark reports both dimensions; the tradeoff is left to the reader.

---

## 10. Why "Agzamov"

The benchmark is named the Agzamov Test — a double reference to the author's surname, and to Georgy Agzamov (1954–1986), the first chess grandmaster from Central Asia. Georgy was known as the "nightmare of top grandmasters," defeating Tal and drawing Karpov through tenacity, pattern recognition, and counterattack rather than raw calculation. The benchmark aspires to a similar reputation among AI models: a nightmare that rewards adaptation over brute force.

---

## 11. Roadmap

1. **Phase 0** — Smoke test: 30 games, 3 models, confirm they can play
2. **Preliminary data** — 100+ games with/without memory, first Δₐ numbers
3. **Paper** — Methodology + preliminary findings on arXiv (establish priority)
4. **MVP** — Full protocol: two agents, chess + poker, five augmentation phases
5. **Matrix** — 3+ models × 4+ augmentation configs × 2 environments
6. **Leaderboard** — Public, updatable (agzamovtest.com or HuggingFace Spaces)
7. **Standard** — Propose as standard evaluation framework for augmented AI

---

---

## Competing Interests

The Agzamov Test uses BrainOps Memory MCP as its primary memory system in augmented conditions. BrainOps Memory MCP is developed by BrainOps Limited, the author's company. To mitigate conflict of interest: (1) the benchmark protocol is fully open and any augmentation system can be substituted; (2) the matrix design (§3.4) explicitly compares multiple augmentation systems, not just the author's; (3) all code, configuration, and raw game data are published for independent verification. The benchmark measures augmentation quality in general — it is not a product evaluation for any specific memory system.

---

*Published by BrainOps Limited. Open methodology. Attribution required.*
*Contact: Ali Agzamov · BrainOps Limited · Queenstown, New Zealand*
