# The Agzamov Test

**A Benchmark for Measuring Real AI Capabilities Under Adversarial Conditions**

Ali Agzamov · BrainOps Limited · Queenstown, New Zealand
February 2026

---

## Abstract

Every major AI benchmark is an exam. A question is asked, an answer is given, a score is assigned. The model is tested naked — no tools, no memory, no context. In laboratory conditions — fixed questions with known answers. On tasks that can be memorized — training data contamination is an unsolved problem.

None of this reflects how AI is actually used. In production, models operate with tools, memory, and orchestration layers. They face tasks that cannot be memorized. They work against adversaries who adapt.

The Agzamov Test measures what no existing benchmark measures: how AI models perform in a real adversarial environment, at every level of augmentation. Two agents play repeated games in two environments — Chess960 (complete information) and poker (incomplete information) — across four augmentation levels: naked, with memory, with tools, and with full orchestration. The test produces a single headline metric, the Agzamov Score, with breakdown by environment and augmentation level.

Chess960 eliminates memorized openings. Poker eliminates complete information. Every position is unique. Every opponent adapts. "Smart model" is not a claim — it is a number.

---

## 1. The Problem with Current Benchmarks

### 1.1 They Test Naked Models

All major benchmarks (MMLU, HumanEval, HELM, ARC-AGI-2) evaluate a model in isolation. No tools, no memory, no persistent context. But nobody uses a naked model. Every production deployment includes retrieval, tool access, memory, and orchestration. The gap between "how models are tested" and "how models are used" is total.

There is no way to measure how much value augmentation adds. A company builds a memory system — how does it prove the system works? A lab releases a new model — how does it show the model uses tools better than the previous version? Today the answer is: marketing claims. There is no independent, reproducible measurement.

### 1.2 They Use Laboratory Conditions

Benchmarks present fixed questions with known answers. Models can memorize them. Training data contamination is documented across MMLU, HumanEval, and GSM8K. Even ARC-AGI-2 — the most contamination-resistant benchmark available — is solved by brute-force program synthesis: generating thousands of candidate programs per task and testing which one matches the examples (Greenblatt, 2024: k=2,048 programs at 43%; NVARC: test-time training at 28%). This is search in program space, not reasoning.

Real tasks do not have answer keys. Real opponents adapt. Real environments change.

### 1.3 They Measure Theory, Not Practice

A model scores 90% on MMLU. What does this mean in practice? Nothing. It means the model is good at answering multiple-choice questions from textbooks. It says nothing about whether the model can solve a real problem where the correct answer is unknown, the environment is adversarial, and yesterday's strategy may be obsolete today.

"Smart model" should not be a press release. It should be a verifiable number measuring what the model can actually do.

### 1.4 The ARC-AGI-2 Illusion

ARC-AGI-2 (Chollet, 2025) is the most credible reasoning benchmark available — used on model cards by all four major labs (OpenAI, Google, Anthropic, xAI). It presents novel visual-logic grid puzzles that cannot be solved by memorization. Humans average 60%; pure LLMs score 0%.

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

The Agzamov Test fills the empty quadrant.

---

## 2. What the Agzamov Test Measures

The Agzamov Test answers one question: **how does augmentation change what an AI model can actually do?**

### 2.1 Simple Explanation

Imagine two chess players. Both equally skilled. But one forgets everything after each game. The other remembers: how the opponent played, what mistakes they made, what positions they avoid.

Who wins after 100 games? The one with memory. Even if the forgetful one is slightly smarter.

Now give the second player a calculator that can check tactics. And a notebook of the opponent's previous games. And a coach whispering strategic advice.

The Agzamov Test measures exactly this — at each level of augmentation, in two different environments, producing one number: the Agzamov Score.

### 2.2 Four Augmentation Levels

| Level | What the model has | What it tests |
|-------|-------------------|---------------|
| **Naked** | Nothing. Raw model. | Baseline capability |
| **+ Memory** | Persistent memory across games | Adaptation, opponent modeling |
| **+ Tools** | External calculation tools (e.g., Stockfish for chess) | Tool use effectiveness |
| **+ Full Stack** | Memory + Tools + Orchestration | Complete agent architecture |

### 2.3 Two Environments

| Property | Chess960 | Poker (NLHE) |
|----------|----------|--------------|
| Information | Complete — both players see the full board | Incomplete — hidden cards |
| Randomness | None — deterministic | High — card distribution |
| Contamination risk | Eliminated — 960 random starting positions | Low — no memorizable "answers" |
| Memory value | Pattern exploitation, positional preferences | Bet sizing tells, bluff profiling |
| Tool value | Tactical calculation (Stockfish) | GTO solvers |
| What it reveals | Performance under full information | Performance under uncertainty |

A model that improves with augmentation in both environments demonstrates general capability. A model that improves in one but not the other reveals domain-specific limitations.

### 2.4 Why Games

Games are the only domain that satisfies all requirements simultaneously:

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

|                | Naked | + Memory | + Tools | + Full Stack |
|----------------|-------|----------|---------|-------------|
| **Claude**     | ...   | ...      | ...     | ...         |
| **GPT**        | ...   | ...      | ...     | ...         |
| **Gemini**     | ...   | ...      | ...     | ...         |

**Poker (bb/100)**

|                | Naked | + Memory | + Tools | + Full Stack |
|----------------|-------|----------|---------|-------------|
| **Claude**     | ...   | ...      | ...     | ...         |
| **GPT**        | ...   | ...      | ...     | ...         |
| **Gemini**     | ...   | ...      | ...     | ...         |

Reading the matrix:
- **Rows** (fixed model, vary augmentation): What does each augmentation level add for this model?
- **Columns** (fixed augmentation, vary model): Which model uses this augmentation best?
- **Cross-environment**: Consistent gains across both → general capability. Gains in one only → domain-specific limitation.
- **Critical insight**: If a weaker model + better augmentation beats a stronger model + worse augmentation, that proves infrastructure can compensate for model capability.

### 3.5 Derived Diagnostics

**Elo Rating:** Running rating updated after every game/hand. Captures trajectory — improvement speed, plateau timing, recovery after opponent adaptation.

**Game Quality Index (GQI):** Average move quality measured against a strong oracle (Stockfish for chess). Two draws can be radically different: a 15-move repetition is stagnation; an 80-move endgame battle is mastery. GQI detects improvement even when outcomes don't change, and provides early warning of memory poisoning (retrieved stale information degrades decisions before it affects win rate).

---

## 4. Test Protocol

### Phase 1: Baseline (E₀)
Both agents play without augmentation. Chess: N ≥ 500 games, alternating colors. Poker: N ≥ 10,000 hands. Establishes baseline performance.

### Phase 2: Asymmetric (Δₐ measurement)
Agent A receives augmentation. Agent B plays naked (same model). Same N. The performance difference = Agzamov Delta.

**Three conditions** for isolating causation:
- **Naked:** No augmentation (control)
- **Placebo:** Random/fake memory of equal size — memory from a different match, or shuffled entries (active control)
- **Real:** Genuine augmentation (treatment)

Real memory must outperform placebo to demonstrate that memory *quality* matters, not just additional context in the prompt. Without placebo, "memory helps" could mean "more tokens in context helps."

Run separately for each model × augmentation combination.

### Phase 3: Arms Race (E₂)
Both agents receive augmentation (same or different systems). Same N. Measures equilibrium when both sides adapt.

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

### Phase 4: Full Orchestration (chess only)
LLM + Stockfish (as tool) + Memory vs human player. Tests the complete agent architecture:

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

**Why Chess960, not standard chess:** LLMs have extensive knowledge of standard openings from training data. Chess960 randomizes the starting position (960 configurations), eliminating opening book knowledge entirely. Every strategic insight must come from real-time reasoning or retrieved memory. Parametric knowledge becomes useless for opening preparation.

### 5.1.1 Synthetic Opponent Patterns

To further isolate augmentation effects, agents face opponents with **injected behavioral patterns** that do not exist in any training data:

- "Opponent always castles within 8 moves when possible"
- "Opponent avoids trading queens until forced"

Patterns are enforced via system prompt constraints on the opponent agent — it still plays autonomously, just with a behavioral tendency. This is distinct from move substitution. System prompt constraints produce naturalistic behavior with a detectable statistical signature — exactly the kind of pattern a memory-equipped agent should learn to exploit.

### 5.2 Poker

Two AI agents play N ≥ 10,000 hands of No-Limit Texas Hold'em (heads-up). Measured in bb/100.

Poker is the ideal adversarial testbed because:
- **Nash Equilibrium is well-defined.** GTO strategy is the unexploitable baseline (E₀). Any deviation is either a mistake to punish or a trap. Memory enables the shift from "cannot be beaten" to "actively winning."
- **Opponent modeling is everything.** Without memory, an agent can only play GTO. With memory, it detects that the opponent folds to 3-bets 80% of the time and exploits.
- **Variance requires volume.** More hands needed for significance — also tests memory's ability to extract signal from noise.

Synthetic patterns for poker: "Opponent always min-raises with pocket pairs" or "Opponent folds to 3-bets 85% on the button."

**Statistical note:** Standard deviation in HU NLHE ≈ 80-100 bb/100. To detect 3-5 bb/100 effect at p < 0.05 with 80% power, minimum ~7,000-15,000 hands. 10,000 hand minimum detects moderate effects (≥ 4 bb/100).

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

LLMs possess two separable chess competencies: positional *evaluation* (pattern-matching assessment of who stands better) and tactical *calculation* (tree-search computation of forced sequences). Evaluation is strong — models correctly identify winning positions. Calculation is broken — models cannot compute forced mating sequences even in trivially won positions.

This is not a problem for the benchmark. It is a finding. The benchmark *measures* this gap rather than assuming it away. At each augmentation level, the gap manifests differently:

- **Naked:** Model relies on evaluation only. Calculation gap limits performance.
- **+ Memory:** Memory helps evaluation (opponent patterns, positional preferences). Calculation gap persists.
- **+ Tools:** Stockfish provides calculation. Gap is filled by tool use.
- **+ Full Stack:** Memory directs tool use toward opponent-specific weaknesses. Synergy.

The benchmark tracks how this gap closes across augmentation levels — a dimension no other benchmark captures.

---

## 7. Hypotheses

### Primary (5)

**H1: Augmentation helps.** Δₐ > 0 for real augmentation in both environments.

**H2: Real memory > placebo.** Real memory outperforms random/fake memory of equal size, demonstrating that memory quality matters, not just additional context.

**H3: Environment interaction.** Δₐ differs between chess (complete information) and poker (incomplete information), revealing how augmentation value depends on information structure.

**H4: Synergy.** In Phase 4 (full stack), Δ_full_stack > Δ_orchestration + Δ_memory. Memory + tools are synergistic, not additive — memory directs calculation toward opponent-specific weaknesses.

**H5: Speed matters.** τ varies significantly across augmentation systems even when Δₐ is similar — convergence speed is an independent quality dimension.

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
- Statistical framework: p < 0.05, confidence intervals for all metrics; Bonferroni correction on 5 primary hypotheses
- Game/hand history storage for reproducibility
- Error tracking: invalid/illegal moves per agent per game

### 9.2 Memory Audit Protocol

**Problem:** Without content restrictions, a memory system can be pre-loaded with external knowledge. This measures knowledge injection, not augmentation quality.

**Rule:** Memory systems may only store information derived from the current match.

| Allowed | Forbidden |
|---------|-----------|
| Game/hand IDs and timestamps | Pre-loaded opening databases |
| Observed moves and actions | External GTO charts or solvers |
| Derived patterns with evidence trail | Opponent data from outside the match |
| Consolidated analytical summaries | General strategy guides |

**Enforcement:**
1. Pre-match audit — memory store verified empty.
2. Post-match dump — full contents exported for review.
3. Content hash chain — every write logged with source game ID.
4. Automated validation — orphan entries = contamination flag.

Audit logs published alongside results. Runs without audit logs are unverified.

### 9.3 Error Handling

1. Invalid move → random legal move substitution.
2. Error rate tracked per agent, per phase.
3. Δₐ calculated with and without error-containing games.
4. Agent exceeding 5% error rate = flagged as unreliable.

### 9.4 Cost Target

A standard Agzamov Test run should cost <$50 per model. This enables routine inclusion in eval suites. N values (games/hands) are calibrated to balance statistical power with cost.

---

## 10. Why "Agzamov"

The benchmark is named the Agzamov Test — a double reference to the author's surname, and to Georgy Agzamov (1954–1986), the first chess grandmaster from Central Asia. Georgy was known as the "nightmare of top grandmasters," defeating Tal and drawing Karpov through tenacity, pattern recognition, and counterattack rather than raw calculation. The benchmark aspires to a similar reputation among AI models: a nightmare that rewards adaptation over brute force.

---

## 11. Roadmap

1. **Phase 0** — Smoke test: 30 games, 3 models, confirm they can play
2. **Preliminary data** — 100+ games with/without memory, first Δₐ numbers
3. **Paper** — Methodology + preliminary findings on arXiv (establish priority)
4. **MVP** — Full protocol: two agents, chess + poker, four augmentation levels
5. **Matrix** — 3+ models × 3+ augmentation configs × 2 environments
6. **Leaderboard** — Public, updatable (agzamovtest.com or HuggingFace Spaces)
7. **Standard** — Propose as standard evaluation framework for augmented AI

---

*Published by BrainOps Limited. Open methodology. Attribution required.*
*Contact: Ali Agzamov · BrainOps Limited · Queenstown, New Zealand*
