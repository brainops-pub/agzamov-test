# The Agzamov Test

**A Game-Theoretic Benchmark for Measuring AI Memory Infrastructure Quality Under Adversarial Conditions**

Ali Agzamov · BrainOps Limited · Queenstown, New Zealand
February 2026

---

## Abstract

Current AI benchmarks evaluate models in static, exam-like conditions — a question is asked, an answer is given, a score is assigned. This tells us nothing about how a model performs in dynamic environments where adversaries adapt, conditions change, and historical context matters.

The Agzamov Test is a dual-axis adversarial benchmark that uses repeated game-theoretic competition to measure two things simultaneously: (1) a model's ability to use memory effectively, and (2) the quality of the memory infrastructure itself. The test employs two game formats — Chess960 (complete information, opening knowledge eliminated) and poker (incomplete information) — to evaluate memory across fundamentally different strategic domains, using synthetic opponent patterns to isolate memory effects from parametric knowledge. It produces three primary outputs — the Agzamov Delta (Δₐ), the convergence rate (τ), and a model×memory performance matrix — plus two derived diagnostics: per-agent Elo trajectories and a Game Quality Index (GQI) based on move-level analysis. The test includes five phases: baseline, asymmetric memory, arms race with explicit strategy-shift recovery protocol, full orchestration with sub-phase decomposition, and positional stress testing from disadvantageous positions. A Memory Audit Protocol ensures that only match-derived information enters the memory system, preventing knowledge injection confounds.

---

## 1. The Problem with Current Benchmarks

All major AI benchmarks (MMLU, HumanEval, HELM, etc.) are exams. They measure what a model knows at a single point in time. A model scores 90% — it's declared intelligent.

But in real-world applications, AI models operate in dynamic environments. Coding agents debug across sessions, remembering past failures. Trading systems adapt to changing market conditions. Diagnostic systems accumulate patient history. None of this is captured by static benchmarks.

Memory infrastructure developers face a worse problem. The only available evaluation methods are synthetic retrieval benchmarks — precision, recall, MRR — that measure whether a system can find a document in a database. They say nothing about whether memory helps an agent perform better under adversarial pressure, where an opponent actively tries to make stored knowledge obsolete.

A model scoring 90% on MMLU may be useless in a dynamic environment. A memory system scoring 99% on retrieval benchmarks may collapse when patterns shift adversarially. Shooting range accuracy ≠ battlefield performance.

The Agzamov Test replaces the exam with a tournament. It replaces the shooting range with a battlefield.

---

## 2. How It Works — Simple Explanation

Imagine two chess players. Both equally skilled. But one forgets everything after each game — every match is a fresh start. The other remembers: how the opponent played, what mistakes they made, what positions they avoid.

Who wins after 100 games? Obviously the one with memory. Even if the forgetful one is slightly smarter.

The Agzamov Test does exactly this with AI models. We sit two models down and make them play against each other many times. First both without memory — we record the score. Then we give one of them memory and measure how much the score changes. That change is the **Agzamov Delta** — a single number that captures how much memory actually helps.

We then repeat with different models and different memory systems. The result is a table showing exactly what drives performance: the model's brain, or the tools we gave it.

We run this in two formats: Chess960 (where both players see everything, but starting positions are randomized to eliminate memorized openings) and poker (where you can't see your opponent's cards). If memory helps in both — it works everywhere.

---

## 3. Two Game Formats

### 3.1 Why Two Games

No single game captures the full range of strategic reasoning. Chess and poker represent two fundamentally different classes of strategic problems:

| Property | Chess | Poker (NLHE) |
|----------|-------|--------------|
| Information | Complete — both players see the full board | Incomplete — hidden cards, unknown intentions |
| Randomness | None — deterministic | High — card distribution |
| Opponent modeling | Useful but optional — position is objectively evaluable | Essential — you cannot play optimally without reading opponent |
| Bluff/deception | Minimal | Core mechanic |
| Memory value | Pattern exploitation, opening preparation | Bet sizing tells, bluff frequency profiling, tilt detection |
| Nash Equilibrium | Theoretical single optimal strategy (unknown) | Well-defined GTO (Game Theory Optimal) baseline |

A memory system that helps in chess but not poker (or vice versa) has a fundamental limitation. A system that helps in both demonstrates general-purpose adaptive capability.

### 3.2 Chess Format

Two AI agents play N ≥ 500 games of Chess960 (Fischer Random). Colors alternate every game to eliminate first-move advantage. Win = 1 point, draw = 0.5, loss = 0.

**Why Chess960, not standard chess:** LLMs have extensive knowledge of standard chess openings from training data. A model playing the Sicilian Defense may be recalling training data, not using external memory. Chess960 randomizes the starting position (960 possible configurations), eliminating opening book knowledge entirely. This isolates the variable we are testing: every strategic insight must come from either real-time calculation or retrieved memory of past games against this specific opponent. Parametric knowledge (training data) becomes useless for opening preparation — only episodic memory (what happened in our previous games) matters.

Chess960 tests memory's ability to exploit patterns in an environment where the opponent's position is fully visible but no prior knowledge of "standard play" applies. Memory value comes from recognizing recurring tactical preferences specific to this opponent: "this opponent weakens kingside when pressured on the queenside," "this opponent plays passively in asymmetric structures."

### 3.2.1 Synthetic Opponent Patterns

To further isolate memory from parametric knowledge, agents in all formats face opponents with **injected behavioral patterns** that do not exist in any training data:

- **Chess960:** "Opponent always castles within 8 moves when possible" or "Opponent avoids trading queens until forced"
- **Poker:** "Opponent always min-raises with pocket pairs" or "Opponent folds to 3-bets 85% of the time on the button"

These patterns are configurable in the test YAML and are unknown to the agent before the match. A memory-equipped agent must discover and exploit them through observation. A naked agent cannot — each game/hand is independent. The delta between discovery speed and exploitation accuracy directly measures memory quality, with zero confound from training data.

**Pattern injection method:** Synthetic patterns are enforced via **system prompt constraints** on the opponent agent (e.g., "You must always castle within 8 moves when castling is legal" or "You must never trade queens unless forced"). The opponent agent still plays autonomously within these constraints — it makes real decisions, just with a behavioral tendency baked in. This is methodologically distinct from move substitution (where the harness overrides the agent's choice), which would make the opponent a script rather than a player. System prompt constraints produce naturalistic behavior with a detectable statistical signature — exactly the kind of pattern a memory-equipped agent should learn to exploit. The specific constraints used in each test run are published with results for reproducibility.

### 3.3 Poker Format

Two AI agents play N ≥ 10,000 hands of No-Limit Texas Hold'em (heads-up). Measured in big blinds won per 100 hands (bb/100) — the standard poker metric.

Poker is the ideal adversarial memory testbed because:

- **Nash Equilibrium is well-defined.** GTO (Game Theory Optimal) poker strategy is the baseline — it cannot be exploited but also cannot exploit. This is our E₀. Any deviation from GTO is either a mistake to punish or a deliberate trap.
- **Opponent modeling is everything.** Without memory, an agent can only play GTO. With memory, it can detect that Opponent folds to 3-bets 80% of the time and exploit relentlessly. This is pure Δₐ.
- **Bluffing creates rich memory targets.** Showdown data reveals when opponent was bluffing. Over hundreds of hands, a memory-equipped agent builds a statistical profile that a memoryless agent cannot.
- **Variance requires volume.** Poker has inherent randomness (card distribution), so more hands are needed for statistical significance. This also tests memory's ability to extract signal from noise — a harder and more realistic challenge.

Higher hand count (10,000 vs 500 for chess) compensates for poker's higher variance while also stress-testing memory at scale.

**Statistical note on sample size:** Standard deviation in heads-up NLHE ≈ 80-100 bb/100. To detect a realistic memory effect of 3-5 bb/100 at p < 0.05 with 80% power, minimum sample size is approximately 7,000-15,000 hands depending on effect size. The 10,000 hand minimum is sufficient for detecting moderate effects (≥ 4 bb/100). For smaller effects or when comparing two memory systems against each other (where the delta between deltas is smaller), 15,000-20,000 hands may be required. Preliminary runs should be used to estimate effect size and adjust sample requirements. All-in EV adjustment (replacing actual pot outcomes with expected value at all-in) can reduce variance by ~30%, lowering sample requirements accordingly. With local models, running 20,000+ hands costs nothing — sample size should default to the maximum practical volume.

---

## 4. Five Metrics

The Agzamov Test produces **three primary outputs** and **two derived diagnostics**:

- **Primary:** Agzamov Delta (Δₐ), Convergence Rate (τ), Model×Memory Matrix
- **Derived:** Elo Rating, Game Quality Index (GQI)

Primary outputs are the core deliverables of any test run. Derived diagnostics provide additional resolution for interpretation and debugging but are not required for a valid result.

### 4.1 Agzamov Delta (Δₐ)

The core metric. Difference in performance between memory-equipped and naked play.

```
Δₐ = Performance(model + memory) − Performance(model, naked)
```

For chess: measured in win rate percentage points.
For poker: measured in bb/100 (big blinds per 100 hands).

Example (chess): Model wins 50% without memory, 65% with memory. Δₐ = +15 percentage points.
Example (poker): Model earns +2 bb/100 without memory, +18 bb/100 with memory. Δₐ = +16 bb/100.

Higher delta = memory infrastructure adds more value.

A negative delta means memory is actively hurting performance — retrieval noise degrades decision-making. This is a critical finding if observed: bad memory is worse than no memory.

### 4.2 Convergence Rate (τ)

How quickly memory starts helping. Measured as the number of games (chess) or hands (poker) needed to reach 95% of the maximum performance level.

In the first few games, memory is empty — nothing to remember. As games accumulate, the model builds a profile of its opponent. Performance climbs. Eventually it plateaus — the opponent is fully modeled.

τ is the game/hand number where the curve hits 95% of that plateau.

**Why τ matters:**

Two memory systems can produce identical Δₐ. But one reaches peak performance in 20 games, the other in 200. In production, this is the difference between "useful from day one" and "useful after six months of data collection."

**Recovery τ** — a second dimension. When the opponent changes strategy, performance drops. How quickly does it recover? Good memory adapts fast (low recovery τ). Bad memory stays stuck on outdated patterns (high recovery τ).

Recovery τ measures resilience — the ability to handle adversarial shift. In dynamic real-world environments, this may be more important than Δₐ itself.

**τ in poker vs chess:** Poker τ is expected to be higher (more hands needed) due to variance. But the ratio τ_poker / τ_chess reveals how efficiently memory extracts signal from noisy vs clean environments — an independent quality dimension.

### 4.3 Model × Memory Matrix

The tool for isolating what drives performance.

Take 3 models (Claude, GPT, Gemini) and 3 memory systems (ours, competitor A, none). Run all 9 combinations in both game formats. Results go into a matrix:

**Chess (win rate %)**

|                | No Memory | Memory A (ours) | Memory B (competitor) |
|----------------|-----------|-----------------|----------------------|
| **Claude**     | 50%       | 65%             | 57%                  |
| **GPT**        | 48%       | 61%             | 55%                  |
| **Gemini**     | 52%       | 58%             | 54%                  |

**Poker (bb/100)**

|                | No Memory | Memory A (ours) | Memory B (competitor) |
|----------------|-----------|-----------------|----------------------|
| **Claude**     | +2        | +18             | +9                   |
| **GPT**        | +1        | +14             | +8                   |
| **Gemini**     | +3        | +11             | +7                   |

Reading the matrix:

- **Read rows** (fixed model, vary memory): Shows how well each memory system works with this model. If our memory consistently beats competitor's across all models → our memory is objectively better.
- **Read columns** (fixed memory, vary model): Shows which model is best at using memory. If Claude extracts more value from the same memory than Gemini → Claude is better at leveraging external tools.
- **Cross-game comparison**: If a memory system shows high Δₐ in chess but low in poker (or vice versa), it reveals domain-specific limitations. A general-purpose memory system should show consistent gains across both formats.
- **Diagonal insight**: If a weaker model + better memory beats a stronger model + worse memory, that proves memory infrastructure can compensate for model capability. This is the strongest possible argument for investing in memory tools over chasing the latest model.

### 4.4 Elo Rating

A fourth metric that smooths the volatility of raw win rates. Each agent maintains a running Elo rating throughout the test, updated after every game/hand using standard Elo calculation (K-factor = 32 for chess, K = 16 for poker due to higher volume).

**Why Elo in addition to win rate:**

Win rate captures the final outcome. Elo captures the trajectory — how quickly an agent improves, when it plateaus, and how sharply it drops when the opponent adapts. In Phase 3 (Arms Race), Elo graphs of both agents reveal the mutual adaptation dynamic in a way that aggregate win rate cannot: who leads the learning curve, who recovers faster, and whether the system converges to a stable equilibrium or oscillates.

### 4.5 Game Quality Index (GQI)

Win rate and Elo only capture outcomes — who won, who lost, who drew. They say nothing about the quality of decisions that led to those outcomes. Two draws can be radically different: a 15-move repetition is stagnation; an 80-move endgame battle is mastery.

GQI measures the average quality of moves throughout a game, independent of the result.

**Chess GQI:** Calculated using Stockfish post-game analysis. For every move, Stockfish evaluates the position before and after. The difference between the model's chosen move and the best available move is the centipawn loss (CPL). Average CPL across all moves in a game = that game's GQI. Lower CPL = higher quality play.

A memory-equipped agent playing draws with average CPL of 15 is objectively stronger than a naked agent drawing with CPL of 45. The games look identical on the scoreboard, but the logs reveal a massive quality gap.

**Why GQI matters:**

1. **Resolves the draw compression problem.** When two strong models draw 70% of chess games, win rate becomes insensitive. GQI remains sensitive — it detects improvement even when outcomes don't change.
2. **Detects "memory poisoning."** If a memory system retrieves stale or irrelevant information, the agent may make worse moves than baseline. GQI will show CPL increasing even if win rate hasn't dropped yet — it's an early warning system.
3. **Measures defensive quality.** An agent that loses slowly and resists well (low CPL despite being in a losing position) is extracting more from its memory than one that collapses quickly.

**Scope of GQI:** GQI is not a proxy for human-like style or creative play. It measures tactical correctness relative to a strong oracle (Stockfish). For our purpose — isolating whether memory improves decision quality under pressure — this is exactly what we want. A memory-equipped agent that consistently finds Stockfish-approved moves is demonstrating that retrieved information improves decision-making, regardless of whether those moves "look human."

**Poker GQI (future work):** An analogous metric for poker would compare the expected value (EV) of the agent's chosen action against a solver's optimal line at each decision point. This is computationally expensive (solver queries per hand × decision points) but theoretically sound. We flag this as a future extension; for the initial benchmark, poker performance is measured via bb/100, Elo, and τ — all outcome-based metrics with sufficient sensitivity for memory effect detection.

---

## 5. Theoretical Foundation

### 5.1 Game Theory Basis

The Agzamov Test is grounded in the distinction between one-shot and repeated games — a fundamental concept in game theory since von Neumann and Morgenstern (1944).

**One-shot game (no memory):** Each encounter is independent. The agent has no information about the opponent beyond what is observable in the current game state. Strategies are static. In poker, the optimal one-shot strategy is GTO — unexploitable but also non-exploiting. In chess, without opponent-specific knowledge, a model relies purely on position evaluation.

**Repeated game (with memory):** The Folk Theorem (Aumann & Shapley, 1994) establishes that in infinitely repeated games, any individually rational outcome can be sustained as a Nash Equilibrium through adaptive strategies — including punishment, reputation building, and opponent exploitation. While our games are finitely repeated (N = 500 or 10,000), for sufficiently large N the Folk Theorem dynamics emerge as a practical approximation.

The Agzamov Delta measures exactly this shift:

```
Δₐ = E_repeated − E_one-shot
```

This is not an arbitrary metric. It captures the measurable strategic value that memory adds, grounded in eight decades of game theory.

### 5.2 Why Poker Strengthens the Theoretical Foundation

The Nash Equilibrium concern with chess is legitimate: chess is a deterministic, complete-information game where the theoretical Nash Equilibrium is a single optimal strategy (though unknown in practice). Models don't reach 50/50 because of Nash — they reach it because they're approximately equal in skill.

Poker resolves this cleanly. In heads-up No-Limit Hold'em, GTO strategy is a true mixed-strategy Nash Equilibrium — it involves randomized bet sizes, bluff frequencies, and call thresholds. Playing GTO is the provable baseline (E₀). Any deviation from GTO is either exploitable or exploiting. Memory enables the transition from "cannot be beaten" (GTO) to "actively winning" (exploitative play). The delta between these two is pure Δₐ, with clean theoretical grounding.

### 5.3 Why Adversarial Testing

Synthetic retrieval benchmarks (precision, recall, MRR) measure memory in a vacuum: "Did the system find the right document?" The Agzamov Test asks a harder question: "Did the system find the right information at the right moment under pressure, when the opponent is actively trying to make stored knowledge obsolete?"

An opponent who detects they are being exploited will change strategy. Memory that cannot handle this shift becomes a liability — the agent acts on outdated information and performs worse than if it had no memory at all. Only adversarial testing reveals this failure mode.

### 5.4 Scope and Limitations

We do not claim that chess and poker exhaust real-world complexity. We use them as canonical, theory-grounded laboratories where repeated adversarial dynamics and memory effects can be precisely measured with established analytical tools. The principles tested — adaptation, opponent modeling, pattern exploitation under pressure — transfer to any domain where an agent faces a changing environment and benefits from historical context. Chess and poker are chosen because they provide controlled complexity, well-understood baselines (Stockfish evaluation, GTO strategy), and decades of human performance data for calibration.

### 5.5 Disentangling Model vs Memory Contributions

A predictable objection: "You're measuring model+memory together — how do you know which one is responsible?"

The model×memory matrix (§4.3) is specifically designed to answer this:

- **If Δₐ ≈ 0 across an entire row** (fixed model, all memory systems): The bottleneck is the model. It cannot reason over retrieved information regardless of retrieval quality. This is a finding about the model, not a test failure.
- **If Δₐ is consistently higher in one column** (fixed memory, all models): That memory system provides genuine value independent of which model uses it. This is the strongest evidence for memory infrastructure quality.
- **If Δₐ varies both by row and column**: Both model capability and memory quality matter, and their interaction reveals which combinations are synergistic vs redundant.

The matrix does not merely display results — it is the analytical instrument for attribution. No single run can disentangle model from memory. The full matrix can.

---

## 6. Test Protocol

### Phase 1: Baseline (E₀)
Both agents play without memory. Chess: N ≥ 500 games, alternating colors. Poker: N ≥ 10,000 hands. Establishes baseline performance for each model.

### Phase 2: Asymmetric (Δₐ measurement)
Agent A receives memory infrastructure. Agent B plays naked (same model, no memory). Same N as Phase 1. The performance difference from Phase 1 = Agzamov Delta. Run separately for each model×memory combination.

### Phase 3: Arms Race (E₂)
Both agents receive memory (same or different memory systems). Same N. Measures equilibrium when both sides adapt.

Interpretation:
- Both have identical memory → E₂ ≈ E₀: memory cancels out, only model capability matters.
- One has superior memory → E₂ ≠ E₀: memory quality creates persistent advantage even when both sides are equipped.
- Both have memory but different architectures → difference reveals which architecture wins under competitive pressure.

**Recovery τ Protocol:** At a pre-defined trigger point (configurable, default: game 300 for chess, hand 5000 for poker), Agent B's strategy is forcibly shifted to measure how quickly Agent A's memory adapts. The shift follows one of these scripted patterns:

```yaml
strategy_shift:
  trigger_chess: 300        # game number
  trigger_poker: 5000       # hand number
  shift_type: one of:
    - "aggressive_to_passive"   # opponent stops initiating, plays defensively
    - "exploitative_to_gto"     # opponent switches from exploiting to unexploitable
    - "pattern_break"           # opponent reverses a previously consistent pattern
  measurement:
    pre_shift_baseline: [last 50 games / 500 hands before trigger]
    recovery_window: games/hands until performance returns within 5% of pre-shift baseline
    recovery_τ: number of games/hands in recovery window
```

Recovery τ is measured as the number of games/hands from trigger to recovery. Multiple shift types are tested sequentially within the same match to measure adaptation across different disruption modes.

### Phase 4: Orchestration (chess only)
LLM + Stockfish (as MCP tool) + Memory vs human expert player. This phase tests not the model or memory alone, but the full agent orchestration stack:

- **Layer 1 — Tactics:** Stockfish provides positional calculation (the "expert witness")
- **Layer 2 — Memory:** Memory MCP stores opponent history, weaknesses, psychological profile (the "case archive")
- **Layer 3 — Strategy:** LLM acts as meta-coordinator (the "barrister") — Stockfish says move A is objectively best, but memory shows opponent collapses in closed positions, so LLM chooses move B
- **Layer 4 — Adaptation:** Real-time pattern detection and strategy adjustment during the game

**Sub-phases for isolating memory contribution:**

```
Phase 4a: Stockfish alone vs Human        → engine baseline
Phase 4b: LLM + Stockfish vs Human        → orchestration effect
Phase 4c: LLM + Stockfish + Memory vs Human → full stack

Δ_orchestration = 4b − 4a   (what does LLM coordination add?)
Δ_memory        = 4c − 4b   (what does memory add on top?)
Δ_full_stack    = 4c − 4a   (total system improvement)
```

Without these sub-phases, it is impossible to attribute performance improvement to memory vs orchestration. Phase 4a establishes that Stockfish alone sets a high bar. Phase 4b shows whether LLM coordination adds or subtracts value. Phase 4c reveals whether memory provides additional lift when the orchestration layer already exists.

Phase 4 is not "AI plays chess." It is an agent orchestration system using chess as the environment. The same architectural patterns apply to coding, trading, medical diagnostics, and any domain requiring tactical execution + historical context + strategic judgment.

### Phase 5: Positional Stress Test (chess only)

Phases 1-3 start from the initial position and let models play full games. Phase 5 drops agents into pre-configured positions with known Stockfish evaluations — including disadvantageous ones — and measures how they perform.

**Position categories:**

- **Equal** (eval 0.0 ± 0.5): Tests whether memory improves play quality when outcome is uncertain.
- **Slight disadvantage** (eval -1.5 to -3.0): A human master can hold these. Tests whether memory of opponent-specific weaknesses helps an agent survive or equalize.
- **Severe disadvantage** (eval -3.0 to -5.0): Objectively lost. Measures survival time — how many moves the agent resists before collapse.

**Position library:** A curated set of 50-100 positions across all categories, selected from grandmaster games and endgame studies. Both agents play both sides of each position, alternating colors.

**Metrics:**

- **Survival index:** Moves survived beyond expected collapse point (based on Stockfish's win probability curve). Memory-equipped agent that survives 40 moves in a -4.0 position vs naked agent that collapses in 15 = direct evidence of adaptive value.
- **Recovery rate:** In slight-disadvantage positions, how often does the agent equalize or win? With memory of this specific opponent's tendencies, a -2.0 position may become holdable.
- **GQI under pressure:** Average centipawn loss specifically in disadvantageous positions. Memory should improve decision quality most when under stress — just as it does in real-world applications.

**Why this phase matters:**

Phase 5 adapts a standard evaluation methodology from adversarial professions to AI benchmarking. In law, moot court gives students unwinnable cases — evaluation focuses on argument quality, not verdict. In military staff exercises, officers face scenarios with inferior numbers and bad terrain — the test is adaptation speed and decision quality under pressure, not victory. In investment banking case interviews, candidates analyze failing companies — the measure is diagnostic clarity, not a rescue plan.

The principle is universal: **competence is revealed under pressure, not in comfort**. An AI agent that plays well from equal positions demonstrates baseline capability. An agent that fights intelligently from losing positions — leveraging memory of opponent-specific weaknesses to find survival paths — demonstrates the adaptive resilience that matters in real-world deployment. Coding agents face broken codebases. Trading systems face market crashes. Diagnostic systems face atypical presentations. Phase 5 tests the quality that separates a useful tool from a brittle one.

---

## 7. Hypotheses

**H1:** Δₐ > 0 for all tested memory systems in both game formats. Memory provides measurable performance increase in adversarial repeated games.

**H2:** Different memory architectures produce measurably different Δₐ with the same model. Semantic search, keyword search, and knowledge graph architectures show distinct performance profiles.

**H3:** Δₐ is larger in poker than chess for the same model×memory combination, because incomplete information environments reward opponent modeling more heavily.

**H4:** Semantic search outperforms FTS (full-text search) under adversarial conditions, because opponent adaptation makes exact-match patterns unreliable.

**H5:** Knowledge graph shows increasing advantage in later games (n > 50 in chess, n > 2000 in poker), as relational patterns between opponent behaviors become more valuable than raw recall.

**H6:** There exists a memory quality floor below which Δₐ ≤ 0. Bad memory (high retrieval noise) is worse than no memory.

**H7:** τ varies significantly across memory architectures even when Δₐ is similar — speed of learning is an independent quality axis.

**H8:** Recovery τ (after opponent strategy shift) correlates more strongly with memory architecture quality than initial τ.

**H9:** The ratio τ_poker / τ_chess characterizes a memory system's noise tolerance — its ability to extract signal from stochastic environments.

**H10:** Weaker models (e.g., Haiku-class) may show Δₐ ≈ 0 even with high-quality memory, because they lack the reasoning capability to act on retrieved information. This is not a memory failure — it is a model capability floor. The matrix will reveal this: if Δₐ is near zero across all memory systems for a given model, the bottleneck is the model, not the tools.

**H11:** Chess Δₐ will be compressed relative to poker Δₐ due to high draw rates between models of similar capability. Two strong models playing "correct" chess will draw 60-80% of games, leaving limited room for memory-driven exploitation. Poker, with its inherent variance and bluffing dynamics, provides wider performance spread and therefore more sensitive Δₐ measurement. This makes poker the primary format for precise memory evaluation, while chess serves as a secondary validation in complete-information environments.

**H12:** In Phase 5 (Positional Stress Test), memory-equipped agents will show the largest relative improvement over naked agents in slight-disadvantage positions (-1.5 to -3.0). In equal positions, both agents play reasonably well. In severely lost positions, no amount of memory saves a hopeless situation. The sweet spot — where memory makes the most difference — is positions that are bad but recoverable, where knowledge of the opponent's specific weaknesses can turn a probable loss into a draw or a draw into a win.

**H13:** Temporally weighted memory (recent games weighted higher than older games) outperforms uniform memory in Phase 3 (Arms Race), where the opponent continuously adapts. In an arms race, information from game 50 is likely obsolete by game 300 — the opponent has already changed strategy. Memory systems with built-in temporal decay or recency bias will show lower recovery τ and higher Δₐ in Phase 3 specifically. This tests whether memory architectures handle staleness, not just retrieval accuracy.

**H14:** Memory-equipped agents will show equal or lower error rates (invalid/illegal moves) compared to naked agents. Memory context provides additional grounding for move generation. If the opposite is observed — memory increases error rate — it indicates that memory context is confusing the model's move generation, a critical failure mode worth documenting.

### 7.1 Emergent Hypotheses from Early Testing

The following hypotheses emerged from Phase 0 smoke testing (February 2026), where we observed unexpected model behavior that revealed fundamental LLM limitations independent of memory effects.

**H15 (Evaluation-Calculation Gap):** LLMs possess two separable chess competencies: positional *evaluation* (pattern-matching assessment of who stands better) and tactical *calculation* (tree-search computation of forced move sequences). Evaluation is strong — models correctly identify winning positions, material advantages, and strategic plans. Calculation is fundamentally broken — models cannot compute forced mating sequences even in trivially won positions (e.g., 2R+N+B vs lone King). This gap is not a training data issue but a structural limitation: evaluation is pattern matching (next-token prediction over chess commentary), while calculation requires recursive tree traversal (if-then branching over multiple levels), which autoregressive generation cannot perform. We predict this gap will be measurable as a distinct metric: Evaluation Accuracy (% of positions where the model correctly identifies the better side) vs Calculation Accuracy (% of forced sequences the model executes correctly). We expect Evaluation Accuracy > 85% and Calculation Accuracy < 30% for current frontier models.

**H16 (Generality of the EC Gap):** The Evaluation-Calculation Gap observed in chess is a domain-general limitation that manifests wherever tasks require multi-step if-then reasoning with branching. Legal reasoning (if plaintiff argues X, defendant responds Y, judge applies Z), medical treatment planning (if drug A then monitor B, if B exceeds threshold then adjust C), financial modeling (if we cut price, competitor responds X, market shifts Y) — all require the same tree-traversal capability that LLMs lack. Chess makes this limitation precisely measurable because positions have objective evaluations (Stockfish) and forced sequences have verifiable correctness. The Agzamov Test thus serves as a proxy measurement for a limitation that affects all LLM applications requiring multi-step contingent reasoning.

**H17 (Memory Helps Evaluation, Not Calculation):** Memory will disproportionately improve evaluation-dependent play (opening/middlegame strategy, positional decisions, opponent pattern recognition) while providing minimal benefit to calculation-dependent play (endgame technique, tactical combinations, forced sequences). Δₐ measured in the middlegame phase of games will be significantly higher than Δₐ measured in endgame phases. This predicts that the GQI improvement from memory will plateau or reverse in simplified positions where calculation dominates over evaluation — a testable prediction that cleanly separates memory's contribution from the model's structural limitation.

**H18 (Hybrid Architecture Prediction):** Phase 4 (LLM + Stockfish + Memory) will show a non-linear performance jump over Phase 2 (LLM + Memory alone), because Stockfish supplies exactly the calculation capability that LLMs lack. The combination fills both gaps: memory provides opponent-specific evaluation context, Stockfish provides forced-sequence calculation. We predict Phase 4 Δ_full_stack will exceed the sum of Δ_orchestration + Δ_memory — the components are synergistic, not additive — because memory directs Stockfish's calculation toward opponent-specific weaknesses that Stockfish alone would not prioritize.

**H19 (Adjudication as Measurement Instrument):** Without material adjudication, model-vs-random baseline data is systematically corrupted: won positions are recorded as draws (max_moves termination), inflating perceived draw rates and depressing measured win rates. This biases Δₐ measurement by compressing the baseline (E₀ appears lower than actual model capability). Material adjudication with threshold ≥ 10 piece-value points sustained over ≥ 6 consecutive plies produces results consistent with true model capability, as validated by Stockfish post-game evaluation of adjudicated positions. Any Agzamov Test run without adjudication should report adjudication-corrected results alongside raw results for comparability.

**H20 (Endgame as Calculation Purity Test):** The endgame phase — specifically positions where one side has decisive material advantage against a lone king — serves as a pure calculation test with zero evaluation ambiguity. The model knows it is winning (evaluation = correct), but cannot execute the win (calculation = failed). The number of moves to adjudication in won endgame positions, normalized against the theoretical optimal (known from endgame tablebases), provides a clean "Calculation Efficiency Score" independent of memory effects. This score should be approximately equal for memory-equipped and naked agents (confirming H17: memory does not help calculation), and should vary significantly across model families (revealing architectural differences in sequential reasoning capability).

---

## 8. Open Questions

1. Does the Δₐ ↔ memory quality relationship scale linearly, or are there phase transitions (sudden jumps in performance at certain quality thresholds)?

2. Can superior memory compensate for inferior model capability? (Claude + bad memory vs Gemini + great memory — which axis dominates?)

3. Is there a ceiling to Δₐ regardless of memory quality? At some point, the opponent may become fully predictable and further memory improvement yields no benefit.

4. Does multi-game memory transfer across opponents? If memory trained on Opponent A helps against Opponent B, that suggests general strategic learning, not just opponent-specific memorization.

5. How does the chess Δₐ correlate with poker Δₐ for the same memory system? Strong correlation → memory quality is domain-independent. Weak correlation → different memory architectures suit different problem classes.

6. In Phase 3 (both with memory), does an arms race dynamic emerge where both agents continuously shift strategies, or does the system converge to a stable equilibrium?

7. Can the test be extended to multi-agent environments (3+ players in poker) where coalition dynamics and reputation effects add complexity?

---

## 9. Implementation Requirements

- Chess engine: python-chess for Phases 1-3 (Chess960 mode); Stockfish via MCP for Phase 4
- Poker engine: heads-up NLHE implementation with standard hand evaluation
- Memory MCP server: BrainOps implementation (SQLite + Neo4j + embeddings) as primary test subject
- Competitor memory systems for comparison matrix
- Model API access: Claude, GPT, Gemini (minimum 3 providers); local models (Llama, Qwen) for cost-free preliminary testing
- Statistical framework: p < 0.05 significance threshold, confidence intervals reported for all metrics
- Game/hand history storage for post-hoc analysis and reproducibility
- τ calculation: sliding window win rate with exponential smoothing, 95% threshold detection
- Error tracking: invalid/illegal moves logged per agent per game; games with errors flagged and reported separately (see §9.2)

### 9.1 Memory Audit Protocol

**Problem:** Without content restrictions, a memory system can be pre-loaded with external knowledge — opening databases, GTO charts, opponent profiles from public data. This would measure knowledge injection, not memory infrastructure quality.

**Content restrictions:** Memory systems under test may only store information derived from the current match:

| Allowed | Forbidden |
|---------|-----------|
| Game/hand IDs and timestamps | Pre-loaded opening databases |
| Observed moves and actions | External GTO charts or solvers |
| Derived patterns with evidence trail | Opponent data from outside the match |
| Consolidated analytical summaries | General chess/poker strategy guides |
| Win/loss outcomes and contexts | Training data regurgitation triggers |

**Enforcement:**

1. **Pre-match audit:** Memory store is verified empty before Phase 1 begins. Any pre-existing content = disqualification.
2. **Post-match dump:** Full memory contents exported after each phase. Available for manual or automated review.
3. **Content hash chain:** Every memory write is logged with timestamp, source game ID, and content hash. Orphan entries (no source game) = contamination flag.
4. **Automated validation:** Script checks that every stored memory references a valid game ID from the current match. Violations reported in results.
5. **Blind match assignment:** Memory system does not know which opponent model it will face until Phase 1 begins. No pre-profiling.

Memory Audit Protocol logs are published alongside results. Any test run without audit logs is considered unverified.

### 9.2 Error Handling Protocol

LLM agents may produce invalid moves (illegal chess moves, out-of-turn poker actions). These are handled as follows:

1. **Invalid move → random legal move** substitution (standard practice in LLM chess).
2. **Error rate tracked** per agent, per phase: total errors / total moves.
3. **Games with errors flagged** but not excluded from primary analysis. Reported separately as:
   - Error rate per agent (e.g., "Agent A: 12 errors in 500 games, Agent B: 3 errors")
   - Δₐ calculated both with and without error-containing games
   - If error rates differ significantly between memory/no-memory conditions, this is a finding (memory may increase or decrease move validity)
4. **High error rate threshold:** If any agent exceeds 5% error rate in a phase, results for that agent are flagged as unreliable.

---

## 10. Why "Agzamov"

The benchmark is named the Agzamov Test — a double reference to the author's surname, and to Georgy Agzamov (1954–1986), the first chess grandmaster from Central Asia. Georgy was known as the "nightmare of top grandmasters," defeating Tal and drawing Karpov through tenacity, pattern recognition, and counterattack rather than raw calculation. The benchmark aspires to a similar reputation among AI models: a nightmare that rewards adaptation and memory over brute computational force.

---

## 11. Roadmap

1. **Paper** — Publish on arXiv as preprint (establish priority, enable citation)
2. **Implementation** — MVP: two agents, chess + poker, with/without BrainOps memory
3. **First Results** — Measure Δₐ and τ in both formats for BrainOps Memory MCP
4. **Matrix** — Expand to 3 models × 3 memory systems × 2 game formats
5. **Leaderboard** — Public leaderboard on HuggingFace Spaces or agzamovtest.com
6. **Community** — Open methodology, open source test harness, attribution required (CC BY 4.0)
7. **Standard** — Propose as standard evaluation framework for AI memory infrastructure

---

*Published by BrainOps Limited. Open methodology. Attribution required.*
*Contact: Ali Agzamov · BrainOps Limited · Queenstown, New Zealand*
