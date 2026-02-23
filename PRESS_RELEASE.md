# The Agzamov Test: Does AI Memory Actually Work?

**BrainOps Limited · Queenstown, New Zealand · February 2026**

---

## The Problem Nobody Talks About

Every AI company ships memory features. "Your assistant remembers you now!" But here's an uncomfortable question: **does memory actually make AI better, or does it just make it slower?**

Current memory benchmarks measure retrieval — can the system find the right fact in a pile of facts? That's like testing a chess player by asking them to name openings. It tells you nothing about whether they can *play*.

Shooting range accuracy does not equal battlefield performance. Precision and recall scores tell you nothing about what happens when an opponent is actively trying to make your stored knowledge obsolete.

There is no benchmark that answers: **"Does giving an AI memory produce measurably better behavior under pressure?"**

Until now.

---

## What Is the Agzamov Test?

The Agzamov Test is a dual-axis adversarial benchmark that uses repeated game-theoretic competition to measure two things: (1) whether a model can use memory effectively, and (2) whether the memory infrastructure itself is any good.

The test puts AI models through two fundamentally different competitive environments:

- **Chess960** (Fischer Random Chess) — complete information, 960 randomized starting positions, zero opening memorization possible. Every strategic insight must come from reasoning or memory of past games. 500+ games per test.

- **No-Limit Texas Hold'em** (heads-up SNG turbo tournaments) — incomplete information, hidden cards, bluffing, escalating blind pressure. PokerStars-style Sit & Go format: 1500 starting chips, blinds increase every 10 hands, winner-takes-all. 500+ tournaments per test.

If memory helps in both — it works everywhere. If it helps in one but not the other — that tells you something important about what kind of memory you've built.

---

## How It Works

### Five Phases

**Phase 0 — Sanity Check.** Can the model beat a random-move opponent 70%+ of the time? If it can't play basic chess, there's nothing to measure.

**Phase 1 — Baseline.** Two identical models play each other, neither with memory. This establishes the noise floor — how much variation is just randomness.

**Phase 2 — The Real Test.** Same model, same opponent — but one side gets access to a memory system. It can remember patterns from previous games, recall what worked, avoid what didn't. The other side starts fresh every game. The performance gap — **Δₐ (Agzamov Delta)** — is the test's primary metric.

**Phase 3 — Arms Race.** Both sides get memory. Does the advantage disappear, or does memory quality create a persistent edge even when both sides are equipped? This phase includes forced strategy shifts to test how fast memory adapts when the opponent changes behavior.

**Phase 4 — Full Stack Orchestration.** LLM + Stockfish engine + Memory vs a human expert. Not testing the model alone — testing the entire agent architecture: tactical engine, memory system, and LLM as strategic coordinator. Sub-phases isolate what each layer contributes.

**Phase 5 — Stress Test.** Drop agents into losing positions. An agent that plays well from equal positions demonstrates baseline capability. An agent that fights intelligently from disadvantageous positions — leveraging memory of opponent-specific weaknesses — demonstrates the adaptive resilience that matters in real-world deployment.

### The Dual-Axis Insight

What makes this benchmark unique is that it separates model capability from memory quality:

- **Fix the model, vary the memory** → compares memory systems (BrainOps vs competitor A vs competitor B)
- **Fix the memory, vary the model** → compares models (Claude vs GPT vs Gemini)

Run all combinations and you get a **Model × Memory Matrix** — a table showing exactly what drives performance. If a weaker model with better memory beats a stronger model with worse memory, that proves infrastructure can compensate for raw capability. That's the strongest possible argument for investing in memory tools.

---

## Three Numbers That Matter

### Δₐ — The Agzamov Delta

The core metric. Performance with memory minus performance without memory. For chess: win rate percentage points. For poker: tournament win rate difference.

Higher delta = memory adds more value. Negative delta = memory is actively hurting — retrieval noise is degrading decisions. Zero delta = the industry has a problem.

### τ — Convergence Rate

How fast memory starts helping. Two systems can produce identical Δₐ, but one reaches peak performance in 20 games, the other in 200. In production, that's the difference between "useful from day one" and "useful after six months."

**Recovery τ** measures something harder: when the opponent changes strategy, how quickly does memory adapt? Good memory recovers fast. Bad memory stays stuck on outdated patterns. In dynamic real-world environments, this may be more important than Δₐ itself.

### GQI — Game Quality Index

Win rate only captures who won. GQI measures how well each move was played, using Stockfish post-game analysis. Two draws can be radically different — a 15-move repetition is stagnation, an 80-move endgame battle is mastery. GQI detects improvement even when outcomes don't change, and catches "memory poisoning" where retrieved information makes decisions worse before win rate drops.

---

## Why Tournament Poker, Not Cash Games?

The poker format was chosen deliberately. In heads-up cash games, an agent can fold everything except premium hands (AA, KK) and break even indefinitely. No pressure, no need to adapt, no reason to use memory. Δₐ collapses to zero — not because memory is useless, but because the format doesn't force decisions.

SNG turbo tournaments eliminate this escape route. Blinds escalate every 10 hands. By the final stage, with a 1500 stack facing 200/400 blinds, every hand is a push-or-fold decision. You can't wait for aces.

One tournament, three phases of poker:
1. **Deep stack (75bb)** — full poker: raises, 3-bets, bluffs, reads
2. **Medium stack (15bb)** — aggression, position, timing pressure
3. **Short stack (3-4bb)** — pure math: push-or-fold ranges

Each phase tests memory differently. Deep stack: remember betting patterns. Medium stack: remember how the opponent reacts to pressure. Short stack: remember their calling range.

Here's the key insight: Nash equilibrium — the "optimal" poker strategy — works on an infinite horizon. On a finite 50-80 hand tournament, playing Nash means leaving edge on the table. The only way to gain edge on a short horizon is to **deviate from Nash based on information about your opponent**. That information comes from memory. The Folk Theorem from game theory formalizes this: in repeated games, knowledge of history enables exploitation strategies that are impossible in one-shot encounters.

SNG turbo doesn't just test memory. It creates a format where memory is the **only path to sustainable advantage**.

---

## Why Games?

Games are not the point. Games are the *instrument*.

We needed a domain where:
- **Every action is measurable** — no subjective quality scores
- **The environment is adversarial** — a passive benchmark can't reveal how memory performs when someone is actively trying to beat you
- **The opponent adapts** — unlike static benchmarks, the opposing player changes strategy
- **Parametric knowledge is useless** — Chess960 eliminates opening memorization, synthetic opponent patterns don't exist in training data
- **The theory is rigorous** — the Folk Theorem from game theory establishes that in repeated games, memory enables strategies (punishment, exploitation, reputation) that are impossible in one-shot encounters. This isn't arbitrary — it's grounded in 80 years of mathematics.

The same principles apply to any domain where an agent faces a changing environment and benefits from historical context: coding, trading, medical diagnostics, security.

---

## Why This Matters

The AI industry is spending billions on memory infrastructure. RAG pipelines, vector databases, long-context windows, persistent memory APIs — the entire ecosystem assumes that giving AI access to its past makes it better.

**Nobody has proven this in a rigorous, adversarial setting.**

The Agzamov Test is designed to answer this with defensible numbers. If Δₐ is positive and statistically significant, memory works. If it's zero, the industry needs to rethink its assumptions. If it's negative, bad memory is worse than no memory — and that's a finding worth billions in wasted infrastructure investment.

### 14 Testable Hypotheses

This isn't just a tool — it's a research program. Among the hypotheses:

- Memory provides measurable advantage in both complete and incomplete information environments
- Semantic search outperforms keyword search under adversarial conditions
- Knowledge graphs show increasing advantage in later games as relational patterns accumulate
- There exists a memory quality floor below which Δₐ ≤ 0 — bad memory is worse than none
- Weaker models may show Δₐ ≈ 0 even with excellent memory — they lack the reasoning to act on retrieved information
- Memory-equipped agents will show lower error rates (fewer illegal moves) than naked agents

Each hypothesis is testable with the benchmark as built. Each result is publishable.

---

## Integrity

### Memory Audit Protocol

Memory systems under test may only store information derived from the current match. No pre-loaded databases, no external strategy guides, no training data injection. Every memory write is logged with timestamp and source game ID. Orphan entries with no source = contamination. Audit logs published with all results.

### Error Handling

Invalid moves are tracked per agent, per phase. Games with errors are flagged but not excluded. If error rates differ between memory and no-memory conditions, that's a finding — memory may improve or degrade move validity.

### Reproducibility

All random seeds logged. Full game histories stored. Config files versioned. Memory snapshots at regular intervals. Any test run without audit logs is considered unverified.

---

## The Name

The benchmark is named after its creator, **Ali Agzamov**, founder of BrainOps Limited.

By pure coincidence, **Georgy Agzamov** (1954–1986) was the first chess grandmaster from Central Asia — known as the "nightmare of top grandmasters," defeating Tal and drawing Karpov through tenacity, pattern recognition, and counterattack rather than raw calculation. The benchmark aspires to a similar reputation among AI models: a test that rewards adaptation and memory over brute computational force.

---

## Current Status

The Agzamov Test is in active development and validation. The chess MVP is built (11 Python modules, 256 tests passing) and running Phase 0 smoke tests against Claude Sonnet 4. Poker module is planned for Phase B.

The benchmark is designed to be fully open-source (Apache 2.0) and model-agnostic — any LLM that can output a move can be tested. A public leaderboard is planned for HuggingFace Spaces.

### Roadmap

1. **arXiv preprint** — establish priority, enable citation
2. **First results** — Δₐ and τ for BrainOps Memory MCP with Claude
3. **Full matrix** — 3 models × 3 memory systems × 2 game formats
4. **Public leaderboard** — open submissions, reproducible results
5. **Standard** — proposed as standard evaluation framework for AI memory infrastructure

---

## Technical Summary

| Component | Detail |
|---|---|
| Game formats | Chess960 (500+ games) + Heads-up NLHE SNG Turbo (500+ tournaments) |
| Primary metrics | Δₐ (Agzamov Delta), τ (convergence rate), Model×Memory Matrix |
| Derived diagnostics | Elo trajectories, GQI (Game Quality Index via Stockfish) |
| Poker format | SNG Heads-Up Turbo: 1500 chips, blinds +10 hands, winner-takes-all |
| Statistical tests | Binomial test (Phase 0), two-proportion z-test (chess), permutation test (poker), p < 0.05 |
| Engine | python-chess, Stockfish for post-game analysis |
| Memory interface | Any MCP-compatible system (remember/recall/forget) |
| Phases | 0 (sanity) → 1 (baseline) → 2 (asymmetric) → 3 (arms race) → 4 (orchestration) → 5 (stress test) |
| Cost estimate | MVP ~$100-200, full matrix ~$500-1000 |
| Stack | Python 3.12+, scipy, python-chess, Anthropic/OpenAI/Google SDKs |

---

**BrainOps Limited** — Building infrastructure for AI that actually remembers.

*Ali Agzamov · Queenstown, New Zealand*
