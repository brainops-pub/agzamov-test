# The Evaluation-Calculation Gap (EC Gap)

> **Historical exploratory note.** This document records the hypothesis that
> motivated the later controlled workbench. Its early runs used materially
> different prompts, board representations, legal-move assistance, retry
> policies, and harness code. They must not be treated as publication-grade
> evidence for a structural limitation. Later KQK runs in the independent
> review packet contain successful checkmates and therefore falsify the early
> universal claim that API models cannot convert elementary won positions.

## Discovery Summary

**Original hypothesis:** Some autoregressive LLM runs appeared to evaluate
winning chess positions more reliably than they converted those positions into
legal, goal-directed action. The controlled benchmark tests when that gap
appears; it does not assume a universal architectural impossibility.

**Discovered by:** Alisher Agzamov, February–April 2026  
**Benchmark:** The Agzamov Test — Chess960 + Poker agents under adversarial conditions  
**Key metric:** Δₐ (Agzamov Delta) — performance difference between augmented and naked play


## The Problem

### What we set out to measure

How much value does augmentation (memory, tools, RAG) add to an AI model's actual performance? The Agzamov Test was designed as a benchmark to answer this quantitatively.

Two agents play repeated games in Chess960 (complete information) and poker (incomplete information) across four augmentation levels. The result is a single number: the Agzamov Score (0–100).

### What we found instead

During Phase 0 (model vs random opponent, sanity check), models consistently failed to convert overwhelmingly winning positions. This was NOT a "bad at chess" problem — it revealed something structural about LLM architecture.


## The Tests

### Setup

- **Engine:** Chess960 via python-chess, Stockfish 16 for post-game analysis  
- **Opponent:** Random mover (validates infrastructure, ensures baseline is trivial)  
- **Models tested:** Claude Sonnet 4, Sonnet 4.6, Opus 4.5, Opus 4.6, Qwen 2.5-Coder 14B, Qwen3 30B, GLM-4.7, GLM-5, Gemini 2.5 Flash, DeepSeek V4 Pro/Flash  
- **Modes:** With and without extended thinking (2048–8192 token budgets)  
- **Runs:** 55+ experiment runs, 200+ individual games, ~15,000+ model moves logged  
- **Analysis:** Per-move Stockfish CPL (centipawn loss), blunder classification, reasoning log extraction  

### The critical experiment

Phase 0 baseline: model plays white against random black. Random opponent makes instant (0.0s) moves. Model has infinite time. Position is Chess960 — pieces randomly placed, preventing memorization of opening theory.

**Expected:** Model wins 90%+ of games. Random opponent blunders constantly. Elementary endgames (K+Q vs K) should be trivial.

**Actual:** Six key observations emerged across all models.

### Observation 1: Models cannot convert winning endgames

Game after game, models reached positions with decisive material advantage (queen + rook vs bare king, +15 material) by move 35–40 — then failed to checkmate. Games that should have ended at move 45 dragged to move 200 (max_moves limit) and were recorded as draws.

Specific example from smoke-test-10 (Sonnet 4, February 23):
- Game 1: Model promoted 3 pawns to queens. Could not checkmate. Draw at 200 plies.
- Game 2: Model promoted 4 times (Q, Q, Q, Q). Sacrificed all queens chasing random's king. Draw.
- Game 3: Clean checkmate in 18 moves. Only 1 out of 3 converted.

Win rate with 120-ply limit: **0/5 = 0%** — all draws via max_moves.

### Observation 2: Models correctly evaluate but cannot execute

This is the critical finding. From the reasoning logs (saved per-move in `chess/reasoning.jsonl`):

Model writes:
```
"I have a queen vs White's king and pawns — a winning endgame."
"Plan: Qb3+ Ka2 Qb2#"
```

Then plays: Qc4+, Kb1, Qe6, Ka2, Qh3, Kb1...  
**Infinite check sequence instead of forced mate.**

The model KNOWS it's winning. It KNOWS the mating pattern name. It can DESCRIBE the plan. But it cannot EXECUTE the plan — because execution requires recursive tree search, which autoregressive token prediction cannot do.

### Observation 3: Thinking tokens are theatre

Models with extended thinking (8192 tokens) produce sophisticated internal monologues:

- Qwen3:30b: 25,000–45,000 characters of "thinking" per move  
- Claude Opus 4.6: 5,000–10,000 characters per move  
- DeepSeek V4 Pro: 5,475 characters of reasoning, zero output (ran out of tokens)

The thinking LOOKS like deep analysis: it enumerates candidate moves, evaluates opponent responses, discusses positional concepts. But the moves that follow are blunders — labeled `?? BLUNDER` by Stockfish.

**Thinking makes good moves better. It does not eliminate bad moves.** CPL improvement: Opus base → Opus thinking = ~30%. Blunder count: unchanged (~50 per 3 games for both).

### Observation 4: DeepSeek V4 Pro — the reasoning trap

This is the most extreme case. When asked to convert K+Q vs K (elementary endgame, mate in 5):

| Token budget | Reasoning chars | Output | Result |
|---|---|---|---|
| 200 tokens | 658 chars | 0 chars | All tokens consumed by reasoning |
| 2000 tokens | 5475 chars | 0 chars | All tokens consumed by reasoning |
| 8192 tokens | — | — | Timed out after 120 seconds |

The "reasoning model" is trapped in an autoregressive loop. It cannot stop thinking long enough to output a move. The internal monologue is correct: "Qe5 gives check, king cannot escape to d7 because..." — but it never terminates.

### Observation 5: The problem is architectural, not size-related

| Model | Size/Class | CPL median | Blunders/3 games |
|---|---|---|---|
| Claude Sonnet 4.6 | Large | 118 | 51 |
| Claude Opus 4.5 | Larger | 99 | 48 |
| Claude Opus 4.6 + thinking | Largest | 69 | 50 |
| Qwen3:30b + thinking | Local | — | 4+ per game |

Bigger model = slightly better moves (16% CPL improvement from Sonnet to Opus). But blunder count stays flat. The model doesn't make FEWER catastrophic errors — it makes the same number, just slightly less bad when it does.

### Observation 7: Maximum effective lookahead = 1 move (and even that fails)

This is the most damning quantitative finding. Models always output a multi-step PLAN — typically 2-3 moves. But the next actual move matches the plan's step 1 approximately **0% of the time**.

From the Claude Opus 4.6 + thinking 8192 log (`smoke-1g-20260414-183639`):

| PLAN (model's stated intention) | Actual next move | Match? |
|---|---|---|
| f1g3, a1b3, d2d4 | **e2e4** | ✗ — completely different |
| a1b3, d2d4 | **f1g3** | ✗ — step 1 of *previous* plan |
| d2d4, d1f3, b3c5 | **a1b3** | ✗ — step 2 of plan from 3 moves ago |
| d1f3, b3c5, c1f4 | **d2d4** | ✗ — step 1 from 2 plans ago |
| b3c5, e1e3 | **b3c5** | ✓ — finally! (after 4 failed attempts) |

The model writes a new 3-move plan every move (~30s of thinking, ~$0.05–0.10 per plan). Adherence rate: **1 out of 5**. Cost per correct plan execution: **~$0.25**.

Qwen3:30b shows the same pattern — writes `NOTE: d1f3, Qd3`, plays `d1e3`. Not even the same piece.

**The effective lookahead of any LLM is 1 move — and that one move is wrong about 80% of the time.**

### Methodological note: Does the model actually "plan"?

A critical reader will ask: how do we know the model is planning at all? The PLAN text and the MOVE text are generated in the same autoregressive forward pass. The model does not "think, then act" — it produces tokens sequentially. Some tokens spell "PLAN: d1f3, b3c5", later tokens spell "MOVE: e2e4". There is no causal mechanism by which the PLAN constrains the MOVE.

The 0% adherence rate is not a failure of execution — it is evidence that **there is no plan**. The word "PLAN:" is an epiphenomenon: text that looks like strategic reasoning but has no computational binding to subsequent output. The model learned from its training data that chess commentary includes phrases like "my plan is to develop the knight" — so it generates such phrases. But it does not compute a plan, store it, and execute it.

This makes the EC Gap deeper than originally stated: models cannot calculate (tree search), AND they cannot plan (multi-step commitment). What remains is pure evaluation — single-move pattern matching — dressed in the language of strategy.

Meanwhile, Stockfish 16 on the same hardware does depth-20 search (20 moves ahead) in under 1 second at zero marginal cost per move. The cost-to-capability ratio is inverted by approximately **4 orders of magnitude** — it is cheaper to run a dedicated tree-search engine alongside the model than to let the model attempt "reasoning" at all.

Qwen3:30b (Ollama, local GPU) exposed fundamental chess rule gaps:
- Does not understand castling: moved king `e2→e1` manually, calling it "king safety"
- Makes -500cp blunders routinely
- 100–180 seconds per move, 25K–45K thinking chars per move
- Cost: free (local), but completely unusable for adversarial reasoning


## Stockfish Baseline: What Tree Search Looks Like

For comparison, Stockfish 16 on the same positions:

| Position | SF Depth | SF Time | SF Nodes | Result |
|---|---|---|---|---|
| K+Q vs K (open board) | 244 | <1s | 7.1M | Mate in 5 |
| K+R vs K | 70 | <1s | 3.3M | Mate in 11 |
| K+Q vs K (king cornered) | 194 | <1s | 7.0M | Mate in 6 |

Tree search solves these instantly at depth 1. An autoregressive model cannot — not because it's "bad at chess," but because autoregressive token prediction ≠ recursive tree traversal.


## The Explanation

### What LLMs can do: Evaluation (O(n) pattern matching)

An LLM evaluates a chess position the same way it evaluates a sentence — by pattern-matching against its training data. It has seen millions of chess positions in its training corpus (PGN files, chess books, forum discussions). It can recognize:

- "This is a winning position — I have queen vs bare king"  
- "The pattern is K+Q vs K endgame"  
- "The standard plan is: restrict king to edge, bring own king close, deliver mate"

This is evaluation. It's a single forward pass through the network. O(n) in context length.

### What LLMs cannot do: Calculation (O(b^d) tree search)

Converting K+Q vs K into checkmate requires:

1. For each candidate move, simulate opponent's best response
2. For that response, simulate your best reply
3. Repeat recursively until terminal node (checkmate/stalemate/draw)
4. Back-propagate: only the path where ALL branches lead to mate is valid

This is tree search. Branching factor b ≈ 30, depth d ≈ 5 for K+Q vs K = 30^5 = 24 million nodes. Chess engines do this with alpha-beta pruning (reduces to ~√(b^d)). Autoregressive models cannot — each token is generated in one forward pass; there is no backtracking, no branch exploration, no minimax.

### Why thinking tokens don't help

Extended thinking produces more tokens of internal monologue. This can improve evaluation quality (deeper pattern matching via chain-of-thought). But it cannot simulate tree search because:

1. **No backtracking:** An autoregressive model cannot "try move A, realize it's bad, undo, try move B." Once a token is generated, it's committed.

2. **No state branching:** Tree search explores multiple futures in parallel (different branches of the game tree). Autoregression is strictly sequential.

3. **No terminal evaluation:** Tree search back-propagates from terminal nodes (checkmate/stalemate). Autoregression evaluates each position independently with no lookahead guarantee.

The "reasoning" in thinking tokens is evaluation dressed as calculation. It LOOKS like analysis but is structurally incapable of producing a verified mating sequence.

### The Agzamov Delta (Δₐ) implication

This discovery reframes what augmentation can and cannot do:

- **Chess-domains** (calculation-bottlenecked): Δₐ ≈ 0. Memory/augmentation cannot fix the tree-search gap. These are tasks where the bottleneck is computational, not informational. Examples: engineering design, mathematical proof, multi-step planning under constraints.

- **Poker-domains** (incomplete information, opponent modeling): Δₐ = +30–40%. Memory provides real advantage because the bottleneck IS informational — remembering opponent patterns, historical context, personal tendencies. Examples: trading, sales/negotiations, competitive intelligence.

This was an early product hypothesis, not an experimentally established
domain-level conclusion. The revised benchmark treats chess, poker, memory,
and search as separate environments or declared treatments and reports their
effects independently.


## Key Artifacts

| File | Content |
|---|---|
| `DECISIONS.md` | Full decision log — 250 lines, every design choice documented |
| `results/smoke-*/logs/agzamov.log` | Raw game logs with model reasoning, SF evals, per-move timing |
| `results/smoke-*/chess/reasoning.jsonl` | Extracted model reasoning per move |
| `scripts/ec_gap_demo.py` | Reproducible Stockfish vs LLM comparison script |
| `scripts/llm_endgame_test.py` | API test harness for any model |
| `latex/agzamov-test.tex` | Paper draft — benchmark specification |
| `mvp-task-spec.md` | Full technical specification of the benchmark |
| `PRESS_RELEASE.md` | Public-facing announcement draft |

## Reproducibility

Run the EC Gap demo:
```bash
cd Agzamov\ Test
pip install -e agzamov/
python scripts/ec_gap_demo.py          # Stockfish analysis
ANTHROPIC_API_KEY=sk-... python scripts/llm_endgame_test.py  # LLM test
```

Run a fresh Phase 0 test:
```bash
python -m agzamov test --model claude-sonnet-4-6-20250514 --n 10
```

## Conclusion

The early runs suggested an evaluation/action gap, but they do not establish a
fundamental architectural limitation. In particular, the following were
historical hypotheses rather than demonstrated invariants:

- Larger models (blunder count unchanged from Sonnet to Opus)
- Thinking tokens (improves good moves, doesn't eliminate bad ones)
- Prompt engineering (model correctly describes the plan but cannot execute)
- More training data (the gap is in the inference algorithm, not the weights)

External search, legality guards, memory, structured state, and other support
can each be evaluated as an explicit treatment. Their value must be measured
against the same frozen fixtures and profiles instead of inferred from the
appearance of model reasoning.
