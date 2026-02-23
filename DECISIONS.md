# Agzamov Test — Decision Log

## 2026-02-23 | Domain extrapolation: where memory matters most

**Insight from CPL analysis:** Chess EC Gap maps directly to real-world domains. The key question is whether a domain is "chess" (evaluation-dominant, calculation-bottlenecked) or "poker" (incomplete information, opponent modeling).

**Domain classification:**

| Domain | Type | Base LLM coverage | Memory Δₐ | Why |
|---|---|---|---|---|
| Trading | Poker | 40-50% | **+30-40%** | Incomplete info, opponent modeling (other market participants), personal pattern history |
| Sales/negotiations | Poker | 50-60% | **+25-35%** | Opponent modeling, deal history, relationship context |
| Law | Chess (eval) | 70-80% | +10-15% | Pattern matching from precedent (already in weights), memory adds case continuity |
| Medicine | Chess (eval) | 70-80% | +10-15% | Diagnostic pattern matching (in weights), memory adds patient history |
| Engineering | Chess (calc) | 40-50% | +5-10% | Bottleneck is multi-step calculation, memory helps marginally |

**Why trading is poker, not chess:**
- Incomplete information (can't see other positions)
- Stochastic environment (market unpredictable beyond 1-2 steps)
- Opponent modeling = other market participants' behavior
- "Last 3 times AAPL beat earnings by 5%+, stock dropped 2% next day" — specific pattern from YOUR trading history, not in training data
- Decision making = evaluation + opponent modeling, both amplified by memory
- No tree search needed — market is unpredictable beyond 1-2 steps anyway

**Why law is chess:**
- "This case resembles Smith v. Jones" — pattern matching already in LLM weights
- Precedent-based reasoning = large pattern database, pre-trained
- Minimal "calculation" — legal reasoning is analogical, not tree-search
- Memory adds: case continuity (6 months of context), specific judge/court patterns, firm-specific practice

**Implication for BrainOps:** Target poker-domains (trading, sales, competitive intelligence) where memory = competitive advantage. Chess-domains (law, medicine) will buy memory as nice-to-have, not must-have.

---

## 2026-02-23 | 3-model Chess960 comparison: EC Gap is architectural

**Test:** Phase 0 sanity (3 games each, model vs random, Chess960).

| Model | Wins | Checkmates | Adj. | Errors | Avg moves | Cost |
|---|---|---|---|---|---|---|
| Sonnet 4.6 | 3/3 | 0 | 3 | 1 illegal | 45.7 | $0.56 |
| Opus 4.5 | 3/3 | 0 | 3 | 0 | 45.0 | $2.27 |
| Opus 4.6 (thinking) | 3/3 | 1 (M1) | 2 | 0 | 45.7 | $9.67 |

**Post-game Stockfish analysis (CPL capped at 500, depth 15):**

| Model | CPL avg | CPL median | avg GQI | Blunders | Mistakes |
|---|---|---|---|---|---|
| Sonnet 4.6 | 203.3 | 118.0 | 202.9 | 51 | 41 |
| Opus 4.5 | 175.6 | 99.0 | 181.1 | 48 | 40 |
| Opus 4.6 (thinking) | 164.9 | **69.0** | 174.2 | 50 | 34 |

**Key findings:**

1. **Base model size helps ~16%.** Opus 4.5 median CPL 99 vs Sonnet 118 — larger model plays slightly better moves. But both fail at the same things (0 checkmates, same blunder count).

2. **Thinking = biggest quality jump.** Median CPL 69 vs 99 (Opus base) — thinking improves per-move quality by ~30% beyond model size. Combined effect: Sonnet 118 → Opus thinking 69 = 42% improvement.

3. **EC Gap confirmed architectural.** All three models blunder at similar rates (~50 per 3 games). Thinking reduces CPL but not blunder count — it makes good moves better, doesn't eliminate bad ones.

4. **Cost scaling:** Sonnet $0.56 → Opus 4.5 $2.27 (4x, -16% CPL) → Opus thinking $9.67 (17x, -42% CPL). Diminishing returns per dollar.

**Implication for benchmark:** Δₐ (memory advantage) should manifest in position evaluation improvement (which LLMs CAN do), not in tactical calculation (which they structurally cannot). Memory helps "know what worked" — not "calculate 5 moves ahead." Use Sonnet 4.6 for Phase 1-2 bulk testing ($0.19/game) — CPL quality sufficient to measure Δₐ.

---

## 2026-02-23 | Unique run names + old results purged

**Bug:** `cli.py test --n 5` hardcoded name `smoke-test-5`. JSONL uses append mode → every re-run appended to the same file. Result: 5 forfeits + 5 Sonnet 4 games + 2 Sonnet 4.6 games + 5 adjudication games all in one JSONL. Summary JSON only reflected the last run, but raw data was garbage.

**Fix:** Run names now include UTC timestamp: `smoke-3g-20260223-094512`. Each run is isolated.

**Cleanup:** All old results directories purged (dry-run-test, smoke-test-1, smoke-test-5, smoke-test-10, smoke-test-30). Fresh start with clean data.

**Lesson:** Append-mode JSONL + fixed run names = data corruption on re-runs. Always use unique identifiers for experiment runs.

---

## 2026-02-23 | Material adjudication added to chess_engine.py

**Change:** Added automatic material adjudication to `Chess960Game.is_game_over()`. When one side has ≥ 10 piece-value points advantage (Q=9, R=5, B=3, N=3, P=1) sustained for 6 consecutive plies after ply 40, the game is adjudicated as a win for the leading side.

**Problem discovered:** In smoke-test-10, the model achieved decisive material advantage against the random opponent by move ~35 (2R+N+B vs lone King = +16 material), but could not compute the forced mating sequence. It spent 160+ moves shuffling checks without closing the game. Result: draw by max_moves. Three games showed this pattern — won positions recorded as draws.

**Root cause (Evaluation-Calculation Gap — see H15-H20):** LLMs have strong positional evaluation (pattern matching from training data) but cannot perform recursive tree-search calculation needed for forced mating sequences. The model correctly identified "massive material advantage, need to checkmate" but could not compute the 8-10 move forced mate. This is structural, not a training gap — autoregressive token prediction ≠ tree traversal.

**Why adjudication, not longer/shorter max_moves:** Changing max_moves doesn't solve the problem. The model won't mate in 120, 200, or 500 moves — it lacks the calculation capability. Adjudication recognizes the objective reality of the position and records the correct result.

**Parameters chosen:**
- `ADJUDICATION_MIN_PLY = 40` — no adjudication before move 20 (protects against early tactical swings)
- `ADJUDICATION_MATERIAL_THRESHOLD = 10` — conservative (Q+P minimum); won't trigger on minor advantages
- `ADJUDICATION_CONSECUTIVE = 6` — 3 full moves of sustained advantage (handles temporary sacrifices)

**Precedent:** TCEC (Top Chess Engine Championship) uses eval-based adjudication. Our material-based version is simpler but sufficient for LLM-vs-random where the gap is always large, not marginal.

**Impact on Δₐ measurement:** Phase 0 baseline (model vs random) should now show 90%+ win rate instead of ~30% wins + 70% false draws. This gives clean E₀ data for Δₐ calculation in Phase 2.

**New hypotheses added:** H15-H20 in agzamov-test.md documenting the Evaluation-Calculation Gap as a fundamental LLM limitation revealed by the test.

---

## 2026-02-23 | max_moves_per_game: 120 → 200 (reverted)

**Change:** Reverted from 120 plies back to 200 plies (100 full moves).

**Evidence from smoke-test-5 (5 games, 120-ply limit, Sonnet 4):**
- Win rate: 0/5 = 0%. ALL five games drew via max_moves.
- Zero errors across 300 total moves — model plays perfectly legal chess.
- Cost: $1.82 for 5 games (~$0.36/game).
- Model reaches winning endgame (Q vs K+pawns) by move 40-50 but can't checkmate within remaining 10-20 moves.

**Key finding from reasoning logs:** Model correctly evaluates its position as winning ("I have a queen vs White's king and pawns — a winning endgame"), writes correct mating plans ("Qb3+ Ka2 Qb2#"), but executes infinite check sequences instead of forcing mate. This is a **tactical calculation limitation**, not a position evaluation failure.

**Why revert to 200:** 120 plies doesn't give enough runway to convert. The model needs 20-40 extra moves of shuffling before stumbling into mate. At 200 plies, at least some games should convert.

**No prompt hints added.** User decision: don't give endgame hints — the benchmark should measure whether models can assess and convert positions independently. If Sonnet 4 can't do it, that's a valid finding.

**Next:** Try Sonnet 4.6 / Opus — newer models may calculate mate sequences better.

---

## 2026-02-23 | Reasoning log added

**Change:** Model reasoning (full text before MOVE:) now saved to `chess/reasoning.jsonl` per move.

**Why:** Debugging the 0/5 win rate required manually scanning 18k-line debug logs to find model responses buried among HTTP headers. Now reasoning is in a clean JSONL: game_id, ply, agent, reasoning, note.

**Usage:** Post-mortem analysis — understand WHY models fail, not just THAT they fail. This is especially important for the paper: if we can show that models correctly evaluate positions but fail at tactical execution, that's a publishable insight about the gap between LLM "understanding" and "doing."

---

## 2026-02-23 | numpy JSON serialization fix

**Bug:** `np.False_` and `np.float64` from scipy's binomial test are not JSON serializable. `json.dump()` in `save_stats()` crashed when saving Phase 0 summary.

**Fix:** Explicit `bool()` and `float()` casts on `passed`, `significant`, and `p_value` in `_run_phase_0()`.

---

## 2026-02-23 | Poker format: SNG Heads-Up Turbo (not cash game)

**Decision:** Use PokerStars-style SNG (Sit & Go) Heads-Up Turbo tournaments instead of cash game format.

**Previous design (vision-implementation.md):** Cash game — reset stacks to 100bb each hand, metric bb/100 over 10,000 hands.

**New design:** SNG Heads-Up Turbo — starting stack 1500, blinds 10/20 escalating every 10 hands, winner-takes-all. 500 tournaments per test.

**Why SNG beats cash for this benchmark:**

1. **Cash allows degenerate GTO exploits.** In heads-up cash, an agent without memory can fold everything except AA/KK and break even or lose minimally. No pressure to play — Δₐ collapses to ≈0 because the format doesn't force decisions, not because memory is useless.

2. **SNG turbo creates escalating pressure.** As blinds grow, stack-to-blind ratio shrinks. Final stage: 1500 stack / 400 blind = 3.75bb → push-or-fold. Agent cannot wait for premium hands. Every decision is ±EV under pressure.

3. **Memory becomes directly actionable.** If you remember opponent folds 80% to pushes → shove any two. If opponent calls wide → tighten. This is exactly what the benchmark tests: adaptive behavior under adversarial pressure using recalled information.

4. **Clean binary outcome.** Each tournament produces a winner and loser. No ambiguous "slightly winning over 10,000 hands" — either you won the tournament or you didn't.

**Blind warning feature:** Like PokerStars, notify the model N hands before blinds increase. Adds realistic time pressure and tests whether memory-equipped agents plan ahead for blind level transitions.

**Metrics:**
- Δₐ poker = tournament win rate (memory) − tournament win rate (no memory)
- τ poker = tournaments until memory-equipped agent reaches stable edge
- Secondary: ICM-adjusted decision quality within tournaments

---

## 2026-02-23 | Pre-run fixes from GPT audit

**Changes (3 bugs found by external GPT-4o audit):**

1. **`start_time`/`end_time` now populated** — `chess_engine.py:to_result()` was returning empty strings. Now writes ISO 8601 UTC timestamps. Needed for future τ (convergence time) analysis in Phase 2.

2. **LLM API healthcheck before run** — `orchestrator.py:run()` now sends a trivial API call before starting any games. If credits are exhausted or auth is broken, run aborts immediately with `aborted: llm_healthcheck_failed` instead of producing 20 forfeit games.

3. **Phase 0 stats summary** — `_run_phase_0()` now writes `stats/phase_0_summary.json` with: model_score, win_rate, p_value, error breakdown, result_reasons, avg_moves, avg_duration, cost. Previously Phase 0 returned only a bool; Phases 1-2 already saved summaries.

**Tests:** 256 passed after changes.

**Source:** External audit by GPT-4o reviewing smoke-test-10 logs and JSONL. Gemini 3.1 Pro auditor missed these. Lesson: cross-model audit finds different bugs.

---

## 2026-02-23 | Phased spending strategy

**Decision:** Don't run full Phase 0 (30 games) blindly. Validate in small increments first.

**Evidence from smoke-test-10 (3 games completed):**
- Win rate: 1/3 = 33% (threshold is 70%)
- Game 3: clean checkmate in 18 moves (152s, $0.05)
- Games 1-2: model dominated but couldn't convert endgame → draws (818s+, ~$0.35 each)
- Zero illegal moves across all 3 games
- Model plays strong opening/middlegame but catastrophic endgame (promotes queens then sacrifices them all)

**Cost estimates (Sonnet 4, Chess960):**
- Quick win (~20 model moves): ~$0.05
- Max-length game (60 model moves at 120 ply limit): ~$0.20
- 30 games Phase 0: ~$4-8
- 400 games Phase 1-2: ~$80-120

**Strategy:**
1. Micro-test: 5 games (~$1) with 120-ply limit. Validate new settings work, measure win rate.
2. Calibrate: if win rate ≥60% → proceed to 30 games. If 30-60% → tune prompt. If <30% → reconsider model or thresholds.
3. Phase 0 full: 30 games (~$6). Only after micro-test passes.
4. Phase 1-2: 400 games (~$100). Only after clean Phase 0.

**Rationale:** Total validation cost $1-7 before committing $100+ to the main experiment. Prevents burning budget on a broken pipeline or a model that can't pass sanity check.

---

## 2026-02-23 | max_moves_per_game: 200 → 120

**Change:** Reduced from 200 plies (100 full moves) to 120 plies (60 full moves).

**Evidence from smoke-test-10 and smoke-test-5:**
- Game 1 (smoke-test-10): model vs random, hit 200-ply limit → draw. Model promoted 3 pawns to queens but couldn't checkmate. Duration: 818s (~13.6 min). Random promoted 2 pawns to knights.
- Game 2 (smoke-test-10): random vs model, 199 plies, insufficient_material draw. Model promoted 4 times (Q, Q, Q, Q) then sacrificed all queens chasing random's king. Duration: 828s (~13.8 min).
- Game 1 (smoke-test-5): still running at ply 200+, 7 errors (1 format + 6 illegal) — errors increase in late endgame as model hallucinates in complex positions.

**Problem:** Random agent never resigns. Model dominates by move 30-40 but enters K+B vs K endgames that are technically drawn or require perfect play to convert. Games become 13+ minute grinds with no chess value. Error rate increases in late endgame — model starts hallucinating illegal moves when position is unusual (bare kings, promoted knights, etc.).

**Human chess context:** Average game is 40-60 moves. Professional games rarely exceed 80. Players resign in hopeless positions.

**Why 120 (60 full moves):**
- Enough for complete opening + middlegame + early endgame
- Model should be able to win against random within 60 moves if it's competent
- If model can't win by move 60 against random, that's a meaningful signal (not just endgame technique)
- Cuts game time from ~13 min to ~8 min, making 5-game smoke tests ~40 min instead of ~65 min
- Reduces late-game hallucination errors

**Risk:** Some legitimate wins might become draws. Acceptable tradeoff — Phase 0 is sanity check, not Elo measurement.

---

## 2026-02-23 | __main__.py added

**Change:** Added `__main__.py` so package can run via `python -m agzamov`.

**Reason:** `pip install -e .` puts `agzamov.exe` in a Scripts dir that's not on PATH in Git Bash on Windows. `python -m agzamov` works from any terminal without PATH config.

---

## 2026-02-22 | System prompt includes benchmark context

**Change:** Updated `_build_system_prompt` in agent.py to tell the model it's being evaluated.

**Before:** "You are playing a series of Chess960 games."
**After:** "You are an AI participating in a structured Chess960 evaluation. Your chess skill, strategic reasoning, and ability to adapt are being measured. Play seriously — every game counts toward your overall performance score."

**Reason:** Without context, model might play casually or experimentally. Benchmark requires best effort. Transparency — model should know it's being tested.
